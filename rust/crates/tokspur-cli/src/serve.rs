//! Foreground serving lifecycle (`tokspur serve`): preflight verification, `SIGINT`/`SIGTERM` handling, RAII child-group escalation (`SIGTERM -> SIGKILL`), non-system descendant `drain_owned`, and `process.json` receipt ownership.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::fs;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use tokspur_json::Json;
use tokspur_pack::MTP_SIDECAR_SHA256;
use tokspur_sys::fd::pause;
use tokspur_sys::host::{
    hw_memsize, list_live_bsd_identities, process_bsd_identity, process_executable_path,
};
use tokspur_sys::net::port_available;
use tokspur_sys::process::{kill, killpg};
use tokspur_sys::{
    ESRCH, SIGKILL, SIGTERM, install_cli_stop_handlers, restore_cli_signal_handlers, stop_requested,
};

use crate::error::CliError;
use crate::layout::RuntimeLayout;
use crate::state::{
    StateLock, check_adoption, inference_processes, isolated_environment, private_dir,
    read_json_file, write_private_json,
};
use crate::verify::verify_build;

/// Bounded escalation and descendant-drain timeouts matching `src/tokspur/cli.py:369-386`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrainTimeouts {
    pub group_term_ms: u64,
    pub group_kill_ms: u64,
    pub owned_wait_ms: u64,
    pub owned_kill_ms: u64,
}

impl Default for DrainTimeouts {
    fn default() -> Self {
        Self {
            group_term_ms: 60_000,
            group_kill_ms: 15_000,
            owned_wait_ms: 45_000,
            owned_kill_ms: 15_000,
        }
    }
}

impl DrainTimeouts {
    pub fn parse_csv(text: &str) -> Result<Self, CliError> {
        let parts: Vec<&str> = text.split(',').collect();
        if parts.len() != 4 {
            return Err(CliError::Usage(
                "--drain-timeouts-ms expects four comma-separated millisecond values".to_owned(),
            ));
        }
        let nums: Vec<u64> = parts
            .iter()
            .map(|p| {
                p.parse::<u64>().map_err(|_| {
                    CliError::Usage("invalid integer in --drain-timeouts-ms".to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            group_term_ms: nums[0],
            group_kill_ms: nums[1],
            owned_wait_ms: nums[2],
            owned_kill_ms: nums[3],
        })
    }
}

/// Runtime, operator and supervision flags for `tokspur serve`, matching `src/tokspur/cli.py`.
#[derive(Clone, PartialEq)]
pub struct ServeOptions {
    pub raw_gguf: bool,
    pub pack_v3: bool,
    pub derived_pack: Option<PathBuf>,
    pub gdn_prefill_only: bool,
    pub derived_pack_trust_mode: String,
    pub output_packet: u32,
    pub worker_burst_pacing: bool,
    pub first_output_packet: Option<u32>,
    pub prefill_dense: String,
    pub prefill_grouped: String,
    pub known_token_capacity: u32,
    pub mtp_draft_slice: bool,
    pub host_logit_views: Option<bool>,
    pub logit_topk: Option<bool>,
    pub ple_async_gather: Option<bool>,
    pub mtp_fast_round: Option<bool>,
    pub dflash2_defer_context: Option<bool>,
    pub dflash2_verify_rows: Option<String>,
    pub prepared_validation: bool,
    pub verify_cost_profile: Option<PathBuf>,
    pub dflash2_draft_policy: String,
    pub dflash2_max_drafts: Option<u32>,
    pub dflash2_vocab_slice: Option<u32>,
    pub verify_nax: bool,
    pub verify_nax_selection: Option<bool>,
    pub verify_nax_m8: bool,
    pub verify_m8_splits: Option<PathBuf>,
    pub verify_hc_gdn_fused: Option<bool>,
    pub speed_preset: bool,
    pub mtp_depth: Option<u32>,
    pub dflash2_precision: Option<String>,
    pub drafter: String,
    pub dflash2_dir: Option<PathBuf>,
    pub concurrent_dispatch: bool,
    pub q8_gemv_rg4: String,
    pub q8_three_msg: String,
    pub prefill_multiseg: bool,
    pub prefill_swizzle: bool,
    pub prefill_router_fast: Option<bool>,
    pub qsa_prefill_fused: bool,
    pub hc_consumer_fused: bool,
    pub ple_overlap: bool,
    pub shared_prefix_cache: Option<bool>,
    pub mtp_lookup: String,
    pub mtp_lookup_match: Option<u32>,
    pub tool_enum_validation: String,
    pub tool_arg_validation: String,
    pub private_prefix_trace: bool,
    pub telemetry_stdout: bool,
    pub request_timeout_seconds: u32,
    pub checkpoint_dir: Option<PathBuf>,
    pub checkpoint_cap_gib: Option<f64>,
    pub checkpoint_floor_gib: f64,
    pub checkpoint_policy: String,
    pub overlap_prefill: bool,
    pub timeouts: DrainTimeouts,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            raw_gguf: false,
            pack_v3: false,
            derived_pack: None,
            gdn_prefill_only: false,
            derived_pack_trust_mode: "full".to_owned(),
            output_packet: 8,
            worker_burst_pacing: false,
            first_output_packet: None,
            prefill_dense: "f32-staged".to_owned(),
            prefill_grouped: "f32".to_owned(),
            known_token_capacity: 2048,
            mtp_draft_slice: false,
            host_logit_views: None,
            logit_topk: None,
            ple_async_gather: None,
            mtp_fast_round: None,
            dflash2_defer_context: None,
            dflash2_verify_rows: None,
            prepared_validation: false,
            verify_cost_profile: None,
            dflash2_draft_policy: "always".to_owned(),
            dflash2_max_drafts: None,
            dflash2_vocab_slice: None,
            verify_nax: false,
            verify_nax_selection: None,
            verify_nax_m8: false,
            verify_m8_splits: None,
            verify_hc_gdn_fused: None,
            speed_preset: false,
            mtp_depth: None,
            dflash2_precision: None,
            drafter: "mtp".to_owned(),
            dflash2_dir: None,
            concurrent_dispatch: true,
            q8_gemv_rg4: "off".to_owned(),
            q8_three_msg: "off".to_owned(),
            prefill_multiseg: false,
            prefill_swizzle: true,
            prefill_router_fast: None,
            qsa_prefill_fused: true,
            hc_consumer_fused: true,
            ple_overlap: true,
            shared_prefix_cache: None,
            mtp_lookup: "off".to_owned(),
            mtp_lookup_match: None,
            tool_enum_validation: "strict".to_owned(),
            tool_arg_validation: "strict".to_owned(),
            private_prefix_trace: false,
            telemetry_stdout: false,
            request_timeout_seconds: 3600,
            checkpoint_dir: None,
            checkpoint_cap_gib: None,
            checkpoint_floor_gib: 20.0,
            checkpoint_policy: "recompute-cost".to_owned(),
            overlap_prefill: false,
            timeouts: DrainTimeouts::default(),
        }
    }
}

impl fmt::Debug for ServeOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServeOptions")
            .field("raw_gguf", &self.raw_gguf)
            .field("pack_v3", &self.pack_v3)
            .field(
                "derived_pack",
                &self.derived_pack.as_ref().map(|_| "[redacted]"),
            )
            .field("gdn_prefill_only", &self.gdn_prefill_only)
            .field("derived_pack_trust_mode", &self.derived_pack_trust_mode)
            .field("output_packet", &self.output_packet)
            .field("worker_burst_pacing", &self.worker_burst_pacing)
            .field("first_output_packet", &self.first_output_packet)
            .field("prefill_dense", &self.prefill_dense)
            .field("prefill_grouped", &self.prefill_grouped)
            .field("known_token_capacity", &self.known_token_capacity)
            .field("mtp_draft_slice", &self.mtp_draft_slice)
            .field("host_logit_views", &self.host_logit_views)
            .field("logit_topk", &self.logit_topk)
            .field("ple_async_gather", &self.ple_async_gather)
            .field("mtp_fast_round", &self.mtp_fast_round)
            .field("dflash2_defer_context", &self.dflash2_defer_context)
            .field("dflash2_verify_rows", &self.dflash2_verify_rows)
            .field("prepared_validation", &self.prepared_validation)
            .field(
                "verify_cost_profile",
                &self.verify_cost_profile.as_ref().map(|_| "[redacted]"),
            )
            .field("dflash2_draft_policy", &self.dflash2_draft_policy)
            .field("dflash2_max_drafts", &self.dflash2_max_drafts)
            .field("dflash2_vocab_slice", &self.dflash2_vocab_slice)
            .field("verify_nax", &self.verify_nax)
            .field("verify_nax_selection", &self.verify_nax_selection)
            .field("verify_nax_m8", &self.verify_nax_m8)
            .field(
                "verify_m8_splits",
                &self.verify_m8_splits.as_ref().map(|_| "[redacted]"),
            )
            .field("verify_hc_gdn_fused", &self.verify_hc_gdn_fused)
            .field("speed_preset", &self.speed_preset)
            .field("mtp_depth", &self.mtp_depth)
            .field("dflash2_precision", &self.dflash2_precision)
            .field("drafter", &self.drafter)
            .field(
                "dflash2_dir",
                &self.dflash2_dir.as_ref().map(|_| "[redacted]"),
            )
            .field("concurrent_dispatch", &self.concurrent_dispatch)
            .field("q8_gemv_rg4", &self.q8_gemv_rg4)
            .field("q8_three_msg", &self.q8_three_msg)
            .field("prefill_multiseg", &self.prefill_multiseg)
            .field("prefill_swizzle", &self.prefill_swizzle)
            .field("prefill_router_fast", &self.prefill_router_fast)
            .field("qsa_prefill_fused", &self.qsa_prefill_fused)
            .field("hc_consumer_fused", &self.hc_consumer_fused)
            .field("ple_overlap", &self.ple_overlap)
            .field("shared_prefix_cache", &self.shared_prefix_cache)
            .field("mtp_lookup", &self.mtp_lookup)
            .field("mtp_lookup_match", &self.mtp_lookup_match)
            .field("tool_enum_validation", &self.tool_enum_validation)
            .field("tool_arg_validation", &self.tool_arg_validation)
            .field("private_prefix_trace", &self.private_prefix_trace)
            .field("telemetry_stdout", &self.telemetry_stdout)
            .field(
                "checkpoint_dir",
                &self.checkpoint_dir.as_ref().map(|_| "[redacted]"),
            )
            .field("checkpoint_cap_gib", &self.checkpoint_cap_gib)
            .field("checkpoint_floor_gib", &self.checkpoint_floor_gib)
            .field("checkpoint_policy", &self.checkpoint_policy)
            .field("overlap_prefill", &self.overlap_prefill)
            .field("timeouts", &self.timeouts)
            .finish()
    }
}

/// Resolved speculative drafter asset forwarded to `tokspur-server`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeAsset<'a> {
    MtpSidecar(&'a Path),
    DFlash2Dir(&'a Path),
}

impl fmt::Debug for SpeculativeAsset<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MtpSidecar(_) => formatter.write_str("MtpSidecar([redacted])"),
            Self::DFlash2Dir(_) => formatter.write_str("DFlash2Dir([redacted])"),
        }
    }
}

/// Validates individual option bounds and cross-flag compatibility before launching `tokspur-server`.
pub fn validate_serve_options(opts: &ServeOptions) -> Result<(), CliError> {
    if opts.gdn_prefill_only {
        return Err(CliError::Validation(
            "--gdn-prefill-only was removed because its source could not be verified".to_owned(),
        ));
    }
    if opts.prefill_router_fast == Some(true) {
        return Err(CliError::Validation(
            "--native-prefill-router-fast requires the removed router-window implementation"
                .to_owned(),
        ));
    }
    if !(1..=86400).contains(&opts.request_timeout_seconds) {
        return Err(CliError::Validation(
            "request timeout requires 1..86400 seconds".to_owned(),
        ));
    }
    if opts.raw_gguf && (opts.pack_v3 || opts.derived_pack.is_some()) {
        return Err(CliError::Validation(
            "--raw-gguf cannot be combined with --pack-v3 or --derived-pack".to_owned(),
        ));
    }
    if !matches!(opts.derived_pack_trust_mode.as_str(), "full" | "local-seal") {
        return Err(CliError::Validation(format!(
            "Unsupported derived_pack_trust_mode: {}",
            opts.derived_pack_trust_mode
        )));
    }
    if !matches!(opts.output_packet, 8 | 64) {
        return Err(CliError::Validation(format!(
            "Unsupported output_packet: {}",
            opts.output_packet
        )));
    }
    if let Some(first_pkt) = opts.first_output_packet {
        if !(1..=64).contains(&first_pkt) {
            return Err(CliError::Validation(format!(
                "Unsupported first_output_packet: {first_pkt} (expected 1..64)"
            )));
        }
        if first_pkt > opts.output_packet {
            return Err(CliError::Validation(format!(
                "--first-output-packet ({first_pkt}) cannot exceed --output-packet ({})",
                opts.output_packet
            )));
        }
    }
    if !matches!(
        opts.prefill_dense.as_str(),
        "f32-staged" | "nax" | "f32" | "f16"
    ) {
        return Err(CliError::Validation(format!(
            "Unsupported prefill_dense mode: {}",
            opts.prefill_dense
        )));
    }
    if !matches!(
        opts.prefill_grouped.as_str(),
        "f32" | "auto" | "serial" | "nax"
    ) {
        return Err(CliError::Validation(format!(
            "Unsupported prefill_grouped mode: {}",
            opts.prefill_grouped
        )));
    }
    let dense_staged_or_nax = matches!(opts.prefill_dense.as_str(), "f32-staged" | "nax");
    if dense_staged_or_nax && opts.prefill_grouped == "serial" {
        return Err(CliError::Validation(
            "--prefill-grouped serial requires --prefill-dense f32 or f16".to_owned(),
        ));
    }
    if opts.prefill_grouped == "nax" && !dense_staged_or_nax {
        return Err(CliError::Validation(
            "--prefill-grouped nax requires --prefill-dense f32-staged or nax".to_owned(),
        ));
    }
    if !matches!(opts.known_token_capacity, 512 | 1024 | 2048) {
        return Err(CliError::Validation(format!(
            "Unsupported known_token_capacity: {} (expected 512, 1024 or 2048)",
            opts.known_token_capacity
        )));
    }
    if !matches!(opts.drafter.as_str(), "mtp" | "dflash2") {
        return Err(CliError::Validation(format!(
            "Unsupported drafter mode: {}",
            opts.drafter
        )));
    }
    if !matches!(opts.q8_gemv_rg4.as_str(), "off" | "on" | "wide") {
        return Err(CliError::Validation(format!(
            "Unsupported q8_gemv_rg4 mode: {}",
            opts.q8_gemv_rg4
        )));
    }
    if !matches!(opts.q8_three_msg.as_str(), "off" | "on" | "wide") {
        return Err(CliError::Validation(format!(
            "Unsupported q8_three_msg mode: {}",
            opts.q8_three_msg
        )));
    }
    if !matches!(opts.mtp_lookup.as_str(), "off" | "bypass" | "hybrid") {
        return Err(CliError::Validation(format!(
            "Unsupported mtp_lookup mode: {}",
            opts.mtp_lookup
        )));
    }
    if let Some(match_len) = opts.mtp_lookup_match
        && !(2..=16).contains(&match_len)
    {
        return Err(CliError::Validation(format!(
            "Unsupported mtp_lookup_match: {match_len} (expected 2..16)"
        )));
    }
    if !matches!(opts.tool_enum_validation.as_str(), "strict" | "executor") {
        return Err(CliError::Validation(format!(
            "Unsupported tool_enum_validation mode: {}",
            opts.tool_enum_validation
        )));
    }
    if !matches!(opts.tool_arg_validation.as_str(), "strict" | "executor") {
        return Err(CliError::Validation(format!(
            "Unsupported tool_arg_validation mode: {}",
            opts.tool_arg_validation
        )));
    }
    if let Some(gib) = opts.checkpoint_cap_gib
        && (!gib.is_finite() || !(0.0..=32.0).contains(&gib))
    {
        return Err(CliError::Validation(
            "checkpoint_cap_gib must be between 0.0 and 32.0 GiB".to_owned(),
        ));
    }
    if !opts.checkpoint_floor_gib.is_finite() || !(0.0..=100.0).contains(&opts.checkpoint_floor_gib)
    {
        return Err(CliError::Validation(
            "checkpoint_floor_gib must be between 0.0 and 100.0 GiB".to_owned(),
        ));
    }
    if !matches!(opts.checkpoint_policy.as_str(), "recompute-cost" | "lru") {
        return Err(CliError::Validation(format!(
            "Unsupported checkpoint_policy: {}",
            opts.checkpoint_policy
        )));
    }

    if let Some(depth) = opts.mtp_depth
        && !matches!(depth, 2 | 3)
    {
        return Err(CliError::Validation(format!(
            "Unsupported mtp_depth: {depth} (expected 2 or 3)"
        )));
    }
    if let Some(ref prec) = opts.dflash2_precision
        && !matches!(prec.as_str(), "q8_0" | "bf16")
    {
        return Err(CliError::Validation(format!(
            "Unsupported dflash2_precision: {prec} (expected q8_0 or bf16)"
        )));
    }
    if let Some(md) = opts.dflash2_max_drafts
        && !(1..=7).contains(&md)
    {
        return Err(CliError::Validation(format!(
            "Unsupported dflash2_max_drafts: {md} (expected 1..7)"
        )));
    }
    if let Some(vs) = opts.dflash2_vocab_slice
        && !(vs == 0 || ((256..248_320).contains(&vs) && vs.is_multiple_of(32)))
    {
        return Err(CliError::Validation(format!(
            "Unsupported dflash2_vocab_slice: {vs} (expected 0 or a multiple of 32 in 256..248319)"
        )));
    }
    if opts.verify_hc_gdn_fused == Some(true)
        && !opts.verify_nax
        && !opts.verify_nax_m8
        && opts.drafter != "dflash2"
    {
        return Err(CliError::Validation(
            "--native-verify-hc-gdn-fused on requires --verify-nax or --drafter dflash2".to_owned(),
        ));
    }

    if opts.drafter == "dflash2" {
        if opts.dflash2_dir.is_none() {
            return Err(CliError::Validation(
                "--drafter dflash2 requires --dflash2-dir".to_owned(),
            ));
        }
        if opts.mtp_depth.is_some() {
            return Err(CliError::Validation(
                "--mtp-depth is incompatible with --drafter dflash2".to_owned(),
            ));
        }
        if opts.mtp_draft_slice {
            return Err(CliError::Validation(
                "--mtp-draft-slice is incompatible with --drafter dflash2".to_owned(),
            ));
        }
        if opts.mtp_fast_round == Some(true) {
            return Err(CliError::Validation(
                "--native-mtp-fast-round on is incompatible with --drafter dflash2".to_owned(),
            ));
        }
        if opts.mtp_lookup != "off" || opts.mtp_lookup_match.is_some() {
            return Err(CliError::Validation(
                "--mtp-lookup is incompatible with --drafter dflash2".to_owned(),
            ));
        }
    } else if opts.dflash2_dir.is_some() {
        return Err(CliError::Validation(
            "--dflash2-dir requires --drafter dflash2".to_owned(),
        ));
    } else if opts.dflash2_precision.is_some() {
        return Err(CliError::Validation(
            "--dflash2-precision requires --drafter dflash2".to_owned(),
        ));
    } else if opts.dflash2_max_drafts.is_some() {
        return Err(CliError::Validation(
            "--dflash2-max-drafts requires --drafter dflash2".to_owned(),
        ));
    } else if opts.dflash2_vocab_slice.is_some() {
        return Err(CliError::Validation(
            "--dflash2-vocab-slice requires --drafter dflash2".to_owned(),
        ));
    }
    if opts.verify_m8_splits.is_some() && !opts.verify_nax_m8 {
        return Err(CliError::Validation(
            "--native-verify-m8-splits requires explicit --verify-nax-m8".to_owned(),
        ));
    }
    if opts.verify_cost_profile.is_some()
        && (opts.verify_nax
            || opts.verify_nax_m8
            || opts.verify_hc_gdn_fused == Some(true)
            || opts.mtp_fast_round == Some(true))
    {
        return Err(CliError::Validation(
            "measured controllers require exact target verification and fast round off".to_owned(),
        ));
    }
    if opts.verify_cost_profile.is_some()
        && opts.drafter == "dflash2"
        && opts.verify_nax_selection != Some(false)
    {
        return Err(CliError::Validation(
            "measured DFlash2 requires explicit target NAX off".to_owned(),
        ));
    }
    if opts.dflash2_draft_policy == "auto"
        && (opts.drafter != "dflash2"
            || opts.verify_cost_profile.is_none()
            || opts.dflash2_verify_rows.as_deref() != Some("auto"))
    {
        return Err(CliError::Validation("--native-dflash2-draft-policy auto requires DFlash2, a cost profile and automatic verify rows".to_owned()));
    }
    if opts
        .verify_cost_profile
        .as_ref()
        .is_some_and(|p| !p.is_absolute())
    {
        return Err(CliError::Validation(
            "--native-verify-cost-profile requires an absolute path".to_owned(),
        ));
    }
    if let Some(ref rows_mode) = opts.dflash2_verify_rows {
        if !matches!(rows_mode.as_str(), "all" | "auto") {
            return Err(CliError::Validation(format!(
                "Unsupported dflash2_verify_rows mode: {rows_mode}"
            )));
        }
        if opts.drafter != "dflash2" {
            return Err(CliError::Validation(
                "--native-dflash2-verify-rows requires --drafter dflash2".to_owned(),
            ));
        }
    }
    if opts.prefill_router_fast == Some(true)
        && opts.prefill_dense != "nax"
        && opts.prefill_grouped != "nax"
    {
        return Err(CliError::Validation(
            "--native-prefill-router-fast on requires --prefill-dense nax or --prefill-grouped nax"
                .to_owned(),
        ));
    }
    Ok(())
}

/// Builds the `--derived-pack*` argument slice forwarded to `tokspur-server`.
pub fn build_derived_pack_flags(opts: &ServeOptions, pack_file: &Path) -> Vec<String> {
    if opts.raw_gguf {
        return Vec::new();
    }
    let mut flags = vec![
        "--derived-pack".to_owned(),
        pack_file.to_string_lossy().into_owned(),
        "--derived-pack-auth-workers".to_owned(),
        "16".to_owned(),
    ];
    if opts.derived_pack_trust_mode != "full" {
        flags.push("--derived-pack-trust-mode".to_owned());
        flags.push(opts.derived_pack_trust_mode.clone());
    }
    flags
}

/// Builds the carrier and runtime argument slice forwarded to `tokspur-server`.
pub fn build_carrier_flags(
    opts: &ServeOptions,
    speculative_asset: SpeculativeAsset<'_>,
    state_dir: &Path,
) -> Result<Vec<String>, CliError> {
    validate_serve_options(opts)?;
    let speculative_flags: Vec<String> = match (opts.drafter.as_str(), speculative_asset) {
        ("mtp", SpeculativeAsset::MtpSidecar(mtp_sidecar)) => {
            let mtp_sidecar_str = mtp_sidecar.to_str().ok_or_else(|| {
                CliError::Validation("MTP sidecar path is not valid UTF-8".to_owned())
            })?;
            let depth_str = opts.mtp_depth.unwrap_or(2).to_string();
            vec![
                "--native-mtp-sidecar".to_owned(),
                mtp_sidecar_str.to_owned(),
                "--native-mtp-depth".to_owned(),
                depth_str,
                "--native-mtp-request-carry".to_owned(),
                "on".to_owned(),
                "--native-mtp-draft-attention".to_owned(),
                "tiled128".to_owned(),
            ]
        }
        ("dflash2", SpeculativeAsset::DFlash2Dir(dflash2_dir)) => {
            let dflash2_str = dflash2_dir.to_str().ok_or_else(|| {
                CliError::Validation("DFlash2 directory path is not valid UTF-8".to_owned())
            })?;
            let mut flags = vec![
                "--native-drafter".to_owned(),
                "dflash2".to_owned(),
                "--native-dflash2-dir".to_owned(),
                dflash2_str.to_owned(),
                "--native-mtp-depth".to_owned(),
                "7".to_owned(),
                "--native-mtp-request-carry".to_owned(),
                "on".to_owned(),
                "--native-mtp-draft-attention".to_owned(),
                "split16".to_owned(),
            ];
            if let Some(ref prec) = opts.dflash2_precision {
                flags.push("--native-dflash2-precision".to_owned());
                flags.push(prec.clone());
            }
            if let Some(md) = opts.dflash2_max_drafts {
                flags.push("--native-dflash2-max-drafts".to_owned());
                flags.push(md.to_string());
            }
            if let Some(vs) = opts.dflash2_vocab_slice {
                flags.push("--native-dflash2-vocab-slice".to_owned());
                flags.push(vs.to_string());
            }
            flags
        }
        _ => {
            return Err(CliError::Validation(
                "speculative asset does not match selected drafter mode".to_owned(),
            ));
        }
    };

    let staged_carrier = if matches!(opts.prefill_dense.as_str(), "f32-staged" | "nax")
        && matches!(opts.prefill_grouped.as_str(), "f32" | "auto" | "nax")
    {
        "on"
    } else {
        "off"
    };

    let mut flags = vec![
        "--native-scalar-channel".to_owned(),
        "on".to_owned(),
        "--native-fast-math".to_owned(),
        "channel".to_owned(),
        "--native-qsa-stage-v8".to_owned(),
        "on".to_owned(),
    ];
    flags.extend(speculative_flags);
    flags.extend([
        "--native-mtp-output-packet".to_owned(),
        opts.output_packet.to_string(),
    ]);
    if let Some(first_pkt) = opts.first_output_packet {
        flags.push("--native-mtp-first-output-packet".to_owned());
        flags.push(first_pkt.to_string());
    }
    flags.extend([
        "--native-qsa-sparse".to_owned(),
        "off".to_owned(),
        "--native-gdn-ported".to_owned(),
        "off".to_owned(),
    ]);
    flags.extend([
        "--native-prefill-input512".to_owned(),
        "on".to_owned(),
        "--native-known-token-capacity".to_owned(),
        opts.known_token_capacity.to_string(),
    ]);
    if opts.raw_gguf {
        flags.push("--native-raw-decode".to_owned());
        flags.push("on".to_owned());
    }
    for (flag, enabled) in [
        ("--native-host-logit-views", opts.host_logit_views),
        ("--native-logit-topk", opts.logit_topk),
        ("--native-ple-async-gather", opts.ple_async_gather),
        ("--native-mtp-fast-round", opts.mtp_fast_round),
        ("--native-prefill-router-fast", opts.prefill_router_fast),
        ("--native-dflash2-defer-context", opts.dflash2_defer_context),
        ("--native-verify-hc-gdn-fused", opts.verify_hc_gdn_fused),
    ] {
        if let Some(enabled) = enabled {
            flags.extend([
                flag.to_owned(),
                if enabled { "on" } else { "off" }.to_owned(),
            ]);
        }
    }
    if let Some(ref path) = opts.verify_m8_splits {
        flags.extend([
            "--native-verify-m8-splits".to_owned(),
            path.to_string_lossy().into_owned(),
        ]);
    }
    if opts.prepared_validation {
        flags.extend(["--native-prepared-validation".to_owned(), "on".to_owned()]);
    }
    if opts.worker_burst_pacing {
        flags.extend(["--worker-burst-pacing".to_owned(), "on".to_owned()]);
    }
    if let Some(ref profile) = opts.verify_cost_profile {
        flags.extend([
            "--native-verify-cost-profile".to_owned(),
            profile.to_string_lossy().into_owned(),
        ]);
    }
    if opts.dflash2_draft_policy == "auto" {
        flags.extend([
            "--native-dflash2-draft-policy".to_owned(),
            "auto".to_owned(),
        ]);
    }
    if let Some(ref rows_mode) = opts.dflash2_verify_rows {
        flags.push("--native-dflash2-verify-rows".to_owned());
        flags.push(rows_mode.clone());
    }
    if opts.mtp_draft_slice {
        flags.push("--native-mtp-draft-slice".to_owned());
        flags.push("on".to_owned());
    }
    if opts.verify_nax || opts.verify_nax_m8 {
        flags.push("--native-verify-nax".to_owned());
        flags.push("on".to_owned());
    } else if opts.verify_nax_selection == Some(false) {
        flags.extend(["--native-verify-nax".to_owned(), "off".to_owned()]);
    }
    if opts.verify_nax_m8 {
        flags.push("--native-verify-nax-m8".to_owned());
        flags.push("on".to_owned());
    }
    flags.extend(
        [
            "--native-prefill-dense",
            opts.prefill_dense.as_str(),
            "--native-prefill-grouped",
            opts.prefill_grouped.as_str(),
            "--native-prefill-hc-rows",
            staged_carrier,
            "--native-prefill-router-window",
            "off",
            "--native-prefill-hc-down-window",
            "off",
            "--native-prefill-hc-down-f32",
            "off",
            "--native-prefill-final-logits",
            "on",
            "--native-q8-three-rows",
            "on",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    if opts.concurrent_dispatch {
        flags.push("--native-concurrent-dispatch".to_owned());
        flags.push("on".to_owned());
    }
    if opts.q8_gemv_rg4 != "off" {
        flags.push("--native-q8-gemv-rg4".to_owned());
        flags.push(opts.q8_gemv_rg4.clone());
    }
    if opts.q8_three_msg != "off" {
        flags.push("--native-q8-three-msg".to_owned());
        flags.push(opts.q8_three_msg.clone());
    }
    if opts.prefill_multiseg {
        flags.push("--native-prefill-multiseg".to_owned());
        flags.push("on".to_owned());
    }
    if opts.prefill_swizzle {
        flags.push("--native-prefill-swizzle".to_owned());
        flags.push("on".to_owned());
    }
    if opts.qsa_prefill_fused {
        flags.push("--native-qsa-prefill-fused".to_owned());
        flags.push("on".to_owned());
    }
    if opts.hc_consumer_fused {
        flags.push("--native-hc-consumer-fused".to_owned());
        flags.push("on".to_owned());
    }
    if opts.ple_overlap {
        flags.push("--native-ple-overlap".to_owned());
        flags.push("on".to_owned());
    }
    let effective_shared_prefix = opts.shared_prefix_cache.unwrap_or(opts.drafter == "mtp");
    if effective_shared_prefix {
        flags.push("--shared-prefix-cache".to_owned());
        flags.push("on".to_owned());
    }
    if opts.mtp_lookup != "off" {
        flags.push("--native-mtp-lookup".to_owned());
        flags.push(opts.mtp_lookup.clone());
    }
    if let Some(match_len) = opts.mtp_lookup_match {
        flags.push("--native-mtp-lookup-match".to_owned());
        flags.push(match_len.to_string());
    }
    flags.push("--tool-enum-validation".to_owned());
    flags.push(opts.tool_enum_validation.clone());
    if opts.tool_arg_validation != "strict" {
        flags.push("--tool-arg-validation".to_owned());
        flags.push(opts.tool_arg_validation.clone());
    }
    if opts.private_prefix_trace {
        flags.push("--private-prefix-trace".to_owned());
        flags.push("on".to_owned());
    }
    if opts.telemetry_stdout {
        flags.push("--telemetry-stdout".to_owned());
        flags.push("on".to_owned());
    }
    let effective_checkpoint_cap = opts.checkpoint_cap_gib.unwrap_or(0.0);
    if effective_checkpoint_cap > 0.0 {
        let resolved_ckpt_dir = match opts.checkpoint_dir.as_ref() {
            Some(custom) if custom.is_absolute() => custom.clone(),
            Some(custom) => std::env::current_dir()
                .map_err(|source| CliError::Io {
                    context: "cannot resolve relative checkpoint directory".to_owned(),
                    source,
                })?
                .join(custom),
            None => state_dir.join("checkpoints"),
        };
        let ckpt_dir_str = resolved_ckpt_dir.to_str().ok_or_else(|| {
            CliError::Validation("checkpoint directory path is not valid UTF-8".to_owned())
        })?;
        flags.extend([
            "--checkpoint-ssd".to_owned(),
            "on".to_owned(),
            "--checkpoint-cap-gib".to_owned(),
            effective_checkpoint_cap.to_string(),
            "--checkpoint-floor-gib".to_owned(),
            opts.checkpoint_floor_gib.to_string(),
            "--checkpoint-dir".to_owned(),
            ckpt_dir_str.to_owned(),
            "--checkpoint-policy".to_owned(),
            opts.checkpoint_policy.clone(),
        ]);
    }
    if opts.overlap_prefill {
        flags.push("--overlap-prefill".to_owned());
        flags.push("on".to_owned());
    }
    Ok(flags)
}

pub(crate) struct SignalGuard;

impl SignalGuard {
    pub(crate) fn install() -> Result<Self, CliError> {
        if !install_cli_stop_handlers() {
            return Err(CliError::Validation(
                "cannot install SIGINT/SIGTERM/SIGHUP stop handlers".to_owned(),
            ));
        }
        Ok(Self)
    }
}

impl Drop for SignalGuard {
    fn drop(&mut self) {
        // Keep cooperative handlers through process exit after a requested stop, so a signal burst cannot interrupt the completed drain while the CLI returns its status.
        if !stop_requested() {
            restore_cli_signal_handlers();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OwnedDescendantKey {
    pid: i32,
    start_tvsec: u64,
    start_tvusec: u64,
}

fn exit_status_code(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        code
    } else if let Some(sig) = status.signal() {
        128 + sig
    } else {
        1
    }
}

fn send_group_signal(pgid: i32, sig: i32) {
    if let Err(err) = killpg(pgid, sig)
        && err.raw_os_error() != Some(ESRCH)
    {
        // Best-effort group signal during drain.
    }
}

fn wait_child_until(child: &mut Child, timeout_ms: u64) -> Result<Option<ExitStatus>, CliError> {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        if let Some(status) = child.try_wait().map_err(|source| CliError::Io {
            context: "failed polling server child status".to_owned(),
            source,
        })? {
            return Ok(Some(status));
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(None);
        }
        let sleep_ms = deadline.saturating_duration_since(now).as_millis().min(25) as i32;
        pause(sleep_ms.max(1));
    }
}

struct SupervisorGuard {
    child: Child,
    child_pid: i32,
    seen: BTreeSet<OwnedDescendantKey>,
    owned: BTreeSet<OwnedDescendantKey>,
    process_receipt_path: PathBuf,
    timeouts: DrainTimeouts,
    finished: bool,
}

impl SupervisorGuard {
    fn new(child: Child, process_receipt_path: PathBuf, timeouts: DrainTimeouts) -> Self {
        let child_pid = child.id() as i32;
        Self {
            child,
            child_pid,
            seen: BTreeSet::new(),
            owned: BTreeSet::new(),
            process_receipt_path,
            timeouts,
            finished: false,
        }
    }

    fn refresh_descendants(&mut self) {
        let Ok(procs) = list_live_bsd_identities() else {
            return;
        };
        let mut by_ppid: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
        for (idx, p) in procs.iter().enumerate() {
            by_ppid.entry(p.ppid).or_default().push(idx);
        }
        let mut queue = VecDeque::new();
        let mut visited = BTreeSet::new();
        queue.push_back(self.child_pid);
        visited.insert(self.child_pid);

        while let Some(parent_pid) = queue.pop_front() {
            let Some(children) = by_ppid.get(&parent_pid) else {
                continue;
            };
            for &idx in children {
                let proc_snap = &procs[idx];
                if !visited.insert(proc_snap.pid) {
                    continue;
                }
                queue.push_back(proc_snap.pid);
                if proc_snap.is_zombie() {
                    continue;
                }
                let key = OwnedDescendantKey {
                    pid: proc_snap.pid,
                    start_tvsec: proc_snap.start_tvsec,
                    start_tvusec: proc_snap.start_tvusec,
                };
                if !self.seen.insert(key) {
                    continue;
                }
                // Exclude Apple system XPC/Metal shader compiler daemons (`cli.py:354-357`).
                let is_system_service = process_executable_path(proc_snap.pid)
                    .as_deref()
                    .is_some_and(|exe| {
                        exe.starts_with("/System/") || exe.starts_with("/usr/libexec/")
                    });
                if !is_system_service {
                    self.owned.insert(key);
                }
            }
        }
    }

    fn live_owned(&self) -> Vec<i32> {
        let mut remaining = Vec::new();
        for item in &self.owned {
            if let Ok(Some(snap)) = process_bsd_identity(item.pid)
                && snap.start_tvsec == item.start_tvsec
                && snap.start_tvusec == item.start_tvusec
                && !snap.is_zombie()
            {
                remaining.push(item.pid);
            }
        }
        remaining
    }

    fn drain_owned(&self, timeout_ms: u64) -> Vec<i32> {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            let remaining = self.live_owned();
            if remaining.is_empty() || Instant::now() >= deadline {
                return remaining;
            }
            pause(50);
        }
    }

    fn finish(&mut self, natural_status: Option<ExitStatus>) -> Result<ExitStatus, CliError> {
        self.finished = true;
        self.refresh_descendants();

        let status = if let Some(done) = natural_status {
            done
        } else if let Some(done) = self.child.try_wait().map_err(|source| CliError::Io {
            context: "failed polling server child".to_owned(),
            source,
        })? {
            done
        } else {
            send_group_signal(self.child_pid, SIGTERM);
            if let Some(done) = wait_child_until(&mut self.child, self.timeouts.group_term_ms)? {
                done
            } else {
                send_group_signal(self.child_pid, SIGKILL);
                match wait_child_until(&mut self.child, self.timeouts.group_kill_ms)? {
                    Some(done) => done,
                    None => {
                        return Err(CliError::Validation(
                            "Server process group did not exit after SIGKILL; retained process receipt"
                                .to_owned(),
                        ));
                    }
                }
            }
        };

        let mut remaining = self.drain_owned(self.timeouts.owned_wait_ms);
        if !remaining.is_empty() {
            for &pid in &remaining {
                let _ignored = kill(pid, SIGKILL);
            }
            remaining = self.drain_owned(self.timeouts.owned_kill_ms);
        }
        if !remaining.is_empty() {
            return Err(CliError::Validation(format!(
                "Owned processes have not exited: {remaining:?}; retained process receipt"
            )));
        }

        if let Err(err) = fs::remove_file(&self.process_receipt_path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            return Err(CliError::Io {
                context: "cannot remove state/process.json after clean drain".to_owned(),
                source: err,
            });
        }
        Ok(status)
    }
}

impl Drop for SupervisorGuard {
    fn drop(&mut self) {
        if !self.finished {
            let _ignored = self.finish(None);
        }
    }
}

fn resolve_tokenizer_and_template(
    target_dir: &Path,
    state_dir: &Path,
    configured: &Json,
) -> (PathBuf, PathBuf) {
    let derived = state_dir.join("gguf").join(crate::text_assets::DERIVED_DIR);
    let derived_tokenizer = derived.join("tokenizer.json");
    let derived_template = derived.join("chat_template.jinja");
    if derived_tokenizer.is_file() && derived_template.is_file() {
        return (derived_tokenizer, derived_template);
    }
    let mut tokenizer = target_dir.join("tokenizer.json");
    let mut template = target_dir.join("chat_template.jinja");
    if tokenizer.is_file() && template.is_file() {
        return (tokenizer, template);
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    let gguf_dir = state_dir.join("gguf");
    let internal_model = configured
        .find("internal_model")
        .ok()
        .flatten()
        .and_then(|v| v.string().ok())
        .and_then(|b| std::str::from_utf8(b).ok());

    if gguf_dir.is_dir()
        && let Ok(entries) = fs::read_dir(&gguf_dir)
    {
        let mut dirs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        if let Some(prefix) = internal_model {
            for d in &dirs {
                if d.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
                {
                    candidates.push(d.clone());
                }
            }
        }
        for d in dirs {
            if !candidates.contains(&d) {
                candidates.push(d);
            }
        }
    }
    if let Some(parent) = target_dir.parent()
        && parent.is_dir()
        && !candidates.contains(&parent.to_path_buf())
    {
        candidates.push(parent.to_path_buf());
    }

    for cand in candidates {
        let cand_tok = cand.join("tokenizer.json");
        let cand_tmpl = cand.join("chat_template.jinja");
        if cand_tok.is_file() && !tokenizer.is_file() {
            tokenizer = cand_tok;
        }
        if cand_tmpl.is_file() && !template.is_file() {
            template = cand_tmpl;
        }
        if tokenizer.is_file() && template.is_file() {
            break;
        }
    }
    (tokenizer, template)
}

pub(crate) fn default_derived_pack_path(tokspur_base: &Path, pack_v3: bool) -> PathBuf {
    let (asa_rel, legacy_rel) = if pack_v3 {
        (
            "derived/tokspur-pack-v3.asa.gguf",
            "derived/tokspur-pack-v3.gguf",
        )
    } else {
        (
            "derived/tokspur-pack-v2.asa.gguf",
            "derived/tokspur-pack-v2.gguf",
        )
    };
    let asa_path = tokspur_base.join(asa_rel);
    if asa_path.is_file() {
        return asa_path;
    }
    let legacy_path = tokspur_base.join(legacy_rel);
    if legacy_path.is_file() {
        return legacy_path;
    }
    asa_path
}

pub(crate) fn resolve_derived_pack_path(
    tokspur_base: &Path,
    pack_v3: bool,
    prefer_v3: bool,
) -> PathBuf {
    if !pack_v3 && prefer_v3 {
        for rel in [
            "derived/tokspur-pack-v3.asa.gguf",
            "derived/tokspur-pack-v3.gguf",
        ] {
            let candidate = tokspur_base.join(rel);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    default_derived_pack_path(tokspur_base, pack_v3)
}

/// Runs `tokspur serve [options]` and returns the process exit code (`0` on `SIGINT`/`SIGTERM` drain, otherwise the child's exit code).
pub fn serve_foreground(layout: &RuntimeLayout, opts: &ServeOptions) -> Result<i32, CliError> {
    validate_serve_options(opts)?;
    let tokspur_base = layout.tokspur_base()?;
    let state = private_dir(&tokspur_base, Path::new("state"))?;
    let _lock = StateLock::acquire(&tokspur_base)?;

    let configured = read_json_file(&tokspur_base.join("configured.json"))?;
    let adoption = read_json_file(&tokspur_base.join("adoption.json"))?;
    let assets_profile = read_json_file(&layout.assets_profile_path())?;
    check_adoption(&adoption, &assets_profile)?;

    let port = configured
        .find("port")
        .ok()
        .flatten()
        .and_then(|v| v.unsigned_integer().ok())
        .and_then(|p| u16::try_from(p).ok())
        .ok_or_else(|| CliError::Validation("missing valid port in configured.json".to_owned()))?;
    if !port_available(port) {
        return Err(CliError::Validation(
            "Requested port is occupied".to_owned(),
        ));
    }

    let workers = inference_processes()?;
    if !workers.is_empty() {
        return Err(CliError::Validation(format!(
            "Other inference workers found: {workers:?}. Stop them explicitly first."
        )));
    }

    let hw_profile = read_json_file(&layout.hardware_profile_path())?;
    let required_gib = hw_profile
        .find("hardware")
        .ok()
        .flatten()
        .and_then(|h| h.find("unified_memory_gib").ok().flatten())
        .and_then(|v| v.unsigned_integer().ok())
        .unwrap_or(128);
    let total_mem = hw_memsize().map_err(|source| CliError::Io {
        context: "cannot query hw.memsize".to_owned(),
        source,
    })?;
    if total_mem < required_gib.saturating_mul(1024 * 1024 * 1024) {
        return Err(CliError::Validation(format!(
            "The selected profile requires {required_gib} GiB unified memory"
        )));
    }

    // Verify source/build/package provenance and exact launch targets before touching weights.
    let targets = verify_build(layout)?;

    // The server reads and checks state/api-key itself, so the key never enters its environment.
    let env = isolated_environment(&tokspur_base)?;

    let target_obj = adoption
        .find("target")
        .ok()
        .flatten()
        .ok_or_else(|| CliError::Validation("adoption.json missing target".to_owned()))?;
    let target_dir = PathBuf::from(
        target_obj
            .find("directory")
            .ok()
            .flatten()
            .and_then(|v| v.string().ok())
            .and_then(|b| std::str::from_utf8(b).ok())
            .unwrap_or(""),
    );
    let (tokenizer, template) = resolve_tokenizer_and_template(&target_dir, &state, &configured);

    let pack_file = match opts.derived_pack.as_ref() {
        Some(custom) if custom.is_absolute() => custom.clone(),
        Some(custom) => std::env::current_dir()
            .map_err(|source| CliError::Io {
                context: "cannot resolve relative derived pack path".to_owned(),
                source,
            })?
            .join(custom),
        None => resolve_derived_pack_path(&tokspur_base, opts.pack_v3, opts.speed_preset),
    };
    if !opts.raw_gguf && !pack_file.is_file() {
        return Err(CliError::Validation(format!(
            "Lossless derived weight pack not found at {}; run pack construction or pass --raw-gguf to allow raw GGUF fallback",
            pack_file.display()
        )));
    }
    let derived_pack_flags = build_derived_pack_flags(opts, &pack_file);

    let speculative_path: PathBuf;
    let speculative_asset = if opts.drafter == "dflash2" {
        let raw_dflash2 = opts.dflash2_dir.as_deref().ok_or_else(|| {
            CliError::Validation("--drafter dflash2 requires --dflash2-dir".to_owned())
        })?;
        speculative_path = fs::canonicalize(raw_dflash2).map_err(|source| CliError::Io {
            context: format!(
                "DFlash2 checkpoint directory not found: {}",
                raw_dflash2.display()
            ),
            source,
        })?;
        if !speculative_path.is_dir() {
            return Err(CliError::Validation(format!(
                "DFlash2 checkpoint directory not found: {}",
                speculative_path.display()
            )));
        }
        SpeculativeAsset::DFlash2Dir(&speculative_path)
    } else {
        let draft_assets = adoption
            .find("draft")
            .ok()
            .flatten()
            .and_then(|d| d.find("assets").ok().flatten())
            .and_then(|a| a.array().ok())
            .ok_or_else(|| CliError::Validation("adoption.json missing draft assets".to_owned()))?;
        let mut mtp_path: Option<PathBuf> = None;
        for asset in draft_assets {
            if asset
                .find("sha256")
                .ok()
                .flatten()
                .and_then(|v| v.string().ok())
                == Some(MTP_SIDECAR_SHA256.as_bytes())
                && let Some(p) = asset
                    .find("path")
                    .ok()
                    .flatten()
                    .and_then(|v| v.string().ok())
                    .and_then(|b| std::str::from_utf8(b).ok())
            {
                mtp_path = Some(PathBuf::from(p));
                break;
            }
        }
        if mtp_path.is_none()
            && draft_assets.len() == 1
            && let Some(p) = draft_assets[0]
                .find("path")
                .ok()
                .flatten()
                .and_then(|v| v.string().ok())
                .and_then(|b| std::str::from_utf8(b).ok())
        {
            mtp_path = Some(PathBuf::from(p));
        }
        speculative_path = mtp_path.ok_or_else(|| {
            CliError::Validation(
                "Adopted draft assets do not contain verified MTP sidecar (blk.48); re-run tokspur adopt"
                    .to_owned(),
            )
        })?;
        if !speculative_path.is_file() {
            return Err(CliError::Validation(format!(
                "Adopted MTP sidecar file not found: {}",
                speculative_path.display()
            )));
        }
        SpeculativeAsset::MtpSidecar(&speculative_path)
    };

    let shards: Vec<String> = target_obj
        .find("assets")
        .ok()
        .flatten()
        .and_then(|a| a.array().ok())
        .ok_or_else(|| CliError::Validation("adoption.json missing target assets".to_owned()))?
        .iter()
        .filter_map(|a| a.find("path").ok().flatten())
        .filter_map(|v| v.string().ok())
        .filter_map(|b| std::str::from_utf8(b).ok())
        .map(str::to_owned)
        .collect();

    let model_alias = configured
        .find("model")
        .ok()
        .flatten()
        .and_then(|v| v.string().ok())
        .and_then(|b| std::str::from_utf8(b).ok())
        .unwrap_or("tokspur");

    let carrier_flags = build_carrier_flags(opts, speculative_asset, &state)?;

    // Declare SignalGuard before SupervisorGuard so the handlers remain active throughout the child-group drain.
    let _signal_guard = SignalGuard::install()?;
    if stop_requested() {
        return Ok(0);
    }

    let mut cmd = Command::new(&targets.server_bin);
    cmd.stdin(Stdio::piped())
        .env_clear()
        .envs(&env)
        .process_group(0)
        .arg("--worker")
        .arg(&targets.worker_bin)
        .arg("--policy")
        .arg(&targets.policy)
        .arg("--library")
        .arg(&targets.metallib)
        .arg("--tokenizer")
        .arg(&tokenizer)
        .arg("--template")
        .arg(&template)
        .arg("--state-dir")
        .arg(&state)
        .arg("--model-alias")
        .arg(model_alias)
        .arg("--port")
        .arg(port.to_string())
        .arg("--owner-lifetime-stdin")
        .arg("on")
        .arg("--request-timeout-seconds")
        .arg(opts.request_timeout_seconds.to_string())
        .arg("--allow-full-model")
        .arg("true")
        .args(&derived_pack_flags)
        .args(&carrier_flags)
        .args(&shards);

    let child = cmd.spawn().map_err(|source| CliError::Io {
        context: "failed to spawn tokspur-server".to_owned(),
        source,
    })?;
    let child_pid = child.id();

    let process_receipt = state.join("process.json");
    let mut supervisor = SupervisorGuard::new(child, process_receipt, opts.timeouts);

    // SupervisorGuard owns the child before process.json is written, so a failed receipt write still drains the child group with stop handlers active.
    write_private_json(
        &tokspur_base,
        Path::new("state/process.json"),
        &Json::new_object([
            ("pid", Json::integer(u64::from(child_pid))),
            ("supervisor", Json::integer(u64::from(std::process::id()))),
        ]),
    )?;

    let mut natural_exit: Option<ExitStatus> = None;
    while !stop_requested() {
        supervisor.refresh_descendants();
        if let Some(status) = supervisor.child.try_wait().map_err(|source| CliError::Io {
            context: "failed polling tokspur-server child".to_owned(),
            source,
        })? {
            natural_exit = Some(status);
            break;
        }
        pause(200);
    }

    let final_status = supervisor.finish(natural_exit)?;
    if stop_requested() {
        Ok(0)
    } else {
        Ok(exit_status_code(final_status))
    }
}
