//! Standalone Rust CLI library (`tokspur-cli`) implementing `doctor`, `adopt`, `configure`, `plan`, `verify-build`, `serve`, and `pack {plan,build}`.

#![forbid(unsafe_code)]

pub mod error;
pub mod layout;
mod model_config;
mod pack_command;
pub mod serve;
pub mod state;
mod text_assets;
pub mod verify;

use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use tokspur_gguf::ModelFiles;
use tokspur_json::Json;
use tokspur_pack::{
    DEFAULT_CHUNK_BYTES, DEFAULT_RESERVE_BYTES, build_derived_pack_with_progress, plan_static_pack,
};
use tokspur_sys::net::port_available;

use crate::error::CliError;
use crate::layout::RuntimeLayout;
use crate::serve::{DrainTimeouts, ServeOptions, serve_foreground};
use crate::state::{
    StateLock, adopt_weights, check_adopted_weights, configure_runtime, format_pretty_sorted_json,
    inference_processes, private_dir, read_json_file,
};
use crate::verify::verify_build;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Help topic for `--help` and `-h` requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HelpTopic {
    General,
    Doctor,
    CheckModelConfig,
    Adopt,
    Configure,
    Plan,
    VerifyBuild,
    Serve,
    Pack,
    PackPlan,
    PackBuild,
}

/// Lossless derived pack format selected for planning or construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackFormat {
    V2,
    V3,
}

impl PackFormat {
    fn parse(value: Option<&String>) -> Result<Self, CliError> {
        match value.map(String::as_str) {
            Some("v2") => Ok(Self::V2),
            Some("v3") => Ok(Self::V3),
            _ => Err(CliError::Usage("--format requires v2 or v3".to_owned())),
        }
    }

    const fn default_chunk_mib(self) -> u64 {
        match self {
            Self::V2 => DEFAULT_CHUNK_BYTES / (1 << 20),
            Self::V3 => 16,
        }
    }

    const fn filename(self) -> &'static str {
        match self {
            Self::V2 => "tokspur-pack-v2.asa.gguf",
            Self::V3 => "tokspur-pack-v3.asa.gguf",
        }
    }
}

/// Parsed CLI subcommand.
#[derive(Clone, PartialEq)]
pub enum Command {
    PrintVersion,
    PrintHelp(HelpTopic),
    Doctor,
    CheckModelConfig(PathBuf),
    Adopt {
        target: Option<PathBuf>,
        draft: Option<PathBuf>,
        check_only: bool,
    },
    Configure {
        port: Option<u32>,
    },
    Plan,
    VerifyBuild,
    Serve(Box<ServeOptions>),
    PackPlan {
        format: PackFormat,
        target_dir: Option<PathBuf>,
        chunk_mib: Option<u64>,
        reserve_gb: f64,
        json: bool,
        check: bool,
    },
    PackBuild {
        format: PackFormat,
        output: Option<PathBuf>,
        reserve_gb: f64,
        chunk_mib: u64,
    },
}

impl fmt::Debug for Command {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrintVersion => formatter.write_str("PrintVersion"),
            Self::PrintHelp(topic) => formatter.debug_tuple("PrintHelp").field(topic).finish(),
            Self::Doctor => formatter.write_str("Doctor"),
            Self::CheckModelConfig(_) => formatter.write_str("CheckModelConfig([redacted])"),
            Self::Adopt {
                target,
                draft,
                check_only,
            } => formatter
                .debug_struct("Adopt")
                .field("target", &target.as_ref().map(|_| "[redacted]"))
                .field("draft", &draft.as_ref().map(|_| "[redacted]"))
                .field("check_only", check_only)
                .finish(),
            Self::Configure { port } => formatter
                .debug_struct("Configure")
                .field("port", port)
                .finish(),
            Self::Plan => formatter.write_str("Plan"),
            Self::VerifyBuild => formatter.write_str("VerifyBuild"),
            Self::Serve(opts) => formatter.debug_tuple("Serve").field(opts).finish(),
            Self::PackPlan {
                format,
                target_dir,
                chunk_mib,
                reserve_gb,
                json,
                check,
            } => formatter
                .debug_struct("PackPlan")
                .field("format", format)
                .field("target_dir", &target_dir.as_ref().map(|_| "[redacted]"))
                .field("chunk_mib", chunk_mib)
                .field("reserve_gb", reserve_gb)
                .field("json", json)
                .field("check", check)
                .finish(),
            Self::PackBuild {
                format,
                output,
                reserve_gb,
                chunk_mib,
            } => formatter
                .debug_struct("PackBuild")
                .field("format", format)
                .field("output", &output.as_ref().map(|_| "[redacted]"))
                .field("reserve_gb", reserve_gb)
                .field("chunk_mib", chunk_mib)
                .finish(),
        }
    }
}

/// Parsed CLI invocation including optional `--state-root`.
#[derive(Clone, PartialEq)]
pub struct CliArgs {
    pub state_root: Option<PathBuf>,
    pub command: Command,
}

impl fmt::Debug for CliArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CliArgs")
            .field(
                "state_root",
                &self.state_root.as_ref().map(|_| "[redacted]"),
            )
            .field("command", &self.command)
            .finish()
    }
}

const HELP_TEXT: &str = "\
usage: tokspur [--version] [--state-root <abs-path>] <command> [options]

commands:
  check-model-config  Compare a local config.json with the Flash-Next text profile
  doctor            Read-only readiness summary; never loads a model
  adopt             Fully hash existing weights without copying them
  configure         Write isolated settings; never starts inference
  plan              Show the selected native policy
  verify-build      Verify built or staged executables, metallib and provenance
  serve             Foreground API; refuses other inference workers
  pack plan         Plan the selected lossless pack layout and check 50 GiB reserve
  pack build        Stream and seal a lossless pack with byte-exact inverse proof
";

const DOCTOR_HELP_TEXT: &str = "\
usage: tokspur [--state-root <abs-path>] doctor [-h | --help]

Read-only readiness summary; never loads a model.
";

const ADOPT_HELP_TEXT: &str = "\
usage: tokspur [--state-root <abs-path>] adopt [--target <dir> --draft <dir> | --check-only]

options:
  --target <dir>    Existing target GGUF directory to verify in place
  --draft <dir>     Existing MTP draft sidecar directory to verify in place
  --check-only      Verify existing adopted weights without rehashing
  -h, --help        Show this help message and exit
";

const CONFIGURE_HELP_TEXT: &str = "\
usage: tokspur [--state-root <abs-path>] configure [--port <port>]

options:
  --port <port>     Unprivileged local TCP port (1024..65535)
  -h, --help        Show this help message and exit
";

const PLAN_HELP_TEXT: &str = "\
usage: tokspur [--state-root <abs-path>] plan [-h | --help]

Show the selected native hardware and policy profile.
";

const VERIFY_BUILD_HELP_TEXT: &str = "\
usage: tokspur [--state-root <abs-path>] verify-build [-h | --help]

Verify build-manifest.json or staged package provenance and launch artefacts.
";

const SERVE_HELP_TEXT: &str = "\
usage: tokspur [--state-root <abs-path>] serve [options]

options:
  --raw-gguf                                  Explicit opt-in to raw GGUF instead of derived pack
  --pack-v3                                   Require planar derived pack v3 (the default when installed; v2 is the fallback)
  --derived-pack <file>                       Explicit derived pack GGUF file path
  --gdn-prefill-only                          Retired GDN prefill experiment
  --derived-pack-trust-mode {full,local-seal} Derived pack verification trust mode (default: full)
  --output-packet {8,64}                      Speculative output packet size (default: 8)
  --first-output-packet <1..64>               Opt-in initial speculative output packet cap before steady-state packets
  --prefill-dense {f32-staged,nax,f32,f16}    Dense Q8 prefill kernel mode (default: f32-staged)
  --prefill-grouped {f32,auto,serial,nax}     Grouped MoE prefill kernel mode (default: f32)
  --known-token-capacity {512,1024,2048}           Known-token IPC packet capacity (default: 2048)
  --native-host-logit-views {off,on}          Opt-in native host logit views (default: off)
  --native-logit-topk {off,on}                Opt-in GPU two-stage logit top-k verification (default: off)
  --native-ple-async-gather {off,on}          Opt-in asynchronous 16-bucket PLE gather overlap (default: off)
  --native-mtp-fast-round {off,on}            Opt-in GPU-resident MTP fast round and fused priming (default: off)
  --native-verify-hc-gdn-fused {off,on}       Opt-in fused HC+GDN multi-row verification (default: off)
  --native-dflash2-defer-context {off,on}     Opt-in native deferred DFlash2 context (default: off)
  --native-prepared-validation {off,on}     Opt-in scoped sampling scan reuse (default: off)
  --worker-burst-pacing {off,on}            Opt-in pacing between worker packets (default: off)
  --native-verify-cost-profile PATH         Qualified measured controller costs (default: absent)
  --native-dflash2-draft-policy {always,auto} Opt-in DFlash2 drafting decisions (default: always)
  --native-dflash2-verify-rows {all,auto}     Opt-in DFlash2 verify row budget (default: all)
  --dflash2-max-drafts <1..7>                 Opt-in DFlash2 maximum drafted proposals per round
  --dflash2-vocab-slice <n>                   Opt-in DFlash2 draft vocabulary slice (0 or multiple of 32)
  --mtp-draft-slice                           Opt-in top-32K MTP draft vocabulary slice
  --verify-nax, --no-verify-nax               Opt-in Metal 4 NAX batched speculative verification (default: off)
  --native-verify-nax {off,on}                Explicit native batched verification mode (default: off)
  --verify-nax-m8                             Opt-in Metal 4 matmul2d(8,32,32) speculative verification
  --native-verify-m8-splits <path>             Explicit M8 split policy (requires --verify-nax-m8)
  --request-timeout-seconds <1..86400>         Overall request deadline (default: 3600 seconds)
  --speed-preset                              Opt-in Pack v3, NAX, MTP3/DFlash2, first8/output64 and executor speed preset
  --mtp-depth {2,3}                           Opt-in MTP speculative lookahead depth (default: 2)
  --drafter {mtp,dflash2}                     Speculative drafter mode (default: mtp)
  --dflash2-dir <dir>                         Opt-in DFlash2 safetensors checkpoint directory
  --dflash2-precision {q8_0,bf16}             Opt-in DFlash2 dense projection precision (default: q8_0)
  --concurrent-dispatch, --no-concurrent-dispatch Metal concurrent compute dispatch (default: on)
  --q8-gemv-rg4 {off,on,wide}                 Retired Q8 GEMV experiment (default: off)
  --q8-three-msg {off,on,wide}                Retired multi-SG Q8 experiment (default: off)
  --prefill-multiseg, --no-prefill-multiseg   Retired multi-segment prefill (default: off)
  --prefill-swizzle, --no-prefill-swizzle     Swizzled QSA prefill projection (default: on)
  --qsa-prefill-fused, --no-qsa-prefill-fused Fused QSA prefill operator (default: on)
  --hc-consumer-fused, --no-hc-consumer-fused Fused HC consumer operator (default: on)
  --ple-overlap, --no-ple-overlap             Chunk-ahead PLE prefill overlap (default: on)
  --shared-prefix-cache, --no-shared-prefix-cache Host-RAM shared system-prefix cache (default: on for mtp)
  --mtp-lookup {off,bypass,hybrid}            Opt-in MTP prompt-lookup mode (default: off)
  --mtp-lookup-match <2..16>                  Opt-in MTP prompt-lookup match length
  --tool-enum-validation {strict,executor}    OpenAI tool enum schema policy (default: strict)
  --tool-arg-validation {strict,executor}     OpenAI tool argument schema policy (default: strict)
  --private-prefix-trace                      Opt-in private prefix reuse trace telemetry
  --telemetry-stdout                          Opt-in structured JSONL telemetry on stdout
  --checkpoint-dir, --checkpoint-ssd-dir <dir> Custom SSD checkpoint directory (default: .tokspur/state/checkpoints)
  --checkpoint-cap-gib, --checkpoint-ssd-budget-gib <gib> SSD checkpoint capacity budget in GiB (0.0..32.0, default: 0.0, opt in with 8.0)
  --checkpoint-floor-gib, --checkpoint-ssd-floor-gib <gib> APFS free space floor in GiB (0.0..100.0, default: 20.0)
  --checkpoint-policy {recompute-cost,lru}    Eviction priority policy (default: recompute-cost)
  --overlap-prefill                           Enable 16-token overlap background prefill with boundary protection
  --drain-timeouts-ms <ms,ms,ms,ms>           Graceful drain timeouts in milliseconds
  -h, --help                                  Show this help message and exit
";

const PACK_HELP_TEXT: &str = "\
usage: tokspur [--state-root <abs-path>] pack <plan|build> [options]

subcommands:
  plan              Plan the selected lossless pack layout and check 50 GiB reserve
  build             Stream and seal a lossless pack with byte-exact inverse proof
";

const PACK_PLAN_HELP_TEXT: &str = "\
usage: tokspur [--state-root <abs-path>] pack plan [--format v2|v3] [--target-dir <dir>] [--reserve-gb <gb>] [--chunk-mib <mib>] [--json] [--check]

options:
  --format <v2|v3>   Pack layout (default: v3); v3 plans use adopted shards and the MTP sidecar
  --target-dir <dir> Directory whose volume free space is checked against reserve
  --reserve-gb <gb>  Required free disk reserve in GiB (default: 50.0)
  --chunk-mib <mib>  V3 streaming chunk size in MiB (1..64, default: 16)
  --json             Emit structured JSON plan output
  --check            Exit non-zero if the disk or streaming RAM budget check fails
  -h, --help         Show this help message and exit
";

const PACK_BUILD_HELP_TEXT: &str = "\
usage: tokspur [--state-root <abs-path>] pack build [--format v2|v3] [--output <file>] [--reserve-gb <gb>] [--chunk-mib <mib>]

options:
  --format <v2|v3>   Pack layout (default: v3)
  --output <file>    Destination file (default: derived/tokspur-pack-<format>.asa.gguf); v3 requires a .gguf suffix
  --reserve-gb <gb>  Required free disk reserve in GiB (default: 50.0)
  --chunk-mib <mib>  Streaming copy chunk size in MiB (v2 default: 4; v3: 1..64, default: 16)
  -h, --help         Show this help message and exit
";

impl HelpTopic {
    const fn text(self) -> &'static str {
        match self {
            Self::General => HELP_TEXT,
            Self::Doctor => DOCTOR_HELP_TEXT,
            Self::CheckModelConfig => {
                "usage: tokspur check-model-config <config.json>\n\nCompare the text architecture and dimensions offline before downloading weights. Exit 0 means the checked fields match; asset adoption is a separate check.\n"
            }
            Self::Adopt => ADOPT_HELP_TEXT,
            Self::Configure => CONFIGURE_HELP_TEXT,
            Self::Plan => PLAN_HELP_TEXT,
            Self::VerifyBuild => VERIFY_BUILD_HELP_TEXT,
            Self::Serve => SERVE_HELP_TEXT,
            Self::Pack => PACK_HELP_TEXT,
            Self::PackPlan => PACK_PLAN_HELP_TEXT,
            Self::PackBuild => PACK_BUILD_HELP_TEXT,
        }
    }
}

pub fn parse_args<I, S>(args: I) -> Result<CliArgs, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let raw: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut state_root: Option<PathBuf> = None;
    let mut filtered: Vec<String> = Vec::with_capacity(raw.len());
    let mut idx = 0usize;
    while idx < raw.len() {
        match raw[idx].as_str() {
            "--state-root" => {
                idx += 1;
                let Some(val) = raw.get(idx) else {
                    return Err(CliError::Usage(
                        "--state-root requires an argument".to_owned(),
                    ));
                };
                state_root = Some(PathBuf::from(val));
            }
            arg if arg.starts_with("--state-root=") => {
                let val = &arg["--state-root=".len()..];
                state_root = Some(PathBuf::from(val));
            }
            _ => filtered.push(raw[idx].clone()),
        }
        idx += 1;
    }

    if filtered.is_empty() {
        return Err(CliError::Usage(HELP_TEXT.to_owned()));
    }
    match filtered[0].as_str() {
        "--version" | "-V" => {
            return Ok(CliArgs {
                state_root,
                command: Command::PrintVersion,
            });
        }
        "--help" | "-h" | "help" => {
            return Ok(CliArgs {
                state_root,
                command: Command::PrintHelp(HelpTopic::General),
            });
        }
        _ => {}
    }

    let sub = filtered[0].as_str();
    let rest = &filtered[1..];
    let command = match sub {
        "check-model-config" => match rest {
            [arg] if matches!(arg.as_str(), "--help" | "-h") => {
                Command::PrintHelp(HelpTopic::CheckModelConfig)
            }
            [path] if !path.starts_with('-') => Command::CheckModelConfig(PathBuf::from(path)),
            _ => {
                return Err(CliError::Usage(
                    "usage: tokspur check-model-config <config.json>".to_owned(),
                ));
            }
        },
        "doctor" => {
            if matches!(rest.first().map(String::as_str), Some("--help" | "-h")) {
                return Ok(CliArgs {
                    state_root,
                    command: Command::PrintHelp(HelpTopic::Doctor),
                });
            }
            reject_extra_args("doctor", rest)?;
            Command::Doctor
        }
        "plan" => {
            if matches!(rest.first().map(String::as_str), Some("--help" | "-h")) {
                return Ok(CliArgs {
                    state_root,
                    command: Command::PrintHelp(HelpTopic::Plan),
                });
            }
            reject_extra_args("plan", rest)?;
            Command::Plan
        }
        "verify-build" => {
            if matches!(rest.first().map(String::as_str), Some("--help" | "-h")) {
                return Ok(CliArgs {
                    state_root,
                    command: Command::PrintHelp(HelpTopic::VerifyBuild),
                });
            }
            reject_extra_args("verify-build", rest)?;
            Command::VerifyBuild
        }
        "adopt" => {
            let mut target = None;
            let mut draft = None;
            let mut check_only = false;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--target" => {
                        i += 1;
                        let val = rest.get(i).ok_or_else(|| {
                            CliError::Usage("--target requires a directory path".to_owned())
                        })?;
                        target = Some(PathBuf::from(val));
                    }
                    "--draft" => {
                        i += 1;
                        let val = rest.get(i).ok_or_else(|| {
                            CliError::Usage("--draft requires a directory path".to_owned())
                        })?;
                        draft = Some(PathBuf::from(val));
                    }
                    "--check-only" => check_only = true,
                    "--help" | "-h" => {
                        return Ok(CliArgs {
                            state_root,
                            command: Command::PrintHelp(HelpTopic::Adopt),
                        });
                    }
                    other => {
                        return Err(CliError::Usage(format!(
                            "unrecognised adopt argument: {other}"
                        )));
                    }
                }
                i += 1;
            }
            Command::Adopt {
                target,
                draft,
                check_only,
            }
        }
        "configure" => {
            let mut port = None;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--port" => {
                        i += 1;
                        let val = rest.get(i).ok_or_else(|| {
                            CliError::Usage("--port requires an integer".to_owned())
                        })?;
                        let parsed = val.parse::<u32>().map_err(|_| {
                            CliError::Usage(format!("invalid --port integer: {val}"))
                        })?;
                        port = Some(parsed);
                    }
                    "--help" | "-h" => {
                        return Ok(CliArgs {
                            state_root,
                            command: Command::PrintHelp(HelpTopic::Configure),
                        });
                    }
                    other => {
                        return Err(CliError::Usage(format!(
                            "unrecognised configure argument: {other}"
                        )));
                    }
                }
                i += 1;
            }
            Command::Configure { port }
        }
        "serve" => {
            let mut opts = ServeOptions::default();
            let mut prefill_dense_explicit = false;
            let mut prefill_grouped_explicit = false;
            let mut verify_nax_explicit = false;
            let mut prepared_validation_explicit = false;
            let mut output_packet_explicit = false;
            let mut tool_enum_explicit = false;
            let mut tool_arg_explicit = false;
            let mut i = 0;
            while i < rest.len() {
                let (flag, inline_val) = match rest[i].split_once('=') {
                    Some((k, v)) if k.starts_with("--") => (k, Some(v)),
                    _ => (rest[i].as_str(), None),
                };
                let take_val = |flag: &str,
                                inline: Option<&str>,
                                idx: &mut usize,
                                hint: &str|
                 -> Result<String, CliError> {
                    if let Some(v) = inline {
                        return Ok(v.to_owned());
                    }
                    *idx += 1;
                    rest.get(*idx)
                        .cloned()
                        .ok_or_else(|| CliError::Usage(format!("{flag} {hint}")))
                };
                let forbid_val = |flag: &str, inline: Option<&str>| -> Result<(), CliError> {
                    if inline.is_some() {
                        return Err(CliError::Usage(format!("{flag} does not take a value")));
                    }
                    Ok(())
                };
                match flag {
                    "--raw-gguf" => {
                        forbid_val(flag, inline_val)?;
                        opts.raw_gguf = true;
                    }
                    "--pack-v3" => {
                        forbid_val(flag, inline_val)?;
                        opts.pack_v3 = true;
                    }
                    "--derived-pack" => {
                        let val = take_val(flag, inline_val, &mut i, "requires a GGUF file path")?;
                        if val.is_empty() {
                            return Err(CliError::Usage(
                                "--derived-pack cannot be empty".to_owned(),
                            ));
                        }
                        opts.derived_pack = Some(PathBuf::from(val));
                    }
                    "--gdn-prefill-only" => {
                        forbid_val(flag, inline_val)?;
                        opts.gdn_prefill_only = true;
                    }
                    "--derived-pack-trust-mode" => {
                        let val =
                            take_val(flag, inline_val, &mut i, "requires full or local-seal")?;
                        if val != "full" && val != "local-seal" {
                            return Err(CliError::Usage(format!(
                                "invalid --derived-pack-trust-mode '{val}' (expected full or local-seal)"
                            )));
                        }
                        opts.derived_pack_trust_mode = val;
                    }
                    "--output-packet" => {
                        let val = take_val(flag, inline_val, &mut i, "requires 8 or 64")?;
                        let pkt = val.parse::<u32>().map_err(|_| {
                            CliError::Usage(format!("invalid --output-packet integer: {val}"))
                        })?;
                        if pkt != 8 && pkt != 64 {
                            return Err(CliError::Usage(format!(
                                "invalid --output-packet '{pkt}' (expected 8 or 64)"
                            )));
                        }
                        opts.output_packet = pkt;
                        output_packet_explicit = true;
                    }
                    "--first-output-packet" | "--native-mtp-first-output-packet" => {
                        let val = take_val(flag, inline_val, &mut i, "requires 1..64")?;
                        let pkt = val.parse::<u32>().map_err(|_| {
                            CliError::Usage(format!("invalid {flag} integer: {val}"))
                        })?;
                        if !(1..=64).contains(&pkt) {
                            return Err(CliError::Usage(format!(
                                "invalid {flag} '{pkt}' (expected 1..64)"
                            )));
                        }
                        opts.first_output_packet = Some(pkt);
                    }
                    "--prefill-dense" => {
                        let val = take_val(
                            flag,
                            inline_val,
                            &mut i,
                            "requires f32-staged, nax, f32 or f16",
                        )?;
                        if !matches!(val.as_str(), "f32-staged" | "nax" | "f32" | "f16") {
                            return Err(CliError::Usage(format!(
                                "invalid --prefill-dense '{val}' (expected f32-staged, nax, f32 or f16)"
                            )));
                        }
                        opts.prefill_dense = val;
                        prefill_dense_explicit = true;
                    }
                    "--prefill-grouped" => {
                        let val = take_val(
                            flag,
                            inline_val,
                            &mut i,
                            "requires f32, auto, serial or nax",
                        )?;
                        if !matches!(val.as_str(), "f32" | "auto" | "serial" | "nax") {
                            return Err(CliError::Usage(format!(
                                "invalid --prefill-grouped '{val}' (expected f32, auto, serial or nax)"
                            )));
                        }
                        opts.prefill_grouped = val;
                        prefill_grouped_explicit = true;
                    }
                    "--known-token-capacity" => {
                        let val = take_val(flag, inline_val, &mut i, "requires 512, 1024 or 2048")?;
                        let cap = val.parse::<u32>().map_err(|_| {
                            CliError::Usage(format!(
                                "invalid --known-token-capacity integer: {val}"
                            ))
                        })?;
                        if !matches!(cap, 512 | 1024 | 2048) {
                            return Err(CliError::Usage(format!(
                                "invalid --known-token-capacity '{cap}' (expected 512, 1024 or 2048)"
                            )));
                        }
                        opts.known_token_capacity = cap;
                    }
                    "--native-host-logit-views"
                    | "--native-logit-topk"
                    | "--native-ple-async-gather"
                    | "--native-mtp-fast-round"
                    | "--mtp-fast-round"
                    | "--native-prefill-router-fast"
                    | "--prefill-router-fast"
                    | "--native-verify-hc-gdn-fused"
                    | "--native-dflash2-defer-context"
                    | "--native-verify-nax" => {
                        let value = take_val(flag, inline_val, &mut i, "requires off or on")?;
                        let enabled = match value.as_str() {
                            "off" => false,
                            "on" => true,
                            _ => {
                                return Err(CliError::Usage(format!(
                                    "invalid {flag} '{value}' (expected off or on)"
                                )));
                            }
                        };
                        match flag {
                            "--native-host-logit-views" => opts.host_logit_views = Some(enabled),
                            "--native-logit-topk" => opts.logit_topk = Some(enabled),
                            "--native-ple-async-gather" => opts.ple_async_gather = Some(enabled),
                            "--native-mtp-fast-round" | "--mtp-fast-round" => {
                                opts.mtp_fast_round = Some(enabled);
                            }
                            "--native-prefill-router-fast" | "--prefill-router-fast" => {
                                opts.prefill_router_fast = Some(enabled);
                            }
                            "--native-verify-hc-gdn-fused" => {
                                opts.verify_hc_gdn_fused = Some(enabled);
                            }
                            "--native-verify-nax" => {
                                opts.verify_nax = enabled;
                                opts.verify_nax_selection = Some(enabled);
                                verify_nax_explicit = true;
                            }
                            _ => opts.dflash2_defer_context = Some(enabled),
                        }
                    }
                    "--mtp-draft-slice" => {
                        forbid_val(flag, inline_val)?;
                        opts.mtp_draft_slice = true;
                    }
                    "--verify-nax" => {
                        forbid_val(flag, inline_val)?;
                        opts.verify_nax = true;
                        opts.verify_nax_selection = Some(true);
                        verify_nax_explicit = true;
                    }
                    "--no-verify-nax" => {
                        forbid_val(flag, inline_val)?;
                        opts.verify_nax = false;
                        opts.verify_nax_selection = Some(false);
                        verify_nax_explicit = true;
                    }
                    "--verify-nax-m8" => {
                        forbid_val(flag, inline_val)?;
                        opts.verify_nax_m8 = true;
                    }
                    "--request-timeout-seconds" => {
                        let value = take_val(flag, inline_val, &mut i, "requires seconds")?;
                        let seconds = value.parse::<u32>().map_err(|_| {
                            CliError::Usage("request timeout requires 1..86400 seconds".to_owned())
                        })?;
                        if !(1..=86400).contains(&seconds) {
                            return Err(CliError::Usage(
                                "request timeout requires 1..86400 seconds".to_owned(),
                            ));
                        }
                        opts.request_timeout_seconds = seconds;
                    }
                    "--native-dflash2-verify-rows" => {
                        let val = take_val(flag, inline_val, &mut i, "requires all or auto")?;
                        if val != "all" && val != "auto" {
                            return Err(CliError::Usage(format!(
                                "invalid --native-dflash2-verify-rows '{val}' (expected all or auto)"
                            )));
                        }
                        opts.dflash2_verify_rows = Some(val);
                    }
                    "--native-prepared-validation" => {
                        let val = take_val(flag, inline_val, &mut i, "requires off or on")?;
                        if !matches!(val.as_str(), "off" | "on") {
                            return Err(CliError::Usage(
                                "--native-prepared-validation requires off or on".to_owned(),
                            ));
                        }
                        opts.prepared_validation = val == "on";
                        prepared_validation_explicit = true;
                    }
                    "--worker-burst-pacing" => {
                        let val = take_val(flag, inline_val, &mut i, "requires off or on")?;
                        if !matches!(val.as_str(), "off" | "on") {
                            return Err(CliError::Usage(
                                "--worker-burst-pacing requires off or on".to_owned(),
                            ));
                        }
                        opts.worker_burst_pacing = val == "on";
                    }
                    "--native-verify-m8-splits" => {
                        opts.verify_m8_splits = Some(PathBuf::from(take_val(
                            flag,
                            inline_val,
                            &mut i,
                            "requires a path",
                        )?));
                    }
                    "--native-verify-cost-profile" => {
                        opts.verify_cost_profile = Some(PathBuf::from(take_val(
                            flag,
                            inline_val,
                            &mut i,
                            "requires an absolute path",
                        )?));
                    }
                    "--native-dflash2-draft-policy" => {
                        let val = take_val(flag, inline_val, &mut i, "requires always or auto")?;
                        if !matches!(val.as_str(), "always" | "auto") {
                            return Err(CliError::Usage(
                                "--native-dflash2-draft-policy requires always or auto".to_owned(),
                            ));
                        }
                        opts.dflash2_draft_policy = val;
                    }
                    "--speed-preset" => {
                        forbid_val(flag, inline_val)?;
                        opts.speed_preset = true;
                    }
                    "--mtp-depth" => {
                        let val = take_val(flag, inline_val, &mut i, "requires 2 or 3")?;
                        let depth = val.parse::<u32>().map_err(|_| {
                            CliError::Usage(format!("invalid --mtp-depth integer: {val}"))
                        })?;
                        if !matches!(depth, 2 | 3) {
                            return Err(CliError::Usage(format!(
                                "invalid --mtp-depth '{depth}' (expected 2 or 3)"
                            )));
                        }
                        opts.mtp_depth = Some(depth);
                    }
                    "--drafter" => {
                        let val = take_val(flag, inline_val, &mut i, "requires mtp or dflash2")?;
                        if val != "mtp" && val != "dflash2" {
                            return Err(CliError::Usage(format!(
                                "invalid --drafter '{val}' (expected mtp or dflash2)"
                            )));
                        }
                        opts.drafter = val;
                    }
                    "--dflash2-dir" => {
                        let val = take_val(flag, inline_val, &mut i, "requires a directory path")?;
                        if val.is_empty() {
                            return Err(CliError::Usage(
                                "--dflash2-dir cannot be empty".to_owned(),
                            ));
                        }
                        opts.dflash2_dir = Some(PathBuf::from(val));
                    }
                    "--dflash2-precision" => {
                        let val = take_val(flag, inline_val, &mut i, "requires q8_0 or bf16")?;
                        if !matches!(val.as_str(), "q8_0" | "bf16") {
                            return Err(CliError::Usage(format!(
                                "invalid --dflash2-precision '{val}' (expected q8_0 or bf16)"
                            )));
                        }
                        opts.dflash2_precision = Some(val);
                    }
                    "--dflash2-max-drafts" | "--native-dflash2-max-drafts" => {
                        let val = take_val(flag, inline_val, &mut i, "requires 1..7")?;
                        let md = val.parse::<u32>().map_err(|_| {
                            CliError::Usage(format!("invalid {flag} integer: {val}"))
                        })?;
                        if !(1..=7).contains(&md) {
                            return Err(CliError::Usage(format!(
                                "invalid {flag} '{md}' (expected 1..7)"
                            )));
                        }
                        opts.dflash2_max_drafts = Some(md);
                    }
                    "--dflash2-vocab-slice" | "--native-dflash2-vocab-slice" => {
                        let val = take_val(
                            flag,
                            inline_val,
                            &mut i,
                            "requires 0, full or multiple of 32",
                        )?;
                        let vs = if val == "full" {
                            0u32
                        } else {
                            val.parse::<u32>().map_err(|_| {
                                CliError::Usage(format!("invalid {flag} integer: {val}"))
                            })?
                        };
                        if !(vs == 0 || ((256..248_320).contains(&vs) && vs.is_multiple_of(32))) {
                            return Err(CliError::Usage(format!(
                                "invalid {flag} '{val}' (expected 0, full or a multiple of 32 in 256..248319)"
                            )));
                        }
                        opts.dflash2_vocab_slice = Some(vs);
                    }
                    "--concurrent-dispatch" => {
                        forbid_val(flag, inline_val)?;
                        opts.concurrent_dispatch = true;
                    }
                    "--no-concurrent-dispatch" => {
                        forbid_val(flag, inline_val)?;
                        opts.concurrent_dispatch = false;
                    }
                    "--q8-gemv-rg4" => {
                        let val = take_val(flag, inline_val, &mut i, "requires off, on or wide")?;
                        if !matches!(val.as_str(), "off" | "on" | "wide") {
                            return Err(CliError::Usage(format!(
                                "invalid --q8-gemv-rg4 '{val}' (expected off, on or wide)"
                            )));
                        }
                        opts.q8_gemv_rg4 = val;
                    }
                    "--q8-three-msg" => {
                        let val = take_val(flag, inline_val, &mut i, "requires off, on or wide")?;
                        if !matches!(val.as_str(), "off" | "on" | "wide") {
                            return Err(CliError::Usage(format!(
                                "invalid --q8-three-msg '{val}' (expected off, on or wide)"
                            )));
                        }
                        opts.q8_three_msg = val;
                    }
                    "--prefill-multiseg" => {
                        forbid_val(flag, inline_val)?;
                        opts.prefill_multiseg = true;
                    }
                    "--no-prefill-multiseg" => {
                        forbid_val(flag, inline_val)?;
                        opts.prefill_multiseg = false;
                    }
                    "--prefill-swizzle" => {
                        forbid_val(flag, inline_val)?;
                        opts.prefill_swizzle = true;
                    }
                    "--no-prefill-swizzle" => {
                        forbid_val(flag, inline_val)?;
                        opts.prefill_swizzle = false;
                    }
                    "--qsa-prefill-fused" => {
                        forbid_val(flag, inline_val)?;
                        opts.qsa_prefill_fused = true;
                    }
                    "--no-qsa-prefill-fused" => {
                        forbid_val(flag, inline_val)?;
                        opts.qsa_prefill_fused = false;
                    }
                    "--hc-consumer-fused" => {
                        forbid_val(flag, inline_val)?;
                        opts.hc_consumer_fused = true;
                    }
                    "--no-hc-consumer-fused" => {
                        forbid_val(flag, inline_val)?;
                        opts.hc_consumer_fused = false;
                    }
                    "--ple-overlap" => {
                        forbid_val(flag, inline_val)?;
                        opts.ple_overlap = true;
                    }
                    "--no-ple-overlap" => {
                        forbid_val(flag, inline_val)?;
                        opts.ple_overlap = false;
                    }
                    "--shared-prefix-cache" => {
                        forbid_val(flag, inline_val)?;
                        opts.shared_prefix_cache = Some(true);
                    }
                    "--no-shared-prefix-cache" => {
                        forbid_val(flag, inline_val)?;
                        opts.shared_prefix_cache = Some(false);
                    }
                    "--mtp-lookup" => {
                        let val =
                            take_val(flag, inline_val, &mut i, "requires off, bypass or hybrid")?;
                        if !matches!(val.as_str(), "off" | "bypass" | "hybrid") {
                            return Err(CliError::Usage(format!(
                                "invalid --mtp-lookup '{val}' (expected off, bypass or hybrid)"
                            )));
                        }
                        opts.mtp_lookup = val;
                    }
                    "--mtp-lookup-match" => {
                        let val = take_val(flag, inline_val, &mut i, "requires an integer 2..16")?;
                        let match_len = val.parse::<u32>().map_err(|_| {
                            CliError::Usage(format!("invalid --mtp-lookup-match integer: {val}"))
                        })?;
                        if !(2..=16).contains(&match_len) {
                            return Err(CliError::Usage(format!(
                                "invalid --mtp-lookup-match '{match_len}' (expected 2..16)"
                            )));
                        }
                        opts.mtp_lookup_match = Some(match_len);
                    }
                    "--tool-enum-validation" => {
                        let val =
                            take_val(flag, inline_val, &mut i, "requires strict or executor")?;
                        if val != "strict" && val != "executor" {
                            return Err(CliError::Usage(format!(
                                "invalid --tool-enum-validation '{val}' (expected strict or executor)"
                            )));
                        }
                        opts.tool_enum_validation = val;
                        tool_enum_explicit = true;
                    }
                    "--tool-arg-validation" => {
                        let val =
                            take_val(flag, inline_val, &mut i, "requires strict or executor")?;
                        if val != "strict" && val != "executor" {
                            return Err(CliError::Usage(format!(
                                "invalid --tool-arg-validation '{val}' (expected strict or executor)"
                            )));
                        }
                        opts.tool_arg_validation = val;
                        tool_arg_explicit = true;
                    }
                    "--private-prefix-trace" => {
                        forbid_val(flag, inline_val)?;
                        opts.private_prefix_trace = true;
                    }
                    "--telemetry-stdout" => {
                        forbid_val(flag, inline_val)?;
                        opts.telemetry_stdout = true;
                    }
                    "--checkpoint-dir" | "--checkpoint-ssd-dir" => {
                        let val = take_val(flag, inline_val, &mut i, "requires a directory path")?;
                        if val.is_empty() {
                            return Err(CliError::Usage(format!("{flag} cannot be empty")));
                        }
                        opts.checkpoint_dir = Some(PathBuf::from(val));
                    }
                    "--checkpoint-cap-gib" | "--checkpoint-ssd-budget-gib" => {
                        let val = take_val(flag, inline_val, &mut i, "requires a number")?;
                        let gib = val.parse::<f64>().map_err(|_| {
                            CliError::Usage(format!("invalid {flag} number: {val}"))
                        })?;
                        if !gib.is_finite() || !(0.0..=32.0).contains(&gib) {
                            return Err(CliError::Usage(format!(
                                "{flag} must be between 0.0 and 32.0 GiB"
                            )));
                        }
                        opts.checkpoint_cap_gib = Some(gib);
                    }
                    "--checkpoint-floor-gib" | "--checkpoint-ssd-floor-gib" => {
                        let val = take_val(flag, inline_val, &mut i, "requires a number")?;
                        let gib = val.parse::<f64>().map_err(|_| {
                            CliError::Usage(format!("invalid {flag} number: {val}"))
                        })?;
                        if !gib.is_finite() || !(0.0..=100.0).contains(&gib) {
                            return Err(CliError::Usage(format!(
                                "{flag} must be between 0.0 and 100.0 GiB"
                            )));
                        }
                        opts.checkpoint_floor_gib = gib;
                    }
                    "--checkpoint-policy" => {
                        let val =
                            take_val(flag, inline_val, &mut i, "requires recompute-cost or lru")?;
                        if val != "recompute-cost" && val != "lru" {
                            return Err(CliError::Usage(format!(
                                "invalid --checkpoint-policy '{val}' (expected recompute-cost or lru)"
                            )));
                        }
                        opts.checkpoint_policy = val;
                    }
                    "--overlap-prefill" => {
                        forbid_val(flag, inline_val)?;
                        opts.overlap_prefill = true;
                    }
                    "--drain-timeouts-ms" => {
                        let val = take_val(
                            flag,
                            inline_val,
                            &mut i,
                            "requires four comma-separated integers",
                        )?;
                        opts.timeouts = DrainTimeouts::parse_csv(&val)?;
                    }
                    "--help" | "-h" => {
                        forbid_val(flag, inline_val)?;
                        return Ok(CliArgs {
                            state_root,
                            command: Command::PrintHelp(HelpTopic::Serve),
                        });
                    }
                    other => {
                        return Err(CliError::Usage(format!(
                            "unrecognised serve argument: {other}"
                        )));
                    }
                }
                i += 1;
            }
            if opts.speed_preset {
                if !prefill_dense_explicit {
                    opts.prefill_dense = "nax".to_owned();
                }
                if !prefill_grouped_explicit {
                    opts.prefill_grouped = "nax".to_owned();
                }
                if !verify_nax_explicit {
                    opts.verify_nax = true;
                }
                if opts.verify_hc_gdn_fused.is_none() && (opts.verify_nax || opts.verify_nax_m8) {
                    opts.verify_hc_gdn_fused = Some(true);
                }
                if opts.host_logit_views.is_none() {
                    opts.host_logit_views = Some(true);
                }
                if opts.logit_topk.is_none() {
                    opts.logit_topk = Some(true);
                }
                if opts.ple_async_gather.is_none() {
                    opts.ple_async_gather = Some(true);
                }
                if !prepared_validation_explicit && opts.verify_cost_profile.is_none() {
                    opts.prepared_validation = true;
                }
                if !tool_enum_explicit {
                    opts.tool_enum_validation = "executor".to_owned();
                }
                if !tool_arg_explicit {
                    opts.tool_arg_validation = "executor".to_owned();
                }
                if opts.drafter == "mtp" {
                    if opts.mtp_depth.is_none() {
                        opts.mtp_depth = Some(3);
                    }
                    opts.mtp_draft_slice = true;
                } else if opts.drafter == "dflash2" {
                    if opts.verify_nax {
                        opts.verify_nax_m8 = true;
                    }
                    if opts.dflash2_precision.is_none() {
                        opts.dflash2_precision = Some("q8_0".to_owned());
                    }
                    if opts.dflash2_defer_context.is_none() {
                        opts.dflash2_defer_context = Some(true);
                    }
                    if opts.dflash2_max_drafts.is_none() {
                        opts.dflash2_max_drafts = Some(7);
                    }
                    if opts.dflash2_vocab_slice.is_none() {
                        opts.dflash2_vocab_slice = Some(32_768);
                    }
                    if opts.dflash2_verify_rows.is_none() {
                        opts.dflash2_verify_rows = Some("auto".to_owned());
                    }
                }
                if !output_packet_explicit {
                    opts.output_packet = 64;
                }
                if opts.output_packet == 64 && opts.first_output_packet.is_none() {
                    opts.first_output_packet = Some(8);
                }
            }
            Command::Serve(Box::new(opts))
        }
        "pack" => {
            let Some(action) = rest.first().map(String::as_str) else {
                return Err(CliError::Usage(
                    "pack requires subcommand 'plan' or 'build'".to_owned(),
                ));
            };
            if matches!(action, "--help" | "-h") {
                return Ok(CliArgs {
                    state_root,
                    command: Command::PrintHelp(HelpTopic::Pack),
                });
            }
            let pack_rest = &rest[1..];
            match action {
                "plan" => {
                    let mut target_dir = None;
                    let mut chunk_mib = None;
                    let mut format = PackFormat::V3;
                    let mut reserve_gb = 50.0f64;
                    let mut json = false;
                    let mut check = false;
                    let mut i = 0;
                    while i < pack_rest.len() {
                        match pack_rest[i].as_str() {
                            "--chunk-mib" => {
                                i += 1;
                                chunk_mib = Some(parse_chunk_mib(pack_rest.get(i))?);
                            }
                            "--format" => {
                                i += 1;
                                format = PackFormat::parse(pack_rest.get(i))?;
                            }
                            "--target-dir" => {
                                i += 1;
                                let val = pack_rest.get(i).ok_or_else(|| {
                                    CliError::Usage("--target-dir requires a path".to_owned())
                                })?;
                                target_dir = Some(PathBuf::from(val));
                            }
                            "--reserve-gb" => {
                                i += 1;
                                let val = pack_rest.get(i).ok_or_else(|| {
                                    CliError::Usage("--reserve-gb requires a number".to_owned())
                                })?;
                                reserve_gb = parse_nonnegative_gb(val)?;
                            }
                            "--json" => json = true,
                            "--check" => check = true,
                            "--help" | "-h" => {
                                return Ok(CliArgs {
                                    state_root,
                                    command: Command::PrintHelp(HelpTopic::PackPlan),
                                });
                            }
                            other => {
                                return Err(CliError::Usage(format!(
                                    "unrecognised pack plan argument: {other}"
                                )));
                            }
                        }
                        i += 1;
                    }
                    if chunk_mib.is_some() && format == PackFormat::V2 {
                        return Err(CliError::Usage(
                            "unrecognised pack plan argument: --chunk-mib".to_owned(),
                        ));
                    }
                    Command::PackPlan {
                        chunk_mib,
                        format,
                        target_dir,
                        reserve_gb,
                        json,
                        check,
                    }
                }
                "build" => {
                    let mut output = None;
                    let mut format = PackFormat::V3;
                    let mut reserve_gb = 50.0f64;
                    let mut chunk_mib = None;
                    let mut i = 0;
                    while i < pack_rest.len() {
                        match pack_rest[i].as_str() {
                            "--chunk-mib" => {
                                i += 1;
                                chunk_mib = Some(parse_chunk_mib(pack_rest.get(i))?);
                            }
                            "--format" => {
                                i += 1;
                                format = PackFormat::parse(pack_rest.get(i))?;
                            }
                            "--output" => {
                                i += 1;
                                let val = pack_rest.get(i).ok_or_else(|| {
                                    CliError::Usage("--output requires a file path".to_owned())
                                })?;
                                output = Some(PathBuf::from(val));
                            }
                            "--reserve-gb" => {
                                i += 1;
                                let val = pack_rest.get(i).ok_or_else(|| {
                                    CliError::Usage("--reserve-gb requires a number".to_owned())
                                })?;
                                reserve_gb = parse_nonnegative_gb(val)?;
                            }
                            "--help" | "-h" => {
                                return Ok(CliArgs {
                                    state_root,
                                    command: Command::PrintHelp(HelpTopic::PackBuild),
                                });
                            }
                            other => {
                                return Err(CliError::Usage(format!(
                                    "unrecognised pack build argument: {other}"
                                )));
                            }
                        }
                        i += 1;
                    }
                    Command::PackBuild {
                        format,
                        output,
                        reserve_gb,
                        chunk_mib: chunk_mib.unwrap_or_else(|| format.default_chunk_mib()),
                    }
                }
                other => {
                    return Err(CliError::Usage(format!(
                        "unknown pack subcommand '{other}' (expected plan or build)"
                    )));
                }
            }
        }
        other => {
            return Err(CliError::Usage(format!("unknown subcommand '{other}'")));
        }
    };

    Ok(CliArgs {
        state_root,
        command,
    })
}

fn parse_chunk_mib(value: Option<&String>) -> Result<u64, CliError> {
    value
        .ok_or_else(|| CliError::Usage("--chunk-mib requires an integer".to_owned()))?
        .parse()
        .map_err(|_| CliError::Usage("invalid --chunk-mib integer".to_owned()))
}

fn parse_nonnegative_gb(raw: &str) -> Result<f64, CliError> {
    let val = raw
        .parse::<f64>()
        .map_err(|_| CliError::Usage(format!("invalid --reserve-gb number: {raw}")))?;
    if !val.is_finite() || val < 0.0 || val > 1_000_000.0 {
        return Err(CliError::Usage(
            "--reserve-gb must be a finite non-negative number".to_owned(),
        ));
    }
    Ok(val)
}

fn reject_extra_args(cmd: &str, rest: &[String]) -> Result<(), CliError> {
    if let Some(first) = rest.first() {
        return Err(CliError::Usage(format!(
            "unexpected argument '{first}' for {cmd}"
        )));
    }
    Ok(())
}

/// Executes a parsed `CliArgs` invocation, writing normal output to `out` and returning the process exit code.
pub fn execute(cli: CliArgs, out: &mut dyn Write) -> Result<i32, CliError> {
    match cli.command {
        Command::PrintVersion => {
            writeln!(out, "{VERSION}").map_err(|source| CliError::Io {
                context: "stdout write failed".to_owned(),
                source,
            })?;
            return Ok(0);
        }
        Command::PrintHelp(topic) => {
            out.write_all(topic.text().as_bytes())
                .map_err(|source| CliError::Io {
                    context: "stdout write failed".to_owned(),
                    source,
                })?;
            return Ok(0);
        }
        Command::CheckModelConfig(ref path) => return model_config::check_file(path, out),
        _ => {}
    }

    let layout = RuntimeLayout::detect(cli.state_root.as_deref())?;

    match cli.command {
        Command::PrintVersion | Command::PrintHelp(_) | Command::CheckModelConfig(_) => Ok(0),
        Command::Plan => {
            let profile = read_json_file(&layout.hardware_profile_path())?;
            let formatted = format_pretty_sorted_json(&profile)?;
            out.write_all(&formatted).map_err(|source| CliError::Io {
                context: "stdout write failed".to_owned(),
                source,
            })?;
            Ok(0)
        }
        Command::VerifyBuild => {
            verify_build(&layout)?;
            writeln!(
                out,
                "Pinned build artefacts, system library links, Metal library and sources verified."
            )
            .map_err(|source| CliError::Io {
                context: "stdout write failed".to_owned(),
                source,
            })?;
            Ok(0)
        }
        Command::Doctor => {
            let tokspur_base = layout.tokspur_base()?;
            let profile = read_json_file(&layout.hardware_profile_path())?;
            let default_port = profile
                .find("port")
                .ok()
                .flatten()
                .and_then(|v| v.unsigned_integer().ok())
                .and_then(|p| u16::try_from(p).ok())
                .unwrap_or(18081);
            let targets = layout.resolve_launch_targets()?;
            let workers: Vec<Json> = inference_processes()?
                .into_iter()
                .map(|pid| {
                    Json::new_object([
                        ("pid", Json::integer(u64::try_from(pid).unwrap_or(0))),
                        ("runtime", Json::from("inference worker")),
                    ])
                })
                .collect();

            let summary = Json::new_object([
                ("version", Json::from(VERSION)),
                ("cli_implementation", Json::from("rust")),
                ("layout", Json::from(layout.kind_name())),
                ("native_built", Json::from(targets.worker_bin.is_file())),
                ("server_built", Json::from(targets.server_bin.is_file())),
                (
                    "weights_adopted",
                    Json::from(tokspur_base.join("adoption.json").is_file()),
                ),
                (
                    "configured",
                    Json::from(tokspur_base.join("configured.json").is_file()),
                ),
                (
                    "default_port_available",
                    Json::from(port_available(default_port)),
                ),
                ("other_inference_workers", Json::Array(workers)),
            ]);
            let formatted = format_pretty_sorted_json(&summary)?;
            out.write_all(&formatted).map_err(|source| CliError::Io {
                context: "stdout write failed".to_owned(),
                source,
            })?;
            Ok(0)
        }
        Command::Adopt {
            target,
            draft,
            check_only,
        } => {
            if check_only {
                check_adopted_weights(&layout, out)?;
            } else {
                let (Some(t), Some(d)) = (target, draft) else {
                    return Err(CliError::Validation(
                        "--target and --draft are required unless --check-only is specified"
                            .to_owned(),
                    ));
                };
                adopt_weights(&layout, &t, &d, out)?;
            }
            Ok(0)
        }
        Command::Configure { port } => {
            let resolved_port = if let Some(p) = port {
                p
            } else {
                let profile = read_json_file(&layout.hardware_profile_path())?;
                profile
                    .find("port")
                    .ok()
                    .flatten()
                    .and_then(|v| v.unsigned_integer().ok())
                    .and_then(|p| u32::try_from(p).ok())
                    .unwrap_or(18081)
            };
            configure_runtime(&layout, resolved_port, out)?;
            Ok(0)
        }
        Command::Serve(opts) => serve_foreground(&layout, &opts),
        Command::PackPlan {
            format,
            target_dir,
            chunk_mib,
            reserve_gb,
            json,
            check,
        } => {
            let default_base = layout.tokspur_base()?;
            if format == PackFormat::V3 {
                return pack_command::plan_v3(
                    &layout,
                    target_dir.as_deref(),
                    reserve_gb,
                    chunk_mib.unwrap_or_else(|| format.default_chunk_mib()),
                    json,
                    check,
                    out,
                );
            }
            let dir = target_dir.unwrap_or_else(|| {
                serve::default_derived_pack_path(&default_base, false)
                    .parent()
                    .unwrap_or(&default_base)
                    .to_path_buf()
            });
            let existing_pack = serve::default_derived_pack_path(&default_base, false);
            let existing_lanes =
                default_base.join("derived/lane-layout-v1/flash-next-lanes-v1.gguf");
            let reserve_bytes = (reserve_gb * (1024.0 * 1024.0 * 1024.0)) as u64;
            let plan = plan_static_pack(
                &dir,
                Some(&existing_pack),
                Some(&existing_lanes),
                reserve_bytes,
            )?;
            if json {
                let raw_summary = plan.summary_json()?;
                out.write_all(&raw_summary).map_err(|source| CliError::Io {
                    context: "stdout write failed".to_owned(),
                    source,
                })?;
                out.write_all(b"\n").map_err(|source| CliError::Io {
                    context: "stdout write failed".to_owned(),
                    source,
                })?;
            } else {
                writeln!(
                    out,
                    "=== Tokspur Pack Planner: tokspur-pack-v2 ===\nTensor count: {}\nPayload bytes: {}\nPlanned file bytes: {}\nPeak additional allocation: {}\nRequired reserve: {}\nAvailable disk space: {}\nBudget check status: {}\nBudget message: {}",
                    plan.tensors.len(),
                    plan.payload_bytes,
                    plan.planned_file_bytes,
                    plan.peak_additional_allocation,
                    plan.reserve_bytes,
                    plan.available_space,
                    if plan.budget_passed { "PASS" } else { "FAIL" },
                    plan.budget_message
                )
                .map_err(|source| CliError::Io {
                    context: "stdout write failed".to_owned(),
                    source,
                })?;
            }
            Ok(if !check || plan.budget_passed { 0 } else { 1 })
        }
        Command::PackBuild {
            format,
            output,
            reserve_gb,
            chunk_mib,
        } => {
            let tokspur_base = layout.tokspur_base()?;
            if let (RuntimeLayout::Staged(staged), Some(path)) = (&layout, &output) {
                let absolute = std::path::absolute(path).map_err(|source| CliError::Io {
                    context: "cannot resolve pack output".to_owned(),
                    source,
                })?;
                layout::ensure_no_symlink_components(&absolute, "pack output")?;
                if layout::at_or_below_directory(&absolute, &staged.package_dir)? {
                    return Err(CliError::Validation(
                        "pack output must be outside the immutable package directory".to_owned(),
                    ));
                }
            }
            let _lock = StateLock::acquire(&tokspur_base)?;
            if format == PackFormat::V3 {
                return pack_command::build_v3(
                    &layout,
                    output.as_deref(),
                    reserve_gb,
                    chunk_mib,
                    out,
                );
            }
            let (shard_paths, mtp_path) = pack_command::adopted_pack_sources(&layout)?;
            let out_path = if let Some(explicit) = output {
                explicit
            } else {
                let derived_dir = private_dir(&tokspur_base, Path::new("derived"))?;
                derived_dir.join(format.filename())
            };

            let source_models =
                ModelFiles::new(&shard_paths).map_err(tokspur_pack::PackError::Gguf)?;
            let mtp_models = ModelFiles::new(&[mtp_path]).map_err(tokspur_pack::PackError::Gguf)?;

            let reserve_bytes = if reserve_gb == 50.0 {
                DEFAULT_RESERVE_BYTES
            } else {
                (reserve_gb * (1024.0 * 1024.0 * 1024.0)) as u64
            };
            let chunk_bytes = chunk_mib.saturating_mul(1024 * 1024);

            let _signal_guard = serve::SignalGuard::install()?;
            let mut reported = std::time::Instant::now();
            let mut progress_output = std::io::stderr().lock();
            let built = build_derived_pack_with_progress(
                &source_models,
                Some(&mtp_models),
                &out_path,
                reserve_bytes,
                chunk_bytes,
                |written, total| {
                    if written == 0 || reported.elapsed().as_secs() >= 5 {
                        if writeln!(
                            progress_output,
                            "pack build: {written}/{total} bytes verified"
                        )
                        .is_err()
                        {
                            return false;
                        }
                        reported = std::time::Instant::now();
                    }
                    !tokspur_sys::stop_requested()
                },
            )?;
            let receipt = Json::new_object([
                ("recipe", Json::from(built.recipe)),
                ("version", Json::integer(u64::from(built.version))),
                ("tensor_count", Json::integer(built.tensors.len() as u64)),
                ("file_bytes", Json::integer(built.file_bytes)),
                ("data_offset", Json::integer(built.data_offset)),
                (
                    "output",
                    Json::from(out_path.to_string_lossy().into_owned()),
                ),
            ]);
            let formatted = format_pretty_sorted_json(&receipt)?;
            out.write_all(&formatted).map_err(|source| CliError::Io {
                context: "stdout write failed".to_owned(),
                source,
            })?;
            Ok(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::{SpeculativeAsset, build_carrier_flags, build_derived_pack_flags};

    #[test]
    fn pack_chunk_defaults_match_native_and_plan_accepts_v3_only_override() {
        for (format, expected) in [("v2", 4), ("v3", 16)] {
            let Command::PackBuild { chunk_mib, .. } =
                parse_args(["pack", "build", "--format", format])
                    .unwrap()
                    .command
            else {
                panic!("expected build")
            };
            assert_eq!(chunk_mib, expected);
        }
        assert!(parse_args(["pack", "plan", "--format", "v2", "--chunk-mib", "2"]).is_err());
        assert!(matches!(
            parse_args(["pack", "plan", "--chunk-mib", "2", "--format", "v3"])
                .unwrap()
                .command,
            Command::PackPlan {
                chunk_mib: Some(2),
                ..
            }
        ));
    }

    #[test]
    fn pack_v3_is_the_default_and_format_is_explicit() {
        for action in ["plan", "build"] {
            let implicit = parse_args(["pack", action]).unwrap();
            assert_eq!(
                parse_args(["pack", action, "--format", "v3"]).unwrap(),
                implicit
            );
            assert_ne!(
                parse_args(["pack", action, "--format", "v2"]).unwrap(),
                implicit
            );
            assert!(parse_args(["pack", action, "--format"]).is_err());
            assert!(parse_args(["pack", action, "--format", "v4"]).is_err());
        }
    }

    #[test]
    fn cli_args_and_serve_options_redact_paths_in_debug() {
        let parsed = parse_args([
            "--state-root",
            "/Users/secret/state-root",
            "serve",
            "--checkpoint-dir",
            "/Users/secret/ckpt-dir",
            "--drafter",
            "dflash2",
            "--dflash2-dir",
            "/Users/secret/dflash2-dir",
        ])
        .unwrap();
        let dbg = format!("{parsed:?}");
        assert!(!dbg.contains("secret"), "leaked path in Debug: {dbg}");
        assert!(dbg.contains("[redacted]"));

        let asset_dbg = format!(
            "{:?} {:?}",
            SpeculativeAsset::MtpSidecar(Path::new("/Users/secret/mtp.gguf")),
            SpeculativeAsset::DFlash2Dir(Path::new("/Users/secret/dflash2")),
        );
        assert!(
            !asset_dbg.contains("secret"),
            "leaked speculative asset path in Debug: {asset_dbg}"
        );

        let adopt = parse_args([
            "adopt",
            "--target",
            "/Users/secret/target",
            "--draft",
            "/Users/secret/draft",
        ])
        .unwrap();
        assert!(!format!("{adopt:?}").contains("secret"));
    }

    #[test]
    fn serve_options_default_and_opt_in_flags_match_python_cli() {
        let default_cli = parse_args(["serve"]).unwrap();
        let Command::Serve(default_opts) = default_cli.command else {
            panic!("expected Serve");
        };
        let pack = Path::new("/tmp/derived/tokspur-pack-v2.asa.gguf");
        let sidecar = Path::new("/tmp/draft/mtp.gguf");
        let dflash2 = Path::new("/tmp/draft/dflash2");
        let state = Path::new("/tmp/state");
        assert_eq!(
            build_derived_pack_flags(&default_opts, pack),
            vec![
                "--derived-pack",
                "/tmp/derived/tokspur-pack-v2.asa.gguf",
                "--derived-pack-auth-workers",
                "16",
            ]
        );
        let default_carrier =
            build_carrier_flags(&default_opts, SpeculativeAsset::MtpSidecar(sidecar), state)
                .unwrap();
        for pair in [
            ["--native-mtp-sidecar", "/tmp/draft/mtp.gguf"],
            ["--native-mtp-depth", "2"],
            ["--native-mtp-draft-attention", "tiled128"],
            ["--native-mtp-output-packet", "8"],
            ["--native-known-token-capacity", "2048"],
            ["--native-prefill-dense", "f32-staged"],
            ["--native-prefill-grouped", "f32"],
            ["--native-gdn-ported", "off"],
            ["--native-prefill-router-window", "off"],
            ["--native-prefill-hc-down-window", "off"],
            ["--native-qsa-sparse", "off"],
            ["--native-concurrent-dispatch", "on"],
            ["--native-prefill-swizzle", "on"],
            ["--native-qsa-prefill-fused", "on"],
            ["--native-hc-consumer-fused", "on"],
            ["--native-ple-overlap", "on"],
            ["--shared-prefix-cache", "on"],
            ["--tool-enum-validation", "strict"],
        ] {
            assert!(
                default_carrier.windows(2).any(|w| w == pair),
                "missing default pair {pair:?} in {default_carrier:?}"
            );
        }
        for opt_in_only in [
            "--native-q8-gemv-rg4",
            "--native-q8-three-msg",
            "--native-prefill-multiseg",
            "--native-raw-decode",
            "--native-verify-nax",
            "--native-verify-nax-m8",
            "--native-dflash2-verify-rows",
            "--native-drafter",
            "--native-mtp-draft-slice",
            "--native-mtp-lookup",
            "--native-mtp-lookup-match",
            "--overlap-prefill",
            "--private-prefix-trace",
            "--telemetry-stdout",
        ] {
            assert!(
                !default_carrier.iter().any(|f| f == opt_in_only),
                "unexpected opt-in flag {opt_in_only} in default carrier: {default_carrier:?}"
            );
        }

        // Verify explicit opt-out flags disable every default-on exact optimisation.
        let opt_out = parse_args([
            "serve",
            "--known-token-capacity=512",
            "--no-concurrent-dispatch",
            "--q8-gemv-rg4=off",
            "--q8-three-msg",
            "off",
            "--no-prefill-multiseg",
            "--no-prefill-swizzle",
            "--no-qsa-prefill-fused",
            "--no-hc-consumer-fused",
            "--no-ple-overlap",
            "--no-shared-prefix-cache",
            "--checkpoint-cap-gib",
            "0",
        ])
        .unwrap();
        let Command::Serve(opt_out_opts) = opt_out.command else {
            panic!("expected Serve");
        };
        let opt_out_carrier =
            build_carrier_flags(&opt_out_opts, SpeculativeAsset::MtpSidecar(sidecar), state)
                .unwrap();
        assert!(
            opt_out_carrier
                .windows(2)
                .any(|w| w == ["--native-known-token-capacity", "512"])
        );
        for disabled in [
            "--native-concurrent-dispatch",
            "--native-q8-gemv-rg4",
            "--native-q8-three-msg",
            "--native-prefill-multiseg",
            "--native-prefill-swizzle",
            "--native-qsa-prefill-fused",
            "--native-hc-consumer-fused",
            "--native-ple-overlap",
            "--shared-prefix-cache",
            "--checkpoint-ssd",
        ] {
            assert!(
                !opt_out_carrier.iter().any(|f| f == disabled),
                "opt-out failed to disable {disabled}: {opt_out_carrier:?}"
            );
        }

        let opt_in = parse_args([
            "serve",
            "--derived-pack-trust-mode",
            "local-seal",
            "--output-packet",
            "64",
            "--prefill-grouped",
            "auto",
            "--known-token-capacity",
            "2048",
            "--mtp-draft-slice",
            "--q8-gemv-rg4",
            "wide",
            "--q8-three-msg",
            "wide",
            "--mtp-lookup",
            "hybrid",
            "--mtp-lookup-match",
            "4",
            "--tool-enum-validation",
            "executor",
            "--private-prefix-trace",
            "--telemetry-stdout",
            "--checkpoint-cap-gib",
            "4.0",
            "--checkpoint-floor-gib",
            "25.0",
            "--checkpoint-policy",
            "lru",
            "--overlap-prefill",
        ])
        .unwrap();
        let Command::Serve(opts) = opt_in.command else {
            panic!("expected Serve");
        };
        assert_eq!(
            build_derived_pack_flags(&opts, pack),
            vec![
                "--derived-pack",
                "/tmp/derived/tokspur-pack-v2.asa.gguf",
                "--derived-pack-auth-workers",
                "16",
                "--derived-pack-trust-mode",
                "local-seal",
            ]
        );
        let carrier =
            build_carrier_flags(&opts, SpeculativeAsset::MtpSidecar(sidecar), state).unwrap();
        for pair in [
            ["--native-mtp-output-packet", "64"],
            ["--native-known-token-capacity", "2048"],
            ["--native-mtp-draft-slice", "on"],
            ["--native-prefill-grouped", "auto"],
            ["--native-concurrent-dispatch", "on"],
            ["--native-q8-gemv-rg4", "wide"],
            ["--native-q8-three-msg", "wide"],
            ["--native-mtp-lookup", "hybrid"],
            ["--native-mtp-lookup-match", "4"],
            ["--tool-enum-validation", "executor"],
            ["--private-prefix-trace", "on"],
            ["--telemetry-stdout", "on"],
            ["--checkpoint-ssd", "on"],
            ["--checkpoint-cap-gib", "4"],
            ["--checkpoint-floor-gib", "25"],
            ["--checkpoint-dir", "/tmp/state/checkpoints"],
            ["--checkpoint-policy", "lru"],
            ["--overlap-prefill", "on"],
        ] {
            assert!(
                carrier.windows(2).any(|w| w == pair),
                "missing pair {pair:?} in {carrier:?}"
            );
        }

        // DFlash2 leaves checkpoint caches off unless explicitly requested.
        let nax_dflash2 = parse_args([
            "serve",
            "--prefill-dense",
            "nax",
            "--prefill-grouped",
            "nax",
            "--verify-nax",
            "--drafter",
            "dflash2",
            "--dflash2-dir",
            "/tmp/draft/dflash2",
        ])
        .unwrap();
        let Command::Serve(dflash2_opts) = nax_dflash2.command else {
            panic!("expected Serve");
        };
        let dflash2_carrier =
            build_carrier_flags(&dflash2_opts, SpeculativeAsset::DFlash2Dir(dflash2), state)
                .unwrap();
        for pair in [
            ["--native-drafter", "dflash2"],
            ["--native-dflash2-dir", "/tmp/draft/dflash2"],
            ["--native-mtp-depth", "7"],
            ["--native-mtp-draft-attention", "split16"],
            ["--native-prefill-dense", "nax"],
            ["--native-prefill-grouped", "nax"],
            ["--native-verify-nax", "on"],
        ] {
            assert!(
                dflash2_carrier.windows(2).any(|w| w == pair),
                "missing DFlash2/NAX pair {pair:?} in {dflash2_carrier:?}"
            );
        }
        assert!(!dflash2_carrier.iter().any(|f| f == "--checkpoint-ssd"));
        assert!(!dflash2_carrier.iter().any(|f| f == "--shared-prefix-cache"));

        let mut cached_dflash2 = dflash2_opts.clone();
        cached_dflash2.shared_prefix_cache = Some(true);
        cached_dflash2.checkpoint_cap_gib = Some(4.0);
        let flags = build_carrier_flags(
            &cached_dflash2,
            SpeculativeAsset::DFlash2Dir(dflash2),
            state,
        )
        .unwrap();
        for pair in [
            ["--shared-prefix-cache", "on"],
            ["--checkpoint-ssd", "on"],
            ["--checkpoint-cap-gib", "4"],
        ] {
            assert!(flags.windows(2).any(|window| window == pair));
        }

        // Verify cross-flag incompatibilities are rejected cleanly.
        for bad_args in [
            vec!["serve", "--drafter", "dflash2"],
            vec!["serve", "--dflash2-dir", "/tmp/draft/dflash2"],
            vec![
                "serve",
                "--drafter",
                "dflash2",
                "--dflash2-dir",
                "/tmp/draft/dflash2",
                "--mtp-draft-slice",
            ],
            vec![
                "serve",
                "--drafter",
                "dflash2",
                "--dflash2-dir",
                "/tmp/draft/dflash2",
                "--mtp-lookup",
                "bypass",
            ],
            vec!["serve", "--prefill-grouped", "serial"],
            vec![
                "serve",
                "--prefill-dense",
                "f32",
                "--prefill-grouped",
                "nax",
            ],
        ] {
            let parsed_bad = parse_args(bad_args.clone()).unwrap();
            let Command::Serve(bad_opts) = parsed_bad.command else {
                panic!("expected Serve");
            };
            let asset = if bad_opts.drafter == "dflash2" {
                SpeculativeAsset::DFlash2Dir(dflash2)
            } else {
                SpeculativeAsset::MtpSidecar(sidecar)
            };
            assert!(
                build_carrier_flags(&bad_opts, asset, state).is_err(),
                "expected incompatible flags {bad_args:?} to be rejected"
            );
        }

        let raw_cli = parse_args(["serve", "--raw-gguf"]).unwrap();
        let Command::Serve(raw_opts) = raw_cli.command else {
            panic!("expected Serve");
        };
        assert!(build_derived_pack_flags(&raw_opts, pack).is_empty());
        let raw_carrier =
            build_carrier_flags(&raw_opts, SpeculativeAsset::MtpSidecar(sidecar), state).unwrap();
        assert!(
            raw_carrier
                .windows(2)
                .any(|w| w == ["--native-raw-decode", "on"])
        );
    }

    #[test]
    fn subcommand_help_prints_subcommand_flags() {
        let serve_help = parse_args(["serve", "--help"]).unwrap();
        assert_eq!(serve_help.command, Command::PrintHelp(HelpTopic::Serve));
        let mut out = Vec::new();
        assert_eq!(execute(serve_help, &mut out).unwrap(), 0);
        let text = String::from_utf8(out).unwrap();
        for expected in [
            "--raw-gguf",
            "--derived-pack-trust-mode",
            "local-seal",
            "--output-packet",
            "--prefill-dense",
            "--prefill-grouped",
            "--known-token-capacity",
            "--mtp-draft-slice",
            "--verify-nax",
            "--no-verify-nax",
            "--native-verify-nax",
            "--drafter",
            "--dflash2-dir",
            "--concurrent-dispatch",
            "--no-concurrent-dispatch",
            "--q8-gemv-rg4",
            "--q8-three-msg",
            "--prefill-multiseg",
            "--prefill-swizzle",
            "--qsa-prefill-fused",
            "--hc-consumer-fused",
            "--ple-overlap",
            "--shared-prefix-cache",
            "--mtp-lookup",
            "--mtp-lookup-match",
            "--tool-enum-validation",
            "--private-prefix-trace",
            "--telemetry-stdout",
            "--checkpoint-dir",
            "--checkpoint-cap-gib",
            "--checkpoint-floor-gib",
            "--checkpoint-policy",
            "--overlap-prefill",
            "--drain-timeouts-ms",
        ] {
            assert!(
                text.contains(expected),
                "missing '{expected}' in serve --help: {text}"
            );
        }

        for (args, expected_topic, marker) in [
            (vec!["--help"], HelpTopic::General, "commands:"),
            (vec!["doctor", "-h"], HelpTopic::Doctor, "doctor"),
            (vec!["adopt", "--help"], HelpTopic::Adopt, "--check-only"),
            (vec!["configure", "-h"], HelpTopic::Configure, "--port"),
            (vec!["plan", "--help"], HelpTopic::Plan, "plan"),
            (
                vec!["verify-build", "--help"],
                HelpTopic::VerifyBuild,
                "verify-build",
            ),
            (vec!["pack", "--help"], HelpTopic::Pack, "subcommands:"),
            (
                vec!["pack", "plan", "--help"],
                HelpTopic::PackPlan,
                "--reserve-gb",
            ),
            (
                vec!["pack", "build", "-h"],
                HelpTopic::PackBuild,
                "--chunk-mib",
            ),
        ] {
            let parsed = parse_args(args).unwrap();
            assert_eq!(parsed.command, Command::PrintHelp(expected_topic));
            let mut buf = Vec::new();
            assert_eq!(execute(parsed, &mut buf).unwrap(), 0);
            let rendered = String::from_utf8(buf).unwrap();
            assert!(rendered.contains(marker));
        }
    }

    #[test]
    fn native_sync_selectors_are_opt_in_typed_and_forwarded() {
        use crate::serve::build_carrier_flags;
        let Command::Serve(defaults) = parse_args(["serve"]).unwrap().command else {
            panic!("expected serve")
        };
        assert_eq!(defaults.host_logit_views, None);
        assert_eq!(defaults.dflash2_defer_context, None);
        let default_flags = build_carrier_flags(
            &defaults,
            SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
            Path::new("/tmp/state"),
        )
        .unwrap();
        for flag in [
            "--native-host-logit-views",
            "--native-dflash2-defer-context",
        ] {
            assert!(!default_flags.iter().any(|word| word == flag));
            for value in ["off", "on"] {
                let Command::Serve(opts) = parse_args(["serve", flag, value]).unwrap().command
                else {
                    panic!("expected serve")
                };
                let selected = if flag == "--native-host-logit-views" {
                    opts.host_logit_views
                } else {
                    opts.dflash2_defer_context
                };
                assert_eq!(selected, Some(value == "on"));
                let inline = format!("{flag}={value}");
                assert_eq!(
                    parse_args(["serve", inline.as_str()]).unwrap().command,
                    Command::Serve(opts.clone())
                );
                let forwarded = build_carrier_flags(
                    &opts,
                    SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                    Path::new("/tmp/state"),
                )
                .unwrap();
                assert!(forwarded.windows(2).any(|pair| pair == [flag, value]));
            }
            for value in ["true", "false", "1", "", "auto"] {
                assert!(matches!(
                    parse_args(["serve", flag, value]),
                    Err(CliError::Usage(_))
                ));
            }
            assert!(matches!(
                parse_args(["serve", flag]),
                Err(CliError::Usage(_))
            ));
        }
    }

    #[test]
    fn known_token_capacity_admits_1024() {
        for cap in ["512", "1024", "2048"] {
            let parsed = parse_args(["serve", "--known-token-capacity", cap]).unwrap();
            let Command::Serve(opts) = parsed.command else {
                panic!("expected Serve");
            };
            assert_eq!(opts.known_token_capacity, cap.parse::<u32>().unwrap());
            let flags = build_carrier_flags(
                &opts,
                SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                Path::new("/tmp/state"),
            )
            .unwrap();
            assert!(
                flags
                    .windows(2)
                    .any(|pair| pair == ["--native-known-token-capacity", cap])
            );
        }
        let err = parse_args(["serve", "--known-token-capacity", "256"]).unwrap_err();
        assert!(matches!(err, CliError::Usage(_)));
    }

    #[test]
    fn verify_nax_m8_enables_fused_verification_without_separate_nax_flag() {
        for args in [
            &[
                "serve",
                "--verify-nax-m8",
                "--native-verify-hc-gdn-fused",
                "on",
            ][..],
            &[
                "serve",
                "--speed-preset",
                "--no-verify-nax",
                "--verify-nax-m8",
            ][..],
            &[
                "serve",
                "--verify-nax-m8",
                "--native-verify-nax",
                "off",
                "--native-verify-hc-gdn-fused",
                "on",
            ][..],
        ] {
            let Command::Serve(opts) = parse_args(args.iter().copied()).unwrap().command else {
                panic!("expected Serve");
            };
            assert!(!opts.verify_nax);
            assert!(opts.verify_nax_m8);
            assert_eq!(opts.verify_hc_gdn_fused, Some(true));
            let flags = build_carrier_flags(
                &opts,
                SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                Path::new("/tmp/state"),
            )
            .unwrap();
            for flag in [
                "--native-verify-nax",
                "--native-verify-nax-m8",
                "--native-verify-hc-gdn-fused",
            ] {
                assert!(flags.windows(2).any(|pair| pair == [flag, "on"]));
            }
        }
    }

    #[test]
    fn dflash2_verify_rows_are_forwarded_and_refused_for_mtp() {
        for mode in ["all", "auto"] {
            let Command::Serve(opts) = parse_args([
                "serve",
                "--drafter",
                "dflash2",
                "--dflash2-dir",
                "/tmp/dflash2",
                "--native-dflash2-verify-rows",
                mode,
            ])
            .unwrap()
            .command
            else {
                panic!("expected Serve");
            };
            let flags = build_carrier_flags(
                &opts,
                SpeculativeAsset::DFlash2Dir(Path::new("/tmp/dflash2")),
                Path::new("/tmp/state"),
            )
            .unwrap();
            assert!(
                flags
                    .windows(2)
                    .any(|pair| pair == ["--native-dflash2-verify-rows", mode])
            );

            let Command::Serve(mtp_opts) =
                parse_args(["serve", "--native-dflash2-verify-rows", mode])
                    .unwrap()
                    .command
            else {
                panic!("expected Serve");
            };
            let error = build_carrier_flags(
                &mtp_opts,
                SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                Path::new("/tmp/state"),
            )
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                "--native-dflash2-verify-rows requires --drafter dflash2"
            );
        }
        assert!(matches!(
            parse_args(["serve", "--native-dflash2-verify-rows", "off"]),
            Err(CliError::Usage(_))
        ));
    }

    #[test]
    fn speed_preset_preserves_explicit_packet_overrides_in_either_order() {
        use crate::serve::build_carrier_flags;
        let Command::Serve(defaults) = parse_args(["serve"]).unwrap().command else {
            panic!("expected Serve");
        };
        assert_eq!(defaults.output_packet, 8);
        assert_eq!(defaults.first_output_packet, None);
        for (packet, first) in [("8", None), ("8", Some("3")), ("64", Some("3"))] {
            for preset_first in [false, true] {
                for inline in [false, true] {
                    let mut args = vec!["serve".to_owned()];
                    if preset_first {
                        args.push("--speed-preset".to_owned());
                    }
                    if inline {
                        args.push(format!("--output-packet={packet}"));
                    } else {
                        args.extend(["--output-packet".to_owned(), packet.to_owned()]);
                    }
                    if let Some(value) = first {
                        args.extend(["--first-output-packet".to_owned(), value.to_owned()]);
                    }
                    if !preset_first {
                        args.push("--speed-preset".to_owned());
                    }
                    let Command::Serve(selected) = parse_args(args).unwrap().command else {
                        panic!("expected Serve");
                    };
                    assert_eq!(selected.output_packet, packet.parse::<u32>().unwrap());
                    assert_eq!(
                        selected.first_output_packet,
                        first.map(|v| v.parse::<u32>().unwrap())
                    );
                    let flags = build_carrier_flags(
                        &selected,
                        SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                        Path::new("/tmp/state"),
                    )
                    .unwrap();
                    assert!(
                        flags
                            .windows(2)
                            .any(|p| p == ["--native-mtp-output-packet", packet])
                    );
                    assert_eq!(
                        flags
                            .iter()
                            .any(|v| v == "--native-mtp-first-output-packet"),
                        first.is_some()
                    );
                }
            }
        }
    }

    #[test]
    fn mtp_depth_dflash2_precision_and_speed_preset_flags() {
        use crate::serve::build_carrier_flags;
        let Command::Serve(mtp3) = parse_args(["serve", "--mtp-depth", "3"]).unwrap().command
        else {
            panic!("expected Serve");
        };
        let mtp3_flags = build_carrier_flags(
            &mtp3,
            SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
            Path::new("/tmp/state"),
        )
        .unwrap();
        assert!(
            mtp3_flags
                .windows(2)
                .any(|pair| pair == ["--native-mtp-depth", "3"])
        );

        let Command::Serve(preset) = parse_args(["serve", "--speed-preset"]).unwrap().command
        else {
            panic!("expected Serve");
        };
        assert_eq!(preset.prefill_dense, "nax");
        assert_eq!(preset.prefill_grouped, "nax");
        assert_eq!(preset.prefill_router_fast, None);
        assert!(!preset.gdn_prefill_only);
        assert!(preset.verify_nax);
        assert!(!preset.verify_nax_m8);
        assert_eq!(preset.dflash2_verify_rows, None);
        assert_eq!(preset.verify_hc_gdn_fused, Some(true));
        assert_eq!(preset.host_logit_views, Some(true));
        assert_eq!(preset.logit_topk, Some(true));
        assert_eq!(preset.ple_async_gather, Some(true));
        assert!(preset.prepared_validation);
        assert_eq!(preset.tool_enum_validation, "executor");
        assert_eq!(preset.tool_arg_validation, "executor");
        assert_eq!(preset.mtp_depth, Some(3));
        assert!(preset.mtp_draft_slice);
        assert_eq!(preset.mtp_fast_round, None);
        assert_eq!(preset.output_packet, 64);
        assert_eq!(preset.first_output_packet, Some(8));

        let Command::Serve(preset_fast_on) =
            parse_args(["serve", "--speed-preset", "--native-mtp-fast-round", "on"])
                .unwrap()
                .command
        else {
            panic!("expected Serve");
        };
        assert_eq!(preset_fast_on.mtp_fast_round, Some(true));

        let Command::Serve(preset_overrides) = parse_args([
            "serve",
            "--speed-preset",
            "--mtp-depth",
            "2",
            "--no-verify-nax",
            "--tool-enum-validation",
            "strict",
            "--tool-arg-validation",
            "strict",
        ])
        .unwrap()
        .command
        else {
            panic!("expected Serve");
        };
        assert_eq!(preset_overrides.mtp_depth, Some(2));
        assert!(!preset_overrides.verify_nax);
        assert_eq!(preset_overrides.verify_hc_gdn_fused, None);
        assert_eq!(preset_overrides.tool_enum_validation, "strict");
        assert_eq!(preset_overrides.tool_arg_validation, "strict");

        let Command::Serve(preset_df) = parse_args([
            "serve",
            "--speed-preset",
            "--drafter",
            "dflash2",
            "--dflash2-dir",
            "/tmp/dflash2",
        ])
        .unwrap()
        .command
        else {
            panic!("expected Serve");
        };
        assert!(preset_df.verify_nax);
        assert!(preset_df.verify_nax_m8);
        assert!(preset_df.prepared_validation);
        assert_eq!(preset_df.dflash2_precision.as_deref(), Some("q8_0"));
        assert_eq!(preset_df.dflash2_defer_context, Some(true));
        assert_eq!(preset_df.dflash2_max_drafts, Some(7));
        assert_eq!(preset_df.dflash2_vocab_slice, Some(32_768));
        assert_eq!(preset_df.dflash2_verify_rows.as_deref(), Some("auto"));
        assert_eq!(preset_df.output_packet, 64);
        assert_eq!(preset_df.first_output_packet, Some(8));

        for preset_first in [false, true] {
            let mut args = vec![
                "serve",
                "--drafter",
                "dflash2",
                "--dflash2-dir",
                "/tmp/dflash2",
            ];
            let overrides = [
                "--dflash2-precision",
                "bf16",
                "--dflash2-max-drafts",
                "4",
                "--native-dflash2-verify-rows",
                "all",
                "--native-prepared-validation",
                "off",
            ];
            if preset_first {
                args.push("--speed-preset");
                args.extend(overrides);
            } else {
                args.extend(overrides);
                args.push("--speed-preset");
            }
            let Command::Serve(df_override) = parse_args(args).unwrap().command else {
                panic!("expected Serve");
            };
            assert_eq!(df_override.dflash2_precision.as_deref(), Some("bf16"));
            assert_eq!(df_override.dflash2_max_drafts, Some(4));
            assert_eq!(df_override.dflash2_verify_rows.as_deref(), Some("all"));
            assert!(!df_override.prepared_validation);
        }

        let Command::Serve(preset64) =
            parse_args(["serve", "--speed-preset", "--output-packet", "64"])
                .unwrap()
                .command
        else {
            panic!("expected Serve");
        };
        assert_eq!(preset64.first_output_packet, Some(8));
        let preset64_flags = build_carrier_flags(
            &preset64,
            SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
            Path::new("/tmp/state"),
        )
        .unwrap();
        assert!(
            preset64_flags
                .windows(2)
                .any(|pair| pair == ["--native-mtp-first-output-packet", "8"])
        );
        assert!(
            !preset64_flags
                .iter()
                .any(|arg| arg == "--native-mtp-fast-round")
        );
        assert!(
            preset64_flags
                .windows(2)
                .any(|pair| pair == ["--native-mtp-depth", "3"])
        );
        assert!(
            preset64_flags
                .windows(2)
                .any(|pair| pair == ["--native-verify-nax", "on"])
        );
        assert!(
            preset64_flags
                .windows(2)
                .any(|pair| pair == ["--native-verify-hc-gdn-fused", "on"])
        );
        assert!(
            !preset64_flags
                .iter()
                .any(|flag| flag == "--native-gdn-prefill-ported")
        );
        assert!(
            !preset64_flags
                .iter()
                .any(|flag| flag == "--native-prefill-router-fast")
        );
        assert!(
            preset64_flags
                .windows(2)
                .any(|pair| pair == ["--tool-arg-validation", "executor"])
        );

        let Command::Serve(bad_first) = parse_args([
            "serve",
            "--output-packet",
            "8",
            "--first-output-packet",
            "16",
        ])
        .unwrap()
        .command
        else {
            panic!("expected Serve");
        };
        assert!(
            build_carrier_flags(
                &bad_first,
                SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                Path::new("/tmp/state"),
            )
            .is_err()
        );

        let Command::Serve(vnax_off) = parse_args(["serve", "--verify-nax", "--no-verify-nax"])
            .unwrap()
            .command
        else {
            panic!("expected Serve");
        };
        assert!(!vnax_off.verify_nax);
        let Command::Serve(vnax_native_on) = parse_args(["serve", "--native-verify-nax", "on"])
            .unwrap()
            .command
        else {
            panic!("expected Serve");
        };
        assert!(vnax_native_on.verify_nax);
        let Command::Serve(vnax_native_off) =
            parse_args(["serve", "--verify-nax", "--native-verify-nax=off"])
                .unwrap()
                .command
        else {
            panic!("expected Serve");
        };
        assert!(!vnax_native_off.verify_nax);

        let Command::Serve(df_bf16) = parse_args([
            "serve",
            "--drafter",
            "dflash2",
            "--dflash2-dir",
            "/tmp/dflash2",
            "--dflash2-precision",
            "bf16",
        ])
        .unwrap()
        .command
        else {
            panic!("expected Serve");
        };
        let df_flags = build_carrier_flags(
            &df_bf16,
            SpeculativeAsset::DFlash2Dir(Path::new("/tmp/dflash2")),
            Path::new("/tmp/state"),
        )
        .unwrap();
        assert!(
            df_flags
                .windows(2)
                .any(|pair| pair == ["--native-dflash2-precision", "bf16"])
        );

        let Command::Serve(exec_args) = parse_args(["serve", "--tool-arg-validation", "executor"])
            .unwrap()
            .command
        else {
            panic!("expected Serve");
        };
        assert_eq!(exec_args.tool_arg_validation, "executor");
        let exec_flags = build_carrier_flags(
            &exec_args,
            SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
            Path::new("/tmp/state"),
        )
        .unwrap();
        assert!(
            exec_flags
                .windows(2)
                .any(|pair| pair == ["--tool-arg-validation", "executor"])
        );

        let Command::Serve(rf_bad) = parse_args(["serve", "--native-prefill-router-fast", "on"])
            .unwrap()
            .command
        else {
            panic!("expected Serve");
        };
        assert!(
            build_carrier_flags(
                &rf_bad,
                SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                Path::new("/tmp/state"),
            )
            .is_err()
        );
        let Command::Serve(rf_off) = parse_args([
            "serve",
            "--speed-preset",
            "--native-prefill-router-fast",
            "off",
        ])
        .unwrap()
        .command
        else {
            panic!("expected Serve");
        };
        let rf_flags = build_carrier_flags(
            &rf_off,
            SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
            Path::new("/tmp/state"),
        )
        .unwrap();
        assert!(
            rf_flags
                .windows(2)
                .any(|pair| pair == ["--native-prefill-router-fast", "off"])
        );
        assert!(parse_args(["serve", "--tool-arg-validation", "permissive"]).is_err());
    }

    #[test]
    fn default_derived_pack_path_prefers_asa_and_falls_back_to_legacy_gguf() {
        use std::fs;
        let dir = std::env::temp_dir().join(format!(
            "tokspur_asa_pack_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let derived = dir.join("derived");
        fs::create_dir_all(&derived).unwrap();

        for (pack_v3, asa_name, legacy_name) in [
            (false, "tokspur-pack-v2.asa.gguf", "tokspur-pack-v2.gguf"),
            (true, "tokspur-pack-v3.asa.gguf", "tokspur-pack-v3.gguf"),
        ] {
            assert_eq!(
                serve::default_derived_pack_path(&dir, pack_v3),
                derived.join(asa_name)
            );

            let legacy = derived.join(legacy_name);
            fs::write(&legacy, b"legacy").unwrap();
            assert_eq!(serve::default_derived_pack_path(&dir, pack_v3), legacy);

            let asa = derived.join(asa_name);
            fs::write(&asa, b"asa").unwrap();
            assert_eq!(serve::default_derived_pack_path(&dir, pack_v3), asa);

            fs::remove_file(&asa).unwrap();
            fs::remove_file(&legacy).unwrap();
        }

        let v2_asa = derived.join("tokspur-pack-v2.asa.gguf");
        assert_eq!(
            serve::resolve_derived_pack_path(&dir, false, true),
            derived.join("tokspur-pack-v3.asa.gguf")
        );
        fs::write(&v2_asa, b"v2").unwrap();
        assert_eq!(serve::resolve_derived_pack_path(&dir, false, true), v2_asa);

        let v3_legacy = derived.join("tokspur-pack-v3.gguf");
        fs::write(&v3_legacy, b"v3-legacy").unwrap();
        assert_eq!(
            serve::resolve_derived_pack_path(&dir, false, true),
            v3_legacy
        );

        let v3_asa = derived.join("tokspur-pack-v3.asa.gguf");
        fs::write(&v3_asa, b"v3-asa").unwrap();
        assert_eq!(serve::resolve_derived_pack_path(&dir, false, true), v3_asa);
        assert_eq!(serve::resolve_derived_pack_path(&dir, false, false), v2_asa);

        fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod runtime_check_tests;
