//! Private V8 and branch8 reply admission, ported one to one from `native/src/serve_verify8.cpp` (CPU only: no target, tokenizer, IO or model work happens here).
//!
//! Evaluation order and short-circuiting follow the C++ exactly, because the first failing field or check decides which message a caller reports. Unsigned arithmetic wraps where the C++ arithmetic would.

use std::fmt;

use tokspur_json::{Json, JsonError, from_chars_f64};

use crate::qsa::{QsaError, QsaShape, qsa_dispatches};

/// The C++ `FormatError` raised by V8/branch8 admission.
#[derive(Clone, PartialEq, Eq)]
pub enum Verify8Error {
    /// `get`: a required field is absent.
    MissingField(String),
    /// `need`: an admission check refused the value.
    Refused(&'static str),
    /// A JSON accessor or serialization refused a field.
    Json(JsonError),
    /// The dispatch formula refused a position.
    Dispatch(QsaError),
}

impl fmt::Debug for Verify8Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(name) => formatter
                .debug_struct("MissingField")
                .field("name_bytes", &name.len())
                .finish(),
            Self::Refused(reason) => formatter.debug_tuple("Refused").field(reason).finish(),
            Self::Json(error) => formatter.debug_tuple("Json").field(error).finish(),
            Self::Dispatch(error) => formatter.debug_tuple("Dispatch").field(error).finish(),
        }
    }
}

impl fmt::Display for Verify8Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(name) => write!(formatter, "private V8 missing field: {name}"),
            Self::Refused(reason) => formatter.write_str(reason),
            Self::Json(error) => error.fmt(formatter),
            Self::Dispatch(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Verify8Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Dispatch(error) => Some(error),
            Self::MissingField(_) | Self::Refused(_) => None,
        }
    }
}

impl From<JsonError> for Verify8Error {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

impl From<QsaError> for Verify8Error {
    fn from(error: QsaError) -> Self {
        Self::Dispatch(error)
    }
}

type Result<T> = std::result::Result<T, Verify8Error>;

fn need(ok: bool, why: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Verify8Error::Refused(why))
    }
}

fn get<'a>(v: &'a Json, k: &str) -> Result<&'a Json> {
    v.find(k)?
        .ok_or_else(|| Verify8Error::MissingField(k.to_owned()))
}

fn integer(v: &Json, k: &str) -> Result<u64> {
    Ok(get(v, k)?.unsigned_integer()?)
}

fn real_value(value: &Json) -> Result<f64> {
    let lexeme = value.number_lexeme()?;
    from_chars_f64(lexeme)
        .filter(|n| n.is_finite())
        .ok_or(Verify8Error::Refused("private V8 nonfinite number"))
}

fn real(v: &Json, k: &str) -> Result<f64> {
    real_value(get(v, k)?)
}

/// `std::max({1.0, |a|, |b|})`, which compares with `<` from the left.
fn max3(a: f64, b: f64, c: f64) -> f64 {
    let mut largest = a;
    if largest < b {
        largest = b;
    }
    if largest < c {
        largest = c;
    }
    largest
}

fn close(a: f64, b: f64) -> Result<()> {
    need(
        (a - b).abs() <= 1e-6 * max3(1.0, a.abs(), b.abs()),
        "private V8 costs do not reconcile",
    )
}

fn str_is(value: &Json, expected: &str) -> Result<bool> {
    Ok(value.string()? == expected.as_bytes())
}

fn lowercase_hex64(hash: &[u8]) -> bool {
    hash.len() == 64 && hash.iter().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'))
}

fn keys(v: &Json, allowed: &[&str], branch8: bool, stage8: bool, fast_math: bool) -> Result<()> {
    for (k, _) in v.object()? {
        let k = k.as_slice();
        if cfg!(feature = "fast-math") && fast_math && k == b"qsa_channel_round_calls" {
            continue;
        }
        if cfg!(feature = "qsa-stage-v8") && stage8 && k == b"qsa_stage_v8_round_calls" {
            continue;
        }
        if cfg!(feature = "branch8")
            && branch8
            && matches!(
                k,
                b"broker_epoch"
                    | b"command_id"
                    | b"lease_id"
                    | b"broker_protocol"
                    | b"broker_batch_width"
            )
        {
            continue;
        }
        need(
            allowed.iter().any(|name| name.as_bytes() == k),
            "private V8 unknown event/command field",
        )?;
    }
    Ok(())
}

fn plain_keys(v: &Json, allowed: &[&str]) -> Result<()> {
    keys(v, allowed, false, false, false)
}

const HC: [&str; 5] = ["scalar", "many1", "many2", "many4", "many8"];

#[cfg(feature = "qsa-stage-v8")]
const STAGE8_FIELDS: [&str; 7] = [
    "qsa_stage_v8_prepared",
    "qsa_stage_v8_enabled",
    "qsa_stage_v8_scratch_logical_bytes",
    "qsa_stage_v8_scratch_charged_bytes",
    "qsa_stage_v8_calls",
    "qsa_stage_v8_physical_dispatches",
    "qsa_stage_v8_extra_dispatches",
];

#[cfg(feature = "qsa-stage-v8")]
fn stage8_metadata(v: &Json) -> Result<bool> {
    let prepared = v.find("qsa_stage_v8_prepared")?;
    let event = v.find("event")?;
    let done = match event {
        Some(event) => str_is(event, "done")?,
        None => false,
    };
    for (name, _) in v.object()? {
        if name.starts_with(b"qsa_stage_v8_") {
            let known = STAGE8_FIELDS
                .iter()
                .any(|field| field.as_bytes() == name.as_slice());
            need(
                prepared.is_some() && (known || (done && name == b"qsa_stage_v8_request_calls")),
                "private staged-QSA unknown/incomplete metadata family",
            )?;
        }
    }
    let Some(prepared) = prepared else {
        return Ok(false);
    };
    need(
        prepared.boolean()?
            && integer(v, "qsa_stage_v8_scratch_logical_bytes")? == 196_896
            && integer(v, "qsa_stage_v8_scratch_charged_bytes")? == 212_992,
        "private staged-QSA must prepare the exact OFF/ON storage",
    )?;
    let enabled = get(v, "qsa_stage_v8_enabled")?.boolean()?;
    let calls = integer(v, "qsa_stage_v8_calls")?;
    need(
        calls % 12 == 0
            && calls <= u64::MAX / 2
            && integer(v, "qsa_stage_v8_physical_dispatches")? == 2 * calls
            && integer(v, "qsa_stage_v8_extra_dispatches")? == calls
            && (enabled || calls == 0),
        "private staged-QSA physical cost identity",
    )?;
    if done {
        integer(v, "qsa_stage_v8_request_calls")?;
    }
    Ok(true)
}

#[cfg(feature = "qsa-stage-v8")]
fn stage8_same(value: &Json, prepared: &Json) -> Result<()> {
    let present = stage8_metadata(prepared)?;
    need(
        stage8_metadata(value)? == present,
        "private staged-QSA selector presence changed",
    )?;
    if present {
        for field in &STAGE8_FIELDS[..4] {
            need(
                get(value, field)?.dump()? == get(prepared, field)?.dump()?,
                "private staged-QSA immutable policy changed",
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "fast-math")]
const FAST_FIELDS: [&str; 8] = [
    "ple_gate_transform",
    "ple_gate_transform_schema",
    "qsa_channel_chronological_mode",
    "qsa_channel_chronological_schema",
    "qsa_channel_factor_logical_bytes",
    "qsa_channel_factor_charged_bytes",
    "qsa_channel_calls",
    "qsa_channel_extra_dispatches",
];

#[cfg(feature = "fast-math")]
fn fast_metadata(value: &Json) -> Result<Option<bool>> {
    let mode = value.find("qsa_channel_chronological_mode")?;
    let event = value.find("event")?;
    let done = match event {
        Some(event) => str_is(event, "done")?,
        None => false,
    };
    for (name, _) in value.object()? {
        if name.starts_with(b"qsa_channel_") || name.starts_with(b"ple_gate_") {
            let known = FAST_FIELDS
                .iter()
                .any(|field| field.as_bytes() == name.as_slice());
            need(
                mode.is_some() && (known || (done && name == b"qsa_channel_request_calls")),
                "private fast-math unknown/incomplete metadata family",
            )?;
        }
    }
    let Some(mode) = mode else {
        return Ok(None);
    };
    need(
        str_is(mode, "off")? || str_is(mode, "on")?,
        "private fast-math channel mode",
    )?;
    need(
        stage8_metadata(value)?
            && get(value, "qsa_stage_v8_enabled")?.boolean()?
            && str_is(get(value, "ple_gate_transform")?, "signed-root")?
            && integer(value, "ple_gate_transform_schema")? == 1
            && integer(value, "qsa_channel_chronological_schema")? == 1
            && integer(value, "qsa_channel_factor_logical_bytes")? == 393_792
            && integer(value, "qsa_channel_factor_charged_bytes")? == 409_600,
        "private fast-math fixed PLE/channel preparation",
    )?;
    let channel = str_is(mode, "on")?;
    let calls = integer(value, "qsa_channel_calls")?;
    need(
        calls % 12 == 0
            && integer(value, "qsa_channel_extra_dispatches")? == calls
            && if channel {
                calls == integer(value, "qsa_stage_v8_calls")?
            } else {
                calls == 0
            },
        "private fast-math physical cost identity",
    )?;
    if done {
        integer(value, "qsa_channel_request_calls")?;
    }
    Ok(Some(channel))
}

#[cfg(feature = "fast-math")]
fn fast_same(value: &Json, prepared: &Json) -> Result<()> {
    let mode = fast_metadata(prepared)?;
    need(
        fast_metadata(value)? == mode,
        "private fast-math selector presence/mode changed",
    )?;
    if mode.is_some() {
        for field in &FAST_FIELDS[..6] {
            need(
                get(value, field)?.dump()? == get(prepared, field)?.dump()?,
                "private fast-math immutable policy changed",
            )?;
        }
    }
    Ok(())
}

/// `verify8_fast_math`: absent, or explicit staged (`false`) / channel (`true`).
#[cfg(feature = "fast-math")]
pub fn verify8_fast_math(value: &Json) -> Result<Option<bool>> {
    fast_metadata(value)
}

fn zero_routes(v: &Json, name: &str, count: usize) -> Result<()> {
    let r = get(v, name)?;
    need(r.object()?.len() == count, "private V8 route shape")?;
    if name.starts_with("wide_q8_exact_") {
        plain_keys(
            r,
            &["qkv_scalar", "qkv_many1", "output_scalar", "output_many1"],
        )?;
    } else {
        plain_keys(
            r,
            &[
                "qkv_many2",
                "qkv_many4",
                "qkv_many8",
                "output_many2",
                "output_many4",
                "output_many8",
            ],
        )?;
    }
    for (_, n) in r.object()? {
        need(n.unsigned_integer()? == 0, "private V8 inactive routes")?;
    }
    Ok(())
}

const FIXED_POLICY: [(&str, u64); 28] = [
    ("small_gemv_column_splits", 16),
    ("block_gemv_formats", 3),
    ("known_token_capacity", 8),
    ("hc_q8_splits", 0),
    ("hc_q8_scratch_logical_bytes", 0),
    ("hc_q8_scratch_charged_bytes", 0),
    ("recurrent_replay_captures", 0),
    ("mapped_weight_region_limit_bytes", 0),
    ("capture_resource_bytes", 0),
    ("capture_binding_aliases", 0),
    ("capture_binding_alias_span_bytes", 0),
    ("capture_binding_alias_resource_bytes", 0),
    ("live_captures", 0),
    ("profile_counter_payload_bytes", 0),
    ("verification_max_tokens", 8),
    ("native_limit_bytes", 96 << 30),
    ("hc_exact_rows_per_group", 1),
    ("hc_exact_unroll", 16),
    ("hc_exact_scratch_logical_bytes", 0),
    ("hc_exact_scratch_charged_bytes", 0),
    ("wide_q8_exact_rows_per_group", 4),
    ("wide_q8_exact_unroll", 16),
    ("wide_q8_exact_scratch_logical_bytes", 0),
    ("wide_q8_exact_scratch_charged_bytes", 0),
    ("wide_q8_many_rows_per_group", 4),
    ("wide_q8_many_unroll", 4),
    ("wide_q8_many_scratch_logical_bytes", 0),
    ("wide_q8_many_scratch_charged_bytes", 0),
];

const STORAGE_FIELDS: [&str; 5] = [
    "checkpoint_slots",
    "checkpoint_combined_budget_bytes",
    "checkpoint_host_bytes",
    "checkpoint_store_host_bytes",
    "checkpoint_gpu_bytes",
];

const GRAPH_FIELDS: [&str; 5] = [
    "tracked_metal_buffer_bytes",
    "peak_metal_buffer_bytes",
    "gpu_allocation_count",
    "gpu_buffer_limit_bytes",
    "imported_weight_backing_bytes",
];

fn fixed(v: &Json, branch8: bool) -> Result<()> {
    #[cfg(feature = "fast-math")]
    {
        need(
            fast_metadata(v)?.is_none() || branch8,
            "private fast HTTP math requires the branch8 wire",
        )?;
        need(
            v.find("licensed_q4_routed_mode")?.is_none()
                && v.find("batch_phase_clock_mode")?.is_none(),
            "private fast HTTP refuses unrelated numerical/telemetry tiers",
        )?;
    }
    #[cfg(feature = "qsa-stage-v8")]
    {
        let staged = stage8_metadata(v)?;
        need(
            !staged || branch8,
            "private staged HTTP requires the branch8 wire",
        )?;
    }
    for (k, n) in FIXED_POLICY {
        need(integer(v, k)? == n, "private V8 fixed policy differs")?;
    }
    if cfg!(feature = "branch8") && branch8 {
        need(
            integer(v, "checkpoint_slots")? != 0
                && integer(v, "checkpoint_slots")? <= 4
                && integer(v, "checkpoint_combined_budget_bytes")? != 0
                && integer(v, "checkpoint_combined_budget_bytes")? <= (512 << 20)
                && get(v, "verification8_branch_reuse")?.boolean()?,
            "private branch8 history consent/configuration",
        )?;
        let mut total: u64 = 0;
        for name in [
            "checkpoint_host_bytes",
            "checkpoint_store_host_bytes",
            "checkpoint_gpu_bytes",
        ] {
            let bytes = integer(v, name)?;
            need(
                bytes != 0
                    && bytes <= integer(v, "checkpoint_combined_budget_bytes")?.wrapping_sub(total),
                "private branch8 history budget",
            )?;
            total = total.wrapping_add(bytes);
        }
    } else {
        for name in STORAGE_FIELDS {
            need(
                integer(v, name)? == 0,
                "private V8 without explicit branch8 consent cannot use history",
            )?;
        }
    }
    need(
        str_is(get(v, "stack_execution")?, "direct")?
            && str_is(get(v, "hc_q8_numerical_tier")?, "ordinary")?
            && str_is(get(v, "hc_exact_mode")?, "combined")?
            && get(v, "known_residual_reuse")?.boolean()?
            && get(v, "prompt_lookup")?.boolean()?
            && get(v, "verification_head_many")?.boolean()?
            && !get(v, "replay_prepared")?.boolean()?
            && !get(v, "wide_q8_exact_enabled")?.boolean()?
            && !get(v, "wide_q8_many_enabled")?.boolean()?
            && !get(v, "timings_qualified")?.boolean()?,
        "private V8 policy drift",
    )?;
    need(
        v.find("expert_route_trace")?.is_none()
            && v.find("trace_expert_routes")?.is_none()
            && v.find("moe_token_tile_enabled")?.is_none(),
        "private V8 unrelated feature metadata",
    )?;
    zero_routes(v, "wide_q8_exact_routes", 4)?;
    zero_routes(v, "wide_q8_many_routes", 6)
}

/// This private endpoint binds the adopted fixed 48-layer architecture: 12 QSA layers, compression 4 and a 2048-token selection budget. 2222 includes 18 ordinary QSA dispatches per layer. Selection begins at token 2052 and adds sort/merge work; never bless an observed count.
fn ordinary(prefix: u32, stage8_enabled: bool, channel_enabled: bool) -> Result<u64> {
    const SHAPE: QsaShape = QsaShape {
        channels: 2560,
        query_heads: 24,
        kv_heads: 2,
        head_width: 256,
        index_heads: 4,
        index_width: 128,
        rotary_width: 64,
        compression: 4,
        token_budget: 2048,
        capacity: 65536,
    };
    Ok(2222u64
        .wrapping_add(12u64.wrapping_mul(qsa_dispatches(SHAPE, prefix)?.wrapping_sub(18)))
        .wrapping_add(12 * u64::from(stage8_enabled))
        .wrapping_add(12 * u64::from(channel_enabled)))
}

fn sampling(v: &Json, policy: &Json, seed: u64, prefix: u32) -> Result<()> {
    let s = get(v, "sampling")?;
    need(
        get(s, "valid")?.boolean()?
            && integer(s, "seed")? == seed
            && integer(s, "entropy_prefix")? == u64::from(prefix)
            && integer(s, "entropy_words")? == 0,
        "private greedy V8 sampling ancestry",
    )?;
    for (k, n) in policy.object()? {
        // `real(s, k)`, then `real(pair, "x")` over `{"x": n}`, i.e. `n` itself.
        let observed = s
            .find(k)?
            .ok_or_else(|| Verify8Error::MissingField(String::from_utf8_lossy(k).into_owned()))?;
        need(
            real_value(observed)? == real_value(n)?,
            "private V8 sampling policy changed",
        )?;
    }
    Ok(())
}

fn hc_shape(r: &Json) -> Result<()> {
    need(r.object()?.len() == HC.len(), "private V8 HC route shape")?;
    for name in HC {
        integer(r, name)?;
    }
    Ok(())
}

fn ready(v: &Json, branch8: bool) -> Result<()> {
    fixed(v, branch8)?;
    let transport = if cfg!(feature = "branch8") && branch8 {
        get(v, "checkpoint_broker")?.boolean()? && integer(v, "broker_epoch")? != 0
    } else {
        !get(v, "checkpoint_broker")?.boolean()? && integer(v, "broker_epoch")? == 0
    };
    need(
        str_is(get(v, "event")?, "ready")?
            && integer(v, "maximum_generated_tokens")? == 4096
            && transport
            && !get(v, "weight_copy")?.boolean()?
            && str_is(get(v, "gpu_profile_mode")?, "disabled")?
            && get(v, "allow_prompt_lookup_experiments")?.boolean()?
                == expected_lookup_experiments(v)?,
        "private V8 ready identity",
    )?;
    #[cfg(feature = "qsa-stage-v8")]
    if stage8_metadata(v)? {
        need(
            integer(v, "qsa_stage_v8_calls")? == 0,
            "private staged-QSA pre-ready model work",
        )?;
    }
    for name in [
        "allow_request_kernel_experiments",
        "allow_request_stack_experiments",
        "allow_request_prefill_experiments",
    ] {
        need(
            !get(v, name)?.boolean()?,
            "private V8 request-time experiment",
        )?;
    }
    need(
        integer(v, "lookup_match_tokens")? == 16
            && integer(v, "lookup_history_tokens")? == 65536
            && integer(v, "lookup_proposal_tokens")? == 7
            && integer(v, "lookup_verification_budget_bytes")? == (128 << 20),
        "private V8 capability",
    )?;
    let mut total: u64 = 0;
    for name in [
        "lookup_caller_host_bytes",
        "verification_host_bytes",
        "verification_gpu_bytes",
    ] {
        let bytes = integer(v, name)?;
        need(
            bytes != 0 && bytes <= (128u64 << 20).wrapping_sub(total),
            "private V8 combined budget",
        )?;
        total = total.wrapping_add(bytes);
    }
    need(
        total == integer(v, "lookup_verification_charged_bytes")?
            && integer(v, "verification_tape_payload_bytes")? != 0
            && integer(v, "verification_tape_payload_bytes")?
                <= integer(v, "verification_gpu_bytes")?,
        "private V8 actual verifier charge",
    )?;
    let live = integer(v, "tracked_metal_buffer_bytes")?;
    let peak = integer(v, "peak_metal_buffer_bytes")?;
    let budget = integer(v, "gpu_buffer_limit_bytes")?;
    need(
        budget == (96u64 << 30) - (512 << 20) - (2 << 30)
            && live != 0
            && live <= peak
            && peak <= budget
            && integer(v, "imported_weight_backing_bytes")? <= live
            && integer(v, "gpu_allocation_count")? != 0,
        "private V8 graph ledger",
    )?;
    hc_shape(get(v, "hc_exact_routes")?)?;
    for name in HC {
        need(
            integer(get(v, "hc_exact_routes")?, name)? == 0,
            "private V8 pre-ready routes",
        )?;
    }
    Ok(())
}

/// The ready check's `allow_prompt_lookup_experiments` expectation: staged builds expect it off exactly when staged metadata is present.
fn expected_lookup_experiments(v: &Json) -> Result<bool> {
    #[cfg(feature = "qsa-stage-v8")]
    {
        Ok(!stage8_metadata(v)?)
    }
    #[cfg(not(feature = "qsa-stage-v8"))]
    {
        let _ = v;
        Ok(true)
    }
}

/// `validate_verify8_ready`.
pub fn validate_verify8_ready(v: &Json) -> Result<()> {
    ready(v, false)
}

#[cfg(feature = "branch8")]
mod branch {
    use super::*;

    pub(super) const COST_FIELDS: [&str; 10] = [
        "lookup_appended_tokens",
        "lookup_indexed_grams",
        "lookup_truncated_grams",
        "lookup_cleared_grams",
        "lookup_resets",
        "lookup_truncations",
        "lookup_queries",
        "lookup_candidates_examined",
        "lookup_token_comparisons",
        "lookup_work_limited_queries",
    ];

    pub(super) fn grams(prefix: u64) -> u64 {
        if prefix < 16 { 0 } else { prefix - 15 }
    }

    pub(super) fn branch_identity(value: &Json) -> Result<()> {
        need(
            str_is(get(value, "broker_protocol")?, "branch8-v1")?
                && integer(value, "broker_batch_width")? == 8
                && integer(value, "broker_epoch")? != 0,
            "private branch8 transport identity",
        )?;
        integer(value, "command_id")?;
        integer(value, "lease_id")?;
        Ok(())
    }

    pub(super) fn frozen_branch_storage(value: &Json, prepared: &Json) -> Result<()> {
        for name in STORAGE_FIELDS {
            need(
                integer(value, name)? == integer(prepared, name)?,
                "private branch8 prepared history changed",
            )?;
        }
        // The C++ re-derives the snapshot per field; it is a pure function of `value`, so one derivation yields the same result and first error.
        let session = branch8_session(value)?;
        for name in GRAPH_FIELDS {
            need(
                integer(session, name)? == integer(prepared, name)?,
                "private branch8 frozen graph changed",
            )?;
        }
        #[cfg(feature = "qsa-stage-v8")]
        stage8_same(session, prepared)?;
        #[cfg(feature = "fast-math")]
        fast_same(session, prepared)?;
        Ok(())
    }

    const SESSION_FIELDS: &[&str] = &[
        "committed_prefix",
        "sampling",
        "logits_available",
        "continuation_allowed",
        "stop_id",
        "final_logit_top8",
        "hc_exact_final_logit_sha256",
        "hc_exact_routes",
        "lookup_history_epoch",
        "lookup_appended_tokens",
        "lookup_indexed_grams",
        "lookup_truncated_grams",
        "lookup_cleared_grams",
        "lookup_resets",
        "lookup_truncations",
        "lookup_queries",
        "lookup_candidates_examined",
        "lookup_token_comparisons",
        "lookup_work_limited_queries",
        "tracked_metal_buffer_bytes",
        "peak_metal_buffer_bytes",
        "gpu_allocation_count",
        "gpu_buffer_limit_bytes",
        "imported_weight_backing_bytes",
        #[cfg(feature = "qsa-stage-v8")]
        "qsa_stage_v8_prepared",
        #[cfg(feature = "qsa-stage-v8")]
        "qsa_stage_v8_enabled",
        #[cfg(feature = "qsa-stage-v8")]
        "qsa_stage_v8_scratch_logical_bytes",
        #[cfg(feature = "qsa-stage-v8")]
        "qsa_stage_v8_scratch_charged_bytes",
        #[cfg(feature = "qsa-stage-v8")]
        "qsa_stage_v8_calls",
        #[cfg(feature = "qsa-stage-v8")]
        "qsa_stage_v8_physical_dispatches",
        #[cfg(feature = "qsa-stage-v8")]
        "qsa_stage_v8_extra_dispatches",
        #[cfg(feature = "fast-math")]
        "ple_gate_transform",
        #[cfg(feature = "fast-math")]
        "ple_gate_transform_schema",
        #[cfg(feature = "fast-math")]
        "qsa_channel_chronological_mode",
        #[cfg(feature = "fast-math")]
        "qsa_channel_chronological_schema",
        #[cfg(feature = "fast-math")]
        "qsa_channel_factor_logical_bytes",
        #[cfg(feature = "fast-math")]
        "qsa_channel_factor_charged_bytes",
        #[cfg(feature = "fast-math")]
        "qsa_channel_calls",
        #[cfg(feature = "fast-math")]
        "qsa_channel_extra_dispatches",
    ];

    /// Checks a sorted, unique top-eight logit list (`final_logit_top8`).
    pub(super) fn sorted_top8(
        top: &[Json],
        sorted: &'static str,
        duplicate: &'static str,
    ) -> Result<()> {
        let mut last = f64::INFINITY;
        let mut last_id = 0;
        for (i, entry) in top.iter().enumerate() {
            plain_keys(entry, &["id", "logit"])?;
            let id = integer(entry, "id")?;
            let score = real(entry, "logit")?;
            need(
                id < 248_320 && (score < last || (i > 0 && score == last && id > last_id)),
                sorted,
            )?;
            for earlier in &top[..i] {
                need(integer(earlier, "id")? != id, duplicate)?;
            }
            last = score;
            last_id = id;
        }
        Ok(())
    }

    /// `branch8_session`: the fresh native snapshot inside a branch8 reply.
    pub fn branch8_session(reply: &Json) -> Result<&Json> {
        branch_identity(reply)?;
        let state = get(reply, "branch8_session")?;
        plain_keys(state, SESSION_FIELDS)?;
        #[cfg(feature = "qsa-stage-v8")]
        stage8_metadata(state)?;
        #[cfg(feature = "fast-math")]
        fast_metadata(state)?;
        let prefix = integer(state, "committed_prefix")?;
        need(
            prefix <= 65536 && integer(state, "lookup_history_epoch")? != 0,
            "private branch8 state prefix/epoch",
        )?;
        for name in COST_FIELDS {
            integer(state, name)?;
        }
        for name in GRAPH_FIELDS {
            integer(state, name)?;
        }
        hc_shape(get(state, "hc_exact_routes")?)?;
        let policy = get(state, "sampling")?;
        need(
            get(policy, "valid")?.boolean()?
                && integer(policy, "entropy_prefix")? == prefix
                && integer(policy, "entropy_words")? == 0
                && real(policy, "temperature")? == 0.0
                && integer(policy, "top_k")? <= 248_320
                && real(policy, "top_p")? > 0.0
                && real(policy, "top_p")? <= 1.0
                && real(policy, "min_p")? >= 0.0
                && real(policy, "min_p")? <= 1.0,
            "private branch8 greedy sampler state",
        )?;
        integer(policy, "seed")?;
        let logits = get(state, "logits_available")?.boolean()?;
        let stop = get(state, "stop_id")?;
        need(
            logits == (prefix != 0)
                && (stop.is_null() || (logits && stop.unsigned_integer()? < 248_320))
                && get(state, "continuation_allowed")?.boolean()? == (logits && stop.is_null()),
            "private branch8 state logits/terminal",
        )?;
        let top = get(state, "final_logit_top8")?.array()?;
        need(
            top.len() == if prefix != 0 { 8 } else { 0 },
            "private branch8 state top8 shape",
        )?;
        if prefix != 0 {
            let hash = get(state, "hc_exact_final_logit_sha256")?.string()?;
            need(lowercase_hex64(hash), "private branch8 full-logit identity")?;
            sorted_top8(
                top,
                "private branch8 sorted state logits",
                "private branch8 duplicate state logit",
            )?;
        } else {
            need(
                get(state, "hc_exact_final_logit_sha256")?.is_null(),
                "private branch8 empty state hash",
            )?;
        }
        // A fresh native snapshot must agree with every duplicate field in its actual envelope. No historical packet is rewritten to impersonate it.
        for (name, value) in state.object()? {
            if let Some(outside) = reply.find(name)? {
                need(
                    outside.dump()? == value.dump()?,
                    "private branch8 snapshot/envelope disagreement",
                )?;
            }
        }
        Ok(state)
    }

    /// `validate_branch8_ready`.
    pub fn validate_branch8_ready(value: &Json) -> Result<()> {
        ready(value, true)?;
        let state = branch8_session(value)?;
        need(
            integer(value, "command_id")? == 0
                && integer(value, "lease_id")? == 0
                && integer(state, "committed_prefix")? == 0
                && integer(state, "lookup_history_epoch")? == 1,
            "private branch8 fresh readiness ancestry",
        )?;
        for name in COST_FIELDS {
            need(
                integer(state, name)? == 0,
                "private branch8 pre-ready history work",
            )?;
        }
        frozen_branch_storage(value, value)
    }

    /// `validate_branch8_control`.
    pub fn validate_branch8_control(
        prepared: &Json,
        previous: &Json,
        job: &Json,
        value: &Json,
        cold_resync: bool,
    ) -> Result<()> {
        branch_identity(value)?;
        let state = branch8_session(value)?;
        let old = branch8_session(previous)?;
        keys(
            value,
            &[
                "event",
                "op",
                "outcome",
                "model_steps",
                "generated_ids",
                "common_prefix",
                "committed_prefix",
                "sampling",
                "stop_id",
                "logits_available",
                "continuation_allowed",
                "checkpoint_slots",
                "checkpoint_combined_budget_bytes",
                "checkpoint_host_bytes",
                "checkpoint_store_host_bytes",
                "checkpoint_gpu_bytes",
                "checkpoint_retained",
                "control_wall_ms",
                "timings_qualified",
                "branch8_session",
            ],
            true,
            false,
            false,
        )?;
        frozen_branch_storage(value, prepared)?;
        need(
            str_is(get(value, "event")?, "control_done")?
                && get(value, "op")?.string()? == get(job, "op")?.string()?
                && integer(value, "model_steps")? == 0
                && get(value, "generated_ids")?.array()?.is_empty()
                && !get(value, "timings_qualified")?.boolean()?
                && real(value, "control_wall_ms")? >= 0.0
                && integer(value, "checkpoint_retained")? <= integer(prepared, "checkpoint_slots")?,
            "private branch8 zero-model-step control envelope",
        )?;
        let operation = get(job, "op")?.string()?;
        let prefix = integer(state, "committed_prefix")?;
        let before = integer(old, "committed_prefix")?;
        let outcome = get(value, "outcome")?.string()?;
        let common = integer(value, "common_prefix")?;
        let command = integer(previous, "command_id")?;
        need(
            command < u64::MAX
                && integer(value, "broker_epoch")? == integer(prepared, "broker_epoch")?
                && if cold_resync {
                    integer(value, "command_id")? > command
                } else {
                    integer(value, "command_id")? == command + 1
                },
            "private branch8 control command ancestry",
        )?;
        let lease = integer(previous, "lease_id")?;
        need(
            if operation == b"branch_begin" {
                integer(value, "lease_id")? > lease
            } else {
                lease != 0 && integer(value, "lease_id")? == lease
            },
            "private branch8 control lease ancestry",
        )?;
        if operation == b"branch_capture" {
            plain_keys(job, &["op"])?;
            need(
                !cold_resync
                    && prefix != 0
                    && common == before
                    && prefix == before
                    && outcome == b"captured"
                    && state.dump()? == old.dump()?,
                "private branch8 capture changed current state or lifetime costs",
            )?;
            return Ok(());
        }
        need(
            operation == b"branch_begin",
            "private branch8 unknown control",
        )?;
        plain_keys(
            job,
            &[
                "op",
                "prompt_ids",
                "generation_requested",
                "allow_restore",
                "sampling",
                "seed",
            ],
        )?;
        let count = get(job, "prompt_ids")?.array()?.len() as u64;
        let allow = get(job, "allow_restore")?.boolean()?;
        need(
            count != 0
                && count <= 65536
                && (!cold_resync || !allow)
                && (allow || common == 0)
                && common <= before.min(count)
                && prefix <= common
                && ((outcome == b"restored" && prefix != 0 && allow && !cold_resync)
                    || (outcome == b"cold_miss" && prefix == 0)),
            "private branch8 ancestor restore/cold selection",
        )?;
        for id in get(job, "prompt_ids")?.array()? {
            need(
                id.unsigned_integer()? < 248_320,
                "private branch8 future ID",
            )?;
        }
        let policy = get(job, "sampling")?;
        plain_keys(policy, &["temperature", "top_k", "top_p", "min_p"])?;
        need(
            real(policy, "temperature")? == 0.0
                && integer(policy, "top_k")? <= 248_320
                && real(policy, "top_p")? > 0.0
                && real(policy, "top_p")? <= 1.0
                && real(policy, "min_p")? >= 0.0
                && real(policy, "min_p")? <= 1.0,
            "private branch8 fresh greedy policy",
        )?;
        sampling(
            state,
            get(job, "sampling")?,
            integer(job, "seed")?,
            prefix as u32,
        )?;
        need(
            !get(job, "generation_requested")?.boolean()?
                || prefix < count
                || get(state, "stop_id")?.is_null(),
            "private branch8 restored terminal cannot generate",
        )?;
        for name in HC {
            let a = integer(get(old, "hc_exact_routes")?, name)?;
            let b = integer(get(state, "hc_exact_routes")?, name)?;
            need(
                if cold_resync { b >= a } else { b == a },
                "private branch8 control changed GPU execution cost",
            )?;
        }
        #[cfg(feature = "qsa-stage-v8")]
        {
            stage8_same(state, old)?;
            if stage8_metadata(state)? {
                let a = integer(old, "qsa_stage_v8_calls")?;
                let b = integer(state, "qsa_stage_v8_calls")?;
                need(
                    if cold_resync { b >= a } else { b == a },
                    "private staged-QSA zero-step control changed execution costs",
                )?;
            }
        }
        let epoch = integer(old, "lookup_history_epoch")?;
        need(
            epoch < u64::MAX
                && if cold_resync {
                    integer(state, "lookup_history_epoch")? > epoch
                } else {
                    integer(state, "lookup_history_epoch")? == epoch + 1
                },
            "private branch8 fresh lookup epoch",
        )?;
        for name in COST_FIELDS {
            let a = integer(old, name)?;
            let b = integer(state, name)?;
            let mut delta: u64 = 0;
            if name == "lookup_resets" {
                delta = u64::from(outcome == b"cold_miss");
            }
            if name == "lookup_truncations" {
                delta = u64::from(outcome == b"restored");
            }
            if name == "lookup_truncated_grams" && outcome == b"restored" {
                delta = grams(before).wrapping_sub(grams(prefix));
            }
            // Only the exact worker-side predecessor identifies what a cold reset clears. An HTTP observer may have missed a completed ancestor restore before cancellation, so its stale longer prefix cannot lower-bound that work. Resync still requires monotonic costs, a new reset/epoch, a fresh command/lease and an explicit zero-prefix cold miss above.
            if name == "lookup_cleared_grams" && outcome == b"cold_miss" && !cold_resync {
                delta = grams(before);
            }
            need(
                b >= a
                    && if cold_resync {
                        b - a >= delta
                    } else {
                        b - a == delta
                    },
                "private branch8 control lifetime-history cost mismatch",
            )?;
        }
        Ok(())
    }
}

#[cfg(feature = "branch8")]
pub use branch::{branch8_session, validate_branch8_control, validate_branch8_ready};

/// The `Verify8Reply` constructor arguments.
#[derive(Clone, Copy)]
pub struct Verify8Context<'a> {
    pub ready: &'a Json,
    pub previous: &'a Json,
    pub job: &'a Json,
    pub before: u32,
    pub policy: &'a Json,
    pub seed: u64,
    pub reset_resync: bool,
    pub branch8: bool,
}

/// `Verify8Reply`: the shared CPU-only gate for one V8 model command. It buffers at most 16 ordered round/token records; `events` is available only after `finish` admits the complete done.
pub struct Verify8Reply<'a> {
    ready: &'a Json,
    previous: &'a Json,
    job: &'a Json,
    sampling: &'a Json,
    seed: u64,
    base: u32,
    position: u32,
    inputs: u32,
    maximum: u32,
    events: Vec<Json>,
    ids: [u32; 8],
    routes: [u64; 5],
    count: usize,
    rounds: u32,
    pending: u32,
    round_count: u32,
    accepted: u32,
    evaluated_total: u32,
    proposed_total: u32,
    ordinary_total: u32,
    recovery_rounds: u32,
    verified_rounds: u32,
    dispatches: u64,
    round_wall: f64,
    batch_wall: f64,
    batch_gpu: f64,
    reset: bool,
    reset_resync: bool,
    stopped: bool,
    finished: bool,
    branch8: bool,
    branch_epoch: u64,
    branch_command: u64,
    branch_lease: u64,
    stage8_prepared: bool,
    stage8_enabled: bool,
    #[cfg(feature = "qsa-stage-v8")]
    stage8_before: u64,
    fast_math_prepared: bool,
    channel_enabled: bool,
    #[cfg(feature = "fast-math")]
    channel_before: u64,
}

impl fmt::Debug for Verify8Reply<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Generated IDs and events stay out of debug output.
        formatter
            .debug_struct("Verify8Reply")
            .field("base", &self.base)
            .field("position", &self.position)
            .field("inputs", &self.inputs)
            .field("maximum", &self.maximum)
            .field("events", &self.events.len())
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

const ROUND_FIELDS: &[&str] = &[
    "event",
    "round_id",
    "round_base",
    "committed_prefix",
    "proposed_tokens",
    "verified_tokens",
    "verification_max_tokens",
    "verification_head_dispatches",
    "accepted_verified_tokens",
    "ordinary_tokens",
    "useful_tokens",
    "round_wall_ms",
    "lookup_and_publication_ms",
    "gather_wall_ms",
    "verification_including_recovery_wall_ms",
    "recovery_batch_wall_ms",
    "fallback_wall_ms",
    "batch_wall_ms",
    "batch_gpu_ms",
    "hc_exact_round_routes",
    "wide_q8_exact_round_routes",
    "wide_q8_many_round_routes",
    "dispatches",
    "capture_executes",
    "useful_tokens_per_second",
    "accepted_tokens_per_second",
    "timings_qualified",
    "gpu_profile",
];

const TOKEN_FIELDS: &[&str] = &[
    "event",
    "id",
    "source_logit",
    "committed_prefix",
    "token_prefix",
    "round_id",
    "round_base",
    "round_committed_prefix",
    "verified",
    "batched_commit",
    "entropy_words",
    "gpu_profile",
];

const CLOCK_FIELDS: [&str; 10] = [
    "round_wall_ms",
    "lookup_and_publication_ms",
    "gather_wall_ms",
    "verification_including_recovery_wall_ms",
    "recovery_batch_wall_ms",
    "fallback_wall_ms",
    "batch_wall_ms",
    "batch_gpu_ms",
    "useful_tokens_per_second",
    "accepted_tokens_per_second",
];

fn popcount(n: u64) -> u64 {
    u64::from((n as u32).count_ones())
}

impl<'a> Verify8Reply<'a> {
    /// `Verify8Reply(ready, previous, job, before, policy, seed, reset_resync, branch8)`. Without the `branch8` feature, `branch8` is ignored.
    pub fn new(context: Verify8Context<'a>) -> Result<Self> {
        let Verify8Context {
            ready,
            previous,
            job,
            before,
            policy,
            seed,
            reset_resync,
            branch8,
        } = context;
        let branch8 = cfg!(feature = "branch8") && branch8;
        // Member initializers, in the C++ declaration order.
        let reduced = Self::previous_state(previous, branch8)?;
        let inputs = get(job, "tokens")?.array()?.len() as u32;
        let maximum = integer(job, "generate")? as u32;
        let reset = get(job, "reset")?.boolean()?;
        let (branch_epoch, branch_command, branch_lease) = if branch8 {
            (
                integer(ready, "broker_epoch")?,
                integer(previous, "command_id")?.wrapping_add(1),
                integer(previous, "lease_id")?,
            )
        } else {
            (0, 0, 0)
        };
        let mut reply = Self {
            ready,
            previous: reduced,
            job,
            sampling: policy,
            seed,
            base: before,
            position: before,
            inputs,
            maximum,
            events: Vec::with_capacity(16),
            ids: [0; 8],
            routes: [0; 5],
            count: 0,
            rounds: 0,
            pending: 0,
            round_count: 0,
            accepted: 0,
            evaluated_total: 0,
            proposed_total: 0,
            ordinary_total: 0,
            recovery_rounds: 0,
            verified_rounds: 0,
            dispatches: 0,
            round_wall: 0.0,
            batch_wall: 0.0,
            batch_gpu: 0.0,
            reset,
            reset_resync,
            stopped: false,
            finished: false,
            branch8,
            branch_epoch,
            branch_command,
            branch_lease,
            stage8_prepared: false,
            stage8_enabled: false,
            #[cfg(feature = "qsa-stage-v8")]
            stage8_before: 0,
            fast_math_prepared: false,
            channel_enabled: false,
            #[cfg(feature = "fast-math")]
            channel_before: 0,
        };
        reply.construct(before, policy, seed)?;
        Ok(reply)
    }

    fn previous_state(previous: &'a Json, branch8: bool) -> Result<&'a Json> {
        #[cfg(feature = "branch8")]
        if branch8 && !previous.is_null() {
            return branch8_session(previous);
        }
        let _ = branch8;
        Ok(previous)
    }

    /// The C++ constructor body.
    fn construct(&mut self, before: u32, policy: &Json, seed: u64) -> Result<()> {
        let job = self.job;
        #[cfg(feature = "fast-math")]
        {
            let fast = fast_metadata(self.ready)?;
            self.fast_math_prepared = fast.is_some();
            self.channel_enabled = fast.unwrap_or(false);
            if !self.previous.is_null() {
                fast_same(self.previous, self.ready)?;
            }
            if self.fast_math_prepared {
                self.channel_before = integer(self.previous, "qsa_channel_calls")?;
            }
        }
        #[cfg(feature = "qsa-stage-v8")]
        {
            self.stage8_prepared = stage8_metadata(self.ready)?;
            if !self.previous.is_null() {
                stage8_same(self.previous, self.ready)?;
            }
            if self.stage8_prepared {
                need(
                    self.branch8,
                    "private staged HTTP requires branch8 admission",
                )?;
                self.stage8_enabled = get(self.ready, "qsa_stage_v8_enabled")?.boolean()?;
                self.stage8_before = integer(self.previous, "qsa_stage_v8_calls")?;
            }
        }
        if cfg!(feature = "branch8") {
            need(
                !self.branch8
                    || (!self.reset
                        && !self.reset_resync
                        && !self.previous.is_null()
                        && self.branch_epoch != 0
                        && self.branch_command != 0
                        && self.branch_lease != 0),
                "private branch8 model work requires a separately admitted cold/restore control",
            )?;
        }
        plain_keys(
            job,
            &[
                "tokens",
                "generate",
                "reset",
                "stop_ids",
                "sampling",
                "seed",
                "prompt_lookup",
            ],
        )?;
        need(
            get(job, "tokens")?.array()?.len() <= 8
                && integer(job, "generate")? <= 8
                && (self.inputs != 0) != (self.maximum != 0)
                && before <= 65536,
            "private V8 one bounded input OR generation batch required",
        )?;
        plain_keys(policy, &["temperature", "top_k", "top_p", "min_p"])?;
        need(
            real(policy, "temperature")? == 0.0
                && integer(policy, "top_k")? <= 248_320
                && real(policy, "top_p")? > 0.0
                && real(policy, "top_p")? <= 1.0
                && real(policy, "min_p")? >= 0.0
                && real(policy, "min_p")? <= 1.0,
            "private V8 is greedy-only",
        )?;
        if self.reset {
            self.base = 0;
            self.position = 0;
            need(
                self.inputs != 0
                    && self.maximum == 0
                    && get(job, "sampling")?.dump()? == policy.dump()?
                    && integer(job, "seed")? == seed,
                "private V8 reset must install policy/seed with input",
            )?;
        } else {
            need(
                job.find("sampling")?.is_none()
                    && job.find("seed")?.is_none()
                    && !self.previous.is_null()
                    && integer(self.previous, "committed_prefix")? == u64::from(before)
                    && (self.inputs != 0
                        || get(self.previous, "continuation_allowed")?.boolean()?),
                "private V8 continuation policy/lineage",
            )?;
            sampling(self.previous, policy, seed, before)?;
        }
        need(
            self.base
                .wrapping_add(self.inputs)
                .wrapping_add(self.maximum)
                <= 65536
                && (self.inputs != 0 || self.base != 0),
            "private V8 context bound",
        )?;
        for id in get(job, "tokens")?.array()? {
            need(id.unsigned_integer()? < 248_320, "private V8 input ID")?;
        }
        need(
            if self.reset {
                get(job, "prompt_lookup")?.boolean()?
            } else {
                job.find("prompt_lookup")?.is_none()
            } && (!self.reset_resync || self.reset),
            "private V8 reset-only lookup/resync policy",
        )?;
        if self.maximum != 0 {
            let stops = get(job, "stop_ids")?.array()?;
            need(
                match stops.as_slice() {
                    [one] => one.unsigned_integer()? < 248_320,
                    [first, second] => matches!(
                        (first.unsigned_integer()?, second.unsigned_integer()?),
                        (248_046, 248_044) | (248_044, 248_046)
                    ),
                    _ => false,
                },
                "private V8 requires one terminal or the pinned terminal pair",
            )?;
        } else {
            need(
                job.find("stop_ids")?.is_none(),
                "private V8 prefill stop list",
            )?;
        }
        Ok(())
    }

    /// One ordered round or token record; buffers at most 16.
    pub fn event(&mut self, v: &Json) -> Result<()> {
        #[cfg(feature = "branch8")]
        if self.branch8 {
            branch::branch_identity(v)?;
            need(
                integer(v, "broker_epoch")? == self.branch_epoch
                    && integer(v, "command_id")? == self.branch_command
                    && integer(v, "lease_id")? == self.branch_lease,
                "private branch8 event epoch/command/lease mismatch",
            )?;
        }
        need(
            !self.finished && self.maximum != 0 && !self.stopped && self.events.len() < 16,
            "private V8 unsolicited event",
        )?;
        let kind = get(v, "event")?.string()?;
        if kind == b"generation_round" {
            self.round(v)?;
        } else {
            self.token(v, kind)?;
        }
        self.events.push(v.clone());
        Ok(())
    }

    fn round(&mut self, v: &Json) -> Result<()> {
        keys(
            v,
            ROUND_FIELDS,
            self.branch8,
            self.stage8_prepared,
            self.fast_math_prepared,
        )?;
        need(
            self.pending == 0
                && {
                    self.rounds = self.rounds.wrapping_add(1);
                    self.rounds <= 8
                }
                && integer(v, "round_id")? == u64::from(self.rounds)
                && integer(v, "round_base")? == u64::from(self.position),
            "private V8 round chronology",
        )?;
        let n = integer(v, "verified_tokens")?;
        let k = integer(v, "accepted_verified_tokens")?;
        let o = integer(v, "ordinary_tokens")?;
        let p = integer(v, "proposed_tokens")?;
        let count = integer(v, "useful_tokens")?;
        // `maximum_ - count_` is size_t; `65536 - position_` is unsigned int.
        let left = (self.maximum as u64).wrapping_sub(self.count as u64);
        let room = u64::from(65536u32.wrapping_sub(self.position));
        need(
            p <= 7
                && (n == 0 || (2..=7).contains(&n))
                && k <= n
                && n <= p
                && n <= left
                && n <= room
                && o <= 1
                && count == k.wrapping_add(o)
                && count != 0
                && count <= left
                && if n == 0 {
                    k == 0 && o == 1
                } else {
                    k >= 1 && o == u64::from(k < n)
                }
                && integer(v, "verification_max_tokens")? == 8
                && integer(v, "verification_head_dispatches")? == popcount(n)
                && integer(v, "committed_prefix")? == u64::from(self.position).wrapping_add(count),
            "private V8 round acceptance/correction/depth bound",
        )?;
        self.accepted = k as u32;
        self.round_count = count as u32;
        self.pending = self.round_count;
        self.verified_rounds = self.verified_rounds.wrapping_add(1);
        if n == 0 {
            self.verified_rounds = self.verified_rounds.wrapping_sub(1);
        }
        self.evaluated_total = self.evaluated_total.wrapping_add(n as u32);
        self.proposed_total = self.proposed_total.wrapping_add(p as u32);
        self.ordinary_total = self.ordinary_total.wrapping_add(o as u32);
        self.recovery_rounds = self.recovery_rounds.wrapping_add(u32::from(k > 0 && k < n));
        let stage_enabled = cfg!(feature = "qsa-stage-v8") && self.stage8_enabled;
        #[cfg(feature = "qsa-stage-v8")]
        if self.stage8_prepared {
            let expected = if self.stage8_enabled { 12 * (n + o) } else { 0 };
            need(
                integer(v, "qsa_stage_v8_round_calls")? == expected,
                "private staged-QSA exact round work",
            )?;
        }
        let channel_enabled = cfg!(feature = "fast-math") && self.channel_enabled;
        #[cfg(feature = "fast-math")]
        {
            if self.fast_math_prepared {
                let expected = if channel_enabled { 12 * (n + o) } else { 0 };
                need(
                    integer(v, "qsa_channel_round_calls")? == expected,
                    "private fast-math exact round work",
                )?;
            } else {
                need(
                    v.find("qsa_channel_round_calls")?.is_none(),
                    "unexpected fast-math round metadata",
                )?;
            }
        }
        let mut serial: u64 = 0;
        for t in 0..n as u32 {
            serial = serial.wrapping_add(ordinary(
                self.position.wrapping_add(t),
                stage_enabled,
                channel_enabled,
            )?);
        }
        let verified = if n != 0 {
            serial
                .wrapping_sub(72u64.wrapping_mul(n.wrapping_sub(1)))
                .wrapping_sub(253u64.wrapping_mul(n.wrapping_sub(popcount(n))))
        } else {
            0
        };
        let recovery_work = if k != 0 && k < n { 72 } else { 0 };
        let fallback = if o != 0 {
            ordinary(
                self.position.wrapping_add(k as u32),
                stage_enabled,
                channel_enabled,
            )?
        } else {
            0
        };
        let work = verified.wrapping_add(recovery_work).wrapping_add(fallback);
        need(
            integer(v, "dispatches")? == work
                && integer(v, "capture_executes")? == 0
                && !get(v, "timings_qualified")?.boolean()?
                && get(v, "gpu_profile")?.is_null(),
            "private V8 physical work/profile",
        )?;
        self.dispatches = self.dispatches.wrapping_add(work);
        let r = get(v, "hc_exact_round_routes")?;
        hc_shape(r)?;
        for (i, name) in HC.iter().enumerate() {
            let expected = if i == 0 {
                25u64.wrapping_mul(n).wrapping_add(97u64.wrapping_mul(o))
            } else if n & (1u64 << (i - 1)) != 0 {
                72
            } else {
                0
            };
            need(
                integer(r, name)? == expected,
                "private V8 round HC route coverage",
            )?;
            self.routes[i] = self.routes[i].wrapping_add(expected);
        }
        zero_routes(v, "wide_q8_exact_round_routes", 4)?;
        zero_routes(v, "wide_q8_many_round_routes", 6)?;
        for name in CLOCK_FIELDS {
            need(real(v, name)? >= 0.0, "private V8 negative clock")?;
        }
        let wall = real(v, "round_wall_ms")?;
        let verify = real(v, "verification_including_recovery_wall_ms")?;
        let recovery = real(v, "recovery_batch_wall_ms")?;
        close(
            wall,
            real(v, "lookup_and_publication_ms")?
                + real(v, "gather_wall_ms")?
                + verify
                + real(v, "fallback_wall_ms")?,
        )?;
        need(
            wall > 0.0
                && real(v, "batch_wall_ms")? > 0.0
                && real(v, "batch_gpu_ms")? > 0.0
                && real(v, "batch_wall_ms")? <= wall + 1e-6
                && if n != 0 {
                    verify > 0.0
                } else {
                    real(v, "gather_wall_ms")? == 0.0 && verify == 0.0 && recovery == 0.0
                }
                && if k != 0 && k < n {
                    recovery > 0.0
                } else {
                    recovery == 0.0
                }
                && if o != 0 {
                    real(v, "fallback_wall_ms")? > 0.0
                } else {
                    real(v, "fallback_wall_ms")? == 0.0
                }
                && recovery <= verify + 1e-6
                && real(v, "batch_gpu_ms")? <= real(v, "batch_wall_ms")? + 1e-6,
            "private V8 clock attribution",
        )?;
        close(
            real(v, "useful_tokens_per_second")?,
            if wall != 0.0 {
                count as f64 * 1000.0 / wall
            } else {
                0.0
            },
        )?;
        close(
            real(v, "accepted_tokens_per_second")?,
            if wall != 0.0 {
                k as f64 * 1000.0 / wall
            } else {
                0.0
            },
        )?;
        self.round_wall += wall;
        self.batch_wall += real(v, "batch_wall_ms")?;
        self.batch_gpu += real(v, "batch_gpu_ms")?;
        Ok(())
    }

    fn token(&mut self, v: &Json, kind: &[u8]) -> Result<()> {
        keys(v, TOKEN_FIELDS, self.branch8, false, false)?;
        let offset = self.round_count.wrapping_sub(self.pending);
        let id = integer(v, "id")?;
        let round_end = u64::from(self.position.wrapping_add(self.round_count));
        need(
            kind == b"token"
                && self.pending != 0
                && (self.count as u64) < u64::from(self.maximum)
                && id < 248_320
                && integer(v, "round_id")? == u64::from(self.rounds)
                && integer(v, "round_base")? == u64::from(self.position)
                && integer(v, "committed_prefix")? == round_end
                && integer(v, "round_committed_prefix")? == round_end
                && integer(v, "token_prefix")?
                    == u64::from(self.position.wrapping_add(offset).wrapping_add(1))
                && get(v, "verified")?.boolean()? == (offset < self.accepted)
                && get(v, "batched_commit")?.boolean()? == (self.accepted >= 2)
                && integer(v, "entropy_words")? == 0
                && get(v, "gpu_profile")?.is_null(),
            "private V8 token ancestry/source",
        )?;
        let score = real(v, "source_logit")?;
        if self.count == 0 {
            let top = get(self.previous, "final_logit_top8")?.array()?;
            need(
                top.len() == 8 && id == integer(&top[0], "id")? && score == real(&top[0], "logit")?,
                "private V8 first greedy ID/source differs from retained logits",
            )?;
        }
        self.ids[self.count] = id as u32;
        self.count += 1;
        self.pending -= 1;
        let stops = get(self.job, "stop_ids")?.array()?;
        self.stopped = false;
        for stop in stops {
            if id == stop.unsigned_integer()? {
                self.stopped = true;
                break;
            }
        }
        need(
            !self.stopped || self.pending == 0,
            "private V8 token after first terminal",
        )?;
        if self.pending == 0 {
            self.position = self.position.wrapping_add(self.round_count);
        }
        Ok(())
    }

    /// Admits the complete done; must succeed before `events` is available.
    pub fn finish(&mut self, v: &Json) -> Result<()> {
        need(
            !self.finished && self.pending == 0 && str_is(get(v, "event")?, "done")?,
            "private V8 incomplete round/done",
        )?;
        fixed(v, self.branch8)?;
        #[cfg(feature = "fast-math")]
        {
            fast_same(v, self.ready)?;
            if self.fast_math_prepared {
                let work = self
                    .inputs
                    .wrapping_add(self.evaluated_total)
                    .wrapping_add(self.ordinary_total);
                let expected = if self.channel_enabled {
                    12u64.wrapping_mul(u64::from(work))
                } else {
                    0
                };
                need(
                    expected <= u64::MAX - self.channel_before
                        && integer(v, "qsa_channel_request_calls")? == expected
                        && integer(v, "qsa_channel_calls")? == self.channel_before + expected,
                    "private fast-math exact request/lifetime cost",
                )?;
            }
        }
        #[cfg(feature = "qsa-stage-v8")]
        {
            stage8_same(v, self.ready)?;
            if self.stage8_prepared {
                let work = self
                    .inputs
                    .wrapping_add(self.evaluated_total)
                    .wrapping_add(self.ordinary_total);
                let expected = if self.stage8_enabled {
                    12u64.wrapping_mul(u64::from(work))
                } else {
                    0
                };
                need(
                    expected <= u64::MAX - self.stage8_before
                        && integer(v, "qsa_stage_v8_request_calls")? == expected
                        && integer(v, "qsa_stage_v8_calls")? == self.stage8_before + expected,
                    "private staged-QSA exact request/lifetime cost",
                )?;
            }
        }
        #[cfg(feature = "branch8")]
        if self.branch8 {
            let session = branch8_session(v)?;
            #[cfg(feature = "qsa-stage-v8")]
            stage8_same(session, self.ready)?;
            #[cfg(feature = "fast-math")]
            fast_same(session, self.ready)?;
            let _ = session;
            need(
                integer(v, "broker_epoch")? == self.branch_epoch
                    && integer(v, "command_id")? == self.branch_command
                    && integer(v, "lease_id")? == self.branch_lease,
                "private branch8 completion epoch/command/lease mismatch",
            )?;
            for name in STORAGE_FIELDS {
                need(
                    integer(v, name)? == integer(self.ready, name)?,
                    "private branch8 persistent storage drift",
                )?;
            }
            need(
                integer(v, "checkpoint_retained")? <= integer(self.ready, "checkpoint_slots")?
                    && str_is(get(v, "checkpoint_reuse")?, "off")?
                    && integer(v, "checkpoint_restored_tokens")? == 0
                    && integer(v, "replayed_input_tokens")? == u64::from(self.inputs)
                    && real(v, "checkpoint_capture_wall_ms")? == 0.0
                    && real(v, "checkpoint_restore_wall_ms")? == 0.0,
                "private branch8 model command cannot hide control work",
            )?;
        }
        if self.inputs != 0 {
            self.position = self.base.wrapping_add(self.inputs);
        }
        need(
            integer(v, "committed_prefix")? == u64::from(self.position)
                && self.count == get(v, "generated_ids")?.array()?.len()
                && (self.inputs != 0
                    || (self.count != 0
                        && (self.stopped || self.count as u64 == u64::from(self.maximum))))
                && get(v, "known_token_prefill")?.boolean()?,
            "private V8 terminal prefix/count",
        )?;
        for i in 0..self.count {
            need(
                get(v, "generated_ids")?.array()?[i].unsigned_integer()? == u64::from(self.ids[i]),
                "private V8 token/done disagreement",
            )?;
        }
        let status = if self.inputs != 0 {
            "prefilled"
        } else if self.stopped {
            "stopped"
        } else {
            "length"
        };
        need(
            str_is(get(v, "status")?, status)?
                && if self.inputs != 0 {
                    get(v, "finish_reason")?.is_null()
                } else {
                    str_is(
                        get(v, "finish_reason")?,
                        if self.stopped { "stop" } else { "length" },
                    )?
                }
                && get(v, "continuation_allowed")?.boolean()? != self.stopped,
            "private V8 completion/continuation",
        )?;
        if self.stopped {
            need(
                integer(v, "stop_id")? == u64::from(self.ids[self.count - 1]),
                "private V8 terminal ID",
            )?;
        } else {
            need(get(v, "stop_id")?.is_null(), "private V8 unexpected stop")?;
        }
        sampling(v, self.sampling, self.seed, self.position)?;
        for name in GRAPH_FIELDS {
            need(
                integer(v, name)? == integer(self.ready, name)?,
                "private V8 frozen graph ledger drift",
            )?;
        }
        let hash = get(v, "hc_exact_final_logit_sha256")?.string()?;
        need(lowercase_hex64(hash), "private V8 final logit identity")?;
        hc_shape(get(v, "hc_exact_routes")?)?;
        hc_shape(get(v, "hc_exact_request_routes")?)?;
        if self.inputs != 0 {
            self.routes[0] = u64::from(24u32.wrapping_mul(self.inputs).wrapping_add(1));
            for i in 1..HC.len() {
                self.routes[i] = if self.inputs & (1u32 << (i - 1)) != 0 {
                    72
                } else {
                    0
                };
            }
        }
        let previous = if self.previous.is_null() {
            self.ready
        } else {
            self.previous
        };
        for (i, name) in HC.iter().enumerate() {
            let old = integer(get(previous, "hc_exact_routes")?, name)?;
            let now = integer(get(v, "hc_exact_routes")?, name)?;
            need(
                now >= old
                    && if self.reset_resync {
                        now - old >= self.routes[i]
                    } else {
                        now - old == self.routes[i]
                    }
                    && integer(get(v, "hc_exact_request_routes")?, name)? == self.routes[i],
                "private V8 completed HC routes",
            )?;
        }
        zero_routes(v, "wide_q8_exact_request_routes", 4)?;
        zero_routes(v, "wide_q8_many_request_routes", 6)?;
        need(
            integer(v, "lookup_rounds")? == u64::from(self.rounds)
                && integer(v, "lookup_verified_rounds")? == u64::from(self.verified_rounds)
                && integer(v, "lookup_recovery_rounds")? == u64::from(self.recovery_rounds)
                && integer(v, "lookup_proposed_tokens")? == u64::from(self.proposed_total)
                && integer(v, "lookup_accepted_tokens")?
                    == (self.count as u64).wrapping_sub(u64::from(self.ordinary_total))
                && integer(v, "lookup_ordinary_tokens")? == u64::from(self.ordinary_total),
            "private V8 round summary",
        )?;
        let epoch = if self.previous.is_null() {
            1
        } else {
            integer(self.previous, "lookup_history_epoch")?
        };
        need(
            epoch < u64::MAX
                && if self.reset_resync {
                    integer(v, "lookup_history_epoch")? > epoch
                } else {
                    integer(v, "lookup_history_epoch")? == epoch + u64::from(self.reset)
                },
            "private V8 history epoch",
        )?;
        for (all, delta) in [
            ("lookup_appended_tokens", "lookup_request_appended_tokens"),
            ("lookup_indexed_grams", "lookup_request_indexed_grams"),
            ("lookup_queries", "lookup_request_queries"),
            ("lookup_resets", "lookup_request_resets"),
        ] {
            let old = if self.previous.is_null() {
                0
            } else {
                integer(self.previous, all)?
            };
            let now = integer(v, all)?;
            need(
                now >= old
                    && if self.reset_resync {
                        now - old >= integer(v, delta)?
                    } else {
                        now - old == integer(v, delta)?
                    },
                "private V8 history cost delta",
            )?;
        }
        #[cfg(feature = "branch8")]
        if self.branch8 {
            for name in [
                "lookup_truncated_grams",
                "lookup_cleared_grams",
                "lookup_truncations",
            ] {
                need(
                    integer(v, name)? == integer(self.previous, name)?,
                    "private branch8 model work altered ancestry-maintenance costs",
                )?;
            }
            for name in [
                "lookup_candidates_examined",
                "lookup_token_comparisons",
                "lookup_work_limited_queries",
            ] {
                need(
                    integer(v, name)? >= integer(self.previous, name)?,
                    "private branch8 lookup execution costs rewound",
                )?;
            }
        }
        let grams = |n: u32| if n < 16 { 0u32 } else { n - 15 };
        need(
            integer(v, "lookup_request_appended_tokens")?
                == u64::from(self.inputs).wrapping_add(self.count as u64)
                && integer(v, "lookup_request_indexed_grams")?
                    == u64::from(grams(self.position).wrapping_sub(grams(self.base)))
                && integer(v, "lookup_request_queries")? == u64::from(self.rounds)
                && integer(v, "lookup_request_resets")? == u64::from(self.reset),
            "private V8 committed history work",
        )?;
        close(real(v, "lookup_round_wall_ms")?, self.round_wall)?;
        close(real(v, "generation_batch_wall_ms")?, self.batch_wall)?;
        close(real(v, "generation_batch_gpu_ms")?, self.batch_gpu)?;
        need(
            real(v, "prefill_wall_ms")? >= 0.0
                && real(v, "request_wall_ms")? > 0.0
                && real(v, "prefill_wall_ms")? + self.round_wall
                    <= real(v, "request_wall_ms")? + 1e-6
                && get(v, "gpu_profile")?.is_null(),
            "private V8 request clock/profile",
        )?;
        let top = get(v, "final_logit_top8")?.array()?;
        need(top.len() == 8, "private V8 final top8")?;
        let mut last = f64::INFINITY;
        let mut last_id = 0;
        for (i, entry) in top.iter().enumerate() {
            plain_keys(entry, &["id", "logit"])?;
            let id = integer(entry, "id")?;
            let score = real(entry, "logit")?;
            need(
                id < 248_320 && (score < last || (i > 0 && score == last && id > last_id)),
                "private V8 sorted unique final logits",
            )?;
            for earlier in &top[..i] {
                need(
                    integer(earlier, "id")? != id,
                    "private V8 duplicate final ID",
                )?;
            }
            last = score;
            last_id = id;
        }
        self.finished = true;
        Ok(())
    }

    /// The admitted records, in order.
    pub fn events(&self) -> Result<&[Json]> {
        need(self.finished, "private V8 events not admitted")?;
        Ok(&self.events)
    }
}
