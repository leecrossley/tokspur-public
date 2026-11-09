//! Native MTP carrier wire protocol and reply admission, ported from `native/src/mtp_wire.cpp`.
//!
//! Provides validation and state tracking for multi-turn prefix caching,
//! sampled positive-temperature generation, and mid-stream cancellation.

use std::fmt;

use tokspur_json::{Json, JsonError, Kind, from_chars_f64};

/// The error type raised by MTP wire admission.
#[derive(Clone, PartialEq, Eq)]
pub enum MtpWireError {
    MissingField(String),
    Refused(&'static str),
    Json(JsonError),
}

impl fmt::Debug for MtpWireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(name) => formatter
                .debug_struct("MissingField")
                .field("name_bytes", &name.len())
                .finish(),
            Self::Refused(reason) => formatter.debug_tuple("Refused").field(reason).finish(),
            Self::Json(error) => formatter.debug_tuple("Json").field(error).finish(),
        }
    }
}

impl fmt::Display for MtpWireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(name) => write!(formatter, "missing MTP wire field: {name}"),
            Self::Refused(reason) => formatter.write_str(reason),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for MtpWireError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::MissingField(_) | Self::Refused(_) => None,
        }
    }
}

impl From<JsonError> for MtpWireError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

pub type Result<T> = std::result::Result<T, MtpWireError>;

fn need(ok: bool, message: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(MtpWireError::Refused(message))
    }
}

fn get<'a>(v: &'a Json, k: &str) -> Result<&'a Json> {
    v.find(k)?
        .ok_or_else(|| MtpWireError::MissingField(k.to_owned()))
}

fn integer(v: &Json, k: &str) -> Result<u64> {
    Ok(get(v, k)?.unsigned_integer()?)
}

fn real_value(value: &Json) -> Result<f64> {
    let lexeme = value.number_lexeme()?;
    from_chars_f64(lexeme)
        .filter(|n| n.is_finite())
        .ok_or(MtpWireError::Refused("nonfinite MTP wire number"))
}

fn real(v: &Json, k: &str) -> Result<f64> {
    real_value(get(v, k)?)
}

fn keys(value: &Json, allowed: &[&str]) -> Result<()> {
    for (name, _) in value.object()? {
        need(
            allowed
                .iter()
                .any(|candidate| candidate.as_bytes() == name.as_slice()),
            "unknown MTP event/command field",
        )?;
    }
    Ok(())
}

fn delta(previous: &Json, done: &Json, name: &str, expected: u64, resync: bool) -> Result<()> {
    let before = integer(previous, name)?;
    let after = integer(done, name)?;
    need(
        after >= before
            && (if resync {
                after - before >= expected
            } else {
                after - before == expected
            }),
        "MTP lifetime counter differs",
    )
}

pub const DEFAULT_CONTROLLER_POLICY_DIGEST: [u8; 32] = [
    0xa9, 0x72, 0x87, 0x4d, 0x5a, 0xdb, 0x82, 0xb0, 0x2e, 0xea, 0x0d, 0x7d, 0xa5, 0xd5, 0x24, 0x2a,
    0x86, 0xe4, 0xaa, 0xa3, 0xff, 0x16, 0x6d, 0xc3, 0xab, 0x58, 0x19, 0xdb, 0x7f, 0x2c, 0xbf, 0x5a,
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredObservation {
    pub generation: u64,
    pub actions: u64,
    pub last_decode_ms: f64,
    pub acceptance: [f64; 7],
    pub cost_multiplier: f64,
    pub rate: f64,
    pub history_size: u32,
    pub since_deep_probe: u32,
    pub converged: bool,
    pub calibrated: bool,
}

impl MeasuredObservation {
    pub fn parse(v: &Json) -> Result<Self> {
        keys(
            v,
            &[
                "generation",
                "actions",
                "last_decode_ms",
                "acceptance",
                "cost_multiplier",
                "rate",
                "history_size",
                "since_deep_probe",
                "converged",
                "calibrated",
            ],
        )?;
        let mut acceptance = [0.0; 7];
        let a = get(v, "acceptance")?.array()?;
        need(a.len() == 7, "invalid measured acceptance shape")?;
        for (out, value) in acceptance.iter_mut().zip(a) {
            *out = real_value(value)?;
            need(
                (0.0..=1.0).contains(out),
                "invalid measured acceptance estimate",
            )?;
        }
        let generation = integer(v, "generation")?;
        let actions = integer(v, "actions")?;
        let last_decode_ms = real(v, "last_decode_ms")?;
        need(
            actions <= 32768
                && last_decode_ms >= 0.0
                && ((actions == 0) == (last_decode_ms == 0.0)),
            "incoherent measured action observation",
        )?;
        let cost_multiplier = real(v, "cost_multiplier")?;
        let rate = real(v, "rate")?;
        let history_size = integer(v, "history_size")?;
        let since_deep_probe = integer(v, "since_deep_probe")?;
        need(
            generation != 0
                && cost_multiplier > 0.0
                && rate >= 0.0
                && (1..=32).contains(&history_size)
                && since_deep_probe <= 16,
            "measured observation bounds exceeded",
        )?;
        Ok(Self {
            generation,
            actions,
            last_decode_ms,
            acceptance,
            cost_multiplier,
            rate,
            history_size: history_size as u32,
            since_deep_probe: since_deep_probe as u32,
            converged: get(v, "converged")?.boolean()?,
            calibrated: get(v, "calibrated")?.boolean()?,
        })
    }

    pub fn json(&self) -> Result<Json> {
        let number = |v| -> Result<Json> {
            let text = tokspur_json::to_chars_shortest(v)
                .ok_or(MtpWireError::Refused("invalid measured number"))?;
            Ok(tokspur_json::parse_json(text.as_bytes())?)
        };
        let acceptance: Result<Vec<Json>> = self.acceptance.iter().map(|&v| number(v)).collect();
        Ok(Json::new_object([
            ("generation", Json::integer(self.generation)),
            ("actions", Json::integer(self.actions)),
            ("last_decode_ms", number(self.last_decode_ms)?),
            ("acceptance", Json::Array(acceptance?)),
            ("cost_multiplier", number(self.cost_multiplier)?),
            ("rate", number(self.rate)?),
            ("history_size", Json::integer(u64::from(self.history_size))),
            (
                "since_deep_probe",
                Json::integer(u64::from(self.since_deep_probe)),
            ),
            ("converged", Json::Bool(self.converged)),
            ("calibrated", Json::Bool(self.calibrated)),
        ]))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageAControllerState {
    pub ewma_alpha1: f64,
    pub current_mode: u8,
    pub consecutive_ordinary_steps: u8,
    pub controller_policy_digest: [u8; 32],
    pub measured: Option<MeasuredObservation>,
}

impl Default for StageAControllerState {
    fn default() -> Self {
        Self {
            ewma_alpha1: 0.45,
            current_mode: 0,
            consecutive_ordinary_steps: 0,
            controller_policy_digest: DEFAULT_CONTROLLER_POLICY_DIGEST,
            measured: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MtpGenerationCommand {
    pub id: u64,
    pub remaining: u32,
    pub begin: bool,
    pub abort: bool,
    pub committed: u32,
    pub emitted: u32,
    pub pending: u32,
    pub remaining_compute: u32,
    pub total_budget: u32,
    pub stage_a_controller: Option<StageAControllerState>,
    pub v2: bool,
}

pub fn mtp_generation_command(value: &Json) -> Result<MtpGenerationCommand> {
    need(
        value.kind() == Kind::Object,
        "MTP generation command must be an object",
    )?;
    let mut out = MtpGenerationCommand::default();
    let id_field = get(value, "id")?;
    let id = id_field.unsigned_integer()?;
    need(id != 0, "MTP generation requires a nonzero ID")?;
    out.id = id;
    if let Some(abort_val) = value.find("abort")? {
        need(
            abort_val.boolean()? && value.object()?.len() == 2,
            "invalid MTP generation abort",
        )?;
        out.abort = true;
    } else {
        let begin_val = get(value, "begin")?;
        let remaining_val = get(value, "remaining")?;
        out.begin = begin_val.boolean()?;
        let n = remaining_val.unsigned_integer()?;
        need(
            (1..=32768).contains(&n),
            "MTP generation output budget outside 1..32768",
        )?;
        out.remaining = n as u32;
        out.total_budget = out.remaining;
        out.remaining_compute = out.remaining;
        if value.object()?.len() == 3 {
            // v1 packet
        } else {
            let committed = get(value, "committed")?.unsigned_integer()?;
            let emitted = get(value, "emitted")?.unsigned_integer()?;
            let pending = get(value, "pending")?.unsigned_integer()?;
            keys(
                value,
                &[
                    "id",
                    "begin",
                    "remaining",
                    "committed",
                    "emitted",
                    "pending",
                    "remaining_compute",
                    "total_budget",
                    "stage_a_controller",
                ],
            )?;
            out.v2 = true;
            out.committed = committed as u32;
            out.emitted = emitted as u32;
            out.pending = pending as u32;
            if let Some(rc) = value.find("remaining_compute")? {
                out.remaining_compute = rc.unsigned_integer()? as u32;
            }
            if let Some(tb) = value.find("total_budget")? {
                out.total_budget = tb.unsigned_integer()? as u32;
            }
            if let Some(sc) = value.find("stage_a_controller")? {
                need(
                    sc.kind() == Kind::Object,
                    "stage_a_controller must be an object",
                )?;
                let mut ctrl = StageAControllerState::default();
                if let Some(ea) = sc.find("ewma_alpha1")? {
                    ctrl.ewma_alpha1 = real_value(ea)?;
                }
                if let Some(cm) = sc.find("current_mode")? {
                    ctrl.current_mode = cm.unsigned_integer()? as u8;
                }
                if let Some(co) = sc.find("consecutive_ordinary_steps")? {
                    ctrl.consecutive_ordinary_steps = co.unsigned_integer()? as u8;
                }
                if let Some(cpd) = sc.find("controller_policy_digest")? {
                    need(
                        cpd.kind() == Kind::Array,
                        "controller_policy_digest must be an array",
                    )?;
                    let arr = cpd.array()?;
                    need(
                        arr.len() == 32,
                        "controller_policy_digest must be 32 elements",
                    )?;
                    for (i, elem) in arr.iter().enumerate() {
                        let byte = elem.unsigned_integer()?;
                        need(byte <= 255, "controller policy digest byte outside range")?;
                        ctrl.controller_policy_digest[i] = byte as u8;
                    }
                }
                if let Some(m) = sc.find("measured")? {
                    ctrl.measured = Some(MeasuredObservation::parse(m)?);
                }
                out.stage_a_controller = Some(ctrl);
            }
        }
    }
    Ok(out)
}

pub fn mtp_draft_attention(value: &Json) -> Result<&'static str> {
    let field = value.find("native_mtp_draft_attention")?;
    match field {
        Some(f) => {
            let s = f.string()?;
            match s {
                b"serial" => Ok("serial"),
                b"tiled128" => Ok("tiled128"),
                b"tiled256" => Ok("tiled256"),
                b"split16" => Ok("split16"),
                _ => Err(MtpWireError::Refused("unknown MTP draft-attention mode")),
            }
        }
        None => Ok("serial"),
    }
}

pub fn mtp_depth(value: &Json) -> Result<u32> {
    let depth = integer(value, "native_mtp_depth")?;
    need(
        depth == 1 || depth == 2 || depth == 3 || depth == 7,
        "unsupported native MTP depth",
    )?;
    Ok(depth as u32)
}

pub fn mtp_output_packet(value: &Json) -> Result<u32> {
    let cap = value.find("native_mtp_output_packet")?;
    let schema = value.find("native_mtp_output_packet_schema")?;
    match (cap, schema) {
        (Some(c), Some(s)) => {
            let limit = c.unsigned_integer()?;
            need(
                s.unsigned_integer()? == 1 && (limit == 8 || limit == 64),
                "MTP output-packet metadata bound/schema",
            )?;
            Ok(limit as u32)
        }
        (None, None) => Ok(8),
        _ => Err(MtpWireError::Refused(
            "MTP output-packet metadata is incomplete",
        )),
    }
}

const PRIMING_FIELDS: [&str; 3] = [
    "native_mtp_priming_batch_capacity",
    "native_mtp_priming_batch_bytes",
    "native_mtp_priming_batch_submits",
];

pub fn mtp_input_packet(value: &Json) -> Result<u32> {
    let cap = value.find("native_mtp_input_packet_max")?;
    let schema = value.find("native_mtp_input_packet_schema")?;
    let mut priming_count = 0;
    for (name, _) in value.object()? {
        if name.starts_with(b"native_mtp_input_packet") {
            need(
                name == b"native_mtp_input_packet_max" || name == b"native_mtp_input_packet_schema",
                "unknown MTP input-packet metadata",
            )?;
        }
        if name.starts_with(b"native_mtp_priming_batch") {
            need(
                PRIMING_FIELDS.iter().any(|field| name == field.as_bytes()),
                "unknown MTP priming-batch metadata",
            )?;
            priming_count += 1;
        }
    }
    match (cap, schema) {
        (Some(c), Some(s)) => {
            let limit = c.unsigned_integer()?;
            need(
                s.unsigned_integer()? == 1 && matches!(limit, 512 | 1024 | 2048 | 4096 | 8192),
                "MTP input-packet metadata bound/schema",
            )?;
            let depth = mtp_depth(value)?;
            let attention = mtp_draft_attention(value)?;
            need(
                mtp_request_carry(value)?
                    && ((matches!(depth, 2 | 3) && attention == "tiled128")
                        || (depth == 7 && attention == "split16")),
                "MTP input512 requires fixed depth-two/three tiled128 or depth-seven split16 scoped carry",
            )?;
            let dense = get(value, "native_prefill_dense")?.string()?;
            need(
                matches!(dense, b"f16" | b"f32" | b"nax"),
                "MTP input512 requires explicit dense precision metadata",
            )?;
            let grouped_nax = value
                .find("native_prefill_grouped")?
                .map(Json::string)
                .transpose()?
                == Some(b"nax".as_slice());
            let expected_capacity = if depth == 7 || dense == b"nax" || grouped_nax {
                limit.min(2048)
            } else {
                limit.min(512)
            };
            let maximum_bytes = if expected_capacity >= 1024 || dense == b"nax" {
                160u64 << 20
            } else {
                32u64 << 20
            };
            need(
                priming_count == PRIMING_FIELDS.len(),
                "MTP priming-batch capacity/workspace bound",
            )?;
            let bytes = integer(value, PRIMING_FIELDS[1])?;
            need(
                integer(value, PRIMING_FIELDS[0])? == expected_capacity
                    && bytes > 0
                    && bytes <= maximum_bytes,
                "MTP priming-batch capacity/workspace bound",
            )?;
            integer(value, PRIMING_FIELDS[2])?;
            Ok(limit as u32)
        }
        (None, None) => {
            need(
                priming_count == 0,
                "MTP batch priming requires explicit input512 metadata",
            )?;
            Ok(8)
        }
        _ => Err(MtpWireError::Refused(
            "MTP input-packet metadata is incomplete",
        )),
    }
}

fn validate_runtime_policy_identity(ready: &Json, done: &Json) -> Result<()> {
    for name in [
        "native_controller_policy_digest",
        "native_prepared_validation",
        "native_measured_controller",
        "native_measured_timing_schema",
        "native_dflash2_draft_policy",
        "native_target_m8_split_digest",
        "native_verify_narrow_rows",
        "native_pack_shared_mapping",
    ] {
        need(
            ready.find(name)? == done.find(name)?,
            "controller or validation policy differs from startup negotiation",
        )?;
    }
    Ok(())
}

fn validate_prefill_identity(ready: &Json, observed: &Json) -> Result<()> {
    let packet = mtp_input_packet(ready)?;
    need(
        mtp_input_packet(observed)? == packet,
        "MTP input-packet mode differs",
    )?;
    if packet >= 512 {
        for name in [PRIMING_FIELDS[0], PRIMING_FIELDS[1], "native_prefill_dense"] {
            need(
                get(ready, name)? == get(observed, name)?,
                "MTP immutable prefill preparation drift",
            )?;
        }
        for name in [
            "native_prefill_grouped",
            "native_prefill_grouped_kernel",
            "native_prefill_dense_kernel",
        ] {
            need(
                ready.find(name)? == observed.find(name)?,
                "MTP immutable grouped-prefill preparation drift",
            )?;
        }
    }
    Ok(())
}

fn validate_measured_timing_packet(
    ready: &Json,
    previous: &Json,
    done: &Json,
    begin: bool,
    generation: u64,
    actions: u32,
) -> Result<()> {
    let measured = ready.find("native_measured_controller")?;
    let packet = done.find("native_measured_decode_observations")?;
    let Some(measured) = measured else {
        return need(
            packet.is_none() && done.find("native_measured_timing_schema")?.is_none(),
            "unsolicited measured timing packet",
        );
    };
    need(
        measured.boolean()?
            && integer(ready, "native_measured_timing_schema")? == 1
            && integer(done, "native_measured_timing_schema")? == 1,
        "measured timing negotiation differs",
    )?;
    let packet = packet.ok_or(MtpWireError::Refused("missing measured timing packet"))?;
    keys(packet, &["first_action", "milliseconds"])?;
    let first = integer(packet, "first_action")?;
    let times = get(packet, "milliseconds")?.array()?;
    need(
        actions <= 64 && times.len() == actions as usize,
        "measured timing packet omitted or duplicated an action",
    )?;
    if actions == 0 {
        return need(first == 0, "empty measured timing packet has an action");
    }
    let state = |value: &Json| {
        MeasuredObservation::parse(get(
            get(get(value, "native_mtp_generation")?, "stage_a_controller")?,
            "measured",
        )?)
    };
    let before = if begin {
        0
    } else {
        let old = state(previous)?;
        need(
            old.generation == generation,
            "measured timing previous generation differs",
        )?;
        old.actions
    };
    let current = state(done)?;
    need(
        first == before + 1
            && current.generation == generation
            && current.actions == before + u64::from(actions),
        "measured timing action frontier differs",
    )?;
    let mut last = 0.0;
    for time in times {
        last = real_value(time)?;
        need(last > 0.0, "invalid measured cost or probability")?;
    }
    need(
        last == current.last_decode_ms,
        "measured timing final observation differs",
    )
}

pub fn mtp_request_carry(value: &Json) -> Result<bool> {
    let scope = value.find("native_mtp_carry_scope")?;
    let generation = value.find("native_mtp_generation")?;
    match (scope, generation) {
        (Some(sc), Some(generation_obj)) => {
            let sc_str = sc.string()?;
            need(
                sc_str == b"http-generation-v1" || sc_str == b"http-generation-v2",
                "unknown MTP carried-terminal scope",
            )?;
            if sc_str == b"http-generation-v1" {
                keys(generation_obj, &["id", "active", "remaining", "pending"])?;
            } else {
                keys(
                    generation_obj,
                    &[
                        "id",
                        "active",
                        "remaining",
                        "pending",
                        "committed",
                        "emitted",
                        "overflow",
                        "remaining_compute",
                        "total_budget",
                        "continuation_allowed",
                        "stop_queued",
                        "stop_drained",
                        "stage_a_controller",
                    ],
                )?;
            }
            let active = get(generation_obj, "active")?.boolean()?;
            let pending = get(generation_obj, "pending")?.boolean()?;
            let remaining = integer(generation_obj, "remaining")?;
            need(
                if active {
                    integer(generation_obj, "id")? != 0 && remaining != 0 && remaining <= 32768
                } else {
                    remaining == 0 && !pending
                },
                "invalid MTP generation lifetime metadata",
            )?;
            Ok(true)
        }
        (None, None) => Ok(false),
        _ => Err(MtpWireError::Refused(
            "MTP generation metadata presence differs",
        )),
    }
}

pub fn mtp_final_input_logits(value: &Json) -> Result<bool> {
    let policy = value.find("native_prefill_output_policy")?;
    if let Some(p) = policy {
        need(
            mtp_input_packet(value)? >= 512 && p.string()? == b"final-input-v1",
            "MTP final-input policy differs",
        )?;
        Ok(true)
    } else {
        Ok(false)
    }
}

pub fn mtp_hc_rows(value: &Json) -> Result<bool> {
    let mut count = 0;
    for (name, _) in value.object()? {
        if name.starts_with(b"native_prefill_hc_rows") {
            count += 1;
        }
    }
    Ok(count > 0)
}

pub fn mtp_router_window(value: &Json) -> Result<bool> {
    let mut count = 0;
    for (name, _) in value.object()? {
        if name.starts_with(b"native_prefill_router_window") {
            count += 1;
        }
    }
    Ok(count > 0)
}

pub fn mtp_hc_down_window(value: &Json) -> Result<bool> {
    let mut count = 0;
    for (name, _) in value.object()? {
        if name.starts_with(b"native_prefill_hc_down_window") {
            count += 1;
        }
    }
    Ok(count > 0)
}

pub fn mtp_qsa_sparse(value: &Json) -> Result<bool> {
    let mut count = 0;
    for (name, _) in value.object()? {
        if name.starts_with(b"native_qsa_sparse_") {
            count += 1;
        }
    }
    Ok(count > 0)
}

pub fn mtp_gdn_ported(value: &Json) -> Result<bool> {
    let mut count = 0;
    for (name, _) in value.object()? {
        if name.starts_with(b"native_gdn_ported_") {
            count += 1;
        }
    }
    Ok(count > 0)
}

pub fn mtp_hc_down_f32(value: &Json) -> Result<bool> {
    let mut count = 0;
    for (name, _) in value.object()? {
        if name.starts_with(b"native_prefill_hc_down_f32_") {
            count += 1;
        }
    }
    Ok(count > 0)
}

pub fn mtp_gdn_old_order(value: &Json) -> Result<bool> {
    let mut count = 0;
    for (name, _) in value.object()? {
        if name.starts_with(b"native_gdn_old_order_") {
            count += 1;
        }
    }
    Ok(count > 0)
}

pub fn validate_mtp_ready(
    ready: &Json,
    expected_draft_attention: &str,
    expected_request_carry: bool,
    expected_depth: u32,
) -> Result<()> {
    need(
        get(ready, "event")?.string()? == b"ready",
        "owned worker readiness/execution identity mismatch",
    )?;
    need(
        integer(ready, "maximum_generated_tokens")? == 4096,
        "maximum generated tokens mismatch",
    )?;
    need(
        get(ready, "stack_execution")?.string()? == b"direct",
        "MTP requires direct stack execution",
    )?;
    need(
        mtp_depth(ready)? == expected_depth,
        "MTP depth startup selection differs",
    )?;
    need(
        mtp_draft_attention(ready)? == expected_draft_attention,
        "MTP draft-attention startup mode differs",
    )?;
    need(
        mtp_request_carry(ready)? == expected_request_carry,
        "MTP request-carry startup mode differs",
    )?;
    if mtp_input_packet(ready)? >= 512 {
        need(
            integer(ready, PRIMING_FIELDS[2])? == 0,
            "MTP ready already submitted batch priming",
        )?;
    }
    if expected_request_carry {
        let generation = get(ready, "native_mtp_generation")?;
        need(
            integer(generation, "id")? == 0 && !get(generation, "active")?.boolean()?,
            "MTP ready contains a generation",
        )?;
    }
    for name in [
        "native_mtp_head_calls",
        "native_mtp_prime_calls",
        "native_mtp_rounds",
        "native_mtp_proposed",
        "native_mtp_accepted",
        "native_mtp_recovery_rounds",
        "native_mtp_carried_anchors",
        "native_mtp_discarded_anchors",
    ] {
        need(integer(ready, name)? == 0, "MTP ready executed work")?;
    }
    Ok(())
}

fn stop(id: u32, job: &Json) -> Result<bool> {
    if let Some(stops) = job.find("stop_ids")? {
        for item in stops.array()? {
            if item.unsigned_integer()? == u64::from(id) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// MTP wire reply session and admission tracker.
pub struct MtpWireReply<'a> {
    ready: &'a Json,
    previous: &'a Json,
    job: &'a Json,
    depth: u32,
    scoped: bool,
    abort: bool,
    generation: MtpGenerationCommand,
    reset: bool,
    resync: bool,
    inputs: u32,
    maximum: u32,
    base: u32,
    position: u32,
    rounds: u32,
    accepted: u32,
    proposed: u32,
    head_calls: u32,
    prime_extra: u32,
    rejected: u32,
    ordinary: u32,
    carried: u32,
    discarded: u64,
    pending_tokens: u32,
    round_end: u32,
    carry: bool,
    stopped: bool,
    finished: bool,
    restored_input_tokens: u32,
    final_input_logits: bool,
    final_input: bool,
    last_words: u64,
    events: Vec<Json>,
    ids: Vec<u32>,
    token_words: Vec<u64>,
}

impl<'a> MtpWireReply<'a> {
    pub fn new(ready: &'a Json, previous: &'a Json, job: &'a Json, resync: bool) -> Result<Self> {
        validate_prefill_identity(ready, previous)?;
        let depth = mtp_depth(ready)?;
        let scoped = mtp_request_carry(ready)?;
        let final_input_logits = mtp_final_input_logits(ready)?;

        if scoped {
            keys(
                job,
                &[
                    "tokens",
                    "generate",
                    "reset",
                    "stop_ids",
                    "sampling",
                    "seed",
                    "mtp_generation",
                    "prefill_final",
                ],
            )?;
        } else {
            keys(
                job,
                &[
                    "tokens", "generate", "reset", "stop_ids", "sampling", "seed",
                ],
            )?;
        }

        let mut generation = MtpGenerationCommand::default();
        let mut abort = false;
        if let Some(cmd) = job.find("mtp_generation")? {
            need(scoped, "MTP generation command lacks explicit scope")?;
            generation = mtp_generation_command(cmd)?;
            abort = generation.abort;
        }

        let reset = get(job, "reset")?.boolean()?;
        let tokens = get(job, "tokens")?.array()?;
        let inputs = tokens.len() as u32;
        let outputs = integer(job, "generate")?;

        let input_packet_limit = mtp_input_packet(ready)?;
        let output_packet_limit = mtp_output_packet(ready)?;

        need(
            (inputs >= 1 && inputs <= input_packet_limit && outputs == 0)
                || (inputs == 0 && outputs >= 1 && outputs <= u64::from(output_packet_limit))
                || (abort && inputs == 0 && outputs == 0),
            "MTP packet exceeds negotiated input or output bound",
        )?;

        let maximum = outputs as u32;
        let mut final_input = true;
        if let Some(final_val) = job.find("prefill_final")? {
            need(
                final_input_logits && inputs > 0,
                "MTP final-input flag presence differs",
            )?;
            final_input = final_val.boolean()?;
        }

        let event = get(previous, "event")?.string()?;
        need(
            event == b"ready" || event == b"done",
            "MTP previous ancestry event differs",
        )?;
        need(event != b"ready" || reset, "MTP first packet must reset")?;

        need(!reset || inputs > 0, "MTP reset must contain known inputs")?;
        if reset {
            let policy = get(job, "sampling")?;
            keys(policy, &["temperature", "top_k", "top_p", "min_p"])?;
            let temp = real(policy, "temperature")?;
            let top_k = integer(policy, "top_k")?;
            let top_p = real(policy, "top_p")?;
            let min_p = real(policy, "min_p")?;
            need(
                (0.0..=100.0).contains(&temp)
                    && top_k <= 248320
                    && top_p > 0.0
                    && top_p <= 1.0
                    && (0.0..=1.0).contains(&min_p),
                "MTP reset sampling policy bound",
            )?;
            integer(job, "seed")?;
        }

        let before = if event == b"ready" {
            0
        } else {
            integer(previous, "committed_prefix")?
        };
        need(before <= 65536, "MTP previous context bound")?;
        let base = if reset { 0 } else { before as u32 };
        let pending_overflow = if generation.v2 && !generation.begin {
            generation.pending
        } else {
            0
        };
        let mut position = base;
        let mut pending_tokens = 0;
        let mut round_end = 0;
        if pending_overflow > 0 {
            let max_overflow = if depth == 7 { 7 } else { depth };
            need(
                pending_overflow <= max_overflow && pending_overflow <= position,
                "invalid MTP pending overflow bound",
            )?;
            position -= pending_overflow;
            pending_tokens = pending_overflow;
            round_end = base;
        }
        need(
            u64::from(base) + u64::from(inputs) + u64::from(maximum) <= 65536,
            "MTP packet exceeds context reservation",
        )?;

        let mut last_words = 0u64;
        if !reset {
            let sampling = get(previous, "sampling")?;
            need(
                get(sampling, "valid")?.boolean()?
                    && integer(sampling, "entropy_prefix")? == u64::from(base),
                "MTP previous entropy frontier differs",
            )?;
            if maximum > 0 {
                need(
                    get(previous, "continuation_allowed")?.boolean()?,
                    "MTP cannot continue stopped prefix without input",
                )?;
            }
            last_words = integer(sampling, "entropy_words")?;
        }

        let mut carry = false;
        let mut discarded = 0u64;
        if scoped {
            let previous_generation = get(previous, "native_mtp_generation")?;
            let previous_id = integer(previous_generation, "id")?;
            let previous_remaining = integer(previous_generation, "remaining")?;
            let active = get(previous_generation, "active")?.boolean()?;
            let pending = get(previous_generation, "pending")?.boolean()?;
            if inputs > 0 {
                need(
                    job.find("mtp_generation")?.is_none(),
                    "known inputs cannot carry MTP generation commands",
                )?;
                discarded = if !resync && pending { 1 } else { 0 };
            } else {
                need(
                    job.find("mtp_generation")?.is_some(),
                    "MTP output/control requires a generation command",
                )?;
                if abort {
                    need(
                        active && generation.id == previous_id && job.find("stop_ids")?.is_none(),
                        "MTP abort identity differs",
                    )?;
                    discarded = if pending { 1 } else { 0 };
                } else {
                    need(
                        u64::from(generation.remaining) >= u64::from(maximum),
                        "MTP packet exceeds remaining generation budget",
                    )?;
                    if generation.begin {
                        need(
                            !active && generation.id > previous_id,
                            "MTP overlapping/stale generation begin",
                        )?;
                    } else {
                        need(
                            active
                                && generation.id == previous_id
                                && u64::from(generation.remaining) == previous_remaining,
                            "MTP continuation lifetime differs",
                        )?;
                        carry = pending;
                    }
                }
            }
        }

        for token in tokens {
            need(token.unsigned_integer()? < 248320, "MTP input token bound")?;
        }

        if let Some(stops) = job.find("stop_ids")? {
            let arr = stops.array()?;
            need(arr.len() <= 16, "MTP stop set bound")?;
            for item in arr {
                need(
                    item.unsigned_integer()? < 248320,
                    "MTP invalid/duplicate stop ID",
                )?;
            }
        }

        Ok(Self {
            ready,
            previous,
            job,
            depth,
            scoped,
            abort,
            generation,
            reset,
            resync,
            inputs,
            maximum,
            base,
            position,
            rounds: 0,
            accepted: 0,
            proposed: 0,
            head_calls: 0,
            prime_extra: 0,
            rejected: 0,
            ordinary: 0,
            carried: 0,
            discarded,
            pending_tokens,
            round_end,
            carry,
            stopped: false,
            finished: false,
            restored_input_tokens: 0,
            final_input_logits,
            final_input,
            last_words,
            events: Vec::new(),
            ids: Vec::new(),
            token_words: Vec::new(),
        })
    }

    pub fn event(&mut self, value: &Json) -> Result<()> {
        need(
            !self.finished && self.maximum > 0 && !self.stopped,
            "MTP unsolicited/excess/post-terminal event",
        )?;
        let type_val = get(value, "event")?.string()?;
        if type_val == b"mtp_round" {
            let count = integer(value, "count")? as u32;
            let accepted = integer(value, "accepted")? as u32;
            let proposals = if let Some(p) = value.find("proposed")? {
                p.unsigned_integer()? as u32
            } else {
                self.depth
            };
            let head_calls_field = value.find("head_calls")?;
            let prime_extra_field = value.find("prime_extra")?;
            need(
                head_calls_field.is_some() == prime_extra_field.is_some()
                    && (head_calls_field.is_none() || matches!(self.depth, 2 | 3)),
                "MTP round lookup-bypass accounting incomplete",
            )?;
            let round_head_calls = match head_calls_field {
                Some(v) => v.unsigned_integer()? as u32,
                None => proposals,
            };
            let round_prime_extra = match prime_extra_field {
                Some(v) => v.unsigned_integer()? as u32,
                None => 0,
            };
            need(
                (round_head_calls == proposals && round_prime_extra == 0)
                    || (round_head_calls == 0 && round_prime_extra == 1),
                "MTP round head/prime bypass accounting invalid",
            )?;
            let count_valid = if self.generation.v2 {
                (self.ids.len() as u32) < self.maximum
            } else {
                count <= self.maximum - (self.ids.len() as u32)
            };
            need(
                self.pending_tokens == 0
                    && count >= 1
                    && count <= proposals + 1
                    && accepted <= proposals
                    && count == 1 + accepted
                    && count_valid
                    && integer(value, "base")? == u64::from(self.position)
                    && integer(value, "committed_prefix")? == u64::from(self.position + count)
                    && get(value, "lookahead_anchor")?.boolean()?
                    && get(value, "used_carried_anchor")?.boolean()? == self.carry,
                "MTP round prefix/shape/carry differs",
            )?;
            if self.carry {
                self.carried += 1;
            }
            self.carry = get(value, "pending_anchor")?.boolean()?;
            self.rounds += 1;
            self.proposed += proposals;
            self.head_calls += round_head_calls;
            self.prime_extra += round_prime_extra;
            if accepted < proposals {
                self.rejected += 1;
            }
            self.accepted += accepted;
            let tokens_in_packet = if self.generation.v2 {
                count.min(self.maximum - (self.ids.len() as u32))
            } else {
                count
            };
            self.pending_tokens = tokens_in_packet;
            self.round_end = self.position + count;
        } else if type_val == b"token" {
            let id = integer(value, "id")? as u32;
            let words = integer(value, "entropy_words")?;
            need(
                (self.ids.len() as u32) < self.maximum && id < 248320 && words >= self.last_words,
                "MTP token/entropy bound",
            )?;
            if self.pending_tokens > 0 {
                need(
                    integer(value, "committed_prefix")? == u64::from(self.round_end)
                        && integer(value, "round_committed_prefix")? == u64::from(self.round_end)
                        && integer(value, "token_prefix")? == u64::from(self.position + 1),
                    "MTP batched token prefix differs",
                )?;
                self.pending_tokens -= 1;
            } else {
                need(
                    integer(value, "committed_prefix")? == u64::from(self.position + 1),
                    "MTP ordinary token prefix differs",
                )?;
                self.ordinary += 1;
                if self.carry {
                    self.carried += 1;
                }
                self.carry = false;
            }
            self.position += 1;
            self.ids.push(id);
            self.token_words.push(words);
            self.last_words = words;
            self.stopped = stop(id, self.job)?;
            need(
                !self.stopped || (self.pending_tokens == 0 && !self.carry),
                "MTP terminal appeared before consumed round end",
            )?;
        } else {
            return Err(MtpWireError::Refused("MTP unknown native event"));
        }
        self.events.push(value.clone());
        Ok(())
    }

    pub fn finish(&mut self, done: &Json) -> Result<()> {
        validate_runtime_policy_identity(self.ready, done)?;
        validate_measured_timing_packet(
            self.ready,
            self.previous,
            done,
            self.generation.begin,
            self.generation.id,
            self.rounds + self.ordinary,
        )?;
        validate_prefill_identity(self.ready, done)?;
        if mtp_input_packet(self.ready)? >= 512 {
            need(
                integer(done, PRIMING_FIELDS[2])? >= integer(self.previous, PRIMING_FIELDS[2])?,
                "MTP priming submit counter rewound",
            )?;
        }
        need(
            !self.finished && self.pending_tokens == 0 && get(done, "event")?.string()? == b"done",
            "MTP incomplete/duplicate completion",
        )?;
        let count = self.ids.len() as u32;
        let expected_prefix = if self.generation.v2 && self.round_end > 0 {
            u64::from(self.round_end.max(self.position))
        } else {
            u64::from(self.base + self.inputs + count)
        };
        let actual_prefix = integer(done, "committed_prefix")?;
        need(
            actual_prefix == expected_prefix
                && (if self.inputs > 0 {
                    count == 0
                } else if self.stopped {
                    count >= 1
                } else {
                    count == self.maximum
                }),
            "MTP completion prefix/count differs",
        )?;

        // Native input_at is the restored portion of this packet, not a lifetime cache counter or an absolute prefix. A carrier that omits both fields reports no native reuse.
        let restored_input_tokens = match (
            done.find("checkpoint_restored_tokens")?,
            done.find("replayed_input_tokens")?,
        ) {
            (None, None) => 0,
            (Some(restored), Some(replayed)) => {
                let restored = restored.unsigned_integer()?;
                let replayed = replayed.unsigned_integer()?;
                need(
                    restored <= u64::from(self.inputs)
                        && replayed == u64::from(self.inputs) - restored,
                    "MTP restored/replayed input accounting differs",
                )?;
                restored as u32
            }
            _ => {
                return Err(MtpWireError::Refused(
                    "MTP restored/replayed input metadata incomplete",
                ));
            }
        };

        let ids = get(done, "generated_ids")?.array()?;
        need(
            ids.len() == self.ids.len(),
            "MTP completion ID count differs",
        )?;
        for (i, token) in ids.iter().enumerate() {
            need(
                token.unsigned_integer()? == u64::from(self.ids[i]),
                "MTP token/DONE ID disagreement",
            )?;
        }

        let sampled = get(done, "sampling")?;
        need(
            get(sampled, "valid")?.boolean()?,
            "MTP sampler ancestry differs",
        )?;

        let available = !self.final_input_logits || self.inputs == 0 || self.final_input;
        need(
            get(done, "continuation_allowed")?.boolean()? == (available && !self.stopped),
            "MTP continuation flag differs",
        )?;

        if self.inputs > 0 || self.abort {
            need(
                get(done, "status")?.string()? == b"prefilled",
                "MTP input emitted terminal state",
            )?;
        } else if self.stopped {
            need(
                get(done, "status")?.string()? == b"stopped"
                    && integer(done, "stop_id")? == u64::from(self.ids[count as usize - 1]),
                "MTP stop completion differs",
            )?;
        } else {
            need(
                get(done, "status")?.string()? == b"length",
                "MTP length completion differs",
            )?;
        }

        if self.scoped {
            let observed = get(done, "native_mtp_generation")?;
            let remaining = if self.maximum > 0 && !self.stopped {
                self.generation.remaining.saturating_sub(count)
            } else {
                0
            };
            need(
                get(observed, "active")?.boolean()? == (remaining > 0)
                    && integer(observed, "remaining")? == u64::from(remaining)
                    && get(observed, "pending")?.boolean()? == (remaining > 0 && self.carry),
                "MTP generation completion identity/budget/pending differs",
            )?;
            if self.maximum > 0 && remaining == 0 && self.carry {
                self.discarded += 1;
            }
        } else {
            self.discarded = if self.carry { 1 } else { 0 };
        }

        let primes = if self.depth == 7 {
            if self.inputs > 0 {
                u64::from(self.inputs)
            } else {
                u64::from(self.rounds + self.accepted + self.ordinary)
            }
        } else if self.inputs > 0 {
            u64::from(self.inputs.saturating_sub(if self.reset { 1 } else { 0 }))
        } else {
            u64::from(self.accepted + self.ordinary)
        };
        let expected = [
            u64::from(self.head_calls),
            primes + u64::from(self.prime_extra),
            u64::from(self.rounds),
            u64::from(self.proposed),
            u64::from(self.accepted),
            u64::from(self.rejected),
            u64::from(self.carried),
            self.discarded,
        ];
        let names = [
            "native_mtp_head_calls",
            "native_mtp_prime_calls",
            "native_mtp_rounds",
            "native_mtp_proposed",
            "native_mtp_accepted",
            "native_mtp_recovery_rounds",
            "native_mtp_carried_anchors",
            "native_mtp_discarded_anchors",
        ];
        for i in 0..names.len() {
            delta(self.previous, done, names[i], expected[i], self.resync)?;
        }

        self.restored_input_tokens = restored_input_tokens;
        self.finished = true;
        Ok(())
    }

    /// Restored input tokens from a fully admitted completion, bounded by this command's input count.
    pub fn restored_input_tokens(&self) -> Result<u32> {
        need(
            self.finished,
            "MTP input accounting requires admitted completion",
        )?;
        Ok(self.restored_input_tokens)
    }

    pub fn events(&self) -> &[Json] {
        &self.events
    }

    pub fn ready(&self) -> &Json {
        self.ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokspur_json::parse_json;

    fn parse(text: &str) -> Json {
        parse_json(text.as_bytes()).expect("valid test JSON")
    }

    #[test]
    fn lookup_bypass_round_accounting_and_refusals() {
        let ready = parse(
            r#"{"event":"ready","native_mtp_depth":2,"native_mtp_carry_scope":"http-generation-v1","native_mtp_generation":{"id":0,"active":false,"remaining":0,"pending":false}}"#,
        );
        let prior = parse(
            r#"{"event":"done","committed_prefix":4,"continuation_allowed":true,"sampling":{"valid":true,"entropy_prefix":4,"entropy_words":0},"native_mtp_generation":{"id":0,"active":false,"remaining":0,"pending":false},"native_mtp_head_calls":0,"native_mtp_prime_calls":3,"native_mtp_rounds":0,"native_mtp_proposed":0,"native_mtp_accepted":0,"native_mtp_recovery_rounds":0,"native_mtp_carried_anchors":0,"native_mtp_discarded_anchors":0}"#,
        );
        let job = parse(
            r#"{"tokens":[],"generate":3,"reset":false,"mtp_generation":{"id":1,"begin":true,"remaining":6}}"#,
        );

        let mut reply = MtpWireReply::new(&ready, &prior, &job, false).expect("valid job");
        reply
            .event(&parse(
                r#"{"event":"mtp_round","base":4,"committed_prefix":7,"count":3,"accepted":2,"proposed":2,"head_calls":0,"prime_extra":1,"lookahead_anchor":true,"used_carried_anchor":false,"pending_anchor":true}"#,
            ))
            .expect("lookup-bypass round admitted");
        for pos in 0..3u32 {
            reply
                .event(&parse(&format!(
                    r#"{{"event":"token","id":42,"entropy_words":0,"committed_prefix":7,"round_committed_prefix":7,"token_prefix":{}}}"#,
                    5 + pos
                )))
                .expect("batched token admitted");
        }
        let done = parse(
            r#"{"event":"done","committed_prefix":7,"generated_ids":[42,42,42],"status":"length","continuation_allowed":true,"sampling":{"valid":true,"entropy_prefix":7,"entropy_words":0},"native_mtp_generation":{"id":1,"active":true,"remaining":3,"pending":true},"native_mtp_head_calls":0,"native_mtp_prime_calls":6,"native_mtp_rounds":1,"native_mtp_proposed":2,"native_mtp_accepted":2,"native_mtp_recovery_rounds":0,"native_mtp_carried_anchors":0,"native_mtp_discarded_anchors":0}"#,
        );
        reply.finish(&done).expect("lookup-bypass done admitted");

        for (hc, pe) in [(0, 0), (1, 1), (2, 1)] {
            let mut bad = MtpWireReply::new(&ready, &prior, &job, false).expect("valid job");
            assert!(
                bad.event(&parse(&format!(
                    r#"{{"event":"mtp_round","base":4,"committed_prefix":7,"count":3,"accepted":2,"proposed":2,"head_calls":{hc},"prime_extra":{pe},"lookahead_anchor":true,"used_carried_anchor":false,"pending_anchor":true}}"#
                )))
                .is_err(),
                "malformed bypass pair ({hc}, {pe}) must be refused"
            );
        }
    }
    fn with(value: &Json, name: &str, replacement: Json) -> Json {
        let mut fields = value.object().unwrap().to_vec();
        fields.retain(|(key, _)| key != name.as_bytes());
        fields.push((name.as_bytes().to_vec(), replacement));
        Json::Object(fields)
    }

    fn prefill_ready(capacity: u32, depth: u32, dense: &str, grouped: &str) -> Json {
        let full = depth == 7 || dense == "nax" || grouped == "nax";
        let priming = if full {
            capacity.min(2048)
        } else {
            capacity.min(512)
        };
        let attention = if depth == 7 { "split16" } else { "tiled128" };
        parse(&format!(
            r#"{{"event":"ready","maximum_generated_tokens":4096,"stack_execution":"direct","native_mtp_input_packet_schema":1,"native_mtp_input_packet_max":{capacity},"native_mtp_depth":{depth},"native_mtp_draft_attention":"{attention}","native_prefill_dense":"{dense}","native_prefill_grouped":"{grouped}","native_mtp_priming_batch_capacity":{priming},"native_mtp_priming_batch_bytes":1024,"native_mtp_priming_batch_submits":0,"native_mtp_carry_scope":"http-generation-v1","native_mtp_generation":{{"id":0,"active":false,"remaining":0,"pending":false}},"native_mtp_head_calls":0,"native_mtp_prime_calls":0,"native_mtp_rounds":0,"native_mtp_proposed":0,"native_mtp_accepted":0,"native_mtp_recovery_rounds":0,"native_mtp_carried_anchors":0,"native_mtp_discarded_anchors":0}}"#
        ))
    }

    #[test]
    fn priming_metadata_tracks_internal_capacity_and_refuses_wrong_bounds() {
        for capacity in [512, 1024, 2048, 8192] {
            for (depth, dense, grouped) in [
                (2, "f32", "f32"),
                (2, "f16", "serial"),
                (2, "nax", "f32"),
                (2, "f32", "nax"),
                (3, "f32", "f32"),
                (3, "nax", "nax"),
                (7, "f32", "f32"),
            ] {
                let ready = prefill_ready(capacity, depth, dense, grouped);
                let attention = if depth == 7 { "split16" } else { "tiled128" };
                assert_eq!(mtp_input_packet(&ready).unwrap(), capacity);
                validate_mtp_ready(&ready, attention, true, depth).unwrap();
                let expected = integer(&ready, PRIMING_FIELDS[0]).unwrap();
                for wrong in [0, 1, 256, 512, 1024, 2048, 4096] {
                    if wrong != expected {
                        assert!(
                            mtp_input_packet(&with(
                                &ready,
                                PRIMING_FIELDS[0],
                                Json::integer(wrong)
                            ))
                            .is_err()
                        );
                    }
                }
                let max_bytes = if expected >= 1024 || dense == "nax" {
                    160u64 << 20
                } else {
                    32u64 << 20
                };
                for bytes in [0, max_bytes + 1] {
                    assert!(
                        mtp_input_packet(&with(&ready, PRIMING_FIELDS[1], Json::integer(bytes)))
                            .is_err()
                    );
                }
                assert!(
                    mtp_input_packet(&with(&ready, PRIMING_FIELDS[1], Json::integer(max_bytes)))
                        .is_ok()
                );
                for field in PRIMING_FIELDS {
                    let mut fields = ready.object().unwrap().to_vec();
                    fields.retain(|(name, _)| name != field.as_bytes());
                    assert!(mtp_input_packet(&Json::Object(fields)).is_err());
                    assert!(mtp_input_packet(&with(&ready, field, Json::Bool(false))).is_err());
                }
                assert!(
                    validate_mtp_ready(
                        &with(&ready, PRIMING_FIELDS[2], Json::integer(1)),
                        attention,
                        true,
                        depth
                    )
                    .is_err()
                );
                assert!(
                    mtp_input_packet(&with(
                        &ready,
                        "native_mtp_priming_batch_extra",
                        Json::integer(0)
                    ))
                    .is_err()
                );
            }
        }
        assert!(mtp_input_packet(&parse(r#"{"native_mtp_priming_batch_capacity":512}"#)).is_err());
    }

    #[test]
    fn prefill_reply_refuses_metadata_drift_and_submit_counter_rewind() {
        let ready = prefill_ready(2048, 2, "f32", "f32");
        let job = parse(
            r#"{"tokens":[42],"generate":0,"reset":true,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":0}"#,
        );
        let mut done = with(&ready, "event", Json::from("done"));
        for (name, value) in [
            ("committed_prefix", Json::integer(1)),
            ("generated_ids", Json::Array(vec![])),
            (
                "sampling",
                parse(r#"{"valid":true,"entropy_prefix":1,"entropy_words":0}"#),
            ),
            ("continuation_allowed", Json::Bool(true)),
            ("status", Json::from("prefilled")),
            (PRIMING_FIELDS[2], Json::integer(1)),
        ] {
            done = with(&done, name, value);
        }
        MtpWireReply::new(&ready, &ready, &job, false)
            .unwrap()
            .finish(&done)
            .unwrap();
        for (field, changed) in [
            (PRIMING_FIELDS[0], Json::integer(2048)),
            (PRIMING_FIELDS[1], Json::integer(2048)),
            ("native_mtp_input_packet_max", Json::integer(1024)),
            ("native_prefill_dense", Json::from("f16")),
            ("native_prefill_grouped_kernel", Json::from("unexpected")),
        ] {
            assert!(
                MtpWireReply::new(&ready, &with(&ready, field, changed.clone()), &job, false)
                    .is_err()
            );
            assert!(
                MtpWireReply::new(&ready, &ready, &job, false)
                    .unwrap()
                    .finish(&with(&done, field, changed))
                    .is_err()
            );
        }
        let previous = with(&done, PRIMING_FIELDS[2], Json::integer(2));
        assert!(
            MtpWireReply::new(&ready, &previous, &job, false)
                .unwrap()
                .finish(&done)
                .is_err()
        );
    }
}

#[cfg(test)]
#[path = "measured_observation_tests.rs"]
mod measured_observation_tests;
