//! One persistent owned target, ported one to one from `native/src/owned_worker.cpp`.
//!
//! Fixed argv and one command in flight. Each admitted command performs at most one model step (one input OR one output), or in private V8 mode at most eight inputs OR eight outputs. Cancellation drains that command before reporting; the next request must explicitly reset. EOF, malformed framing and uncertain failures kill/drain the child, invalidate admission and never trigger restart, fallback or retry. Reentrant use is impossible: `request` takes `&mut self`.

use std::collections::BTreeSet;
use std::ffi::CString;
use std::fmt;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tokspur_json::{Json, JsonError, JsonLimits, Kind, from_chars_f64, parse_json_limited};
use tokspur_sys::fd::{self as sysfd, PairError};
use tokspur_sys::process::{self, GuardianFds, GuardianPlan};
use tokspur_sys::{EAGAIN, EINTR, POLLIN, POLLNVAL, PollFd, SIGKILL};

#[cfg(feature = "verify8")]
use crate::verify8::{Verify8Context, Verify8Error, Verify8Reply};

const LINE_LIMIT: usize = 1024 * 1024;
const STDERR_LIMIT: usize = 64 * 1024;
const POLL_SLICE_MS: i32 = 25;

/// Pace between drained commands so the model retains its packet and state semantics.
#[derive(Default)]
struct BurstPacing {
    outstanding: Option<(Instant, Duration)>,
}

impl BurstPacing {
    // At 60% duty, each three seconds of work incur two seconds of debt.
    const BURST_SLACK: Duration = Duration::from_nanos(28_000_000_000 * 2 / 3);

    fn debt_at(&self, now: Instant) -> Duration {
        self.outstanding.map_or(Duration::ZERO, |(end, debt)| {
            debt.saturating_sub(now.saturating_duration_since(end))
        })
    }

    fn delay_at(&self, now: Instant) -> Duration {
        self.debt_at(now).saturating_sub(Self::BURST_SLACK)
    }

    fn charge(&mut self, started: Instant, ended: Instant) {
        let elapsed = ended.saturating_duration_since(started);
        let debt = self.debt_at(started);
        self.outstanding = Some((ended, debt.saturating_add(elapsed.saturating_mul(2) / 3)));
    }
}

/// `WorkerError` / `WorkerCancelled`, plus the C++ `FormatError` paths that escape `request` unconverted.
pub enum WorkerError {
    /// `WorkerError`: refused, failed or ambiguous without an inner system/codec error.
    Refused(&'static str),
    StartupOutput {
        source: Box<WorkerError>,
        stderr: Vec<u8>,
    },
    /// `WorkerError`: pipe or socketpair creation/securing failed.
    Channel {
        create: &'static str,
        source: PairError,
    },
    /// `WorkerError`: guardian `fork` failed.
    GuardianLaunch(std::io::Error),
    /// `WorkerError`: `make_nonblocking` failed on a parent pipe/socket.
    PipeConfigure(std::io::Error),
    /// `WorkerError`: reading the guardian status pipe failed.
    StatusPipeRead(std::io::Error),
    /// `WorkerError`: reading worker stdout or stderr failed.
    PipeRead(std::io::Error),
    /// `WorkerError`: `poll` on worker descriptors failed.
    Poll(std::io::Error),
    /// `WorkerError`: sending a control byte to the guardian failed.
    ControlSend(std::io::Error),
    /// `WorkerError`: parsing a worker output line as JSON failed.
    MalformedReply(JsonError),
    /// `WorkerError`: startup configuration failed with an inner cause.
    StartupConfig(Box<dyn std::error::Error + Send + Sync>),
    /// `WorkerError`: MTP readiness validation failed during startup.
    MtpReady(crate::mtp_wire::MtpWireError),
    /// `WorkerError`: command encoding/admission failed before submission.
    EncodeBeforeSubmit(Box<WorkerError>),
    /// `WorkerError`: cancellation probe failed before submission.
    CancelProbe(CallbackFailed),
    /// `WorkerError`: MTP command validation failed before submission.
    MtpBeforeSubmit(crate::mtp_wire::MtpWireError),
    /// `WorkerError`: submitted command failed or became ambiguous; worker invalidated.
    CommandInvalidated(Option<Box<dyn std::error::Error + Send + Sync>>),
    /// `WorkerCancelled`: cancelled before submission or after a drain.
    Cancelled(&'static str),
    /// A JSON accessor refused the command before the submission guard.
    Json(JsonError),
    /// V8 admission refused the command before submission.
    #[cfg(feature = "verify8")]
    Admission(Verify8Error),
    /// MTP admission refused the command.
    Mtp(crate::mtp_wire::MtpWireError),
}

impl fmt::Debug for WorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartupOutput { stderr, .. } => formatter
                .debug_struct("StartupOutput")
                .field("bytes", &stderr.len())
                .finish_non_exhaustive(),
            Self::Refused(reason) => formatter.debug_tuple("Refused").field(reason).finish(),
            Self::Channel { create, .. } => formatter
                .debug_struct("Channel")
                .field("create", create)
                .field("source", &"[redacted]")
                .finish(),
            Self::GuardianLaunch(_) => formatter.write_str("GuardianLaunch([redacted])"),
            Self::PipeConfigure(_) => formatter.write_str("PipeConfigure([redacted])"),
            Self::StatusPipeRead(_) => formatter.write_str("StatusPipeRead([redacted])"),
            Self::PipeRead(_) => formatter.write_str("PipeRead([redacted])"),
            Self::Poll(_) => formatter.write_str("Poll([redacted])"),
            Self::ControlSend(_) => formatter.write_str("ControlSend([redacted])"),
            Self::MalformedReply(_) => formatter.write_str("MalformedReply([redacted])"),
            Self::StartupConfig(_) => formatter.write_str("StartupConfig([redacted])"),
            Self::MtpReady(_) => formatter.write_str("MtpReady([redacted])"),
            Self::EncodeBeforeSubmit(_) => formatter.write_str("EncodeBeforeSubmit([redacted])"),
            Self::CancelProbe(_) => formatter.write_str("CancelProbe(CallbackFailed)"),
            Self::MtpBeforeSubmit(_) => formatter.write_str("MtpBeforeSubmit([redacted])"),
            Self::CommandInvalidated(Some(_)) => {
                formatter.write_str("CommandInvalidated(Some([redacted]))")
            }
            Self::CommandInvalidated(None) => formatter.write_str("CommandInvalidated(None)"),
            Self::Cancelled(reason) => formatter.debug_tuple("Cancelled").field(reason).finish(),
            Self::Json(error) => formatter.debug_tuple("Json").field(error).finish(),
            #[cfg(feature = "verify8")]
            Self::Admission(error) => formatter.debug_tuple("Admission").field(error).finish(),
            Self::Mtp(error) => formatter.debug_tuple("Mtp").field(error).finish(),
        }
    }
}

impl fmt::Display for WorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartupOutput { source, .. } => source.fmt(formatter),
            Self::Refused(reason) | Self::Cancelled(reason) => formatter.write_str(reason),
            Self::Channel { create, source } => formatter.write_str(match source {
                PairError::Create(_) => create,
                PairError::Secure(_) => "cannot secure worker pipe descriptor",
                PairError::NoSigpipe(_) => "cannot suppress local control-stream SIGPIPE",
            }),
            Self::GuardianLaunch(_) => formatter.write_str("cannot launch owned worker guardian"),
            Self::PipeConfigure(_) => formatter.write_str("cannot configure bounded worker pipe"),
            Self::StatusPipeRead(_) => formatter.write_str("worker guardian status pipe failed"),
            Self::PipeRead(_) => formatter.write_str("owned worker pipe read failed"),
            Self::Poll(_) => formatter.write_str("owned worker poll failed"),
            Self::ControlSend(_) => formatter.write_str("owned guardian control channel failed"),
            Self::MalformedReply(_) => formatter.write_str("owned worker emitted malformed JSON"),
            Self::StartupConfig(_) => {
                formatter.write_str("owned worker startup configuration failed")
            }
            Self::MtpReady(_) => formatter.write_str("MTP startup readiness validation failed"),
            Self::EncodeBeforeSubmit(_) => {
                formatter.write_str("owned worker encoding refused before submission")
            }
            Self::CancelProbe(_) => {
                formatter.write_str("owned cancellation probe failed before submission")
            }
            Self::MtpBeforeSubmit(_) => {
                formatter.write_str("MTP command refused before submission")
            }
            Self::CommandInvalidated(_) => formatter.write_str(
                "owned command failed or became ambiguous; worker invalidated without retry",
            ),
            Self::Json(error) => error.fmt(formatter),
            #[cfg(feature = "verify8")]
            Self::Admission(error) => error.fmt(formatter),
            Self::Mtp(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for WorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StartupOutput { source, .. } => Some(source.as_ref()),
            Self::Channel { source, .. } => Some(source),
            Self::GuardianLaunch(error)
            | Self::PipeConfigure(error)
            | Self::StatusPipeRead(error)
            | Self::PipeRead(error)
            | Self::Poll(error)
            | Self::ControlSend(error) => Some(error),
            Self::MalformedReply(error) | Self::Json(error) => Some(error),
            Self::StartupConfig(error) => Some(error.as_ref()),
            Self::MtpReady(error) | Self::MtpBeforeSubmit(error) | Self::Mtp(error) => Some(error),
            Self::EncodeBeforeSubmit(error) => Some(error.as_ref()),
            Self::CancelProbe(error) => Some(error),
            Self::CommandInvalidated(Some(error)) => Some(error.as_ref()),
            #[cfg(feature = "verify8")]
            Self::Admission(error) => Some(error),
            Self::Refused(_) | Self::Cancelled(_) | Self::CommandInvalidated(None) => None,
        }
    }
}

impl WorkerError {
    /// Bounded pre-readiness stderr for the private command-line diagnostic only.
    pub fn startup_output(&self) -> Option<&[u8]> {
        match self {
            Self::StartupOutput { stderr, .. } => Some(stderr),
            _ => None,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        match self {
            Self::StartupOutput { source, .. } => source.is_cancelled(),
            _ => matches!(self, Self::Cancelled(_)),
        }
    }

    /// True when the error corresponds to C++ `WorkerError` rather than `WorkerCancelled` or `FormatError`.
    pub fn is_refused(&self) -> bool {
        match self {
            Self::StartupOutput { source, .. } => source.is_refused(),
            Self::Cancelled(_) | Self::Json(_) | Self::Mtp(_) => false,
            #[cfg(feature = "verify8")]
            Self::Admission(_) => false,
            Self::Refused(_)
            | Self::Channel { .. }
            | Self::GuardianLaunch(_)
            | Self::PipeConfigure(_)
            | Self::StatusPipeRead(_)
            | Self::PipeRead(_)
            | Self::Poll(_)
            | Self::ControlSend(_)
            | Self::MalformedReply(_)
            | Self::StartupConfig(_)
            | Self::MtpReady(_)
            | Self::EncodeBeforeSubmit(_)
            | Self::CancelProbe(_)
            | Self::MtpBeforeSubmit(_)
            | Self::CommandInvalidated(_) => true,
        }
    }
}

impl From<JsonError> for WorkerError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

impl From<crate::mtp_wire::MtpWireError> for WorkerError {
    fn from(error: crate::mtp_wire::MtpWireError) -> Self {
        Self::Mtp(error)
    }
}

#[cfg(feature = "verify8")]
impl From<Verify8Error> for WorkerError {
    fn from(error: Verify8Error) -> Self {
        Self::Admission(error)
    }
}

type Result<T> = std::result::Result<T, WorkerError>;

fn require(ok: bool, message: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(WorkerError::Refused(message))
    }
}

/// `OwnedWorkerOptions`.
pub struct WorkerOptions {
    pub executable: PathBuf,
    pub policy: PathBuf,
    pub library: PathBuf,
    pub shards: Vec<PathBuf>,
    /// Startup only: `direct` or `recurrent-replay`.
    pub stack_execution: String,
    pub checkpoint_slots: u32,
    /// Paired opt-in; enables zero-step broker controls.
    pub checkpoint_budget_bytes: u64,
    pub startup_timeout: Duration,
    pub step_timeout: Duration,
    pub drain_timeout: Duration,
    /// Experimental host pacing between completed inference commands.
    pub burst_pacing: bool,
    /// Parent only; never invoked after fork.
    pub startup_cancelled: Option<Box<dyn Fn() -> bool + Send>>,
    /// Immutable, private greedy-only HTTP capability.
    #[cfg(feature = "verify8")]
    pub verify8: bool,
    /// Explicit new broker wire; never inferred from slots.
    #[cfg(feature = "branch8")]
    pub branch8: bool,
    /// Absent is not prepared-off.
    #[cfg(feature = "qsa-stage-v8")]
    pub stage8_present: bool,
    #[cfg(feature = "qsa-stage-v8")]
    pub stage8_enabled: bool,
    /// Explicit staged/channel pair; both use corrected signed-root PLE.
    #[cfg(feature = "fast-math")]
    pub fast_channel: Option<bool>,
    pub scalar_channel: bool,
    pub mtp_sidecar: PathBuf,
    pub dflash2_dir: PathBuf,
    pub drafter: Option<String>,
    pub dflash2_precision: Option<String>,
    pub mtp_depth: Option<u32>,
    pub mtp_request_carry: bool,
    pub mtp_draft_attention: Option<String>,
    pub mtp_output_packet: Option<u32>,
    pub qsa_sparse: Option<bool>,
    pub gdn_ported: Option<bool>,
    pub gdn_prefill_ported: Option<bool>,
    pub prefill_input512: bool,
    pub prefill_dense: Option<String>,
    pub prefill_grouped: Option<String>,
    pub prefill_hc_rows: Option<bool>,
    pub prefill_router_window: Option<bool>,
    pub prefill_hc_down_window: Option<bool>,
    pub prefill_hc_down_f32: Option<bool>,
    pub prefill_final_logits: bool,
    pub gdn_old_order_batch: bool,
    pub q8_three_rows: Option<bool>,
    pub concurrent_dispatch: Option<bool>,
    pub derived_pack: Option<PathBuf>,
    pub derived_pack_auth_workers: Option<u32>,
    pub derived_pack_trust_mode: Option<String>,
    pub raw_decode: Option<bool>,
    pub mtp_draft_slice: Option<bool>,
    pub verify_nax: Option<bool>,
    pub verify_nax_m8: Option<bool>,
    pub verify_hc_gdn_fused: Option<bool>,
    pub host_logit_views: Option<bool>,
    pub logit_topk: Option<bool>,
    pub ple_async_gather: Option<bool>,
    pub mtp_fast_round: Option<bool>,
    pub dflash2_defer_context: Option<bool>,
    pub dflash2_verify_rows: Option<String>,
    pub prepared_validation: bool,
    pub verify_cost_profile: Option<PathBuf>,
    pub dflash2_draft_policy: String,
    pub verify_m8_splits: Option<PathBuf>,
    pub dflash2_max_drafts: Option<u32>,
    pub dflash2_vocab_slice: Option<u32>,
    pub known_token_capacity: Option<u32>,
    pub checkpoint_ssd: bool,
    pub checkpoint_ssd_budget_bytes: u64,
    pub checkpoint_ssd_floor_bytes: u64,
    pub checkpoint_ssd_dir: PathBuf,
    pub checkpoint_policy: String,
    pub overlap_prefill: bool,
    pub swiglu_fused: Option<bool>,
    pub q8_gemv_rg4: Option<String>,
    pub q8_three_msg: Option<String>,
    pub moe_three_union: Option<bool>,
    pub mtp_lookup: Option<String>,
    pub mtp_lookup_match: Option<u32>,
    pub prefill_m64: Option<String>,
    pub prefill_multiseg: Option<bool>,
    pub prefill_swizzle: Option<bool>,
    pub prefill_router_fast: Option<bool>,
    pub qsa_prefill_fused: Option<bool>,
    pub hc_consumer_fused: Option<bool>,
    pub ple_overlap: Option<bool>,
    pub shared_prefix_cache: Option<bool>,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            executable: PathBuf::new(),
            policy: PathBuf::new(),
            library: PathBuf::new(),
            shards: Vec::new(),
            stack_execution: "direct".to_owned(),
            checkpoint_slots: 0,
            checkpoint_budget_bytes: 0,
            startup_timeout: Duration::from_millis(1_200_000),
            step_timeout: Duration::from_millis(60_000),
            drain_timeout: Duration::from_millis(10_000),
            burst_pacing: false,
            startup_cancelled: None,
            #[cfg(feature = "verify8")]
            verify8: false,
            #[cfg(feature = "branch8")]
            branch8: false,
            #[cfg(feature = "qsa-stage-v8")]
            stage8_present: false,
            #[cfg(feature = "qsa-stage-v8")]
            stage8_enabled: false,
            #[cfg(feature = "fast-math")]
            fast_channel: None,
            scalar_channel: false,
            mtp_sidecar: PathBuf::new(),
            dflash2_dir: PathBuf::new(),
            drafter: None,
            dflash2_precision: None,
            mtp_depth: None,
            mtp_request_carry: false,
            mtp_draft_attention: None,
            mtp_output_packet: None,
            qsa_sparse: None,
            gdn_ported: None,
            gdn_prefill_ported: None,
            prefill_input512: false,
            prefill_dense: None,
            prefill_grouped: None,
            prefill_hc_rows: None,
            prefill_router_window: None,
            prefill_hc_down_window: None,
            prefill_hc_down_f32: None,
            prefill_final_logits: false,
            gdn_old_order_batch: false,
            q8_three_rows: None,
            concurrent_dispatch: None,
            derived_pack: None,
            derived_pack_auth_workers: None,
            derived_pack_trust_mode: None,
            raw_decode: None,
            mtp_draft_slice: None,
            verify_nax: None,
            verify_nax_m8: None,
            verify_hc_gdn_fused: None,
            host_logit_views: None,
            logit_topk: None,
            ple_async_gather: None,
            mtp_fast_round: None,
            dflash2_defer_context: None,
            dflash2_verify_rows: None,
            prepared_validation: false,
            verify_cost_profile: None,
            dflash2_draft_policy: "always".to_owned(),
            verify_m8_splits: None,
            dflash2_max_drafts: None,
            dflash2_vocab_slice: None,
            known_token_capacity: None,
            checkpoint_ssd: false,
            checkpoint_ssd_budget_bytes: 8 * (1024 * 1024 * 1024),
            checkpoint_ssd_floor_bytes: 20 * (1024 * 1024 * 1024),
            checkpoint_ssd_dir: PathBuf::new(),
            checkpoint_policy: "recompute-cost".to_owned(),
            overlap_prefill: false,
            swiglu_fused: None,
            q8_gemv_rg4: None,
            q8_three_msg: None,
            moe_three_union: None,
            mtp_lookup: None,
            mtp_lookup_match: None,
            prefill_m64: None,
            prefill_multiseg: None,
            prefill_swizzle: None,
            prefill_router_fast: None,
            qsa_prefill_fused: None,
            hc_consumer_fused: None,
            ple_overlap: None,
            shared_prefix_cache: None,
        }
    }
}

pub fn scalar_channel_transport(options: &WorkerOptions) -> bool {
    #[cfg(feature = "verify8")]
    let verify8 = options.verify8;
    #[cfg(not(feature = "verify8"))]
    let verify8 = false;

    #[cfg(feature = "branch8")]
    let branch8 = options.branch8;
    #[cfg(not(feature = "branch8"))]
    let branch8 = false;

    #[cfg(feature = "qsa-stage-v8")]
    let stage8 = options.stage8_present && options.stage8_enabled;
    #[cfg(not(feature = "qsa-stage-v8"))]
    let stage8 = false;

    #[cfg(feature = "fast-math")]
    let fast_channel = options.fast_channel == Some(true);
    #[cfg(not(feature = "fast-math"))]
    let fast_channel = false;

    options.scalar_channel
        && options.stack_execution == "direct"
        && !verify8
        && !branch8
        && options.checkpoint_slots <= 4
        && options.checkpoint_budget_bytes <= (512 << 20)
        && stage8
        && fast_channel
}

pub fn has_speculative(options: &WorkerOptions) -> bool {
    !options.mtp_sidecar.as_os_str().is_empty() || !options.dflash2_dir.as_os_str().is_empty()
}

pub fn owned_concurrent_transport(options: &WorkerOptions) -> bool {
    let mtp2_ok = !options.mtp_sidecar.as_os_str().is_empty()
        && options.dflash2_dir.as_os_str().is_empty()
        && matches!(options.mtp_depth, Some(2 | 3))
        && options.mtp_request_carry
        && options.mtp_draft_attention.as_deref() == Some("tiled128");
    let dflash2_ok = !options.dflash2_dir.as_os_str().is_empty()
        && options.mtp_sidecar.as_os_str().is_empty()
        && options.mtp_depth == Some(7)
        && options.mtp_request_carry
        && options.mtp_draft_attention.as_deref() == Some("split16");
    scalar_channel_transport(options) && (mtp2_ok || dflash2_ok)
}

impl fmt::Debug for WorkerOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerOptions")
            .field("stack_execution", &self.stack_execution)
            .field("checkpoint_slots", &self.checkpoint_slots)
            .field("checkpoint_budget_bytes", &self.checkpoint_budget_bytes)
            .field("burst_pacing", &self.burst_pacing)
            .field("shards", &self.shards.len())
            .finish_non_exhaustive()
    }
}

fn worker_field<'a>(value: &'a Json, name: &str) -> Result<&'a Json> {
    value
        .find(name)?
        .ok_or(WorkerError::Refused("missing owned broker field"))
}

fn worker_integer(value: &Json, name: &str) -> Result<u64> {
    Ok(worker_field(value, name)?.unsigned_integer()?)
}

fn worker_number(value: &Json) -> Result<f64> {
    let text = value.number_lexeme()?;
    from_chars_f64(text)
        .filter(|number| number.is_finite())
        .ok_or(WorkerError::Refused("invalid owned broker numeric field"))
}

fn worker_keys(value: &Json, allowed: &[&str]) -> Result<()> {
    for (name, _) in value.object()? {
        require(
            allowed
                .iter()
                .any(|candidate| candidate.as_bytes() == name.as_slice()),
            "unsupported owned broker command field",
        )?;
    }
    Ok(())
}

fn event_is(value: &Json, name: &str) -> Result<bool> {
    Ok(match value.find("event")? {
        Some(event) => event.kind() == Kind::String && event.string()? == name.as_bytes(),
        None => false,
    })
}

/// Startup failures before the constructor's conversion: a `WorkerError` passes through, anything else becomes the generic configuration refusal while retaining its cause.
enum StartError {
    Worker(WorkerError),
    Cause(Box<dyn std::error::Error + Send + Sync>),
}

impl From<WorkerError> for StartError {
    fn from(error: WorkerError) -> Self {
        if error.is_refused() || error.is_cancelled() {
            Self::Worker(error)
        } else {
            Self::Cause(Box::new(error))
        }
    }
}

impl From<JsonError> for StartError {
    fn from(error: JsonError) -> Self {
        Self::Cause(Box::new(error))
    }
}

#[cfg(feature = "verify8")]
impl From<Verify8Error> for StartError {
    fn from(error: Verify8Error) -> Self {
        Self::Cause(Box::new(error))
    }
}

fn pair_error(source: PairError, create: &'static str) -> WorkerError {
    WorkerError::Channel { create, source }
}

fn fixed_file(path: &Path) -> std::result::Result<PathBuf, StartError> {
    require(
        path.is_absolute() && !path.as_os_str().as_bytes().contains(&0),
        "worker assets require absolute fixed paths",
    )?;
    let result = std::fs::canonicalize(path).map_err(|error| StartError::Cause(Box::new(error)))?;
    let regular = std::fs::metadata(&result).is_ok_and(|state| state.is_file());
    require(regular, "worker asset must be a regular file")?;
    Ok(result)
}

fn fixed_dir(path: &Path) -> std::result::Result<PathBuf, StartError> {
    require(
        path.is_absolute() && !path.as_os_str().as_bytes().contains(&0),
        "worker assets require absolute fixed paths",
    )?;
    let result = std::fs::canonicalize(path).map_err(|error| StartError::Cause(Box::new(error)))?;
    let is_dir = std::fs::metadata(&result).is_ok_and(|state| state.is_dir());
    require(is_dir, "worker asset must be a directory")?;
    Ok(result)
}

fn milliseconds_until(deadline: Instant, now: Instant) -> i64 {
    // duration_cast<milliseconds>(deadline - now): truncates toward zero.
    if deadline >= now {
        i64::try_from((deadline - now).as_millis()).unwrap_or(i64::MAX)
    } else {
        -i64::try_from((now - deadline).as_millis()).unwrap_or(i64::MAX)
    }
}

/// The parent-side channel ends and process/framing state.
struct Io {
    control: Option<OwnedFd>,
    status: Option<OwnedFd>,
    input: Option<OwnedFd>,
    output: Option<OwnedFd>,
    error: Option<OwnedFd>,
    guardian_pid: i32,
    worker_pid: i32,
    armed: bool,
    worker_exited: bool,
    guardian_reaped: bool,
    out_eof: bool,
    err_eof: bool,
    status_eof: bool,
    failed: bool,
    closed: bool,
    starting: bool,
    worker_status: i32,
    guardian_status: i32,
    stderr_bytes: usize,
    startup_stderr: Vec<u8>,
    pending: Vec<u8>,
    notice_bytes: Vec<u8>,
    step_timeout: Duration,
    drain_timeout: Duration,
    startup_cancelled: Option<Box<dyn Fn() -> bool + Send>>,
}

fn raw(fd: &Option<OwnedFd>, eof: bool) -> i32 {
    if eof {
        -1
    } else {
        fd.as_ref().map_or(-1, AsRawFd::as_raw_fd)
    }
}

impl Io {
    fn drop_channels(&mut self) {
        self.control = None;
        self.status = None;
        self.input = None;
        self.output = None;
        self.error = None;
    }

    fn status_input(&mut self) -> Result<()> {
        let mut data = [0u8; 128];
        loop {
            let Some(fd) = self.status.as_ref() else {
                return Err(WorkerError::Refused("worker guardian status pipe failed"));
            };
            match sysfd::read(fd.as_fd(), &mut data) {
                Ok(0) => {
                    self.status_eof = true;
                    break;
                }
                Ok(n) => {
                    self.notice_bytes.extend_from_slice(&data[..n]);
                    require(
                        self.notice_bytes.len() <= process::NOTICE_BYTES * 4,
                        "worker guardian status framing overflow",
                    )?;
                    while self.notice_bytes.len() >= process::NOTICE_BYTES {
                        let note: Vec<u8> =
                            self.notice_bytes.drain(..process::NOTICE_BYTES).collect();
                        let kind = i32::from_ne_bytes([note[0], note[1], note[2], note[3]]);
                        let value = i32::from_ne_bytes([note[4], note[5], note[6], note[7]]);
                        if kind == process::ARMED_NOTICE
                            && !self.armed
                            && value == self.guardian_pid
                        {
                            self.armed = true;
                        } else if kind == process::LAUNCHED_NOTICE
                            && self.armed
                            && self.worker_pid < 0
                            && value > 1
                        {
                            self.worker_pid = value;
                        } else if kind == process::EXIT_NOTICE && self.armed && !self.worker_exited
                        {
                            self.worker_exited = true;
                            self.worker_status = value;
                        } else {
                            return Err(WorkerError::Refused(
                                "worker guardian state transition refused",
                            ));
                        }
                    }
                }
                Err(error) => match error.raw_os_error() {
                    Some(EINTR) => continue,
                    Some(EAGAIN) => break,
                    _ => return Err(WorkerError::StatusPipeRead(error)),
                },
            }
        }
        require(
            !self.status_eof || self.notice_bytes.is_empty(),
            "truncated worker guardian status",
        )
    }

    fn read_stream(&mut self, standard_output: bool, discard: bool) -> Result<()> {
        let mut data = [0u8; 4096];
        for _ in 0..16 {
            let stream = if standard_output {
                &self.output
            } else {
                &self.error
            };
            let Some(fd) = stream.as_ref() else {
                return Err(WorkerError::Refused("owned worker pipe read failed"));
            };
            let n = match sysfd::read(fd.as_fd(), &mut data) {
                Ok(0) => {
                    if standard_output {
                        self.out_eof = true;
                    } else {
                        self.err_eof = true;
                    }
                    break;
                }
                Ok(n) => n,
                Err(error) => match error.raw_os_error() {
                    Some(EINTR) => continue,
                    Some(EAGAIN) => break,
                    _ => return Err(WorkerError::PipeRead(error)),
                },
            };
            if standard_output && !discard {
                self.pending.extend_from_slice(&data[..n]);
                let line = self
                    .pending
                    .iter()
                    .position(|&c| c == b'\n')
                    .unwrap_or(self.pending.len());
                require(
                    line <= LINE_LIMIT && self.pending.len() <= LINE_LIMIT + 65536,
                    "owned worker output framing limit exceeded",
                )?;
            } else if !standard_output {
                self.stderr_bytes = STDERR_LIMIT.min(self.stderr_bytes + n);
                if self.starting {
                    let keep = (4096 - self.startup_stderr.len()).min(n);
                    self.startup_stderr.extend_from_slice(&data[..keep]);
                }
            }
        }
        Ok(())
    }

    fn pump(&mut self, deadline: Instant, discard: bool, wait_ms: i32) -> Result<()> {
        if !discard && self.starting && self.startup_cancelled.as_ref().is_some_and(|probe| probe())
        {
            return Err(WorkerError::Refused(
                "owned startup cancelled before readiness; no automatic restart",
            ));
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(());
        }
        let mut watches = [
            PollFd {
                fd: raw(&self.output, self.out_eof),
                events: POLLIN,
                revents: 0,
            },
            PollFd {
                fd: raw(&self.error, self.err_eof),
                events: POLLIN,
                revents: 0,
            },
            PollFd {
                fd: raw(&self.status, self.status_eof),
                events: POLLIN,
                revents: 0,
            },
        ];
        let remaining = milliseconds_until(deadline, Instant::now());
        let timeout = remaining.max(1).min(i64::from(wait_ms)) as i32;
        if let Err(error) = sysfd::poll(&mut watches, timeout)
            && error.raw_os_error() != Some(EINTR)
        {
            return Err(WorkerError::Poll(error));
        }
        for (i, watch) in watches.iter().enumerate() {
            if watch.revents == 0 {
                continue;
            }
            require(
                watch.revents & POLLNVAL == 0,
                "owned worker descriptor became invalid",
            )?;
            if i == 0 && !self.out_eof {
                self.read_stream(true, discard)?;
            }
            if i == 1 && !self.err_eof {
                self.read_stream(false, false)?;
            }
            if i == 2 && !self.status_eof {
                self.status_input()?;
            }
        }
        Ok(())
    }

    fn send_control(&mut self, byte: u8, deadline: Instant) -> Result<()> {
        loop {
            require(
                Instant::now() < deadline,
                "owned guardian handshake deadline expired",
            )?;
            let Some(fd) = self.control.as_ref() else {
                return Err(WorkerError::Refused(
                    "owned guardian control channel failed",
                ));
            };
            match sysfd::send(fd.as_fd(), &[byte], 0) {
                Ok(1) => return Ok(()),
                Ok(_) => {
                    return Err(WorkerError::Refused(
                        "owned guardian control channel failed",
                    ));
                }
                Err(error) => match error.raw_os_error() {
                    Some(EINTR) => continue,
                    Some(EAGAIN) => self.pump(deadline, false, POLL_SLICE_MS)?,
                    _ => {
                        return Err(WorkerError::ControlSend(error));
                    }
                },
            }
        }
    }

    fn next(&mut self, deadline: Instant) -> Result<Json> {
        loop {
            require(
                Instant::now() < deadline,
                "owned worker response deadline expired",
            )?;
            if let Some(newline) = self.pending.iter().position(|&c| c == b'\n') {
                require(
                    newline != 0 && newline <= LINE_LIMIT,
                    "owned worker emitted an empty/oversized line",
                )?;
                let line: Vec<u8> = self.pending.drain(..=newline).take(newline).collect();
                return parse_json_limited(&line, JsonLimits::new(LINE_LIMIT, 300_000, 12))
                    .map_err(WorkerError::MalformedReply);
            }
            require(
                !self.out_eof && !self.worker_exited && !self.status_eof,
                "owned worker ended before a complete response",
            )?;
            self.pump(deadline, false, POLL_SLICE_MS)?;
        }
    }

    fn idle_check(&mut self) -> Result<()> {
        self.pump(Instant::now() + self.step_timeout, false, 0)?;
        require(
            !self.failed
                && !self.closed
                && !self.worker_exited
                && !self.out_eof
                && !self.status_eof
                && self.pending.is_empty(),
            "owned worker is not drained and ready for another command",
        )
    }

    fn shutdown(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        if self.guardian_pid <= 0 {
            self.closed = true;
            self.drop_channels();
            return Ok(());
        }
        self.input = None;
        self.pending.clear();
        let mut deadline = Instant::now() + self.drain_timeout;
        let mut forced = false;
        loop {
            if self.worker_exited
                && self.out_eof
                && self.err_eof
                && self.status_eof
                && self.notice_bytes.is_empty()
            {
                break;
            }
            if self.worker_pid > 0
                && self.status_eof
                && !self.worker_exited
                && process::exited_without_reaping(self.guardian_pid)
            {
                // If the guardian itself crashes, keep its PID unreaped while checking the worker's exact owned session/group. A recycled PID in another session is never a permissible signal target.
                if process::session_of(self.worker_pid) == self.guardian_pid
                    && process::group_of(self.worker_pid) == self.worker_pid
                {
                    best_effort_kill(-self.worker_pid);
                    best_effort_kill(self.worker_pid);
                }
                if self.out_eof && self.err_eof && process::process_absent(self.worker_pid) {
                    // No exact guardian-to-worker wait record exists. Stop the child, reap our guardian, but never claim verified normal closure for this exceptional path.
                    self.guardian_status = process::reap_child(self.guardian_pid);
                    self.guardian_reaped = true;
                    return Err(WorkerError::Refused(
                        "guardian crashed; worker stopped but exact child wait evidence is missing",
                    ));
                }
            }
            if self.worker_pid < 0 && self.status_eof {
                // No exec acknowledgement was sent. A failed guardian cannot have started the target; close its gate/lifetime endpoint and verify the exact guardian wait rather than inventing a ready.
                self.control = None;
                let (result, status, error) = process::wait_nohang(self.guardian_pid);
                if result == self.guardian_pid {
                    self.guardian_status = status;
                    self.guardian_reaped = true;
                    self.closed = true;
                    self.drop_channels();
                    return Ok(());
                }
                require(
                    result == 0 || (result < 0 && error == EINTR),
                    "unlaunched guardian waitpid failed",
                )?;
            }
            if Instant::now() >= deadline {
                if forced {
                    return Err(WorkerError::Refused(
                        "owned worker exit/EOF drain could not be verified; guardian remains responsible",
                    ));
                }
                self.control = None;
                forced = true;
                deadline = Instant::now() + Duration::from_millis(1500);
            }
            self.pump(deadline, true, POLL_SLICE_MS)?;
        }
        self.control = None;
        if !self.guardian_reaped && self.guardian_pid > 0 {
            loop {
                let (result, status, error) = process::wait_nohang(self.guardian_pid);
                if result == self.guardian_pid {
                    self.guardian_status = status;
                    self.guardian_reaped = true;
                    break;
                }
                if result < 0 && error == EINTR {
                    continue;
                }
                require(
                    result == 0 && Instant::now() < deadline,
                    "exact owned guardian waitpid could not be verified",
                )?;
                sysfd::pause(POLL_SLICE_MS);
            }
        }
        require(
            self.guardian_reaped
                && process::exited_zero(self.guardian_status)
                && self.worker_status != -1,
            "owned guardian/worker exit was not verified",
        )?;
        self.closed = true;
        self.drop_channels();
        Ok(())
    }
}

/// The C++ ignores `kill` results while stopping an orphaned worker group.
fn best_effort_kill(pid: i32) {
    if process::kill(pid, SIGKILL).is_err() {
        // Nothing to do: the target is already gone or not ours to signal.
    }
}

/// `OwnedWorker`.
pub struct OwnedWorker {
    io: Io,
    options: WorkerOptions,
    pacing: Option<BurstPacing>,
    ready_value: Json,
    last_response: Json,
    needs_reset: bool,
    broker_epoch: u64,
    broker_command: u64,
    broker_lease: u64,
    broker_position: u32,
    #[cfg(feature = "verify8")]
    verify_previous: Json,
    #[cfg(feature = "verify8")]
    verify_sampling: Json,
    #[cfg(feature = "verify8")]
    verify_seed: u64,
    #[cfg(feature = "verify8")]
    verify_position: u32,
}

impl fmt::Debug for OwnedWorker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedWorker")
            .field("guardian_pid", &self.io.guardian_pid)
            .field("worker_pid", &self.io.worker_pid)
            .field("closed", &self.io.closed)
            .field("failed", &self.io.failed)
            .finish_non_exhaustive()
    }
}

impl Drop for OwnedWorker {
    fn drop(&mut self) {
        if self.io.shutdown().is_err() {
            // `~Impl` swallows shutdown failures: the guardian stays responsible for the group, and the channels close below.
        }
        self.io.drop_channels();
    }
}

/// What a command admits before any byte is written.
struct Admitted {
    output_count: u64,
    input_count: u64,
    reset: bool,
    begin: bool,
    allow_restore: bool,
    control: bool,
    operation: Vec<u8>,
    mtp_abort: bool,
}

/// Failures inside the submitted-command block before its conversion.
enum CommandError {
    Cancelled(&'static str),
    Cause(Option<Box<dyn std::error::Error + Send + Sync>>),
}

impl From<WorkerError> for CommandError {
    fn from(error: WorkerError) -> Self {
        match error {
            WorkerError::Cancelled(reason) => Self::Cancelled(reason),
            other => Self::Cause(Some(Box::new(other))),
        }
    }
}

impl From<JsonError> for CommandError {
    fn from(error: JsonError) -> Self {
        Self::Cause(Some(Box::new(error)))
    }
}

#[cfg(feature = "verify8")]
impl From<Verify8Error> for CommandError {
    fn from(error: Verify8Error) -> Self {
        Self::Cause(Some(Box::new(error)))
    }
}

impl From<crate::mtp_wire::MtpWireError> for CommandError {
    fn from(error: crate::mtp_wire::MtpWireError) -> Self {
        Self::Cause(Some(Box::new(error)))
    }
}

impl From<CallbackFailed> for CommandError {
    fn from(error: CallbackFailed) -> Self {
        Self::Cause(Some(Box::new(error)))
    }
}

/// A probe or callback failed; the worker records it and drains first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallbackFailed;

impl fmt::Display for CallbackFailed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("worker callback failed")
    }
}

impl std::error::Error for CallbackFailed {}

/// `cancelled`: `Ok(true)` cancels; `Err` is a failed probe.
pub type CancelProbe<'a> = &'a mut dyn FnMut() -> std::result::Result<bool, CallbackFailed>;
/// `on_token`: receives token events (and, in V8 mode, admitted records).
pub type TokenSink<'a> = &'a mut dyn FnMut(&Json) -> std::result::Result<(), CallbackFailed>;

fn cancelled_before_submit(
    pacing: Option<&BurstPacing>,
    deadline: Instant,
    cancelled: &mut Option<CancelProbe<'_>>,
) -> Result<bool> {
    loop {
        if let Some(probe) = cancelled.as_mut()
            && probe().map_err(WorkerError::CancelProbe)?
        {
            return Ok(true);
        }
        let Some(pacing) = pacing else {
            return Ok(false);
        };
        let now = Instant::now();
        let wait = pacing
            .delay_at(now)
            .min(deadline.saturating_duration_since(now));
        if wait.is_zero() {
            return Ok(false);
        }
        std::thread::sleep(wait.min(Duration::from_millis(POLL_SLICE_MS as u64)));
    }
}

impl OwnedWorker {
    /// Validates the fixed startup, launches the guardian and admits the single ready event.
    pub fn new(options: WorkerOptions) -> Result<Self> {
        let mut options = options;
        let io = Io {
            control: None,
            status: None,
            input: None,
            output: None,
            error: None,
            guardian_pid: -1,
            worker_pid: -1,
            armed: false,
            worker_exited: false,
            guardian_reaped: false,
            out_eof: false,
            err_eof: false,
            status_eof: false,
            failed: false,
            closed: false,
            starting: true,
            worker_status: -1,
            guardian_status: -1,
            stderr_bytes: 0,
            startup_stderr: Vec::new(),
            pending: Vec::new(),
            notice_bytes: Vec::new(),
            step_timeout: options.step_timeout,
            drain_timeout: options.drain_timeout,
            startup_cancelled: options.startup_cancelled.take(),
        };
        let mut worker = Self {
            io,
            pacing: options.burst_pacing.then(BurstPacing::default),
            options,
            ready_value: Json::Null,
            last_response: Json::Null,
            needs_reset: false,
            broker_epoch: 0,
            broker_command: 0,
            broker_lease: 0,
            broker_position: 0,
            #[cfg(feature = "verify8")]
            verify_previous: Json::Null,
            #[cfg(feature = "verify8")]
            verify_sampling: Json::Null,
            #[cfg(feature = "verify8")]
            verify_seed: 0,
            #[cfg(feature = "verify8")]
            verify_position: 0,
        };
        match worker.start() {
            Ok(()) => Ok(worker),
            Err(StartError::Worker(error)) => {
                worker.invalidate();
                let stderr = std::mem::take(&mut worker.io.startup_stderr);
                Err(if stderr.is_empty() {
                    error
                } else {
                    WorkerError::StartupOutput {
                        source: Box::new(error),
                        stderr,
                    }
                })
            }
            Err(StartError::Cause(cause)) => {
                worker.invalidate();
                let stderr = std::mem::take(&mut worker.io.startup_stderr);
                let error = WorkerError::StartupConfig(cause);
                Err(if stderr.is_empty() {
                    error
                } else {
                    WorkerError::StartupOutput {
                        source: Box::new(error),
                        stderr,
                    }
                })
            }
        }
    }

    fn broker_storage(&self, value: &Json) -> Result<()> {
        require(
            worker_integer(value, "checkpoint_slots")? == u64::from(self.options.checkpoint_slots)
                && worker_integer(value, "checkpoint_combined_budget_bytes")?
                    == self.options.checkpoint_budget_bytes,
            "owned broker checkpoint configuration changed",
        )?;
        let mut total: u64 = 0;
        for name in [
            "checkpoint_host_bytes",
            "checkpoint_store_host_bytes",
            "checkpoint_gpu_bytes",
        ] {
            let bytes = worker_integer(value, name)?;
            require(
                bytes != 0 && bytes <= self.options.checkpoint_budget_bytes.wrapping_sub(total),
                "owned broker storage exceeds budget",
            )?;
            total = total.wrapping_add(bytes);
            if self.broker_epoch != 0 {
                require(
                    bytes == worker_integer(&self.ready_value, name)?,
                    "owned broker prepared storage changed",
                )?;
            }
        }
        Ok(())
    }

    fn broker_identity(&self, value: &Json, command: u64, lease: u64) -> Result<()> {
        require(
            worker_integer(value, "broker_epoch")? == self.broker_epoch
                && worker_integer(value, "command_id")? == command
                && worker_integer(value, "lease_id")? == lease,
            "stale owned broker reply identity",
        )?;
        #[cfg(feature = "branch8")]
        if self.options.branch8 {
            require(
                worker_field(value, "broker_protocol")?.string()? == b"branch8-v1"
                    && worker_integer(value, "broker_batch_width")? == 8,
                "owned branch8 wire identity changed",
            )?;
        }
        Ok(())
    }

    /// The fixed target argv (after path canonicalization), `argv[0]` first.
    fn words(&self) -> Vec<Vec<u8>> {
        fn text(words: &mut Vec<Vec<u8>>, items: &[&str]) {
            words.extend(items.iter().map(|item| item.as_bytes().to_vec()));
        }
        let o = &self.options;
        let path = |p: &Path| p.as_os_str().as_bytes().to_vec();
        let mut words = vec![path(&o.executable)];
        text(&mut words, &["target", "--policy"]);
        words.push(path(&o.policy));
        text(&mut words, &["--library"]);
        words.push(path(&o.library));
        text(
            &mut words,
            &[
                "--allow-full-model",
                "true",
                "--small-gemv-splits",
                "16",
                "--block-gemv-formats",
                "3",
                "--stack-execution",
                &o.stack_execution,
            ],
        );
        #[cfg(feature = "verify8")]
        if o.verify8 {
            #[cfg(feature = "qsa-stage-v8")]
            let lookup_experiments = if o.stage8_present { "false" } else { "true" };
            #[cfg(not(feature = "qsa-stage-v8"))]
            let lookup_experiments = "true";
            text(
                &mut words,
                &[
                    "--known-token-capacity",
                    "8",
                    "--reuse-residual-projections",
                    "true",
                    "--prompt-lookup",
                    "true",
                    "--verification-budget-bytes",
                    "134217728",
                    "--allow-prompt-lookup-experiments",
                    lookup_experiments,
                    "--verification-max-tokens",
                    "8",
                    "--hc-exact",
                    "combined",
                    "--wide-q8-exact",
                    "off",
                    "--wide-q8-many",
                    "off",
                ],
            );
        }
        #[cfg(feature = "branch8")]
        if o.branch8 {
            text(
                &mut words,
                &[
                    "--verification8-branch-reuse",
                    "true",
                    "--worker-broker8",
                    "true",
                ],
            );
        }
        #[cfg(feature = "qsa-stage-v8")]
        if o.stage8_present {
            text(
                &mut words,
                &[
                    "--qsa-stage-v8",
                    if o.stage8_enabled { "on" } else { "off" },
                ],
            );
        }
        #[cfg(feature = "fast-math")]
        if let Some(channel) = o.fast_channel {
            text(
                &mut words,
                &[
                    "--ple-gate-transform",
                    "signed-root",
                    "--qsa-channel-chronological",
                    if channel { "on" } else { "off" },
                ],
            );
        }
        if o.checkpoint_slots != 0 {
            let slots = o.checkpoint_slots.to_string();
            let budget = o.checkpoint_budget_bytes.to_string();
            text(
                &mut words,
                &[
                    "--checkpoint-slots",
                    &slots,
                    "--checkpoint-budget-bytes",
                    &budget,
                    "--worker-broker",
                    "true",
                ],
            );
        }
        if o.checkpoint_ssd {
            let budget = o.checkpoint_ssd_budget_bytes.to_string();
            let floor = o.checkpoint_ssd_floor_bytes.to_string();
            text(
                &mut words,
                &[
                    "--checkpoint-ssd",
                    "on",
                    "--checkpoint-ssd-budget-bytes",
                    &budget,
                    "--checkpoint-ssd-floor-bytes",
                    &floor,
                    "--checkpoint-policy",
                    &o.checkpoint_policy,
                ],
            );
            if !o.checkpoint_ssd_dir.as_os_str().is_empty() {
                words.push(b"--checkpoint-ssd-dir".to_vec());
                words.push(path(&o.checkpoint_ssd_dir));
            }
        }
        if o.overlap_prefill {
            text(&mut words, &["--overlap-prefill", "on"]);
        }
        if let Some(swiglu) = o.swiglu_fused {
            text(
                &mut words,
                &["--swiglu-fused", if swiglu { "on" } else { "off" }],
            );
        }
        if let Some(ref rg4) = o.q8_gemv_rg4 {
            text(&mut words, &["--q8-gemv-rg4", rg4]);
        }
        if let Some(ref msg) = o.q8_three_msg {
            text(&mut words, &["--q8-three-msg", msg]);
        }
        if let Some(union_moe) = o.moe_three_union {
            text(
                &mut words,
                &["--moe-three-union", if union_moe { "on" } else { "off" }],
            );
        }
        if let Some(ref lookup) = o.mtp_lookup {
            text(&mut words, &["--mtp-lookup", lookup]);
        }
        if let Some(match_width) = o.mtp_lookup_match {
            let match_str = match_width.to_string();
            text(&mut words, &["--mtp-lookup-match", &match_str]);
        }
        if let Some(ref m64) = o.prefill_m64 {
            text(&mut words, &["--prefill-m64", m64]);
        }
        if let Some(multiseg) = o.prefill_multiseg {
            text(
                &mut words,
                &["--prefill-multiseg", if multiseg { "on" } else { "off" }],
            );
        }
        if let Some(swizzle) = o.prefill_swizzle {
            text(
                &mut words,
                &["--prefill-swizzle", if swizzle { "on" } else { "off" }],
            );
        }
        if let Some(qsa_fused) = o.qsa_prefill_fused {
            text(
                &mut words,
                &["--qsa-prefill-fused", if qsa_fused { "on" } else { "off" }],
            );
        }
        if let Some(hc_fused) = o.hc_consumer_fused {
            text(
                &mut words,
                &["--hc-consumer-fused", if hc_fused { "on" } else { "off" }],
            );
        }
        if let Some(ple_ov) = o.ple_overlap {
            text(
                &mut words,
                &["--ple-overlap", if ple_ov { "on" } else { "off" }],
            );
        }
        for (flag, enabled) in [
            ("--native-host-logit-views", o.host_logit_views),
            ("--native-logit-topk", o.logit_topk),
            ("--native-ple-async-gather", o.ple_async_gather),
            ("--native-mtp-fast-round", o.mtp_fast_round),
            ("--native-prefill-router-fast", o.prefill_router_fast),
            ("--native-dflash2-defer-context", o.dflash2_defer_context),
        ] {
            if let Some(enabled) = enabled {
                text(&mut words, &[flag, if enabled { "on" } else { "off" }]);
            }
        }
        if o.prepared_validation {
            text(&mut words, &["--native-prepared-validation", "on"]);
        }
        if let Some(ref profile) = o.verify_cost_profile {
            text(&mut words, &["--native-verify-cost-profile"]);
            words.push(path(profile));
        }
        if let Some(ref splits) = o.verify_m8_splits {
            text(&mut words, &["--native-verify-m8-splits"]);
            words.push(path(splits));
        }
        if o.dflash2_draft_policy == "auto" {
            text(&mut words, &["--native-dflash2-draft-policy", "auto"]);
        }
        if let Some(ref rows_mode) = o.dflash2_verify_rows {
            text(&mut words, &["--native-dflash2-verify-rows", rows_mode]);
        }
        if let Some(spc) = o.shared_prefix_cache {
            text(
                &mut words,
                &["--shared-prefix-cache", if spc { "on" } else { "off" }],
            );
        }
        if o.scalar_channel {
            text(
                &mut words,
                &[
                    "--hc-exact",
                    "combined",
                    "--wide-q8-exact",
                    "off",
                    "--wide-q8-many",
                    "off",
                ],
            );
            if has_speculative(o) {
                let cap =
                    o.known_token_capacity
                        .unwrap_or(if o.prefill_input512 { 512 } else { 8 });
                let cap_str = cap.to_string();
                if o.prefill_input512 {
                    let dense = o.prefill_dense.as_deref().unwrap_or("f16");
                    text(
                        &mut words,
                        &[
                            "--known-token-capacity",
                            &cap_str,
                            "--reuse-residual-projections",
                            "true",
                            "--native-prefill-chunks",
                            "fused",
                            "--native-prefill-dense",
                            dense,
                        ],
                    );
                } else {
                    text(
                        &mut words,
                        &[
                            "--known-token-capacity",
                            &cap_str,
                            "--reuse-residual-projections",
                            "true",
                            "--known-qsa-rows",
                            "q8",
                        ],
                    );
                }
                if !o.dflash2_dir.as_os_str().is_empty() {
                    text(&mut words, &["--drafter", "dflash2", "--dflash2-dir"]);
                    words.push(path(&o.dflash2_dir));
                    if let Some(ref prec) = o.dflash2_precision {
                        text(&mut words, &["--dflash2-precision", prec]);
                    }
                    if let Some(max_drafts) = o.dflash2_max_drafts {
                        let s = max_drafts.to_string();
                        text(&mut words, &["--dflash2-max-drafts", &s]);
                    }
                    if let Some(vocab_slice) = o.dflash2_vocab_slice {
                        let s = vocab_slice.to_string();
                        text(&mut words, &["--dflash2-vocab-slice", &s]);
                    }
                } else {
                    words.push(b"--mtp-sidecar".to_vec());
                    words.push(path(&o.mtp_sidecar));
                }
                text(&mut words, &["--mtp-lookahead-anchor", "on"]);
                if let Some(decode) = o.raw_decode {
                    text(
                        &mut words,
                        &["--raw-decode", if decode { "on" } else { "off" }],
                    );
                }
                if let Some(draft_slice) = o.mtp_draft_slice {
                    text(
                        &mut words,
                        &[
                            "--native-mtp-draft-slice",
                            if draft_slice { "on" } else { "off" },
                        ],
                    );
                }
                if o.prefill_final_logits {
                    text(&mut words, &["--native-prefill-final-logits", "on"]);
                }
                if let Some(hc_rows) = o.prefill_hc_rows {
                    text(
                        &mut words,
                        &[
                            "--native-prefill-hc-rows",
                            if hc_rows { "on" } else { "off" },
                        ],
                    );
                }
                if let Some(router) = o.prefill_router_window {
                    text(
                        &mut words,
                        &[
                            "--native-prefill-router-window",
                            if router { "on" } else { "off" },
                        ],
                    );
                }
                if let Some(hc_down) = o.prefill_hc_down_window {
                    text(
                        &mut words,
                        &[
                            "--native-prefill-hc-down-window",
                            if hc_down { "on" } else { "off" },
                        ],
                    );
                }
                if let Some(qsa) = o.qsa_sparse {
                    text(
                        &mut words,
                        &["--native-qsa-sparse", if qsa { "on" } else { "off" }],
                    );
                }
                if let Some(gdn) = o.gdn_ported {
                    text(
                        &mut words,
                        &["--native-gdn-ported", if gdn { "on" } else { "off" }],
                    );
                }
                if let Some(gdn_pre) = o.gdn_prefill_ported {
                    text(
                        &mut words,
                        &[
                            "--native-gdn-prefill-ported",
                            if gdn_pre { "on" } else { "off" },
                        ],
                    );
                }
                if let Some(hc_down_f32) = o.prefill_hc_down_f32 {
                    text(
                        &mut words,
                        &[
                            "--native-prefill-hc-down-f32",
                            if hc_down_f32 { "on" } else { "off" },
                        ],
                    );
                }
                if o.gdn_old_order_batch {
                    text(&mut words, &["--native-gdn-old-order-batch", "on"]);
                }
                if let Some(ref grouped) = o.prefill_grouped {
                    text(&mut words, &["--native-prefill-grouped", grouped]);
                }
                if let Some(vnax) = o.verify_nax {
                    text(
                        &mut words,
                        &["--verify-nax", if vnax { "on" } else { "off" }],
                    );
                }
                if let Some(vnax_m8) = o.verify_nax_m8 {
                    text(
                        &mut words,
                        &["--verify-nax-m8", if vnax_m8 { "on" } else { "off" }],
                    );
                }
                if let Some(vhcg) = o.verify_hc_gdn_fused {
                    text(
                        &mut words,
                        &["--verify-hc-gdn-fused", if vhcg { "on" } else { "off" }],
                    );
                }
                if let Some(depth) = o.mtp_depth {
                    let depth_str = depth.to_string();
                    text(&mut words, &["--mtp-depth", &depth_str]);
                }
                if o.mtp_request_carry {
                    text(&mut words, &["--mtp-request-carry", "on"]);
                }
                if let Some(packet) = o.mtp_output_packet {
                    let packet_str = packet.to_string();
                    text(&mut words, &["--mtp-output-packet", &packet_str]);
                }
                if let Some(ref attention) = o.mtp_draft_attention {
                    text(&mut words, &["--mtp-draft-attention", attention]);
                }
                if let Some(concurrent) = o.concurrent_dispatch {
                    text(
                        &mut words,
                        &[
                            "--concurrent-dispatch",
                            if concurrent { "on" } else { "off" },
                        ],
                    );
                }
                if let Some(q8_rows) = o.q8_three_rows {
                    text(
                        &mut words,
                        &["--q8-three-rows", if q8_rows { "on" } else { "off" }],
                    );
                }
            }
        }
        if let Some(ref pack) = o.derived_pack {
            text(&mut words, &["--derived-pack"]);
            words.push(path(pack));
            if let Some(workers) = o.derived_pack_auth_workers {
                let workers_str = workers.to_string();
                text(&mut words, &["--derived-pack-auth-workers", &workers_str]);
            }
            if let Some(ref mode) = o.derived_pack_trust_mode {
                text(&mut words, &["--derived-pack-trust-mode", mode]);
            }
        }

        for shard in &o.shards {
            words.push(path(shard));
        }
        words
    }

    /// `argv` the target would receive, for differential checks.
    pub fn argv(&self) -> Vec<Vec<u8>> {
        self.words()
    }

    fn start(&mut self) -> std::result::Result<(), StartError> {
        let o = &self.options;
        require(
            o.stack_execution == "direct" || o.stack_execution == "recurrent-replay",
            "unsupported owned startup execution mode",
        )?;
        require(
            o.checkpoint_slots <= 4
                && o.checkpoint_budget_bytes <= (512 << 20)
                && (o.checkpoint_slots != 0) == (o.checkpoint_budget_bytes != 0)
                && (o.checkpoint_slots == 0 || o.stack_execution == "direct"),
            "owned broker requires paired bounded direct preparation",
        )?;
        let executable = fixed_file(&o.executable)?;
        self.options.executable = executable;
        let o = &self.options;
        let expected_name = self.expected_executable_name()?;
        let executable_ok = o.executable.file_name().is_some_and(|name| {
            name.as_bytes() == expected_name.as_bytes()
                || (expected_name == "tokspur-native"
                    && name.as_bytes() == b"scalar-channel-target")
                || (expected_name == "scalar-channel-target"
                    && name.as_bytes() == b"tokspur-native")
        }) && CString::new(o.executable.as_os_str().as_bytes())
            .is_ok_and(|path| process::is_executable(&path));
        require(executable_ok, self.executable_refusal())?;
        self.options.policy = fixed_file(&self.options.policy)?;
        self.options.library = fixed_file(&self.options.library)?;
        require(
            self.options.mtp_sidecar.as_os_str().is_empty()
                || self.options.dflash2_dir.as_os_str().is_empty(),
            "cannot combine MTP sidecar and DFlash2 drafter",
        )?;
        if let Some(ref prec) = self.options.dflash2_precision {
            require(
                !self.options.dflash2_dir.as_os_str().is_empty()
                    && matches!(prec.as_str(), "q8_0" | "bf16"),
                "dflash2 precision requires --dflash2-dir and q8_0 or bf16",
            )?;
        }
        if !self.options.mtp_sidecar.as_os_str().is_empty() {
            self.options.mtp_sidecar = fixed_file(&self.options.mtp_sidecar)?;
        }
        if !self.options.dflash2_dir.as_os_str().is_empty() {
            self.options.dflash2_dir = fixed_dir(&self.options.dflash2_dir)?;
        }
        require(
            self.options.shards.len() == 4,
            "owned target requires four fixed shards",
        )?;
        let mut distinct = BTreeSet::new();
        for i in 0..self.options.shards.len() {
            let shard = fixed_file(&self.options.shards[i])?;
            require(
                distinct.insert(shard.clone()),
                "owned target shard paths must be distinct",
            )?;
            self.options.shards[i] = shard;
        }
        for timeout in [
            self.options.startup_timeout,
            self.options.step_timeout,
            self.options.drain_timeout,
        ] {
            require(
                !timeout.is_zero() && timeout <= Duration::from_secs(24 * 3600),
                "invalid worker deadline bound",
            )?;
        }
        let ceiling = process::open_files_limit()
            .ok()
            .filter(|limit| *limit <= 1_048_576)
            .ok_or(WorkerError::Refused(
                "cannot bound inherited worker descriptors",
            ))?;
        let argv = self.words();
        // No loader/preload, Python, remote-model or runtime-selection variables.
        let environment = [
            b"PATH=/usr/bin:/bin:/usr/sbin:/sbin".to_vec(),
            b"LANG=C".to_vec(),
            b"LC_ALL=C".to_vec(),
        ];
        let (control, control_child) = sysfd::socket_pair()
            .map_err(|e| pair_error(e, "cannot create worker control stream"))?;
        let (status, status_child) =
            sysfd::pipe_pair().map_err(|e| pair_error(e, "cannot create worker pipe"))?;
        let (input, input_child) = sysfd::socket_pair()
            .map_err(|e| pair_error(e, "cannot create worker control stream"))?;
        let (output, output_child) =
            sysfd::pipe_pair().map_err(|e| pair_error(e, "cannot create worker pipe"))?;
        let (error, error_child) =
            sysfd::pipe_pair().map_err(|e| pair_error(e, "cannot create worker pipe"))?;
        let (gate_read, gate_write) =
            sysfd::pipe_pair().map_err(|e| pair_error(e, "cannot create worker pipe"))?;
        // Parent ends are [0]; pipes return (read, write).
        let fds = GuardianFds {
            control: control_child.as_raw_fd(),
            status: status_child.as_raw_fd(),
            input: input_child.as_raw_fd(),
            output: output_child.as_raw_fd(),
            error: error_child.as_raw_fd(),
            gate_read: gate_read.as_raw_fd(),
            gate_write: gate_write.as_raw_fd(),
        };
        let plan = GuardianPlan::new(
            fds,
            ceiling as i32,
            process::current_pid(),
            &argv,
            &environment,
        )
        .map_err(|error| StartError::Cause(Box::new(error)))?;
        self.io.control = Some(control);
        self.io.status = Some(status);
        self.io.input = Some(input);
        self.io.output = Some(output);
        self.io.error = Some(error);
        let guardian = process::fork_guardian(&plan);
        drop((
            control_child,
            status_child,
            input_child,
            output_child,
            error_child,
            gate_read,
            gate_write,
        ));
        self.io.guardian_pid = guardian.map_err(WorkerError::GuardianLaunch)?;
        for fd in [
            &self.io.control,
            &self.io.status,
            &self.io.input,
            &self.io.output,
            &self.io.error,
        ]
        .into_iter()
        .flatten()
        {
            sysfd::make_nonblocking(fd.as_fd()).map_err(WorkerError::PipeConfigure)?;
        }
        let deadline = Instant::now() + self.options.startup_timeout;
        while !self.io.armed {
            require(
                Instant::now() < deadline && !self.io.status_eof,
                "owned guardian did not arm before the startup deadline",
            )?;
            self.io.pump(deadline, false, POLL_SLICE_MS)?;
        }
        self.io.send_control(b'L', deadline)?;
        while self.io.worker_pid < 0 {
            require(
                Instant::now() < deadline && !self.io.status_eof,
                "owned guardian did not identify its worker",
            )?;
            self.io.pump(deadline, false, POLL_SLICE_MS)?;
        }
        self.io.send_control(b'E', deadline)?;
        self.ready_value = self.io.next(deadline)?;
        require(
            self.ready_value.kind() == Kind::Object && event_is(&self.ready_value, "ready")?,
            "owned target did not emit a single ready event",
        )?;
        #[cfg(feature = "verify8")]
        if self.options.verify8 {
            #[cfg(feature = "branch8")]
            let branch8 = self.options.branch8;
            #[cfg(not(feature = "branch8"))]
            let branch8 = false;
            if branch8 {
                #[cfg(feature = "branch8")]
                crate::verify8::validate_branch8_ready(&self.ready_value)?;
                self.verify_previous = self.ready_value.clone();
            } else {
                crate::verify8::validate_verify8_ready(&self.ready_value)?;
            }
        }
        #[cfg(feature = "qsa-stage-v8")]
        {
            let staged = self.ready_value.find("qsa_stage_v8_prepared")?;
            let matches = staged.is_some() == self.options.stage8_present
                && match staged {
                    None => true,
                    Some(staged) => {
                        staged.boolean()?
                            && worker_field(&self.ready_value, "qsa_stage_v8_enabled")?.boolean()?
                                == self.options.stage8_enabled
                    }
                };
            require(
                matches,
                "owned staged-QSA readiness does not match the exact requested startup arm",
            )?;
        }
        #[cfg(feature = "fast-math")]
        require(
            crate::verify8::verify8_fast_math(&self.ready_value)? == self.options.fast_channel,
            "owned fast-math readiness does not match the exact requested startup arm",
        )?;
        if self.options.scalar_channel {
            if has_speculative(&self.options) {
                let default_attn = if !self.options.dflash2_dir.as_os_str().is_empty() {
                    "split16"
                } else {
                    "serial"
                };
                let default_depth = if !self.options.dflash2_dir.as_os_str().is_empty() {
                    7
                } else {
                    1
                };
                crate::mtp_wire::validate_mtp_ready(
                    &self.ready_value,
                    self.options
                        .mtp_draft_attention
                        .as_deref()
                        .unwrap_or(default_attn),
                    self.options.mtp_request_carry,
                    self.options.mtp_depth.unwrap_or(default_depth),
                )
                .map_err(WorkerError::MtpReady)?;
                if let Some(ref prec) = self.options.dflash2_precision {
                    require(
                        worker_field(&self.ready_value, "native_dflash2_precision")?.string()?
                            == prec.as_bytes(),
                        "owned DFlash2 precision readiness differs",
                    )?;
                }
            }
            self.last_response = self.ready_value.clone();
        }
        if let Some(concurrent) = self.options.concurrent_dispatch {
            require(
                worker_integer(&self.ready_value, "concurrent_dispatch_schema")? == 1,
                "owned concurrent dispatch schema must be 1",
            )?;
            let expected_mode = if concurrent {
                b"on".as_slice()
            } else {
                b"off".as_slice()
            };
            require(
                worker_field(&self.ready_value, "concurrent_dispatch_mode")?.string()?
                    == expected_mode,
                "owned concurrent dispatch mode does not match requested option",
            )?;
            require(
                worker_integer(&self.ready_value, "concurrent_dispatch_barriers")? == 0,
                "owned concurrent dispatch barriers at startup must be zero",
            )?;
        } else if let Some(field) = self.ready_value.find("concurrent_dispatch_schema")? {
            require(
                field.is_null(),
                "unexpected owned concurrent dispatch readiness",
            )?;
        }
        if self.options.prepared_validation {
            require(
                worker_field(&self.ready_value, "native_prepared_validation")?.boolean()?,
                "prepared validation startup differs",
            )?;
        }
        if self.options.verify_cost_profile.is_some() {
            require(
                worker_field(&self.ready_value, "native_measured_controller")?.boolean()?
                    && worker_integer(&self.ready_value, "native_measured_timing_schema")? == 1,
                "measured controller startup differs",
            )?;
            let digest =
                worker_field(&self.ready_value, "native_controller_policy_digest")?.array()?;
            require(
                digest.len() == 32
                    && digest
                        .iter()
                        .all(|b| b.unsigned_integer().is_ok_and(|v| v <= 255)),
                "invalid negotiated controller digest",
            )?;
        }
        if self.options.dflash2_draft_policy == "auto" {
            require(
                worker_field(&self.ready_value, "native_dflash2_draft_policy")?.string()?
                    == b"auto",
                "drafting policy startup differs",
            )?;
        }
        if self.options.verify_m8_splits.is_some() {
            require(
                worker_field(&self.ready_value, "native_target_m8_split_digest")?
                    .string()?
                    .len()
                    == 64,
                "M8 split policy startup differs",
            )?;
        }
        for (name, requested) in [
            ("native_host_logit_views", self.options.host_logit_views),
            ("native_logit_topk", self.options.logit_topk),
            ("native_ple_async_gather", self.options.ple_async_gather),
            ("native_mtp_fast_round", self.options.mtp_fast_round),
            (
                "native_prefill_router_fast",
                self.options.prefill_router_fast,
            ),
            (
                "native_verify_hc_gdn_fused",
                self.options.verify_hc_gdn_fused,
            ),
            (
                "native_dflash2_defer_context",
                self.options.dflash2_defer_context,
            ),
            ("native_prefill_gdn_tb12", self.options.gdn_prefill_ported),
        ] {
            if let Some(requested) = requested {
                require(
                    worker_field(&self.ready_value, name)?.boolean()? == requested,
                    "owned host synchronisation readiness does not match requested option",
                )?;
            }
        }
        for (name, requested) in [
            ("native_dflash2_max_drafts", self.options.dflash2_max_drafts),
            (
                "native_dflash2_vocab_slice",
                self.options.dflash2_vocab_slice,
            ),
        ] {
            if let Some(requested) = requested {
                require(
                    worker_integer(&self.ready_value, name)? == u64::from(requested),
                    "owned DFlash2 numeric readiness does not match requested option",
                )?;
            }
        }
        if self.options.checkpoint_slots != 0 {
            require(
                worker_field(&self.ready_value, "checkpoint_broker")?.boolean()?,
                "owned target did not prepare broker controls",
            )?;
            self.broker_storage(&self.ready_value)?;
            self.broker_epoch = worker_integer(&self.ready_value, "broker_epoch")?;
            require(self.broker_epoch != 0, "owned broker epoch is invalid")?;
        } else if let Some(broker) = self.ready_value.find("checkpoint_broker")? {
            require(!broker.boolean()?, "unexpected owned broker preparation")?;
        }
        self.io.idle_check()?;
        self.io.starting = false;
        Ok(())
    }

    /// The executable basename the explicit private startup mode requires.
    fn expected_executable_name(&self) -> std::result::Result<&'static str, StartError> {
        #[cfg(feature = "verify8")]
        {
            let o = &self.options;
            if scalar_channel_transport(o) {
                return Ok("scalar-channel-target");
            }
            #[cfg(feature = "branch8")]
            let branch_transport =
                o.branch8 && o.verify8 && o.checkpoint_slots != 0 && o.checkpoint_budget_bytes != 0;
            #[cfg(not(feature = "branch8"))]
            let branch_transport = false;
            #[cfg(feature = "branch8")]
            require(
                !o.branch8 || (branch_transport && o.stack_execution == "direct"),
                "branch8 worker requires explicit V8/history/direct consent",
            )?;
            #[cfg(feature = "qsa-stage-v8")]
            require(
                (!o.stage8_enabled || o.stage8_present)
                    && (!o.stage8_present || (branch_transport && o.stack_execution == "direct")),
                "staged HTTP worker requires selector presence and explicit branch8/V8/direct/history consent",
            )?;
            #[cfg(feature = "fast-math")]
            require(
                o.fast_channel.is_none()
                    || (branch_transport && o.stage8_present && o.stage8_enabled),
                "fast HTTP math requires explicit staged-ON branch8/V8/history consent",
            )?;
            require(
                !o.verify8
                    || (o.stack_execution == "direct"
                        && (branch_transport
                            || (o.checkpoint_slots == 0 && o.checkpoint_budget_bytes == 0))),
                "private HTTP V8 requires fixed direct execution without checkpoints",
            )?;
            Ok(if branch_transport {
                "branch8-target"
            } else if o.verify8 {
                "verify8-target"
            } else {
                "tokspur-native"
            })
        }
        #[cfg(not(feature = "verify8"))]
        {
            Ok("tokspur-native")
        }
    }

    fn executable_refusal(&self) -> &'static str {
        if cfg!(feature = "verify8") {
            "owned worker executable does not match the explicit private startup mode"
        } else {
            "only an explicit owned tokspur-native executable is admitted"
        }
    }

    /// `invalidate()`: fail, require a reset, close the lifetime endpoint and drain. Shutdown failures are swallowed, as in the C++.
    fn invalidate(&mut self) {
        self.io.failed = true;
        self.needs_reset = true;
        self.io.control = None;
        if self.io.shutdown().is_err() {
            // The guardian remains responsible; admission is already failed.
        }
    }

    pub fn ready(&self) -> &Json {
        &self.ready_value
    }

    /// No active command, no failure/EOF, and nothing readable on the status pipe.
    pub fn alive(&self) -> bool {
        let io = &self.io;
        if io.failed || io.closed || io.worker_exited || io.out_eof || io.status_eof {
            return false;
        }
        let mut status = [PollFd {
            fd: raw(&io.status, false),
            events: POLLIN,
            revents: 0,
        }];
        sysfd::poll(&mut status, 0).is_ok_and(|ready| ready == 0)
    }

    pub fn options(&self) -> &WorkerOptions {
        &self.options
    }

    pub fn last_response(&self) -> &Json {
        &self.last_response
    }

    pub fn worker_pid(&self) -> Option<u32> {
        if self.io.worker_pid > 0 {
            Some(self.io.worker_pid as u32)
        } else {
            None
        }
    }

    pub fn guardian_pid(&self) -> Option<u32> {
        if self.io.guardian_pid > 0 {
            Some(self.io.guardian_pid as u32)
        } else {
            None
        }
    }

    pub fn abort_mtp_generation(&mut self, deadline: Instant) -> Result<()> {
        if self.io.failed || self.io.closed {
            return Ok(());
        }
        let generation = match self.last_response.find("native_mtp_generation") {
            Ok(Some(g)) => g,
            _ => return Ok(()),
        };
        let active = worker_field(generation, "active")
            .ok()
            .and_then(|a| a.boolean().ok())
            .unwrap_or(false);
        if !active {
            return Ok(());
        }
        let id = worker_integer(generation, "id")?;
        let job = Json::new_object([
            ("tokens", Json::Array(Vec::new())),
            ("generate", Json::integer(0)),
            ("reset", Json::Bool(false)),
            (
                "mtp_generation",
                Json::new_object([("id", Json::integer(id)), ("abort", Json::Bool(true))]),
            ),
        ]);
        let result = self.request(&job, deadline, None, None);
        if result.is_err() {
            self.invalidate();
            return result.map(|_| ());
        }
        self.needs_reset = true;
        Ok(())
    }

    /// Throws unless the exact guardian wait, worker exit and pipe drain are verified.
    pub fn close(&mut self) -> Result<()> {
        self.io.shutdown()
    }

    /// Validates, submits and drains one command.
    pub fn request(
        &mut self,
        body: &Json,
        deadline: Instant,
        mut cancelled: Option<CancelProbe<'_>>,
        mut on_token: Option<TokenSink<'_>>,
    ) -> Result<Json> {
        // Validate admission before a single request byte is written.
        require(
            !self.io.failed && !self.io.closed,
            "owned worker context is unavailable; no automatic restart",
        )?;
        require(
            body.kind() == Kind::Object,
            "owned worker command must be an object",
        )?;
        let admitted = match self.admit(body) {
            Ok(admitted) => admitted,
            Err(error) if error.is_refused() => return Err(error),
            Err(other) => return Err(WorkerError::EncodeBeforeSubmit(Box::new(other))),
        };
        let pacing = self.pacing.as_ref().filter(|_| {
            admitted.control || admitted.input_count != 0 || admitted.output_count != 0
        });
        if cancelled_before_submit(pacing, deadline, &mut cancelled)? {
            self.needs_reset = true;
            return Err(WorkerError::Cancelled(
                "owned request cancelled before submission",
            ));
        }
        let pace_work = pacing.is_some();
        let deadline = deadline.min(Instant::now() + self.options.step_timeout);
        require(
            Instant::now() < deadline,
            "owned request deadline expired before submission",
        )?;
        let mut command = self.broker_command;
        let mut lease = self.broker_lease;
        let before = self.broker_position;
        #[cfg(feature = "verify8")]
        let verify_inputs = if self.options.verify8 && !admitted.control && !admitted.mtp_abort {
            let policy = if admitted.reset {
                worker_field(body, "sampling")?.clone()
            } else {
                self.verify_sampling.clone()
            };
            let seed = if admitted.reset {
                worker_integer(body, "seed")?
            } else {
                self.verify_seed
            };
            Some((policy, seed))
        } else {
            None
        };
        // The gate reads the readiness and previous reply; snapshots keep it independent of the worker state the command updates.
        let mtp_snapshots = if has_speculative(&self.options) {
            Some((self.ready_value.clone(), self.last_response.clone()))
        } else {
            None
        };
        let mut mtp_reply = match &mtp_snapshots {
            Some((ready, previous)) => Some(
                crate::mtp_wire::MtpWireReply::new(ready, previous, body, false)
                    .map_err(WorkerError::MtpBeforeSubmit)?,
            ),
            None => None,
        };
        #[cfg(feature = "verify8")]
        let snapshots = verify_inputs
            .as_ref()
            .map(|_| (self.ready_value.clone(), self.verify_previous.clone()));
        #[cfg(feature = "verify8")]
        let mut verify = match (&verify_inputs, &snapshots) {
            (Some((policy, seed)), Some((ready, previous))) => {
                #[cfg(feature = "branch8")]
                let branch8 = self.options.branch8;
                #[cfg(not(feature = "branch8"))]
                let branch8 = false;
                Some(Verify8Reply::new(Verify8Context {
                    ready,
                    previous,
                    job: body,
                    before: self.verify_position,
                    policy,
                    seed: *seed,
                    reset_resync: false,
                    branch8,
                })?)
            }
            _ => None,
        };
        let mut wire = body.clone();
        if self.broker_epoch != 0 {
            require(command < u64::MAX, "owned broker command space exhausted")?;
            command += 1;
            if admitted.begin || admitted.reset {
                require(lease < u64::MAX, "owned broker lease space exhausted")?;
                lease += 1;
            }
            let mut fields = body.object()?.clone();
            fields.push((b"broker_epoch".to_vec(), Json::integer(self.broker_epoch)));
            fields.push((b"command_id".to_vec(), Json::integer(command)));
            fields.push((b"lease_id".to_vec(), Json::integer(lease)));
            fields.push((
                b"expected_prefix".to_vec(),
                Json::integer(u64::from(before)),
            ));
            #[cfg(feature = "branch8")]
            if self.options.branch8 {
                fields.push((b"broker_protocol".to_vec(), Json::from("branch8-v1")));
                fields.push((b"broker_batch_width".to_vec(), Json::integer(8)));
            }
            wire = Json::Object(fields);
        }
        let mut bytes = wire.dump_limited(LINE_LIMIT - 1)?;
        bytes.push(b'\n');
        let submission = Submission {
            body,
            bytes: &bytes,
            admitted: &admitted,
            command,
            lease,
            before,
            deadline,
            mtp_reply: mtp_reply.as_mut(),
            #[cfg(feature = "verify8")]
            verify: verify.as_mut(),
            #[cfg(feature = "verify8")]
            verify_inputs: verify_inputs.as_ref(),
            #[cfg(not(feature = "verify8"))]
            _snapshots: std::marker::PhantomData,
        };
        let started = pace_work.then(Instant::now);
        let outcome = self.submit(submission, &mut cancelled, &mut on_token);
        if let (Some(pacing), Some(started)) = (&mut self.pacing, started) {
            pacing.charge(started, Instant::now());
        }
        match outcome {
            Ok(value) => Ok(value),
            Err(CommandError::Cancelled(reason)) => Err(WorkerError::Cancelled(reason)),
            Err(CommandError::Cause(cause)) => {
                self.invalidate();
                Err(WorkerError::CommandInvalidated(cause))
            }
        }
    }

    /// The command-shape admission block (C++ `try` before submission).
    fn admit(&self, body: &Json) -> Result<Admitted> {
        let tokens = body.find("tokens")?;
        let mut admitted = Admitted {
            output_count: 0,
            input_count: 0,
            reset: false,
            begin: false,
            allow_restore: false,
            control: false,
            operation: Vec::new(),
            mtp_abort: false,
        };
        if let Some(op) = body.find("op")? {
            require(
                self.broker_epoch != 0,
                "owned broker controls require explicit startup preparation",
            )?;
            admitted.operation = op.string()?.to_vec();
            admitted.control = true;
            admitted.begin = admitted.operation == b"branch_begin";
            require(
                admitted.begin || admitted.operation == b"branch_capture",
                "unknown owned broker operation",
            )?;
            if admitted.begin {
                worker_keys(
                    body,
                    &[
                        "op",
                        "prompt_ids",
                        "generation_requested",
                        "allow_restore",
                        "sampling",
                        "seed",
                    ],
                )?;
                let prompt = worker_field(body, "prompt_ids")?.array()?;
                require(
                    !prompt.is_empty() && prompt.len() <= 65536,
                    "owned broker future prompt bound",
                )?;
                for token in prompt {
                    require(
                        token.unsigned_integer()? < 248_320,
                        "owned broker future token bound",
                    )?;
                }
                worker_field(body, "generation_requested")?.boolean()?;
                admitted.allow_restore = worker_field(body, "allow_restore")?.boolean()?;
                require(
                    !admitted.allow_restore || self.broker_lease != 0,
                    "owned broker reuse requires an established lease",
                )?;
                let policy = worker_field(body, "sampling")?;
                worker_keys(policy, &["temperature", "top_k", "top_p", "min_p"])?;
                let top_p = worker_number(worker_field(policy, "top_p")?)?;
                let min_p = worker_number(worker_field(policy, "min_p")?)?;
                require(
                    worker_number(worker_field(policy, "temperature")?)? == 0.0
                        && worker_integer(policy, "top_k")? <= 248_320
                        && top_p > 0.0
                        && top_p <= 1.0
                        && (0.0..=1.0).contains(&min_p),
                    "owned broker requires bounded greedy sampling",
                )?;
                worker_integer(body, "seed")?;
            } else {
                worker_keys(body, &["op"])?;
            }
        } else {
            let generate = body.find("generate")?;
            let clear = body.find("reset")?;
            let (Some(tokens), Some(generate), Some(clear)) = (tokens, generate, clear) else {
                return Err(WorkerError::Refused("owned worker command shape refused"));
            };
            require(
                tokens.kind() == Kind::Array
                    && generate.kind() == Kind::Number
                    && clear.kind() == Kind::Boolean,
                "owned worker command shape refused",
            )?;
            admitted.output_count = generate.unsigned_integer()?;
            admitted.input_count = tokens.array()?.len() as u64;
            admitted.reset = clear.boolean()?;
            for token in tokens.array()? {
                require(
                    token.unsigned_integer()? < 248_320,
                    "owned input token is outside vocabulary",
                )?;
            }
            let (inputs, outputs) = (admitted.input_count, admitted.output_count);
            let mut mtp_abort = false;
            if let Some(generation) = body.find("mtp_generation")? {
                if let Some(abort) = generation.find("abort")? {
                    mtp_abort = abort.boolean()?;
                } else {
                    worker_keys(
                        generation,
                        &[
                            "id",
                            "abort",
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
                }
            }
            admitted.mtp_abort = mtp_abort;
            if has_speculative(&self.options) {
                let input_limit = crate::mtp_wire::mtp_input_packet(&self.ready_value)? as u64;
                let output_limit = crate::mtp_wire::mtp_output_packet(&self.ready_value)? as u64;
                require(
                    (inputs >= 1 && inputs <= input_limit && outputs == 0)
                        || (inputs == 0 && outputs >= 1 && outputs <= output_limit)
                        || (mtp_abort && inputs == 0 && outputs == 0 && !admitted.reset),
                    "MTP commands exceed the negotiated input or output bound",
                )?;
                if self.options.scalar_channel {
                    if self.options.mtp_request_carry {
                        worker_keys(
                            body,
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
                        worker_keys(
                            body,
                            &[
                                "tokens", "generate", "reset", "stop_ids", "sampling", "seed",
                            ],
                        )?;
                    }
                }
            } else if mtp_abort {
                require(
                    inputs == 0 && outputs == 0 && !admitted.reset,
                    "MTP abort must have zero tokens and no reset",
                )?;
            } else if self.v8_mode() {
                require(
                    inputs <= 8 && outputs <= 8 && (inputs != 0) != (outputs != 0),
                    "private HTTP V8 commands require <=8 input OR output tokens",
                )?;
            } else {
                require(
                    inputs <= 1 && outputs <= 1 && inputs + outputs <= 1,
                    "owned worker commands are bounded to one model step",
                )?;
            }
            if self.broker_epoch != 0 {
                if self.branch8_mode() {
                    worker_keys(
                        body,
                        &[
                            "tokens",
                            "generate",
                            "reset",
                            "stop_ids",
                            "sampling",
                            "seed",
                            "prompt_lookup",
                            "mtp_generation",
                        ],
                    )?;
                } else {
                    worker_keys(
                        body,
                        &[
                            "tokens",
                            "generate",
                            "reset",
                            "stop_ids",
                            "sampling",
                            "seed",
                            "mtp_generation",
                        ],
                    )?;
                }
            }
        }
        require(
            !self.needs_reset
                || admitted.reset
                || (admitted.begin && !admitted.allow_restore)
                || admitted.mtp_abort,
            "cancelled/ambiguous context requires an explicit cold reset",
        )?;
        require(
            self.broker_epoch == 0 || admitted.begin || admitted.reset || self.broker_lease != 0,
            "owned broker requires an initial reset or begin",
        )?;
        Ok(admitted)
    }

    fn v8_mode(&self) -> bool {
        #[cfg(feature = "verify8")]
        {
            self.options.verify8
        }
        #[cfg(not(feature = "verify8"))]
        {
            false
        }
    }

    fn branch8_mode(&self) -> bool {
        #[cfg(feature = "branch8")]
        {
            self.options.branch8
        }
        #[cfg(not(feature = "branch8"))]
        {
            false
        }
    }
}

/// One encoded command and everything its drain needs.
struct Submission<'s, 'a> {
    body: &'s Json,
    bytes: &'s [u8],
    admitted: &'s Admitted,
    command: u64,
    lease: u64,
    before: u32,
    deadline: Instant,
    mtp_reply: Option<&'s mut crate::mtp_wire::MtpWireReply<'a>>,
    #[cfg(feature = "verify8")]
    verify: Option<&'s mut Verify8Reply<'a>>,
    #[cfg(feature = "verify8")]
    verify_inputs: Option<&'s (Json, u64)>,
    #[cfg(not(feature = "verify8"))]
    _snapshots: std::marker::PhantomData<&'a ()>,
}

impl OwnedWorker {
    /// The C++ submitted-command `try` block. Any error except a cancellation becomes the caller's invalidating refusal.
    fn submit(
        &mut self,
        submission: Submission<'_, '_>,
        cancelled: &mut Option<CancelProbe<'_>>,
        on_token: &mut Option<TokenSink<'_>>,
    ) -> std::result::Result<Json, CommandError> {
        let Submission {
            body,
            bytes,
            admitted,
            command,
            lease,
            before,
            deadline,
            mut mtp_reply,
            #[cfg(feature = "verify8")]
            mut verify,
            #[cfg(feature = "verify8")]
            verify_inputs,
            ..
        } = submission;
        let mut cancellation = false;
        let mut timed_out = false;
        let mut consumer_failure = false;
        let mut drain_deadline = deadline;
        self.io.idle_check()?;
        let mut sent = 0;
        while sent < bytes.len() {
            require(
                Instant::now() < deadline,
                "owned request submission deadline expired",
            )?;
            let Some(input) = self.io.input.as_ref() else {
                return Err(CommandError::Cause(None));
            };
            match sysfd::send(input.as_fd(), &bytes[sent..], 0) {
                Ok(n) if n > 0 => sent += n,
                Ok(_) => return Err(CommandError::Cause(None)),
                Err(error) => match error.raw_os_error() {
                    Some(EINTR) => continue,
                    Some(EAGAIN) => self.io.pump(deadline, false, POLL_SLICE_MS)?,
                    _ => return Err(CommandError::Cause(Some(Box::new(error)))),
                },
            }
        }
        if self.broker_epoch != 0 {
            self.broker_command = command;
            self.broker_lease = lease;
        }
        let mut token_events: u64 = 0;
        let mut observed: Option<u64> = None;
        loop {
            if !cancellation && !timed_out && !consumer_failure {
                match cancelled.as_mut().map(|probe| probe()) {
                    Some(Ok(value)) => cancellation = value,
                    Some(Err(_)) => consumer_failure = true,
                    None => {}
                }
                timed_out = Instant::now() >= deadline;
                if cancellation || timed_out || consumer_failure {
                    self.needs_reset = true;
                    drain_deadline = deadline.max(Instant::now() + self.options.drain_timeout);
                }
            }
            let draining = cancellation || timed_out || consumer_failure;
            if !self.io.pending.contains(&b'\n') {
                require(
                    !self.io.out_eof && !self.io.worker_exited && !self.io.status_eof,
                    "owned target ended during a command",
                )?;
                require(
                    !draining || Instant::now() < drain_deadline,
                    "owned cancelled/deadline command did not drain in time",
                )?;
                self.io.pump(
                    if draining { drain_deadline } else { deadline },
                    false,
                    POLL_SLICE_MS,
                )?;
                continue;
            }
            let value = self
                .io
                .next(if draining { drain_deadline } else { deadline })?;
            require(
                value.kind() == Kind::Object,
                "owned target emitted a nonobject event",
            )?;
            if let Some(reply) = mtp_reply.as_deref_mut()
                && !event_is(&value, "done")?
            {
                reply.event(&value)?;
                continue;
            }
            #[cfg(feature = "verify8")]
            if let Some(reply) = verify.as_deref_mut()
                && !event_is(&value, "done")?
            {
                if self.broker_epoch != 0 {
                    self.broker_identity(&value, command, lease)?;
                }
                reply.event(&value)?;
                continue; // No token/round callback before complete admission.
            }
            if event_is(&value, "token")? || event_is(&value, "mtp_round")? {
                if event_is(&value, "token")? {
                    token_events += 1;
                    require(
                        !admitted.control && token_events <= 1,
                        "owned target emitted unsolicited/excess token events",
                    )?;
                    if self.broker_epoch != 0 {
                        self.broker_identity(&value, command, lease)?;
                        let id = worker_integer(&value, "id")?;
                        let base = if admitted.reset { 0 } else { u64::from(before) };
                        require(
                            admitted.output_count == 1
                                && id < 248_320
                                && worker_integer(&value, "committed_prefix")? == base + 1,
                            "owned broker token publication mismatch",
                        )?;
                        observed = Some(id);
                    }
                }
                if !draining
                    && let Some(sink) = on_token.as_mut()
                    && sink(&value).is_err()
                {
                    consumer_failure = true;
                    self.needs_reset = true;
                    drain_deadline = deadline.max(Instant::now() + self.options.drain_timeout);
                }
            } else if event_is(
                &value,
                if admitted.control {
                    "control_done"
                } else {
                    "done"
                },
            )? {
                let mtp_events = match mtp_reply.as_deref_mut() {
                    Some(reply) => {
                        reply.finish(&value)?;
                        Some(reply.events().to_vec())
                    }
                    None => None,
                };
                if self.options.scalar_channel {
                    self.last_response = value.clone();
                }
                #[cfg(feature = "verify8")]
                let admitted_events = match verify.as_deref_mut() {
                    Some(reply) => {
                        reply.finish(&value)?;
                        let events = reply.events()?.to_vec();
                        self.verify_previous = value.clone();
                        if let Some((policy, seed)) = verify_inputs {
                            self.verify_sampling = policy.clone();
                            self.verify_seed = *seed;
                        }
                        self.verify_position = worker_integer(&value, "committed_prefix")? as u32;
                        Some(events)
                    }
                    None => None,
                };
                if self.broker_epoch != 0 {
                    self.broker_identity(&value, command, lease)?;
                    self.broker_storage(&value)?;
                    let position = worker_integer(&value, "committed_prefix")?;
                    require(
                        position <= 65536
                            && worker_integer(&value, "checkpoint_retained")?
                                <= u64::from(self.options.checkpoint_slots),
                        "owned broker prefix/retention bound",
                    )?;
                    let generated = worker_field(&value, "generated_ids")?.array()?;
                    if admitted.control {
                        self.control_reply(body, &value, admitted, before, position, token_events)?;
                    } else {
                        #[cfg(feature = "verify8")]
                        let verified = admitted_events.is_some();
                        #[cfg(not(feature = "verify8"))]
                        let verified = false;
                        if verified {
                            #[cfg(feature = "verify8")]
                            require(
                                position == u64::from(self.verify_position),
                                "owned branch8 verified prefix mismatch",
                            )?;
                        } else {
                            let base = if admitted.reset { 0 } else { u64::from(before) };
                            require(
                                position == base + admitted.input_count + admitted.output_count
                                    && generated.len() as u64 == admitted.output_count
                                    && token_events == admitted.output_count
                                    && match observed {
                                        None => true,
                                        Some(id) => generated[0].unsigned_integer()? == id,
                                    },
                                "owned broker step publication mismatch",
                            )?;
                        }
                    }
                    let sampling = worker_field(&value, "sampling")?;
                    require(
                        worker_field(sampling, "valid")?.boolean()?
                            && worker_integer(sampling, "entropy_prefix")? == position,
                        "owned broker entropy publication mismatch",
                    )?;
                    self.broker_position = position as u32;
                }
                self.io.idle_check()?;
                if consumer_failure {
                    return Err(CommandError::Cause(Some(Box::new(CallbackFailed))));
                }
                if cancellation {
                    return Err(CommandError::Cancelled(
                        "owned request cancelled after its current step drained; reset required",
                    ));
                }
                if timed_out {
                    return Err(CommandError::Cause(None));
                }
                if let (Some(events), Some(sink)) = (mtp_events, on_token.as_mut()) {
                    for event in &events {
                        let stop = match cancelled.as_mut().map(|probe| probe()) {
                            Some(Ok(value)) => value,
                            Some(Err(error)) => return Err(CommandError::from(error)),
                            None => false,
                        };
                        if stop {
                            self.needs_reset = true;
                            return Err(CommandError::Cancelled(
                                "MTP cancelled after admitted drain; reset required",
                            ));
                        }
                        self.needs_reset = true;
                        sink(event)?;
                    }
                }
                #[cfg(feature = "verify8")]
                if let (Some(events), Some(sink)) = (admitted_events, on_token.as_mut()) {
                    for event in &events {
                        let stop = match cancelled.as_mut().map(|probe| probe()) {
                            Some(Ok(value)) => value,
                            Some(Err(error)) => return Err(CommandError::from(error)),
                            None => false,
                        };
                        if stop {
                            self.needs_reset = true;
                            return Err(CommandError::Cancelled(
                                "private V8 cancelled after admitted drain; reset required",
                            ));
                        }
                        // Callback failures invalidate, not a second drain of an already completed command.
                        sink(event)?;
                    }
                }
                self.needs_reset = false;
                self.last_response = value.clone();
                return Ok(value);
            } else {
                return Err(CommandError::Cause(None));
            }
        }
    }

    /// Zero-model-step control completion checks.
    fn control_reply(
        &mut self,
        body: &Json,
        value: &Json,
        admitted: &Admitted,
        before: u32,
        position: u64,
        token_events: u64,
    ) -> Result<()> {
        #[cfg(feature = "branch8")]
        if self.options.branch8 {
            crate::verify8::validate_branch8_control(
                &self.ready_value,
                &self.verify_previous,
                body,
                value,
                false,
            )?;
            self.verify_previous = value.clone();
            self.verify_position = position as u32;
            if admitted.begin {
                self.verify_sampling = worker_field(body, "sampling")?.clone();
                self.verify_seed = worker_integer(body, "seed")?;
            }
        }
        let generated = worker_field(value, "generated_ids")?.array()?;
        require(
            worker_field(value, "op")?.string()? == admitted.operation.as_slice()
                && worker_integer(value, "model_steps")? == 0
                && generated.is_empty()
                && token_events == 0,
            "owned control executed model work",
        )?;
        let outcome = worker_field(value, "outcome")?.string()?;
        let common = worker_integer(value, "common_prefix")?;
        let before = u64::from(before);
        if admitted.begin {
            let count = worker_field(body, "prompt_ids")?.array()?.len() as u64;
            require(
                common <= before.min(count)
                    && (admitted.allow_restore || common == 0)
                    && position <= common
                    && ((outcome == b"cold_miss" && position == 0)
                        || (outcome == b"restored" && position != 0 && admitted.allow_restore)),
                "owned broker restore selection mismatch",
            )?;
            let sampling = worker_field(value, "sampling")?;
            require(
                worker_integer(sampling, "seed")? == worker_integer(body, "seed")?,
                "owned broker seed changed",
            )?;
            for (name, parameter) in worker_field(body, "sampling")?.object()? {
                let observed = sampling
                    .find(name)?
                    .ok_or(WorkerError::Refused("missing owned broker field"))?;
                require(
                    worker_number(observed)? == worker_number(parameter)?,
                    "owned broker sampling policy changed",
                )?;
            }
        } else {
            require(
                outcome == b"captured" && position == before && common == before,
                "owned broker capture prefix mismatch",
            )?;
        }
        let logits = worker_field(value, "logits_available")?.boolean()?;
        let stop = worker_field(value, "stop_id")?;
        require(
            logits == (position != 0)
                && (stop.is_null() || (logits && stop.unsigned_integer()? < 248_320))
                && worker_field(value, "continuation_allowed")?.boolean()?
                    == (logits && stop.is_null()),
            "owned broker terminal/logit metadata mismatch",
        )
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use tokspur_json::parse_json;

    use super::*;

    #[test]
    fn burst_pacing_charges_only_work_and_recovers_when_idle() {
        let mut pacing = BurstPacing::default();
        let start = Instant::now();
        let mut now = start + Duration::from_secs(28);
        pacing.charge(start, now);
        assert_eq!(pacing.delay_at(now), Duration::ZERO);
        pacing.charge(now, now + Duration::from_secs(3));
        now += Duration::from_secs(3);
        assert_eq!(pacing.delay_at(now), Duration::from_secs(2));
        now += Duration::from_secs(2);
        assert_eq!(pacing.delay_at(now), Duration::ZERO);
        assert_eq!(
            pacing.debt_at(now + BurstPacing::BURST_SLACK),
            Duration::ZERO
        );
    }

    #[test]
    fn pacing_polls_cancellation_and_respects_the_request_deadline() {
        let now = Instant::now();
        let pacing = BurstPacing {
            outstanding: Some((now, BurstPacing::BURST_SLACK + Duration::from_secs(60))),
        };
        let mut polls = 0;
        let mut probe = || {
            polls += 1;
            Ok(polls >= 2)
        };
        assert!(
            cancelled_before_submit(
                Some(&pacing),
                now + Duration::from_secs(60),
                &mut Some(&mut probe),
            )
            .unwrap()
        );
        assert_eq!(polls, 2);
        assert!(!cancelled_before_submit(Some(&pacing), now, &mut None).unwrap());
    }

    #[test]
    fn worker_error_retains_immediate_sources_and_redacts_debug() {
        let json_err = parse_json(b"\"\xc0\x80\"").unwrap_err();
        let malformed = WorkerError::MalformedReply(json_err);
        assert_eq!(malformed.to_string(), "owned worker emitted malformed JSON");
        assert!(malformed.is_refused());
        let first = malformed
            .source()
            .expect("MalformedReply must expose JsonError");
        assert!(
            first.source().is_some(),
            "JsonError must expose UnicodeError"
        );

        let io_err = std::io::Error::other("/Users/secret/worker-pipe");
        let pipe = WorkerError::PipeRead(io_err);
        assert_eq!(pipe.to_string(), "owned worker pipe read failed");
        assert!(pipe.source().is_some());
        assert!(!format!("{pipe:?}").contains("secret"));
    }
}
