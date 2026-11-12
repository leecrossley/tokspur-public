//! `ServeSession`, ported one to one from `native/src/serve_protocol.cpp`.
//!
//! CPU-only seams: production binds the codec to the owned tokenizer and renderer and the target to exactly one owned worker; tests supply deterministic fixtures.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use tokspur_json::{Json, JsonError, Kind};
use tokspur_text::chat::RenderedChat;
use tokspur_text::response::{
    ResponseKind, ResponseLimits, ResponseParser, ResponseRecord, ResponseStatus,
};
use tokspur_text::response_protocol::{
    RESPONSE_MAXIMUM_ACCUMULATED_IDS, ResponseAdmission, ResponseProtocolError,
    parse_native_response_with_controls,
};

use crate::admission::{
    AdmissionError, AdmittedChat, ToolArgValidation, ToolEnumValidation, admit_chat_with_policies,
    broker_storage, chunk, field, number, parser_manifest, real, require,
    validate_executor_tool_arguments, validate_tool_value_with_policy, validate_worker_ready,
    worker_execution,
};
use crate::http::HttpError;
use crate::request_timing::PhaseStamps;
use tokspur_wire::worker::WorkerError;

#[cfg(feature = "verify8")]
use tokspur_wire::verify8::{Verify8Context, Verify8Error, Verify8Reply};

/// Everything `ServeSession::complete` (and its callbacks) can fail with; `Display` is the C++ `what()` text.
pub enum ServeError {
    /// Admission or a session check refused (`FormatError`).
    Admission(AdmissionError),
    /// A text-layer operation (render, encode, decode, final parse) failed.
    Text(Box<dyn std::error::Error + Send + Sync>),
    /// The worker refused, failed or was cancelled.
    Worker(WorkerError),
    /// V8/branch8 reply admission refused.
    #[cfg(feature = "verify8")]
    Verify8(Verify8Error),
    /// Streaming to the client failed.
    Http(HttpError),
}

impl fmt::Debug for ServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => formatter.debug_tuple("Admission").field(error).finish(),
            Self::Text(_) => formatter.write_str("Text([redacted])"),
            Self::Worker(error) => formatter.debug_tuple("Worker").field(error).finish(),
            #[cfg(feature = "verify8")]
            Self::Verify8(error) => formatter.debug_tuple("Verify8").field(error).finish(),
            Self::Http(error) => formatter.debug_tuple("Http").field(error).finish(),
        }
    }
}

impl fmt::Display for ServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => error.fmt(formatter),
            Self::Text(error) => error.fmt(formatter),
            Self::Worker(error) => error.fmt(formatter),
            #[cfg(feature = "verify8")]
            Self::Verify8(error) => error.fmt(formatter),
            Self::Http(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ServeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            Self::Text(error) => Some(error.as_ref()),
            Self::Worker(error) => Some(error),
            #[cfg(feature = "verify8")]
            Self::Verify8(error) => Some(error),
            Self::Http(error) => Some(error),
        }
    }
}

impl From<AdmissionError> for ServeError {
    fn from(error: AdmissionError) -> Self {
        Self::Admission(error)
    }
}

impl From<JsonError> for ServeError {
    fn from(error: JsonError) -> Self {
        Self::Admission(AdmissionError::Json(error))
    }
}

impl From<WorkerError> for ServeError {
    fn from(error: WorkerError) -> Self {
        Self::Worker(error)
    }
}

#[cfg(feature = "verify8")]
impl From<Verify8Error> for ServeError {
    fn from(error: Verify8Error) -> Self {
        Self::Verify8(error)
    }
}

impl From<HttpError> for ServeError {
    fn from(error: HttpError) -> Self {
        Self::Http(error)
    }
}

impl From<tokspur_wire::mtp_wire::MtpWireError> for ServeError {
    fn from(error: tokspur_wire::mtp_wire::MtpWireError) -> Self {
        Self::Admission(AdmissionError::Mtp(error))
    }
}

type Result<T> = std::result::Result<T, ServeError>;

/// Renders an admitted request into prompt text (`ServeCodec::render`).
pub type RenderFn = Box<dyn Fn(&Json) -> Result<RenderedChat> + Send>;
/// Encodes renderer output into token IDs while retaining its literal byte ranges (`ServeCodec::encode`).
pub type EncodeFn = Box<dyn Fn(&RenderedChat) -> Result<Vec<u32>> + Send>;
/// Decodes generated IDs into their exact bytes (`ServeCodec::decode`).
pub type DecodeFn = Box<dyn Fn(&[u32]) -> Result<Vec<u8>> + Send>;
/// Receives one streaming chunk (`ServeSink::chunk`).
pub type ChunkSink<'a> = &'a mut dyn FnMut(&Json) -> Result<()>;

/// `ServeCodec`.
pub struct ServeCodec {
    pub render: RenderFn,
    pub encode: EncodeFn,
    pub decode: DecodeFn,
    pub admission: ResponseAdmission,
    /// Added-token IDs of the response frame markers (`<think>`, `</think>`, `<tool_call>`, `</tool_call>`). Empty keeps the byte-spelled framing.
    pub control_ids: Vec<u32>,
}

impl fmt::Debug for ServeCodec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ServeCodec").finish_non_exhaustive()
    }
}

/// The token-event callback a target receives.
pub type TokenCallback<'a> = &'a mut dyn FnMut(&Json) -> Result<()>;
/// `ServeTarget`: submits one worker command and returns its completion.
pub type ServeTarget<'a> = &'a mut dyn FnMut(&Json, TokenCallback<'_>) -> Result<Json>;

/// Receives one trace event (`ServeSink::prefix_trace`).
pub type TraceSink<'a> = &'a mut dyn FnMut(&Json) -> Result<()>;

/// `ServeSink`.
pub struct ServeSink<'a> {
    pub chunk: Option<ChunkSink<'a>>,
    pub cancelled: &'a mut dyn FnMut() -> bool,
    pub new_seed: &'a mut dyn FnMut() -> u64,
    pub prefix_trace: Option<TraceSink<'a>>,
}

/// The `ServeSession` constructor arguments after the codec.
#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub alias: Vec<u8>,
    /// Immutable startup choice, never a request field.
    pub stack_execution: String,
    pub tool_enums: ToolEnumValidation,
    pub tool_args: ToolArgValidation,
    pub broker_ready: Json,
    #[cfg(feature = "verify8")]
    pub verify8: bool,
    #[cfg(feature = "verify8")]
    pub verify_ready: Json,
    #[cfg(feature = "branch8")]
    pub branch8: bool,
    pub mtp_ready: Json,
    pub mtp_first_output_packet: Option<u32>,
    pub overlap_prefill: bool,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            alias: b"tokspur".to_vec(),
            stack_execution: "direct".to_owned(),
            tool_enums: ToolEnumValidation::Strict,
            tool_args: ToolArgValidation::Strict,
            broker_ready: Json::Null,
            #[cfg(feature = "verify8")]
            verify8: false,
            #[cfg(feature = "verify8")]
            verify_ready: Json::Null,
            #[cfg(feature = "branch8")]
            branch8: false,
            mtp_ready: Json::Null,
            mtp_first_output_packet: None,
            overlap_prefill: false,
        }
    }
}

/// One persistent conversation lane over one owned target.
pub struct ServeSession {
    codec: ServeCodec,
    alias: Vec<u8>,
    stack_execution: String,
    tool_enums: ToolEnumValidation,
    tool_args: ToolArgValidation,
    committed: Vec<u32>,
    sampling: Vec<u8>,
    seed: u64,
    reusable: bool,
    broker_ready: Json,
    broker_epoch: u64,
    broker_command: u64,
    broker_lease: u64,
    broker_resync: bool,
    #[cfg(feature = "verify8")]
    verify8: bool,
    #[cfg(feature = "verify8")]
    verify_ready: Json,
    #[cfg(feature = "verify8")]
    verify_previous: Json,
    /// Only a cold reset may rebase metadata after a drained cancellation.
    #[cfg(feature = "verify8")]
    verify_resync: bool,
    #[cfg(feature = "branch8")]
    branch8: bool,
    pub mtp: bool,
    pub mtp_ready: Json,
    pub mtp_previous: Json,
    pub mtp_first_output_packet: Option<u32>,
    pub mtp_resync: bool,
    pub mtp_generation_id: u64,
    pub mtp_committed: u64,
    pub mtp_emitted: u64,
    pub mtp_pending: u64,
    pub mtp_stage_a_controller: Option<tokspur_wire::mtp_wire::StageAControllerState>,
    pub overlap_prefill: bool,
    /// Reasoning this session generated, keyed by the assistant message's ordinal in the request history.
    saved_reasoning: Vec<(usize, Vec<u8>)>,
    /// Off by default; only the opt-in `request_timing` record reads the stamps.
    phase_stamps_enabled: bool,
    phase_stamps: PhaseStamps,
}

impl fmt::Debug for ServeSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Committed token IDs belong to a real conversation; count them only.
        formatter
            .debug_struct("ServeSession")
            .field("committed_tokens", &self.committed.len())
            .field("reusable", &self.reusable)
            .finish_non_exhaustive()
    }
}

fn is(value: &Json, expected: &str) -> std::result::Result<bool, JsonError> {
    Ok(value.string()? == expected.as_bytes())
}

impl ServeSession {
    pub fn new(codec: ServeCodec, options: SessionOptions) -> Result<Self> {
        let mtp = !options.mtp_ready.is_null();
        let mut session = Self {
            codec,
            alias: options.alias,
            stack_execution: options.stack_execution,
            tool_enums: options.tool_enums,
            tool_args: options.tool_args,
            committed: Vec::new(),
            sampling: Vec::new(),
            seed: 0,
            reusable: false,
            broker_ready: options.broker_ready,
            broker_epoch: 0,
            broker_command: 0,
            broker_lease: 0,
            broker_resync: true,
            #[cfg(feature = "verify8")]
            verify8: options.verify8,
            #[cfg(feature = "verify8")]
            verify_ready: options.verify_ready,
            #[cfg(feature = "verify8")]
            verify_previous: Json::Null,
            #[cfg(feature = "verify8")]
            verify_resync: false,
            #[cfg(feature = "branch8")]
            branch8: options.branch8,
            mtp,
            mtp_ready: options.mtp_ready.clone(),
            mtp_previous: if mtp { options.mtp_ready } else { Json::Null },
            mtp_first_output_packet: options.mtp_first_output_packet,
            mtp_resync: false,
            mtp_generation_id: 0,
            mtp_committed: 0,
            mtp_emitted: 0,
            mtp_pending: 0,
            mtp_stage_a_controller: None,
            overlap_prefill: options.overlap_prefill,
            saved_reasoning: Vec::new(),
            phase_stamps_enabled: false,
            phase_stamps: PhaseStamps::default(),
        };
        require(!session.alias.is_empty(), "invalid serving codec")?;
        require(
            session.stack_execution == "direct" || session.stack_execution == "recurrent-replay",
            "unsupported serving startup execution mode",
        )?;
        if session.mtp {
            let draft = tokspur_wire::mtp_wire::mtp_draft_attention(&session.mtp_ready)?;
            let carry = tokspur_wire::mtp_wire::mtp_request_carry(&session.mtp_ready)?;
            let depth = tokspur_wire::mtp_wire::mtp_depth(&session.mtp_ready)?;
            tokspur_wire::mtp_wire::validate_mtp_ready(&session.mtp_ready, draft, carry, depth)?;
            session.committed.reserve(65536);
        }
        if let Some(first_pkt) = session.mtp_first_output_packet {
            require(
                session.mtp,
                "MTP first-output-packet requires native MTP readiness",
            )?;
            let cap = tokspur_wire::mtp_wire::mtp_output_packet(&session.mtp_ready)?;
            require(
                (1..=cap).contains(&first_pkt),
                "MTP first-output-packet outside negotiated output packet bound",
            )?;
        }
        #[cfg(feature = "verify8")]
        if session.verify8 {
            require(
                session.stack_execution == "direct",
                "private HTTP V8 cannot use replay",
            )?;
            if session.branch8_mode() {
                #[cfg(feature = "branch8")]
                {
                    require(
                        !session.broker_ready.is_null()
                            && session.broker_ready.dump()? == session.verify_ready.dump()?,
                        "branch8 requires the same explicit broker/verifier readiness",
                    )?;
                    tokspur_wire::verify8::validate_branch8_ready(&session.verify_ready)?;
                    session.verify_previous = session.verify_ready.clone();
                }
            } else {
                require(
                    session.broker_ready.is_null(),
                    "private HTTP V8 cannot use checkpoints without explicit branch8 consent",
                )?;
                tokspur_wire::verify8::validate_verify8_ready(&session.verify_ready)?;
            }
            session.committed.reserve(65536);
        } else {
            require(
                session.verify_ready.is_null(),
                "unexpected private V8 readiness",
            )?;
        }
        #[cfg(feature = "branch8")]
        require(
            !session.branch8 || session.verify8,
            "branch8 requires explicit private V8 scheduling",
        )?;
        if !session.broker_ready.is_null() {
            let slots = field(&session.broker_ready, "checkpoint_slots")?.unsigned_integer()?;
            require(slots != 0 && slots <= 4, "serving broker slot bound")?;
            let budget = field(&session.broker_ready, "checkpoint_combined_budget_bytes")?
                .unsigned_integer()?;
            validate_worker_ready(
                &session.broker_ready,
                &session.stack_execution,
                slots as u32,
                budget,
            )?;
            session.broker_epoch =
                field(&session.broker_ready, "broker_epoch")?.unsigned_integer()?;
            session.committed.reserve(65536); // Before any restore publication.
        }
        Ok(session)
    }

    fn branch8_mode(&self) -> bool {
        #[cfg(feature = "branch8")]
        {
            self.branch8
        }
        #[cfg(not(feature = "branch8"))]
        {
            false
        }
    }

    fn verify8_mode(&self) -> bool {
        #[cfg(feature = "verify8")]
        {
            self.verify8
        }
        #[cfg(not(feature = "verify8"))]
        {
            false
        }
    }

    /// Turns on the phase stamps that the opt-in `request_timing` record reads. Off, no stamp is taken.
    pub fn set_phase_stamps(&mut self, enabled: bool) {
        self.phase_stamps_enabled = enabled;
        self.phase_stamps = PhaseStamps::default();
    }

    /// The latest request's phase stamps; all empty unless [`Self::set_phase_stamps`] turned them on.
    pub fn phase_stamps(&self) -> &PhaseStamps {
        &self.phase_stamps
    }

    /// The first output packet's worker command has returned; `batches` counts packets so far.
    fn stamp_output_packet(&mut self, batches: u64, tokens: usize) {
        if self.phase_stamps_enabled && batches == 1 {
            self.phase_stamps.first_packet_done = Some(std::time::Instant::now());
            self.phase_stamps.first_packet_tokens = tokens as u64;
        }
    }

    /// Drops reusable state after any failure.
    pub fn invalidate(&mut self) {
        self.saved_reasoning.clear();
        self.committed.clear();
        self.sampling.clear();
        self.reusable = false;
        self.seed = 0;
        self.broker_resync = true;
        self.mtp_resync = true;
        self.mtp_committed = 0;
        self.mtp_emitted = 0;
        self.mtp_pending = 0;
        self.mtp_stage_a_controller = None;
        #[cfg(feature = "verify8")]
        {
            self.verify_resync = true;
        }
    }

    pub fn committed_tokens(&self) -> usize {
        self.committed.len()
    }

    /// The request with this session's own reasoning restored to assistant turns that arrived without it.
    fn with_saved_reasoning(&self, render: &Json) -> Result<Option<Json>> {
        if self.saved_reasoning.is_empty() {
            return Ok(None);
        }
        let mut fields = render.object()?.clone();
        let Some((_, messages)) = fields.iter_mut().find(|(name, _)| name == b"messages") else {
            return Ok(None);
        };
        let mut list = messages.array()?.clone();
        let mut ordinal = 0;
        let mut filled = false;
        for message in &mut list {
            if field(message, "role")?.string()? != b"assistant" {
                continue;
            }
            let absent = message.find("reasoning_content")?.is_none_or(Json::is_null);
            if absent
                && let Some((_, text)) = self.saved_reasoning.iter().find(|(at, _)| *at == ordinal)
            {
                let mut entry = message.object()?.clone();
                entry.retain(|(name, _)| name != b"reasoning_content");
                entry.push((b"reasoning_content".to_vec(), Json::String(text.clone())));
                *message = Json::Object(entry);
                filled = true;
            }
            ordinal += 1;
        }
        // Nothing restored renders the same prompt again, so skip the second render and encode.
        if !filled {
            return Ok(None);
        }
        *messages = Json::Array(list);
        Ok(Some(Json::Object(fields)))
    }

    /// Serves one chat completion. Refusals before worker state can change keep reusable state. Later failures invalidate it.
    pub fn complete(
        &mut self,
        input: &Json,
        response_id: &[u8],
        target: ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
    ) -> Result<Json> {
        let request_start = std::time::Instant::now();
        if self.phase_stamps_enabled {
            self.phase_stamps = PhaseStamps {
                request_start: Some(request_start),
                ..PhaseStamps::default()
            };
        }
        let request =
            admit_chat_with_policies(input, &self.alias, self.tool_enums, self.tool_args)?;
        crate::admission::validate_calibrated_sampling(&request.sampling, &self.mtp_ready)?;
        if !self.mtp && self.verify8_mode() {
            require(
                number(field(&request.sampling, "temperature")?, 0.0, 100.0)? == 0.0,
                "private HTTP V8 is greedy-only; nonzero temperature is explicitly refused",
            )?;
        }
        let rendered = (self.codec.render)(&request.render)?;
        let mut prompt = (self.codec.encode)(&rendered)?;
        if self.reusable
            && !prompt.starts_with(&self.committed)
            && let Some(filled) = self.with_saved_reasoning(&request.render)?
        {
            let candidate = (self.codec.encode)(&(self.codec.render)(&filled)?)?;
            if candidate.len() < 65536 && candidate.starts_with(&self.committed) {
                prompt = candidate;
            }
        }
        require(
            !prompt.is_empty() && prompt.len() < 65536,
            "request leaves no generation space in the 64K context",
        )?;
        let maximum_tokens = (request.maximum_tokens as usize).min(65536 - prompt.len()) as u32;
        for &id in &prompt {
            require(
                id < self.codec.admission.vocabulary,
                "rendered token outside vocabulary",
            )?;
        }
        let policy = request.sampling.dump()?;
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(AdmissionError::Clock)?
            .as_secs();
        let append = self.reusable
            && self.sampling == policy
            && request.seed.is_none_or(|seed| seed == self.seed)
            && prompt.len() >= self.committed.len()
            && prompt.starts_with(&self.committed);
        let cached_prefix = if append { self.committed.len() } else { 0 };
        let mut common = 0;
        if self.reusable {
            while common < self.committed.len()
                && common < prompt.len()
                && self.committed[common] == prompt[common]
            {
                common += 1;
            }
        }
        let seed = if append {
            self.seed
        } else if let Some(seed) = request.seed {
            seed
        } else {
            (sink.new_seed)()
        };
        require(
            !response_id.is_empty() && response_id.len() <= 96,
            "response ID bound",
        )?;
        let parser = ResponseParser::new_with_executor_parameters(
            request.manifest.clone(),
            request.thinking,
            ResponseLimits::default(),
            self.tool_args == ToolArgValidation::Executor,
        )
        .map_err(|error| ServeError::Text(Box::new(error)))?
        .with_token_framing();
        let parser = if self.codec.control_ids.is_empty() {
            parser
        } else {
            parser.with_control_tokens()
        };
        let mut completion = Completion {
            request_start,
            maximum_tokens,
            request: &request,
            prompt: &prompt,
            policy: &policy,
            seed,
            created,
            response_id,
            append,
            common,
            cached_prefix,
            parser,
            cursor: StreamCursor::default(),
            streamed_content: Vec::new(),
            streamed_thinking: Vec::new(),
            generated: Vec::with_capacity(maximum_tokens as usize),
            position: 0,
            state_touched: false,
            counters: Counters::default(),
        };
        let result = completion.run(self, target, sink);
        if self.phase_stamps_enabled {
            self.phase_stamps.output_packets = completion.counters.output_batches;
            self.phase_stamps.max_packet_tokens = completion.counters.output_width as u64;
            self.phase_stamps.completion_tokens = completion.generated.len() as u64;
        }
        if result.is_err() && completion.state_touched {
            self.invalidate();
        }
        result
    }
}

#[derive(Default)]
struct Counters {
    input_batches: u64,
    restored_input_tokens: u64,
    output_batches: u64,
    generation_rounds: u64,
    #[allow(dead_code)]
    verified_rounds: u64,
    accepted_tokens: u64,
    input_width: usize,
    output_width: usize,
    overlap_pipelined: bool,
    #[allow(dead_code)]
    checkpoint_captures: u64,
    #[allow(dead_code)]
    checkpoint_restores: u64,
    #[allow(dead_code)]
    checkpoint_cold_resyncs: u64,
}

/// How far the parser's text and thinking records have been streamed.
///
/// While a response is incomplete the parser only appends: a new record goes on the end and only the last record's text grows (tool records stay withheld until `finish`). The unsent bytes are therefore the tail of the record at `record` plus every record after it, so a token costs its own bytes, not the whole response so far.
#[derive(Default)]
struct StreamCursor {
    record: usize,
    offset: usize,
}

/// The bytes one `StreamCursor::advance` found unsent.
#[derive(Default)]
struct StreamDelta {
    thinking: Vec<u8>,
    content: Vec<u8>,
}

impl StreamCursor {
    fn advance(&mut self, records: &[ResponseRecord]) -> Result<StreamDelta> {
        const NONMONOTONIC: &str = "nonmonotonic response parser";
        let mut delta = StreamDelta::default();
        let Some(last) = records.last() else {
            require(self.record == 0 && self.offset == 0, NONMONOTONIC)?;
            return Ok(delta);
        };
        require(self.record < records.len(), NONMONOTONIC)?;
        let mut offset = self.offset;
        for record in &records[self.record..] {
            require(offset <= record.text.len(), NONMONOTONIC)?;
            let unsent = &record.text[offset..];
            match record.kind {
                ResponseKind::Text => delta.content.extend_from_slice(unsent),
                ResponseKind::Thinking => delta.thinking.extend_from_slice(unsent),
                ResponseKind::Tool => {}
            }
            offset = 0;
        }
        self.record = records.len() - 1;
        self.offset = last.text.len();
        Ok(delta)
    }
}

/// The per-request state the C++ lambdas capture by reference.
struct Completion<'r> {
    request_start: std::time::Instant,
    maximum_tokens: u32,
    request: &'r AdmittedChat,
    prompt: &'r [u32],
    policy: &'r [u8],
    seed: u64,
    created: u64,
    response_id: &'r [u8],
    append: bool,
    common: usize,
    cached_prefix: usize,
    parser: ResponseParser,
    cursor: StreamCursor,
    streamed_content: Vec<u8>,
    streamed_thinking: Vec<u8>,
    generated: Vec<u32>,
    position: usize,
    state_touched: bool,
    counters: Counters,
}

fn cancelled(sink: &mut ServeSink<'_>) -> Result<()> {
    Ok(require(
        !(sink.cancelled)(),
        "request cancelled or deadline expired",
    )?)
}

impl Completion<'_> {
    fn emit(
        &self,
        session: &ServeSession,
        sink: &mut ServeSink<'_>,
        delta: Json,
        finish: Json,
    ) -> Result<()> {
        if self.request.stream {
            require(sink.chunk.is_some(), "missing streaming sink")?;
            cancelled(sink)?;
            let mut event = chunk(
                self.response_id,
                &session.alias,
                self.created,
                delta,
                finish,
            );
            if self.request.include_usage {
                let mut fields = event.object()?.clone();
                fields.push((b"usage".to_vec(), Json::Null));
                event = Json::Object(fields);
            }
            if let Some(send) = sink.chunk.as_mut() {
                send(&event)?;
            }
        }
        Ok(())
    }

    fn flush(&mut self, session: &ServeSession, sink: &mut ServeSink<'_>) -> Result<()> {
        // The finished response is still checked against everything streamed, in `run`.
        let unsent = self.cursor.advance(self.parser.records())?;
        if !unsent.thinking.is_empty() {
            let delta =
                Json::new_object([("reasoning_content", Json::from(unsent.thinking.as_slice()))]);
            self.emit(session, sink, delta, Json::Null)?;
        }
        if !unsent.content.is_empty() {
            let delta = Json::new_object([("content", Json::from(unsent.content.as_slice()))]);
            self.emit(session, sink, delta, Json::Null)?;
        }
        self.streamed_thinking.extend_from_slice(&unsent.thinking);
        self.streamed_content.extend_from_slice(&unsent.content);
        Ok(())
    }

    fn broker_reply(&self, session: &mut ServeSession, value: &Json, fresh: bool) -> Result<()> {
        if session.broker_epoch == 0 {
            return Ok(());
        }
        if session.branch8_mode() {
            require(
                is(field(value, "broker_protocol")?, "branch8-v1")?
                    && field(value, "broker_batch_width")?.unsigned_integer()? == 8,
                "serving branch8 wire identity changed",
            )?;
        }
        require(
            field(value, "broker_epoch")?.unsigned_integer()? == session.broker_epoch,
            "serving worker epoch changed",
        )?;
        let command = field(value, "command_id")?.unsigned_integer()?;
        let lease = field(value, "lease_id")?.unsigned_integer()?;
        require(
            command > session.broker_command
                && (session.broker_resync || command == session.broker_command.wrapping_add(1))
                && if fresh {
                    lease > session.broker_lease
                } else {
                    lease != 0 && lease == session.broker_lease
                },
            "stale serving broker command/lease",
        )?;
        broker_storage(value, &session.broker_ready)?;
        require(
            field(value, "checkpoint_retained")?.unsigned_integer()?
                <= field(&session.broker_ready, "checkpoint_slots")?.unsigned_integer()?,
            "serving broker retained count exceeds capacity",
        )?;
        session.broker_command = command;
        session.broker_lease = lease;
        session.broker_resync = false;
        Ok(())
    }

    fn sampling_matches(&self, value: &Json, prefix: usize) -> Result<()> {
        let sampled = field(value, "sampling")?;
        require(
            field(sampled, "valid")?.boolean()?
                && field(sampled, "entropy_prefix")?.unsigned_integer()? == prefix as u64
                && field(sampled, "seed")?.unsigned_integer()? == self.seed,
            "worker sampling prefix/seed mismatch",
        )?;
        for (name, value) in self.request.sampling.object()? {
            require(
                number(field(sampled, name)?, 0.0, 1e9)? == number(value, 0.0, 1e9)?,
                "worker sampling policy changed",
            )?;
        }
        Ok(())
    }

    fn control(
        &mut self,
        session: &mut ServeSession,
        target: &mut ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
        job: Json,
        operation: &str,
        fresh: bool,
    ) -> Result<Json> {
        cancelled(sink)?;
        let done = target(&job, &mut |_: &Json| {
            Err(AdmissionError::Refused("zero-step broker control emitted a token").into())
        })?;
        cancelled(sink)?;
        require(
            is(field(&done, "event")?, "control_done")?
                && is(field(&done, "op")?, operation)?
                && field(&done, "model_steps")?.unsigned_integer()? == 0
                && field(&done, "generated_ids")?.array()?.is_empty(),
            "invalid zero-step broker completion",
        )?;
        #[cfg(feature = "branch8")]
        if session.branch8 {
            tokspur_wire::verify8::validate_branch8_control(
                &session.verify_ready,
                &session.verify_previous,
                &job,
                &done,
                session.verify_resync,
            )?;
            if operation == "branch_capture" {
                self.counters.checkpoint_captures += 1;
            } else {
                self.counters.checkpoint_restores +=
                    u64::from(is(field(&done, "outcome")?, "restored")?);
                self.counters.checkpoint_cold_resyncs += u64::from(session.verify_resync);
            }
            session.verify_previous = done.clone();
            session.verify_resync = false;
        }
        self.broker_reply(session, &done, fresh)?;
        let prefix = field(&done, "committed_prefix")?.unsigned_integer()?;
        require(
            prefix <= self.prompt.len() as u64,
            "broker restored beyond future prompt",
        )?;
        self.sampling_matches(&done, prefix as usize)?;
        let logits = field(&done, "logits_available")?.boolean()?;
        let stop = field(&done, "stop_id")?;
        require(
            logits == (prefix != 0)
                && (stop.is_null()
                    || (logits
                        && (stop.unsigned_integer()?
                            == u64::from(session.codec.admission.im_end)
                            || stop.unsigned_integer()?
                                == u64::from(session.codec.admission.end_of_text))))
                && field(&done, "continuation_allowed")?.boolean()? == (logits && stop.is_null()),
            "invalid broker terminal metadata",
        )?;
        Ok(done)
    }

    fn step(
        &mut self,
        session: &mut ServeSession,
        target: &mut ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
        job: Json,
        input_token: Option<u32>,
        decode: bool,
    ) -> Result<Json> {
        cancelled(sink)?;
        let mut events: Vec<u32> = Vec::new();
        let position = self.position;
        let vocabulary = session.codec.admission.vocabulary;
        let (epoch, next_command, lease) = (
            session.broker_epoch,
            session.broker_command.wrapping_add(1),
            session.broker_lease,
        );
        let done = target(&job, &mut |event: &Json| {
            require(
                decode && is(field(event, "event")?, "token")?,
                "unexpected worker token event",
            )?;
            let id = field(event, "id")?.unsigned_integer()?;
            require(
                id < u64::from(vocabulary) && events.is_empty(),
                "invalid worker token count/ID",
            )?;
            require(
                field(event, "committed_prefix")?.unsigned_integer()? == position as u64 + 1,
                "worker token prefix mismatch",
            )?;
            events.push(id as u32);
            if epoch != 0 {
                require(
                    field(event, "broker_epoch")?.unsigned_integer()? == epoch
                        && field(event, "command_id")?.unsigned_integer()? == next_command
                        && field(event, "lease_id")?.unsigned_integer()? == lease,
                    "stale serving broker token event",
                )?;
            }
            Ok(())
        })?;
        require(
            is(field(&done, "event")?, "done")?
                && field(&done, "committed_prefix")?.unsigned_integer()?
                    == self.position as u64 + 1,
            "worker completion prefix mismatch",
        )?;
        worker_execution(&done, &session.stack_execution)?;
        self.broker_reply(session, &done, field(&job, "reset")?.boolean()?)?;
        self.sampling_matches(&done, self.position + 1)?;
        let ids = field(&done, "generated_ids")?.array()?;
        require(
            ids.len() == usize::from(decode) && events.len() == ids.len(),
            "worker generated count mismatch",
        )?;
        if decode {
            require(
                ids[0].unsigned_integer()? == u64::from(events[0]),
                "worker token/done disagreement",
            )?;
        }
        self.position += 1;
        let token = match input_token {
            Some(token) => token,
            None => events[0],
        };
        session.committed.push(token);
        Ok(done)
    }

    #[cfg(feature = "verify8")]
    fn run_many(
        &mut self,
        session: &mut ServeSession,
        target: &mut ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
        job: Json,
    ) -> Result<Json> {
        cancelled(sink)?;
        // The gate reads readiness and the previous reply; snapshots keep it independent of the session state this command updates.
        let ready = session.verify_ready.clone();
        let previous = session.verify_previous.clone();
        let mut reply = Verify8Reply::new(Verify8Context {
            ready: &ready,
            previous: &previous,
            job: &job,
            before: self.position as u32,
            policy: &self.request.sampling,
            seed: self.seed,
            reset_resync: session.verify_resync,
            branch8: session.branch8_mode(),
        })?;
        let branch8 = session.branch8_mode();
        let (epoch, next_command, lease) = (
            session.broker_epoch,
            session.broker_command.wrapping_add(1),
            session.broker_lease,
        );
        let done = target(&job, &mut |event: &Json| {
            if branch8 {
                require(
                    field(event, "broker_epoch")?.unsigned_integer()? == epoch
                        && field(event, "command_id")?.unsigned_integer()? == next_command
                        && field(event, "lease_id")?.unsigned_integer()? == lease
                        && is(field(event, "broker_protocol")?, "branch8-v1")?
                        && field(event, "broker_batch_width")?.unsigned_integer()? == 8,
                    "stale serving branch8 batched event",
                )?;
            }
            Ok(reply.event(event)?)
        })?;
        reply.finish(&done)?;
        drop(reply);
        cancelled(sink)?; // No response parsing or history publication before full admission.
        if branch8 {
            self.broker_reply(session, &done, field(&job, "reset")?.boolean()?)?;
        }
        let tokens = field(&job, "tokens")?.array()?;
        let count = tokens.len();
        if count != 0 {
            self.counters.input_batches += 1;
            self.counters.input_width = self.counters.input_width.max(count);
            for id in tokens {
                session.committed.push(id.unsigned_integer()? as u32);
            }
        } else {
            self.counters.output_batches += 1;
            let generated = field(&done, "generated_ids")?.array()?;
            self.counters.output_width = self.counters.output_width.max(generated.len());
            session.stamp_output_packet(self.counters.output_batches, generated.len());
            self.counters.generation_rounds += field(&done, "lookup_rounds")?.unsigned_integer()?;
            self.counters.verified_rounds +=
                field(&done, "lookup_verified_rounds")?.unsigned_integer()?;
            self.counters.accepted_tokens +=
                field(&done, "lookup_accepted_tokens")?.unsigned_integer()?;
            for id in generated {
                session.committed.push(id.unsigned_integer()? as u32);
            }
        }
        self.position = field(&done, "committed_prefix")?.unsigned_integer()? as usize;
        session.verify_previous = done.clone();
        session.verify_resync = false;
        Ok(done)
    }

    fn run_mtp(
        &mut self,
        session: &mut ServeSession,
        target: &mut ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
        job: Json,
    ) -> Result<Json> {
        cancelled(sink)?;
        let ready = session.mtp_ready.clone();
        let previous = session.mtp_previous.clone();
        let mut reply =
            tokspur_wire::mtp_wire::MtpWireReply::new(&ready, &previous, &job, session.mtp_resync)?;
        let done = target(&job, &mut |event: &Json| {
            reply.event(event)?;
            Ok(())
        })?;
        reply.finish(&done)?;
        cancelled(sink)?;
        worker_execution(&done, &session.stack_execution)?;
        let committed_prefix = field(&done, "committed_prefix")?.unsigned_integer()?;
        self.sampling_matches(&done, committed_prefix as usize)?;
        let inputs = field(&job, "tokens")?.array()?;
        if !inputs.is_empty() {
            self.counters.restored_input_tokens += u64::from(reply.restored_input_tokens()?);
            self.counters.input_batches += 1;
            self.counters.input_width = self.counters.input_width.max(inputs.len());
            if let Some(ov) = done.find("prefill_generation_overlap")?
                && ov.kind() == Kind::String
                && ov.string()? == b"16_token_b10_pipelined"
            {
                self.counters.overlap_pipelined = true;
            }
            for id in inputs {
                session.committed.push(id.unsigned_integer()? as u32);
            }
        } else {
            let outputs = field(&done, "generated_ids")?.array()?;
            self.counters.output_batches += 1;
            self.counters.output_width = self.counters.output_width.max(outputs.len());
            session.stamp_output_packet(self.counters.output_batches, outputs.len());
            let done_rounds = field(&done, "native_mtp_rounds")?.unsigned_integer()?;
            let prev_rounds =
                field(&session.mtp_previous, "native_mtp_rounds")?.unsigned_integer()?;
            self.counters.generation_rounds += done_rounds.saturating_sub(prev_rounds);
            let done_acc = field(&done, "native_mtp_accepted")?.unsigned_integer()?;
            let prev_acc =
                field(&session.mtp_previous, "native_mtp_accepted")?.unsigned_integer()?;
            self.counters.accepted_tokens += done_acc.saturating_sub(prev_acc);
            for id in outputs {
                session.committed.push(id.unsigned_integer()? as u32);
            }
            if let Some(gen_meta) = done.find("native_mtp_generation")? {
                if let Some(comm) = gen_meta.find("committed")? {
                    session.mtp_committed = comm.unsigned_integer()?;
                }
                if let Some(emit) = gen_meta.find("emitted")? {
                    session.mtp_emitted = emit.unsigned_integer()?;
                }
                if let Some(ovf) = gen_meta.find("overflow")? {
                    session.mtp_pending = ovf.unsigned_integer()?;
                }
                if let Some(sc) = gen_meta.find("stage_a_controller")?
                    && sc.kind() == Kind::Object
                {
                    let mut ctrl = tokspur_wire::mtp_wire::StageAControllerState::default();
                    if let Some(ea) = sc.find("ewma_alpha1")? {
                        ctrl.ewma_alpha1 = number(ea, 0.0, 1.0)?;
                    }
                    if let Some(cm) = sc.find("current_mode")? {
                        ctrl.current_mode = cm.unsigned_integer()? as u8;
                    }
                    if let Some(co) = sc.find("consecutive_ordinary_steps")? {
                        ctrl.consecutive_ordinary_steps = co.unsigned_integer()? as u8;
                    }
                    if let Some(cpd) = sc.find("controller_policy_digest")?
                        && cpd.kind() == Kind::Array
                    {
                        let arr = cpd.array()?;
                        if arr.len() == 32 {
                            for (i, elem) in arr.iter().enumerate() {
                                ctrl.controller_policy_digest[i] = elem.unsigned_integer()? as u8;
                            }
                        }
                    }
                    if let Some(m) = sc.find("measured")? {
                        ctrl.measured =
                            Some(tokspur_wire::mtp_wire::MeasuredObservation::parse(m)?);
                    }
                    if let Some(expected) =
                        session.mtp_ready.find("native_controller_policy_digest")?
                    {
                        require(
                            sc.find("controller_policy_digest")? == Some(expected),
                            "controller policy differs from startup negotiation",
                        )?;
                    }
                    session.mtp_stage_a_controller = Some(ctrl);
                }
            }
        }
        self.position = committed_prefix as usize;
        session.mtp_previous = done.clone();
        session.mtp_resync = false;
        Ok(done)
    }

    /// Private replay evidence for a refused response, before session invalidation removes its tokens.
    fn response_failure(
        &self,
        sink: &mut ServeSink<'_>,
        phase: &'static str,
        finish_reason: Option<&'static str>,
        fallback_reason: &'static str,
        primary: ServeError,
    ) -> ServeError {
        if let Some(trace) = sink.prefix_trace.as_mut() {
            let reason = if self.parser.error().is_empty() {
                fallback_reason
            } else {
                self.parser.error()
            };
            // Parser errors contain fixed descriptions, never generated bytes or tool values. IDs and the admitted parser manifest remain confined to the already opt-in, bounded private trace.
            let event = Json::new_object([
                ("event", Json::from("response_validation_failed")),
                ("schema", Json::integer(1)),
                ("response_id", Json::from(self.response_id)),
                ("phase", Json::from(phase)),
                (
                    "finish_reason",
                    finish_reason.map_or(Json::Null, Json::from),
                ),
                ("reason", Json::from(reason)),
                (
                    "generated_count",
                    Json::integer(self.generated.len() as u64),
                ),
                (
                    "generated_scope",
                    Json::from("generated-ids-consumed-by-session"),
                ),
                (
                    "generated_ids",
                    Json::Array(
                        self.generated
                            .iter()
                            .map(|&id| Json::integer(u64::from(id)))
                            .collect(),
                    ),
                ),
                (
                    "parser_bytes_received",
                    Json::integer(self.parser.bytes_received() as u64),
                ),
                ("initial_thinking", Json::Bool(self.request.thinking)),
                ("parser_manifest", parser_manifest(&self.request.manifest)),
            ]);
            if trace(&event).is_err() {
                // The primary validation error takes precedence over a diagnostic-write failure, preserving the public HTTP error bytes.
                return primary;
            }
        }
        primary
    }

    /// Feeds one generated ID's bytes to the parser and streams new deltas.
    fn feed(&mut self, session: &ServeSession, sink: &mut ServeSink<'_>, id: u32) -> Result<()> {
        if session.codec.admission.is_stop(id) {
            return Ok(());
        }
        let bytes = (session.codec.decode)(&[id])?;
        let status = if session.codec.control_ids.contains(&id) {
            self.parser.feed_control(&bytes)
        } else {
            self.parser.feed(&bytes)
        };
        if status == ResponseStatus::Error {
            return Err(self.response_failure(
                sink,
                "feed",
                None,
                "response parser rejected generated bytes",
                AdmissionError::Refused("response bytes/frame validation failed").into(),
            ));
        }
        if self.request.stream {
            self.flush(session, sink)?;
        }
        Ok(())
    }

    /// The C++ `try` block of `complete`.
    fn run(
        &mut self,
        session: &mut ServeSession,
        mut target: ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
    ) -> Result<Json> {
        let target = &mut target;
        let request = self.request;
        let prompt = self.prompt;
        cancelled(sink)?;
        if let Some(trace) = sink.prefix_trace.as_mut() {
            let ids = Json::Array(
                prompt
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect(),
            );
            let prior = Json::Array(
                session
                    .committed
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect(),
            );
            let event = Json::new_object([
                ("event", Json::from("prompt_prefix")),
                ("schema", Json::integer(1)),
                ("response_id", Json::from(self.response_id)),
                ("prompt_ids", ids),
                ("prior_committed_ids", prior),
                (
                    "prior_committed_tokens",
                    Json::integer(session.committed.len() as u64),
                ),
                ("common_prefix", Json::integer(self.common as u64)),
                ("reusable", Json::Bool(session.reusable)),
                ("append", Json::Bool(self.append)),
                (
                    "sampling_matches",
                    Json::Bool(session.sampling == self.policy),
                ),
                (
                    "seed_matches",
                    Json::Bool(self.request.seed.is_none_or(|s| s == session.seed)),
                ),
                ("prompt_tokens", Json::integer(prompt.len() as u64)),
                ("cached_tokens", Json::integer(self.cached_prefix as u64)),
                (
                    "new_tokens",
                    Json::integer((prompt.len() - self.cached_prefix) as u64),
                ),
            ]);
            trace(&event)?;
        }
        self.emit(
            session,
            sink,
            Json::new_object([("role", Json::from("assistant"))]),
            Json::Null,
        )?;
        self.state_touched = true;
        let prefill_start = std::time::Instant::now();
        if session.phase_stamps_enabled {
            session.phase_stamps.prefill_start = Some(prefill_start);
        }
        self.position = if self.append {
            session.committed.len()
        } else {
            0
        };
        let mut began = false;
        let mut cached_prefix = self.cached_prefix;
        if session.broker_epoch != 0
            && !self.append
            && (session.branch8_mode()
                || session.mtp
                || number(field(&request.sampling, "temperature")?, 0.0, 100.0)? == 0.0)
        {
            let future = Json::Array(
                prompt
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect(),
            );
            let reuse = session.reusable;
            let job = Json::new_object([
                ("op", Json::from("branch_begin")),
                ("prompt_ids", future),
                ("generation_requested", Json::Bool(true)),
                ("allow_restore", Json::Bool(reuse)),
                ("sampling", request.sampling.clone()),
                ("seed", Json::integer(self.seed)),
            ]);
            let done = self.control(session, target, sink, job, "branch_begin", true)?;
            let prefix = field(&done, "committed_prefix")?.unsigned_integer()?;
            let lcp = field(&done, "common_prefix")?.unsigned_integer()?;
            let outcome = field(&done, "outcome")?.string()?;
            require(
                lcp == if reuse { self.common as u64 } else { 0 }
                    && prefix <= lcp
                    && ((outcome == b"cold_miss" && prefix == 0)
                        || (outcome == b"restored" && prefix != 0 && reuse)),
                "broker exact-LCP transaction mismatch",
            )?;
            require(
                prefix < prompt.len() as u64 || field(&done, "stop_id")?.is_null(),
                "terminal restored prompt cannot generate without new input",
            )?;
            // Reserved at preparation; shrinking cannot allocate after restore.
            session.committed.resize(prefix as usize, 0);
            self.position = prefix as usize;
            cached_prefix = self.position;
            began = true;
        } else if !self.append {
            session.committed.clear();
        }
        if !self.append {
            session.reusable = false;
        }
        let input_start = self.position;
        if session.mtp {
            while self.position < prompt.len() {
                cancelled(sink)?;
                let input_packet =
                    tokspur_wire::mtp_wire::mtp_input_packet(&session.mtp_ready)? as usize;
                let mut n = input_packet.min(prompt.len() - self.position);
                if session.overlap_prefill && prompt.len() - self.position > 16 {
                    let bulk_remaining = (prompt.len() - 16) - self.position;
                    n = n.min(bulk_remaining);
                }
                let ids = prompt[self.position..self.position + n]
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect();
                let fresh = !self.append && !began && self.position == 0;
                let mut job = vec![
                    (b"tokens".to_vec(), Json::Array(ids)),
                    (b"generate".to_vec(), Json::integer(0)),
                    (b"reset".to_vec(), Json::Bool(fresh)),
                ];
                if fresh {
                    job.push((b"sampling".to_vec(), request.sampling.clone()));
                    job.push((b"seed".to_vec(), Json::integer(self.seed)));
                }
                if tokspur_wire::mtp_wire::mtp_final_input_logits(&session.mtp_ready)? {
                    job.push((
                        b"prefill_final".to_vec(),
                        Json::Bool(self.position + n == prompt.len()),
                    ));
                }
                self.run_mtp(session, target, sink, Json::Object(job))?;
            }
        } else if session.verify8_mode() {
            #[cfg(feature = "verify8")]
            while self.position < prompt.len() {
                let mut n = 8.min(prompt.len() - self.position);
                if session.branch8_mode() {
                    n = n.min(512 - self.position % 512); // Never step over a useful ancestor boundary.
                }
                let ids = prompt[self.position..self.position + n]
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect();
                let fresh = !self.append && !began && self.position == 0;
                let mut job = vec![
                    (b"tokens".to_vec(), Json::Array(ids)),
                    (b"generate".to_vec(), Json::integer(0)),
                    (b"reset".to_vec(), Json::Bool(fresh)),
                ];
                if fresh {
                    job.push((b"sampling".to_vec(), request.sampling.clone()));
                    job.push((b"seed".to_vec(), Json::integer(self.seed)));
                    job.push((b"prompt_lookup".to_vec(), Json::Bool(true)));
                }
                self.run_many(session, target, sink, Json::Object(job))?;
                if session.branch8_mode()
                    && (self.position.is_multiple_of(512) || self.position == prompt.len())
                {
                    let capture = Json::new_object([("op", Json::from("branch_capture"))]);
                    let captured =
                        self.control(session, target, sink, capture, "branch_capture", false)?;
                    require(
                        is(field(&captured, "outcome")?, "captured")?
                            && field(&captured, "committed_prefix")?.unsigned_integer()?
                                == self.position as u64
                            && field(&captured, "common_prefix")?.unsigned_integer()?
                                == self.position as u64,
                        "branch8 scheduled capture mismatch",
                    )?;
                }
            }
        } else {
            for (i, &token) in prompt.iter().enumerate().skip(self.position) {
                let fresh = !self.append && !began && i == 0;
                let mut job = vec![
                    (
                        b"tokens".to_vec(),
                        Json::Array(vec![Json::integer(u64::from(token))]),
                    ),
                    (b"generate".to_vec(), Json::integer(0)),
                    (b"reset".to_vec(), Json::Bool(fresh)),
                ];
                if fresh {
                    job.push((b"sampling".to_vec(), request.sampling.clone()));
                    job.push((b"seed".to_vec(), Json::integer(self.seed)));
                }
                let done =
                    self.step(session, target, sink, Json::Object(job), Some(token), false)?;
                require(
                    is(field(&done, "status")?, "prefilled")?
                        && field(&done, "finish_reason")?.is_null()
                        && field(&done, "stop_id")?.is_null(),
                    "invalid prefill termination",
                )?;
            }
        }
        if session.broker_epoch != 0 && self.position > input_start && !session.branch8_mode() {
            let capture = Json::new_object([("op", Json::from("branch_capture"))]);
            let done = self.control(session, target, sink, capture, "branch_capture", false)?;
            require(
                is(field(&done, "outcome")?, "captured")?
                    && field(&done, "committed_prefix")?.unsigned_integer()?
                        == self.position as u64
                    && field(&done, "common_prefix")?.unsigned_integer()? == self.position as u64,
                "broker input-boundary capture mismatch",
            )?;
        }
        // Active/broker reuse precedes the submitted input packets. Each admitted packet contributes only its own restored suffix, once per request.
        require(
            self.counters.restored_input_tokens <= (prompt.len() - cached_prefix) as u64,
            "restored input accounting exceeds uncached prompt",
        )?;
        cached_prefix += self.counters.restored_input_tokens as usize;
        let admission = session.codec.admission;
        let stop_ids = || {
            Json::Array(vec![
                Json::integer(u64::from(admission.im_end)),
                Json::integer(u64::from(admission.end_of_text)),
            ])
        };
        let generation_start = std::time::Instant::now();
        if session.phase_stamps_enabled {
            session.phase_stamps.generation_start = Some(generation_start);
        }
        let mut stopped = false;
        if session.mtp {
            session.mtp_committed = 0;
            session.mtp_emitted = 0;
            session.mtp_pending = 0;
            session.mtp_stage_a_controller = None;
            let scoped_carry = tokspur_wire::mtp_wire::mtp_request_carry(&session.mtp_ready)?;
            if scoped_carry {
                require(
                    session.mtp_generation_id < u64::MAX,
                    "MTP generation ID exhausted",
                )?;
                session.mtp_generation_id += 1;
            }
            let output_packet =
                tokspur_wire::mtp_wire::mtp_output_packet(&session.mtp_ready)? as usize;
            while self.generated.len() < self.maximum_tokens as usize {
                let packet_cap = match session.mtp_first_output_packet {
                    Some(first_pkt) if self.generated.is_empty() => {
                        (first_pkt as usize).min(output_packet)
                    }
                    _ => output_packet,
                };
                let n = packet_cap.min(self.maximum_tokens as usize - self.generated.len());
                let mut job = vec![
                    (b"tokens".to_vec(), Json::Array(Vec::new())),
                    (b"generate".to_vec(), Json::integer(n as u64)),
                    (b"reset".to_vec(), Json::Bool(false)),
                    (b"stop_ids".to_vec(), stop_ids()),
                ];
                if scoped_carry {
                    let is_v2 =
                        if let Some(scope) = session.mtp_ready.find("native_mtp_carry_scope")? {
                            scope.string()? == b"http-generation-v2"
                        } else {
                            false
                        };
                    let remaining = (self.maximum_tokens as usize - self.generated.len()) as u64;
                    let gen_obj = if is_v2 {
                        let total_budget = self.maximum_tokens as u64;
                        let committed = session.mtp_committed;
                        let emitted = session.mtp_emitted;
                        let pending = session.mtp_pending;
                        let remaining_compute = total_budget.saturating_sub(committed);
                        let mut fields: Vec<(&str, Json)> = vec![
                            ("id", Json::integer(session.mtp_generation_id)),
                            ("begin", Json::Bool(self.generated.is_empty())),
                            ("remaining", Json::integer(remaining)),
                            ("committed", Json::integer(committed)),
                            ("emitted", Json::integer(emitted)),
                            ("pending", Json::integer(pending)),
                            ("remaining_compute", Json::integer(remaining_compute)),
                            ("total_budget", Json::integer(total_budget)),
                        ];
                        let ctrl_obj;
                        if let Some(ref ctrl) = session.mtp_stage_a_controller {
                            let digest_arr: Vec<Json> = ctrl
                                .controller_policy_digest
                                .iter()
                                .map(|&b| Json::integer(b as u64))
                                .collect();
                            ctrl_obj = Json::new_object([
                                ("ewma_alpha1", real(ctrl.ewma_alpha1)?),
                                ("current_mode", Json::integer(ctrl.current_mode as u64)),
                                (
                                    "consecutive_ordinary_steps",
                                    Json::integer(ctrl.consecutive_ordinary_steps as u64),
                                ),
                                ("controller_policy_digest", Json::Array(digest_arr)),
                            ]);
                            let ctrl_obj = if let Some(measured) = ctrl.measured {
                                let mut object = ctrl_obj.object()?.to_vec();
                                object.push((b"measured".to_vec(), measured.json()?));
                                Json::Object(object)
                            } else {
                                ctrl_obj
                            };
                            fields.push(("stage_a_controller", ctrl_obj));
                        }
                        Json::new_object(fields)
                    } else {
                        Json::new_object([
                            ("id", Json::integer(session.mtp_generation_id)),
                            ("begin", Json::Bool(self.generated.is_empty())),
                            ("remaining", Json::integer(remaining)),
                        ])
                    };
                    job.push((b"mtp_generation".to_vec(), gen_obj));
                }
                let done = self.run_mtp(session, target, sink, Json::Object(job))?;
                for token in field(&done, "generated_ids")?.array()? {
                    let id = token.unsigned_integer()? as u32;
                    require(
                        id != admission.im_start && !stopped,
                        "MTP unexpected generated role/terminal",
                    )?;
                    stopped = admission.is_stop(id);
                    self.generated.push(id);
                    self.feed(session, sink, id)?;
                }
                if stopped {
                    break;
                }
            }
        } else if session.verify8_mode() {
            #[cfg(feature = "verify8")]
            while self.generated.len() < self.maximum_tokens as usize {
                let n = 8.min(self.maximum_tokens as usize - self.generated.len());
                let job = Json::new_object([
                    ("tokens", Json::Array(Vec::new())),
                    ("generate", Json::integer(n as u64)),
                    ("reset", Json::Bool(false)),
                    ("stop_ids", stop_ids()),
                ]);
                let done = self.run_many(session, target, sink, job)?;
                for token in field(&done, "generated_ids")?.array()? {
                    let id = token.unsigned_integer()? as u32;
                    require(
                        id != admission.im_start && !stopped,
                        "private V8 unexpected generated role/terminal",
                    )?;
                    stopped = admission.is_stop(id);
                    self.generated.push(id);
                    self.feed(session, sink, id)?;
                }
                if stopped {
                    break;
                }
            }
        } else {
            for _ in 0..self.maximum_tokens {
                let job = Json::new_object([
                    ("tokens", Json::Array(Vec::new())),
                    ("generate", Json::integer(1)),
                    ("reset", Json::Bool(false)),
                    ("stop_ids", stop_ids()),
                ]);
                let done = self.step(session, target, sink, job, None, true)?;
                let id = session.committed.last().copied().unwrap_or(0);
                require(
                    id != admission.im_start,
                    "unexpected generated role/terminal token",
                )?;
                stopped = admission.is_stop(id);
                require(
                    is(
                        field(&done, "finish_reason")?,
                        if stopped { "stop" } else { "length" },
                    )? && is(
                        field(&done, "status")?,
                        if stopped { "stopped" } else { "length" },
                    )?,
                    "worker termination mismatch",
                )?;
                if stopped {
                    require(
                        field(&done, "stop_id")?.unsigned_integer()? == u64::from(id)
                            && !field(&done, "continuation_allowed")?.boolean()?,
                        "invalid terminal admission",
                    )?;
                } else {
                    require(
                        field(&done, "stop_id")?.is_null()
                            && field(&done, "continuation_allowed")?.boolean()?,
                        "invalid continuation admission",
                    )?;
                }
                self.generated.push(id);
                self.feed(session, sink, id)?;
                if stopped {
                    break;
                }
            }
        }
        // Nonstreaming still validates each token above. Assemble its parser
        // records once for the same final prefix check used by streaming.
        if !request.stream {
            self.flush(session, sink)?;
        }
        let ids = Json::Array(
            self.generated
                .iter()
                .map(|&id| Json::integer(u64::from(id)))
                .collect(),
        );
        let mut final_fields = vec![
            (b"ids".to_vec(), ids),
            (b"tools".to_vec(), parser_manifest(&request.manifest)),
            (b"initial_thinking".to_vec(), Json::Bool(request.thinking)),
            (
                b"finish_reason".to_vec(),
                Json::from(if stopped { "stop" } else { "length" }),
            ),
            (
                b"stop_id".to_vec(),
                if stopped {
                    Json::integer(u64::from(*self.generated.last().ok_or(
                        AdmissionError::Refused("stopped response lacks terminal token"),
                    )?))
                } else {
                    Json::Null
                },
            ),
        ];
        if session.tool_args == ToolArgValidation::Executor {
            final_fields.push((b"executor_parameters".to_vec(), Json::Bool(true)));
        }
        let final_request = Json::Object(final_fields);
        let decode = |ids: &[u32]| (session.codec.decode)(ids);
        let finished = parse_native_response_with_controls(
            &final_request,
            admission,
            Some(&decode),
            RESPONSE_MAXIMUM_ACCUMULATED_IDS,
            &session.codec.control_ids,
        )
        .map_err(|error| {
            if sink.prefix_trace.is_some() && stopped {
                // Only after the authoritative accumulated parser has refused: finalise the existing incremental parser to recover its detailed reason without changing admission or success paths.
                self.parser.finish();
            }
            let reason = match &error {
                ResponseProtocolError::Admit(reason) => *reason,
                ResponseProtocolError::Access(_) => "final response field validation failed",
                ResponseProtocolError::Configure(_) => "final response parser configuration failed",
                ResponseProtocolError::Decode(_) => "final response decoding failed",
            };
            self.response_failure(
                sink,
                "finish",
                Some(if stopped { "stop" } else { "length" }),
                reason,
                ServeError::Text(Box::new(error)),
            )
        })?;
        require(
            !request.single_tool_call || field(&finished, "tool_calls")?.array()?.len() <= 1,
            "parallel tool calls are disabled for this request",
        )?;
        let mut calls: Vec<Json> = Vec::new();
        for call in field(&finished, "tool_calls")?.array()? {
            let function = field(call, "function")?;
            let name = field(function, "name")?.string()?;
            let Some(index) = request
                .names
                .iter()
                .position(|candidate| candidate.as_slice() == name)
            else {
                return Err(AdmissionError::Refused("unadmitted final tool").into());
            };
            if session.tool_args == ToolArgValidation::Executor {
                validate_executor_tool_arguments(field(function, "arguments")?)?;
            } else {
                validate_tool_value_with_policy(
                    &request.schemas[index],
                    field(function, "arguments")?,
                    session.tool_enums,
                )?;
            }
            let mut id = self.response_id.to_vec();
            id.extend_from_slice(format!("_call_{}", calls.len()).as_bytes());
            calls.push(Json::new_object([
                ("id", Json::String(id)),
                ("type", Json::from("function")),
                (
                    "function",
                    Json::new_object([
                        ("name", Json::from(name)),
                        (
                            "arguments",
                            Json::String(field(function, "arguments")?.dump()?),
                        ),
                    ]),
                ),
            ]));
        }
        let choice = request.tool_choice.as_slice();
        require(
            (!request.named_tool_choice && (choice == b"auto" || choice == b"none"))
                || !stopped
                || !calls.is_empty(),
            "required tool was not produced",
        )?;
        cancelled(sink)?;
        let content = field(&finished, "content")?.string()?.to_vec();
        let thinking = field(&finished, "reasoning_content")?.string()?.to_vec();
        require(
            content.starts_with(&self.streamed_content)
                && thinking.starts_with(&self.streamed_thinking),
            "final response disagrees with streamed prefix",
        )?;
        if thinking.len() > self.streamed_thinking.len() {
            let delta = Json::new_object([(
                "reasoning_content",
                Json::from(&thinking[self.streamed_thinking.len()..]),
            )]);
            self.emit(session, sink, delta, Json::Null)?;
        }
        if content.len() > self.streamed_content.len() {
            let delta = Json::new_object([(
                "content",
                Json::from(&content[self.streamed_content.len()..]),
            )]);
            self.emit(session, sink, delta, Json::Null)?;
        }
        for (i, call) in calls.iter().enumerate() {
            let mut fields = call.object()?.clone();
            fields.push((b"index".to_vec(), Json::integer(i as u64)));
            self.emit(
                session,
                sink,
                Json::new_object([("tool_calls", Json::Array(vec![Json::Object(fields)]))]),
                Json::Null,
            )?;
        }
        let finish = if !calls.is_empty() {
            "tool_calls"
        } else if stopped {
            "stop"
        } else {
            "length"
        };
        self.emit(session, sink, Json::Object(Vec::new()), Json::from(finish))?;
        session.sampling = self.policy.to_vec();
        session.seed = self.seed;
        session.reusable = true;
        if !self.append {
            session.saved_reasoning.clear();
        }
        if !thinking.is_empty() {
            let mut ordinal = 0;
            for message in field(&request.render, "messages")?.array()? {
                ordinal += usize::from(field(message, "role")?.string()? == b"assistant");
            }
            session.saved_reasoning.push((ordinal, thinking.clone()));
        }
        let generation_end = std::time::Instant::now();
        if session.phase_stamps_enabled {
            session.phase_stamps.generation_end = Some(generation_end);
        }
        let admission_ns = prefill_start
            .saturating_duration_since(self.request_start)
            .as_nanos() as u64;
        let prefill_ns = generation_start
            .saturating_duration_since(prefill_start)
            .as_nanos() as u64;
        let generation_ns = generation_end
            .saturating_duration_since(generation_start)
            .as_nanos() as u64;
        let server_admission_wall_ms = admission_ns as f64 / 1e6;
        let server_prefill_wall_ms = prefill_ns as f64 / 1e6;
        let server_generation_wall_ms = generation_ns as f64 / 1e6;
        let server_request_wall_ms =
            server_admission_wall_ms + server_prefill_wall_ms + server_generation_wall_ms;
        let server_generation_throughput =
            if !self.generated.is_empty() && server_generation_wall_ms > 0.0 {
                (self.generated.len() as f64 * 1000.0) / server_generation_wall_ms
            } else {
                0.0
            };
        let generated = self.generated.len() as u64;
        let prompt_tokens = prompt.len() as u64;
        let prefill_generation_overlap = if self.counters.overlap_pipelined {
            "16_token_b10_pipelined"
        } else {
            "none"
        };
        let mut message = vec![
            (b"role".to_vec(), Json::from("assistant")),
            (b"content".to_vec(), Json::String(content)),
        ];
        if !thinking.is_empty() {
            message.push((b"reasoning_content".to_vec(), Json::String(thinking)));
        }
        if !calls.is_empty() {
            message.push((b"tool_calls".to_vec(), Json::Array(calls)));
        }
        let usage = Json::new_object([
            ("prompt_tokens", Json::integer(prompt_tokens)),
            ("completion_tokens", Json::integer(generated)),
            ("total_tokens", Json::integer(prompt_tokens + generated)),
            (
                "prompt_tokens_details",
                Json::new_object([("cached_tokens", Json::integer(cached_prefix as u64))]),
            ),
        ]);
        let result = vec![
            (b"id".to_vec(), Json::from(self.response_id)),
            (b"object".to_vec(), Json::from("chat.completion")),
            (b"created".to_vec(), Json::integer(self.created)),
            (b"model".to_vec(), Json::from(session.alias.clone())),
            (
                b"choices".to_vec(),
                Json::Array(vec![Json::new_object([
                    ("index", Json::integer(0)),
                    ("message", Json::Object(message)),
                    ("finish_reason", Json::from(finish)),
                ])]),
            ),
            (b"usage".to_vec(), usage.clone()),
        ];
        if let Some(trace) = sink.prefix_trace.as_mut() {
            let fields: Vec<(&str, Json)> = vec![
                ("event", Json::from("completion")),
                ("schema", Json::integer(1)),
                ("response_id", Json::from(self.response_id)),
                ("prompt_tokens", Json::integer(prompt_tokens)),
                ("cached_tokens", Json::integer(cached_prefix as u64)),
                (
                    "new_tokens",
                    Json::integer(prompt_tokens - cached_prefix as u64),
                ),
                ("completion_tokens", Json::integer(generated)),
                ("total_tokens", Json::integer(prompt_tokens + generated)),
                ("finish_reason", Json::from(finish)),
                ("stopped_on_eos", Json::Bool(stopped)),
                ("server_admission_wall_ms", real(server_admission_wall_ms)?),
                ("server_prefill_wall_ms", real(server_prefill_wall_ms)?),
                (
                    "server_generation_wall_ms",
                    real(server_generation_wall_ms)?,
                ),
                ("server_request_wall_ms", real(server_request_wall_ms)?),
                (
                    "server_generation_throughput",
                    real(server_generation_throughput)?,
                ),
                (
                    "prefill_generation_overlap",
                    Json::from(prefill_generation_overlap),
                ),
            ];
            let mut byte_fields: Vec<(Vec<u8>, Json)> = fields
                .into_iter()
                .map(|(k, v)| (k.as_bytes().to_vec(), v))
                .collect();
            if session.mtp_previous.kind() == Kind::Object {
                for trace_key in [
                    "peak_metal_buffer_bytes",
                    "memory",
                    "ple_cache",
                    "stateless_prefill_ple_rows",
                    "checkpoint_ssd_enabled",
                    "checkpoint_ssd_occupied_bytes",
                    "checkpoint_ssd_spills",
                    "checkpoint_ssd_restores",
                    "checkpoint_ssd_restored_tokens",
                ] {
                    if let Some(v) = session.mtp_previous.find(trace_key)? {
                        byte_fields.push((trace_key.as_bytes().to_vec(), v.clone()));
                    }
                }
            }
            let trace_event = Json::Object(byte_fields);
            trace(&trace_event)?;
        }
        if self.request.stream && self.request.include_usage {
            cancelled(sink)?;
            let final_chunk = Json::new_object([
                ("id", Json::from(self.response_id)),
                ("object", Json::from("chat.completion.chunk")),
                ("created", Json::integer(self.created)),
                ("model", Json::from(session.alias.clone())),
                ("choices", Json::Array(Vec::new())),
                ("usage", usage),
            ]);
            if let Some(send) = sink.chunk.as_mut() {
                send(&final_chunk)?;
            }
        }
        let result = self.private_details(session, result)?;
        Ok(Json::Object(result))
    }

    /// `tokspur_private`: scheduling counters, never a performance claim.
    fn private_details(
        &self,
        session: &ServeSession,
        mut result: Vec<(Vec<u8>, Json)>,
    ) -> Result<Vec<(Vec<u8>, Json)>> {
        if session.mtp {
            let c = &self.counters;
            let mut details = vec![
                (b"native_mtp".to_vec(), Json::Bool(true)),
                (
                    b"native_mtp_protocol".to_vec(),
                    field(&session.mtp_ready, "native_mtp_protocol")?.clone(),
                ),
                (
                    b"known_input_batches".to_vec(),
                    Json::integer(c.input_batches),
                ),
                (
                    b"maximum_known_input_width".to_vec(),
                    Json::integer(c.input_width as u64),
                ),
                (
                    b"generation_batches".to_vec(),
                    Json::integer(c.output_batches),
                ),
                (
                    b"maximum_generation_width".to_vec(),
                    Json::integer(c.output_width as u64),
                ),
                (b"mtp_rounds".to_vec(), Json::integer(c.generation_rounds)),
                (
                    b"mtp_accepted_tokens".to_vec(),
                    Json::integer(c.accepted_tokens),
                ),
                (b"performance_qualified".to_vec(), Json::Bool(false)),
            ];
            if let Ok(512 | 1024 | 2048 | 4096 | 8192) =
                tokspur_wire::mtp_wire::mtp_input_packet(&session.mtp_ready)
            {
                for name in &[
                    "native_mtp_input_packet_schema",
                    "native_mtp_input_packet_max",
                    "native_mtp_priming_batch_capacity",
                    "native_mtp_priming_batch_bytes",
                    "native_prefill_dense",
                ] {
                    if let Ok(v) = field(&session.mtp_ready, name) {
                        details.push((name.as_bytes().to_vec(), v.clone()));
                    }
                }
                for name in &[
                    "native_prefill_grouped",
                    "native_prefill_grouped_kernel",
                    "native_prefill_dense_kernel",
                ] {
                    if let Ok(v) = field(&session.mtp_ready, name) {
                        details.push((name.as_bytes().to_vec(), v.clone()));
                    }
                }
            }
            if let Some(first_pkt) = session.mtp_first_output_packet {
                details.push((
                    b"native_mtp_first_output_packet".to_vec(),
                    Json::integer(u64::from(first_pkt)),
                ));
            }
            result.push((b"tokspur_private".to_vec(), Json::Object(details)));
        } else {
            #[cfg(feature = "verify8")]
            if session.verify8 {
                let c = &self.counters;
                let details = vec![
                    (b"native_verify8".to_vec(), Json::Bool(true)),
                    (b"verification_max_tokens".to_vec(), Json::integer(8)),
                    (
                        b"known_input_batches".to_vec(),
                        Json::integer(c.input_batches),
                    ),
                    (
                        b"maximum_known_input_width".to_vec(),
                        Json::integer(c.input_width as u64),
                    ),
                    (
                        b"generation_batches".to_vec(),
                        Json::integer(c.output_batches),
                    ),
                    (
                        b"maximum_generation_width".to_vec(),
                        Json::integer(c.output_width as u64),
                    ),
                    (
                        b"lookup_rounds".to_vec(),
                        Json::integer(c.generation_rounds),
                    ),
                    (
                        b"lookup_verified_rounds".to_vec(),
                        Json::integer(c.verified_rounds),
                    ),
                    (
                        b"lookup_accepted_tokens".to_vec(),
                        Json::integer(c.accepted_tokens),
                    ),
                    (
                        b"hc_exact_final_logit_sha256".to_vec(),
                        field(&session.verify_previous, "hc_exact_final_logit_sha256")?.clone(),
                    ),
                    (b"performance_qualified".to_vec(), Json::Bool(false)),
                ];
                #[cfg(feature = "branch8")]
                let details = self.branch8_details(session, details)?;
                result.push((b"tokspur_private".to_vec(), Json::Object(details)));
            }
        }
        Ok(result)
    }

    /// The branch8 additions to `tokspur_private`.
    #[cfg(feature = "branch8")]
    fn branch8_details(
        &self,
        session: &ServeSession,
        mut details: Vec<(Vec<u8>, Json)>,
    ) -> Result<Vec<(Vec<u8>, Json)>> {
        let c = &self.counters;
        {
            if session.branch8 {
                details.push((b"branch8_reuse".to_vec(), Json::Bool(true)));
                details.push((b"broker_protocol".to_vec(), Json::from("branch8-v1")));
                details.push((
                    b"checkpoint_captures".to_vec(),
                    Json::integer(c.checkpoint_captures),
                ));
                details.push((
                    b"checkpoint_restores".to_vec(),
                    Json::integer(c.checkpoint_restores),
                ));
                details.push((
                    b"checkpoint_cold_resyncs".to_vec(),
                    Json::integer(c.checkpoint_cold_resyncs),
                ));
                details.push((
                    b"checkpoint_slots".to_vec(),
                    field(&session.verify_ready, "checkpoint_slots")?.clone(),
                ));
            }
        }
        Ok(details)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec() -> ServeCodec {
        ServeCodec {
            control_ids: Vec::new(),
            render: Box::new(|_| Err(AdmissionError::Refused("unused").into())),
            encode: Box::new(|_| Ok(Vec::new())),
            decode: Box::new(|_| Ok(Vec::new())),
            admission: ResponseAdmission {
                vocabulary: 300,
                im_end: 1,
                end_of_text: 2,
                im_start: 3,
            },
        }
    }

    #[test]
    fn session_debug_counts_committed_tokens_without_their_ids() {
        // Arrange
        let mut session = ServeSession::new(codec(), SessionOptions::default()).unwrap();
        session.committed = vec![151_644, 872, 198];

        // Act
        let rendered = format!("{session:?}");

        // Assert
        assert_eq!(
            rendered,
            "ServeSession { committed_tokens: 3, reusable: false, .. }"
        );
        assert!(!rendered.contains("151644") && !rendered.contains("872"));
    }

    /// What `flush` did before the cursor: rebuild both streams from every record.
    fn rebuilt(records: &[ResponseRecord]) -> (Vec<u8>, Vec<u8>) {
        let mut thinking = Vec::new();
        let mut content = Vec::new();
        for record in records {
            match record.kind {
                ResponseKind::Text => content.extend_from_slice(&record.text),
                ResponseKind::Thinking => thinking.extend_from_slice(&record.text),
                ResponseKind::Tool => {}
            }
        }
        (thinking, content)
    }

    #[test]
    fn stream_cursor_emits_the_same_deltas_as_a_full_rebuild() {
        // Arrange: thinking, text either side of a withheld tool call, multi-byte UTF-8 and a held `<` marker prefix.
        let response =
            "plan 日本 <b\n</think>answer 🙂 a<b <tool_call><function=ping></function></tool_call>";
        let tools = vec![tokspur_text::response::ToolSpec::new("ping", Vec::new())];
        let mut parser = ResponseParser::new(tools, true, ResponseLimits::default()).unwrap();
        let mut cursor = StreamCursor::default();
        let (mut thinking, mut content) = (Vec::new(), Vec::new());

        // Act: one byte per feed, the smallest step the cursor can see.
        for byte in response.bytes() {
            assert_eq!(parser.feed(&[byte]), ResponseStatus::Incomplete);
            let before = (thinking.len(), content.len());
            let delta = cursor.advance(parser.records()).unwrap();
            thinking.extend_from_slice(&delta.thinking);
            content.extend_from_slice(&delta.content);

            // Assert: every step matches the rebuild, byte for byte.
            let (all_thinking, all_content) = rebuilt(parser.records());
            assert_eq!(delta.thinking, all_thinking[before.0..]);
            assert_eq!(delta.content, all_content[before.1..]);
        }
        assert!(!thinking.is_empty() && !content.is_empty());
        assert_eq!(parser.records().len(), 2);
    }

    #[test]
    fn stream_cursor_refuses_records_that_shrink() {
        // Arrange
        let mut parser = ResponseParser::new(Vec::new(), false, ResponseLimits::default()).unwrap();
        parser.feed(b"hello world");
        let mut cursor = StreamCursor::default();
        cursor.advance(parser.records()).unwrap();
        let mut shorter = parser.records().to_vec();
        shorter[0].text.truncate(5);

        // Act + Assert
        assert!(cursor.advance(&shorter).is_err());
        assert!(cursor.advance(&[]).is_err());
    }

    #[test]
    fn codec_debug_shows_no_fields() {
        // Act
        let rendered = format!("{:?}", codec());

        // Assert
        assert_eq!(rendered, "ServeCodec { .. }");
    }

    #[test]
    fn omitted_reasoning_requires_an_exact_prefix_and_prework_failure_preserves_state() {
        let encode = |text: &[u8]| text.iter().map(|b| u32::from(*b) + 4).collect::<Vec<_>>();
        let mut text = codec();
        text.encode = Box::new(move |raw| Ok(encode(&raw.prompt)));
        text.render = Box::new(|request| {
            let mut prompt = Vec::new();
            for message in field(request, "messages")?.array()? {
                let assistant = field(message, "role")?.string()? == b"assistant";
                if assistant && let Some(reasoning) = message.find("reasoning_content")? {
                    prompt.extend_from_slice(b"[reasoning:");
                    prompt.extend_from_slice(reasoning.string()?);
                    prompt.push(b']');
                }
                prompt.extend_from_slice(field(message, "content")?.string()?);
                if assistant {
                    prompt.push(b';');
                }
            }
            Ok(RenderedChat {
                prompt,
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        });
        let mut session = ServeSession::new(text, SessionOptions::default()).unwrap();
        session.reusable = true;
        session.committed = encode(b"A[reasoning:why]Visible;");
        session.saved_reasoning.push((0, b"why".to_vec()));
        for (first, answer, expected_append) in [
            ("A", "Visible", true),
            ("X", "Visible", false),
            ("A", "Edited", false),
        ] {
            let input = tokspur_json::parse_json(format!(r#"{{"model":"tokspur","messages":[{{"role":"user","content":"{first}"}},{{"role":"assistant","content":"{answer}"}},{{"role":"user","content":"B"}}]}}"#).as_bytes()).unwrap();
            let admitted = admit_chat_with_policies(
                &input,
                b"tokspur",
                ToolEnumValidation::Strict,
                ToolArgValidation::Strict,
            )
            .unwrap();
            session.sampling = admitted.sampling.dump().unwrap();
            let before = session.committed.clone();
            let mut observed = Json::Null;
            let mut trace = |value: &Json| {
                observed = value.clone();
                Err(AdmissionError::Refused("controlled trace refusal").into())
            };
            let mut cancelled = || false;
            let mut seed = || 42;
            let mut sink = ServeSink {
                chunk: None,
                cancelled: &mut cancelled,
                new_seed: &mut seed,
                prefix_trace: Some(&mut trace),
            };
            let mut target = |_: &Json, _: TokenCallback<'_>| -> Result<Json> {
                panic!("prework refusal submitted a worker command")
            };
            assert!(
                session
                    .complete(&input, b"test", &mut target, &mut sink)
                    .is_err()
            );
            assert_eq!(
                field(&observed, "append").unwrap().boolean().unwrap(),
                expected_append
            );
            let ids = field(&observed, "prompt_ids")
                .unwrap()
                .array()
                .unwrap()
                .iter()
                .map(|value| value.unsigned_integer().unwrap() as u32)
                .collect::<Vec<_>>();
            assert_eq!(ids.starts_with(&before), expected_append);
            if !expected_append {
                assert!(!ids.windows(3).any(|part| part == encode(b"why")));
            }
            assert_eq!(session.committed, before);
            assert!(session.reusable);
        }
    }
}
