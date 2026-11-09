//! The owned worker's command path, mirrored from `tokspur-wire/src/worker.rs` (`OwnedWorker::request`, `admit`, `submit`, `control_reply`, `broker_storage`, `broker_identity`, `invalidate`, `alive`) statement for statement, with the target pipes replaced by Shuttle channels.
//!
//! Not mirrored: process launch, the guardian, byte framing and JSON line parsing (the channels carry parsed values), the status pipe, and time (step, request and drain deadlines). A target that never answers is therefore not modelled; every model target answers every command.
//!
//! The V8/branch8 gate (`Verify8Reply`) and `validate_branch8_control` are the real `tokspur_wire::verify8` code, as in the owned worker.

use std::collections::VecDeque;

use shuttle::sync::mpsc::{Receiver, Sender, TryRecvError};
use tokspur_json::{Json, JsonError, Kind as JsonKind, from_chars_f64};
#[cfg(feature = "verify8")]
use tokspur_wire::verify8::{Verify8Context, Verify8Error, Verify8Reply};
use tokspur_wire::worker::{CallbackFailed, CancelProbe, TokenSink, WorkerError};

use crate::log::{Event, Identity, Kind, Log, Outcome};

type Result<T> = std::result::Result<T, WorkerError>;

fn require(ok: bool, message: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(WorkerError::Refused(message))
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
        Some(event) => event.kind() == JsonKind::String && event.string()? == name.as_bytes(),
        None => false,
    })
}

/// Which owned worker the session talks to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The mirror as written.
    Faithful,
    /// No worker-side reply checks: every event is forwarded as read and the first done of either kind is returned, so a misbehaving target reaches the session.
    Transparent,
    /// Seeded bug: returns on cancellation without draining the command or checking for leftover events.
    SkipDrain,
    /// Seeded bug: after a drained cancellation, forgets the required reset and resends the command once.
    Retry,
}

/// The `WorkerOptions` fields the command path reads.
pub struct Options {
    pub checkpoint_slots: u32,
    pub checkpoint_budget_bytes: u64,
    #[cfg(feature = "verify8")]
    pub verify8: bool,
    #[cfg(feature = "branch8")]
    pub branch8: bool,
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

impl From<CallbackFailed> for CommandError {
    fn from(error: CallbackFailed) -> Self {
        Self::Cause(Some(Box::new(error)))
    }
}

/// One stamped command and everything its drain needs.
struct Submission<'s, 'a> {
    body: &'s Json,
    wire: Json,
    admitted: &'s Admitted,
    command: u64,
    lease: u64,
    before: u32,
    #[cfg(feature = "verify8")]
    verify: Option<&'s mut Verify8Reply<'a>>,
    #[cfg(feature = "verify8")]
    verify_inputs: Option<&'s (Json, u64)>,
    #[cfg(not(feature = "verify8"))]
    _snapshots: std::marker::PhantomData<&'a ()>,
}

/// The mirrored `OwnedWorker`.
pub struct Wire {
    commands: Option<Sender<Json>>,
    events: Option<Receiver<Json>>,
    /// Events read from the target and not yet processed (`io.pending`).
    pending: VecDeque<Json>,
    eof: bool,
    failed: bool,
    closed: bool,
    options: Options,
    ready_value: Json,
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
    variant: Variant,
    log: Log,
    /// The request whose job is running, for the log.
    pub request: usize,
    /// The command the current submission wrote, until it resolves.
    written: Option<u64>,
}

impl Wire {
    /// The owned worker after `start` admitted the ready event.
    pub fn new(
        ready: Json,
        options: Options,
        commands: Sender<Json>,
        events: Receiver<Json>,
        variant: Variant,
        log: Log,
    ) -> Result<Self> {
        let mut wire = Self {
            commands: Some(commands),
            events: Some(events),
            pending: VecDeque::new(),
            eof: false,
            failed: false,
            closed: false,
            options,
            ready_value: ready,
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
            variant,
            log,
            request: 0,
            written: None,
        };
        // The tail of `start` after the ready event.
        #[cfg(feature = "verify8")]
        if wire.options.verify8 {
            if wire.branch8_mode() {
                #[cfg(feature = "branch8")]
                tokspur_wire::verify8::validate_branch8_ready(&wire.ready_value)?;
                wire.verify_previous = wire.ready_value.clone();
            } else {
                tokspur_wire::verify8::validate_verify8_ready(&wire.ready_value)?;
            }
        }
        if wire.options.checkpoint_slots != 0 {
            require(
                worker_field(&wire.ready_value, "checkpoint_broker")?.boolean()?,
                "owned target did not prepare broker controls",
            )?;
            wire.broker_storage(&wire.ready_value)?;
            wire.broker_epoch = worker_integer(&wire.ready_value, "broker_epoch")?;
            require(wire.broker_epoch != 0, "owned broker epoch is invalid")?;
        }
        Ok(wire)
    }

    /// Whether the worker-side reply checks run.
    fn checks(&self) -> bool {
        self.variant != Variant::Transparent
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

    /// `io.pump` with a zero wait: take whatever the target already wrote.
    fn poll(&mut self) {
        let Some(events) = self.events.as_ref() else {
            self.eof = true;
            return;
        };
        loop {
            match events.try_recv() {
                Ok(value) => self.pending.push_back(value),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.eof = true;
                    return;
                }
            }
        }
    }

    /// `io.pump` until the target writes one more event, or its pipe closes.
    fn pump(&mut self) {
        match self.events.as_ref().map(Receiver::recv) {
            Some(Ok(value)) => self.pending.push_back(value),
            _ => self.eof = true,
        }
    }

    /// `io.idle_check`.
    fn idle_check(&mut self) -> Result<()> {
        self.poll();
        require(
            !self.failed && !self.closed && !self.eof && self.pending.is_empty(),
            "owned worker is not drained and ready for another command",
        )
    }

    /// `invalidate()`: fail, require a reset, and kill and drain the target group (`io.shutdown`, here dropping both pipes).
    fn invalidate(&mut self) {
        self.failed = true;
        self.needs_reset = true;
        self.commands = None;
        self.events = None;
        self.pending.clear();
        self.log.push(Event::Killed);
    }

    /// No failure, close or EOF (the status pipe is not modelled).
    pub fn alive(&self) -> bool {
        !self.failed && !self.closed && !self.eof
    }

    /// `close()` once the generation thread has stopped.
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.commands = None;
        self.events = None;
        self.log.push(Event::Closed);
    }

    /// Validates, submits and drains one command.
    pub fn request(
        &mut self,
        body: &Json,
        mut cancelled: Option<CancelProbe<'_>>,
        mut on_token: Option<TokenSink<'_>>,
    ) -> Result<Json> {
        let admitted = match self.preflight(body, &mut cancelled) {
            Ok(admitted) => admitted,
            Err(error) => {
                self.log.push(Event::Unsent {
                    request: self.request,
                    reason: error.to_string(),
                    cancelled: error.is_cancelled(),
                });
                return Err(error);
            }
        };
        let mut outcome = self.stamp_and_submit(body, &admitted, &mut cancelled, &mut on_token)?;
        if self.variant == Variant::Retry && matches!(outcome, Err(CommandError::Cancelled(_))) {
            // Seeded bug: the cancellation is dropped and the command resent.
            self.needs_reset = false;
            outcome = self.stamp_and_submit(body, &admitted, &mut cancelled, &mut on_token)?;
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

    /// `request` up to the submission: admission and the cancellation probe.
    fn preflight(
        &mut self,
        body: &Json,
        cancelled: &mut Option<CancelProbe<'_>>,
    ) -> Result<Admitted> {
        require(
            !self.failed && !self.closed,
            "owned worker context is unavailable; no automatic restart",
        )?;
        require(
            body.kind() == JsonKind::Object,
            "owned worker command must be an object",
        )?;
        let admitted = match self.admit(body) {
            Ok(admitted) => admitted,
            Err(error) if error.is_refused() => return Err(error),
            Err(other) => return Err(WorkerError::EncodeBeforeSubmit(Box::new(other))),
        };
        let before_cancelled = match cancelled.as_mut() {
            Some(probe) => probe().map_err(WorkerError::CancelProbe)?,
            None => false,
        };
        if before_cancelled {
            self.needs_reset = true;
            return Err(WorkerError::Cancelled(
                "owned request cancelled before submission",
            ));
        }
        // The step deadline and its expiry check: time is not modelled.
        Ok(admitted)
    }

    /// The rest of `request`: broker stamping, V8 setup and `submit`.
    fn stamp_and_submit(
        &mut self,
        body: &Json,
        admitted: &Admitted,
        cancelled: &mut Option<CancelProbe<'_>>,
        on_token: &mut Option<TokenSink<'_>>,
    ) -> Result<std::result::Result<Json, CommandError>> {
        let mut command = self.broker_command;
        let mut lease = self.broker_lease;
        let before = self.broker_position;
        #[cfg(feature = "verify8")]
        let verify_inputs = if self.options.verify8 && !admitted.control && self.checks() {
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
        #[cfg(feature = "verify8")]
        let snapshots = verify_inputs
            .as_ref()
            .map(|_| (self.ready_value.clone(), self.verify_previous.clone()));
        #[cfg(feature = "verify8")]
        let mut verify = match (&verify_inputs, &snapshots) {
            (Some((policy, seed)), Some((ready, previous))) => {
                Some(Verify8Reply::new(Verify8Context {
                    ready,
                    previous,
                    job: body,
                    before: self.verify_position,
                    policy,
                    seed: *seed,
                    reset_resync: false,
                    branch8: self.branch8_mode(),
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
        let submission = Submission {
            body,
            wire,
            admitted,
            command,
            lease,
            before,
            #[cfg(feature = "verify8")]
            verify: verify.as_mut(),
            #[cfg(feature = "verify8")]
            verify_inputs: verify_inputs.as_ref(),
            #[cfg(not(feature = "verify8"))]
            _snapshots: std::marker::PhantomData,
        };
        let outcome = self.submit(submission, cancelled, on_token);
        if let Some(command) = self.written.take() {
            self.log.push(Event::Resolved {
                command,
                outcome: match &outcome {
                    Ok(value) => Outcome::Reply(Identity::of(value)),
                    Err(CommandError::Cancelled(_)) => Outcome::Cancelled,
                    Err(CommandError::Cause(_)) => Outcome::Refused,
                },
            });
        }
        Ok(outcome)
    }

    /// The command-shape admission block.
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
                tokens.kind() == JsonKind::Array
                    && generate.kind() == JsonKind::Number
                    && clear.kind() == JsonKind::Boolean,
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
            if self.v8_mode() {
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
        }
        require(
            !self.needs_reset || admitted.reset || (admitted.begin && !admitted.allow_restore),
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

    /// The submitted-command block. Any error except a cancellation becomes the caller's invalidating refusal.
    fn submit(
        &mut self,
        submission: Submission<'_, '_>,
        cancelled: &mut Option<CancelProbe<'_>>,
        on_token: &mut Option<TokenSink<'_>>,
    ) -> std::result::Result<Json, CommandError> {
        let Submission {
            body,
            wire,
            admitted,
            command,
            lease,
            before,
            #[cfg(feature = "verify8")]
            mut verify,
            #[cfg(feature = "verify8")]
            verify_inputs,
            ..
        } = submission;
        let mut cancellation = false;
        let mut consumer_failure = false;
        let skip_drain = self.variant == Variant::SkipDrain;
        if !skip_drain {
            self.idle_check()?;
        }
        let Some(commands) = self.commands.as_ref() else {
            return Err(CommandError::Cause(None));
        };
        if commands.send(wire).is_err() {
            return Err(CommandError::Cause(None)); // The pipe failed: submission is ambiguous.
        }
        if self.broker_epoch != 0 {
            self.broker_command = command;
            self.broker_lease = lease;
        }
        self.written = Some(command);
        self.log.push(Event::Sent {
            request: self.request,
            id: Identity {
                epoch: self.broker_epoch,
                command,
                lease,
            },
            kind: Kind::of(body),
        });
        let mut token_events: u64 = 0;
        let mut observed: Option<u64> = None;
        loop {
            if !cancellation && !consumer_failure {
                match cancelled.as_mut().map(|probe| probe()) {
                    Some(Ok(value)) => cancellation = value,
                    Some(Err(_)) => consumer_failure = true,
                    None => {}
                }
                // `timed_out`: time is not modelled.
                if cancellation || consumer_failure {
                    self.needs_reset = true;
                }
                if cancellation && skip_drain {
                    return Err(CommandError::Cancelled(
                        "owned request cancelled after its current step drained; reset required",
                    ));
                }
            }
            let draining = cancellation || consumer_failure;
            if self.pending.is_empty() {
                require(!self.eof, "owned target ended during a command")?;
                // The drain deadline: time is not modelled.
                self.pump();
                continue;
            }
            let Some(value) = self.pending.pop_front() else {
                return Err(CommandError::Cause(None));
            };
            let terminal = if self.checks() {
                event_is(
                    &value,
                    if admitted.control {
                        "control_done"
                    } else {
                        "done"
                    },
                )
            } else {
                // Transparent: the first done of either kind ends the command, and the session judges it.
                event_is(&value, "done")
                    .and_then(|done| Ok(done || event_is(&value, "control_done")?))
            }
            .unwrap_or(false);
            self.log.push(Event::Consumed {
                id: Identity::of(&value),
                done: terminal,
            });
            require(
                value.kind() == JsonKind::Object,
                "owned target emitted a nonobject event",
            )?;
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
            if !terminal && !self.checks() {
                // Transparent: every non-terminal event goes to the session.
                if !draining
                    && let Some(sink) = on_token.as_mut()
                    && sink(&value).is_err()
                {
                    consumer_failure = true;
                    self.needs_reset = true;
                }
            } else if event_is(&value, "token")? {
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
                if !draining
                    && let Some(sink) = on_token.as_mut()
                    && sink(&value).is_err()
                {
                    consumer_failure = true;
                    self.needs_reset = true;
                }
            } else if terminal {
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
                    let position = worker_integer(&value, "committed_prefix")?;
                    if self.checks() {
                        self.broker_identity(&value, command, lease)?;
                        self.broker_storage(&value)?;
                        require(
                            position <= 65536
                                && worker_integer(&value, "checkpoint_retained")?
                                    <= u64::from(self.options.checkpoint_slots),
                            "owned broker prefix/retention bound",
                        )?;
                        let generated = worker_field(&value, "generated_ids")?.array()?;
                        if admitted.control {
                            self.control_reply(
                                body,
                                &value,
                                admitted,
                                before,
                                position,
                                token_events,
                            )?;
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
                    }
                    self.broker_position = position as u32;
                }
                if !skip_drain {
                    self.idle_check()?;
                }
                if consumer_failure {
                    return Err(CommandError::Cause(Some(Box::new(CallbackFailed))));
                }
                if cancellation {
                    return Err(CommandError::Cancelled(
                        "owned request cancelled after its current step drained; reset required",
                    ));
                }
                // `timed_out`: time is not modelled.
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
            tokspur_wire::verify8::validate_branch8_control(
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
