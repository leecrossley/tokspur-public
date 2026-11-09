//! The shared event log and the checker of the five properties.
//!
//! Every model component appends to one log in execution order. The log is a `std` mutex, not a Shuttle one: Shuttle runs every model thread on one OS thread, the lock is never held across a Shuttle operation, and recording must not add scheduling points of its own.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use tokspur_json::Json;

/// What a broker command does to the lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `branch_begin`; `restore` is its `allow_restore`.
    Begin { restore: bool },
    /// `branch_capture`.
    Capture,
    /// A model step; `reset` starts a new lease from an empty context.
    Step { reset: bool },
}

impl Kind {
    /// Classifies a session job (before the owned worker adds wire fields).
    pub fn of(job: &Json) -> Self {
        let flag = |name: &str| {
            job.find(name)
                .ok()
                .flatten()
                .and_then(|value| value.boolean().ok())
                .unwrap_or(false)
        };
        match job.find("op").ok().flatten() {
            Some(op) if op.string().is_ok_and(|op| op == b"branch_begin") => Self::Begin {
                restore: flag("allow_restore"),
            },
            Some(_) => Self::Capture,
            None => Self::Step {
                reset: flag("reset"),
            },
        }
    }

    /// Opens a new lease: the owned worker increments `lease_id`.
    pub fn fresh(self) -> bool {
        matches!(self, Self::Begin { .. } | Self::Step { reset: true })
    }

    /// Starts from an empty context: a reset, or a begin that may not restore.
    pub fn cold(self) -> bool {
        matches!(
            self,
            Self::Begin { restore: false } | Self::Step { reset: true }
        )
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Begin { restore: true } => "begin(restore)",
            Self::Begin { restore: false } => "begin(cold)",
            Self::Capture => "capture",
            Self::Step { reset: true } => "step(reset)",
            Self::Step { reset: false } => "step",
        })
    }
}

/// The broker identity a command or reply carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub epoch: u64,
    pub command: u64,
    pub lease: u64,
}

impl Identity {
    /// The identity fields of a worker event; an absent field reads as 0.
    pub fn of(value: &Json) -> Self {
        let number = |name: &str| {
            value
                .find(name)
                .ok()
                .flatten()
                .and_then(|value| value.unsigned_integer().ok())
                .unwrap_or(0)
        };
        Self {
            epoch: number("broker_epoch"),
            command: number("command_id"),
            lease: number("lease_id"),
        }
    }
}

impl fmt::Display for Identity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "#{} lease {} epoch {}",
            self.command, self.lease, self.epoch
        )
    }
}

/// How the owned worker resolved a submitted command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The done went back to the session; this is the identity it carried.
    Reply(Identity),
    /// Refused after submission; the owned worker invalidated itself.
    Refused,
    /// Cancelled. The owned worker reports this only after the done drained.
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// The accept loop queued request `request` for the generation thread.
    Queued { request: usize },
    /// The accept loop answered `request` itself (429 queue full, 503 worker unavailable); it never reaches the generation thread.
    Answered { request: usize, status: u16 },
    /// The client went away (or the request deadline passed). Logged as the client decides, just before its store becomes visible.
    Disconnected { request: usize },
    /// The generation thread began `run_job`.
    JobStart { request: usize },
    /// `run_job` finished; `error` is the failure text, if any.
    JobEnd {
        request: usize,
        error: Option<String>,
    },
    /// The generation loop's idle tick invalidated the session.
    Invalidated,
    /// The owned worker wrote a command to the target.
    Sent {
        request: usize,
        id: Identity,
        kind: Kind,
    },
    /// The owned worker refused or cancelled a command before writing it.
    Unsent {
        request: usize,
        reason: String,
        cancelled: bool,
    },
    /// The owned worker read one target event (`done`: the command's last).
    Consumed { id: Identity, done: bool },
    /// The owned worker's `request` returned for a written command.
    Resolved { command: u64, outcome: Outcome },
    /// The owned worker invalidated itself and killed the target.
    Killed,
    /// The server closed the owned worker at shutdown.
    Closed,
}

impl fmt::Display for Event {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued { request } => write!(formatter, "queued r{request}"),
            Self::Answered { request, status } => write!(formatter, "answered r{request} {status}"),
            Self::Disconnected { request } => write!(formatter, "disconnected r{request}"),
            Self::JobStart { request } => write!(formatter, "job r{request} start"),
            Self::JobEnd {
                request,
                error: None,
            } => write!(formatter, "job r{request} ok"),
            Self::JobEnd {
                request,
                error: Some(error),
            } => write!(formatter, "job r{request} failed: {error}"),
            Self::Invalidated => formatter.write_str("idle tick invalidated the session"),
            Self::Sent { request, id, kind } => write!(formatter, "sent r{request} {id} {kind}"),
            Self::Unsent {
                request, reason, ..
            } => write!(formatter, "unsent r{request}: {reason}"),
            Self::Consumed { id, done } => write!(
                formatter,
                "read {id} {}",
                if *done { "done" } else { "event" }
            ),
            Self::Resolved { command, outcome } => match outcome {
                Outcome::Reply(id) => write!(formatter, "resolved #{command} by reply {id}"),
                Outcome::Refused => write!(formatter, "resolved #{command} refused"),
                Outcome::Cancelled => write!(formatter, "resolved #{command} cancelled"),
            },
            Self::Killed => formatter.write_str("worker invalidated and killed"),
            Self::Closed => formatter.write_str("worker closed"),
        }
    }
}

/// One execution's log, shared by every model thread.
#[derive(Clone, Default)]
pub struct Log(Arc<Mutex<Vec<Event>>>);

impl Log {
    pub fn push(&self, event: Event) {
        self.0.lock().unwrap().push(event);
    }

    pub fn take(&self) -> Vec<Event> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

/// The log, one numbered event per line.
pub fn render(log: &[Event]) -> String {
    log.iter()
        .enumerate()
        .map(|(at, event)| format!("{at:4} {event}\n"))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Property {
    /// At most one in-flight command per lease, and nothing is sent under a lease the session no longer holds.
    OneOwner,
    /// Every command resolves exactly once: reply, refusal or drained cancellation.
    NoDoubleResolution,
    /// After a cancellation the in-flight command drains before any cold reset or resync is sent; after any cancellation, failure or invalidation the resync is cold.
    DrainBeforeCold,
    /// A failed or cancelled command is never resent automatically.
    NoImplicitRetry,
    /// Broker epochs, and command and lease ids within an epoch, only increase.
    MonotonicEpochs,
}

impl Property {
    pub fn name(self) -> &'static str {
        match self {
            Self::OneOwner => "one owner per lease",
            Self::NoDoubleResolution => "no double resolution",
            Self::DrainBeforeCold => "drain before cold",
            Self::NoImplicitRetry => "no implicit retry",
            Self::MonotonicEpochs => "monotonic epochs",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub property: Property,
    /// The index of the offending event (the log length for end checks).
    pub at: usize,
    pub detail: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "property '{}' violated at event {}: {}",
            self.property.name(),
            self.at,
            self.detail
        )
    }
}

/// What one or more executions exercised, so a pass is not vacuous.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    pub executions: usize,
    pub jobs: usize,
    pub completed: usize,
    pub commands: usize,
    /// Cancellations the owned worker resolved after reading the done.
    pub drained_cancellations: usize,
    /// Cancellations the owned worker saw before writing the command.
    pub unsent_cancellations: usize,
    /// Cold commands sent after a cancellation or failure.
    pub cold_resyncs: usize,
    /// `branch_begin` commands that allowed a restore.
    pub restores: usize,
    pub refusals: usize,
    /// Requests the accept loop answered itself (429/503).
    pub answered: usize,
    pub idle_invalidations: usize,
    /// Job failure texts and how often each occurred.
    pub failures: BTreeMap<String, usize>,
}

impl Coverage {
    pub fn add(&mut self, other: &Coverage) {
        self.executions += other.executions;
        self.jobs += other.jobs;
        self.completed += other.completed;
        self.commands += other.commands;
        self.drained_cancellations += other.drained_cancellations;
        self.unsent_cancellations += other.unsent_cancellations;
        self.cold_resyncs += other.cold_resyncs;
        self.restores += other.restores;
        self.refusals += other.refusals;
        self.answered += other.answered;
        self.idle_invalidations += other.idle_invalidations;
        for (text, count) in &other.failures {
            *self.failures.entry(text.clone()).or_default() += count;
        }
    }

    /// Jobs that failed with exactly this text.
    pub fn failed_with(&self, text: &str) -> usize {
        self.failures.get(text).copied().unwrap_or(0)
    }
}

/// Checks the five properties over one execution's log.
pub fn check(log: &[Event]) -> Result<Coverage, Violation> {
    let mut checker = Checker::default();
    for (at, event) in log.iter().enumerate() {
        checker.step(at, event)?;
    }
    checker.finish(log.len())
}

struct Command {
    request: usize,
    id: Identity,
    kind: Kind,
    resolutions: usize,
    drained: bool,
}

#[derive(Default)]
struct Checker {
    commands: Vec<Command>,
    /// Written, done not yet read, target not killed.
    in_flight: Option<usize>,
    /// A reply returned to the session whose acceptance is not yet proven.
    reply: Option<(usize, Identity)>,
    /// The lease the session holds: a fresh command's reply it accepted, with no failure, cancellation or invalidation since.
    held: Option<u64>,
    /// A cancellation, failure or invalidation happened; the next fresh command must be cold.
    must_cold: bool,
    /// The request whose command failed or was cancelled.
    failed: Option<usize>,
    worker_dead: bool,
    last: Option<Identity>,
    job: Option<usize>,
    started: BTreeSet<usize>,
    answered: BTreeSet<usize>,
    coverage: Coverage,
}

fn violation<T>(property: Property, at: usize, detail: String) -> Result<T, Violation> {
    Err(Violation {
        property,
        at,
        detail,
    })
}

impl Checker {
    /// The session went on (another command, or a successful job), so it accepted the reply it was given.
    fn accept(&mut self, at: usize) -> Result<(), Violation> {
        let Some((index, carried)) = self.reply.take() else {
            return Ok(());
        };
        let command = &self.commands[index];
        let id = command.id;
        if carried.epoch != id.epoch {
            return violation(
                Property::MonotonicEpochs,
                at,
                format!(
                    "the session accepted a reply from epoch {} for command #{} of epoch {}",
                    carried.epoch, id.command, id.epoch
                ),
            );
        }
        if carried.command != id.command {
            return violation(
                Property::NoDoubleResolution,
                at,
                format!(
                    "the session accepted the reply of command #{} as the resolution of command #{}",
                    carried.command, id.command
                ),
            );
        }
        if carried.lease != id.lease {
            return violation(
                Property::OneOwner,
                at,
                format!(
                    "the session accepted a reply under lease {} for command #{} under lease {}",
                    carried.lease, id.command, id.lease
                ),
            );
        }
        if command.kind.fresh() {
            self.held = Some(id.lease);
        }
        Ok(())
    }

    fn step(&mut self, at: usize, event: &Event) -> Result<(), Violation> {
        match event {
            Event::Queued { .. } | Event::Disconnected { .. } | Event::Closed => {}
            Event::Answered { request, .. } => {
                self.answered.insert(*request);
                self.coverage.answered += 1;
            }
            Event::JobStart { request } => {
                if !self.started.insert(*request) {
                    return violation(
                        Property::NoImplicitRetry,
                        at,
                        format!("request r{request} ran twice"),
                    );
                }
                if self.answered.contains(request) {
                    return violation(
                        Property::NoImplicitRetry,
                        at,
                        format!("request r{request} was refused by the accept loop and then ran"),
                    );
                }
                self.job = Some(*request);
                self.failed = None;
                self.coverage.jobs += 1;
            }
            Event::JobEnd { request, error } => {
                if self.job != Some(*request) {
                    return violation(
                        Property::NoImplicitRetry,
                        at,
                        format!("job r{request} ended without running"),
                    );
                }
                match error {
                    None => {
                        self.accept(at)?;
                        self.coverage.completed += 1;
                    }
                    Some(text) => {
                        // The session invalidated itself: it holds no lease and its next fresh command must be cold.
                        self.reply = None;
                        self.held = None;
                        self.must_cold = true;
                        *self.coverage.failures.entry(text.clone()).or_default() += 1;
                    }
                }
                self.job = None;
                self.failed = None;
            }
            Event::Invalidated => {
                self.held = None;
                self.must_cold = true;
                self.coverage.idle_invalidations += 1;
            }
            Event::Unsent {
                request, cancelled, ..
            } => {
                if *cancelled {
                    self.must_cold = true;
                    self.failed = Some(*request);
                    self.held = None;
                    self.coverage.unsent_cancellations += 1;
                }
            }
            Event::Sent { request, id, kind } => self.sent(at, *request, *id, *kind)?,
            Event::Consumed { id, done } => {
                let Some(index) = self.in_flight else {
                    return violation(
                        Property::NoDoubleResolution,
                        at,
                        format!("a reply carrying {id} was read with no command in flight"),
                    );
                };
                if *done {
                    self.commands[index].drained = true;
                    self.in_flight = None;
                }
            }
            Event::Resolved { command, outcome } => self.resolved(at, *command, outcome)?,
            Event::Killed => {
                // The guardian kills and drains the whole target group.
                self.worker_dead = true;
                if let Some(index) = self.in_flight.take() {
                    self.commands[index].drained = true;
                }
            }
        }
        Ok(())
    }

    fn sent(
        &mut self,
        at: usize,
        request: usize,
        id: Identity,
        kind: Kind,
    ) -> Result<(), Violation> {
        self.accept(at)?;
        let c = id.command;
        if self.worker_dead {
            return violation(
                Property::NoImplicitRetry,
                at,
                format!(
                    "command #{c} sent after the worker was invalidated (no automatic restart)"
                ),
            );
        }
        if self.job != Some(request) {
            return violation(
                Property::NoImplicitRetry,
                at,
                format!("command #{c} sent for request r{request} outside its job"),
            );
        }
        if self.failed == Some(request) {
            return violation(
                Property::NoImplicitRetry,
                at,
                format!(
                    "command #{c} resent for request r{request} after its command failed or was cancelled"
                ),
            );
        }
        if let Some(index) = self.in_flight {
            let other = &self.commands[index];
            return if kind.fresh() {
                violation(
                    Property::DrainBeforeCold,
                    at,
                    format!(
                        "{kind} command #{c} sent while command #{} is undrained",
                        other.id.command
                    ),
                )
            } else {
                violation(
                    Property::OneOwner,
                    at,
                    format!(
                        "command #{c} sent on lease {} while command #{} (lease {}) is in flight",
                        id.lease, other.id.command, other.id.lease
                    ),
                )
            };
        }
        if !kind.fresh() && self.held != Some(id.lease) {
            return violation(
                Property::OneOwner,
                at,
                format!(
                    "{kind} command #{c} sent under lease {}, which the session does not hold (held: {:?})",
                    id.lease, self.held
                ),
            );
        }
        if kind.fresh() && !kind.cold() && self.must_cold {
            return violation(
                Property::DrainBeforeCold,
                at,
                format!(
                    "command #{c} may restore after a cancellation or failure; the resync must be cold"
                ),
            );
        }
        if let Some(last) = self.last {
            if id.epoch < last.epoch {
                return violation(
                    Property::MonotonicEpochs,
                    at,
                    format!("epoch went back from {} to {}", last.epoch, id.epoch),
                );
            }
            if id.epoch == last.epoch {
                if id.command <= last.command {
                    return violation(
                        Property::MonotonicEpochs,
                        at,
                        format!("command id went from #{} to #{c}", last.command),
                    );
                }
                if id.lease < last.lease || (kind.fresh() && id.lease == last.lease) {
                    return violation(
                        Property::MonotonicEpochs,
                        at,
                        format!("lease id went from {} to {} ({kind})", last.lease, id.lease),
                    );
                }
            }
        }
        if kind.cold() {
            if self.must_cold {
                self.coverage.cold_resyncs += 1;
            }
            self.must_cold = false;
        }
        if kind.fresh() {
            self.held = None;
        }
        if kind == (Kind::Begin { restore: true }) {
            self.coverage.restores += 1;
        }
        self.coverage.commands += 1;
        self.in_flight = Some(self.commands.len());
        self.commands.push(Command {
            request,
            id,
            kind,
            resolutions: 0,
            drained: false,
        });
        self.last = Some(id);
        Ok(())
    }

    fn resolved(&mut self, at: usize, command: u64, outcome: &Outcome) -> Result<(), Violation> {
        let Some(index) = self
            .commands
            .iter()
            .rposition(|sent| sent.id.command == command)
        else {
            return violation(
                Property::NoDoubleResolution,
                at,
                format!("command #{command} resolved but never sent"),
            );
        };
        let sent = &mut self.commands[index];
        sent.resolutions += 1;
        if sent.resolutions > 1 {
            return violation(
                Property::NoDoubleResolution,
                at,
                format!("command #{command} resolved twice"),
            );
        }
        let (request, drained) = (sent.request, sent.drained);
        match outcome {
            Outcome::Reply(carried) => self.reply = Some((index, *carried)),
            Outcome::Cancelled | Outcome::Refused => {
                if *outcome == Outcome::Refused {
                    self.coverage.refusals += 1;
                } else if drained {
                    self.coverage.drained_cancellations += 1;
                }
                self.must_cold = true;
                self.failed = Some(request);
                self.held = None;
            }
        }
        Ok(())
    }

    fn finish(mut self, at: usize) -> Result<Coverage, Violation> {
        if let Some(unresolved) = self.commands.iter().find(|sent| sent.resolutions == 0) {
            return violation(
                Property::NoDoubleResolution,
                at,
                format!("command #{} was never resolved", unresolved.id.command),
            );
        }
        if let Some(index) = self.in_flight
            && !self.worker_dead
        {
            return violation(
                Property::DrainBeforeCold,
                at,
                format!(
                    "command #{} was never drained",
                    self.commands[index].id.command
                ),
            );
        }
        self.coverage.executions = 1;
        Ok(self.coverage)
    }
}
