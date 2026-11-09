//! A Shuttle model of the serving layer's broker leases, epochs, cancellation and resync, with negative controls.
//!
//! Three Shuttle threads per execution: the clients and accept loop, the generation thread, and the native target. The generation thread runs the real `ServeSession` inside a mirror of `serve.rs`'s `run_job`; its target closure calls a mirror of `OwnedWorker::request`, which talks to the fake native target over Shuttle channels. Clients disconnect at arbitrary points through a Shuttle atomic that both `ServeSink::cancelled` and the owned worker's cancellation probe read. Every component appends to one log, and `log::check` checks the five properties on it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

#[path = "shuttle_model/log.rs"]
mod log;
#[path = "shuttle_model/native.rs"]
mod native;
#[path = "shuttle_model/serve.rs"]
mod serve;
#[path = "shuttle_model/wire.rs"]
mod wire;

// The branch8 fixtures, as `http_branch8.rs` includes them.
#[cfg(feature = "branch8")]
#[allow(dead_code)]
#[path = "http_private/branch8.rs"]
mod branch8;
#[cfg(feature = "branch8")]
#[allow(dead_code)]
#[path = "http_private/verify8.rs"]
mod verify8;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use shuttle::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use shuttle::sync::mpsc::channel;
use shuttle::thread;
use tokspur_json::{Json, parse_json};
use tokspur_server::session::{ServeCodec, ServeSession, SessionOptions};
use tokspur_text::chat::RenderedChat;
use tokspur_text::response_protocol::ResponseAdmission;

use log::{Coverage, Event, Identity, Kind, Log, Outcome, Property, check, render};
use native::{BrokerTarget, Fault, IM_END, Target, put};
use serve::{Shared, clients, generation_loop};
use wire::{Options, Variant, Wire};

/// Executions per Shuttle run.
const BROKER_RANDOM: usize = 5000;
const BROKER_PCT: usize = 5000;
#[cfg(feature = "branch8")]
const BRANCH8_RANDOM: usize = 1000;
#[cfg(feature = "branch8")]
const BRANCH8_PCT: usize = 1000;
/// The budget within which each seeded bug must be found.
const SEEDED: usize = 400;
const MISBEHAVING: usize = 20;
/// PCT bug depth: up to two priority change points per execution.
const DEPTH: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// The plain broker (`checkpoint_slots` without V8), at every level.
    Broker,
    /// Private V8 with branch8 checkpoint reuse.
    #[cfg(feature = "branch8")]
    Branch8,
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    mode: Mode,
    variant: Variant,
    fault: Fault,
    /// The chance that a client abandons a request.
    cancel_rate: f64,
    /// How many idle-tick timeouts the generation loop may take.
    ticks: usize,
}

impl Scenario {
    fn broker() -> Self {
        Self {
            mode: Mode::Broker,
            variant: Variant::Faithful,
            fault: Fault::None,
            cancel_rate: 0.5,
            ticks: 2,
        }
    }

    #[cfg(feature = "branch8")]
    fn branch8() -> Self {
        Self {
            mode: Mode::Branch8,
            ..Self::broker()
        }
    }

    /// Every client waits for its response; the target misbehaves once.
    fn misbehaving(mode: Mode, variant: Variant, fault: Fault) -> Self {
        Self {
            mode,
            variant,
            fault,
            cancel_rate: 0.0,
            ..Self::broker()
        }
    }
}

/// The plain codec of `serve_protocol.rs`: bytes are IDs 1..=256.
fn encode(text: &[u8]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with(b"<|im_end|>") {
            result.push(IM_END);
            i += 10;
        } else {
            result.push(u32::from(text[i]) + 1);
            i += 1;
        }
    }
    result
}

fn decode(ids: &[u32]) -> Vec<u8> {
    let mut result = Vec::new();
    for &id in ids {
        if id == IM_END {
            result.extend_from_slice(b"<|im_end|>");
        } else {
            assert!(id > 0 && id <= 256, "fixture decode");
            result.push((id - 1) as u8);
        }
    }
    result
}

fn broker_codec() -> ServeCodec {
    ServeCodec {
        render: Box::new(|request: &Json| {
            let messages = request.find("messages").unwrap().unwrap().array().unwrap();
            let last = messages.last().unwrap();
            Ok(RenderedChat {
                prompt: String::from_utf8(
                    last.find("content")
                        .unwrap()
                        .unwrap()
                        .string()
                        .unwrap()
                        .to_vec(),
                )
                .unwrap()
                .replace("[fixture_end]", "<|im_end|>")
                .into_bytes(),
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        }),
        encode: Box::new(|text| Ok(encode(&text.prompt))),
        decode: Box::new(|ids: &[u32]| Ok(decode(ids))),
        admission: ResponseAdmission {
            vocabulary: 4096,
            im_end: IM_END,
            end_of_text: 1001,
            im_start: 1002,
        },
    }
}

fn chat(text: &str, stream: bool) -> Json {
    let text = text.replace("<|im_end|>", "[fixture_end]");
    Json::new_object([
        ("model", Json::from("tokspur")),
        (
            "messages",
            Json::Array(vec![Json::new_object([
                ("role", Json::from("user")),
                ("content", Json::from(text)),
            ])]),
        ),
        ("max_tokens", Json::integer(4)),
        ("stream", Json::Bool(stream)),
        ("enable_thinking", Json::Bool(false)),
    ])
}

/// Cold, append, branch restore, stochastic reset, then a greedy begin whose restore misses.
fn broker_script() -> Vec<Json> {
    let mut sampled = chat("S", false);
    put(&mut sampled, "temperature", parse_json(b"0.5").unwrap());
    put(&mut sampled, "seed", Json::integer(7));
    vec![
        chat("AB", true),
        chat("ABok<|im_end|>C", false),
        chat("ABD", true),
        sampled,
        chat("AB", false),
    ]
}

/// Cold, append, sibling restore, then an exact restore.
#[cfg(feature = "branch8")]
fn branch8_script() -> Vec<Json> {
    vec![
        verify8::chat(b"abc", true, 8),
        verify8::chat(b"abcok<|im_end|>d", false, 8),
        verify8::chat(b"abcz", true, 8),
        verify8::chat(b"abc", false, 8),
    ]
}

fn script(mode: Mode) -> Vec<Json> {
    match mode {
        Mode::Broker => broker_script(),
        #[cfg(feature = "branch8")]
        Mode::Branch8 => branch8_script(),
    }
}

fn session(mode: Mode) -> ServeSession {
    match mode {
        Mode::Broker => ServeSession::new(
            broker_codec(),
            SessionOptions {
                broker_ready: native::broker_ready(),
                ..SessionOptions::default()
            },
        )
        .unwrap(),
        #[cfg(feature = "branch8")]
        Mode::Branch8 => {
            let ready = branch8::prepared();
            ServeSession::new(
                verify8::session_codec(),
                SessionOptions {
                    broker_ready: ready.clone(),
                    verify8: true,
                    verify_ready: ready,
                    branch8: true,
                    ..SessionOptions::default()
                },
            )
            .unwrap()
        }
    }
}

fn target(mode: Mode, fault: Fault) -> Box<dyn Target + Send> {
    match mode {
        Mode::Broker => Box::new(BrokerTarget::new(fault)),
        #[cfg(feature = "branch8")]
        Mode::Branch8 => Box::new(native::Branch8Target::new(fault)),
    }
}

fn ready_and_options(mode: Mode) -> (Json, Options) {
    match mode {
        Mode::Broker => (
            native::broker_ready(),
            Options {
                checkpoint_slots: 2,
                checkpoint_budget_bytes: 1 << 20,
                #[cfg(feature = "verify8")]
                verify8: false,
                #[cfg(feature = "branch8")]
                branch8: false,
            },
        ),
        #[cfg(feature = "branch8")]
        Mode::Branch8 => (
            branch8::prepared(),
            Options {
                checkpoint_slots: 4,
                checkpoint_budget_bytes: 512 << 20,
                verify8: true,
                branch8: true,
            },
        ),
    }
}

/// One execution: spawn the three threads, shut down like `server_cli`, and return the log.
fn execute(scenario: Scenario) -> Vec<Event> {
    let log = Log::default();
    let (commands, command_lines) = channel();
    let (event_lines, events) = channel();
    let written = Arc::new(AtomicUsize::new(0));
    let native = {
        let target = target(scenario.mode, scenario.fault);
        thread::spawn(move || native::serve(target, command_lines, event_lines, written))
    };
    let shared = Arc::new(Shared::new());
    let stopped = Arc::new(AtomicBool::new(false));
    let generation = {
        let (shared, log, stopped) = (Arc::clone(&shared), log.clone(), Arc::clone(&stopped));
        thread::spawn(move || {
            let (ready, options) = ready_and_options(scenario.mode);
            let mut wire = Wire::new(
                ready,
                options,
                commands,
                events,
                scenario.variant,
                log.clone(),
            )
            .unwrap();
            let mut session = session(scenario.mode);
            generation_loop(&mut wire, &mut session, &shared, &log, scenario.ticks);
            stopped.store(true, Ordering::SeqCst);
            wire
        })
    };
    let client = {
        let (shared, log) = (Arc::clone(&shared), log.clone());
        let script = script(scenario.mode);
        thread::spawn(move || clients(script, shared, log, scenario.cancel_rate))
    };
    client.join().unwrap();
    // The clients' shutdown notify can land between the generation thread's check and its wait; the real wait then times out within 100 ms.
    while !stopped.load(Ordering::SeqCst) {
        shared.condition.notify_one();
        thread::yield_now();
    }
    let mut wire = generation.join().unwrap();
    wire.close(); // `worker.close()`, after the generation thread joined.
    native.join().unwrap();
    log.take()
}

/// The Shuttle test body: one execution, then the property check. Passing executions add their coverage to `totals`.
fn model(scenario: Scenario, totals: Arc<Mutex<Coverage>>) -> impl Fn() + Send + Sync {
    move || {
        let log = execute(scenario);
        match check(&log) {
            Ok(coverage) => totals.lock().unwrap().add(&coverage),
            Err(violation) => panic!("{violation}\n{}", render(&log)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Strategy {
    Random,
    Pct,
}

/// `shuttle::check_random` or `shuttle::check_pct` over the model.
fn run(strategy: Strategy, scenario: Scenario, iterations: usize, totals: &Arc<Mutex<Coverage>>) {
    let body = model(scenario, Arc::clone(totals));
    match strategy {
        Strategy::Random => shuttle::check_random(body, iterations),
        Strategy::Pct => shuttle::check_pct(body, iterations, DEPTH),
    }
}

/// Runs the model and returns what the executions covered.
fn explore(strategy: Strategy, scenario: Scenario, iterations: usize) -> Coverage {
    let totals = Arc::default();
    run(strategy, scenario, iterations, &totals);
    let coverage = totals.lock().unwrap().clone();
    eprintln!("{strategy:?} {scenario:?}: {coverage:#?}");
    coverage
}

/// Runs a Shuttle check that must fail; returns its panic message.
fn failure(run: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(run)).expect_err("the Shuttle check passed");
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map(|message| (*message).to_owned())
            .unwrap_or_default(),
    }
}

/// The encoded schedule in a Shuttle failure message.
fn schedule(message: &str) -> &str {
    message
        .split("failing schedule:\n\"\n")
        .nth(1)
        .and_then(|rest| rest.split("\n\"").next())
        .expect("Shuttle failure without a schedule")
}

/// Runs a seeded-bug scenario until Shuttle reports a failure; returns the failure message and the number of the failing execution.
fn seeded(strategy: Strategy, scenario: Scenario) -> (String, usize) {
    let totals = Arc::default();
    let message = failure(|| run(strategy, scenario, SEEDED, &totals));
    let found = totals.lock().unwrap().executions + 1;
    (message, found)
}

/// The failure is the expected property within the budget, and replaying its schedule deterministically fails the same way.
fn assert_found(scenario: Scenario, (message, found): (String, usize), property: Property) {
    let expected = format!("property '{}' violated", property.name());
    assert!(message.contains(&expected), "{message}");
    assert!(found <= SEEDED, "found after {found} executions");
    let encoded = schedule(&message).to_owned();
    let replayed = failure(|| shuttle::replay(model(scenario, Arc::default()), &encoded));
    assert!(replayed.contains(&expected), "{replayed}");
    let violation = message.lines().find(|line| line.contains(&expected));
    eprintln!("found in execution {found}: {}", violation.unwrap_or(""));
}

/// The cancellation, resync and reuse paths every run must reach: queued, session-observed and drained cancellations, cold resyncs and restores.
fn assert_exercised(coverage: &Coverage) {
    assert!(coverage.completed > 0, "{coverage:#?}");
    assert!(coverage.drained_cancellations > 0, "{coverage:#?}");
    assert!(coverage.cold_resyncs > 0, "{coverage:#?}");
    assert!(coverage.restores > 0, "{coverage:#?}");
    for text in [
        "queued request cancelled or expired",
        "request cancelled or deadline expired",
        "owned request cancelled after its current step drained; reset required",
    ] {
        assert!(coverage.failed_with(text) > 0, "{text}: {coverage:#?}");
    }
}

/// The narrow windows random scheduling also reaches (PCT reaches them only when a priority change point falls inside them): a cancellation between the session's check and the owned worker's probe, a disconnect during a response write, and a full queue.
fn assert_narrow_windows(coverage: &Coverage) {
    assert!(coverage.unsent_cancellations > 0, "{coverage:#?}");
    assert!(coverage.answered > 0, "{coverage:#?}");
    let text = "HTTP peer disconnected or sent unexpected bytes";
    assert!(coverage.failed_with(text) > 0, "{coverage:#?}");
}

#[test]
fn broker_properties_hold_under_random_schedules() {
    // Arrange
    let scenario = Scenario::broker();

    // Act
    let coverage = explore(Strategy::Random, scenario, BROKER_RANDOM);

    // Assert
    assert_eq!(coverage.executions, BROKER_RANDOM);
    assert_exercised(&coverage);
    assert_narrow_windows(&coverage);
}

#[test]
fn broker_properties_hold_under_pct_schedules() {
    // Arrange
    let scenario = Scenario::broker();

    // Act
    let coverage = explore(Strategy::Pct, scenario, BROKER_PCT);

    // Assert
    assert_eq!(coverage.executions, BROKER_PCT);
    assert_exercised(&coverage);
}

#[cfg(feature = "branch8")]
#[test]
fn branch8_properties_hold_under_random_schedules() {
    // Arrange
    let scenario = Scenario::branch8();

    // Act
    let coverage = explore(Strategy::Random, scenario, BRANCH8_RANDOM);

    // Assert
    assert_eq!(coverage.executions, BRANCH8_RANDOM);
    assert_exercised(&coverage);
    assert_narrow_windows(&coverage);
    let text = "private V8 cancelled after admitted drain; reset required";
    assert!(coverage.failed_with(text) > 0, "{coverage:#?}");
}

#[cfg(feature = "branch8")]
#[test]
fn branch8_properties_hold_under_pct_schedules() {
    // Arrange
    let scenario = Scenario::branch8();

    // Act
    let coverage = explore(Strategy::Pct, scenario, BRANCH8_PCT);

    // Assert
    assert_eq!(coverage.executions, BRANCH8_PCT);
    assert_exercised(&coverage);
}

#[test]
fn random_schedules_find_a_seeded_skipped_drain() {
    // Arrange
    let scenario = Scenario {
        variant: Variant::SkipDrain,
        ..Scenario::broker()
    };

    // Act
    let failure = seeded(Strategy::Random, scenario);

    // Assert
    assert_found(scenario, failure, Property::DrainBeforeCold);
}

#[test]
fn pct_schedules_find_a_seeded_skipped_drain() {
    // Arrange
    let scenario = Scenario {
        variant: Variant::SkipDrain,
        ..Scenario::broker()
    };

    // Act
    let failure = seeded(Strategy::Pct, scenario);

    // Assert
    assert_found(scenario, failure, Property::DrainBeforeCold);
}

#[test]
fn random_schedules_find_a_seeded_retry() {
    // Arrange
    let scenario = Scenario {
        variant: Variant::Retry,
        ..Scenario::broker()
    };

    // Act
    let failure = seeded(Strategy::Random, scenario);

    // Assert
    assert_found(scenario, failure, Property::NoImplicitRetry);
}

#[test]
fn pct_schedules_find_a_seeded_retry() {
    // Arrange
    let scenario = Scenario {
        variant: Variant::Retry,
        ..Scenario::broker()
    };

    // Act
    let failure = seeded(Strategy::Pct, scenario);

    // Assert
    assert_found(scenario, failure, Property::NoImplicitRetry);
}

#[test]
fn seeded_bugs_need_a_concurrent_cancellation() {
    // Arrange: the same stand-ins, but no client ever disconnects.
    let quiet = |variant| Scenario {
        variant,
        cancel_rate: 0.0,
        ..Scenario::broker()
    };

    // Act
    let skipped = explore(Strategy::Random, quiet(Variant::SkipDrain), MISBEHAVING);
    let retried = explore(Strategy::Random, quiet(Variant::Retry), MISBEHAVING);

    // Assert
    for coverage in [skipped, retried] {
        assert_eq!(coverage.executions, MISBEHAVING);
        assert_eq!(coverage.completed, coverage.jobs);
    }
}

/// A misbehaving target behind the transparent worker: the real session refuses the first request with `refusal`, never accepts the bad reply, and the next request resyncs cold and every later one completes.
fn assert_session_refuses(mode: Mode, fault: Fault, refusal: &str) {
    // Arrange
    let scenario = Scenario::misbehaving(mode, Variant::Transparent, fault);
    let requests = script(mode).len();

    // Act
    let coverage = explore(Strategy::Random, scenario, MISBEHAVING);

    // Assert
    assert_eq!(coverage.failed_with(refusal), MISBEHAVING, "{coverage:#?}");
    assert_eq!(coverage.jobs, requests * MISBEHAVING, "{coverage:#?}");
    assert_eq!(
        coverage.completed,
        (requests - 1) * MISBEHAVING,
        "{coverage:#?}"
    );
    assert_eq!(coverage.cold_resyncs, MISBEHAVING, "{coverage:#?}");
}

#[test]
fn session_refuses_a_stale_epoch_reply() {
    // Command 2 is the first prefill step.
    assert_session_refuses(
        Mode::Broker,
        Fault::StaleEpoch { at: 2 },
        "serving worker epoch changed",
    );
}

#[test]
fn session_refuses_a_stale_lease_reply() {
    // Command 4 is the first input-boundary capture.
    assert_session_refuses(
        Mode::Broker,
        Fault::StaleLease { at: 4 },
        "stale serving broker command/lease",
    );
}

#[test]
fn session_refuses_a_duplicate_reply() {
    // Command 3 (the second prefill step) gets command 2's done again.
    assert_session_refuses(
        Mode::Broker,
        Fault::DuplicateReply { at: 3 },
        "worker completion prefix mismatch",
    );
}

#[cfg(feature = "branch8")]
#[test]
fn branch8_session_refuses_a_stale_epoch_capture() {
    // Command 3 is the scheduled capture at the end of the first prompt.
    assert_session_refuses(
        Mode::Branch8,
        Fault::StaleEpoch { at: 3 },
        "private branch8 control command ancestry",
    );
}

#[cfg(feature = "branch8")]
#[test]
fn branch8_session_refuses_a_stale_lease_capture() {
    assert_session_refuses(
        Mode::Branch8,
        Fault::StaleLease { at: 3 },
        "private branch8 control lease ancestry",
    );
}

#[cfg(feature = "branch8")]
#[test]
fn branch8_session_refuses_a_duplicate_reply() {
    // The capture gets the input batch's done again.
    assert_session_refuses(
        Mode::Branch8,
        Fault::DuplicateReply { at: 3 },
        "invalid zero-step broker completion",
    );
}

#[test]
fn owned_worker_refuses_a_stale_reply_and_never_restarts() {
    // Arrange
    let scenario =
        Scenario::misbehaving(Mode::Broker, Variant::Faithful, Fault::StaleEpoch { at: 2 });

    // Act
    let coverage = explore(Strategy::Random, scenario, MISBEHAVING);

    // Assert
    let text = "owned command failed or became ambiguous; worker invalidated without retry";
    assert_eq!(coverage.failed_with(text), MISBEHAVING, "{coverage:#?}");
    assert_eq!(coverage.refusals, MISBEHAVING, "{coverage:#?}");
    assert_eq!(coverage.jobs, MISBEHAVING, "{coverage:#?}");
    assert_eq!(coverage.answered, 4 * MISBEHAVING, "{coverage:#?}");
}

const EPOCH: u64 = native::EPOCH;

fn id(command: u64, lease: u64) -> Identity {
    Identity {
        epoch: EPOCH,
        command,
        lease,
    }
}

fn sent(request: usize, command: u64, lease: u64, kind: Kind) -> Event {
    Event::Sent {
        request,
        id: id(command, lease),
        kind,
    }
}

fn read(command: u64, lease: u64, done: bool) -> Event {
    Event::Consumed {
        id: id(command, lease),
        done,
    }
}

fn resolved(command: u64, outcome: Outcome) -> Event {
    Event::Resolved { command, outcome }
}

fn reply(command: u64, lease: u64) -> Outcome {
    Outcome::Reply(id(command, lease))
}

fn ended(request: usize, error: Option<&str>) -> Event {
    Event::JobEnd {
        request,
        error: error.map(str::to_owned),
    }
}

const COLD: Kind = Kind::Begin { restore: false };
const STEP: Kind = Kind::Step { reset: false };

/// A cold begin, one step that is cancelled and drained, then a cold resync.
fn drained_cancellation() -> Vec<Event> {
    vec![
        Event::JobStart { request: 0 },
        sent(0, 1, 1, COLD),
        read(1, 1, true),
        resolved(1, reply(1, 1)),
        sent(0, 2, 1, STEP),
        read(2, 1, false),
        read(2, 1, true),
        resolved(2, Outcome::Cancelled),
        ended(
            0,
            Some("owned request cancelled after its current step drained"),
        ),
        Event::JobStart { request: 1 },
        sent(1, 3, 2, COLD),
        read(3, 2, true),
        resolved(3, reply(3, 2)),
        ended(1, None),
    ]
}

fn violated(log: &[Event]) -> Property {
    check(log)
        .expect_err("the checker accepted a bad log")
        .property
}

#[test]
fn checker_accepts_a_drained_cancellation_and_cold_resync() {
    // Arrange
    let log = drained_cancellation();

    // Act
    let coverage = check(&log);

    // Assert
    let coverage = coverage.unwrap();
    assert_eq!(
        (coverage.drained_cancellations, coverage.cold_resyncs),
        (1, 1)
    );
}

#[test]
fn checker_flags_two_in_flight_commands_on_one_lease() {
    // Arrange
    let log = vec![
        Event::JobStart { request: 0 },
        sent(0, 1, 1, COLD),
        read(1, 1, true),
        resolved(1, reply(1, 1)),
        sent(0, 2, 1, STEP),
        sent(0, 3, 1, STEP),
    ];

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::OneOwner);
}

#[test]
fn checker_flags_a_command_under_a_lease_the_session_no_longer_holds() {
    // Arrange: the cancelled job's lease is reused by the next job.
    let mut log = drained_cancellation();
    log.truncate(10);
    log.push(sent(1, 3, 1, Kind::Capture));

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::OneOwner);
}

#[test]
fn checker_flags_a_command_resolved_twice() {
    // Arrange
    let log = vec![
        Event::JobStart { request: 0 },
        sent(0, 1, 1, COLD),
        read(1, 1, true),
        resolved(1, reply(1, 1)),
        resolved(1, reply(1, 1)),
    ];

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::NoDoubleResolution);
}

#[test]
fn checker_flags_a_reply_accepted_for_another_command() {
    // Arrange: command 2 is resolved by command 1's done, and the session goes on as if it were command 2's.
    let log = vec![
        Event::JobStart { request: 0 },
        sent(0, 1, 1, COLD),
        read(1, 1, true),
        resolved(1, reply(1, 1)),
        sent(0, 2, 1, STEP),
        read(1, 1, true),
        resolved(2, reply(1, 1)),
        sent(0, 3, 1, STEP),
    ];

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::NoDoubleResolution);
}

#[test]
fn checker_flags_a_cold_reset_before_the_drain() {
    // Arrange: command 2 is reported cancelled without its done being read.
    let mut log = drained_cancellation();
    log.truncate(6);
    log.extend([
        resolved(2, Outcome::Cancelled),
        ended(0, Some("cancelled")),
        Event::JobStart { request: 1 },
        sent(1, 3, 2, Kind::Step { reset: true }),
    ]);

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::DrainBeforeCold);
}

#[test]
fn checker_flags_a_restoring_resync_after_a_cancellation() {
    // Arrange
    let mut log = drained_cancellation();
    log.truncate(10);
    log.push(sent(1, 3, 2, Kind::Begin { restore: true }));

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::DrainBeforeCold);
}

#[test]
fn checker_flags_a_command_resent_after_its_cancellation() {
    // Arrange: the same job sends again after the drained cancellation.
    let mut log = drained_cancellation();
    log.truncate(8);
    log.push(sent(0, 3, 1, STEP));

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::NoImplicitRetry);
}

#[test]
fn checker_flags_a_command_after_the_worker_was_invalidated() {
    // Arrange
    let log = vec![
        Event::JobStart { request: 0 },
        sent(0, 1, 1, COLD),
        read(1, 1, true),
        resolved(1, Outcome::Refused),
        Event::Killed,
        ended(0, Some("owned command failed")),
        Event::JobStart { request: 1 },
        sent(1, 2, 2, COLD),
    ];

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::NoImplicitRetry);
}

#[test]
fn checker_flags_an_epoch_that_goes_back() {
    // Arrange
    let mut log = drained_cancellation();
    log.truncate(10);
    log.push(Event::Sent {
        request: 1,
        id: Identity {
            epoch: EPOCH - 1,
            command: 3,
            lease: 2,
        },
        kind: COLD,
    });

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::MonotonicEpochs);
}

#[test]
fn checker_flags_a_command_id_that_goes_back() {
    // Arrange
    let mut log = drained_cancellation();
    log.truncate(10);
    log.push(sent(1, 2, 2, COLD));

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::MonotonicEpochs);
}

#[test]
fn checker_flags_a_stale_epoch_reply_the_session_accepted() {
    // Arrange
    let mut log = drained_cancellation();
    log.truncate(3);
    log.extend([
        resolved(
            1,
            Outcome::Reply(Identity {
                epoch: EPOCH - 1,
                command: 1,
                lease: 1,
            }),
        ),
        ended(0, None),
    ]);

    // Act
    let property = violated(&log);

    // Assert
    assert_eq!(property, Property::MonotonicEpochs);
}
