//! Port of `native/tests/test_http_fast_math.cpp`, check for check.
//!
//! Fresh CPU-generated contracts only; no historical receipt is rewritten. Like the C++, this suite first runs every inherited branch8 (and V8) check. The worker launches this binary as `branch8-target`; the fast-math fake answers when `--qsa-channel-chronological` is present.

use std::path::{Path, PathBuf};

use tokspur_json::{Json, parse_json};
use tokspur_wire::verify8::{
    Verify8Error, validate_branch8_control, validate_branch8_ready, verify8_fast_math,
};
use tokspur_wire::worker::OwnedWorker;

use crate::branch8::{
    self, Fake as BranchFake, GRAPH, MAINTENANCE, WIRE, begin_job, branch_reply, snapshot,
    staged_prepared, stamp, worker_files,
};
use crate::verify8::{
    FIXTURE, Round, append_line, array, check, checks, emit, fake_executable, field,
    generation_job, input_job, int, integer, make_transcript, never, number, ok, parse, policy,
    policy_parts, present, put, read_line, refuses, remove, section, soon, text, with, without,
};

const NAMES: [&str; 8] = [
    "ple_gate_transform",
    "ple_gate_transform_schema",
    "qsa_channel_chronological_mode",
    "qsa_channel_chronological_schema",
    "qsa_channel_factor_logical_bytes",
    "qsa_channel_factor_charged_bytes",
    "qsa_channel_calls",
    "qsa_channel_extra_dispatches",
];

fn fields(value: &Json, channel: bool) -> Json {
    let mut value = with(value, "ple_gate_transform", Json::from("signed-root"));
    put(&mut value, "ple_gate_transform_schema", int(1));
    put(
        &mut value,
        "qsa_channel_chronological_mode",
        Json::from(if channel { "on" } else { "off" }),
    );
    put(&mut value, "qsa_channel_chronological_schema", int(1));
    put(&mut value, "qsa_channel_factor_logical_bytes", int(393_792));
    put(&mut value, "qsa_channel_factor_charged_bytes", int(409_600));
    let calls = if channel {
        integer(&value, "qsa_stage_v8_calls")
    } else {
        0
    };
    put(&mut value, "qsa_channel_calls", int(calls));
    put(&mut value, "qsa_channel_extra_dispatches", int(calls));
    value
}

fn annotate(value: &Json, channel: bool) -> Json {
    let event = text(value, "event").to_vec();
    let mut value = value.clone();
    if event == b"generation_round" {
        let rows = integer(&value, "verified_tokens") + integer(&value, "ordinary_tokens");
        let extra = if channel { 12 * rows } else { 0 };
        let dispatches = integer(&value, "dispatches") + extra;
        put(&mut value, "qsa_channel_round_calls", int(extra));
        put(&mut value, "dispatches", int(dispatches));
        return value;
    }
    if event == b"ready" || event == b"done" {
        value = fields(&value, channel);
        if event == b"done" {
            let calls = if channel {
                integer(&value, "qsa_stage_v8_request_calls")
            } else {
                0
            };
            put(&mut value, "qsa_channel_request_calls", int(calls));
        }
    }
    if let Some(state) = ok(value.find("branch8_session")) {
        let annotated = fields(state, channel);
        put(&mut value, "branch8_session", annotated);
    }
    value
}

fn core_ready() -> Json {
    let mut ready = staged_prepared(true);
    for name in ["tracked_metal_buffer_bytes", "peak_metal_buffer_bytes"] {
        let bytes = integer(&ready, name) + 409_600;
        put(&mut ready, name, int(bytes));
    }
    let allocations = integer(&ready, "gpu_allocation_count") + 1;
    put(&mut ready, "gpu_allocation_count", int(allocations));
    let mut state = field(&ready, "branch8_session").clone();
    for name in GRAPH {
        put(&mut state, name, field(&ready, name).clone());
    }
    put(&mut ready, "branch8_session", state);
    ready
}

/// The branch8 fake with every ready/done/state/round annotated.
struct Fake {
    inner: BranchFake,
    channel: bool,
    ready: Json,
    previous: Json,
}

impl Fake {
    fn new(enabled: bool) -> Self {
        let inner = BranchFake::new(core_ready());
        let ready = annotate(&inner.ready, enabled);
        Self {
            previous: ready.clone(),
            inner,
            channel: enabled,
            ready,
        }
    }

    /// `operator()(job, callback)`.
    fn apply(&mut self, job: &Json, mut callback: impl FnMut(&Json)) -> Json {
        let channel = self.channel;
        let mut events = Vec::new();
        let done = self
            .inner
            .apply(job, |event| events.push(annotate(event, channel)));
        self.previous = annotate(&done, channel);
        for event in &events {
            callback(event);
        }
        self.previous.clone()
    }
}

fn admission_and_rounds(channel: bool) {
    let mut fake = Fake::new(channel);
    let ready = fake.ready.clone();
    ok(validate_branch8_ready(&ready));
    check(ok(verify8_fast_math(&ready)) == Some(channel), FIXTURE);
    for name in NAMES {
        refuses(validate_branch8_ready(&without(&ready, name)));
        refuses(validate_branch8_ready(&with(
            &ready,
            "branch8_session",
            without(field(&ready, "branch8_session"), name),
        )));
    }
    for (name, value) in [
        ("ple_gate_transform", Json::from("plain")),
        ("ple_gate_transform_schema", int(2)),
        ("qsa_channel_chronological_schema", Json::Bool(true)),
        ("qsa_channel_chronological_mode", Json::from("invalid")),
        ("qsa_channel_factor_logical_bytes", int(0)),
        ("qsa_channel_factor_charged_bytes", int(393_792)),
        ("qsa_channel_calls", int(12)),
        ("qsa_channel_extra_dispatches", int(12)),
        ("qsa_channel_unknown", int(0)),
        ("licensed_q4_routed_mode", Json::from("on")),
        ("batch_phase_clock_mode", Json::from("on")),
    ] {
        refuses(validate_branch8_ready(&with(&ready, name, value)));
    }
    let ids = [100u32, 101, 102, 103, 104, 105, 106, 107];
    let mut previous = ready.clone();
    let cold_job = begin_job(&ids, false);
    let cold = fake.apply(&cold_job, |_| check(false, FIXTURE));
    ok(validate_branch8_control(
        &ready, &previous, &cold_job, &cold, false,
    ));
    previous = cold;
    let input = input_job(8, false, &policy(), 0);
    let filled = fake.apply(&input, |_| check(false, FIXTURE));
    {
        // The C++ passes a `policy()` temporary that dies with this constructor's full-expression; the reply keeps it alive here.
        let greedy = policy();
        let mut prefill = ok(branch_reply(&ready, &previous, &input, 0, &greedy));
        ok(prefill.finish(&filled));
    }
    previous = filled.clone();
    check(
        integer(&filled, "qsa_channel_request_calls") == if channel { 96 } else { 0 },
        FIXTURE,
    );
    let capture_job = parse(r#"{"op":"branch_capture"}"#);
    let captured = fake.apply(&capture_job, |_| check(false, FIXTURE));
    ok(validate_branch8_control(
        &ready,
        &previous,
        &capture_job,
        &captured,
        false,
    ));
    previous = captured;
    let generation = generation_job(8);
    let mut events = Vec::new();
    let done = fake.apply(&generation, |event| events.push(event.clone()));
    let verify = |stream: &[Json], result: &Json| -> Result<(), Verify8Error> {
        let greedy = policy();
        let mut reply = branch_reply(&ready, &previous, &generation, 8, &greedy)?;
        for event in stream {
            reply.event(event)?;
        }
        reply.finish(result)?;
        check(!reply.events()?.is_empty(), FIXTURE);
        Ok(())
    };
    ok(verify(&events, &done));
    let mut broken = events.clone();
    remove(&mut broken[0], "qsa_channel_round_calls");
    refuses(verify(&broken, &done));
    let mut broken = events.clone();
    put(&mut broken[0], "qsa_channel_round_calls", int(999));
    refuses(verify(&broken, &done));
    let mut broken = events.clone();
    let dispatches = integer(&broken[0], "dispatches") + 1;
    put(&mut broken[0], "dispatches", int(dispatches));
    refuses(verify(&broken, &done));
    refuses(verify(
        &events,
        &with(&done, "qsa_channel_request_calls", int(999)),
    ));
    let bad_state = fields(field(&done, "branch8_session"), !channel);
    refuses(verify(&events, &with(&done, "branch8_session", bad_state)));
    let warm_job = begin_job(&ids, true);
    let restored = fake.apply(&warm_job, |_| check(false, FIXTURE));
    ok(validate_branch8_control(
        &ready, &done, &warm_job, &restored, false,
    ));
    check(
        integer(field(&restored, "branch8_session"), "qsa_channel_calls")
            == integer(&done, "qsa_channel_calls"),
        "channel lifetime counter rewound on restore",
    );
    // Independently generated partial rejection: seven evaluated, three accepted, correction plus four fallback rows = 12 evaluated model rows, NOT the eight committed outputs. Preserve the existing recovery budget.
    let plain_previous = fake.inner.state.clone();
    let rounds = [
        Round::of(7, 3, 1),
        Round::of(0, 0, 1),
        Round::of(0, 0, 1),
        Round::of(0, 0, 1),
        Round::of(0, 0, 1),
    ];
    let mut t = make_transcript(
        &fake.inner.ready,
        &plain_previous,
        generation.clone(),
        &policy(),
        0,
        &rounds,
    );
    for name in MAINTENANCE {
        put(&mut t.done, name, field(&plain_previous, name).clone());
    }
    put(&mut t.done, "logits_available", Json::Bool(true));
    put(
        &mut t.done,
        "checkpoint_retained",
        int(fake.inner.saved.len() as u64),
    );
    put(&mut t.done, "checkpoint_capture_wall_ms", number(0.0));
    put(&mut t.done, "checkpoint_restore_wall_ms", number(0.0));
    let partial_state = snapshot(&t.done, &plain_previous);
    put(&mut t.done, "branch8_session", partial_state);
    let command = fake.inner.command + 1;
    let lease = fake.inner.lease;
    let partial_done = annotate(&stamp(t.done.clone(), command, lease, &Json::Null), channel);
    let mut before = restored.clone();
    let mut state = field(&before, "branch8_session").clone();
    let mut top = array(&state, "final_logit_top8").clone();
    top[0] = with(&top[0], "id", int(200));
    put(&mut state, "final_logit_top8", Json::Array(top));
    put(&mut before, "branch8_session", state);
    let partial_events: Vec<Json> = t
        .events
        .iter()
        .map(|event| annotate(&stamp(event.clone(), command, lease, &Json::Null), channel))
        .collect();
    let verify_partial = |result: &Json| -> Result<(), Verify8Error> {
        let greedy = policy();
        let mut reply = branch_reply(&ready, &before, &generation, 8, &greedy)?;
        for event in &partial_events {
            reply.event(event)?;
        }
        reply.finish(result)?;
        check(!reply.events()?.is_empty(), FIXTURE);
        Ok(())
    };
    ok(verify_partial(&partial_done));
    check(
        integer(&partial_done, "qsa_channel_request_calls") == if channel { 144 } else { 0 },
        FIXTURE,
    );
    if channel {
        refuses(verify_partial(&with(
            &partial_done,
            "qsa_channel_request_calls",
            int(96),
        )));
    }
}

fn target(args: &[String]) -> i32 {
    let mut channel: Option<bool> = None;
    let mut policy_path = PathBuf::new();
    let mut signed_root = false;
    for i in 2..args.len().saturating_sub(1) {
        let (key, value) = (args[i].as_str(), args[i + 1].as_str());
        if key == "--qsa-channel-chronological" {
            if channel.is_some() || (value != "off" && value != "on") {
                return 81;
            }
            channel = Some(value == "on");
        }
        if key == "--ple-gate-transform" {
            signed_root = value == "signed-root";
        }
        if key == "--policy" {
            policy_path = PathBuf::from(value);
        }
    }
    let Some(channel) = channel else {
        return 82;
    };
    if !signed_root || policy_path.as_os_str().is_empty() {
        return 82;
    }
    let (directory, mode) = policy_parts(&policy_path);
    ok(std::fs::write(
        directory.join("started"),
        "explicit fast-math fixture argv\n",
    ));
    let mut fake = Fake::new(channel);
    let mut ready = fake.ready.clone();
    if mode == "wrong-fast-arm" {
        ready = Fake::new(!channel).ready;
    }
    if mode == "missing-fast" {
        ready = core_ready();
    }
    emit(&ready);
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    while let Some(line) = read_line(&mut input) {
        append_line(&directory.join("commands"), "command\n");
        let mut job = ok(parse_json(&line));
        if integer(&job, "broker_epoch") != 4242
            || integer(&job, "command_id") != fake.inner.command + 1
            || integer(&job, "expected_prefix") != fake.inner.history.len() as u64
            || integer(&job, "broker_batch_width") != 8
            || text(&job, "broker_protocol") != b"branch8-v1"
        {
            return 83;
        }
        let begin = present(&job, "op") && text(&job, "op") == b"branch_begin";
        if integer(&job, "lease_id") != fake.inner.lease + u64::from(begin) {
            return 84;
        }
        for name in WIRE {
            remove(&mut job, name);
        }
        let mut done = fake.apply(&job, emit);
        if mode == "bad-fast-cost" && !present(&job, "op") && integer(&job, "generate") != 0 {
            put(&mut done, "qsa_channel_request_calls", int(999));
        }
        emit(&done);
    }
    0
}

fn workers(executable: &Path, channel: bool) {
    let ids = [1u32, 2, 3, 4, 5, 6, 7, 8];
    for mode in ["normal", "bad-fast-cost", "wrong-fast-arm", "missing-fast"] {
        let mut f = worker_files(executable, mode, true, true);
        f.options.fast_channel = Some(channel);
        if mode == "wrong-fast-arm" || mode == "missing-fast" {
            refuses(OwnedWorker::new(f.options()));
            check(f.commands() == 0, FIXTURE);
            continue;
        }
        let mut worker = ok(OwnedWorker::new(f.options()));
        ok(worker.request(&begin_job(&ids, false), soon(), Some(&mut never()), None));
        ok(worker.request(
            &input_job(8, false, &policy(), 0),
            soon(),
            Some(&mut never()),
            None,
        ));
        let mut callbacks = 0usize;
        let generated = worker.request(
            &generation_job(8),
            soon(),
            Some(&mut never()),
            Some(&mut |_: &Json| {
                callbacks += 1;
                Ok(())
            }),
        );
        if mode == "bad-fast-cost" {
            refuses(generated);
            check(
                callbacks == 0 && !worker.alive() && f.commands() == 3,
                FIXTURE,
            );
        } else {
            ok(generated);
            check(callbacks == 6 && worker.alive(), FIXTURE);
        }
        ok(worker.close());
    }
    for present in [false, true] {
        let mut f = worker_files(executable, "normal", present, false);
        f.options.fast_channel = Some(channel);
        refuses(OwnedWorker::new(f.options()));
        check(
            !f.directory.join("started").exists(),
            "fast math without staged ON launched a child",
        );
    }
}

/// The C++ `main`.
pub fn main(args: &[String]) -> i32 {
    if args.len() > 1 && args[1] == "target" {
        if args[2..]
            .iter()
            .any(|arg| arg == "--qsa-channel-chronological")
        {
            return target(args);
        }
        return branch8::main(args);
    }
    let prior = branch8::main(args);
    if prior != 0 {
        return prior;
    }
    for channel in [false, true] {
        admission_and_rounds(channel);
        section(&format!("admission_and_rounds({channel})"));
        workers(&fake_executable(args), channel);
        section(&format!("workers({channel})"));
    }
    println!("Private HTTP fast-math CPU checks={}", checks());
    0
}
