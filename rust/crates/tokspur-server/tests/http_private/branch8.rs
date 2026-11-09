//! Port of `native/tests/test_http_branch8.cpp`, check for check.
//!
//! Original deterministic fixtures. Like the C++, this suite first runs every private-V8 refusal/worker/session check (the `verify8` module); that protocol remains a distinct arm. The fast-math suite includes this module.
//!
//! The worker launches this binary as `branch8-target`; `main` dispatches on that basename, as the C++ does.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::path::Path;

use tokspur_json::{Json, parse_json};
use tokspur_server::session::{ServeSession, ServeSink, SessionOptions, TokenCallback};
use tokspur_wire::verify8::{
    Verify8Context, Verify8Error, Verify8Reply, validate_branch8_control, validate_branch8_ready,
    validate_verify8_ready,
};
use tokspur_wire::worker::{OwnedWorker, WorkerError};

use crate::verify8::{
    self, FIXTURE, Round, TOOL_CALL, WorkerFixture, append_line, array, boolean, cached_tokens,
    chat, check, checks, choice, count, delta, dump, emit, encode_text, fake_executable,
    fake_target, field, fixed_once, generation_job, grams, input_job, int, integer,
    make_transcript, never, number, ok, parse, pause_ms, policy, policy_argument, policy_parts,
    present, put, read_line, refuses, remove, section, session_codec, soon, text, tools, v8_reply,
    with, without,
};

pub const COSTS: [&str; 10] = [
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

pub const GRAPH: [&str; 5] = [
    "tracked_metal_buffer_bytes",
    "peak_metal_buffer_bytes",
    "gpu_allocation_count",
    "gpu_buffer_limit_bytes",
    "imported_weight_backing_bytes",
];

const STORAGE: [&str; 5] = [
    "checkpoint_slots",
    "checkpoint_combined_budget_bytes",
    "checkpoint_host_bytes",
    "checkpoint_store_host_bytes",
    "checkpoint_gpu_bytes",
];

#[cfg(feature = "qsa-stage-v8")]
const STAGE: [&str; 7] = [
    "qsa_stage_v8_prepared",
    "qsa_stage_v8_enabled",
    "qsa_stage_v8_scratch_logical_bytes",
    "qsa_stage_v8_scratch_charged_bytes",
    "qsa_stage_v8_calls",
    "qsa_stage_v8_physical_dispatches",
    "qsa_stage_v8_extra_dispatches",
];

/// Lifetime costs a model command carries forward from the native state.
pub const MAINTENANCE: [&str; 6] = [
    "lookup_truncated_grams",
    "lookup_cleared_grams",
    "lookup_truncations",
    "lookup_candidates_examined",
    "lookup_token_comparisons",
    "lookup_work_limited_queries",
];

/// The broker fields the worker adds to each wire command.
pub const WIRE: [&str; 6] = [
    "broker_epoch",
    "command_id",
    "lease_id",
    "expected_prefix",
    "broker_protocol",
    "broker_batch_width",
];

fn sampled(policy_value: &Json, seed: u64, prefix: u64) -> Json {
    let mut value = with(policy_value, "seed", int(seed));
    put(&mut value, "valid", Json::Bool(true));
    put(&mut value, "entropy_words", int(0));
    put(&mut value, "entropy_prefix", int(prefix));
    value
}

/// `envelope.find(name)`, else `previous.find(name)` when `previous` is set.
fn source<'a>(envelope: &'a Json, previous: &'a Json, name: &str) -> Option<&'a Json> {
    match ok(envelope.find(name)) {
        None if !previous.is_null() => ok(previous.find(name)),
        value => value,
    }
}

/// `snapshot(envelope, previous)`; pass `Json::Null` for the C++ default.
pub fn snapshot(envelope: &Json, previous: &Json) -> Json {
    let mut out = Vec::new();
    for name in [
        "committed_prefix",
        "sampling",
        "logits_available",
        "continuation_allowed",
        "stop_id",
        "final_logit_top8",
        "hc_exact_final_logit_sha256",
        "hc_exact_routes",
        "lookup_history_epoch",
    ] {
        let value = source(envelope, previous, name).expect("fixture snapshot source incomplete");
        out.push((name.as_bytes().to_vec(), value.clone()));
    }
    for name in COSTS {
        let value = source(envelope, previous, name)
            .cloned()
            .unwrap_or_else(|| int(0));
        out.push((name.as_bytes().to_vec(), value));
    }
    for name in GRAPH {
        let value =
            source(envelope, previous, name).expect("fixture graph snapshot source incomplete");
        out.push((name.as_bytes().to_vec(), value.clone()));
    }
    #[cfg(feature = "qsa-stage-v8")]
    if source(envelope, previous, "qsa_stage_v8_prepared").is_some() {
        for name in STAGE {
            let value =
                source(envelope, previous, name).expect("fixture staged-QSA snapshot incomplete");
            out.push((name.as_bytes().to_vec(), value.clone()));
        }
    }
    Json::Object(out)
}

/// `stamp(value, command, lease, state)`; pass `Json::Null` for no state.
pub fn stamp(mut value: Json, command: u64, lease: u64, state: &Json) -> Json {
    put(&mut value, "broker_protocol", Json::from("branch8-v1"));
    put(&mut value, "broker_batch_width", int(8));
    put(&mut value, "broker_epoch", int(4242));
    put(&mut value, "command_id", int(command));
    put(&mut value, "lease_id", int(lease));
    if !state.is_null() {
        put(&mut value, "branch8_session", state.clone());
    }
    value
}

pub fn prepared() -> Json {
    let mut ready = verify8::ready_record();
    for (name, amount) in [
        ("checkpoint_slots", 4u64),
        ("checkpoint_combined_budget_bytes", 512 << 20),
        ("checkpoint_host_bytes", 5_229_136),
        ("checkpoint_store_host_bytes", 368),
        ("checkpoint_gpu_bytes", 472_186_880),
        ("tracked_metal_buffer_bytes", 84_277_411_004),
        ("peak_metal_buffer_bytes", 84_277_411_004),
        ("gpu_allocation_count", 616),
    ] {
        put(&mut ready, name, int(amount));
    }
    put(&mut ready, "verification8_branch_reuse", Json::Bool(true));
    put(&mut ready, "checkpoint_broker", Json::Bool(true));
    let mut state_source = ready.clone();
    put(&mut state_source, "committed_prefix", int(0));
    put(&mut state_source, "sampling", sampled(&policy(), 0, 0));
    put(&mut state_source, "logits_available", Json::Bool(false));
    put(&mut state_source, "continuation_allowed", Json::Bool(false));
    put(&mut state_source, "stop_id", Json::Null);
    put(
        &mut state_source,
        "final_logit_top8",
        Json::Array(Vec::new()),
    );
    put(&mut state_source, "hc_exact_final_logit_sha256", Json::Null);
    put(&mut state_source, "lookup_history_epoch", int(1));
    let state = snapshot(&state_source, &Json::Null);
    stamp(ready, 0, 0, &state)
}

#[cfg(feature = "qsa-stage-v8")]
pub fn staged_prepared(enabled: bool) -> Json {
    let mut ready = prepared();
    put(
        &mut ready,
        "allow_prompt_lookup_experiments",
        Json::Bool(false),
    );
    put(&mut ready, "qsa_stage_v8_prepared", Json::Bool(true));
    put(&mut ready, "qsa_stage_v8_enabled", Json::Bool(enabled));
    for (name, value) in [
        ("qsa_stage_v8_scratch_logical_bytes", 196_896u64),
        ("qsa_stage_v8_scratch_charged_bytes", 212_992),
        ("qsa_stage_v8_calls", 0),
        ("qsa_stage_v8_physical_dispatches", 0),
        ("qsa_stage_v8_extra_dispatches", 0),
        ("tracked_metal_buffer_bytes", 84_277_623_996),
        ("peak_metal_buffer_bytes", 84_277_623_996),
        ("gpu_allocation_count", 617),
    ] {
        put(&mut ready, name, int(value));
    }
    let mut state = field(&ready, "branch8_session").clone();
    for name in GRAPH {
        put(&mut state, name, field(&ready, name).clone());
    }
    for name in STAGE {
        put(&mut state, name, field(&ready, name).clone());
    }
    put(&mut ready, "branch8_session", state);
    ready
}

pub fn begin_job(prompt: &[u32], reuse: bool) -> Json {
    let ids = prompt.iter().map(|&id| int(u64::from(id))).collect();
    Json::new_object([
        ("op", Json::from("branch_begin")),
        ("prompt_ids", Json::Array(ids)),
        ("generation_requested", Json::Bool(true)),
        ("allow_restore", Json::Bool(reuse)),
        ("sampling", policy()),
        ("seed", int(0)),
    ])
}

/// `Verify8Reply(ready, previous, job, before, sampling, 0, false, true)`.
pub fn branch_reply<'a>(
    ready: &'a Json,
    previous: &'a Json,
    job: &'a Json,
    before: u32,
    policy: &'a Json,
) -> Result<Verify8Reply<'a>, Verify8Error> {
    Verify8Reply::new(Verify8Context {
        ready,
        previous,
        job,
        before,
        policy,
        seed: 0,
        reset_resync: false,
        branch8: true,
    })
}

pub struct Saved {
    prefix: u32,
    state: Json,
}

/// The branch8 native fake: zero-step controls over an exact history and up to four saved ancestors, and model commands built by `make_transcript`.
pub struct Fake {
    pub ready: Json,
    pub previous: Json,
    pub state: Json,
    sampling: Json,
    pub command: u64,
    pub lease: u64,
    seed: u64,
    pub history: Vec<u32>,
    output: Vec<u32>,
    next: usize,
    pub saved: VecDeque<Saved>,
    pub input_widths: Vec<u32>,
    pub captured_prefixes: Vec<u32>,
    pub restored_prefixes: Vec<u32>,
    pub corrupt_done: bool,
}

impl Fake {
    pub fn new(prepared_ready: Json) -> Self {
        let state = field(&prepared_ready, "branch8_session").clone();
        Self {
            previous: prepared_ready.clone(),
            ready: prepared_ready,
            state,
            sampling: policy(),
            command: 0,
            lease: 0,
            seed: 0,
            history: Vec::new(),
            output: encode_text(b"ok<|im_end|>"),
            next: 0,
            saved: VecDeque::new(),
            input_widths: Vec::new(),
            captured_prefixes: Vec::new(),
            restored_prefixes: Vec::new(),
            corrupt_done: false,
        }
    }

    /// `plan(text = "ok<|im_end|>")`.
    pub fn plan(&mut self, text: &str) {
        self.output = encode_text(text.as_bytes());
        self.next = 0;
    }

    fn control(&mut self, job: &Json) -> Json {
        let op = text(job, "op").to_vec();
        let before = self.history.len() as u32;
        let mut common = before;
        let outcome;
        if op == b"branch_begin" {
            self.lease += 1;
            self.sampling = field(job, "sampling").clone();
            self.seed = integer(job, "seed");
            common = 0;
            let reuse = boolean(job, "allow_restore");
            let prompt = array(job, "prompt_ids");
            if reuse {
                while (common as usize) < self.history.len()
                    && (common as usize) < prompt.len()
                    && u64::from(self.history[common as usize])
                        == ok(prompt[common as usize].unsigned_integer())
                {
                    common += 1;
                }
            }
            let mut selected: Option<(u32, Json)> = None;
            if reuse {
                for record in &self.saved {
                    if record.prefix <= common
                        && selected
                            .as_ref()
                            .is_none_or(|(prefix, _)| record.prefix >= *prefix)
                    {
                        selected = Some((record.prefix, record.state.clone()));
                    }
                }
            }
            let prefix = selected.as_ref().map_or(0, |(prefix, _)| *prefix);
            let mut replacement = self.state.clone();
            if let Some((_, saved)) = &selected {
                for name in [
                    "committed_prefix",
                    "logits_available",
                    "continuation_allowed",
                    "stop_id",
                    "final_logit_top8",
                    "hc_exact_final_logit_sha256",
                ] {
                    put(&mut replacement, name, field(saved, name).clone());
                }
                let truncations = integer(&self.state, "lookup_truncations") + 1;
                put(&mut replacement, "lookup_truncations", int(truncations));
                let truncated = integer(&self.state, "lookup_truncated_grams")
                    + u64::from(grams(before))
                    - u64::from(grams(prefix));
                put(&mut replacement, "lookup_truncated_grams", int(truncated));
                outcome = "restored";
                self.restored_prefixes.push(prefix);
                self.saved.retain(|record| record.prefix <= prefix);
            } else {
                self.saved.clear();
                outcome = "cold_miss";
                put(&mut replacement, "committed_prefix", int(0));
                put(&mut replacement, "logits_available", Json::Bool(false));
                put(&mut replacement, "continuation_allowed", Json::Bool(false));
                put(&mut replacement, "stop_id", Json::Null);
                put(
                    &mut replacement,
                    "final_logit_top8",
                    Json::Array(Vec::new()),
                );
                put(&mut replacement, "hc_exact_final_logit_sha256", Json::Null);
                let resets = integer(&self.state, "lookup_resets") + 1;
                put(&mut replacement, "lookup_resets", int(resets));
                let cleared =
                    integer(&self.state, "lookup_cleared_grams") + u64::from(grams(before));
                put(&mut replacement, "lookup_cleared_grams", int(cleared));
            }
            let epoch = integer(&self.state, "lookup_history_epoch") + 1;
            put(&mut replacement, "lookup_history_epoch", int(epoch));
            put(
                &mut replacement,
                "sampling",
                sampled(&self.sampling, self.seed, u64::from(prefix)),
            );
            self.state = replacement;
            self.history.truncate(prefix as usize);
            self.next = 0;
        } else {
            check(op == b"branch_capture" && before > 0, FIXTURE);
            if self.saved.len() == 4 {
                self.saved.pop_front();
            }
            self.saved.push_back(Saved {
                prefix: before,
                state: self.state.clone(),
            });
            self.captured_prefixes.push(before);
            outcome = "captured";
        }
        let mut fields = vec![
            (b"event".to_vec(), Json::from("control_done")),
            (b"op".to_vec(), Json::from(op.as_slice())),
            (b"outcome".to_vec(), Json::from(outcome)),
            (b"model_steps".to_vec(), int(0)),
            (b"generated_ids".to_vec(), Json::Array(Vec::new())),
            (b"common_prefix".to_vec(), int(u64::from(common))),
            (b"control_wall_ms".to_vec(), number(1.0)),
            (b"timings_qualified".to_vec(), Json::Bool(false)),
            (
                b"checkpoint_retained".to_vec(),
                int(self.saved.len() as u64),
            ),
        ];
        for name in [
            "committed_prefix",
            "sampling",
            "stop_id",
            "logits_available",
            "continuation_allowed",
        ] {
            fields.push((name.as_bytes().to_vec(), field(&self.state, name).clone()));
        }
        for name in STORAGE {
            fields.push((name.as_bytes().to_vec(), field(&self.ready, name).clone()));
        }
        stamp(Json::Object(fields), self.command, self.lease, &self.state)
    }

    /// `operator()(job, callback)`. A callback failure propagates after the native state has committed, as the C++ exception would.
    pub fn call<E>(
        &mut self,
        job: &Json,
        callback: &mut dyn FnMut(&Json) -> Result<(), E>,
    ) -> Result<Json, E> {
        self.command += 1;
        if present(job, "op") {
            self.previous = self.control(job);
            return Ok(self.previous.clone());
        }
        check(
            !boolean(job, "reset"),
            "new branch wire reset must be a distinct control",
        );
        let n = integer(job, "generate") as u32;
        let input = array(job, "tokens").len() as u32;
        let mut rounds = Vec::new();
        let first = self.next;
        let mut t = 0;
        while t < n && self.next < self.output.len() {
            let terminal = self.output[self.next] == 248_046;
            rounds.push(Round {
                terminal,
                ..Round::of(0, 0, 1)
            });
            if terminal {
                self.next += 1;
                break;
            }
            t += 1;
            self.next += 1;
        }
        let mut transcript = make_transcript(
            &self.ready,
            &self.state,
            job.clone(),
            &self.sampling,
            self.seed,
            &rounds,
        );
        for name in MAINTENANCE {
            put(&mut transcript.done, name, field(&self.state, name).clone());
        }
        let mut ids = Vec::new();
        let mut at = first;
        for event in &mut transcript.events {
            if text(event, "event") == b"token" {
                ids.push(int(u64::from(self.output[at])));
                put(event, "id", int(u64::from(self.output[at])));
                at += 1;
            }
        }
        put(
            &mut transcript.done,
            "generated_ids",
            Json::Array(ids.clone()),
        );
        let mut top = array(&transcript.done, "final_logit_top8").clone();
        let next = self.output.get(self.next).copied().unwrap_or(200);
        top[0] = with(&top[0], "id", int(u64::from(next)));
        put(&mut transcript.done, "final_logit_top8", Json::Array(top));
        put(&mut transcript.done, "logits_available", Json::Bool(true));
        put(
            &mut transcript.done,
            "checkpoint_retained",
            int(self.saved.len() as u64),
        );
        put(
            &mut transcript.done,
            "checkpoint_capture_wall_ms",
            number(0.0),
        );
        put(
            &mut transcript.done,
            "checkpoint_restore_wall_ms",
            number(0.0),
        );
        for id in array(job, "tokens") {
            self.history.push(ok(id.unsigned_integer()) as u32);
        }
        for id in &ids {
            self.history.push(ok(id.unsigned_integer()) as u32);
        }
        if input != 0 {
            self.input_widths.push(input);
        }
        self.state = snapshot(&transcript.done, &self.state);
        self.previous = stamp(
            without(&transcript.done, "branch8_session"),
            self.command,
            self.lease,
            &self.state,
        );
        for event in transcript.events {
            callback(&stamp(event, self.command, self.lease, &Json::Null))?;
        }
        if self.corrupt_done && n != 0 {
            return Ok(with(
                &self.previous,
                "hc_exact_final_logit_sha256",
                Json::from("malformed"),
            ));
        }
        Ok(self.previous.clone())
    }

    /// `operator()` with a callback that cannot fail.
    pub fn apply(&mut self, job: &Json, mut callback: impl FnMut(&Json)) -> Json {
        let Ok(done) = self.call(job, &mut |event: &Json| {
            callback(event);
            Ok::<(), Infallible>(())
        });
        done
    }
}

/// `result.tokspur_private[name]`.
fn private(result: &Json, name: &str) -> u64 {
    integer(field(result, "tokspur_private"), name)
}

fn branch_options(ready: &Json) -> SessionOptions {
    SessionOptions {
        broker_ready: ready.clone(),
        verify8: true,
        verify_ready: ready.clone(),
        branch8: true,
        ..SessionOptions::default()
    }
}

pub fn admission_checks(ready: &Json) {
    let sampling_policy = policy();
    ok(validate_branch8_ready(ready));
    count();
    refuses(validate_verify8_ready(ready));
    refuses(ServeSession::new(
        session_codec(),
        SessionOptions {
            broker_ready: ready.clone(),
            verify8: true,
            verify_ready: ready.clone(),
            ..SessionOptions::default()
        },
    ));
    refuses(ServeSession::new(
        session_codec(),
        SessionOptions {
            broker_ready: ready.clone(),
            branch8: true,
            ..SessionOptions::default()
        },
    ));
    for key in [
        "broker_protocol",
        "broker_batch_width",
        "broker_epoch",
        "command_id",
        "lease_id",
        "verification8_branch_reuse",
        "branch8_session",
    ] {
        refuses(validate_branch8_ready(&without(ready, key)));
    }
    for (key, value) in [
        ("broker_protocol", Json::from("ordinary")),
        ("broker_batch_width", int(1)),
        ("checkpoint_slots", int(0)),
        ("checkpoint_slots", int(5)),
        ("checkpoint_combined_budget_bytes", int((512 << 20) + 1)),
        ("checkpoint_gpu_bytes", int(512 << 20)),
        ("verification8_branch_reuse", Json::Bool(false)),
        ("command_id", int(1)),
        ("lease_id", int(1)),
        ("lookup_verification_budget_bytes", int(512 << 20)),
    ] {
        refuses(validate_branch8_ready(&with(ready, key, value)));
    }
    let session = field(ready, "branch8_session");
    for (key, _) in ok(session.object()) {
        refuses(validate_branch8_ready(&with(
            ready,
            "branch8_session",
            without(session, key),
        )));
        refuses(validate_branch8_ready(&with(
            ready,
            "branch8_session",
            with(session, key, Json::from("wrong")),
        )));
    }
    let mut fake = Fake::new(ready.clone());
    let initial = fake.previous.clone();
    let ids = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let job = begin_job(&ids, false);
    let cold = fake.apply(&job, |_| check(false, FIXTURE));
    ok(validate_branch8_control(
        ready, &initial, &job, &cold, false,
    ));
    count();
    for key in [
        "model_steps",
        "command_id",
        "common_prefix",
        "committed_prefix",
        "checkpoint_retained",
    ] {
        refuses(validate_branch8_control(
            ready,
            &initial,
            &job,
            &with(&cold, key, int(999)),
            false,
        ));
    }
    refuses(validate_branch8_control(
        ready,
        &initial,
        &job,
        &with(&cold, "lease_id", int(0)),
        false,
    ));
    // Fresh broker leases are strictly increasing, not necessarily contiguous. Exact echo of the parent's issued lease is a separate worker boundary.
    ok(validate_branch8_control(
        ready,
        &initial,
        &job,
        &with(&cold, "lease_id", int(999)),
        false,
    ));
    count();
    for name in COSTS {
        let session = field(&cold, "branch8_session");
        let state = with(session, name, int(integer(session, name) + 1));
        refuses(validate_branch8_control(
            ready,
            &initial,
            &job,
            &with(&cold, "branch8_session", state),
            false,
        ));
    }
    for width in 1..=8 {
        let before = fake.previous.clone();
        let input = input_job(width, false, &policy(), 0);
        let done = fake.apply(&input, |_| check(false, FIXTURE));
        let prefix = integer(field(&before, "branch8_session"), "committed_prefix") as u32;
        let mut reply = ok(branch_reply(
            ready,
            &before,
            &input,
            prefix,
            &sampling_policy,
        ));
        refuses(reply.events());
        ok(reply.finish(&done));
        check(ok(reply.events()).is_empty(), FIXTURE);
    }
    let capture_job = parse(r#"{"op":"branch_capture"}"#);
    let before = fake.previous.clone();
    let captured = fake.apply(&capture_job, |_| check(false, FIXTURE));
    ok(validate_branch8_control(
        ready,
        &before,
        &capture_job,
        &captured,
        false,
    ));
    count();
    let invalid_capture = with(
        &captured,
        "branch8_session",
        with(
            field(&captured, "branch8_session"),
            "lookup_queries",
            int(1),
        ),
    );
    refuses(validate_branch8_control(
        ready,
        &before,
        &capture_job,
        &invalid_capture,
        false,
    ));
    fake.plan("ok<|im_end|>");
    let previous = fake.previous.clone();
    let generate = generation_job(8);
    let mut events = Vec::new();
    let done = fake.apply(&generate, |event| events.push(event.clone()));
    let prefix = integer(field(&previous, "branch8_session"), "committed_prefix") as u32;
    let mut reply = ok(branch_reply(
        ready,
        &previous,
        &generate,
        prefix,
        &sampling_policy,
    ));
    for event in &events {
        ok(reply.event(event));
        refuses(reply.events());
    }
    ok(reply.finish(&done));
    check(
        ok(reply.events()).len() == 6 && text(&done, "status") == b"stopped",
        FIXTURE,
    );
    refuses((|| {
        let mut old = v8_reply(ready, &previous, &generate, 36, &sampling_policy, 0)?;
        for event in &events {
            old.event(event)?;
        }
        old.finish(&done)
    })());
    let restore_job = begin_job(&fake.history[..36], true);
    let restored = fake.apply(&restore_job, |_| check(false, FIXTURE));
    ok(validate_branch8_control(
        ready,
        &done,
        &restore_job,
        &restored,
        false,
    ));
    count();
    let restored_state = field(&restored, "branch8_session");
    check(
        integer(&restored, "committed_prefix") == 36
            && integer(restored_state, "lookup_truncated_grams") == 3,
        FIXTURE,
    );
    check(
        dump(field(restored_state, "hc_exact_routes")) == dump(field(&done, "hc_exact_routes")),
        "restoring model state rewound execution cost",
    );
    for n in 2..=7u32 {
        for k in 1..=n {
            let mut rounds = vec![Round::of(n, k, u32::from(k < n))];
            for _ in k + u32::from(k < n)..n {
                rounds.push(Round::of(0, 0, 1));
            }
            let mut t = make_transcript(
                ready,
                restored_state,
                generation_job(n),
                &policy(),
                0,
                &rounds,
            );
            for name in MAINTENANCE {
                put(&mut t.done, name, field(restored_state, name).clone());
            }
            put(&mut t.done, "logits_available", Json::Bool(true));
            put(&mut t.done, "checkpoint_capture_wall_ms", number(0.0));
            put(&mut t.done, "checkpoint_restore_wall_ms", number(0.0));
            let mut output = array(&t.done, "generated_ids").clone();
            output[0] = int(112);
            put(&mut t.done, "generated_ids", Json::Array(output));
            let mut first = true;
            for event in &mut t.events {
                if text(event, "event") == b"token" && first {
                    put(event, "id", int(112));
                    first = false;
                }
            }
            let command = integer(&restored, "command_id") + 1;
            let lease = integer(&restored, "lease_id");
            let state = snapshot(&t.done, restored_state);
            t.done = stamp(without(&t.done, "branch8_session"), command, lease, &state);
            for event in &mut t.events {
                *event = stamp(event.clone(), command, lease, &Json::Null);
            }
            let mut candidate = ok(branch_reply(ready, &restored, &t.job, 36, &sampling_policy));
            for event in &t.events {
                ok(candidate.event(event));
            }
            ok(candidate.finish(&t.done));
            check(ok(candidate.events()).len() == t.events.len(), FIXTURE);
            for (key, value) in [
                ("broker_batch_width", int(1)),
                ("broker_epoch", int(43)),
                ("command_id", int(command + 1)),
                ("lease_id", int(lease + 1)),
            ] {
                let mut bad = ok(branch_reply(ready, &restored, &t.job, 36, &sampling_policy));
                refuses(bad.event(&with(&t.events[0], key, value)));
                refuses(bad.events());
            }
        }
    }
}

pub fn session_checks(ready: Json) {
    let fake = RefCell::new(Fake::new(ready));
    let options = branch_options(&fake.borrow().ready);
    let mut session = ok(ServeSession::new(session_codec(), options));
    let cancelled = Cell::new(false);
    let chunks: RefCell<Vec<Json>> = RefCell::new(Vec::new());
    let mut push = |event: &Json| {
        chunks.borrow_mut().push(event.clone());
        Ok(())
    };
    let mut is_cancelled = || cancelled.get();
    let mut new_seed = || 0;
    let mut sink = ServeSink {
        chunk: Some(&mut push),
        cancelled: &mut is_cancelled,
        new_seed: &mut new_seed,
        prefix_trace: None,
    };
    let mut target = |job: &Json, event: TokenCallback<'_>| fake.borrow_mut().call(job, event);
    refuses(session.complete(
        &chat(&[b'x'; 65536], false, 8),
        b"context-overflow",
        &mut target,
        &mut sink,
    ));
    check(fake.borrow().command == 0, FIXTURE);
    let mut prompt = vec![b'a'; 2052];
    let result = ok(session.complete(
        &chat(&prompt, false, 8),
        b"cold2052",
        &mut target,
        &mut sink,
    ));
    check(
        fake.borrow().captured_prefixes == [512, 1024, 1536, 2048, 2052],
        "branch8 cloned every tile or missed scheduled ancestors",
    );
    check(
        private(&result, "checkpoint_captures") == 5 && cached_tokens(&result) == 0,
        FIXTURE,
    );
    fake.borrow_mut().plan("ok<|im_end|>");
    let result = ok(session.complete(
        &chat(&prompt, false, 8),
        b"exact2052",
        &mut target,
        &mut sink,
    ));
    check(
        cached_tokens(&result) == 2052 && fake.borrow().restored_prefixes.last() == Some(&2052),
        FIXTURE,
    );
    check(
        private(&result, "known_input_batches") == 0
            && private(&result, "checkpoint_restores") == 1,
        FIXTURE,
    );
    if let Some(last) = prompt.last_mut() {
        *last = b'b';
    }
    fake.borrow_mut().plan("ok<|im_end|>");
    let result = ok(session.complete(
        &chat(&prompt, false, 8),
        b"sibling2048",
        &mut target,
        &mut sink,
    ));
    check(
        cached_tokens(&result) == 2048 && fake.borrow().input_widths.last() == Some(&4),
        FIXTURE,
    );
    fake.borrow_mut().plan("ok<|im_end|>");
    let result = ok(session.complete(
        &chat(&prompt[..1536], false, 8),
        b"short1536",
        &mut target,
        &mut sink,
    ));
    check(cached_tokens(&result) == 1536, FIXTURE);
    let append = [&prompt[..1536], b"ok<|im_end|>suffix"].concat();
    fake.borrow_mut().plan("ok<|im_end|>");
    let result = ok(session.complete(&chat(&append, false, 8), b"append", &mut target, &mut sink));
    check(cached_tokens(&result) == 1539, FIXTURE);
    prompt[0] = b'z';
    fake.borrow_mut().plan("ok<|im_end|>");
    let result = ok(session.complete(
        &chat(&prompt, false, 8),
        b"deep-cold",
        &mut target,
        &mut sink,
    ));
    check(cached_tokens(&result) == 0, FIXTURE);
    let tool = with(&chat(b"tool-root", true, 128), "tools", tools());
    fake.borrow_mut().plan(TOOL_CALL);
    let result = ok(session.complete(&tool, b"terminal-tool", &mut target, &mut sink));
    let selected = choice(&result);
    check(
        text(selected, "finish_reason") == b"tool_calls"
            && array(field(selected, "message"), "tool_calls").len() == 1,
        "new branch path lost terminal tool admission",
    );
    fake.borrow_mut().plan("ok<|im_end|>");
    let mut cancel_after_done = |job: &Json, event: TokenCallback<'_>| {
        let done = fake.borrow_mut().call(job, event)?;
        if !present(job, "op") && integer(job, "generate") != 0 {
            cancelled.set(true);
        }
        Ok(done)
    };
    refuses(session.complete(
        &chat(&prompt, true, 8),
        b"cancel",
        &mut cancel_after_done,
        &mut sink,
    ));
    check(session.committed_tokens() == 0, FIXTURE);
    cancelled.set(false);
    fake.borrow_mut().plan("ok<|im_end|>");
    let result = ok(session.complete(
        &chat(&prompt, false, 8),
        b"cold-resync",
        &mut target,
        &mut sink,
    ));
    check(
        cached_tokens(&result) == 0 && private(&result, "checkpoint_cold_resyncs") == 1,
        FIXTURE,
    );
    for &width in &fake.borrow().input_widths {
        check(width != 0 && width <= 8, FIXTURE);
    }
    let command_before = fake.borrow().command;
    refuses(session.complete(
        &with(&chat(b"x", false, 8), "temperature", number(0.5)),
        b"non-greedy",
        &mut target,
        &mut sink,
    ));
    check(fake.borrow().command == command_before, FIXTURE);
    chunks.borrow_mut().clear();
    fake.borrow_mut().plan("ok<|im_end|>");
    fake.borrow_mut().corrupt_done = true;
    refuses(session.complete(
        &chat(b"fresh", true, 8),
        b"bad-snapshot",
        &mut target,
        &mut sink,
    ));
    check(session.committed_tokens() == 0, FIXTURE);
    for event in chunks.borrow().iter() {
        let delta = delta(event);
        check(
            !present(delta, "content") && !present(delta, "tool_calls"),
            FIXTURE,
        );
    }
}

pub fn cancelled_rewind_resync_checks(ready: Json) {
    let fake = RefCell::new(Fake::new(ready));
    let fake_ready = fake.borrow().ready.clone();
    let mut session = ok(ServeSession::new(
        session_codec(),
        branch_options(&fake_ready),
    ));
    let cancelled = Cell::new(false);
    let mut discard = |_: &Json| Ok(());
    let mut is_cancelled = || cancelled.get();
    let mut new_seed = || 0;
    let mut sink = ServeSink {
        chunk: Some(&mut discard),
        cancelled: &mut is_cancelled,
        new_seed: &mut new_seed,
        prefix_trace: None,
    };
    let mut target = |job: &Json, event: TokenCallback<'_>| fake.borrow_mut().call(job, event);
    let prompt = vec![b'a'; 2052];
    ok(session.complete(
        &chat(&prompt, false, 8),
        b"rewind-prime",
        &mut target,
        &mut sink,
    ));
    let stale = fake.borrow().previous.clone();
    check(
        integer(field(&stale, "branch8_session"), "committed_prefix") == 2055
            && fake.borrow().saved.len() == 4,
        "rewind cancellation regression lacks an admitted long prefix and saved ancestors",
    );
    let restored = RefCell::new(Json::Null);
    let begins = |job: &Json| present(job, "op") && text(job, "op") == b"branch_begin";
    let mut cancel_after_restore = |job: &Json, event: TokenCallback<'_>| {
        let worker_before = fake.borrow().previous.clone();
        let done = fake.borrow_mut().call(job, event)?;
        if begins(job) {
            // Exact worker-side admission still succeeds.
            ok(validate_branch8_control(
                &fake_ready,
                &worker_before,
                job,
                &done,
                false,
            ));
            check(
                text(&done, "outcome") == b"restored" && integer(&done, "committed_prefix") == 1536,
                "cancellation happened without actually rewinding the worker",
            );
            *restored.borrow_mut() = done.clone();
            cancelled.set(true); // HTTP's post-target cancellation check prevents reply publication.
        }
        Ok(done)
    };
    fake.borrow_mut().plan("ok<|im_end|>");
    refuses(session.complete(
        &chat(&prompt[..1536], false, 8),
        b"cancel-restored-control",
        &mut cancel_after_restore,
        &mut sink,
    ));
    check(
        cancelled.get()
            && session.committed_tokens() == 0
            && !restored.borrow().is_null()
            && fake.borrow().history.len() == 1536,
        "cancelled shorter control did not leave the worker restored and HTTP invalidated",
    );
    cancelled.set(false);
    fake.borrow_mut().plan("ok<|im_end|>");
    let checked = Cell::new(false);
    let mut cold_after_rewind = |job: &Json, event: TokenCallback<'_>| {
        let worker_before = fake.borrow().previous.clone();
        let done = fake.borrow_mut().call(job, event)?;
        if begins(job) {
            checked.set(true);
            check(
                !boolean(job, "allow_restore")
                    && text(&done, "outcome") == b"cold_miss"
                    && integer(&done, "committed_prefix") == 0
                    && integer(&done, "common_prefix") == 0,
                "rewind resync attempted a restore or reported a cache hit",
            );
            let now = field(&done, "branch8_session");
            let old = field(&stale, "branch8_session");
            let cleared =
                integer(now, "lookup_cleared_grams") - integer(old, "lookup_cleared_grams");
            check(
                cleared == u64::from(grams(1536))
                    && cleared < u64::from(grams(integer(old, "committed_prefix") as u32)),
                "regression does not exercise fewer cleared grams than the stale HTTP prefix",
            );
            // Exact delta uses the authoritative 1536, not the stale 2055.
            ok(validate_branch8_control(
                &fake_ready,
                &worker_before,
                job,
                &done,
                false,
            ));
            // The explicitly cold HTTP resync can rebase.
            ok(validate_branch8_control(
                &fake_ready,
                &stale,
                job,
                &done,
                true,
            ));
            let cleared_less = integer(now, "lookup_cleared_grams") - 1;
            let broken = with(now, "lookup_cleared_grams", int(cleared_less));
            refuses(validate_branch8_control(
                &fake_ready,
                &worker_before,
                job,
                &with(&done, "branch8_session", broken),
                false,
            ));
            let broken = with(now, "lookup_resets", field(old, "lookup_resets").clone());
            refuses(validate_branch8_control(
                &fake_ready,
                &stale,
                job,
                &with(&done, "branch8_session", broken),
                true,
            ));
            let appended = integer(old, "lookup_appended_tokens") - 1;
            let broken = with(now, "lookup_appended_tokens", int(appended));
            refuses(validate_branch8_control(
                &fake_ready,
                &stale,
                job,
                &with(&done, "branch8_session", broken),
                true,
            ));
            refuses(validate_branch8_control(
                &fake_ready,
                &stale,
                &with(job, "allow_restore", Json::Bool(true)),
                &done,
                true,
            ));
            refuses(validate_branch8_control(
                &fake_ready,
                &stale,
                job,
                &with(&done, "outcome", Json::from("restored")),
                true,
            ));
        }
        Ok(done)
    };
    let result = ok(session.complete(
        &chat(&prompt, false, 8),
        b"cold-after-rewind",
        &mut cold_after_rewind,
        &mut sink,
    ));
    check(
        checked.get()
            && cached_tokens(&result) == 0
            && private(&result, "checkpoint_cold_resyncs") == 1
            && private(&result, "checkpoint_restores") == 0,
        "cancelled rewind failed to recover through an explicitly cold, zero-hit resync",
    );
}

pub fn target(args: &[String]) -> i32 {
    #[cfg(feature = "qsa-stage-v8")]
    let (stage_present, stage_enabled) = {
        let mut present = false;
        let mut enabled = false;
        for i in 2..args.len().saturating_sub(1) {
            if args[i] == "--qsa-stage-v8" {
                if present || (args[i + 1] != "off" && args[i + 1] != "on") {
                    return 90;
                }
                present = true;
                enabled = args[i + 1] == "on";
            }
        }
        (present, enabled)
    };
    #[cfg(not(feature = "qsa-stage-v8"))]
    let stage_present = false;
    let fixed: [(&str, &str); 18] = [
        ("--allow-full-model", "true"),
        ("--small-gemv-splits", "16"),
        ("--block-gemv-formats", "3"),
        ("--stack-execution", "direct"),
        ("--known-token-capacity", "8"),
        ("--reuse-residual-projections", "true"),
        ("--prompt-lookup", "true"),
        ("--verification-budget-bytes", "134217728"),
        (
            "--allow-prompt-lookup-experiments",
            if stage_present { "false" } else { "true" },
        ),
        ("--verification-max-tokens", "8"),
        ("--hc-exact", "combined"),
        ("--wide-q8-exact", "off"),
        ("--wide-q8-many", "off"),
        ("--verification8-branch-reuse", "true"),
        ("--worker-broker8", "true"),
        ("--checkpoint-slots", "4"),
        ("--checkpoint-budget-bytes", "536870912"),
        ("--worker-broker", "true"),
    ];
    if args.len() != 2 + 2 * (fixed.len() + 2) + 4 + 2 * usize::from(stage_present) {
        return 91;
    }
    if !fixed_once(args, &fixed) {
        return 92;
    }
    let policy_path = policy_argument(args);
    if policy_path.as_os_str().is_empty() {
        return 93;
    }
    let (directory, mode) = policy_parts(&policy_path);
    ok(std::fs::write(
        directory.join("started"),
        "explicit branch8 fake argv\n",
    ));
    #[cfg(feature = "qsa-stage-v8")]
    let mut fake = Fake::new(if stage_present {
        staged_prepared(stage_enabled)
    } else {
        prepared()
    });
    #[cfg(not(feature = "qsa-stage-v8"))]
    let mut fake = Fake::new(prepared());
    #[cfg_attr(not(feature = "qsa-stage-v8"), allow(unused_mut))]
    let mut initial = fake.ready.clone();
    #[cfg(feature = "qsa-stage-v8")]
    {
        if mode == "wrong-stage-arm" {
            initial = staged_prepared(!stage_enabled);
        }
        if mode == "unprepared-stage" {
            initial = prepared();
        }
    }
    emit(&initial);
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    while let Some(line) = read_line(&mut input) {
        append_line(&directory.join("commands"), "command\n");
        let wire = ok(parse_json(&line));
        if integer(&wire, "broker_epoch") != 4242
            || integer(&wire, "command_id") != fake.command + 1
            || integer(&wire, "expected_prefix") != fake.history.len() as u64
            || text(&wire, "broker_protocol") != b"branch8-v1"
            || integer(&wire, "broker_batch_width") != 8
        {
            return 94;
        }
        let mut job = wire.clone();
        for name in WIRE {
            remove(&mut job, name);
        }
        let begin = present(&job, "op") && text(&job, "op") == b"branch_begin";
        if integer(&wire, "lease_id") != fake.lease + u64::from(begin) {
            return 95;
        }
        let generate = !present(&job, "op") && integer(&job, "generate") != 0;
        if begin {
            fake.plan("ok<|im_end|>");
        }
        let mut done = fake.apply(&job, |event| {
            if mode == "wrong-width" {
                emit(&with(event, "broker_batch_width", int(1)));
            } else {
                emit(event);
            }
        });
        if mode == "wrong-control-lease" && begin {
            put(&mut done, "lease_id", int(999));
        }
        if generate {
            ok(std::fs::write(
                directory.join("events-sent"),
                "events emitted, done withheld\n",
            ));
            if mode == "truncated" {
                return 0;
            }
            if mode == "slow-done" {
                let observed = directory.join("cancel-observed");
                for _ in 0..100 {
                    if observed.exists() {
                        break;
                    }
                    pause_ms(5);
                }
                if !observed.exists() {
                    return 96;
                }
            }
            if mode == "bad-snapshot" {
                let broken = with(field(&done, "branch8_session"), "lookup_resets", int(999));
                put(&mut done, "branch8_session", broken);
            }
            #[cfg(feature = "qsa-stage-v8")]
            if mode == "bad-stage-cost" {
                put(&mut done, "qsa_stage_v8_request_calls", int(999));
            }
        }
        emit(&done);
    }
    0
}

/// `WorkerFiles`: the V8 fixture relinked as `branch8-target` with history.
pub fn worker_files(
    executable: &Path,
    mode: &str,
    stage_present: bool,
    stage_enabled: bool,
) -> WorkerFixture {
    let mut f = WorkerFixture::new(executable, mode);
    let named = f.directory.join("branch8-target");
    if std::fs::hard_link(&f.options.executable, &named).is_err() {
        ok(std::fs::copy(&f.options.executable, &named));
    }
    f.options.executable = named;
    f.options.branch8 = true;
    f.options.checkpoint_slots = 4;
    f.options.checkpoint_budget_bytes = 512 << 20;
    #[cfg(feature = "qsa-stage-v8")]
    {
        f.options.stage8_present = stage_present;
        f.options.stage8_enabled = stage_enabled;
    }
    #[cfg(not(feature = "qsa-stage-v8"))]
    let _ = (stage_present, stage_enabled);
    f
}

/// The C++ `catch (const WorkerCancelled&) { cancelled = true; }`: any other error escapes and fails the run.
fn cancelled<T>(result: Result<T, WorkerError>) -> bool {
    match result {
        Err(WorkerError::Cancelled(_)) => true,
        Err(error) => panic!("{error}"),
        Ok(_) => false,
    }
}

pub fn worker_checks(executable: &Path, stage_present: bool, stage_enabled: bool) {
    let ids = [100u32, 101, 102, 103, 104, 105, 106, 107];
    let cold = begin_job(&ids, false);
    let warm = begin_job(&ids, true);
    let input = input_job(8, false, &policy(), 0);
    let capture = parse(r#"{"op":"branch_capture"}"#);
    {
        let f = worker_files(executable, "normal", stage_present, stage_enabled);
        let mut worker = ok(OwnedWorker::new(f.options()));
        let mut callbacks = 0usize;
        refuses(worker.request(
            &input_job(8, true, &policy(), 0),
            soon(),
            Some(&mut never()),
            None,
        ));
        check(f.commands() == 0, FIXTURE);
        ok(worker.request(&cold, soon(), Some(&mut never()), None));
        ok(worker.request(&input, soon(), Some(&mut never()), None));
        ok(worker.request(&capture, soon(), Some(&mut never()), None));
        ok(worker.request(
            &generation_job(8),
            soon(),
            Some(&mut never()),
            Some(&mut |_: &Json| {
                callbacks += 1;
                Ok(())
            }),
        ));
        check(callbacks == 6, FIXTURE);
        let restored = ok(worker.request(&warm, soon(), Some(&mut never()), None));
        check(
            integer(&restored, "committed_prefix") == 8
                && text(&restored, "outcome") == b"restored",
            FIXTURE,
        );
        ok(worker.request(&generation_job(8), soon(), Some(&mut never()), None));
        ok(worker.close());
        check(!worker.alive(), FIXTURE);
    }
    {
        let f = worker_files(
            executable,
            "wrong-control-lease",
            stage_present,
            stage_enabled,
        );
        let mut worker = ok(OwnedWorker::new(f.options()));
        let mut callbacks = 0usize;
        refuses(worker.request(
            &cold,
            soon(),
            Some(&mut never()),
            Some(&mut |_: &Json| {
                callbacks += 1;
                Ok(())
            }),
        ));
        check(
            callbacks == 0 && !worker.alive() && f.commands() == 1,
            "worker accepted a different fresh lease than the exact issued lease",
        );
        refuses(worker.request(&cold, soon(), Some(&mut never()), None));
        check(
            f.commands() == 1,
            "wrong control-lease echo triggered a retry",
        );
        ok(worker.close());
    }
    for mode in ["wrong-width", "bad-snapshot", "truncated"] {
        let f = worker_files(executable, mode, stage_present, stage_enabled);
        let mut worker = ok(OwnedWorker::new(f.options()));
        let mut callbacks = 0usize;
        ok(worker.request(&cold, soon(), Some(&mut never()), None));
        ok(worker.request(&input, soon(), Some(&mut never()), None));
        refuses(worker.request(
            &generation_job(8),
            soon(),
            Some(&mut never()),
            Some(&mut |_: &Json| {
                callbacks += 1;
                Ok(())
            }),
        ));
        check(
            callbacks == 0 && !worker.alive() && f.commands() == 3,
            FIXTURE,
        );
        ok(worker.close());
    }
    {
        let f = worker_files(executable, "slow-done", stage_present, stage_enabled);
        let mut worker = ok(OwnedWorker::new(f.options()));
        let mut callbacks = 0usize;
        ok(worker.request(&cold, soon(), Some(&mut never()), None));
        ok(worker.request(&input, soon(), Some(&mut never()), None));
        let outcome = worker.request(
            &generation_job(8),
            soon(),
            Some(&mut || {
                if !f.directory.join("events-sent").exists() {
                    return Ok(false);
                }
                ok(std::fs::write(
                    f.directory.join("cancel-observed"),
                    "cancel acknowledged\n",
                ));
                Ok(true)
            }),
            Some(&mut |_: &Json| {
                callbacks += 1;
                Ok(())
            }),
        );
        check(
            cancelled(outcome) && callbacks == 0 && worker.alive(),
            FIXTURE,
        );
        let commands = f.commands();
        refuses(worker.request(&warm, soon(), Some(&mut never()), None));
        check(f.commands() == commands, FIXTURE);
        let reset = ok(worker.request(&cold, soon(), Some(&mut never()), None));
        check(integer(&reset, "committed_prefix") == 0, FIXTURE);
        ok(worker.request(&input, soon(), Some(&mut never()), None));
        ok(worker.close());
    }
    for mode in ["no-consent", "no-verify", "no-history", "replay"] {
        let mut f = worker_files(executable, "normal", stage_present, stage_enabled);
        match mode {
            "no-consent" => f.options.branch8 = false,
            "no-verify" => f.options.verify8 = false,
            "no-history" => {
                f.options.checkpoint_slots = 0;
                f.options.checkpoint_budget_bytes = 0;
            }
            _ => f.options.stack_execution = "recurrent-replay".to_owned(),
        }
        refuses(OwnedWorker::new(f.options()));
        check(
            !f.directory.join("started").exists() && f.commands() == 0,
            FIXTURE,
        );
    }
}

#[cfg(feature = "qsa-stage-v8")]
mod stage {
    use super::*;

    fn stage_cost(value: &Json, calls: u64) -> Json {
        let mut value = with(value, "qsa_stage_v8_calls", int(calls));
        put(
            &mut value,
            "qsa_stage_v8_physical_dispatches",
            int(2 * calls),
        );
        put(&mut value, "qsa_stage_v8_extra_dispatches", int(calls));
        value
    }

    /// `qsa_stage_v8_startup_allowed` from `native/include/tokspur/graph.hpp` in the HTTP stage build. It is a target-side startup rule (used by `native/src/target.cpp`), outside the Rust port, transliterated here so `stage_startup_checks` keeps its 16 check points. Those checks exercise this transliteration, not ported code.
    #[allow(clippy::too_many_arguments)]
    fn qsa_stage_v8_startup_allowed(
        execution: u32,
        profiling: bool,
        splits: u32,
        blocks: u32,
        old_hc: u32,
        capacity: u32,
        kernels: bool,
        scheduling: bool,
        prefill: bool,
        lookup_selectors: bool,
        broker: bool,
        scalar_wide: bool,
        many_wide: bool,
        bounded_branch_broker: bool,
    ) -> bool {
        let transport = !broker || (bounded_branch_broker && capacity == 8);
        execution == 0
            && !profiling
            && splits == 16
            && blocks == 3
            && old_hc == 0
            && (capacity == 0 || capacity == 8)
            && !kernels
            && !scheduling
            && !prefill
            && !lookup_selectors
            && transport
            && !scalar_wide
            && !many_wide
    }

    fn stage_startup_checks() {
        check(
            qsa_stage_v8_startup_allowed(
                0, false, 16, 3, 0, 8, false, false, false, false, false, false, false, false,
            ),
            FIXTURE,
        );
        check(
            !qsa_stage_v8_startup_allowed(
                0, false, 16, 3, 0, 8, false, false, false, false, true, false, false, false,
            ),
            FIXTURE,
        );
        check(
            qsa_stage_v8_startup_allowed(
                0, false, 16, 3, 0, 8, false, false, false, false, true, false, false, true,
            ),
            FIXTURE,
        );
        for unsafe_case in 0..13u32 {
            check(
                !qsa_stage_v8_startup_allowed(
                    u32::from(unsafe_case == 0),
                    unsafe_case == 1,
                    if unsafe_case == 2 { 8 } else { 16 },
                    if unsafe_case == 3 { 1 } else { 3 },
                    if unsafe_case == 4 { 8 } else { 0 },
                    if unsafe_case == 5 { 0 } else { 8 },
                    unsafe_case == 6,
                    unsafe_case == 7,
                    unsafe_case == 8,
                    unsafe_case == 9,
                    true,
                    unsafe_case == 10,
                    unsafe_case == 11,
                    unsafe_case != 12,
                ),
                "staged-QSA broker startup admitted an unsafe fixed-policy combination",
            );
        }
    }

    fn stage_admission_checks(ready: &Json) {
        let enabled = boolean(ready, "qsa_stage_v8_enabled");
        ok(validate_branch8_ready(ready));
        count();
        for name in STAGE {
            refuses(validate_branch8_ready(&without(ready, name)));
            refuses(validate_branch8_ready(&with(
                ready,
                "branch8_session",
                without(field(ready, "branch8_session"), name),
            )));
        }
        for (name, value) in [
            ("qsa_stage_v8_prepared", Json::Bool(false)),
            ("qsa_stage_v8_scratch_logical_bytes", int(0)),
            ("qsa_stage_v8_scratch_charged_bytes", int(0)),
            ("qsa_stage_v8_scratch_charged_bytes", int(212_991)),
            ("qsa_stage_v8_calls", int(1)),
            ("qsa_stage_v8_physical_dispatches", int(1)),
            ("qsa_stage_v8_extra_dispatches", int(1)),
            ("qsa_stage_v8_unknown", int(0)),
            ("allow_prompt_lookup_experiments", Json::Bool(true)),
        ] {
            refuses(validate_branch8_ready(&with(ready, name, value)));
        }
        let mut fake = Fake::new(ready.clone());
        let ids = [100u32, 101, 102, 103, 104, 105, 106, 107];
        let cold_job = begin_job(&ids, false);
        let initial = fake.previous.clone();
        let cold = fake.apply(&cold_job, |_| check(false, FIXTURE));
        ok(validate_branch8_control(
            ready, &initial, &cold_job, &cold, false,
        ));
        count();
        let input = input_job(8, false, &policy(), 0);
        let before_input = fake.previous.clone();
        let filled = fake.apply(&input, |_| check(false, FIXTURE));
        // Keep the sampling Json alive for admission's non-owning reference.
        let greedy = policy();
        let mut input_reply = ok(branch_reply(ready, &before_input, &input, 0, &greedy));
        ok(input_reply.finish(&filled));
        check(ok(input_reply.events()).is_empty(), FIXTURE);
        check(
            integer(&filled, "qsa_stage_v8_request_calls") == if enabled { 96 } else { 0 },
            FIXTURE,
        );
        let mut corrupt_input = ok(branch_reply(ready, &before_input, &input, 0, &greedy));
        refuses(corrupt_input.finish(&with(&filled, "qsa_stage_v8_request_calls", int(999))));
        let capture_job = parse(r#"{"op":"branch_capture"}"#);
        let before_capture = fake.previous.clone();
        let captured = fake.apply(&capture_job, |_| check(false, FIXTURE));
        ok(validate_branch8_control(
            ready,
            &before_capture,
            &capture_job,
            &captured,
            false,
        ));
        count();
        if enabled {
            let session = field(&captured, "branch8_session");
            let bad_state = stage_cost(session, integer(session, "qsa_stage_v8_calls") + 12);
            refuses(validate_branch8_control(
                ready,
                &before_capture,
                &capture_job,
                &with(&captured, "branch8_session", bad_state),
                false,
            ));
        }
        let previous = fake.previous.clone();
        let job = generation_job(7);
        let rounds = [
            Round::of(7, 2, 1),
            Round::of(0, 0, 1),
            Round::of(0, 0, 1),
            Round::of(0, 0, 1),
            Round::of(0, 0, 1),
        ];
        let previous_state = field(&previous, "branch8_session");
        let mut transcript =
            make_transcript(ready, previous_state, job.clone(), &greedy, 0, &rounds);
        // The generic V8 fixture omits these branch-only lifetime fields. Keep the complete native envelope; do not weaken branch reply admission.
        for name in [
            "lookup_candidates_examined",
            "lookup_token_comparisons",
            "lookup_work_limited_queries",
        ] {
            put(
                &mut transcript.done,
                name,
                field(previous_state, name).clone(),
            );
        }
        put(&mut transcript.done, "logits_available", Json::Bool(true));
        put(
            &mut transcript.done,
            "checkpoint_capture_wall_ms",
            number(0.0),
        );
        put(
            &mut transcript.done,
            "checkpoint_restore_wall_ms",
            number(0.0),
        );
        let mut generated = array(&transcript.done, "generated_ids").clone();
        let first_id = integer(&array(previous_state, "final_logit_top8")[0], "id");
        generated[0] = int(first_id);
        put(
            &mut transcript.done,
            "generated_ids",
            Json::Array(generated),
        );
        let command = integer(&previous, "command_id") + 1;
        let lease = integer(&previous, "lease_id");
        let mut first = true;
        for event in &mut transcript.events {
            if first && text(event, "event") == b"token" {
                put(event, "id", int(first_id));
                first = false;
            }
            *event = stamp(event.clone(), command, lease, &Json::Null);
        }
        let state = snapshot(&transcript.done, previous_state);
        transcript.done = stamp(transcript.done.clone(), command, lease, &state);
        let verify = |events: &[Json], done: &Json| -> Result<(), Verify8Error> {
            let mut reply = branch_reply(ready, &previous, &job, 8, &greedy)?;
            for event in events {
                reply.event(event)?;
                refuses(reply.events());
            }
            reply.finish(done)?;
            check(reply.events()?.len() == events.len(), FIXTURE);
            Ok(())
        };
        ok(verify(&transcript.events, &transcript.done));
        check(
            integer(&transcript.done, "qsa_stage_v8_request_calls")
                == if enabled { 144 } else { 0 },
            "staged cost counted accepted rows rather than evaluated rows",
        );
        let mut broken_events = transcript.events.clone();
        let calls = integer(&broken_events[0], "qsa_stage_v8_round_calls") + 1;
        put(
            &mut broken_events[0],
            "qsa_stage_v8_round_calls",
            int(calls),
        );
        refuses(verify(&broken_events, &transcript.done));
        let mut broken_events = transcript.events.clone();
        remove(&mut broken_events[0], "qsa_stage_v8_round_calls");
        refuses(verify(&broken_events, &transcript.done));
        let mut broken_events = transcript.events.clone();
        let old_dispatches = integer(&broken_events[0], "dispatches");
        let dispatches = if enabled {
            old_dispatches - 96
        } else {
            old_dispatches + 12
        };
        put(&mut broken_events[0], "dispatches", int(dispatches));
        refuses(verify(&broken_events, &transcript.done));
        let now = integer(&transcript.done, "qsa_stage_v8_calls");
        let mut bad_done = stage_cost(&transcript.done, now + 1);
        put(
            &mut bad_done,
            "branch8_session",
            stage_cost(field(&transcript.done, "branch8_session"), now + 1),
        );
        refuses(verify(&transcript.events, &bad_done));
        refuses(verify(
            &transcript.events,
            &without(&transcript.done, "qsa_stage_v8_request_calls"),
        ));
        let mut bad_done = with(
            &transcript.done,
            "qsa_stage_v8_enabled",
            Json::Bool(!enabled),
        );
        put(
            &mut bad_done,
            "branch8_session",
            with(
                field(&transcript.done, "branch8_session"),
                "qsa_stage_v8_enabled",
                Json::Bool(!enabled),
            ),
        );
        refuses(verify(&transcript.events, &bad_done));
        let generated_done = fake.apply(&generation_job(8), |_| {});
        let restore_job = begin_job(&ids, true);
        let restored = fake.apply(&restore_job, |_| check(false, FIXTURE));
        ok(validate_branch8_control(
            ready,
            &generated_done,
            &restore_job,
            &restored,
            false,
        ));
        count();
        check(
            integer(field(&restored, "branch8_session"), "qsa_stage_v8_calls")
                == integer(&generated_done, "qsa_stage_v8_calls"),
            "model restore rewound lifetime staged execution work",
        );
        if enabled {
            let stale_cost = integer(field(&captured, "branch8_session"), "qsa_stage_v8_calls");
            check(
                stale_cost < integer(&generated_done, "qsa_stage_v8_calls"),
                "restore fixture did not perform intervening stage work",
            );
            let rewound = with(
                &restored,
                "branch8_session",
                stage_cost(field(&restored, "branch8_session"), stale_cost),
            );
            refuses(validate_branch8_control(
                ready,
                &generated_done,
                &restore_job,
                &rewound,
                false,
            ));
            let cold_before = fake.previous.clone();
            let cold_again = fake.apply(&cold_job, |_| check(false, FIXTURE));
            ok(validate_branch8_control(
                ready,
                &cold_before,
                &cold_job,
                &cold_again,
                false,
            ));
            count();
            let again = field(&cold_again, "branch8_session");
            let advanced = with(
                &cold_again,
                "branch8_session",
                stage_cost(again, integer(again, "qsa_stage_v8_calls") + 12),
            );
            // Exact worker predecessor.
            refuses(validate_branch8_control(
                ready,
                &cold_before,
                &cold_job,
                &advanced,
                false,
            ));
            // A stale HTTP observer may have missed work.
            ok(validate_branch8_control(
                ready,
                &cold_before,
                &cold_job,
                &advanced,
                true,
            ));
            count();
            let fractional = with(
                &cold_again,
                "branch8_session",
                stage_cost(again, integer(again, "qsa_stage_v8_calls") + 1),
            );
            // Whole model rows execute all 12 QSA layers.
            refuses(validate_branch8_control(
                ready,
                &cold_before,
                &cold_job,
                &fractional,
                true,
            ));
        }
    }

    fn stage_worker_checks(executable: &Path, enabled: bool) {
        let ids = [1u32, 2, 3, 4, 5, 6, 7, 8];
        for mode in ["wrong-stage-arm", "unprepared-stage"] {
            let f = worker_files(executable, mode, true, enabled);
            refuses(OwnedWorker::new(f.options()));
            check(
                f.directory.join("started").exists() && f.commands() == 0,
                "stage readiness mismatch was not tested at the actual child boundary",
            );
        }
        {
            let f = worker_files(executable, "bad-stage-cost", true, enabled);
            let mut worker = ok(OwnedWorker::new(f.options()));
            let mut callbacks = 0usize;
            ok(worker.request(&begin_job(&ids, false), soon(), Some(&mut never()), None));
            ok(worker.request(
                &input_job(8, false, &policy(), 0),
                soon(),
                Some(&mut never()),
                None,
            ));
            refuses(worker.request(
                &generation_job(8),
                soon(),
                Some(&mut never()),
                Some(&mut |_: &Json| {
                    callbacks += 1;
                    Ok(())
                }),
            ));
            check(
                callbacks == 0 && !worker.alive() && f.commands() == 3,
                FIXTURE,
            );
            ok(worker.close());
        }
        {
            let f = worker_files(executable, "normal", false, true);
            refuses(OwnedWorker::new(f.options()));
            check(
                !f.directory.join("started").exists(),
                "enabled without selector presence launched a child",
            );
        }
    }

    pub fn stage_checks(executable: &Path) {
        stage_startup_checks();
        section("stage_startup_checks");
        let off = staged_prepared(false);
        let on = staged_prepared(true);
        let absent = prepared();
        check(
            !present(&absent, "qsa_stage_v8_prepared")
                && boolean(&absent, "allow_prompt_lookup_experiments"),
            FIXTURE,
        );
        for name in [
            "qsa_stage_v8_scratch_logical_bytes",
            "qsa_stage_v8_scratch_charged_bytes",
            "tracked_metal_buffer_bytes",
            "peak_metal_buffer_bytes",
            "gpu_allocation_count",
        ] {
            check(
                integer(&off, name) == integer(&on, name),
                "prepared OFF/ON ledgers differ",
            );
        }
        section("stage ledgers");
        for enabled in [false, true] {
            let arm = if enabled { "on" } else { "off" };
            let ready = staged_prepared(enabled);
            admission_checks(&ready);
            section(&format!("stage {arm} admission_checks"));
            stage_admission_checks(&ready);
            section(&format!("stage {arm} stage_admission_checks"));
            session_checks(ready.clone());
            section(&format!("stage {arm} session_checks"));
            cancelled_rewind_resync_checks(ready.clone());
            section(&format!("stage {arm} cancelled_rewind_resync_checks"));
            worker_checks(executable, true, enabled);
            section(&format!("stage {arm} worker_checks"));
            stage_worker_checks(executable, enabled);
            section(&format!("stage {arm} stage_worker_checks"));
        }
    }
}

/// The C++ `main` (`inherited_branch8_checks` once included).
pub fn main(args: &[String]) -> i32 {
    if args.len() > 1 && args[1] == "target" {
        let branch = Path::new(&args[0])
            .file_name()
            .is_some_and(|name| name == "branch8-target");
        return if branch {
            target(args)
        } else {
            fake_target(args)
        };
    }
    let prior = verify8::main(args);
    if prior != 0 {
        return prior;
    }
    admission_checks(&prepared());
    section("admission_checks");
    session_checks(prepared());
    section("branch8 session_checks");
    cancelled_rewind_resync_checks(prepared());
    section("cancelled_rewind_resync_checks");
    worker_checks(&fake_executable(args), false, false);
    section("branch8 worker_checks");
    #[cfg(feature = "qsa-stage-v8")]
    stage::stage_checks(&fake_executable(args));
    println!("Private HTTP branch8 CPU checks={}", checks());
    0
}
