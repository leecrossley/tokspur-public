//! Port of `native/tests/test_http_verify8.cpp`, check for check.
//!
//! Independently generated protocol records only. No model, tokenizer, Metal, external engine or actual inference asset is used by this fixture. The branch8 and fast-math suites include this module, as the C++ suites `#include` this translation unit, and share its check counter.
//!
//! As in C++, the test binary is also the fake target: the worker launches a hard link to it named `verify8-target` and `main` dispatches on `argv[1]`.

use std::cell::{Cell, RefCell};
use std::fmt::Display;
use std::io::{BufRead, Write};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokspur_json::{Json, parse_json};
use tokspur_server::session::{
    ServeCodec, ServeError, ServeSession, ServeSink, SessionOptions, TokenCallback,
};
use tokspur_text::chat::RenderedChat;
use tokspur_text::response_protocol::ResponseAdmission;
use tokspur_wire::verify8::{Verify8Context, Verify8Error, Verify8Reply, validate_verify8_ready};
use tokspur_wire::worker::{CallbackFailed, OwnedWorker, WorkerError, WorkerOptions};

/// The C++ global `checks`, shared by every included suite.
static CHECKS: AtomicUsize = AtomicUsize::new(0);

/// The default `check` message.
pub const FIXTURE: &str = "private HTTP V8 fixture failed";

/// `++checks`.
pub fn count() {
    CHECKS.fetch_add(1, Ordering::Relaxed);
}

pub fn checks() -> usize {
    CHECKS.load(Ordering::Relaxed)
}

/// `TOKSPUR_HTTP_SECTIONS=1` prints the running total after each top-level function.
pub fn section(name: &str) {
    if std::env::var_os("TOKSPUR_HTTP_SECTIONS").is_some() {
        println!("{name} {}", checks());
    }
}

pub fn check(ok: bool, why: &str) {
    count();
    assert!(ok, "{why}");
}

/// An exception escaping a C++ action: print `what()` and fail the run.
pub fn ok<T, E: Display>(result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{error}"),
    }
}

/// Whether the C++ `refuses` would catch this error: it catches `FormatError` and `WorkerError` only. `WorkerCancelled` and fixture failures escape and fail the run.
pub trait Refusal: Display {
    fn refusal(&self) -> bool;
}

impl Refusal for Verify8Error {
    fn refusal(&self) -> bool {
        true // Every V8/branch8 admission failure is a `FormatError`.
    }
}

impl Refusal for WorkerError {
    fn refusal(&self) -> bool {
        !self.is_cancelled()
    }
}

impl Refusal for ServeError {
    fn refusal(&self) -> bool {
        match self {
            Self::Admission(_) | Self::Text(_) | Self::Verify8(_) => true,
            Self::Worker(error) => error.refusal(),
            Self::Http(_) => false,
        }
    }
}

pub fn refuses<T, E: Refusal>(result: Result<T, E>) {
    let refused = match result {
        Ok(_) => false,
        Err(error) => {
            assert!(error.refusal(), "{error}");
            true
        }
    };
    check(refused, "unsafe private V8 operation accepted");
}

pub fn parse(text: &str) -> Json {
    ok(parse_json(text.as_bytes()))
}

pub fn int(value: u64) -> Json {
    Json::integer(value)
}

/// `with` on an owned value: replace the first `key` or append it.
pub fn put(value: &mut Json, key: impl AsRef<[u8]>, replacement: Json) {
    let key = key.as_ref();
    let Json::Object(fields) = value else {
        panic!("expected JSON object");
    };
    match fields.iter_mut().find(|(name, _)| name.as_slice() == key) {
        Some((_, item)) => *item = replacement,
        None => fields.push((key.to_vec(), replacement)),
    }
}

pub fn with(value: &Json, key: impl AsRef<[u8]>, replacement: Json) -> Json {
    let mut result = value.clone();
    put(&mut result, key, replacement);
    result
}

/// `without` on an owned value: drop every field named `key`.
pub fn remove(value: &mut Json, key: impl AsRef<[u8]>) {
    let key = key.as_ref();
    let Json::Object(fields) = value else {
        panic!("expected JSON object");
    };
    fields.retain(|(name, _)| name.as_slice() != key);
}

pub fn without(value: &Json, key: impl AsRef<[u8]>) -> Json {
    let mut result = value.clone();
    remove(&mut result, key);
    result
}

/// `*value.find(key)`.
pub fn field(value: &Json, key: impl AsRef<[u8]>) -> &Json {
    match ok(value.find(key)) {
        Some(item) => item,
        None => panic!("fixture field absent"),
    }
}

pub fn present(value: &Json, key: impl AsRef<[u8]>) -> bool {
    ok(value.find(key)).is_some()
}

pub fn integer(value: &Json, key: impl AsRef<[u8]>) -> u64 {
    ok(field(value, key).unsigned_integer())
}

pub fn boolean(value: &Json, key: impl AsRef<[u8]>) -> bool {
    ok(field(value, key).boolean())
}

pub fn text(value: &Json, key: impl AsRef<[u8]>) -> &[u8] {
    ok(field(value, key).string())
}

pub fn array(value: &Json, key: impl AsRef<[u8]>) -> &Vec<Json> {
    ok(field(value, key).array())
}

pub fn dump(value: &Json) -> Vec<u8> {
    ok(value.dump())
}

/// `std::ostream << std::setprecision(17) << value`: C `%.17g`.
fn precision17(value: f64) -> String {
    let scientific = format!("{value:.16e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("Rust exponent form");
    let exponent: i32 = exponent.parse().expect("Rust exponent digits");
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-4..17).contains(&exponent) {
        let fixed = if exponent >= 0 {
            let point = exponent as usize + 1;
            format!("{}.{}", &digits[..point], &digits[point..])
        } else {
            format!("0.{}{digits}", "0".repeat((-exponent - 1) as usize))
        };
        let fixed = fixed.trim_end_matches('0').trim_end_matches('.');
        format!("{sign}{fixed}")
    } else {
        let (first, rest) = digits.split_at(1);
        let rest = rest.trim_end_matches('0');
        let point = if rest.is_empty() { "" } else { "." };
        let marker = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{first}{point}{rest}e{marker}{:02}", exponent.abs())
    }
}

/// `number`: a JSON number with the lexeme a 17-digit `ostream` prints.
pub fn number(value: f64) -> Json {
    parse(&precision17(value))
}

pub fn policy() -> Json {
    parse(r#"{"temperature":0,"top_k":0,"top_p":1,"min_p":0}"#)
}

pub fn hc(counts: [u64; 5]) -> Json {
    Json::new_object([
        ("scalar", int(counts[0])),
        ("many1", int(counts[1])),
        ("many2", int(counts[2])),
        ("many4", int(counts[3])),
        ("many8", int(counts[4])),
    ])
}

pub fn ready_record() -> Json {
    parse(
        r#"{
      "event":"ready","small_gemv_column_splits":16,"block_gemv_formats":3,"stack_execution":"direct",
      "maximum_generated_tokens":4096,"known_token_capacity":8,"known_residual_reuse":true,
      "prompt_lookup":true,"verification_max_tokens":8,"verification_head_many":true,
      "hc_q8_splits":0,"hc_q8_numerical_tier":"ordinary","hc_q8_scratch_logical_bytes":0,"hc_q8_scratch_charged_bytes":0,
      "replay_prepared":false,"recurrent_replay_captures":0,"mapped_weight_region_limit_bytes":0,
      "capture_resource_bytes":0,"capture_binding_aliases":0,"capture_binding_alias_span_bytes":0,
      "capture_binding_alias_resource_bytes":0,"live_captures":0,"profile_counter_payload_bytes":0,
      "checkpoint_slots":0,"checkpoint_combined_budget_bytes":0,"checkpoint_host_bytes":0,
      "checkpoint_store_host_bytes":0,"checkpoint_gpu_bytes":0,"checkpoint_broker":false,"broker_epoch":0,
      "native_limit_bytes":103079215104,"weight_copy":false,"gpu_profile_mode":"disabled","timings_qualified":false,
      "allow_prompt_lookup_experiments":true,"allow_request_kernel_experiments":false,
      "allow_request_stack_experiments":false,"allow_request_prefill_experiments":false,
      "lookup_match_tokens":16,"lookup_history_tokens":65536,"lookup_proposal_tokens":7,
      "lookup_verification_budget_bytes":134217728,"lookup_caller_host_bytes":2709960,
      "verification_host_bytes":9293008,"verification_gpu_bytes":50076576,
      "lookup_verification_charged_bytes":62079544,"verification_tape_payload_bytes":33140736,
      "tracked_metal_buffer_bytes":83805224124,"peak_metal_buffer_bytes":83805224124,
      "gpu_buffer_limit_bytes":100394860544,"imported_weight_backing_bytes":81848123392,"gpu_allocation_count":612,
      "hc_exact_mode":"combined","hc_exact_rows_per_group":1,"hc_exact_unroll":16,
      "hc_exact_scratch_logical_bytes":0,"hc_exact_scratch_charged_bytes":0,
      "hc_exact_routes":{"scalar":0,"many1":0,"many2":0,"many4":0,"many8":0},
      "wide_q8_exact_enabled":false,"wide_q8_exact_rows_per_group":4,"wide_q8_exact_unroll":16,
      "wide_q8_exact_scratch_logical_bytes":0,"wide_q8_exact_scratch_charged_bytes":0,
      "wide_q8_exact_routes":{"qkv_scalar":0,"qkv_many1":0,"output_scalar":0,"output_many1":0},
      "wide_q8_many_enabled":false,"wide_q8_many_rows_per_group":4,"wide_q8_many_unroll":4,
      "wide_q8_many_scratch_logical_bytes":0,"wide_q8_many_scratch_charged_bytes":0,
      "wide_q8_many_routes":{"qkv_many2":0,"qkv_many4":0,"qkv_many8":0,"output_many2":0,"output_many4":0,"output_many8":0}
    }"#,
    )
}

pub fn input_job(count: u32, reset: bool, sampling: &Json, seed: u64) -> Json {
    let ids = (0..count).map(|i| int(100 + u64::from(i))).collect();
    let mut result = Json::new_object([
        ("tokens", Json::Array(ids)),
        ("generate", int(0)),
        ("reset", Json::Bool(reset)),
    ]);
    if reset {
        put(&mut result, "sampling", sampling.clone());
        put(&mut result, "seed", int(seed));
        put(&mut result, "prompt_lookup", Json::Bool(true));
    }
    result
}

pub fn generation_job(count: u32) -> Json {
    Json::new_object([
        ("tokens", Json::Array(Vec::new())),
        ("generate", int(u64::from(count))),
        ("reset", Json::Bool(false)),
        ("stop_ids", Json::Array(vec![int(248_046)])),
    ])
}

/// `Round{n, k, ordinary, terminal = false, proposed = 0}`.
#[derive(Clone, Copy)]
pub struct Round {
    pub n: u32,
    pub k: u32,
    pub ordinary: u32,
    pub terminal: bool,
    pub proposed: u32,
}

impl Round {
    pub const fn of(n: u32, k: u32, ordinary: u32) -> Self {
        Self {
            n,
            k,
            ordinary,
            terminal: false,
            proposed: 0,
        }
    }

    /// `Round{n, k, ordinary, true}`.
    pub const fn stop(n: u32, k: u32, ordinary: u32) -> Self {
        Self {
            terminal: true,
            ..Self::of(n, k, ordinary)
        }
    }
}

#[derive(Clone)]
pub struct Transcript {
    pub previous: Json,
    pub job: Json,
    pub sampling: Json,
    pub done: Json,
    pub before: u32,
    pub seed: u64,
    pub events: Vec<Json>,
}

pub fn grams(prefix: u32) -> u32 {
    if prefix < 16 { 0 } else { prefix - 15 }
}

fn serial_dispatches(prefix: u32) -> u64 {
    // Independent, fixed-prefix threshold table rather than the runtime helper. Thresholds are the first token of each extra merge level, not powers of 2.
    let tokens = prefix + 1;
    let mut extra: u64 = if tokens >= 2052 { 48 } else { 0 };
    for threshold in [4100u32, 8196, 16388, 32772] {
        if tokens >= threshold {
            extra += 12;
        }
    }
    2222 + (if tokens.is_multiple_of(4) { 36 } else { 0 }) + extra
}

/// `routes[i] = 72` for each of the 1/2/4/8 chunks a width decomposes into.
fn chunk_routes(width: u32, routes: &mut [u64; 5]) {
    for (i, chunk) in [1u32, 2, 4, 8].into_iter().enumerate() {
        if width & chunk != 0 {
            routes[i + 1] = 72;
        }
    }
}

const HC_NAMES: [&str; 5] = ["scalar", "many1", "many2", "many4", "many8"];

pub fn make_transcript(
    ready: &Json,
    previous: &Json,
    job: Json,
    sampling: &Json,
    seed: u64,
    rounds: &[Round],
) -> Transcript {
    let before = if previous.is_null() {
        0
    } else {
        integer(previous, "committed_prefix") as u32
    };
    let mut result = Transcript {
        previous: previous.clone(),
        job,
        sampling: sampling.clone(),
        done: Json::Null,
        before,
        seed,
        events: Vec::new(),
    };
    let reset = boolean(&result.job, "reset");
    let inputs = array(&result.job, "tokens").len() as u32;
    let base = if reset { 0 } else { result.before };
    let mut position = base + inputs;
    let (mut total_ordinary, mut total_proposed, mut verified, mut recovery) =
        (0u32, 0u32, 0u32, 0u32);
    let mut routes = [0u64; 5];
    let mut output: Vec<Json> = Vec::new();
    let (mut wall_sum, mut batch_wall_sum, mut batch_gpu_sum) = (0.0f64, 0.0f64, 0.0f64);
    let mut stopped = false;
    #[cfg(feature = "qsa-stage-v8")]
    let stage_prepared = present(ready, "qsa_stage_v8_prepared");
    #[cfg(feature = "qsa-stage-v8")]
    let stage_enabled = stage_prepared && boolean(ready, "qsa_stage_v8_enabled");
    #[cfg(feature = "qsa-stage-v8")]
    let mut stage_request_calls: u64 = if stage_enabled {
        12 * u64::from(inputs)
    } else {
        0
    };
    // Staged rows add 12 physical dispatches per evaluated model row.
    #[cfg(feature = "qsa-stage-v8")]
    let stage_rows = |rows: u32| {
        if stage_enabled {
            12 * u64::from(rows)
        } else {
            0
        }
    };
    #[cfg(not(feature = "qsa-stage-v8"))]
    let stage_rows = |_: u32| 0u64;
    if inputs != 0 {
        // All N inputs traverse 24 ordinary HC projections. Only the final retained row executes the scalar head; bit chunks cover 72 others.
        routes[0] = u64::from(24 * inputs + 1);
        chunk_routes(inputs, &mut routes);
    }
    for (r, spec) in rounds.iter().enumerate() {
        let count = spec.k + spec.ordinary;
        let proposed = spec.n.max(spec.proposed);
        let partial = spec.k != 0 && spec.k < spec.n;
        let chunks = spec.n.count_ones();
        let mut serial: u64 = 0;
        for t in 0..spec.n {
            serial += serial_dispatches(position + t);
        }
        let forward = if spec.n != 0 {
            serial - u64::from(72 * (spec.n - 1)) - u64::from(253 * (spec.n - chunks))
        } else {
            0
        };
        let dispatches = forward
            + (if partial { 72 } else { 0 })
            + (if spec.ordinary != 0 {
                serial_dispatches(position + spec.k)
            } else {
                0
            })
            + stage_rows(spec.n + spec.ordinary);
        let mut round_routes = [0u64; 5];
        round_routes[0] = u64::from(25 * spec.n + 97 * spec.ordinary);
        chunk_routes(spec.n, &mut round_routes);
        for (total, add) in routes.iter_mut().zip(round_routes) {
            *total += add;
        }
        let gather = if spec.n != 0 { 2.0 } else { 0.0 };
        let verify = if spec.n != 0 { 10.0 } else { 0.0 };
        let fallback = if spec.ordinary != 0 { 4.0 } else { 0.0 };
        let wall = 1.0 + gather + verify + fallback;
        let batch_wall = f64::from(
            (if spec.n != 0 { 8 } else { 0 })
                + (if partial { 2 } else { 0 })
                + (if spec.ordinary != 0 { 3 } else { 0 }),
        );
        let batch_gpu = f64::from(
            (if spec.n != 0 { 7 } else { 0 })
                + (if partial { 1 } else { 0 })
                + (if spec.ordinary != 0 { 2 } else { 0 }),
        );
        let round_id = r as u64 + 1;
        let end = u64::from(position + count);
        #[cfg_attr(not(feature = "qsa-stage-v8"), allow(unused_mut))]
        let mut round = Json::new_object([
            ("event", Json::from("generation_round")),
            ("round_id", int(round_id)),
            ("round_base", int(u64::from(position))),
            ("committed_prefix", int(end)),
            ("proposed_tokens", int(u64::from(proposed))),
            ("verified_tokens", int(u64::from(spec.n))),
            ("verification_max_tokens", int(8)),
            ("verification_head_dispatches", int(u64::from(chunks))),
            ("accepted_verified_tokens", int(u64::from(spec.k))),
            ("ordinary_tokens", int(u64::from(spec.ordinary))),
            ("useful_tokens", int(u64::from(count))),
            ("round_wall_ms", number(wall)),
            ("lookup_and_publication_ms", number(1.0)),
            ("gather_wall_ms", number(gather)),
            ("verification_including_recovery_wall_ms", number(verify)),
            (
                "recovery_batch_wall_ms",
                number(if partial { 2.0 } else { 0.0 }),
            ),
            ("fallback_wall_ms", number(fallback)),
            ("batch_wall_ms", number(batch_wall)),
            ("batch_gpu_ms", number(batch_gpu)),
            ("hc_exact_round_routes", hc(round_routes)),
            (
                "wide_q8_exact_round_routes",
                field(ready, "wide_q8_exact_routes").clone(),
            ),
            (
                "wide_q8_many_round_routes",
                field(ready, "wide_q8_many_routes").clone(),
            ),
            ("dispatches", int(dispatches)),
            ("capture_executes", int(0)),
            (
                "useful_tokens_per_second",
                number(f64::from(count) * 1000.0 / wall),
            ),
            (
                "accepted_tokens_per_second",
                number(f64::from(spec.k) * 1000.0 / wall),
            ),
            ("timings_qualified", Json::Bool(false)),
            ("gpu_profile", Json::Null),
        ]);
        #[cfg(feature = "qsa-stage-v8")]
        if stage_prepared {
            let calls = stage_rows(spec.n + spec.ordinary);
            stage_request_calls += calls;
            put(&mut round, "qsa_stage_v8_round_calls", int(calls));
        }
        result.events.push(round);
        for t in 0..count {
            let id = if spec.terminal && t + 1 == count {
                248_046
            } else {
                200 + output.len() as u64
            };
            output.push(int(id));
            result.events.push(Json::new_object([
                ("event", Json::from("token")),
                ("id", int(id)),
                ("source_logit", number(1.25 + f64::from(t))),
                ("committed_prefix", int(end)),
                ("token_prefix", int(u64::from(position + t + 1))),
                ("round_id", int(round_id)),
                ("round_base", int(u64::from(position))),
                ("round_committed_prefix", int(end)),
                ("verified", Json::Bool(t < spec.k)),
                ("batched_commit", Json::Bool(spec.k >= 2)),
                ("entropy_words", int(0)),
                ("gpu_profile", Json::Null),
            ]));
        }
        position += count;
        total_ordinary += spec.ordinary;
        total_proposed += proposed;
        verified += u32::from(spec.n != 0);
        recovery += u32::from(partial);
        stopped = spec.terminal;
        wall_sum += wall;
        batch_wall_sum += batch_wall;
        batch_gpu_sum += batch_gpu;
    }
    // Ready-only capability/charge fields are deliberately absent from done, matching the actual target schema rather than inventing stronger echoes.
    let mut done = ready.clone();
    for key in [
        "maximum_generated_tokens",
        "checkpoint_broker",
        "broker_epoch",
        "weight_copy",
        "gpu_profile_mode",
        "allow_prompt_lookup_experiments",
        "allow_request_kernel_experiments",
        "allow_request_stack_experiments",
        "allow_request_prefill_experiments",
        "lookup_match_tokens",
        "lookup_history_tokens",
        "lookup_proposal_tokens",
        "lookup_verification_budget_bytes",
        "lookup_caller_host_bytes",
        "verification_host_bytes",
        "verification_gpu_bytes",
        "lookup_verification_charged_bytes",
        "verification_tape_payload_bytes",
    ] {
        remove(&mut done, key);
    }
    let generated = output.len() as u64;
    put(&mut done, "event", Json::from("done"));
    put(&mut done, "committed_prefix", int(u64::from(position)));
    put(&mut done, "generated_ids", Json::Array(output));
    put(&mut done, "known_token_prefill", Json::Bool(true));
    let status = if inputs != 0 {
        "prefilled"
    } else if stopped {
        "stopped"
    } else {
        "length"
    };
    put(&mut done, "status", Json::from(status));
    put(
        &mut done,
        "finish_reason",
        if inputs != 0 {
            Json::Null
        } else {
            Json::from(if stopped { "stop" } else { "length" })
        },
    );
    put(&mut done, "continuation_allowed", Json::Bool(!stopped));
    put(
        &mut done,
        "stop_id",
        if stopped { int(248_046) } else { Json::Null },
    );
    let mut sampled = sampling.clone();
    put(&mut sampled, "valid", Json::Bool(true));
    put(&mut sampled, "seed", int(seed));
    put(&mut sampled, "entropy_prefix", int(u64::from(position)));
    put(&mut sampled, "entropy_words", int(0));
    put(&mut done, "sampling", sampled);
    put(
        &mut done,
        "hc_exact_final_logit_sha256",
        Json::from("a".repeat(64)),
    );
    let top = (0..8u32)
        .map(|i| {
            Json::new_object([
                ("id", int(u64::from(i + 200))),
                ("logit", number(1.25 - f64::from(i) * 0.25)),
            ])
        })
        .collect();
    put(&mut done, "final_logit_top8", Json::Array(top));
    put(&mut done, "hc_exact_request_routes", hc(routes));
    let old_routes = if previous.is_null() {
        field(ready, "hc_exact_routes")
    } else {
        field(previous, "hc_exact_routes")
    };
    let mut cumulative = routes;
    for (total, name) in cumulative.iter_mut().zip(HC_NAMES) {
        *total += integer(old_routes, name);
    }
    put(&mut done, "hc_exact_routes", hc(cumulative));
    put(
        &mut done,
        "wide_q8_exact_request_routes",
        field(ready, "wide_q8_exact_routes").clone(),
    );
    put(
        &mut done,
        "wide_q8_many_request_routes",
        field(ready, "wide_q8_many_routes").clone(),
    );
    put(&mut done, "lookup_rounds", int(rounds.len() as u64));
    put(
        &mut done,
        "lookup_verified_rounds",
        int(u64::from(verified)),
    );
    put(
        &mut done,
        "lookup_recovery_rounds",
        int(u64::from(recovery)),
    );
    put(
        &mut done,
        "lookup_proposed_tokens",
        int(u64::from(total_proposed)),
    );
    put(
        &mut done,
        "lookup_accepted_tokens",
        int(generated - u64::from(total_ordinary)),
    );
    put(
        &mut done,
        "lookup_ordinary_tokens",
        int(u64::from(total_ordinary)),
    );
    let epoch = if previous.is_null() {
        1
    } else {
        integer(previous, "lookup_history_epoch")
    };
    put(
        &mut done,
        "lookup_history_epoch",
        int(epoch + u64::from(reset)),
    );
    for (total, delta, amount) in [
        (
            "lookup_appended_tokens",
            "lookup_request_appended_tokens",
            u64::from(inputs) + generated,
        ),
        (
            "lookup_indexed_grams",
            "lookup_request_indexed_grams",
            u64::from(grams(position) - grams(base)),
        ),
        (
            "lookup_queries",
            "lookup_request_queries",
            rounds.len() as u64,
        ),
        ("lookup_resets", "lookup_request_resets", u64::from(reset)),
    ] {
        put(&mut done, delta, int(amount));
        let old = if previous.is_null() {
            0
        } else {
            integer(previous, total)
        };
        put(&mut done, total, int(old + amount));
    }
    put(&mut done, "lookup_truncated_grams", int(0));
    put(&mut done, "lookup_cleared_grams", int(0));
    put(&mut done, "lookup_truncations", int(0));
    put(&mut done, "replayed_input_tokens", int(u64::from(inputs)));
    put(&mut done, "checkpoint_reuse", Json::from("off"));
    put(&mut done, "checkpoint_restored_tokens", int(0));
    put(&mut done, "checkpoint_retained", int(0));
    put(&mut done, "lookup_round_wall_ms", number(wall_sum));
    put(
        &mut done,
        "generation_batch_wall_ms",
        number(batch_wall_sum),
    );
    put(&mut done, "generation_batch_gpu_ms", number(batch_gpu_sum));
    put(
        &mut done,
        "prefill_wall_ms",
        number(if inputs != 0 { 1.0 } else { 0.0 }),
    );
    put(&mut done, "request_wall_ms", number(wall_sum + 2.0));
    put(&mut done, "gpu_profile", Json::Null);
    #[cfg(feature = "qsa-stage-v8")]
    if stage_prepared {
        let old = if previous.is_null() {
            0
        } else {
            integer(previous, "qsa_stage_v8_calls")
        };
        let calls = old + stage_request_calls;
        put(
            &mut done,
            "qsa_stage_v8_request_calls",
            int(stage_request_calls),
        );
        put(&mut done, "qsa_stage_v8_calls", int(calls));
        put(
            &mut done,
            "qsa_stage_v8_physical_dispatches",
            int(2 * calls),
        );
        put(&mut done, "qsa_stage_v8_extra_dispatches", int(calls));
    }
    result.done = done;
    result
}

/// `Verify8Reply(ready, previous, job, before, sampling, seed)`.
pub fn v8_reply<'a>(
    ready: &'a Json,
    previous: &'a Json,
    job: &'a Json,
    before: u32,
    policy: &'a Json,
    seed: u64,
) -> Result<Verify8Reply<'a>, Verify8Error> {
    Verify8Reply::new(Verify8Context {
        ready,
        previous,
        job,
        before,
        policy,
        seed,
        reset_resync: false,
        branch8: false,
    })
}

pub fn admit(ready: &Json, t: &Transcript) {
    let job_pin = dump(&t.job);
    let policy_pin = dump(&t.sampling);
    let previous_pin = dump(&t.previous);
    let mut reply = ok(v8_reply(
        ready,
        &t.previous,
        &t.job,
        t.before,
        &t.sampling,
        t.seed,
    ));
    refuses(reply.events());
    for event in &t.events {
        ok(reply.event(event));
        refuses(reply.events());
    }
    ok(reply.finish(&t.done));
    check(ok(reply.events()).len() == t.events.len(), FIXTURE);
    for (i, event) in t.events.iter().enumerate() {
        check(
            dump(&ok(reply.events())[i]) == dump(event),
            "admission rewrote source records",
        );
    }
    check(
        dump(&t.job) == job_pin
            && dump(&t.sampling) == policy_pin
            && dump(&t.previous) == previous_pin,
        "sampler or input ancestry was rewritten",
    );
    refuses(reply.event(t.events.first().unwrap_or(&Json::Null)));
    refuses(reply.finish(&t.done));
}

pub fn reject(ready: &Json, t: &Transcript) {
    let mut reply = ok(v8_reply(
        ready,
        &t.previous,
        &t.job,
        t.before,
        &t.sampling,
        t.seed,
    ));
    refuses((|| {
        for event in &t.events {
            reply.event(event)?;
        }
        reply.finish(&t.done)
    })());
    refuses(reply.events());
}

/// `seed_prefix(ready)`: an admitted 16-token input ancestry.
pub fn seed_prefix(ready: &Json) -> Json {
    let mut previous = Json::Null;
    let mut prefix = 16u32;
    while prefix != 0 {
        let count = prefix.min(8);
        let job = input_job(count, previous.is_null(), &policy(), 0);
        let t = make_transcript(ready, &previous, job, &policy(), 0, &[]);
        admit(ready, &t);
        previous = t.done;
        prefix -= count;
    }
    previous
}

fn ready_checks() {
    let ready = ready_record();
    ok(validate_verify8_ready(&ready));
    count();
    for (key, _) in ok(ready.object()) {
        refuses(validate_verify8_ready(&without(&ready, key)));
        for invalid in [Json::Null, Json::from("wrong"), Json::Array(Vec::new())] {
            refuses(validate_verify8_ready(&with(&ready, key, invalid)));
        }
    }
    for (key, value) in [
        ("verification_max_tokens", int(2)),
        ("known_token_capacity", int(4)),
        ("known_residual_reuse", Json::Bool(false)),
        ("stack_execution", Json::from("recurrent-replay")),
        ("hc_exact_mode", Json::from("scalar")),
        ("wide_q8_many_enabled", Json::Bool(true)),
        ("gpu_profile_mode", Json::from("intrusive")),
        ("lookup_proposal_tokens", int(8)),
        ("checkpoint_slots", int(1)),
        ("lookup_verification_budget_bytes", int((128 << 20) + 1)),
        ("lookup_verification_charged_bytes", int(62_079_545)),
        ("verification_gpu_bytes", int(128 << 20)),
        ("verification_host_bytes", int(0)),
        ("verification_tape_payload_bytes", int(50_076_577)),
        ("tracked_metal_buffer_bytes", int(83_805_224_125)),
        ("peak_metal_buffer_bytes", int(100_394_860_545)),
        ("imported_weight_backing_bytes", int(83_805_224_125)),
        ("gpu_allocation_count", int(0)),
        ("trace_expert_routes", Json::Bool(false)),
        ("expert_route_trace", Json::Object(Vec::new())),
        ("moe_token_tile_enabled", Json::Bool(false)),
    ] {
        refuses(validate_verify8_ready(&with(&ready, key, value)));
    }
    for n in [
        Json::Bool(true),
        parse("8.0"),
        parse("8e0"),
        parse("-1"),
        parse("18446744073709551616"),
    ] {
        refuses(validate_verify8_ready(&with(
            &ready,
            "verification_max_tokens",
            n,
        )));
    }
    let mut renamed = without(field(&ready, "wide_q8_exact_routes"), "qkv_scalar");
    put(&mut renamed, "unknown_zero", int(0));
    refuses(validate_verify8_ready(&with(
        &ready,
        "wide_q8_exact_routes",
        renamed,
    )));
}

fn positive_checks() {
    let ready = ready_record();
    for n in 1..=8 {
        let t = make_transcript(
            &ready,
            &Json::Null,
            input_job(n, true, &policy(), 0),
            &policy(),
            0,
            &[],
        );
        admit(&ready, &t);
    }
    let previous = seed_prefix(&ready);
    for n in 2..=7u32 {
        for k in 1..=n {
            // Submitted/evaluated work must itself fit the reservation, not merely the smaller committed prefix. Finish unused output with ordinary work.
            let mut sequence = vec![Round::of(n, k, u32::from(k < n))];
            for _ in k + u32::from(k < n)..n {
                sequence.push(Round::of(0, 0, 1));
            }
            let t = make_transcript(
                &ready,
                &previous,
                generation_job(n),
                &policy(),
                0,
                &sequence,
            );
            admit(&ready, &t);
        }
    }
    let full_eight = [Round::of(7, 7, 0), Round::of(0, 0, 1)];
    let eight = make_transcript(
        &ready,
        &previous,
        generation_job(8),
        &policy(),
        0,
        &full_eight,
    );
    admit(&ready, &eight);
    let eight_serial = [Round::of(0, 0, 1); 8];
    let bounded = make_transcript(
        &ready,
        &previous,
        generation_job(8),
        &policy(),
        0,
        &eight_serial,
    );
    check(bounded.events.len() == 16, FIXTURE);
    admit(&ready, &bounded);
    for rounds in [
        vec![Round::stop(7, 7, 0)],
        vec![Round::stop(7, 2, 1)],
        vec![Round::of(2, 2, 0), Round::stop(0, 0, 1)],
    ] {
        let t = make_transcript(&ready, &previous, generation_job(8), &policy(), 0, &rounds);
        admit(&ready, &t);
    }
    let mut stop_top = array(&previous, "final_logit_top8").clone();
    stop_top[0] = with(&stop_top[0], "id", int(248_046));
    let stop_previous = with(&previous, "final_logit_top8", Json::Array(stop_top));
    let terminal = [Round::stop(0, 0, 1)];
    let stop = make_transcript(
        &ready,
        &stop_previous,
        generation_job(1),
        &policy(),
        0,
        &terminal,
    );
    admit(&ready, &stop);
    for stops in [[248_046, 248_044], [248_044, 248_046]] {
        let mut paired = stop.clone();
        paired.job = with(
            &paired.job,
            "stop_ids",
            Json::Array(stops.into_iter().map(int).collect()),
        );
        admit(&ready, &paired);
        let mut top = array(&paired.previous, "final_logit_top8").clone();
        top[0] = with(&top[0], "id", int(248_044));
        paired.previous = with(&paired.previous, "final_logit_top8", Json::Array(top));
        for event in &mut paired.events {
            if field(event, "event").string().unwrap() == b"token" {
                put(event, "id", int(248_044));
            }
        }
        put(
            &mut paired.done,
            "generated_ids",
            Json::Array(vec![int(248_044)]),
        );
        put(&mut paired.done, "stop_id", int(248_044));
        admit(&ready, &paired);
    }

    let proposed_stop = [Round {
        proposed: 7,
        ..Round::stop(0, 0, 1)
    }];
    let t = make_transcript(
        &ready,
        &stop_previous,
        generation_job(8),
        &policy(),
        0,
        &proposed_stop,
    );
    admit(&ready, &t);
    let append = make_transcript(
        &ready,
        &stop.done,
        input_job(3, false, &policy(), 0),
        &policy(),
        0,
        &[],
    );
    admit(&ready, &append);
    let reset = make_transcript(
        &ready,
        &append.done,
        input_job(8, true, &policy(), 9),
        &policy(),
        9,
        &[],
    );
    admit(&ready, &reset);
    let selected = with(&with(&policy(), "top_k", int(7)), "min_p", number(0.1));
    let t = make_transcript(
        &ready,
        &reset.done,
        input_job(2, true, &selected, 17),
        &selected,
        17,
        &[],
    );
    admit(&ready, &t);
}

fn context_checks() {
    let ready = ready_record();
    let mut previous = Json::Null;
    let mut prefix = 0u32;
    // Carry a generated, fully admitted input ancestry to each boundary.
    for boundary in [
        2044u32, 2048, 2051, 2052, 4096, 4099, 8192, 8195, 16387, 32771, 65528,
    ] {
        while prefix < boundary {
            let count = 8.min(boundary - prefix);
            let job = input_job(count, previous.is_null(), &policy(), 0);
            let fill = make_transcript(&ready, &previous, job, &policy(), 0, &[]);
            admit(&ready, &fill);
            previous = fill.done;
            prefix += count;
        }
        let mut plans = vec![vec![Round::of(0, 0, 1); 8]];
        for accepted in [1u32, 3, 6, 7] {
            let mut plan = vec![Round::of(7, accepted, u32::from(accepted < 7))];
            for _ in accepted + u32::from(accepted < 7)..8 {
                plan.push(Round::of(0, 0, 1));
            }
            plans.push(plan);
        }
        for plan in &plans {
            let t = make_transcript(&ready, &previous, generation_job(8), &policy(), 0, plan);
            admit(&ready, &t);
            for delta in [-1i64, 1] {
                let mut wrong = t.clone();
                let dispatches = integer(&wrong.events[0], "dispatches") as i64 + delta;
                put(&mut wrong.events[0], "dispatches", int(dispatches as u64));
                reject(&ready, &wrong);
            }
        }
        if boundary == 2051 {
            let serial = [Round::of(0, 0, 1); 8];
            let mut old_short_rule =
                make_transcript(&ready, &previous, generation_job(8), &policy(), 0, &serial);
            check(
                integer(&old_short_rule.events[0], "dispatches") == 2306,
                "token2052 includes48 selection dispatches",
            );
            put(&mut old_short_rule.events[0], "dispatches", int(2258));
            reject(&ready, &old_short_rule);
        }
    }
}

fn command_checks() {
    let ready = ready_record();
    let previous = seed_prefix(&ready);
    let greedy = policy();
    let good = generation_job(2);
    let refused = |job: &Json, before: &Json, prefix: u32, p: &Json, seed: u64| {
        refuses(v8_reply(&ready, before, job, prefix, p, seed));
    };
    for (key, value) in [
        ("generate", int(9)),
        ("generate", int(0)),
        ("generate", Json::Bool(true)),
        ("generate", parse("2.0")),
        ("tokens", Json::Array(vec![int(1)])),
        ("reset", Json::Bool(true)),
        ("stop_ids", Json::Array(Vec::new())),
        ("stop_ids", Json::Array(vec![int(248_046), int(2)])),
        ("stop_ids", Json::Array(vec![int(248_320)])),
        ("stop_ids", Json::Array(vec![int(248_046), int(248_046)])),
        ("stop_ids", Json::Array(vec![int(248_044), int(248_320)])),
        ("sampling", greedy.clone()),
        ("seed", int(0)),
        ("prompt_lookup", Json::Bool(false)),
        ("prompt_lookup", Json::Bool(true)),
        ("checkpoint_mode", Json::from("auto")),
        ("op", Json::from("branch_capture")),
        ("unknown", Json::Bool(true)),
    ] {
        refused(&with(&good, key, value), &previous, 16, &policy(), 0);
    }
    for key in ["tokens", "generate", "reset", "stop_ids"] {
        refused(&without(&good, key), &previous, 16, &policy(), 0);
    }
    refused(&good, &Json::Null, 16, &policy(), 0);
    refused(&good, &previous, 15, &policy(), 0);
    refused(&good, &previous, 65537, &policy(), 0);
    refused(
        &good,
        &with(&previous, "continuation_allowed", Json::Bool(false)),
        16,
        &policy(),
        0,
    );
    let entropy = with(field(&previous, "sampling"), "entropy_prefix", int(15));
    refused(
        &good,
        &with(&previous, "sampling", entropy),
        16,
        &policy(),
        0,
    );
    refused(&good, &previous, 16, &greedy, 1);
    for (key, value) in [
        ("temperature", number(0.1)),
        ("top_p", number(0.0)),
        ("top_p", number(1.1)),
        ("min_p", number(-0.1)),
        ("min_p", number(1.1)),
        ("top_k", int(248_321)),
        ("top_k", Json::Bool(true)),
        ("temperature", parse("1e9999")),
        ("unknown", int(0)),
    ] {
        refused(&good, &previous, 16, &with(&greedy, key, value), 0);
    }
    for n in [0u32, 9] {
        refused(
            &input_job(n, true, &greedy, 0),
            &Json::Null,
            0,
            &policy(),
            0,
        );
    }
    let one = input_job(1, true, &greedy, 0);
    refused(
        &with(&one, "tokens", Json::Array(vec![int(248_320)])),
        &Json::Null,
        0,
        &policy(),
        0,
    );
    refused(
        &with(&one, "seed", Json::Bool(true)),
        &Json::Null,
        0,
        &policy(),
        0,
    );
    refused(
        &with(&one, "sampling", with(&greedy, "top_k", int(1))),
        &Json::Null,
        0,
        &policy(),
        0,
    );
    refused(
        &with(&one, "stop_ids", Json::Array(Vec::new())),
        &Json::Null,
        0,
        &policy(),
        0,
    );
    refused(
        &without(&one, "prompt_lookup"),
        &Json::Null,
        0,
        &policy(),
        0,
    );
    refused(
        &with(&one, "prompt_lookup", Json::Bool(false)),
        &Json::Null,
        0,
        &policy(),
        0,
    );
    let end = with(
        &with(&previous, "committed_prefix", int(65535)),
        "sampling",
        with(field(&previous, "sampling"), "entropy_prefix", int(65535)),
    );
    refused(&good, &end, 65535, &policy(), 0); // Entire requested output reservation must fit.
}

fn event_checks() {
    let ready = ready_record();
    let previous = seed_prefix(&ready);
    let full = [Round::of(7, 7, 0)];
    let source = make_transcript(&ready, &previous, generation_job(7), &policy(), 0, &full);
    let modified = |index: usize, change: &dyn Fn(&Json) -> Json| {
        let mut bad = source.clone();
        bad.events[index] = change(&bad.events[index]);
        reject(&ready, &bad);
    };
    for (key, value) in [
        ("round_id", int(2)),
        ("round_base", int(17)),
        ("committed_prefix", int(22)),
        ("verified_tokens", int(8)),
        ("verified_tokens", int(1)),
        ("accepted_verified_tokens", int(0)),
        ("accepted_verified_tokens", int(8)),
        ("ordinary_tokens", int(1)),
        ("proposed_tokens", int(6)),
        ("proposed_tokens", int(8)),
        ("useful_tokens", int(8)),
        ("verification_max_tokens", int(2)),
        ("verification_head_dispatches", int(1)),
        ("dispatches", int(1)),
        ("capture_executes", int(1)),
        ("timings_qualified", Json::Bool(true)),
        ("gpu_profile", Json::Object(Vec::new())),
        ("round_wall_ms", number(-1.0)),
        ("round_wall_ms", parse("1e9999")),
        ("batch_gpu_ms", number(100.0)),
        ("batch_gpu_ms", number(0.0)),
        ("batch_wall_ms", number(100.0)),
        ("round_wall_ms", number(0.0)),
        ("recovery_batch_wall_ms", number(1.0)),
        ("fallback_wall_ms", number(1.0)),
        ("useful_tokens_per_second", number(0.0)),
        ("accepted_tokens_per_second", number(0.0)),
        ("unknown", Json::Bool(true)),
    ] {
        modified(0, &|event| with(event, key, value.clone()));
    }
    for (key, _) in ok(source.events[0].object()) {
        modified(0, &|event| without(event, key));
    }
    for (key, value) in [
        ("round_id", int(2)),
        ("round_base", int(17)),
        ("token_prefix", int(18)),
        ("committed_prefix", int(17)),
        ("round_committed_prefix", int(17)),
        ("verified", Json::Bool(false)),
        ("batched_commit", Json::Bool(false)),
        ("entropy_words", int(1)),
        ("id", int(248_320)),
        ("id", int(201)),
        ("id", Json::Bool(true)),
        ("source_logit", Json::from("1.0")),
        ("source_logit", number(1.5)),
        ("source_logit", parse("1e9999")),
        ("gpu_profile", Json::Object(Vec::new())),
        ("unknown", Json::Bool(true)),
    ] {
        modified(1, &|event| with(event, key, value.clone()));
    }
    for (key, _) in ok(source.events[1].object()) {
        modified(1, &|event| without(event, key));
    }
    {
        let mut bad = source.clone();
        bad.events.remove(0);
        reject(&ready, &bad);
    }
    {
        let mut bad = source.clone();
        bad.events.insert(1, bad.events[0].clone());
        reject(&ready, &bad);
    }
    {
        let mut bad = source.clone();
        bad.events.pop();
        reject(&ready, &bad);
    }
    {
        let mut bad = source.clone();
        bad.events.swap(1, 2);
        reject(&ready, &bad);
    }
    modified(1, &|event| with(event, "id", int(248_046)));
    {
        let mut bad = source.clone();
        bad.events.push(source.events[0].clone());
        reject(&ready, &bad);
    }
    modified(0, &|event| with(event, "hc_exact_round_routes", hc([0; 5])));
    modified(0, &|event| {
        let routes = with(field(&ready, "wide_q8_exact_routes"), "qkv_scalar", int(1));
        with(event, "wide_q8_exact_round_routes", routes)
    });
    {
        let mut bad = source.clone();
        bad.job = generation_job(2);
        reject(&ready, &bad);
    }
    let partial = [Round::stop(7, 2, 1)];
    let correction = make_transcript(&ready, &previous, generation_job(8), &policy(), 0, &partial);
    admit(&ready, &correction);
    {
        let mut bad = correction.clone();
        let last = bad.events.len() - 1;
        put(&mut bad.events[last], "verified", Json::Bool(true));
        reject(&ready, &bad);
    }
    {
        let mut bad = correction.clone();
        put(&mut bad.events[0], "recovery_batch_wall_ms", number(0.0));
        reject(&ready, &bad);
    }
    // An interrupted/uncertain response has no finish, hence no publishable span.
    let mut incomplete = ok(v8_reply(
        &ready,
        &source.previous,
        &source.job,
        source.before,
        &source.sampling,
        source.seed,
    ));
    for event in &source.events {
        ok(incomplete.event(event));
    }
    refuses(incomplete.events());
}

fn done_checks() {
    let ready = ready_record();
    let previous = seed_prefix(&ready);
    let rounds = [Round::of(2, 1, 1)];
    let source = make_transcript(&ready, &previous, generation_job(2), &policy(), 0, &rounds);
    let changed = |change: &dyn Fn(&Json) -> Json| {
        let mut bad = source.clone();
        bad.done = change(&bad.done);
        reject(&ready, &bad);
    };
    for (key, value) in [
        ("committed_prefix", int(17)),
        ("generated_ids", Json::Array(Vec::new())),
        ("status", Json::from("stopped")),
        ("finish_reason", Json::from("stop")),
        ("stop_id", int(248_046)),
        ("continuation_allowed", Json::Bool(false)),
        ("known_token_prefill", Json::Bool(false)),
        ("known_token_capacity", int(4)),
        ("block_gemv_formats", int(1)),
        ("hc_exact_final_logit_sha256", Json::from("A".repeat(64))),
        ("hc_exact_final_logit_sha256", Json::from("a".repeat(63))),
        ("lookup_history_epoch", int(3)),
        ("lookup_request_appended_tokens", int(7)),
        ("lookup_appended_tokens", int(1)),
        ("lookup_indexed_grams", int(0)),
        ("lookup_request_indexed_grams", int(0)),
        ("lookup_rounds", int(2)),
        ("lookup_verified_rounds", int(0)),
        ("lookup_recovery_rounds", int(0)),
        ("lookup_accepted_tokens", int(2)),
        ("lookup_ordinary_tokens", int(0)),
        ("lookup_round_wall_ms", number(0.0)),
        ("generation_batch_gpu_ms", number(0.0)),
        ("prefill_wall_ms", number(-1.0)),
        ("request_wall_ms", parse("1e9999")),
        ("gpu_profile", Json::Object(Vec::new())),
        ("final_logit_top8", Json::Array(Vec::new())),
    ] {
        changed(&|done| with(done, key, value.clone()));
    }
    for key in [
        "tracked_metal_buffer_bytes",
        "peak_metal_buffer_bytes",
        "gpu_allocation_count",
        "gpu_buffer_limit_bytes",
        "imported_weight_backing_bytes",
    ] {
        changed(&|done| with(done, key, int(integer(done, key) + 1)));
    }
    for (key, value) in [
        ("seed", int(42)),
        ("entropy_prefix", int(17)),
        ("entropy_words", int(1)),
        ("temperature", number(1.0)),
        ("valid", Json::Bool(false)),
    ] {
        changed(&|done| {
            let sampling = with(field(done, "sampling"), key, value.clone());
            with(done, "sampling", sampling)
        });
    }
    changed(&|done| with(done, "hc_exact_routes", hc([0; 5])));
    changed(&|done| with(done, "hc_exact_request_routes", hc([0; 5])));
    changed(&|done| {
        let mut ids = array(done, "generated_ids").clone();
        ids[1] = int(42);
        with(done, "generated_ids", Json::Array(ids))
    });
    let top_changed = |done: &Json, change: &dyn Fn(&mut Vec<Json>)| {
        let mut top = array(done, "final_logit_top8").clone();
        change(&mut top);
        with(done, "final_logit_top8", Json::Array(top))
    };
    for key in ["id", "logit", "unknown"] {
        changed(&|done| {
            top_changed(done, &|top| {
                let value = if key == "logit" {
                    parse("1e9999")
                } else {
                    int(200)
                };
                top[1] = with(&top[1], key, value);
            })
        });
    }
    {
        let mut good = source.clone();
        good.done = top_changed(&good.done, &|top| {
            top[1] = with(&top[1], "logit", number(1.25));
        });
        admit(&ready, &good);
    }
    changed(&|done| top_changed(done, &|top| top.swap(0, 1)));
}

pub fn pause_ms(duration: u64) {
    std::thread::sleep(Duration::from_millis(duration));
}

pub fn emit(value: &Json) {
    let mut line = dump(value);
    line.push(b'\n');
    let mut out = std::io::stdout().lock();
    if out.write_all(&line).and_then(|()| out.flush()).is_err() {
        std::process::exit(90);
    }
}

/// `std::ofstream(path, std::ios::app) << text`.
pub fn append_line(path: &Path, text: &str) {
    let mut file = ok(std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path));
    ok(file.write_all(text.as_bytes()));
}

/// `std::getline(std::cin, line)` until EOF.
pub fn read_line(input: &mut impl BufRead) -> Option<Vec<u8>> {
    let mut line = Vec::new();
    if ok(input.read_until(b'\n', &mut line)) == 0 {
        return None;
    }
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    Some(line)
}

/// `(directory, mode)`: the parent and file name of the `--policy` path.
pub fn policy_parts(policy_path: &Path) -> (PathBuf, String) {
    let directory = policy_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let mode = policy_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    (directory, mode)
}

/// Each `(name, value)` pair occurs exactly once as `argv[i], argv[i + 1]`.
pub fn fixed_once(args: &[String], fixed: &[(&str, &str)]) -> bool {
    fixed.iter().all(|(name, value)| {
        (2..args.len().saturating_sub(1))
            .filter(|&i| args[i] == *name && args[i + 1] == *value)
            .count()
            == 1
    })
}

/// The last `--policy` value, or empty.
pub fn policy_argument(args: &[String]) -> PathBuf {
    let mut policy_path = PathBuf::new();
    for i in 2..args.len().saturating_sub(1) {
        if args[i] == "--policy" {
            policy_path = PathBuf::from(&args[i + 1]);
        }
    }
    policy_path
}

pub fn fake_target(args: &[String]) -> i32 {
    // Refuse a changed private startup command rather than silently accepting the fixture's own expected ready fields under a different runtime policy.
    const FIXED: [(&str, &str); 13] = [
        ("--allow-full-model", "true"),
        ("--small-gemv-splits", "16"),
        ("--block-gemv-formats", "3"),
        ("--stack-execution", "direct"),
        ("--known-token-capacity", "8"),
        ("--reuse-residual-projections", "true"),
        ("--prompt-lookup", "true"),
        ("--verification-budget-bytes", "134217728"),
        ("--allow-prompt-lookup-experiments", "true"),
        ("--verification-max-tokens", "8"),
        ("--hc-exact", "combined"),
        ("--wide-q8-exact", "off"),
        ("--wide-q8-many", "off"),
    ];
    if args.len() != 2 + 2 * (FIXED.len() + 2) + 4 {
        return 81;
    }
    if !fixed_once(args, &FIXED) {
        return 82;
    }
    let policy_path = policy_argument(args);
    if policy_path.as_os_str().is_empty() {
        return 83;
    }
    let (directory, mode) = policy_parts(&policy_path);
    ok(std::fs::write(
        directory.join("started"),
        "fixed private argv\n",
    ));
    let ready = ready_record();
    emit(&ready);
    let mut previous = Json::Null;
    let mut installed_policy = Json::Null;
    let mut seed = 0;
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    while let Some(line) = read_line(&mut input) {
        append_line(&directory.join("commands"), "command\n");
        let job = ok(parse_json(&line));
        if boolean(&job, "reset") {
            installed_policy = field(&job, "sampling").clone();
            seed = integer(&job, "seed");
        }
        let n = integer(&job, "generate") as u32;
        let rounds = vec![Round::of(0, 0, 1); n as usize];
        let mut t = make_transcript(&ready, &previous, job, &installed_policy, seed, &rounds);
        if n != 0 && mode == "bad-round" {
            put(&mut t.events[0], "round_id", int(2));
        }
        for event in &t.events {
            emit(event);
        }
        if n != 0 {
            ok(std::fs::write(
                directory.join("events-sent"),
                "native records emitted; terminal not yet sent\n",
            ));
            if mode == "truncated" {
                return 0;
            }
            if mode == "slow-done" {
                // Bounded handshake, not a race against a guessed sleep: the parent observes emitted native events before terminal release.
                let observed = directory.join("cancel-observed");
                for _ in 0..100 {
                    if observed.exists() {
                        break;
                    }
                    pause_ms(5);
                }
                if !observed.exists() {
                    return 84;
                }
            }
            if mode == "bad-done" {
                put(
                    &mut t.done,
                    "hc_exact_final_logit_sha256",
                    Json::from("not-a-complete-hash"),
                );
            }
        }
        emit(&t.done);
        previous = t.done;
    }
    0
}

/// `OwnedWorkerOptions` is copied per worker in C++; its startup probe is not copyable here and every fixture leaves it unset.
pub fn copy_options(options: &WorkerOptions) -> WorkerOptions {
    WorkerOptions {
        executable: options.executable.clone(),
        policy: options.policy.clone(),
        library: options.library.clone(),
        shards: options.shards.clone(),
        stack_execution: options.stack_execution.clone(),
        checkpoint_slots: options.checkpoint_slots,
        checkpoint_budget_bytes: options.checkpoint_budget_bytes,
        startup_timeout: options.startup_timeout,
        step_timeout: options.step_timeout,
        drain_timeout: options.drain_timeout,
        burst_pacing: options.burst_pacing,
        startup_cancelled: None,
        #[cfg(feature = "verify8")]
        verify8: options.verify8,
        #[cfg(feature = "branch8")]
        branch8: options.branch8,
        #[cfg(feature = "qsa-stage-v8")]
        stage8_present: options.stage8_present,
        #[cfg(feature = "qsa-stage-v8")]
        stage8_enabled: options.stage8_enabled,
        #[cfg(feature = "fast-math")]
        fast_channel: options.fast_channel,
        scalar_channel: options.scalar_channel,
        mtp_sidecar: options.mtp_sidecar.clone(),
        dflash2_dir: options.dflash2_dir.clone(),
        dflash2_precision: options.dflash2_precision.clone(),
        drafter: options.drafter.clone(),
        mtp_depth: options.mtp_depth,
        mtp_request_carry: options.mtp_request_carry,
        mtp_draft_attention: options.mtp_draft_attention.clone(),
        mtp_output_packet: options.mtp_output_packet,
        qsa_sparse: options.qsa_sparse,
        gdn_ported: options.gdn_ported,
        gdn_prefill_ported: options.gdn_prefill_ported,
        prefill_input512: options.prefill_input512,
        prefill_dense: options.prefill_dense.clone(),
        prefill_grouped: options.prefill_grouped.clone(),
        prefill_hc_rows: options.prefill_hc_rows,
        prefill_router_window: options.prefill_router_window,
        prefill_hc_down_window: options.prefill_hc_down_window,
        prefill_hc_down_f32: options.prefill_hc_down_f32,
        prefill_final_logits: options.prefill_final_logits,
        gdn_old_order_batch: options.gdn_old_order_batch,
        q8_three_rows: options.q8_three_rows,
        concurrent_dispatch: options.concurrent_dispatch,
        derived_pack: options.derived_pack.clone(),
        derived_pack_auth_workers: options.derived_pack_auth_workers,
        derived_pack_trust_mode: options.derived_pack_trust_mode.clone(),
        raw_decode: options.raw_decode,
        mtp_draft_slice: options.mtp_draft_slice,
        verify_nax: options.verify_nax,
        verify_nax_m8: options.verify_nax_m8,
        verify_hc_gdn_fused: options.verify_hc_gdn_fused,
        host_logit_views: options.host_logit_views,
        logit_topk: options.logit_topk,
        ple_async_gather: options.ple_async_gather,
        mtp_fast_round: options.mtp_fast_round,
        dflash2_defer_context: options.dflash2_defer_context,
        dflash2_verify_rows: options.dflash2_verify_rows.clone(),
        prepared_validation: options.prepared_validation,
        verify_cost_profile: options.verify_cost_profile.clone(),
        verify_m8_splits: options.verify_m8_splits.clone(),
        dflash2_draft_policy: options.dflash2_draft_policy.clone(),
        dflash2_max_drafts: options.dflash2_max_drafts,
        dflash2_vocab_slice: options.dflash2_vocab_slice,
        known_token_capacity: options.known_token_capacity,
        checkpoint_ssd: options.checkpoint_ssd,
        checkpoint_ssd_budget_bytes: options.checkpoint_ssd_budget_bytes,
        checkpoint_ssd_floor_bytes: options.checkpoint_ssd_floor_bytes,
        checkpoint_ssd_dir: options.checkpoint_ssd_dir.clone(),
        checkpoint_policy: options.checkpoint_policy.clone(),
        overlap_prefill: options.overlap_prefill,
        swiglu_fused: options.swiglu_fused,
        q8_gemv_rg4: options.q8_gemv_rg4.clone(),
        q8_three_msg: options.q8_three_msg.clone(),
        moe_three_union: options.moe_three_union,
        mtp_lookup: options.mtp_lookup.clone(),
        mtp_lookup_match: options.mtp_lookup_match,
        prefill_m64: options.prefill_m64.clone(),
        prefill_multiseg: options.prefill_multiseg,
        prefill_swizzle: options.prefill_swizzle,
        prefill_router_fast: options.prefill_router_fast,
        qsa_prefill_fused: options.qsa_prefill_fused,
        hc_consumer_fused: options.hc_consumer_fused,
        ple_overlap: options.ple_overlap,
        shared_prefix_cache: options.shared_prefix_cache,
    }
}

static FIXTURES: AtomicUsize = AtomicUsize::new(0);

pub struct WorkerFixture {
    pub directory: PathBuf,
    pub options: WorkerOptions,
}

impl WorkerFixture {
    pub fn new(executable: &Path, mode: &str) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "tokspur-http-v8-rs-{}-{}",
            std::process::id(),
            FIXTURES.fetch_add(1, Ordering::Relaxed)
        ));
        if std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .is_err()
        {
            panic!("cannot create private V8 fixture directory");
        }
        let target = directory.join("verify8-target");
        if std::fs::hard_link(executable, &target).is_err() {
            ok(std::fs::copy(executable, &target));
        }
        let policy = directory.join(mode);
        let library = directory.join("owned.metallib");
        for file in [&policy, &library] {
            ok(std::fs::write(file, "CPU fixture, never a model\n"));
        }
        let shards = (0..4)
            .map(|i| {
                let file = directory.join(format!("fixture-shard-{i}"));
                ok(std::fs::write(&file, i.to_string()));
                file
            })
            .collect();
        let options = WorkerOptions {
            executable: target,
            policy,
            library,
            shards,
            verify8: true,
            startup_timeout: Duration::from_secs(8),
            step_timeout: Duration::from_secs(3),
            drain_timeout: Duration::from_millis(800),
            ..WorkerOptions::default()
        };
        Self { directory, options }
    }

    pub fn options(&self) -> WorkerOptions {
        copy_options(&self.options)
    }

    pub fn commands(&self) -> usize {
        std::fs::read_to_string(self.directory.join("commands"))
            .map_or(0, |text| text.lines().count())
    }
}

impl Drop for WorkerFixture {
    fn drop(&mut self) {
        if std::fs::remove_dir_all(&self.directory).is_err() {
            // Best-effort cleanup of a temporary fixture, as in C++.
        }
    }
}

pub fn soon() -> Instant {
    Instant::now() + Duration::from_secs(3)
}

/// `[] { return false; }`.
pub fn never() -> impl FnMut() -> Result<bool, CallbackFailed> {
    || Ok(false)
}

fn worker_checks(executable: &Path) {
    {
        let f = WorkerFixture::new(executable, "normal");
        let mut worker = ok(OwnedWorker::new(f.options()));
        let mut callbacks = 0usize;
        let bad = input_job(1, true, &with(&policy(), "temperature", number(1.0)), 0);
        refuses(worker.request(&bad, soon(), Some(&mut never()), None));
        check(
            f.commands() == 0 && worker.alive(),
            "non-greedy admission performed transport IO",
        );
        ok(worker.request(
            &input_job(8, true, &policy(), 0),
            soon(),
            Some(&mut never()),
            Some(&mut |_: &Json| {
                callbacks += 1;
                Ok(())
            }),
        ));
        check(callbacks == 0, FIXTURE);
        ok(worker.request(
            &generation_job(8),
            soon(),
            Some(&mut never()),
            Some(&mut |event: &Json| {
                callbacks += 1;
                check(
                    f.directory.join("events-sent").exists(),
                    "callback before completed native event batch",
                );
                let expected: &[u8] = if callbacks % 2 == 1 {
                    b"generation_round"
                } else {
                    b"token"
                };
                check(
                    text(event, "event") == expected,
                    "worker reordered admitted callbacks",
                );
                Ok(())
            }),
        ));
        check(
            callbacks == 16 && f.commands() == 2 && worker.alive(),
            FIXTURE,
        );
        ok(worker.close());
        check(!worker.alive(), FIXTURE);
    }
    for mode in ["bad-round", "bad-done", "truncated"] {
        let f = WorkerFixture::new(executable, mode);
        let mut worker = ok(OwnedWorker::new(f.options()));
        let mut callbacks = 0usize;
        ok(worker.request(
            &input_job(8, true, &policy(), 0),
            soon(),
            Some(&mut never()),
            None,
        ));
        refuses(worker.request(
            &generation_job(2),
            soon(),
            Some(&mut never()),
            Some(&mut |_: &Json| {
                callbacks += 1;
                Ok(())
            }),
        ));
        check(
            callbacks == 0 && !worker.alive() && f.commands() == 2,
            "uncertain reply exposed tokens or kept admission",
        );
        refuses(worker.request(
            &input_job(1, true, &policy(), 0),
            soon(),
            Some(&mut never()),
            None,
        ));
        check(f.commands() == 2, "failed private worker retried");
        ok(worker.close());
    }
    {
        let f = WorkerFixture::new(executable, "slow-done");
        let mut worker = ok(OwnedWorker::new(f.options()));
        let mut callbacks = 0usize;
        ok(worker.request(
            &input_job(8, true, &policy(), 0),
            soon(),
            Some(&mut never()),
            None,
        ));
        let outcome = worker.request(
            &generation_job(2),
            soon(),
            Some(&mut || {
                if !f.directory.join("events-sent").exists() {
                    return Ok(false);
                }
                ok(std::fs::write(
                    f.directory.join("cancel-observed"),
                    "parent cancellation observed before done\n",
                ));
                Ok(true)
            }),
            Some(&mut |_: &Json| {
                callbacks += 1;
                Ok(())
            }),
        );
        let cancelled = match outcome {
            Err(WorkerError::Cancelled(_)) => true,
            Err(error) => panic!("{error}"),
            Ok(_) => false,
        };
        check(
            cancelled && callbacks == 0 && worker.alive() && f.commands() == 2,
            "mid-response cancellation did not drain without publication",
        );
        refuses(worker.request(&generation_job(1), soon(), Some(&mut never()), None));
        check(f.commands() == 2, FIXTURE);
        ok(worker.request(
            &input_job(1, true, &policy(), 0),
            soon(),
            Some(&mut never()),
            None,
        ));
        check(f.commands() == 3, FIXTURE);
        ok(worker.close());
    }
    {
        let f = WorkerFixture::new(executable, "normal");
        let mut worker = ok(OwnedWorker::new(f.options()));
        ok(worker.request(
            &input_job(8, true, &policy(), 0),
            soon(),
            Some(&mut never()),
            None,
        ));
        let abort_job = Json::new_object([
            ("tokens", Json::Array(Vec::new())),
            ("generate", int(0)),
            ("reset", Json::Bool(false)),
            (
                "mtp_generation",
                Json::new_object([("id", int(42)), ("abort", Json::Bool(true))]),
            ),
        ]);
        let done = ok(worker.request(&abort_job, soon(), Some(&mut never()), None));
        check(
            text(&done, "event") == b"done",
            "MTP abort frame succeeded under verify8 mode",
        );

        let cancel = std::sync::atomic::AtomicBool::new(true);
        let mut probe = || Ok(cancel.load(std::sync::atomic::Ordering::Relaxed));
        let cancel_res = worker.request(&generation_job(1), soon(), Some(&mut probe), None);
        check(
            matches!(cancel_res, Err(WorkerError::Cancelled(_))),
            "request cancelled before submission",
        );

        refuses(worker.request(&generation_job(1), soon(), Some(&mut never()), None));

        let done_reset = ok(worker.request(&abort_job, soon(), Some(&mut never()), None));
        check(
            text(&done_reset, "event") == b"done",
            "MTP abort succeeds when needs_reset == true under verify8 mode",
        );

        ok(worker.close());
    }
    for mode in ["wrong-basename", "disabled", "replay", "checkpoint"] {
        let mut f = WorkerFixture::new(executable, "normal");
        match mode {
            "wrong-basename" => f.options.executable = PathBuf::from("/bin/sh"),
            "disabled" => f.options.verify8 = false,
            "replay" => f.options.stack_execution = "recurrent-replay".to_owned(),
            _ => {
                f.options.checkpoint_slots = 1;
                f.options.checkpoint_budget_bytes = 1 << 20;
            }
        }
        refuses(OwnedWorker::new(f.options()));
        check(
            !f.directory.join("started").exists() && f.commands() == 0,
            FIXTURE,
        );
    }
}

pub fn encode_text(text: &[u8]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut at = 0;
    while at < text.len() {
        if text[at..].starts_with(b"<|im_end|>") {
            result.push(248_046);
            at += 10;
        } else {
            result.push(u32::from(text[at]) + 1);
            at += 1;
        }
    }
    result
}

pub fn session_codec() -> ServeCodec {
    ServeCodec {
        render: Box::new(|request: &Json| {
            let messages = array(request, "messages");
            let last = messages.last().expect("fixture chat has a message");
            Ok(RenderedChat {
                prompt: String::from_utf8(text(last, "content").to_vec())
                    .unwrap()
                    .replace("[fixture_end]", "<|im_end|>")
                    .into_bytes(),
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        }),
        encode: Box::new(|prompt| Ok(encode_text(&prompt.prompt))),
        decode: Box::new(|ids: &[u32]| {
            let mut bytes = Vec::new();
            for &id in ids {
                if id == 248_046 {
                    bytes.extend_from_slice(b"<|im_end|>");
                } else {
                    check(id > 0 && id < 257, FIXTURE);
                    bytes.push((id - 1) as u8);
                }
            }
            Ok(bytes)
        }),
        admission: ResponseAdmission {
            vocabulary: 248_320,
            im_end: 248_046,
            end_of_text: 248_044,
            im_start: 248_045,
        },
    }
}

pub fn chat(text: &[u8], stream: bool, maximum: u32) -> Json {
    let text = String::from_utf8(text.to_vec())
        .unwrap()
        .replace("<|im_end|>", "[fixture_end]");
    Json::new_object([
        ("model", Json::from("tokspur")),
        (
            "messages",
            Json::Array(vec![Json::new_object([
                ("role", Json::from("user")),
                ("content", Json::from(text)),
            ])]),
        ),
        ("stream", Json::Bool(stream)),
        ("enable_thinking", Json::Bool(false)),
        ("temperature", int(0)),
        ("seed", int(0)),
        ("max_tokens", int(u64::from(maximum))),
    ])
}

pub fn tools() -> Json {
    parse(
        r#"[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}]"#,
    )
}

pub const TOOL_CALL: &str = "<tool_call><function=read_file><parameter=path>\nx\n</parameter></function></tool_call><|im_end|>";

/// `choices[0]`.
pub fn choice(value: &Json) -> &Json {
    &array(value, "choices")[0]
}

/// `choices[0].delta` of a streamed chunk.
pub fn delta(chunk: &Json) -> &Json {
    field(choice(chunk), "delta")
}

/// `usage.prompt_tokens_details.cached_tokens`.
pub fn cached_tokens(result: &Json) -> u64 {
    integer(
        field(field(result, "usage"), "prompt_tokens_details"),
        "cached_tokens",
    )
}

struct SessionFake {
    ready: Json,
    previous: Json,
    sampling: Json,
    seed: u64,
    output: Vec<u32>,
    next: usize,
    commands: usize,
    resets: usize,
    bad_done: bool,
}

impl SessionFake {
    fn new() -> Self {
        Self {
            ready: ready_record(),
            previous: Json::Null,
            sampling: Json::Null,
            seed: 0,
            output: Vec::new(),
            next: 0,
            commands: 0,
            resets: 0,
            bad_done: false,
        }
    }

    fn plan(&mut self, text: &str) {
        self.output = encode_text(text.as_bytes());
        self.next = 0;
    }

    fn call(&mut self, job: &Json, callback: TokenCallback<'_>) -> Result<Json, ServeError> {
        self.commands += 1;
        if boolean(job, "reset") {
            self.resets += 1;
            self.sampling = field(job, "sampling").clone();
            self.seed = integer(job, "seed");
        }
        let requested = integer(job, "generate") as u32;
        let mut rounds = Vec::new();
        let first = self.next;
        let mut i = 0;
        while i < requested && self.next < self.output.len() {
            let terminal = self.output[self.next] == 248_046;
            rounds.push(Round {
                terminal,
                ..Round::of(0, 0, 1)
            });
            if terminal {
                self.next += 1;
                break;
            }
            i += 1;
            self.next += 1;
        }
        let mut transcript = make_transcript(
            &self.ready,
            &self.previous,
            job.clone(),
            &self.sampling,
            self.seed,
            &rounds,
        );
        let mut ids = Vec::new();
        let mut at = first;
        for event in &mut transcript.events {
            if text(event, "event") == b"token" {
                ids.push(int(u64::from(self.output[at])));
                put(event, "id", int(u64::from(self.output[at])));
                at += 1;
            }
        }
        put(&mut transcript.done, "generated_ids", Json::Array(ids));
        let mut top = array(&transcript.done, "final_logit_top8").clone();
        let next = self.output.get(self.next).copied().unwrap_or(200);
        top[0] = with(&top[0], "id", int(u64::from(next)));
        put(&mut transcript.done, "final_logit_top8", Json::Array(top));
        for event in &transcript.events {
            callback(event)?;
        }
        self.previous = transcript.done.clone(); // Native state commits even if caller cancels after this done.
        if requested != 0 && self.bad_done {
            return Ok(with(
                &transcript.done,
                "hc_exact_final_logit_sha256",
                Json::from("uncertain"),
            ));
        }
        Ok(transcript.done)
    }
}

fn session_checks() {
    let ready = ready_record();
    let mut session = ok(ServeSession::new(
        session_codec(),
        SessionOptions {
            verify8: true,
            verify_ready: ready.clone(),
            ..SessionOptions::default()
        },
    ));
    let fake = RefCell::new(SessionFake::new());
    fake.borrow_mut().plan("ok<|im_end|>");
    let cancelled = Cell::new(false);
    let chunks: RefCell<Vec<Json>> = RefCell::new(Vec::new());
    let mut push = |value: &Json| {
        chunks.borrow_mut().push(value.clone());
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
        &with(&chat(b"A", false, 8), "temperature", number(0.1)),
        b"non-greedy",
        &mut target,
        &mut sink,
    ));
    check(
        fake.borrow().commands == 0,
        "HTTP non-greedy refusal contacted target",
    );
    let mut cancel_after_done = |job: &Json, event: TokenCallback<'_>| {
        let done = fake.borrow_mut().call(job, event)?;
        if integer(job, "generate") != 0 {
            cancelled.set(true);
        }
        Ok(done)
    };
    refuses(session.complete(
        &chat(b"A", true, 8),
        b"cancel-after-done",
        &mut cancel_after_done,
        &mut sink,
    ));
    check(
        session.committed_tokens() == 0 && fake.borrow().commands == 2,
        "session retained uncertain cancelled prefix",
    );
    for value in chunks.borrow().iter() {
        let delta = delta(value);
        check(
            !present(delta, "tool_calls") && !present(delta, "content"),
            "cancelled native batch released visible content/tools",
        );
    }
    cancelled.set(false);
    chunks.borrow_mut().clear();
    fake.borrow_mut().plan("ok<|im_end|>");
    let result = ok(session.complete(
        &chat(b"AB", false, 8),
        b"cold-recover",
        &mut target,
        &mut sink,
    ));
    check(
        text(choice(&result), "finish_reason") == b"stop" && fake.borrow().resets == 2,
        FIXTURE,
    );
    let reset_count = fake.borrow().resets;
    fake.borrow_mut().plan("ok<|im_end|>");
    let result = ok(session.complete(
        &chat(b"ABok<|im_end|>C", false, 8),
        b"append-after-recover",
        &mut target,
        &mut sink,
    ));
    check(
        fake.borrow().resets == reset_count && cached_tokens(&result) == 5,
        "fresh reset failed to reestablish exact append ancestry",
    );
    let tool_request = with(&chat(b"T", true, 10), "tools", tools());
    fake.borrow_mut().plan(TOOL_CALL);
    chunks.borrow_mut().clear();
    let result = ok(session.complete(&tool_request, b"tool-cutoff", &mut target, &mut sink));
    let selected = choice(&result);
    check(
        text(selected, "finish_reason") == b"length"
            && !present(field(selected, "message"), "tool_calls"),
        "length-cutoff frame released a tool",
    );
    for value in chunks.borrow().iter() {
        check(!present(delta(value), "tool_calls"), FIXTURE);
    }
    fake.borrow_mut().plan("ok<|im_end|>");
    fake.borrow_mut().bad_done = true;
    chunks.borrow_mut().clear();
    refuses(session.complete(&chat(b"X", true, 8), b"bad-batch", &mut target, &mut sink));
    check(session.committed_tokens() == 0, FIXTURE);
    for value in chunks.borrow().iter() {
        check(
            !present(delta(value), "content"),
            "malformed batch released content",
        );
    }
}

/// The fake target this binary's worker checks launch: itself, unless `TOKSPUR_FAKE_TARGET` names another (for example the C++ test binary).
pub fn fake_executable(args: &[String]) -> PathBuf {
    let fake = std::env::var_os("TOKSPUR_FAKE_TARGET").unwrap_or_else(|| args[0].clone().into());
    ok(std::fs::canonicalize(fake))
}

/// The C++ `main` (`previous_http_verify8_main` once included).
pub fn main(args: &[String]) -> i32 {
    if args.len() > 1 && args[1] == "target" {
        return fake_target(args);
    }
    ready_checks();
    section("ready_checks");
    positive_checks();
    section("positive_checks");
    context_checks();
    section("context_checks");
    command_checks();
    section("command_checks");
    event_checks();
    section("event_checks");
    done_checks();
    section("done_checks");
    session_checks();
    section("session_checks");
    worker_checks(&fake_executable(args));
    section("worker_checks");
    println!("Private HTTP V8 CPU checks={}", checks());
    0
}
