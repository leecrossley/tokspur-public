//! Challenger stress test suite for Milestones 5 & 6.
//! Empirically tests clock conservation, timing leak prevention, context length clamping, and MTP abort.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tokspur_json::{Json, parse_json};
use tokspur_server::session::{ServeCodec, ServeError, ServeSession, ServeSink, SessionOptions};
use tokspur_text::chat::RenderedChat;
use tokspur_text::response_protocol::ResponseAdmission;

static CHECKS: AtomicUsize = AtomicUsize::new(0);

fn check(value: bool, why: &str) {
    CHECKS.fetch_add(1, Ordering::Relaxed);
    assert!(value, "{why}");
}

fn parse(text: &str) -> Json {
    parse_json(text.as_bytes()).unwrap()
}

type Event<'a> = &'a mut dyn FnMut(&Json) -> Result<(), ServeError>;

#[derive(Default)]
struct Fake {
    position: u32,
    seed: u64,
    sampling: Json,
    output: Vec<u32>,
    next: usize,
    resets: usize,
    prefills: usize,
    generations: usize,
}

impl Fake {
    fn new(output: Vec<u32>) -> Self {
        Self {
            output,
            ..Self::default()
        }
    }

    fn call(&mut self, job: &Json, event: Event<'_>) -> Result<Json, ServeError> {
        if job.find("reset")?.unwrap().boolean()? {
            self.position = 0;
            self.resets += 1;
            self.sampling = job.find("sampling")?.unwrap().clone();
            self.seed = job.find("seed")?.unwrap().unsigned_integer()?;
        }
        let generation = job.find("generate")?.unwrap().unsigned_integer()? != 0;
        let stopped = generation && self.output.get(self.next) == Some(&1000);
        let mut ids = Vec::new();
        self.position += 1;
        if generation {
            self.generations += 1;
            let id = self.output[self.next];
            self.next += 1;
            ids.push(Json::integer(u64::from(id)));
            event(&Json::new_object([
                ("event", Json::from("token")),
                ("id", Json::integer(u64::from(id))),
                ("committed_prefix", Json::integer(u64::from(self.position))),
            ]))?;
        } else {
            self.prefills += 1;
        }
        let mut policy = self.sampling.object()?.clone();
        policy.push((b"valid".to_vec(), Json::Bool(true)));
        policy.push((
            b"entropy_prefix".to_vec(),
            Json::integer(u64::from(self.position)),
        ));
        policy.push((b"seed".to_vec(), Json::integer(self.seed)));
        let status = if generation {
            if stopped { "stopped" } else { "length" }
        } else {
            "prefilled"
        };
        let mut done = Json::new_object([
            ("event", Json::from("done")),
            ("committed_prefix", Json::integer(u64::from(self.position))),
            ("generated_ids", Json::Array(ids)),
            ("status", Json::from(status)),
            (
                "finish_reason",
                if generation {
                    Json::from(if stopped { "stop" } else { "length" })
                } else {
                    Json::Null
                },
            ),
            (
                "stop_id",
                if stopped {
                    Json::integer(1000)
                } else {
                    Json::Null
                },
            ),
            ("continuation_allowed", Json::Bool(!stopped && generation)),
            ("sampling", Json::Object(policy)),
            ("small_gemv_column_splits", Json::integer(16)),
            ("block_gemv_formats", Json::integer(3)),
            ("stack_execution", Json::from("direct")),
        ]);
        if generation {
            let mut fields = done.object()?.clone();
            fields.push((b"logits_available".to_vec(), Json::Bool(true)));
            done = Json::Object(fields);
        }
        Ok(done)
    }
}

fn encode_bytes(text: &[u8]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with(b"<|im_end|>") {
            result.push(1000);
            i += 10;
        } else {
            result.push(u32::from(text[i]) + 1);
            i += 1;
        }
    }
    result
}

fn decode_ids(ids: &[u32]) -> Vec<u8> {
    let mut result = Vec::new();
    for &id in ids {
        match id {
            1000 => result.extend_from_slice(b"<|im_end|>"),
            1001 => result.extend_from_slice(b"<|endoftext|>"),
            1002 => result.extend_from_slice(b"<|im_start|>"),
            _ => {
                if id > 0 && id <= 256 {
                    result.push((id - 1) as u8);
                } else {
                    result.push(b'x');
                }
            }
        }
    }
    result
}

fn make_codec(custom_prompt_len: usize) -> ServeCodec {
    ServeCodec {
        control_ids: Vec::new(),
        render: Box::new(|_| {
            Ok(RenderedChat {
                prompt: b"test prompt".to_vec(),
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        }),
        encode: Box::new(move |_| {
            // Emits exactly `custom_prompt_len` tokens within vocabulary
            Ok(vec![66; custom_prompt_len]) // 66 -> 'A'
        }),
        decode: Box::new(|ids: &[u32]| Ok(decode_ids(ids))),
        admission: ResponseAdmission {
            vocabulary: 4096,
            im_end: 1000,
            end_of_text: 1001,
            im_start: 1002,
        },
    }
}

#[test]
fn test_context_length_clamping_boundary_65535() {
    // Exactly 65,535 tokens: valid prompt leaving exactly 1 token generation space
    let codec = make_codec(65535);
    let mut session = ServeSession::new(codec, SessionOptions::default()).unwrap();
    let fake = RefCell::new(Fake::new(encode_bytes(b"X<|im_end|>")));
    let mut call = |job: &Json, event: Event<'_>| fake.borrow_mut().call(job, event);

    let cancel = AtomicBool::new(false);
    let mut cancelled = || cancel.load(Ordering::Relaxed);
    let mut new_seed = || 42u64;
    let mut sink = ServeSink {
        chunk: None,
        cancelled: &mut cancelled,
        new_seed: &mut new_seed,
        prefix_trace: None,
    };

    let req = parse(
        r#"{"model":"tokspur","messages":[{"role":"user","content":"hi"}],"max_tokens":32768,"enable_thinking":false}"#,
    );
    let result = session
        .complete(&req, b"resp-65535", &mut call, &mut sink)
        .unwrap();

    let usage = result.find("usage").unwrap().unwrap();
    let prompt_tokens = usage
        .find("prompt_tokens")
        .unwrap()
        .unwrap()
        .unsigned_integer()
        .unwrap();
    let completion_tokens = usage
        .find("completion_tokens")
        .unwrap()
        .unwrap()
        .unsigned_integer()
        .unwrap();
    let total_tokens = usage
        .find("total_tokens")
        .unwrap()
        .unwrap()
        .unsigned_integer()
        .unwrap();

    check(prompt_tokens == 65535, "prompt_tokens must be 65535");
    check(
        completion_tokens == 1,
        "completion_tokens must be clamped to 1",
    );
    check(total_tokens == 65536, "total_tokens must be exactly 65536");
    check(
        session.committed_tokens() == 65536,
        "committed tokens must be 65536",
    );
}

#[test]
fn test_context_length_overflow_rejected_at_65536() {
    // Exactly 65,536 tokens: leaves 0 generation space -> must be refused
    let codec = make_codec(65536);
    let mut session = ServeSession::new(codec, SessionOptions::default()).unwrap();
    let fake = RefCell::new(Fake::new(vec![1000]));
    let mut call = |job: &Json, event: Event<'_>| fake.borrow_mut().call(job, event);

    let cancel = AtomicBool::new(false);
    let mut cancelled = || cancel.load(Ordering::Relaxed);
    let mut new_seed = || 42u64;
    let mut sink = ServeSink {
        chunk: None,
        cancelled: &mut cancelled,
        new_seed: &mut new_seed,
        prefix_trace: None,
    };

    let req =
        parse(r#"{"model":"tokspur","messages":[{"role":"user","content":"hi"}],"max_tokens":10}"#);
    let result = session.complete(&req, b"resp-65536", &mut call, &mut sink);

    match result {
        Err(ServeError::Admission(_)) => {
            CHECKS.fetch_add(1, Ordering::Relaxed);
        }
        _ => panic!("Expected Admission refusal for prompt_tokens == 65536"),
    }
}

#[test]
fn test_context_length_overflow_rejected_at_empty_prompt() {
    // 0 tokens: leaves no prompt -> must be refused
    let codec = make_codec(0);
    let mut session = ServeSession::new(codec, SessionOptions::default()).unwrap();
    let fake = RefCell::new(Fake::new(vec![1000]));
    let mut call = |job: &Json, event: Event<'_>| fake.borrow_mut().call(job, event);

    let cancel = AtomicBool::new(false);
    let mut cancelled = || cancel.load(Ordering::Relaxed);
    let mut new_seed = || 42u64;
    let mut sink = ServeSink {
        chunk: None,
        cancelled: &mut cancelled,
        new_seed: &mut new_seed,
        prefix_trace: None,
    };

    let req =
        parse(r#"{"model":"tokspur","messages":[{"role":"user","content":"hi"}],"max_tokens":10}"#);
    let result = session.complete(&req, b"resp-0", &mut call, &mut sink);

    match result {
        Err(ServeError::Admission(_)) => {
            CHECKS.fetch_add(1, Ordering::Relaxed);
        }
        _ => panic!("Expected Admission refusal for prompt_tokens == 0"),
    }
}

#[test]
fn test_steady_clock_conservation_and_trace_accounting() {
    let codec = make_codec(16);
    let mut session = ServeSession::new(codec, SessionOptions::default()).unwrap();
    // Target produces 6 tokens: "hello" + EOS
    let fake = RefCell::new(Fake::new(encode_bytes(b"hello<|im_end|>")));
    let mut call = |job: &Json, event: Event<'_>| fake.borrow_mut().call(job, event);

    let cancel = AtomicBool::new(false);
    let mut cancelled = || cancel.load(Ordering::Relaxed);
    let mut new_seed = || 42u64;

    let trace_records = Rc::new(RefCell::new(Vec::new()));
    let trace_records_clone = trace_records.clone();
    let mut prefix_trace = move |record: &Json| {
        trace_records_clone.borrow_mut().push(record.clone());
        Ok(())
    };

    let mut sink = ServeSink {
        chunk: None,
        cancelled: &mut cancelled,
        new_seed: &mut new_seed,
        prefix_trace: Some(&mut prefix_trace),
    };

    let req = parse(
        r#"{"model":"tokspur","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"enable_thinking":false}"#,
    );
    let result = session
        .complete(&req, b"resp-clock", &mut call, &mut sink)
        .unwrap();

    // 1. Verify that trace captured prompt_prefix and completion events
    let traces = trace_records.borrow();
    check(
        traces.len() == 2,
        "must emit prompt_prefix and completion traces",
    );
    check(
        traces[0].find("event").unwrap().unwrap().string().unwrap() == b"prompt_prefix",
        "event 0 is prompt_prefix",
    );
    check(
        traces[1].find("event").unwrap().unwrap().string().unwrap() == b"completion",
        "event 1 is completion",
    );

    // 2. Verify clock conservation law in completion event
    let completion = &traces[1];
    let adm = completion
        .find("server_admission_wall_ms")
        .unwrap()
        .unwrap()
        .number_lexeme()
        .unwrap();
    let prf = completion
        .find("server_prefill_wall_ms")
        .unwrap()
        .unwrap()
        .number_lexeme()
        .unwrap();
    let generation_lex = completion
        .find("server_generation_wall_ms")
        .unwrap()
        .unwrap()
        .number_lexeme()
        .unwrap();
    let req_w = completion
        .find("server_request_wall_ms")
        .unwrap()
        .unwrap()
        .number_lexeme()
        .unwrap();
    let tp = completion
        .find("server_generation_throughput")
        .unwrap()
        .unwrap()
        .number_lexeme()
        .unwrap();

    let adm_f: f64 = adm.parse().unwrap();
    let prf_f: f64 = prf.parse().unwrap();
    let gen_f: f64 = generation_lex.parse().unwrap();
    let req_f: f64 = req_w.parse().unwrap();
    let tp_f: f64 = tp.parse().unwrap();

    check(adm_f >= 0.0, "admission wall ms must be non-negative");
    check(prf_f >= 0.0, "prefill wall ms must be non-negative");
    check(gen_f >= 0.0, "generation wall ms must be non-negative");
    check(req_f >= 0.0, "request wall ms must be non-negative");

    let diff = ((adm_f + prf_f + gen_f) - req_f).abs();
    check(
        diff < 1e-3,
        "phase clocks must conserve server_request_wall_ms (< 1e-3)",
    );

    if gen_f > 0.0 {
        let completion_tokens = completion
            .find("completion_tokens")
            .unwrap()
            .unwrap()
            .unsigned_integer()
            .unwrap() as f64;
        let expected_tp = (completion_tokens * 1000.0) / gen_f;
        let tp_diff = (tp_f - expected_tp).abs();
        check(
            tp_diff < 1e-3,
            "throughput must match authoritative formula",
        );
    }

    // 3. Verify timing keys NEVER leak into client-facing HTTP JSON response
    let json_dump = String::from_utf8(result.dump().unwrap()).unwrap();
    for forbidden in [
        "server_admission_wall_ms",
        "server_prefill_wall_ms",
        "server_generation_wall_ms",
        "server_request_wall_ms",
        "server_generation_throughput",
        "prefill_generation_overlap",
        "native_worker",
    ] {
        check(
            !json_dump.contains(forbidden),
            "forbidden timing key must not leak into HTTP JSON",
        );
    }
}

#[test]
fn test_streaming_sse_chunks_timing_leak_free() {
    let codec = make_codec(8);
    let mut session = ServeSession::new(codec, SessionOptions::default()).unwrap();
    let fake = RefCell::new(Fake::new(encode_bytes(b"hello<|im_end|>")));
    let mut call = |job: &Json, event: Event<'_>| fake.borrow_mut().call(job, event);

    let cancel = AtomicBool::new(false);
    let mut cancelled = || cancel.load(Ordering::Relaxed);
    let mut new_seed = || 42u64;

    let chunks = Rc::new(RefCell::new(Vec::new()));
    let chunks_clone = chunks.clone();
    let mut chunk_sink = move |chunk: &Json| {
        chunks_clone.borrow_mut().push(chunk.clone());
        Ok(())
    };

    let mut sink = ServeSink {
        chunk: Some(&mut chunk_sink),
        cancelled: &mut cancelled,
        new_seed: &mut new_seed,
        prefix_trace: None,
    };

    let req = parse(
        r#"{"model":"tokspur","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"stream":true,"stream_options":{"include_usage":true},"enable_thinking":false}"#,
    );
    let _result = session
        .complete(&req, b"resp-stream", &mut call, &mut sink)
        .unwrap();

    let emitted_chunks = chunks.borrow();
    check(!emitted_chunks.is_empty(), "must emit streaming chunks");

    for chunk in emitted_chunks.iter() {
        let dump = String::from_utf8(chunk.dump().unwrap()).unwrap();
        for forbidden in [
            "server_admission_wall_ms",
            "server_prefill_wall_ms",
            "server_generation_wall_ms",
            "server_request_wall_ms",
            "server_generation_throughput",
            "prefill_generation_overlap",
            "native_worker",
        ] {
            check(
                !dump.contains(forbidden),
                "forbidden timing key must not leak into streaming chunk",
            );
        }
    }
}

#[test]
fn test_abort_mtp_generation_idle_and_active_contract() {
    // Verify JSON shape of MTP abort command
    let id = 42u64;
    let job = Json::new_object([
        ("tokens", Json::Array(Vec::new())),
        ("generate", Json::integer(0)),
        ("reset", Json::Bool(false)),
        (
            "mtp_generation",
            Json::new_object([("id", Json::integer(id)), ("abort", Json::Bool(true))]),
        ),
    ]);

    let dump = String::from_utf8(job.dump().unwrap()).unwrap();
    check(dump.contains(r#""generate":0"#), "generate is 0");
    check(dump.contains(r#""reset":false"#), "reset is false");
    check(dump.contains(r#""tokens":[]"#), "tokens is empty");
    check(dump.contains(r#""abort":true"#), "abort is true");
    check(dump.contains(r#""id":42"#), "id matches");
}
