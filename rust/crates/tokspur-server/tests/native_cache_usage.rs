//! Generated worker metadata through the actual serving session and MTP reply admission.
//! These tests cover usage and failure lifecycle; they do not execute native cache storage or a model.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::cell::Cell;
use std::collections::VecDeque;

use tokspur_json::{Json, parse_json};
use tokspur_server::session::{
    ServeCodec, ServeError, ServeSession, ServeSink, SessionOptions, TokenCallback,
};
use tokspur_text::chat::RenderedChat;
use tokspur_text::response_protocol::ResponseAdmission;
use tokspur_wire::mtp_wire::MtpWireReply;

fn get<'a>(value: &'a Json, name: &str) -> &'a Json {
    value.find(name).unwrap().unwrap()
}

fn number(value: &Json, name: &str) -> u64 {
    get(value, name).unsigned_integer().unwrap()
}

fn put(value: &mut Json, name: &str, replacement: Json) {
    let mut fields = value.object().unwrap().clone();
    fields.retain(|(key, _)| key != name.as_bytes());
    fields.push((name.as_bytes().to_vec(), replacement));
    *value = Json::Object(fields);
}

#[derive(Clone)]
enum Counts {
    Restored(u64),
    Legacy,
    Fields(Option<Json>, Option<Json>),
}

impl Counts {
    fn write(&self, done: &mut Json, inputs: u64) {
        let (restored, replayed) = match self {
            Self::Restored(restored) => (
                Some(Json::integer(*restored)),
                Some(Json::integer(inputs.saturating_sub(*restored))),
            ),
            Self::Legacy => (None, None),
            Self::Fields(restored, replayed) => (restored.clone(), replayed.clone()),
        };
        if let Some(restored) = restored {
            put(done, "checkpoint_restored_tokens", restored);
        }
        if let Some(replayed) = replayed {
            put(done, "replayed_input_tokens", replayed);
        }
    }
}

fn ready() -> Json {
    parse_json(br#"{"event":"ready","maximum_generated_tokens":4096,"stack_execution":"direct","small_gemv_column_splits":16,"block_gemv_formats":3,"native_mtp_protocol":"native-mtp-depth2-lookahead-v1","native_mtp_depth":2,"native_mtp_draft_attention":"tiled128","native_mtp_input_packet_schema":1,"native_mtp_input_packet_max":512,"native_prefill_dense":"f32","native_prefill_grouped":"f32","native_mtp_priming_batch_capacity":512,"native_mtp_priming_batch_bytes":1024,"native_mtp_priming_batch_submits":0,"native_mtp_carry_scope":"http-generation-v1","native_mtp_generation":{"id":0,"active":false,"remaining":0,"pending":false},"native_mtp_head_calls":0,"native_mtp_prime_calls":0,"native_mtp_rounds":0,"native_mtp_proposed":0,"native_mtp_accepted":0,"native_mtp_recovery_rounds":0,"native_mtp_carried_anchors":0,"native_mtp_discarded_anchors":0}"#).unwrap()
}

struct Worker {
    ready: Json,
    position: u64,
    primes: u64,
    seed: u64,
    sampling: Json,
    generation_id: u64,
    inputs: VecDeque<Counts>,
    output: Counts,
    input_packets: usize,
    resets: usize,
    cancel_after_input: Option<usize>,
    corrupt_prime_count: bool,
}

impl Worker {
    fn new(inputs: impl IntoIterator<Item = Counts>) -> Self {
        Self {
            ready: ready(),
            position: 0,
            primes: 0,
            seed: 0,
            sampling: Json::Null,
            generation_id: 0,
            inputs: inputs.into_iter().collect(),
            output: Counts::Restored(0),
            input_packets: 0,
            resets: 0,
            cancel_after_input: None,
            corrupt_prime_count: false,
        }
    }

    fn call(
        &mut self,
        job: &Json,
        event: TokenCallback<'_>,
        cancelled: &Cell<bool>,
    ) -> Result<Json, ServeError> {
        let reset = get(job, "reset").boolean().unwrap();
        if reset {
            self.resets += 1;
            self.position = 0;
            self.sampling = get(job, "sampling").clone();
            self.seed = number(job, "seed");
        }
        let inputs = get(job, "tokens").array().unwrap().len() as u64;
        let generate = number(job, "generate");
        let mut ids = Vec::new();
        let counts = if inputs > 0 {
            assert_eq!(generate, 0);
            self.input_packets += 1;
            self.position += inputs;
            // Native restore credits logical prime calls even when it skips their execution.
            self.primes += inputs - u64::from(reset);
            self.inputs.pop_front().unwrap_or(Counts::Restored(0))
        } else {
            assert_eq!(generate, 1);
            self.generation_id = number(get(job, "mtp_generation"), "id");
            self.position += 1;
            self.primes += 1;
            ids.push(Json::integer(u64::from(b'a')));
            event(&Json::new_object([
                ("event", Json::from("token")),
                ("id", Json::integer(u64::from(b'a'))),
                ("committed_prefix", Json::integer(self.position)),
                ("entropy_words", Json::integer(0)),
            ]))?;
            self.output.clone()
        };
        let mut sampled = self.sampling.clone();
        for (name, value) in [
            ("valid", Json::Bool(true)),
            ("entropy_prefix", Json::integer(self.position)),
            ("entropy_words", Json::integer(0)),
            ("seed", Json::integer(self.seed)),
        ] {
            put(&mut sampled, name, value);
        }
        let mut done = self.ready.clone();
        for (name, value) in [
            ("event", Json::from("done")),
            ("committed_prefix", Json::integer(self.position)),
            ("generated_ids", Json::Array(ids)),
            ("sampling", sampled),
            (
                "status",
                Json::from(if inputs > 0 { "prefilled" } else { "length" }),
            ),
            ("continuation_allowed", Json::Bool(true)),
            (
                "native_mtp_prime_calls",
                Json::integer(self.primes + u64::from(self.corrupt_prime_count)),
            ),
            (
                "native_mtp_generation",
                Json::new_object([
                    ("id", Json::integer(self.generation_id)),
                    ("active", Json::Bool(false)),
                    ("remaining", Json::integer(0)),
                    ("pending", Json::Bool(false)),
                ]),
            ),
            // Deliberately unrelated lifetime/absolute-prefix values must never become HTTP usage.
            ("shared_prefix_cache_restored_tokens", Json::integer(98_000)),
            ("checkpoint_ssd_restored_tokens", Json::integer(104_000)),
        ] {
            put(&mut done, name, value);
        }
        counts.write(&mut done, inputs);
        if inputs > 0 && self.cancel_after_input == Some(self.input_packets) {
            cancelled.set(true);
        }
        Ok(done)
    }
}

fn session() -> ServeSession {
    let codec = ServeCodec {
        render: Box::new(|request| {
            let messages = get(request, "messages").array().unwrap();
            Ok(RenderedChat {
                prompt: get(messages.last().unwrap(), "content")
                    .string()
                    .unwrap()
                    .to_vec(),
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        }),
        encode: Box::new(|text| Ok(text.prompt.iter().map(|&byte| u32::from(byte)).collect())),
        decode: Box::new(|ids| Ok(ids.iter().map(|&id| u8::try_from(id).unwrap()).collect())),
        admission: ResponseAdmission {
            vocabulary: 4096,
            im_end: 1000,
            end_of_text: 1001,
            im_start: 1002,
        },
    };
    ServeSession::new(
        codec,
        SessionOptions {
            mtp_ready: ready(),
            ..SessionOptions::default()
        },
    )
    .unwrap()
}

fn complete(
    session: &mut ServeSession,
    worker: &mut Worker,
    prompt: &str,
    stream: bool,
    cancelled: &Cell<bool>,
) -> (Result<Json, ServeError>, Vec<Json>, Vec<Json>) {
    let request = Json::new_object([
        ("model", Json::from("tokspur")),
        (
            "messages",
            Json::Array(vec![Json::new_object([
                ("role", Json::from("user")),
                ("content", Json::from(prompt)),
            ])]),
        ),
        ("max_tokens", Json::integer(1)),
        ("seed", Json::integer(44)),
        ("stream", Json::Bool(stream)),
        ("enable_thinking", Json::Bool(false)),
    ]);
    let mut request = request;
    if stream {
        put(
            &mut request,
            "stream_options",
            Json::new_object([("include_usage", Json::Bool(true))]),
        );
    }
    let mut traces = Vec::new();
    let mut chunks = Vec::new();
    let mut trace = |value: &Json| {
        traces.push(value.clone());
        Ok(())
    };
    let mut chunk = |value: &Json| {
        chunks.push(value.clone());
        Ok(())
    };
    let mut check_cancelled = || cancelled.get();
    let mut seed = || 44;
    let mut sink = ServeSink {
        chunk: Some(&mut chunk),
        cancelled: &mut check_cancelled,
        new_seed: &mut seed,
        prefix_trace: Some(&mut trace),
    };
    let result = session.complete(
        &request,
        b"cache-usage",
        &mut |job, event| worker.call(job, event, cancelled),
        &mut sink,
    );
    (result, traces, chunks)
}

fn cached(response: &Json) -> u64 {
    number(
        get(get(response, "usage"), "prompt_tokens_details"),
        "cached_tokens",
    )
}

fn assert_usage(response: &Json, traces: &[Json], chunks: &[Json], expected: u64) {
    assert_eq!(cached(response), expected);
    let completed = traces
        .iter()
        .find(|item| get(item, "event").string().unwrap() == b"completion")
        .unwrap();
    assert_eq!(number(completed, "cached_tokens"), expected);
    assert_eq!(
        number(completed, "new_tokens"),
        number(get(response, "usage"), "prompt_tokens") - expected
    );
    if !chunks.is_empty() {
        let final_chunk = chunks.last().unwrap();
        assert!(get(final_chunk, "choices").array().unwrap().is_empty());
        assert_eq!(get(final_chunk, "usage"), get(response, "usage"));
    }
}

#[test]
fn accumulates_packets_once_and_preserves_active_prefix_then_resets_request_usage() {
    let mut session = session();
    let mut worker = Worker::new([
        Counts::Restored(512),
        Counts::Restored(128),
        Counts::Restored(0),
    ]);
    let cancelled = Cell::new(false);
    let prompt = "x".repeat(1100);
    let (response, traces, chunks) = complete(&mut session, &mut worker, &prompt, true, &cancelled);
    assert!(
        !chunks.is_empty(),
        "streamed request must publish final usage"
    );
    assert_usage(&response.unwrap(), &traces, &chunks, 640);
    assert_eq!(worker.input_packets, 3);
    assert_eq!(number(&traces[0], "cached_tokens"), 0); // Admission trace precedes native restores.
    assert_eq!(session.committed_tokens(), 1101);
    assert_eq!(worker.resets, 1);

    worker
        .inputs
        .extend([Counts::Restored(256), Counts::Restored(0)]);
    let appended = format!("{prompt}a{}", "y".repeat(600));
    let (response, traces, chunks) =
        complete(&mut session, &mut worker, &appended, false, &cancelled);
    assert_usage(&response.unwrap(), &traces, &chunks, 1357); // 1101 active plus 256 newly restored.
    assert_eq!(worker.resets, 1);
    assert_eq!(number(&traces[0], "cached_tokens"), 1101);

    let (response, traces, chunks) =
        complete(&mut session, &mut worker, "fresh", false, &cancelled);
    assert_usage(&response.unwrap(), &traces, &chunks, 0);
    assert_eq!(worker.resets, 2);
}

#[test]
fn legacy_absence_and_present_zero_both_report_no_native_reuse() {
    for counts in [Counts::Legacy, Counts::Restored(0)] {
        let mut session = session();
        let mut worker = Worker::new([counts.clone()]);
        worker.output = counts;
        let (response, traces, chunks) =
            complete(&mut session, &mut worker, "cold", false, &Cell::new(false));
        assert_usage(&response.unwrap(), &traces, &chunks, 0);
    }
}

#[test]
fn malformed_packet_counts_refuse_without_publishing_usage_and_retry_cold() {
    let bad = [
        Counts::Fields(Some(Json::integer(1)), None),
        Counts::Fields(None, Some(Json::integer(511))),
        Counts::Fields(Some(Json::from("1")), Some(Json::integer(511))),
        Counts::Fields(Some(Json::Null), Some(Json::integer(512))),
        Counts::Fields(Some(Json::Bool(false)), Some(Json::integer(512))),
        Counts::Fields(Some(parse_json(b"1.0").unwrap()), Some(Json::integer(511))),
        Counts::Fields(Some(Json::integer(1)), Some(Json::integer(u64::MAX))),
        Counts::Fields(Some(parse_json(b"-1").unwrap()), Some(Json::integer(513))),
        Counts::Fields(Some(Json::integer(1)), Some(Json::from("511"))),
        Counts::Fields(Some(Json::integer(u64::MAX)), Some(Json::integer(0))),
        Counts::Fields(Some(Json::integer(513)), Some(Json::integer(0))),
        Counts::Fields(Some(Json::integer(1)), Some(Json::integer(510))),
    ];
    for counts in bad {
        let mut session = session();
        let mut worker = Worker::new([Counts::Restored(512), counts]);
        let cancelled = Cell::new(false);
        let (result, traces, chunks) = complete(
            &mut session,
            &mut worker,
            &"x".repeat(1024),
            true,
            &cancelled,
        );
        assert!(result.is_err());
        assert_eq!(session.committed_tokens(), 0);
        assert!(
            !traces
                .iter()
                .any(|item| get(item, "event").string().unwrap() == b"completion")
        );
        assert!(!chunks.iter().any(|item| {
            item.find("usage")
                .unwrap()
                .is_some_and(|usage| !usage.is_null())
        }));
        let (response, traces, chunks) =
            complete(&mut session, &mut worker, "retry", false, &cancelled);
        assert_usage(&response.unwrap(), &traces, &chunks, 0);
        assert_eq!(worker.resets, 2);
    }
}

#[test]
fn generation_cannot_claim_restored_inputs_and_existing_prime_validation_remains_strict() {
    for corrupt_prime in [false, true] {
        let mut session = session();
        let mut worker = Worker::new([Counts::Restored(4)]);
        worker.corrupt_prime_count = corrupt_prime;
        if !corrupt_prime {
            worker.output = Counts::Restored(1);
        }
        let (result, traces, _) =
            complete(&mut session, &mut worker, "cold", false, &Cell::new(false));
        assert!(result.is_err());
        assert_eq!(session.committed_tokens(), 0);
        assert!(
            !traces
                .iter()
                .any(|item| get(item, "event").string().unwrap() == b"completion")
        );
    }
}

#[test]
fn cancellation_after_admitted_restore_discards_request_usage_before_retry() {
    let mut session = session();
    let mut worker = Worker::new([Counts::Restored(512), Counts::Restored(128)]);
    worker.cancel_after_input = Some(2);
    let cancelled = Cell::new(false);
    let (result, traces, chunks) = complete(
        &mut session,
        &mut worker,
        &"x".repeat(1024),
        true,
        &cancelled,
    );
    assert_eq!(
        result.unwrap_err().to_string(),
        "request cancelled or deadline expired"
    );
    assert_eq!(session.committed_tokens(), 0);
    assert!(
        !traces
            .iter()
            .any(|item| get(item, "event").string().unwrap() == b"completion")
    );
    assert!(!chunks.iter().any(|item| {
        item.find("usage")
            .unwrap()
            .is_some_and(|usage| !usage.is_null())
    }));
    cancelled.set(false);
    worker.cancel_after_input = None;
    let (response, traces, chunks) =
        complete(&mut session, &mut worker, "retry", false, &cancelled);
    assert_usage(&response.unwrap(), &traces, &chunks, 0);
    assert_eq!(worker.resets, 2);
}

#[test]
fn restore_count_is_unavailable_until_the_entire_reply_is_admitted() {
    let ready = ready();
    let job = parse_json(br#"{"tokens":[1,2,3,4],"generate":0,"reset":true,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":44}"#).unwrap();
    let mut worker = Worker::new([Counts::Restored(4)]);
    let done = worker
        .call(
            &job,
            &mut |_| panic!("prefill emitted token"),
            &Cell::new(false),
        )
        .unwrap();
    let mut reply = MtpWireReply::new(&ready, &ready, &job, false).unwrap();
    assert!(reply.restored_input_tokens().is_err());
    let mut bad = done.clone();
    put(&mut bad, "native_mtp_prime_calls", Json::integer(99));
    assert!(reply.finish(&bad).is_err());
    assert!(reply.restored_input_tokens().is_err());
    reply.finish(&done).unwrap();
    assert_eq!(reply.restored_input_tokens().unwrap(), 4);
}

#[test]
fn first_output_packet_streams_early_then_switches_to_steady_state_64_packets() {
    let mut ready64 = ready();
    put(
        &mut ready64,
        "native_mtp_output_packet_schema",
        Json::integer(1),
    );
    put(&mut ready64, "native_mtp_output_packet", Json::integer(64));

    let make_codec = || ServeCodec {
        render: Box::new(|request| {
            let messages = get(request, "messages").array().unwrap();
            Ok(RenderedChat {
                prompt: get(messages.last().unwrap(), "content")
                    .string()
                    .unwrap()
                    .to_vec(),
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        }),
        encode: Box::new(|text| Ok(text.prompt.iter().map(|&byte| u32::from(byte)).collect())),
        decode: Box::new(|ids| Ok(ids.iter().map(|&id| u8::try_from(id).unwrap()).collect())),
        admission: ResponseAdmission {
            vocabulary: 4096,
            im_end: 1000,
            end_of_text: 1001,
            im_start: 1002,
        },
    };

    assert!(
        ServeSession::new(
            make_codec(),
            SessionOptions {
                mtp_ready: ready(),
                mtp_first_output_packet: Some(16),
                ..SessionOptions::default()
            },
        )
        .is_err()
    );

    let mut session = ServeSession::new(
        make_codec(),
        SessionOptions {
            mtp_ready: ready64.clone(),
            mtp_first_output_packet: Some(8),
            ..SessionOptions::default()
        },
    )
    .unwrap();

    let request = Json::new_object([
        ("model", Json::from("tokspur")),
        (
            "messages",
            Json::Array(vec![Json::new_object([
                ("role", Json::from("user")),
                ("content", Json::from("hello")),
            ])]),
        ),
        ("max_tokens", Json::integer(80)),
        ("seed", Json::integer(44)),
        ("stream", Json::Bool(true)),
        ("enable_thinking", Json::Bool(false)),
    ]);

    let mut position = 0u64;
    let mut primes = 0u64;
    let mut sampling = Json::Null;
    let mut gen_widths = Vec::new();
    let streamed_content = Cell::new(0usize);
    let mut content_before_second_packet = 0usize;

    let mut chunk_cb = |value: &Json| {
        if let Some(choices) = value.find("choices").unwrap()
            && !choices.array().unwrap().is_empty()
            && choices.array().unwrap()[0]
                .find("delta")
                .unwrap()
                .unwrap()
                .find("content")
                .unwrap()
                .is_some()
        {
            streamed_content.set(streamed_content.get() + 1);
        }
        Ok(())
    };
    let mut check_cancelled = || false;
    let mut new_seed = || 44;
    let mut sink = ServeSink {
        chunk: Some(&mut chunk_cb),
        cancelled: &mut check_cancelled,
        new_seed: &mut new_seed,
        prefix_trace: None,
    };

    let mut target = |job: &Json, event: TokenCallback<'_>| -> Result<Json, ServeError> {
        let reset = get(job, "reset").boolean().unwrap();
        if reset {
            position = 0;
            sampling = get(job, "sampling").clone();
        }
        let inputs = get(job, "tokens").array().unwrap().len() as u64;
        let generate = number(job, "generate");
        let mut ids = Vec::new();
        let mut active = false;
        let mut remaining_after = 0u64;
        let mut gen_id = 0u64;
        if inputs > 0 {
            position += inputs;
            primes += inputs - u64::from(reset);
        } else {
            if generate == 64 && content_before_second_packet == 0 {
                content_before_second_packet = streamed_content.get();
            }
            gen_widths.push(generate);
            let gen_meta = get(job, "mtp_generation");
            gen_id = number(gen_meta, "id");
            let rem = number(gen_meta, "remaining");
            remaining_after = rem - generate;
            active = remaining_after > 0;
            for _ in 0..generate {
                position += 1;
                primes += 1;
                ids.push(Json::integer(u64::from(b'x')));
                event(&Json::new_object([
                    ("event", Json::from("token")),
                    ("id", Json::integer(u64::from(b'x'))),
                    ("committed_prefix", Json::integer(position)),
                    ("entropy_words", Json::integer(0)),
                ]))?;
            }
        }
        let mut sampled = sampling.clone();
        for (name, value) in [
            ("valid", Json::Bool(true)),
            ("entropy_prefix", Json::integer(position)),
            ("entropy_words", Json::integer(0)),
            ("seed", Json::integer(44)),
        ] {
            put(&mut sampled, name, value);
        }
        let mut done = ready64.clone();
        for (name, value) in [
            ("event", Json::from("done")),
            ("committed_prefix", Json::integer(position)),
            ("generated_ids", Json::Array(ids)),
            ("sampling", sampled),
            (
                "status",
                Json::from(if inputs > 0 { "prefilled" } else { "length" }),
            ),
            ("continuation_allowed", Json::Bool(true)),
            ("native_mtp_prime_calls", Json::integer(primes)),
            (
                "native_mtp_generation",
                Json::new_object([
                    ("id", Json::integer(gen_id)),
                    ("active", Json::Bool(active)),
                    ("remaining", Json::integer(remaining_after)),
                    ("pending", Json::Bool(false)),
                ]),
            ),
        ] {
            put(&mut done, name, value);
        }
        Ok(done)
    };

    let response = session
        .complete(&request, b"first-pkt-8", &mut target, &mut sink)
        .unwrap();
    assert_eq!(gen_widths, vec![8, 64, 8]);
    assert!(
        content_before_second_packet > 0,
        "first 8-token packet must flush SSE content chunks before starting steady-state 64-token packet"
    );
    let details = get(&response, "tokspur_private");
    assert_eq!(number(details, "native_mtp_first_output_packet"), 8);
    assert_eq!(number(details, "maximum_generation_width"), 64);
}
