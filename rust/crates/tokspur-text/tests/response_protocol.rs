//! Port of `native/tests/test_response_protocol.cpp`. Each test counts its checks as the C++ `check` and `refused` helpers do; together they make the C++ total of 584. Refusals also assert the exact C++ message.
//!
//! The C++ fixture is `ByteBpe` over 256 byte singletons plus `<|im_end|>`, `<|endoftext|>` and `<|im_start|>` (IDs 256-258) with no merges and an identity `TextProcessor`. `ByteBpe` is ported separately, so `tests/fixture` reproduces that fixture's encode/decode; the differential runner in `rust/tools/diff/chat-response` checks the protocol against the real C++ fixture.

// Test code: helpers outside #[test] functions unwrap as the tests do.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixture;

use std::cell::Cell;
use std::convert::Infallible;
use std::fmt;

use fixture::{Fixture, FixtureError};
use tokspur_json::{Json, parse_json};
use tokspur_text::response_protocol::{
    RESPONSE_MAXIMUM_ACCUMULATED_IDS, RESPONSE_MAXIMUM_IDS, RESPONSE_REQUEST_BYTES,
    ResponseAdmission, ResponseProtocolError, native_response_error, parse_native_response,
};

const FIXTURE: Fixture = Fixture::STANDARD;

fn decode(ids: &[u32]) -> Result<Vec<u8>, FixtureError> {
    FIXTURE.decode_bytes(ids)
}

const ADMISSION: ResponseAdmission = ResponseAdmission {
    vocabulary: Fixture::VOCABULARY,
    im_end: 256,
    end_of_text: 257,
    im_start: 258,
};

/// The C++ `checks` counter with `check` and `refused`.
struct Checks(usize);

impl Checks {
    fn check(&mut self, condition: bool, message: &str) {
        self.0 += 1;
        assert!(condition, "{message}");
    }

    fn refused<E: fmt::Display>(
        &mut self,
        result: Result<Json, ResponseProtocolError<E>>,
        message: &str,
    ) {
        self.0 += 1;
        match result {
            Ok(_) => panic!("invalid native response protocol accepted; expected {message:?}"),
            Err(error) => assert_eq!(error.to_string(), message),
        }
    }
}

fn manifest() -> Json {
    parse_json(
        br#"[{"name":"read_file","parameters":[{"name":"path","type":"string","required":true}]},{"name":"run_tests","parameters":[]}]"#,
    )
    .unwrap()
}

fn envelope_with(ids: &[u32], finish: &str, thinking: bool, schema: Json, stop: Json) -> Json {
    let input = ids.iter().map(|&id| Json::integer(u64::from(id))).collect();
    Json::new_object([
        ("ids", Json::Array(input)),
        ("tools", schema),
        ("initial_thinking", Json::Bool(thinking)),
        ("finish_reason", Json::from(finish)),
        ("stop_id", stop),
    ])
}

/// `envelope(ids, finish)` with the C++ defaults.
fn envelope(ids: &[u32], finish: &str) -> Json {
    envelope_with(ids, finish, false, manifest(), Json::integer(256))
}

/// `envelope(ids, "length", false, manifest(), Json())`.
fn pending(ids: &[u32]) -> Json {
    envelope_with(ids, "length", false, manifest(), Json::Null)
}

fn parse(input: &Json) -> Result<Json, ResponseProtocolError<FixtureError>> {
    parse_native_response(input, ADMISSION, Some(&decode), RESPONSE_MAXIMUM_IDS)
}

fn get<'a>(value: &'a Json, name: &str) -> &'a Json {
    value.find(name).unwrap().unwrap()
}

fn call() -> String {
    "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/日本🙂.rs\n</parameter>\n</function>\n</tool_call>"
        .to_owned()
}

fn all() -> Vec<u32> {
    let batch = format!(
        "Inspecting café\n{}\n<tool_call><function=run_tests>\n</function></tool_call>",
        call()
    );
    FIXTURE.encode(format!("{batch}<|im_end|>").as_bytes())
}

#[test]
fn stopped_response_publishes_calls_and_every_length_prefix_withholds_them() {
    let mut checks = Checks(0);
    let all = all();
    let complete = parse(&envelope(&all, "stop")).unwrap();
    checks.check(
        get(&complete, "status").string().unwrap() == b"complete"
            && get(&complete, "complete").boolean().unwrap(),
        "valid stopped response not complete",
    );
    checks.check(
        get(&complete, "tool_calls").array().unwrap().len() == 2
            && !get(&complete, "tools_executed").boolean().unwrap(),
        "tool count or execution authority",
    );
    let first = &get(&complete, "tool_calls").array().unwrap()[0];
    checks.check(
        get(first, "ordinal").unsigned_integer().unwrap() == 0
            && get(first, "type").string().unwrap() == b"function",
        "call identity fields",
    );
    checks.check(
        get(get(get(first, "function"), "arguments"), "path")
            .string()
            .unwrap()
            == "src/日本🙂.rs".as_bytes(),
        "raw Unicode argument changed",
    );
    checks.check(
        get(&complete, "content").string().unwrap() == "Inspecting café\n".as_bytes()
            && get(&complete, "reasoning_content")
                .string()
                .unwrap()
                .is_empty(),
        "visible/hidden split",
    );
    // Every token boundary uses the same accumulated IDs, never separately replaced token strings. Some boundaries split UTF-8 or XML-like markers.
    for split in 0..all.len() {
        let partial = parse(&pending(&all[..split])).unwrap();
        checks.check(
            get(&partial, "status").string().unwrap() == b"incomplete"
                && !get(&partial, "complete").boolean().unwrap(),
            "length prefix incorrectly completed",
        );
        checks.check(
            get(&partial, "tool_calls").array().unwrap().is_empty(),
            "length prefix released executable call",
        );
        checks.check(
            parse(&envelope(&all, "stop")).unwrap().dump().unwrap() == complete.dump().unwrap(),
            "continuation accumulation changes final parse",
        );
    }
    assert_eq!(checks.0, 5 + 3 * all.len());
    assert_eq!(checks.0, 542, "C++ complete and split-loop checks");
}

#[test]
fn only_terminal_admission_releases_calls_and_thinking_stays_inert() {
    let mut checks = Checks(0);
    // Even a syntactically complete call is pending without a verified terminal.
    let capped = parse(&pending(&FIXTURE.encode(call().as_bytes()))).unwrap();
    checks.check(
        get(&capped, "tool_calls").array().unwrap().is_empty(),
        "complete-looking capped call was executed",
    );
    let hidden = format!("considered {} but this is inert.", call());
    let thought = parse(&envelope_with(
        &FIXTURE.encode(format!("{hidden}</think>\n\n{}<|im_end|>", call()).as_bytes()),
        "stop",
        true,
        manifest(),
        Json::integer(256),
    ))
    .unwrap();
    checks.check(
        get(&thought, "reasoning_content").string().unwrap() == hidden.as_bytes()
            && get(&thought, "tool_calls").array().unwrap().len() == 1,
        "thinking tool text escaped into callable record",
    );
    let plain = parse(&envelope_with(
        &FIXTURE.encode("finished café 🦉<|im_end|>".as_bytes()),
        "stop",
        false,
        Json::Array(Vec::new()),
        Json::integer(256),
    ))
    .unwrap();
    checks.check(
        get(&plain, "content").string().unwrap() == "finished café 🦉".as_bytes()
            && get(&plain, "tool_calls").array().unwrap().is_empty(),
        "plain final response",
    );
    assert_eq!(checks.0, 3);
}

#[test]
fn typed_manifest_maps_each_parameter_type() {
    let mut checks = Checks(0);
    let typed = parse_json(
        br#"[{"name":"typed","parameters":[{"name":"s","type":"string","required":true},{"name":"b","type":"boolean","required":true},{"name":"n","type":"number","required":true},{"name":"o","type":"object","required":true},{"name":"a","type":"array","required":true},{"name":"z","type":"null","required":true}]}]"#,
    )
    .unwrap();
    let typed_body = FIXTURE.encode(
        b"<tool_call><function=typed>\n<parameter=s>\ntrue\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n<parameter=n>\n-2.5\n</parameter>\n<parameter=o>\n{\"x\":1}\n</parameter>\n<parameter=a>\n[1,true]\n</parameter>\n<parameter=z>\nnull\n</parameter>\n</function></tool_call><|im_end|>",
    );
    let typed_result = parse(&envelope_with(
        &typed_body,
        "stop",
        false,
        typed,
        Json::integer(256),
    ))
    .unwrap();
    let arguments = get(
        get(
            &get(&typed_result, "tool_calls").array().unwrap()[0],
            "function",
        ),
        "arguments",
    );
    checks.check(
        get(arguments, "s").string().unwrap() == b"true" && get(arguments, "b").boolean().unwrap(),
        "typed manifest string/boolean mapping",
    );
    checks.check(
        get(arguments, "n").number_lexeme().unwrap() == "-2.5"
            && get(get(arguments, "o"), "x").unsigned_integer().unwrap() == 1
            && get(arguments, "a").array().unwrap().len() == 2
            && get(arguments, "z").is_null(),
        "typed manifest structured mapping",
    );
    assert_eq!(checks.0, 2);
}

#[test]
fn invalid_envelopes_and_frames_are_refused() {
    let mut checks = Checks(0);
    let all = all();
    let encode = |text: &str| FIXTURE.encode(text.as_bytes());
    let invalid = [
        (
            envelope(&[], "stop"),
            "stopped response lacks exact final stop token",
        ),
        (
            envelope_with(&all, "stop", false, manifest(), Json::Null),
            "response stop ID is not an admitted terminal token",
        ),
        (
            envelope_with(&all, "stop", false, manifest(), Json::integer(257)),
            "earlier or unacknowledged response terminal token",
        ),
        (
            pending(&all),
            "earlier or unacknowledged response terminal token",
        ),
        (
            envelope_with(&all, "stop", false, manifest(), Json::integer(999_999)),
            "response stop ID is not an admitted terminal token",
        ),
        (
            envelope(&encode(&call()), "stop"),
            "stopped response lacks exact final stop token",
        ),
        (envelope(&all, "unknown"), "unknown response finish reason"),
        (
            envelope(&encode("x<|endoftext|>"), "stop"),
            "earlier or unacknowledged response terminal token",
        ),
        (
            envelope(&encode("<|im_start|>x<|im_end|>"), "stop"),
            "unexpected response terminal or role marker",
        ),
        (
            envelope(&encode("x<|im_end|>y<|im_end|>"), "stop"),
            "earlier or unacknowledged response terminal token",
        ),
        (
            envelope(&encode(&format!("{}unexpected<|im_end|>", call())), "stop"),
            "invalid response bytes or frames",
        ),
        (
            envelope(&encode("<think>unfinished<|im_end|>"), "stop"),
            "incomplete or invalid stopped response",
        ),
        (
            envelope(
                &encode("<tool_call><function=read_file></function></tool_call><|im_end|>"),
                "stop",
            ),
            "invalid response bytes or frames",
        ),
        (
            envelope(
                &encode("<tool_call><function=unknown></function></tool_call><|im_end|>"),
                "stop",
            ),
            "invalid response bytes or frames",
        ),
    ];
    for (input, message) in &invalid {
        checks.refused(parse(input), message);
    }
    let mut earlier = all.clone();
    earlier.insert(0, ADMISSION.im_end);
    checks.refused(
        parse(&envelope(&earlier, "stop")),
        "earlier or unacknowledged response terminal token",
    );
    checks.refused(
        parse(&pending(&[ADMISSION.vocabulary])),
        "response token outside vocabulary",
    );
    checks.refused(
        parse(&pending(&vec![65; RESPONSE_MAXIMUM_IDS + 1])),
        "response ID count limit",
    );
    let accumulated = pending(&vec![65; RESPONSE_MAXIMUM_ACCUMULATED_IDS]);
    let decode = |ids: &[u32]| FIXTURE.decode_bytes(ids);
    let admitted = parse_native_response(
        &accumulated,
        ADMISSION,
        Some(&decode),
        RESPONSE_MAXIMUM_ACCUMULATED_IDS,
    )
    .unwrap();
    assert_eq!(
        get(&admitted, "content").string().unwrap().len(),
        RESPONSE_MAXIMUM_ACCUMULATED_IDS
    );
    for bound in [0, RESPONSE_MAXIMUM_ACCUMULATED_IDS + 1] {
        assert!(parse_native_response(&accumulated, ADMISSION, Some(&decode), bound).is_err());
    }
    assert_eq!(checks.0, 17);
}

#[test]
fn invalid_manifests_fields_and_admissions_are_refused() {
    let mut checks = Checks(0);
    let all = all();
    for (schema, message) in [
        (
            r#"[{"name":"x","parameters":[],"extra":1}]"#,
            "unknown response protocol field",
        ),
        (
            r#"[{"name":"x","parameters":[{"name":"p","type":"string"}]}]"#,
            "missing explicit response field",
        ),
        (
            r#"[{"name":"x","parameters":[{"name":"p","type":"string","required":true,"extra":1}]}]"#,
            "unknown response protocol field",
        ),
        (
            r#"[{"name":"x","parameters":[{"name":"p","type":"integer","required":true}]}]"#,
            "unsupported response parameter type",
        ),
        (
            r#"[{"name":"x","parameters":[{"name":"p","type":"string","required":"true"}]}]"#,
            "expected JSON boolean",
        ),
        (
            r#"[{"name":"x","parameters":[]},{"name":"x","parameters":[]}]"#,
            "duplicate tool or too many response parameters",
        ),
        (
            r#"[{"name":"bad name","parameters":[]}]"#,
            "unsupported response function name",
        ),
    ] {
        let schema = parse_json(schema.as_bytes()).unwrap();
        checks.refused(
            parse(&envelope_with(
                &all,
                "stop",
                false,
                schema,
                Json::integer(256),
            )),
            message,
        );
    }
    let Json::Object(mut extra) = envelope(&all, "stop") else {
        panic!("envelope is an object")
    };
    extra.push((b"unknown".to_vec(), Json::Bool(true)));
    checks.refused(
        parse(&Json::Object(extra)),
        "unknown response protocol field",
    );
    checks.refused(
        parse(&parse_json(br#"{"ids":[]}"#).unwrap()),
        "missing explicit response field",
    );
    for invalid in [
        ResponseAdmission {
            vocabulary: 0,
            im_end: 0,
            end_of_text: 1,
            im_start: 2,
        },
        ResponseAdmission {
            vocabulary: 259,
            im_end: 256,
            end_of_text: 256,
            im_start: 258,
        },
        ResponseAdmission {
            vocabulary: 259,
            im_end: 256,
            end_of_text: 257,
            im_start: 259,
        },
    ] {
        checks.refused(
            parse_native_response(
                &envelope(&all, "stop"),
                invalid,
                Some(&decode),
                RESPONSE_MAXIMUM_IDS,
            ),
            "invalid tokenizer-derived response admission",
        );
    }
    assert_eq!(checks.0, 12);
}

#[test]
fn decoder_identity_bounds_and_utf8_are_enforced() {
    let mut checks = Checks(0);
    let all = all();
    let encode = |bytes: &[u8]| FIXTURE.encode(bytes);
    let decoded = Cell::new(0usize);
    let counting = |ids: &[u32]| {
        decoded.set(decoded.get() + 1);
        FIXTURE.decode_bytes(ids)
    };
    let errored = parse_native_response(
        &envelope_with(
            &encode(call().as_bytes()),
            "error",
            false,
            manifest(),
            Json::Null,
        ),
        ADMISSION,
        Some(&counting),
        RESPONSE_MAXIMUM_IDS,
    )
    .unwrap();
    checks.check(
        get(&errored, "status").string().unwrap() == b"error"
            && get(&errored, "tool_calls").array().unwrap().is_empty()
            && decoded.get() == 0,
        "upstream error decoded or released calls",
    );
    let wrong_terminal = |_: &[u32]| Ok::<_, Infallible>(b"wrong terminal".to_vec());
    checks.refused(
        parse_native_response(
            &envelope(&all, "stop"),
            ADMISSION,
            Some(&wrong_terminal),
            RESPONSE_MAXIMUM_IDS,
        ),
        "response decoder/terminal identity mismatch",
    );
    let oversized = |_: &[u32]| Ok::<_, Infallible>(vec![b'x'; RESPONSE_REQUEST_BYTES + 1]);
    checks.refused(
        parse_native_response(
            &pending(&[]),
            ADMISSION,
            Some(&oversized),
            RESPONSE_MAXIMUM_IDS,
        ),
        "decoded response byte limit",
    );
    let invalid_utf8 = encode(b"x\xc0\x80<|im_end|>");
    checks.refused(
        parse(&envelope(&invalid_utf8, "stop")),
        "invalid response bytes or frames",
    );
    let invalid_utf8 = encode(b"x\xc3<|im_end|>");
    checks.refused(
        parse(&envelope(&invalid_utf8, "stop")),
        "incomplete or invalid stopped response",
    );
    let partial = encode(b"x\xc3");
    checks.check(
        FIXTURE.decode_bytes(&partial).unwrap() == b"x\xc3"
            && FIXTURE.decode_text(&partial).unwrap() != FIXTURE.decode_bytes(&partial).unwrap(),
        "fixture does not expose replacement corruption",
    );
    checks.check(
        get(&parse(&pending(&partial)).unwrap(), "status")
            .string()
            .unwrap()
            == b"incomplete",
        "valid partial UTF-8 was not retained as incomplete",
    );
    let failure = native_response_error("test refusal");
    checks.check(
        get(&failure, "tool_calls").array().unwrap().is_empty()
            && !get(&failure, "complete").boolean().unwrap()
            && !get(&failure, "tools_executed").boolean().unwrap(),
        "error envelope contains execution authority",
    );
    assert_eq!(checks.0, 8);
}

#[test]
fn missing_decoder_is_refused_only_when_bytes_are_needed() {
    // The C++ refuses an empty `std::function`; `None` is its Rust spelling.
    let refused = parse_native_response::<FixtureError>(
        &envelope(&all(), "stop"),
        ADMISSION,
        None,
        RESPONSE_MAXIMUM_IDS,
    );
    assert_eq!(
        refused.unwrap_err().to_string(),
        "missing raw response decoder"
    );
    let errored = parse_native_response::<FixtureError>(
        &envelope_with(&[], "error", false, manifest(), Json::Null),
        ADMISSION,
        None,
        RESPONSE_MAXIMUM_IDS,
    )
    .unwrap();
    assert_eq!(
        get(&errored, "message").string().unwrap(),
        b"target reported an error; response is not executable"
    );
}

#[test]
fn error_envelope_layout_and_message_bound() {
    let failure = native_response_error("test refusal");
    assert_eq!(
        failure.dump().unwrap(),
        br#"{"event":"response","status":"error","finish_reason":"error","complete":false,"content":"","reasoning_content":"","tool_calls":[],"tools_executed":false,"message":"test refusal"}"#
    );
    let long = native_response_error([b'm'; 3000]);
    assert_eq!(get(&long, "message").string().unwrap().len(), 2048);
}
