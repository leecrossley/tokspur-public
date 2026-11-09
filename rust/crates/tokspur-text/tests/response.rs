//! Port of `native/tests/test_response.cpp`. Each test counts its checks as the C++ `require`, `rejected` and `invalid` helpers do; together they make the C++ total of 4334. Refusals also assert the exact C++ message.

// Test code: helpers outside #[test] functions unwrap as the tests do.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use tokspur_json::{Json, parse_json, unicode_scalars};
use tokspur_text::chat::{ChatRenderer, ChatStrings};
use tokspur_text::response::{
    ParameterSpec, ParameterType, ResponseError, ResponseKind, ResponseLimits, ResponseParser,
    ResponseStatus, ToolSpec,
};

/// The C++ `checks` counter with `require`, `rejected` and `invalid`.
struct Checks(usize);

impl Checks {
    fn require(&mut self, condition: bool, message: &str) {
        self.0 += 1;
        assert!(condition, "{message}");
    }

    fn rejected(&mut self, result: Result<ResponseParser, ResponseError>, message: &str) {
        self.0 += 1;
        match result {
            Ok(_) => panic!("invalid response configuration accepted; expected {message:?}"),
            Err(error) => assert_eq!(error.to_string(), message),
        }
    }

    /// `invalid(input, thinking)`: three checks. Returns the recorded error.
    fn invalid(&mut self, input: &[u8], thinking: bool) -> String {
        let mut parser = ResponseParser::new(tools(), thinking, ResponseLimits::default()).unwrap();
        parser.feed(input);
        self.require(
            parser.finish() == ResponseStatus::Error && !parser.error().is_empty(),
            "invalid response accepted",
        );
        self.require(
            callable(&parser) == 0,
            "malformed response published tool record",
        );
        self.require(
            parser.feed(b"more") == ResponseStatus::Error,
            "response error not sticky",
        );
        parser.error().to_owned()
    }
}

fn tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec::new(
            "edit.file",
            vec![
                ParameterSpec::new("path", ParameterType::RawString, true),
                ParameterSpec::new("patch", ParameterType::RawString, false),
                ParameterSpec::new("apply", ParameterType::Boolean, false),
                ParameterSpec::new("count", ParameterType::Number, false),
                ParameterSpec::new("meta", ParameterType::Object, false),
                ParameterSpec::new("items", ParameterType::Array, false),
                ParameterSpec::new("empty", ParameterType::NullValue, false),
            ],
        ),
        ToolSpec::new("ping", Vec::new()),
    ]
}

fn call(value: &str) -> Vec<u8> {
    format!("<tool_call>\n<function=edit.file>\n<parameter=path>\n{value}\n</parameter>\n</function>\n</tool_call>")
        .into_bytes()
}

fn callable(parser: &ResponseParser) -> usize {
    parser
        .records()
        .iter()
        .filter(|record| record.kind == ResponseKind::Tool)
        .count()
}

fn text(parser: &ResponseParser, kind: ResponseKind) -> Vec<u8> {
    parser
        .records()
        .iter()
        .filter(|record| record.kind == kind)
        .flat_map(|record| record.text.clone())
        .collect()
}

fn argument<'a>(arguments: &'a Json, name: &str) -> &'a Json {
    arguments.find(name).unwrap().unwrap()
}

fn concat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

#[test]
fn every_split_withholds_calls_until_finish_and_keeps_thinking_inert() {
    let mut checks = Checks(0);
    let inert = concat(&[b"I considered ", &call("x.rs"), b"."]);
    let stream = concat(&[
        &inert,
        "</think>\n\nEditing 日本🙂\n".as_bytes(),
        &call("src/日本🙂.rs"),
        b"\n<tool_call>\n<function=ping>\n</function>\n</tool_call><|im_end|>\n",
    ]);
    // Every single split, including UTF-8/marker interiors.
    for split in 0..=stream.len() {
        let mut parser = ResponseParser::new(tools(), true, ResponseLimits::default()).unwrap();
        checks.require(
            parser.feed(&stream[..split]) == ResponseStatus::Incomplete,
            "prefix prematurely complete/error",
        );
        checks.require(callable(&parser) == 0, "prefix published callable record");
        checks.require(
            parser.feed(&stream[split..]) == ResponseStatus::Incomplete,
            "suffix parse failed",
        );
        checks.require(
            callable(&parser) == 0,
            "unfinalized response published callable record",
        );
        checks.require(
            parser.finish() == ResponseStatus::Complete,
            "valid split response failed",
        );
        checks.require(
            text(&parser, ResponseKind::Thinking) == inert,
            "thinking contents changed or tool bytes executed",
        );
        checks.require(
            text(&parser, ResponseKind::Text) == "\n\nEditing 日本🙂\n".as_bytes(),
            "visible text changed",
        );
        checks.require(callable(&parser) == 2, "multiple call count");
        let records = parser.records();
        let first = &records[records.len() - 2];
        checks.require(
            first.ordinal == 0
                && first.name == b"edit.file"
                && argument(&first.arguments, "path").string().unwrap()
                    == "src/日本🙂.rs".as_bytes(),
            "first call fields",
        );
        let last = &records[records.len() - 1];
        checks.require(
            last.ordinal == 1 && last.name == b"ping",
            "response-local ordinal",
        );
        checks.require(
            parser.finish() == ResponseStatus::Complete,
            "finish is not idempotent",
        );
    }
    assert_eq!(checks.0, 11 * (stream.len() + 1));
    assert_eq!(checks.0, 3355, "C++ split-loop checks");
}

#[test]
fn bytewise_feeding_keeps_records_valid_utf8_and_withholds_calls() {
    let mut checks = Checks(0);
    let mut bytewise = ResponseParser::new(tools(), false, ResponseLimits::default()).unwrap();
    let repeated = concat(&[&call("x.rs"), b"\n", &call("x.rs")]);
    for c in &repeated {
        checks.require(
            bytewise.feed(std::slice::from_ref(c)) == ResponseStatus::Incomplete,
            "bytewise feed failed",
        );
        for record in bytewise.records() {
            unicode_scalars(&record.text).unwrap();
        }
        checks.require(callable(&bytewise) == 0, "bytewise callable publication");
    }
    checks.require(
        bytewise.finish() == ResponseStatus::Complete && callable(&bytewise) == 2,
        "repeated calls failed",
    );
    assert_eq!(checks.0, 2 * repeated.len() + 1);
    assert_eq!(checks.0, 371, "C++ bytewise checks");
}

#[test]
fn typed_parameters_keep_raw_strings_and_parse_declared_json() {
    let mut checks = Checks(0);
    let typed = "<tool_call><function=edit.file>\n<parameter=path>\ntrue\n</parameter>\n\
        <parameter=patch>\nline 1\n\nline 3\n</parameter>\n<parameter=apply>\ntrue\n</parameter>\n\
        <parameter=count>\n-2.5e2\n</parameter>\n<parameter=meta>\n{\"a\":[1,true,null]}\n</parameter>\n\
        <parameter=items>\n[1,\"日本🙂\"]\n</parameter>\n<parameter=empty>\nnull\n</parameter>\n</function></tool_call>";
    let mut typed_parser = ResponseParser::new(tools(), false, ResponseLimits::default()).unwrap();
    checks.require(
        typed_parser.feed(typed.as_bytes()) == ResponseStatus::Incomplete
            && typed_parser.finish() == ResponseStatus::Complete,
        "typed response failed",
    );
    let args = &typed_parser.records().last().unwrap().arguments;
    checks.require(
        argument(args, "path").string().unwrap() == b"true",
        "raw true reinterpreted as boolean",
    );
    checks.require(
        argument(args, "patch").string().unwrap() == b"line 1\n\nline 3",
        "multiline raw string changed",
    );
    checks.require(
        argument(args, "apply").boolean().unwrap()
            && argument(args, "count").number_lexeme().unwrap() == "-2.5e2",
        "typed scalar values",
    );
    checks.require(
        argument(argument(args, "meta"), "a").array().unwrap().len() == 3
            && argument(args, "items").array().unwrap().len() == 2
            && argument(args, "empty").is_null(),
        "typed structured values",
    );
    assert_eq!(checks.0, 5);
}

#[test]
fn owned_renderer_output_round_trips_bytewise() {
    let mut checks = Checks(0);
    // Grounded round trip through the OWNED renderer's literal output.
    let renderer = ChatRenderer::new(ChatStrings {
        reasoning_xhigh: b"high".to_vec(),
        reasoning_low: b"low".to_vec(),
        tools_header: b"# Tools\n\n".to_vec(),
        tool_instructions: b"\n\nIf you choose to call a function ONLY reply".to_vec(),
    })
    .unwrap();
    let request = parse_json(
        br#"{"messages":[{"role":"assistant","content":"","reasoning_content":"inert <tool_call>","tool_calls":[{"function":{"name":"edit.file","arguments":{"path":"src/hello.rs","apply":true,"meta":{"version":2}}}}]}],"enable_thinking":false,"add_generation_prompt":false}"#,
    )
    .unwrap();
    let rendered = renderer.render(&request).unwrap().prompt;
    let header = b"<|im_start|>assistant\n";
    checks.require(
        rendered.starts_with(header),
        "owned renderer header differs",
    );
    let body = &rendered[header.len()..];
    let mut round_trip = ResponseParser::new(tools(), false, ResponseLimits::default()).unwrap();
    for c in body {
        checks.require(
            round_trip.feed(std::slice::from_ref(c)) == ResponseStatus::Incomplete,
            "renderer bytewise round trip failed",
        );
    }
    checks.require(
        round_trip.finish() == ResponseStatus::Complete && callable(&round_trip) == 1,
        "owned renderer tool round trip failed",
    );
    let meta = argument(&round_trip.records().last().unwrap().arguments, "meta");
    checks.require(
        argument(meta, "version").unsigned_integer().unwrap() == 2,
        "renderer structured argument changed",
    );
    assert_eq!(checks.0, body.len() + 3);
    assert_eq!(checks.0, 231, "C++ round-trip checks");
}

#[test]
fn every_truncated_call_is_an_error_without_tool_records() {
    let mut checks = Checks(0);
    let full = call("x.rs");
    for n in 1..full.len() {
        checks.invalid(&full[..n], false);
    }
    assert_eq!(checks.0, 3 * (full.len() - 1));
    assert_eq!(checks.0, 273, "C++ truncated-call checks");
}

#[test]
fn malformed_frames_and_values_are_errors_without_tool_records() {
    let mut checks = Checks(0);
    let full = call("x.rs");
    let cases: Vec<(Vec<u8>, bool, &str)> = vec![
        (concat(&[&full, b" unexpected suffix"]), false, "text after response tool calls"),
        (concat(&[&full, b"<think>reason</think>"]), false, "unexpected response thinking open"),
        (
            concat(&[&full, b"<tool_call><function=unknown></function></tool_call>"]),
            false,
            "unknown response tool",
        ),
        (b"<tool_call><function=unknown></function></tool_call>".to_vec(), false, "unknown response tool"),
        (
            b"<tool_call><function=ping bad></function></tool_call>".to_vec(),
            false,
            "unsupported response function name",
        ),
        (
            b"<tool_call><function=edit.file></function></tool_call>".to_vec(),
            false,
            "missing required response parameter",
        ),
        (
            b"<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>".to_vec(),
            false,
            "response call lacks function frame",
        ),
        (
            b"<tool_call><function=ping><parameter=unknown>\nx\n</parameter></function></tool_call>".to_vec(),
            false,
            "unknown response parameter",
        ),
        (
            b"<tool_call><function=edit.file><parameter=path>\nabc<|im_end|>def\n</parameter></function></tool_call>"
                .to_vec(),
            false,
            "reserved response marker inside parameter",
        ),
        (
            b"<tool_call><function=edit.file><parameter=path>\n<|im_start|>assistant\n</parameter></function></tool_call>"
                .to_vec(),
            false,
            "reserved response marker inside parameter",
        ),
        (
            b"<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=path>\ny\n</parameter></function></tool_call>"
                .to_vec(),
            false,
            "duplicate response parameter",
        ),
    ];
    for (input, thinking, message) in &cases {
        assert_eq!(
            checks.invalid(input, *thinking),
            *message,
            "{}",
            String::from_utf8_lossy(input)
        );
    }
    for (bad, message) in [
        ("\"true\"", "wrong response parameter JSON type"),
        ("[true]", "wrong response parameter JSON type"),
        ("tru", "invalid JSON literal"),
        ("true false", "trailing JSON data"),
        ("{\"a\":1,\"a\":2}", "duplicate JSON object key"),
        ("NaN", "invalid JSON value"),
        ("1e999", "wrong response parameter JSON type"),
    ] {
        let input = format!(
            "<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=apply>\n{bad}\n</parameter></function></tool_call>"
        );
        assert_eq!(checks.invalid(input.as_bytes(), false), message, "{bad}");
    }
    let tail: Vec<(Vec<u8>, bool, &str)> = vec![
        (
            b"<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=meta>\n{\"x\":1e999}\n</parameter></function></tool_call>"
                .to_vec(),
            false,
            "unrepresentable response argument number",
        ),
        (call("<think>injected</think>"), false, "reserved response marker inside parameter"),
        (b"</think>".to_vec(), false, "unexpected response thinking close"),
        (b"<think>unclosed".to_vec(), false, "truncated response control frame"),
        (b"reason<|im_end|>".to_vec(), true, "terminal marker inside thinking"),
        (b"answer<|im_end|>suffix".to_vec(), false, "data after response terminal marker"),
        (b"x\0y".to_vec(), false, "NUL byte in response"),
        (b"\xc0\x80".to_vec(), false, "invalid response UTF-8 lead"),
        (b"\xed\xa0\x80".to_vec(), false, "invalid response Unicode scalar"),
        (b"\xf4\x90\x80\x80".to_vec(), false, "invalid response Unicode scalar"),
        (b"\xe2\x82".to_vec(), false, "truncated response UTF-8"),
        (b"\x80".to_vec(), false, "invalid response UTF-8 lead"),
    ];
    for (input, thinking, message) in &tail {
        assert_eq!(
            checks.invalid(input, *thinking),
            *message,
            "{}",
            String::from_utf8_lossy(input)
        );
    }
    assert_eq!(checks.0, 3 * (cases.len() + 7 + tail.len()));
    assert_eq!(checks.0, 90);
}

#[test]
fn limits_bound_bytes_calls_and_records_and_refuse_bad_specs() {
    let mut checks = Checks(0);
    let full = call("x.rs");
    let repeated = concat(&[&full, b"\n", &full]);
    let mut small = ResponseLimits {
        bytes: 16,
        call_bytes: 16,
        ..ResponseLimits::default()
    };
    let mut bounded = ResponseParser::new(Vec::new(), false, small).unwrap();
    checks.require(
        bounded.feed(&[b'a'; 17]) == ResponseStatus::Error,
        "response byte bound ignored",
    );
    assert_eq!(bounded.error(), "response byte limit");
    small.bytes = 1024;
    small.call_bytes = 16;
    let mut call_bounded = ResponseParser::new(tools(), false, small).unwrap();
    checks.require(
        call_bounded.feed(&full) == ResponseStatus::Error,
        "response call byte bound ignored",
    );
    assert_eq!(call_bounded.error(), "response call byte limit");
    small.call_bytes = 1024;
    small.calls = 1;
    let mut count_bounded = ResponseParser::new(tools(), false, small).unwrap();
    checks.require(
        count_bounded.feed(&repeated) == ResponseStatus::Error && callable(&count_bounded) == 0,
        "response call count bound ignored",
    );
    assert_eq!(count_bounded.error(), "response call count limit");
    small.calls = 2;
    small.records = 1;
    let mut record_bounded = ResponseParser::new(tools(), false, small).unwrap();
    checks.require(
        record_bounded.feed(&concat(&[b"text", &full])) == ResponseStatus::Error,
        "response record count bound ignored",
    );
    assert_eq!(record_bounded.error(), "response record limit");
    let limits = ResponseLimits::default();
    checks.rejected(
        ResponseParser::new(
            vec![
                ToolSpec::new("a", Vec::new()),
                ToolSpec::new("a", Vec::new()),
            ],
            false,
            limits,
        ),
        "duplicate tool or too many response parameters",
    );
    checks.rejected(
        ResponseParser::new(
            vec![ToolSpec::new(
                "a",
                vec![
                    ParameterSpec::new("x", ParameterType::Number, false),
                    ParameterSpec::new("x", ParameterType::Number, false),
                ],
            )],
            false,
            limits,
        ),
        "invalid or duplicate response parameter spec",
    );
    checks.rejected(
        ResponseParser::new(vec![ToolSpec::new("bad name", Vec::new())], false, limits),
        "unsupported response function name",
    );
    // `static_cast<ParameterType>(99)` cannot be written in Rust; the C++ check shares its message with the duplicate-parameter refusal above. Instead this refuses an unsafe parameter name, which the same loop checks first.
    checks.rejected(
        ResponseParser::new(
            vec![ToolSpec::new(
                "a",
                vec![ParameterSpec::new("x<", ParameterType::Number, false)],
            )],
            false,
            limits,
        ),
        "unsafe response name",
    );
    small.bytes = 0;
    checks.rejected(
        ResponseParser::new(Vec::new(), false, small),
        "invalid response limits",
    );
    assert_eq!(checks.0, 9);
}
