//! Port of `native/tests/test_chat.cpp` (`json_format`, `rendering`, `data_literals`). Each test counts its checks as the C++ `require` and `rejected` helpers do; together they make the C++ total of 39. Refusals also assert the exact C++ message.

// Test code: helpers outside #[test] functions unwrap as the tests do.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use tokspur_json::{Json, Kind, parse_json};
use tokspur_text::chat::{
    ChatError, ChatRenderer, ChatStrings, chat_trim, extract_flash_next_literals, template_json,
};

/// The C++ `checks` counter with `require` and `rejected`.
struct Checks(usize);

impl Checks {
    fn require(&mut self, condition: bool, message: &str) {
        self.0 += 1;
        assert!(condition, "{message}");
    }

    fn rejected<T>(&mut self, result: Result<T, ChatError>, message: &str) {
        self.0 += 1;
        match result {
            Ok(_) => panic!("invalid chat request accepted; expected {message:?}"),
            Err(error) => assert_eq!(error.to_string(), message),
        }
    }
}

fn parse(text: &str) -> Json {
    parse_json(text.as_bytes()).unwrap()
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn renderer() -> ChatRenderer {
    ChatRenderer::new(ChatStrings {
        reasoning_xhigh: b"XHIGH".to_vec(),
        reasoning_low: b"LOW".to_vec(),
        tools_header: b"# Tools\n\n<tools>".to_vec(),
        tool_instructions: b"\n\nINSTRUCTIONS".to_vec(),
    })
    .unwrap()
}

fn prompt(request: &str) -> Vec<u8> {
    renderer().render(&parse(request)).unwrap().prompt
}

#[test]
fn json_format_matches_python_float_repr_and_escaping() {
    let mut checks = Checks(0);
    for (source, expected) in [
        ("-0", "0"),
        ("0.0", "0.0"),
        ("-0.0", "-0.0"),
        ("1e0", "1.0"),
        ("1e-4", "0.0001"),
        ("1e-5", "1e-05"),
        ("1e15", "1000000000000000.0"),
        ("1e16", "1e+16"),
        ("-1.25e2", "-125.0"),
        ("5e-324", "5e-324"),
        ("1e-9999", "0.0"),
        ("-1e-9999", "-0.0"),
        (
            "123456789012345678901234567890",
            "123456789012345678901234567890",
        ),
    ] {
        checks.require(
            template_json(&parse(source)).unwrap() == expected.as_bytes(),
            "Python-style JSON number mismatch",
        );
    }
    let value = parse(r#"{"z":[true,null,"<é>&\n\t\u0000"],"a":1.0}"#);
    checks.require(
        template_json(&value).unwrap()
            == "{\"z\": [true, null, \"<é>&\\n\\t\\u0000\"], \"a\": 1.0}".as_bytes(),
        "template JSON escaping/order/separators",
    );
    checks.rejected(
        template_json(&parse("1e9999")),
        "nonfinite JSON number in chat",
    );
    assert_eq!(checks.0, 15);
}

#[test]
fn rendering_plain_prompt_with_thinking_disabled() {
    let mut checks = Checks(0);
    checks.require(
        prompt(r#"{"messages":[{"role":"user","content":" hi "}],"enable_thinking":false}"#)
            == b"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "plain disabled-thinking prompt",
    );
    assert_eq!(checks.0, 1);
}

#[test]
fn rendering_merges_leading_system_and_developer_messages() {
    let mut checks = Checks(0);
    checks.require(
        prompt(
            r#"{"messages":[{"role":"system","content":" A "},{"role":"developer","content":[{"text":" B "}]},{"role":"user","content":null}],"reasoning_effort":"low","add_generation_prompt":false}"#,
        ) == b"<|im_start|>system\nLOW\n\nA\nB<|im_end|>\n<|im_start|>user\n<|im_end|>\n",
        "merged system/developer",
    );
    assert_eq!(checks.0, 1);
}

#[test]
fn rendering_keeps_reasoning_only_after_the_last_query_when_not_preserved() {
    let mut checks = Checks(0);
    checks.require(
        prompt(
            r#"{"messages":[{"role":"assistant","content":" old ","reasoning_content":" hide "},{"role":"user","content":"next"},{"role":"assistant","content":" new ","reasoning_content":" keep "}],"preserve_thinking":false,"reasoning_effort":"medium","add_generation_prompt":false}"#,
        ) == b"<|im_start|>assistant\nold<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\nkeep\n</think>\n\nnew<|im_end|>\n",
        "reasoning history boundary",
    );
    assert_eq!(checks.0, 1);
}

#[test]
fn rendering_parses_string_arguments_and_groups_tool_responses() {
    let mut checks = Checks(0);
    let request = parse(
        r#"{"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"","tool_calls":[{"id":"x","function":{"name":"f","arguments":"{\"x\":1.0,\"p\":\"é\"}"}}]},{"role":"tool","content":"one"},{"role":"tool","content":[{"text":"two"}]},{"role":"user","content":"<tool_response>\nthree\n</tool_response>"}],"reasoning_effort":"medium"}"#,
    );
    let tool = renderer().render(&request).unwrap();
    checks.require(
        tool.parsed_tool_arguments == 1,
        "JSON-string argument normalization",
    );
    checks.require(
        contains(
            &tool.prompt,
            "<parameter=x>\n1.0\n</parameter>\n<parameter=p>\né\n</parameter>".as_bytes(),
        ),
        "tool parameter rendering",
    );
    checks.require(
        contains(
            &tool.prompt,
            b"<|im_start|>user\n<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response><|im_end|>\n",
        ),
        "consecutive tool group",
    );
    let arguments = tool.normalized_messages.array().unwrap()[1]
        .find("tool_calls")
        .unwrap()
        .unwrap()
        .array()
        .unwrap()[0]
        .find("function")
        .unwrap()
        .unwrap()
        .find("arguments")
        .unwrap()
        .unwrap();
    checks.require(
        arguments.kind() == Kind::Object,
        "normalized arguments stay structured",
    );
    assert_eq!(checks.0, 4);
}

#[test]
fn rendering_leading_tool_fragment_has_no_user_opener() {
    let mut checks = Checks(0);
    checks.require(
        prompt(r#"{"messages":[{"role":"tool","content":"x"}],"enable_thinking":false}"#)
            == b"\n<tool_response>\nx\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "leading tool fragment matches pinned loop.previtem semantics",
    );
    assert_eq!(checks.0, 1);
}

#[test]
fn rendering_hoists_late_system_messages_in_order_when_asked() {
    let mut checks = Checks(0);
    let request = parse(
        r#"{"messages":[{"role":"system","content":"first"},{"role":"user","content":"Q"},{"role":"developer","content":"second"},{"role":"assistant","content":"A"},{"role":"system","content":"third"}],"late_system_policy":"hoist","enable_thinking":false}"#,
    );
    let late = renderer().render(&request).unwrap();
    checks.require(late.hoisted_system_messages == 2, "late system count");
    checks.require(
        late.prompt
            .starts_with(b"<|im_start|>system\nfirst\nsecond\nthird<|im_end|>\n"),
        "explicit hoist order",
    );
    checks.require(
        late.normalized_messages.array().unwrap()[3]
            .find("role")
            .unwrap()
            .unwrap()
            .string()
            .unwrap()
            == b"user",
        "hoisted ordering",
    );
    assert_eq!(checks.0, 3);
}

#[test]
fn chat_trim_strips_python_whitespace_but_keeps_bom_and_zero_width() {
    let mut checks = Checks(0);
    checks.require(
        chat_trim(" \u{1c}\u{85}\u{3000}x\u{a0}".as_bytes()).unwrap() == b"x",
        "Python-compatible trim",
    );
    checks.require(
        chat_trim(" \u{feff}x\u{200b} ".as_bytes()).unwrap() == "\u{feff}x\u{200b}".as_bytes(),
        "preserve BOM and zero-width characters",
    );
    assert_eq!(checks.0, 2);
}

#[test]
fn rendering_refuses_malformed_requests() {
    let mut checks = Checks(0);
    for (bad, message) in [
        (
            r#"{"messages":[]}"#,
            "chat message count is empty or exceeds limit",
        ),
        (
            r#"{"messages":[{"role":"other"}]}"#,
            "unexpected message role",
        ),
        (
            r#"{"messages":[{"role":"user","content":"Q"},{"role":"system","content":"late"}]}"#,
            "system message must be at the beginning or explicitly hoisted",
        ),
        (
            r#"{"messages":[{"role":"user","content":[{"type":"image","image_url":"x"}]}]}"#,
            "text-only chat does not accept images or video",
        ),
        (
            r#"{"messages":[{"role":"user","content":"Q"}],"reasoning_effort":"bogus"}"#,
            "unexpected reasoning effort",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"bad>name","arguments":{}}}]}]}"#,
            "unsafe tool or parameter name",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"not JSON"}}]}]}"#,
            "invalid JSON literal",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{\"x\":1}"}}]}],"parse_tool_arguments":false}"#,
            "parse nonempty JSON-string tool arguments before raw template rendering",
        ),
    ] {
        checks.rejected(renderer().render(&parse(bad)), message);
    }
    assert_eq!(checks.0, 8);
}

#[test]
fn data_literals_extract_profile_strings_and_refuse_bad_sources() {
    let mut checks = Checks(0);
    let strings = extract_flash_next_literals(
        br##"{% set x = 'Reasoning effort is set to xhigh. X' %}
{% set l = 'Reasoning effort is set to low. L' %}
{{ '# Tools\n\nT' }} {{ '\n\nIf you choose to call a function ONLY reply X' }}
{# 'ignored unmatched quote #}"##,
    )
    .unwrap();
    checks.require(
        strings.tools_header == b"# Tools\n\nT",
        "template literal extraction",
    );
    checks.rejected(
        extract_flash_next_literals(b"'missing profile'"),
        "pinned template literal missing or ambiguous",
    );
    checks.rejected(
        extract_flash_next_literals(b"'\\x00'"),
        "unsupported template literal escape",
    );
    assert_eq!(checks.0, 3);
}

/// Reads the pinned `chat_template.jinja` named by `TOKSPUR_PINNED_TEMPLATE` (SHA-256 `12827f24…aadce`; `rust/tools/diff/chat_response.sh` checks the hash). The template is a model asset and never enters Git, so without the variable this test does nothing.
#[test]
fn pinned_template_literals_extract_and_render() {
    let Some(path) = std::env::var_os("TOKSPUR_PINNED_TEMPLATE") else {
        return;
    };
    let source = std::fs::read(path).unwrap();
    assert!(source.len() <= 64 * 1024);
    let strings = extract_flash_next_literals(&source).unwrap();
    assert!(
        strings
            .reasoning_xhigh
            .starts_with(b"Reasoning effort is set to xhigh.")
    );
    assert!(
        strings
            .reasoning_low
            .starts_with(b"Reasoning effort is set to low.")
    );
    assert!(strings.tools_header.starts_with(b"# Tools\n\n"));
    assert!(
        strings
            .tool_instructions
            .starts_with(b"\n\nIf you choose to call a function ONLY reply")
    );
    let renderer = ChatRenderer::new(strings.clone()).unwrap();
    let request = parse(
        r#"{"messages":[{"role":"user","content":"Q"}],"tools":[{"type":"function","function":{"name":"f"}}],"reasoning_effort":"low"}"#,
    );
    let prompt = renderer.render(&request).unwrap().prompt;
    assert!(
        prompt.starts_with(&[b"<|im_start|>system\n".as_slice(), &strings.reasoning_low].concat())
    );
    assert!(contains(&prompt, &strings.tool_instructions));
}
