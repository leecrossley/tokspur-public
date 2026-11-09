//! `native/src/response_cli.cpp` and the `main` of `native/tests/response_cli_main.cpp` have no C++ unit test; these pin their observable behaviour (stdout lines, stderr, exit status) over the fixture tokenizer. `rust/tools/diff/chat_response.sh` compares the same paths with the C++ binary.

// Test code: helpers outside #[test] functions unwrap as the tests do.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixture;

use std::ffi::OsString;
use std::io::{self, Read, Write};

use fixture::{Fixture, FixtureError, load};
use tokspur_text::response_cli::{
    ResponseCliError, TOKENIZER_PIN, response_cli, response_cli_main,
};

const READY: &str = concat!(
    r#"{"event":"ready","scope":"model-free accumulated-response parser; no tools are executed","#,
    r#""tokenizer_sha256":"8e852142d8ab38bb7678f650bd504ba28c29d9c6357e6ed762bebdbd474689eb","#,
    r#""vocabulary_size":259,"im_end_id":256,"end_of_text_id":257,"im_start_id":258,"#,
    r#""maximum_ids":16384,"maximum_request_bytes":1048576,"tools_executed":false}"#,
);

fn args(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}

fn standard() -> Vec<OsString> {
    args(&[
        "tokspur-native",
        "response",
        "--tokenizer",
        "/assets/fixture.json",
        "--sha256",
        TOKENIZER_PIN,
    ])
}

/// Runs `response_cli` over `input`; returns the status (or the escaping error's text) and stdout.
fn run(arguments: &[OsString], input: &[u8]) -> (Result<i32, String>, Vec<u8>) {
    let mut output = Vec::new();
    let status =
        response_cli(arguments, load, input, &mut output).map_err(|error| error.to_string());
    (status, output)
}

fn lines(output: &[u8]) -> Vec<String> {
    let text = String::from_utf8(output.to_vec()).unwrap();
    assert!(
        text.ends_with('\n'),
        "every protocol line ends with a newline"
    );
    text.lines().map(str::to_owned).collect()
}

fn error_line(message: &str) -> String {
    format!(
        r#"{{"event":"response","status":"error","finish_reason":"error","complete":false,"content":"","reasoning_content":"","tool_calls":[],"tools_executed":false,"message":"{message}"}}"#
    )
}

fn request(text: &str, finish: &str, stop: &str) -> String {
    let ids: Vec<String> = Fixture::STANDARD
        .encode(text.as_bytes())
        .iter()
        .map(u32::to_string)
        .collect();
    format!(
        r#"{{"ids":[{}],"tools":[{{"name":"ping","parameters":[]}}],"initial_thinking":false,"finish_reason":"{finish}","stop_id":{stop}}}"#,
        ids.join(",")
    )
}

#[test]
fn ready_line_then_one_response_line_per_request() {
    let input = format!(
        "{}\n{}\n",
        request(
            "hello <tool_call><function=ping></function></tool_call><|im_end|>",
            "stop",
            "256"
        ),
        request("partial", "length", "null"),
    );
    let (status, output) = run(&standard(), input.as_bytes());
    assert_eq!(status, Ok(0));
    assert_eq!(
        lines(&output),
        [
            READY.to_owned(),
            r#"{"event":"response","status":"complete","finish_reason":"stop","complete":true,"content":"hello ","reasoning_content":"","tool_calls":[{"ordinal":0,"type":"function","function":{"name":"ping","arguments":{}}}],"tools_executed":false}"#.to_owned(),
            r#"{"event":"response","status":"incomplete","finish_reason":"length","complete":false,"content":"partial","reasoning_content":"","tool_calls":[],"tools_executed":false}"#.to_owned(),
        ]
    );
}

#[test]
fn empty_input_prints_only_the_ready_line() {
    let (status, output) = run(&standard(), b"");
    assert_eq!(status, Ok(0));
    assert_eq!(lines(&output), [READY]);
}

#[test]
fn refused_requests_print_error_envelopes_keep_serving_and_exit_one() {
    let input = format!(
        "\nnot json\n{}\n{{\"ids\":[]}}",
        request("ok<|im_end|>", "stop", "256")
    );
    let (status, output) = run(&standard(), input.as_bytes());
    assert_eq!(status, Ok(1));
    assert_eq!(
        lines(&output),
        [
            READY.to_owned(),
            error_line("truncated JSON"),
            error_line("invalid JSON literal"),
            r#"{"event":"response","status":"complete","finish_reason":"stop","complete":true,"content":"ok","reasoning_content":"","tool_calls":[],"tools_executed":false}"#.to_owned(),
            error_line("missing explicit response field"),
        ]
    );
}

#[test]
fn oversized_request_lines_are_drained_and_refused() {
    let mut input = vec![b' '; 1024 * 1024];
    input.extend_from_slice(b"null\n");
    input.extend_from_slice(request("x", "length", "null").as_bytes());
    input.push(b'\n');
    let (status, output) = run(&standard(), &input);
    assert_eq!(status, Ok(1));
    let lines = lines(&output);
    assert_eq!(lines[1], error_line("response request exceeds 1-MiB bound"));
    assert!(lines[2].contains(r#""content":"x""#));
    assert_eq!(lines.len(), 3);
}

#[test]
fn a_line_of_exactly_one_mebibyte_is_admitted() {
    let body = b"null";
    let mut input = vec![b' '; 1024 * 1024 - body.len()];
    input.extend_from_slice(body);
    let (status, output) = run(&standard(), &input);
    assert_eq!(status, Ok(1));
    // The line is read whole: `null` reaches the protocol, not the line bound.
    assert_eq!(lines(&output)[1], error_line("expected JSON object"));
}

#[test]
fn options_are_validated_before_loading_the_tokenizer() {
    let cases: [(&[&str], &str); 8] = [
        (
            &["tokspur-native"],
            "response requires the supported pinned tokenizer and explicit SHA256",
        ),
        (
            &["tokspur-native", "response"],
            "response requires the supported pinned tokenizer and explicit SHA256",
        ),
        (
            &["tokspur-native", "response", "--tokenizer"],
            "duplicate response option or missing value",
        ),
        (
            &[
                "tokspur-native",
                "response",
                "--tokenizer",
                "a",
                "--tokenizer",
                "b",
            ],
            "duplicate response option or missing value",
        ),
        (
            &["tokspur-native", "response", "--bogus"],
            "duplicate response option or missing value",
        ),
        (
            &["tokspur-native", "response", "--bogus", "x"],
            "unknown response option",
        ),
        (
            &[
                "tokspur-native",
                "response",
                "--tokenizer",
                "/assets/fixture.json",
                "--sha256",
                "00",
            ],
            "response requires the supported pinned tokenizer and explicit SHA256",
        ),
        (
            &[
                "tokspur-native",
                "response",
                "--tokenizer",
                "",
                "--sha256",
                TOKENIZER_PIN,
            ],
            "response requires the supported pinned tokenizer and explicit SHA256",
        ),
    ];
    for (arguments, message) in cases {
        let (status, output) = run(&args(arguments), b"");
        assert_eq!(status, Err(message.to_owned()), "{arguments:?}");
        assert!(output.is_empty());
    }
}

#[test]
fn the_subcommand_slot_is_skipped_whatever_it_holds() {
    let shifted = args(&[
        "tokspur-native",
        "--tokenizer",
        "/assets/fixture.json",
        "--sha256",
        TOKENIZER_PIN,
    ]);
    assert_eq!(
        run(&shifted, b"").0,
        Err("unknown response option".to_owned())
    );
}

#[test]
fn tokenizer_load_and_marker_failures_escape() {
    let missing = args(&[
        "tokspur-native",
        "response",
        "--tokenizer",
        "/assets/missing.json",
        "--sha256",
        TOKENIZER_PIN,
    ]);
    let mut output = Vec::new();
    let error = response_cli(&missing, load, io::empty(), &mut output).unwrap_err();
    assert_eq!(error, ResponseCliError::Load(FixtureError::Load));
    assert_eq!(error.to_string(), "stub tokenizer missing");
    let plain = args(&[
        "tokspur-native",
        "response",
        "--tokenizer",
        "/assets/plain.json",
        "--sha256",
        TOKENIZER_PIN,
    ]);
    assert_eq!(
        run(&plain, b"").0,
        Err("response marker is not one exact pinned tokenizer token".to_owned())
    );
}

#[test]
fn main_prints_escaping_errors_to_stderr_and_dispatches_text_commands() {
    let refuse_text =
        |_: &[OsString]| -> Result<i32, FixtureError> { panic!("text_cli must not run") };
    let mut output = Vec::new();
    let mut errors = Vec::new();
    let status = response_cli_main(
        &args(&["tokspur-native", "response"]),
        refuse_text,
        load,
        io::empty(),
        &mut output,
        &mut errors,
    );
    assert_eq!(status, 1);
    assert!(output.is_empty());
    assert_eq!(
        errors,
        b"response requires the supported pinned tokenizer and explicit SHA256\n"
    );

    let mut errors = Vec::new();
    let status = response_cli_main(
        &standard(),
        refuse_text,
        load,
        &b"[]\n"[..],
        &mut output,
        &mut errors,
    );
    assert_eq!(status, 1);
    assert!(errors.is_empty());
    assert_eq!(
        lines(&output),
        [READY.to_owned(), error_line("expected JSON object")]
    );

    for command in ["text", "chat"] {
        let mut errors = Vec::new();
        let status = response_cli_main(
            &args(&["tokspur-native", command]),
            |arguments: &[OsString]| -> Result<i32, FixtureError> {
                assert_eq!(arguments[1], command);
                Ok(7)
            },
            load,
            io::empty(),
            io::sink(),
            &mut errors,
        );
        assert_eq!(status, 7);
        assert!(errors.is_empty());
        let status = response_cli_main(
            &args(&["tokspur-native", command]),
            |_: &[OsString]| -> Result<i32, FixtureError> { Err(FixtureError::Decode) },
            load,
            io::empty(),
            io::sink(),
            &mut errors,
        );
        assert_eq!(status, 1);
        assert_eq!(errors, b"token ID outside vocabulary\n");
    }
}

/// A writer that fails every write, like a closed stdout.
struct Closed;

impl Write for Closed {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::from(io::ErrorKind::BrokenPipe))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_failed_stdout_drops_output_without_changing_the_status() {
    let input = format!("{}\n", request("ok<|im_end|>", "stop", "256"));
    assert_eq!(
        response_cli(&standard(), load, input.as_bytes(), Closed),
        Ok(0)
    );
    assert_eq!(
        response_cli(&standard(), load, &b"bad\n"[..], Closed),
        Ok(1)
    );
}

/// Yields `head`, then fails every read, like a broken stdin.
struct Broken<'a> {
    head: &'a [u8],
}

impl Read for Broken<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.head.read(buffer)?;
        if count == 0 {
            return Err(io::Error::other("read failed"));
        }
        Ok(count)
    }
}

#[test]
fn a_read_error_ends_input_like_end_of_file() {
    // The partial last line is still served, as `std::istream::get` returns it.
    let partial = request("ok<|im_end|>", "stop", "256");
    let reader = io::BufReader::new(Broken {
        head: partial.as_bytes(),
    });
    let mut output = Vec::new();
    assert_eq!(response_cli(&standard(), load, reader, &mut output), Ok(0));
    assert_eq!(lines(&output).len(), 2);

    let reader = io::BufReader::new(Broken { head: b"bad" });
    let mut output = Vec::new();
    assert_eq!(response_cli(&standard(), load, reader, &mut output), Ok(1));
    assert_eq!(
        lines(&output),
        [READY.to_owned(), error_line("invalid JSON value")]
    );
}

/// Interrupts every other read, as a signal can.
struct Interrupted<'a> {
    head: &'a [u8],
    interrupt: bool,
}

impl Read for Interrupted<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.interrupt = !self.interrupt;
        if self.interrupt {
            return Err(io::Error::from(io::ErrorKind::Interrupted));
        }
        let limit = buffer.len().min(3);
        self.head.read(&mut buffer[..limit])
    }
}

#[test]
fn interrupted_reads_are_retried() {
    let input = format!("{}\n", request("ok<|im_end|>", "stop", "256"));
    let reader = io::BufReader::new(Interrupted {
        head: input.as_bytes(),
        interrupt: false,
    });
    let mut output = Vec::new();
    assert_eq!(response_cli(&standard(), load, reader, &mut output), Ok(0));
    assert!(lines(&output)[1].contains(r#""status":"complete""#));
}

#[test]
fn a_request_split_across_many_buffer_fills_is_read_whole() {
    let text = "a".repeat(10_000);
    let input = format!("{}\n", request(&format!("{text}<|im_end|>"), "stop", "256"));
    let reader = io::BufReader::with_capacity(7, input.as_bytes());
    let mut output = Vec::new();
    assert_eq!(response_cli(&standard(), load, reader, &mut output), Ok(0));
    assert!(lines(&output)[1].contains(&text));
}
