//! CLI cases: the C++ `response-cli` binary (`response_cli_main.cpp`, `response_cli.cpp` and `cpp/native_text_stub.cpp`) against `tokspur_text::response_cli::response_cli_main`, comparing stdout, stderr and the exit status.

use std::ffi::OsString;
use std::fmt;
use std::io::{self, Write};
use std::os::unix::ffi::OsStringExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;

use tokspur_text::response_cli::{TOKENIZER_PIN, response_cli_main};

use crate::fixture::{FIXTURE, load};
use crate::generate::{Random, random_envelope};

pub struct CliCase {
    pub label: String,
    pub args: Vec<Vec<u8>>,
    pub stdin: Vec<u8>,
}

#[derive(PartialEq, Eq)]
pub struct CliOutcome {
    pub status: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl CliOutcome {
    /// One byte string for the report.
    pub fn flatten(&self) -> Vec<u8> {
        let mut out = format!("status {:?}\nstdout:\n", self.status).into_bytes();
        out.extend_from_slice(&self.stdout);
        out.extend_from_slice(b"\nstderr:\n");
        out.extend_from_slice(&self.stderr);
        out
    }
}

/// The stub `text_cli`'s refusal.
#[derive(Debug)]
struct TextRefused;

impl fmt::Display for TextRefused {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("text stub refused")
    }
}

pub fn rust(case: &CliCase) -> CliOutcome {
    let args: Vec<OsString> = case
        .args
        .iter()
        .map(|arg| OsString::from_vec(arg.clone()))
        .collect();
    let text_cli = |args: &[OsString]| {
        if args.len() >= 3 && args[2] == "throw" {
            Err(TextRefused)
        } else {
            Ok(7)
        }
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = response_cli_main(
        &args,
        text_cli,
        load,
        case.stdin.as_slice(),
        &mut stdout,
        &mut stderr,
    );
    CliOutcome {
        status: Some(status),
        stdout,
        stderr,
    }
}

pub fn cpp(binary: &Path, case: &CliCase) -> io::Result<CliOutcome> {
    let mut child = Command::new(binary)
        .args(
            case.args[1..]
                .iter()
                .map(|arg| OsString::from_vec(arg.clone())),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("no child stdin"))?;
    let input = case.stdin.clone();
    // The child may exit before reading everything; a broken pipe is then part of the behaviour under test, not a runner failure.
    let writer = thread::spawn(move || match stdin.write_all(&input) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(error) => Err(error),
    });
    let output = child.wait_with_output()?;
    writer
        .join()
        .map_err(|_| io::Error::other("stdin writer panicked"))??;
    Ok(CliOutcome {
        status: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

fn args(values: &[&str]) -> Vec<Vec<u8>> {
    values
        .iter()
        .map(|value| value.as_bytes().to_vec())
        .collect()
}

fn request(text: &[u8], finish: &str, stop: &str) -> Vec<u8> {
    let ids: Vec<String> = FIXTURE.encode(text).iter().map(u32::to_string).collect();
    format!(
        r#"{{"ids":[{}],"tools":[{{"name":"ping","parameters":[]}},{{"name":"read_file","parameters":[{{"name":"path","type":"string","required":true}}]}}],"initial_thinking":false,"finish_reason":"{finish}","stop_id":{stop}}}"#,
        ids.join(",")
    )
    .into_bytes()
}

fn lines(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for part in parts {
        out.extend_from_slice(part);
        out.push(b'\n');
    }
    out
}

pub fn cases(random: &mut Random) -> Vec<CliCase> {
    let standard = || {
        args(&[
            "tokspur-native",
            "response",
            "--tokenizer",
            "/assets/fixture.json",
            "--sha256",
            TOKENIZER_PIN,
        ])
    };
    let ok = request(
        b"hello <tool_call><function=ping></function></tool_call><|im_end|>",
        "stop",
        "256",
    );
    let call = request(b"<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/\xe6\x97\xa5.rs\n</parameter>\n</function>\n</tool_call><|im_end|>", "stop", "256");
    let partial = request(b"x\xc3", "length", "null");
    let errored = request(b"x", "error", "null");
    let invalid = request(b"x<|im_end|>junk<|im_end|>", "stop", "256");
    let mut cases = Vec::new();
    let mut add = |label: &str, args: Vec<Vec<u8>>, stdin: Vec<u8>| {
        cases.push(CliCase {
            label: label.to_owned(),
            args,
            stdin,
        })
    };
    add("empty input", standard(), Vec::new());
    add("one request", standard(), lines(&[&ok]));
    add(
        "mixed requests",
        standard(),
        lines(&[
            &ok,
            b"",
            b"not json",
            &call,
            b"[]",
            &partial,
            &errored,
            &invalid,
            b"{\"ids\":[]}",
            &ok,
        ]),
    );
    add(
        "last line without newline",
        standard(),
        [lines(&[&ok]), ok.clone()].concat(),
    );
    add(
        "invalid last line without newline",
        standard(),
        [lines(&[&ok]), b"nope".to_vec()].concat(),
    );
    add(
        "CRLF lines",
        standard(),
        [ok.as_slice(), b"\r\n", &partial, b"\r\n"].concat(),
    );
    let mut oversized = vec![b' '; 1024 * 1024];
    oversized.extend_from_slice(b"null\n");
    oversized.extend_from_slice(&lines(&[&ok]));
    add("oversized line then a request", standard(), oversized);
    let mut exact = vec![b' '; 1024 * 1024 - ok.len()];
    exact.extend_from_slice(&ok);
    add(
        "request line of exactly 1 MiB",
        standard(),
        lines(&[&exact]),
    );
    let mut oversized_tail = vec![b'x'; 1024 * 1024 + 5];
    oversized_tail.extend_from_slice(b"tail");
    add(
        "oversized last line without newline",
        standard(),
        oversized_tail,
    );
    add(
        "raw control byte",
        standard(),
        lines(&[b"{\"ids\":\"\x01\"}"]),
    );
    add("NUL in line", standard(), lines(&[b"{\"ids\":[]}\x00"]));
    let deep = format!("{}{}", "[".repeat(33), "]".repeat(33));
    add("parse depth 33", standard(), lines(&[deep.as_bytes()]));
    let shallow = format!("{}{}", "[".repeat(32), "]".repeat(32));
    add("parse depth 32", standard(), lines(&[shallow.as_bytes()]));
    let nodes = format!("[{}]", vec!["0"; 150_000].join(","));
    add("150001 nodes", standard(), lines(&[nodes.as_bytes()]));
    let fewer = format!("[{}]", vec!["0"; 149_999].join(","));
    add("150000 nodes", standard(), lines(&[fewer.as_bytes()]));
    add("invalid UTF-8 line", standard(), lines(&[b"\"\xc3\x28\""]));
    let generated: Vec<Vec<u8>> = (0..300)
        .map(|_| {
            let (_, envelope, _) = random_envelope(random);
            envelope.text(random)
        })
        .collect();
    let refs: Vec<&[u8]> = generated.iter().map(Vec::as_slice).collect();
    add("300 generated envelopes", standard(), lines(&refs));
    add("no options", args(&["tokspur-native"]), Vec::new());
    add(
        "subcommand only",
        args(&["tokspur-native", "response"]),
        Vec::new(),
    );
    add(
        "missing value",
        args(&["tokspur-native", "response", "--tokenizer"]),
        Vec::new(),
    );
    add(
        "duplicate option",
        args(&[
            "tokspur-native",
            "response",
            "--tokenizer",
            "a",
            "--tokenizer",
            "b",
        ]),
        Vec::new(),
    );
    add(
        "unknown option without value",
        args(&["tokspur-native", "response", "--bogus"]),
        Vec::new(),
    );
    add(
        "unknown option",
        args(&["tokspur-native", "response", "--bogus", "x"]),
        Vec::new(),
    );
    add(
        "wrong pin",
        args(&[
            "tokspur-native",
            "response",
            "--tokenizer",
            "/assets/fixture.json",
            "--sha256",
            "00",
        ]),
        Vec::new(),
    );
    add(
        "uppercase pin",
        args(&[
            "tokspur-native",
            "response",
            "--tokenizer",
            "/assets/fixture.json",
            "--sha256",
            &TOKENIZER_PIN.to_uppercase(),
        ]),
        Vec::new(),
    );
    add(
        "empty path",
        args(&[
            "tokspur-native",
            "response",
            "--tokenizer",
            "",
            "--sha256",
            TOKENIZER_PIN,
        ]),
        Vec::new(),
    );
    add(
        "pin before path",
        args(&[
            "tokspur-native",
            "response",
            "--sha256",
            TOKENIZER_PIN,
            "--tokenizer",
            "/assets/fixture.json",
        ]),
        lines(&[&ok]),
    );
    add(
        "shifted subcommand slot",
        args(&[
            "tokspur-native",
            "--tokenizer",
            "/assets/fixture.json",
            "--sha256",
            TOKENIZER_PIN,
        ]),
        Vec::new(),
    );
    add(
        "value as option name",
        args(&[
            "tokspur-native",
            "response",
            "--tokenizer",
            "--sha256",
            "--sha256",
            TOKENIZER_PIN,
        ]),
        Vec::new(),
    );
    add(
        "missing tokenizer",
        args(&[
            "tokspur-native",
            "response",
            "--tokenizer",
            "/assets/missing.json",
            "--sha256",
            TOKENIZER_PIN,
        ]),
        lines(&[&ok]),
    );
    add(
        "markers not single tokens",
        args(&[
            "tokspur-native",
            "response",
            "--tokenizer",
            "/assets/plain.json",
            "--sha256",
            TOKENIZER_PIN,
        ]),
        lines(&[&ok]),
    );
    add(
        "text dispatch",
        args(&["tokspur-native", "text"]),
        lines(&[&ok]),
    );
    add(
        "chat dispatch",
        args(&["tokspur-native", "chat", "--anything"]),
        Vec::new(),
    );
    add(
        "text refusal",
        args(&["tokspur-native", "text", "throw"]),
        Vec::new(),
    );
    add(
        "chat refusal",
        args(&["tokspur-native", "chat", "throw"]),
        Vec::new(),
    );
    add(
        "Text is not text",
        args(&["tokspur-native", "Text"]),
        Vec::new(),
    );
    cases
}
