//! Differential runner for the Rust port of `native/src/chat.cpp`, `response.cpp`, `response_protocol.cpp` and `response_cli.cpp`.
//!
//! It generates deterministic cases, runs them through the C++ driver (`cpp/text_driver.cpp`, one process for all cases) and through the Rust port, and compares the outcome bytes case by case. CLI cases run the C++ `response-cli` binary once each. `run.sh` builds the C++ side into `build/rust-diff/chat-response/` and passes the paths:
//!
//! ```text
//! tokspur-text-diff --driver BIN --cli BIN --out DIR [--template PATH] [--seed N]
//! ```
//!
//! The report (`DIR/report.txt`) lists counts per case kind, every distinct refusal message both sides produced, and the first differences in full.

#![forbid(unsafe_code)]

mod case;
mod cli;
mod fixture;
mod generate;

use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, ExitCode, Stdio};
use std::thread;

use case::Wire;

struct Options {
    driver: PathBuf,
    cli: PathBuf,
    out: PathBuf,
    template: Option<PathBuf>,
    seed: u64,
}

impl Options {
    fn parse(mut args: impl Iterator<Item = OsString>) -> Result<Self, Box<dyn Error>> {
        let usage = "usage: tokspur-text-diff --driver BIN --cli BIN --out DIR [--template PATH] [--seed N]";
        let mut driver = None;
        let mut cli = None;
        let mut out = None;
        let mut template = None;
        let mut seed = 0x0074_6f6b_7370_7572;
        args.next();
        while let Some(flag) = args.next() {
            let value = args.next().ok_or(usage)?;
            match flag.to_str() {
                Some("--driver") => driver = Some(PathBuf::from(value)),
                Some("--cli") => cli = Some(PathBuf::from(value)),
                Some("--out") => out = Some(PathBuf::from(value)),
                Some("--template") => template = Some(PathBuf::from(value)),
                Some("--seed") => seed = value.to_str().ok_or(usage)?.parse()?,
                _ => return Err(usage.into()),
            }
        }
        Ok(Self {
            driver: driver.ok_or(usage)?,
            cli: cli.ok_or(usage)?,
            out: out.ok_or(usage)?,
            template,
            seed,
        })
    }
}

fn run_driver(driver: &PathBuf, input: Vec<u8>) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut child = Command::new(driver)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let mut stdin = child.stdin.take().ok_or("no driver stdin")?;
    let writer = thread::spawn(move || stdin.write_all(&input));
    let output = child.wait_with_output()?;
    writer
        .join()
        .map_err(|_| "driver stdin writer panicked")??;
    if !output.status.success() {
        return Err(format!("C++ driver failed: {}", output.status).into());
    }
    Ok(output.stdout)
}

fn frames(mut output: &[u8]) -> Result<Vec<&[u8]>, Box<dyn Error>> {
    let mut frames = Vec::new();
    while !output.is_empty() {
        let (size, rest) = output
            .split_first_chunk::<4>()
            .ok_or("truncated driver frame")?;
        let size = usize::try_from(u32::from_le_bytes(*size))?;
        if size > rest.len() {
            return Err("truncated driver frame".into());
        }
        let (frame, rest) = rest.split_at(size);
        frames.push(frame);
        output = rest;
    }
    Ok(frames)
}

/// Printable ASCII kept, everything else `\xNN`, cut at `limit` bytes.
fn escape(bytes: &[u8], limit: usize) -> String {
    let mut out = String::new();
    for &byte in bytes.iter().take(limit) {
        if (0x20..0x7f).contains(&byte) && byte != b'\\' {
            out.push(char::from(byte));
        } else {
            out.push_str(&format!("\\x{byte:02x}"));
        }
    }
    if bytes.len() > limit {
        out.push_str(&format!("... ({} bytes)", bytes.len()));
    }
    out
}

/// Reads the refusal messages out of an outcome, for the coverage summary.
fn messages(kind: &str, outcome: &[u8], found: &mut Vec<Vec<u8>>) {
    struct Cursor<'a>(&'a [u8]);
    impl Cursor<'_> {
        fn take(&mut self, count: usize) -> Option<&[u8]> {
            if count > self.0.len() {
                return None;
            }
            let (head, tail) = self.0.split_at(count);
            self.0 = tail;
            Some(head)
        }
        fn u32(&mut self) -> Option<usize> {
            let bytes: [u8; 4] = self.take(4)?.try_into().ok()?;
            usize::try_from(u32::from_le_bytes(bytes)).ok()
        }
        fn bytes(&mut self) -> Option<Vec<u8>> {
            let size = self.u32()?;
            Some(self.take(size)?.to_vec())
        }
    }
    /// A `dump_or_error` field: flag 0 is JSON text, flag 1 a dump refusal.
    fn dumped(cursor: &mut Cursor<'_>, found: &mut Vec<Vec<u8>>) -> Option<()> {
        let flag = cursor.take(1)?[0];
        let text = cursor.bytes()?;
        if flag == 1 {
            found.push(text);
        } else {
            envelope_messages(&text, found);
        }
        Some(())
    }
    if kind == "cli" {
        envelope_messages(outcome, found);
        if let Some(at) = outcome.windows(8).position(|window| window == b"stderr:\n") {
            found.extend(
                outcome[at + 8..]
                    .split(|&c| c == b'\n')
                    .filter(|line| !line.is_empty())
                    .map(<[u8]>::to_vec),
            );
        }
        return;
    }
    let Some((&tag, rest)) = outcome.split_first() else {
        return;
    };
    let mut cursor = Cursor(rest);
    if matches!(tag, b'P' | b'E' | b'C') {
        if let Some(message) = cursor.bytes() {
            found.push(message);
        }
        return;
    }
    if tag != b'O' {
        return;
    }
    match kind {
        "render" | "render tree" => {
            if cursor.bytes().is_some() {
                dumped(&mut cursor, found);
            }
        }
        "protocol" => {
            dumped(&mut cursor, found);
        }
        "parser" => {
            // Snapshots: returned, status, error, bytes, records.
            while !cursor.0.is_empty() {
                let Some(_) = cursor.take(2) else { return };
                let Some(error) = cursor.bytes() else { return };
                if !error.is_empty() {
                    found.push(error);
                }
                let Some(_) = cursor.take(8) else { return };
                let Some(records) = cursor.u32() else { return };
                for _ in 0..records {
                    let complete = cursor.take(1).is_some()
                        && cursor.bytes().is_some()
                        && cursor.take(8).is_some()
                        && cursor.bytes().is_some()
                        && dumped(&mut cursor, found).is_some();
                    if !complete {
                        return;
                    }
                }
            }
        }
        _ => {}
    }
}

/// The `"message"` values of response envelopes in `text`.
fn envelope_messages(text: &[u8], found: &mut Vec<Vec<u8>>) {
    let key = b"\"message\":\"";
    let mut rest = text;
    while let Some(at) = rest.windows(key.len()).position(|window| window == key) {
        rest = &rest[at + key.len()..];
        let end = rest.iter().position(|&c| c == b'"').unwrap_or(rest.len());
        found.push(rest[..end].to_vec());
        rest = &rest[end..];
    }
}

/// Significant digits of a decimal: no sign, point, exponent or edge zeros.
fn significant(text: &[u8]) -> Vec<u8> {
    let mantissa = text
        .split(|&c| c == b'e' || c == b'E')
        .next()
        .unwrap_or_default();
    let digits: Vec<u8> = mantissa
        .iter()
        .copied()
        .filter(u8::is_ascii_digit)
        .collect();
    let start = digits
        .iter()
        .position(|&c| c != b'0')
        .unwrap_or(digits.len());
    let end = digits
        .iter()
        .rposition(|&c| c != b'0')
        .map_or(start, |last| last + 1);
    digits[start..end.max(start)].to_vec()
}

/// Whether a `template_json` number case printed digits other than Rust's own `{:e}` shortest digits: an exact tie that `to_chars` broke to even.
fn tie_broken_to_even(json: &[u8], cpp: &[u8]) -> bool {
    let Ok(lexeme) = std::str::from_utf8(json) else {
        return false;
    };
    if !lexeme.contains(['.', 'e', 'E']) {
        return false;
    }
    let Ok(value) = lexeme.parse::<f64>() else {
        return false;
    };
    if !value.is_finite() || value == 0.0 || cpp.first() != Some(&b'O') || cpp.len() < 5 {
        return false;
    }
    significant(format!("{:e}", value.abs()).as_bytes()) != significant(&cpp[5..])
}

#[derive(Default)]
struct Tally {
    cases: usize,
    differences: usize,
    tags: BTreeMap<u8, usize>,
    messages: BTreeMap<Vec<u8>, usize>,
}

#[derive(Default)]
struct Report {
    kinds: BTreeMap<&'static str, Tally>,
    details: String,
    shown: usize,
    ties: usize,
}

impl Report {
    fn record(&mut self, kind: &'static str, label: &str, rust: &[u8], cpp: &[u8]) {
        let tally = self.kinds.entry(kind).or_default();
        tally.cases += 1;
        if let Some(&tag) = cpp.first() {
            *tally.tags.entry(tag).or_default() += 1;
        }
        let mut found = Vec::new();
        messages(kind, cpp, &mut found);
        for message in found {
            *tally.messages.entry(message).or_default() += 1;
        }
        if rust != cpp {
            tally.differences += 1;
            if self.shown < 40 {
                self.shown += 1;
                let at = rust
                    .iter()
                    .zip(cpp)
                    .position(|(a, b)| a != b)
                    .unwrap_or(rust.len().min(cpp.len()));
                let start = at.saturating_sub(64);
                self.details.push_str(&format!(
                    "DIFFERENCE {kind} [{label}] at outcome byte {at}\n  rust: {}\n  c++:  {}\n\n",
                    escape(&rust[start.min(rust.len())..], 600),
                    escape(&cpp[start.min(cpp.len())..], 600),
                ));
            }
        }
    }

    fn differences(&self) -> usize {
        self.kinds.values().map(|tally| tally.differences).sum()
    }

    fn render(&self, seed: u64, template: bool) -> String {
        let mut out = String::new();
        let cases: usize = self.kinds.values().map(|tally| tally.cases).sum();
        out.push_str(&format!(
            "tokspur-text differential: seed {seed}, pinned template {}\n",
            if template { "included" } else { "absent" }
        ));
        out.push_str(&format!(
            "{cases} cases compared, {} differences\n",
            self.differences()
        ));
        out.push_str(&format!(
            "template_json exact shortest-digit ties broken to even: {}\n\n",
            self.ties
        ));
        for (kind, tally) in &self.kinds {
            let tags: Vec<String> = tally
                .tags
                .iter()
                .map(|(tag, count)| format!("{}={count}", char::from(*tag)))
                .collect();
            out.push_str(&format!(
                "{kind}: {} cases, {} differences, outcomes {}\n",
                tally.cases,
                tally.differences,
                tags.join(" ")
            ));
            for (message, count) in &tally.messages {
                out.push_str(&format!("    {count:>7}  {}\n", escape(message, 200)));
            }
        }
        out.push('\n');
        out.push_str(&self.details);
        out
    }
}

fn main() -> Result<ExitCode, Box<dyn Error>> {
    let options = Options::parse(std::env::args_os())?;
    fs::create_dir_all(&options.out)?;
    let template = match &options.template {
        Some(path) => Some(fs::read(path)?),
        None => None,
    };
    let cases = generate::cases(options.seed, template.as_deref());
    let mut wire = Wire::default();
    for labeled in &cases {
        labeled.case.encode(&mut wire);
    }
    let output = run_driver(&options.driver, wire.bytes)?;
    let frames = frames(&output)?;
    if frames.len() != cases.len() {
        return Err(format!(
            "C++ driver returned {} outcomes for {} cases",
            frames.len(),
            cases.len()
        )
        .into());
    }
    let mut report = Report::default();
    for (labeled, cpp) in cases.iter().zip(frames) {
        report.record(
            labeled.case.kind(),
            &labeled.label,
            &labeled.case.evaluate(),
            cpp,
        );
        if let case::Case::TemplateJson { json } = &labeled.case
            && tie_broken_to_even(json, cpp)
        {
            report.ties += 1;
        }
    }
    let mut random = generate::Random::new(options.seed ^ 0x636c69);
    for case in cli::cases(&mut random) {
        let rust = cli::rust(&case).flatten();
        let cpp = cli::cpp(&options.cli, &case)?.flatten();
        report.record("cli", &case.label, &rust, &cpp);
    }
    let text = report.render(options.seed, template.is_some());
    fs::write(options.out.join("report.txt"), &text)?;
    let summary: Vec<&str> = text
        .lines()
        .take_while(|line| !line.starts_with("DIFFERENCE"))
        .filter(|line| !line.starts_with("    "))
        .collect();
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{}", summary.join("\n").trim_end())?;
    writeln!(
        stdout,
        "report: {}",
        options.out.join("report.txt").display()
    )?;
    Ok(if report.differences() == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}
