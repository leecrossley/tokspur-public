//! `dispatch-graph summary <trace>`: dependency shape of each batch. `dispatch-graph check <reference> <candidate>`: equivalence check. `dispatch-graph controls <trace>`: runs the check's controls on a trace. Reports are JSON on stdout; exit 0 only when everything holds.

#![forbid(unsafe_code)]

use std::fmt;
use std::io::Write;
use std::process::ExitCode;

use tokspur_dispatch_graph::check::{Report, check};
use tokspur_dispatch_graph::controls::{Expect, controls};
use tokspur_dispatch_graph::graph::shape;
use tokspur_dispatch_graph::trace::{Trace, TraceError, parse_trace};
use tokspur_json::{Json, JsonError};

enum CliError {
    Usage,
    TraceRead {
        path: String,
        source: std::io::Error,
    },
    TraceParse {
        path: String,
        source: TraceError,
    },
    ReportDump(JsonError),
    StdoutWrite(std::io::Error),
}

impl fmt::Debug for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage => formatter.write_str("Usage"),
            Self::TraceRead { .. } => formatter.write_str("TraceRead([redacted])"),
            Self::TraceParse { source, .. } => {
                formatter.debug_tuple("TraceParse").field(source).finish()
            }
            Self::ReportDump(source) => formatter.debug_tuple("ReportDump").field(source).finish(),
            Self::StdoutWrite(_) => formatter.write_str("StdoutWrite([redacted])"),
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage => formatter.write_str(
                "usage: dispatch-graph summary <trace> | check <reference> <candidate> | controls <trace>",
            ),
            Self::TraceRead { path, source } => write!(formatter, "cannot read {path}: {source}"),
            Self::TraceParse { path, source } => write!(formatter, "{path}: {source}"),
            Self::ReportDump(source) => source.fmt(formatter),
            Self::StdoutWrite(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Usage => None,
            Self::TraceRead { source, .. } | Self::StdoutWrite(source) => Some(source),
            Self::TraceParse { source, .. } => Some(source),
            Self::ReportDump(source) => Some(source),
        }
    }
}

fn load(path: &str) -> Result<Trace, CliError> {
    let bytes = std::fs::read(path).map_err(|source| CliError::TraceRead {
        path: path.to_owned(),
        source,
    })?;
    parse_trace(&bytes).map_err(|source| CliError::TraceParse {
        path: path.to_owned(),
        source,
    })
}

fn integer(value: usize) -> Json {
    Json::integer(value as u64)
}

fn text(value: &str) -> Json {
    Json::String(value.as_bytes().to_vec())
}

fn report_json(report: &Report) -> Json {
    Json::new_object([
        ("passed", Json::Bool(report.passed())),
        ("failures", integer(report.failures)),
        (
            "listed",
            Json::Array(report.listed.iter().map(|message| text(message)).collect()),
        ),
    ])
}

fn summary(trace: &Trace) -> Json {
    let batches = trace
        .batches()
        .map(|batch| {
            let found = shape(batch);
            Json::new_object([
                ("dispatches", integer(found.dispatches)),
                ("barriers", integer(found.barriers)),
                ("dependency_edges", integer(found.edges)),
                ("depth", integer(found.depth)),
                ("widest_level", integer(found.widest)),
            ])
        })
        .collect();
    Json::new_object([
        ("producer", text(&trace.producer)),
        ("pipelines", integer(trace.pipelines.len())),
        ("batches", Json::Array(batches)),
    ])
}

fn run(args: &[String]) -> Result<(Json, bool), CliError> {
    match args {
        [command, path] if command == "summary" => Ok((summary(&load(path)?), true)),
        [command, reference, candidate] if command == "check" => {
            let report = check(&load(reference)?, &load(candidate)?);
            Ok((report_json(&report), report.passed()))
        }
        [command, path] if command == "controls" => {
            let reference = load(path)?;
            let mut held = true;
            let results = controls(&reference)
                .into_iter()
                .map(|control| {
                    let report = check(&reference, &control.candidate);
                    let ok = report.passed() == (control.expect == Expect::Pass);
                    held &= ok;
                    Json::new_object([
                        ("control", text(control.name)),
                        (
                            "expect",
                            text(if control.expect == Expect::Pass {
                                "pass"
                            } else {
                                "fail"
                            }),
                        ),
                        ("held", Json::Bool(ok)),
                        ("report", report_json(&report)),
                    ])
                })
                .collect();
            Ok((
                Json::new_object([
                    ("held", Json::Bool(held)),
                    ("controls", Json::Array(results)),
                ]),
                held,
            ))
        }
        _ => Err(CliError::Usage),
    }
}

fn write_output(json: &Json) -> Result<(), CliError> {
    let mut bytes = json.dump().map_err(CliError::ReportDump)?;
    bytes.push(b'\n');
    std::io::stdout()
        .write_all(&bytes)
        .map_err(CliError::StdoutWrite)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args).and_then(|(json, ok)| write_output(&json).map(|()| ok)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => report_error(&error),
    }
}

fn report_error(error: &CliError) -> ExitCode {
    let line = format!("dispatch-graph: {error}\n");
    if std::io::stderr().write_all(line.as_bytes()).is_err() {
        return ExitCode::from(2);
    }
    ExitCode::from(1)
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn cli_error_retains_source_and_redacts_path_in_debug() {
        let err = load("/nonexistent/secret-user-dir/trace.jsonl").unwrap_err();
        assert!(err.source().is_some());
        let dbg = format!("{err:?}");
        assert!(
            !dbg.contains("secret-user-dir"),
            "unexpected path in Debug: {dbg}"
        );
    }
}
