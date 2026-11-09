//! Port of `native/tests/test_http_branch8.cpp` (private HTTP branch8 build, and its staged-QSA variant `test-http_stage_branch8` under `qsa-stage-v8`).
//!
//! As in C++, the V8 suite runs first and shares the check counter. `TOKSPUR_FAKE_TARGET` swaps in another fake target for the worker checks, e.g. the C++ `test-http_branch8` binary of the same feature level.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

#[path = "http_private/branch8.rs"]
mod branch8;
#[path = "http_private/verify8.rs"]
mod verify8;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_none_or(|first| first != "target") && args.iter().any(|arg| arg == "--list") {
        // `cargo test -- --list` probes harnessless tests; report one case.
        println!("http_branch8: test");
        return;
    }
    std::process::exit(branch8::main(&args));
}
