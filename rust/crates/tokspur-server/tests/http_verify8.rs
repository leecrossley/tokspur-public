//! Port of `native/tests/test_http_verify8.cpp` (private HTTP V8 build).
//!
//! Harnessless, like the C++. `TOKSPUR_FAKE_TARGET` swaps in another fake target for the worker checks, e.g. the C++ `test-http_verify8` binary.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

#[path = "http_private/verify8.rs"]
mod verify8;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_none_or(|first| first != "target") && args.iter().any(|arg| arg == "--list") {
        // `cargo test -- --list` probes harnessless tests; report one case.
        println!("http_verify8: test");
        return;
    }
    std::process::exit(verify8::main(&args));
}
