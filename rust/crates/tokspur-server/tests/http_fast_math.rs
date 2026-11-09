//! Port of `native/tests/test_http_fast_math.cpp` (private HTTP fast-math build).
//!
//! As in C++, the branch8 (and V8) suites run first and share the check counter. `TOKSPUR_FAKE_TARGET` swaps in another fake target for the worker checks, e.g. the C++ `test-http_fast_math` binary.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

#[path = "http_private/branch8.rs"]
mod branch8;
#[path = "http_private/fast_math.rs"]
mod fast_math;
#[path = "http_private/verify8.rs"]
mod verify8;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_none_or(|first| first != "target") && args.iter().any(|arg| arg == "--list") {
        // `cargo test -- --list` probes harnessless tests; report one case.
        println!("http_fast_math: test");
        return;
    }
    std::process::exit(fast_math::main(&args));
}
