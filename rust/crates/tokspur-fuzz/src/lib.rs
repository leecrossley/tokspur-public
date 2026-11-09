//! libFuzzer targets for the Rust port (development tooling, never part of a shipped binary). Targets live in `src/bin/`; `native/rust.mk rust-fuzz` builds them with sanitizer-coverage instrumentation on the pinned stable toolchain and runs each for a bounded time. `native/rust.mk` documents the run settings.
//!
//! Each target is a thin `fuzz_target!` around one `check` function here, so the same checks also run, without fuzzing flags, in this crate's unit tests. A check panics on the first violated property; libFuzzer reports the panic as a crash and saves the input. Inputs are synthetic.
//!
//! [`oracle`] is the only module allowed `unsafe`: the FFI to the unchanged C++ parsers that the differential targets compare against. The crate's own lint table in `Cargo.toml` explains the exception.

#![deny(unsafe_code)]

pub mod bpe;
pub mod encode;
pub mod gguf;
pub mod http;
pub mod json;
#[cfg(feature = "cpp-oracle")]
pub mod oracle;
pub mod response;
pub mod split;
#[cfg(feature = "verify8")]
pub mod wire;
