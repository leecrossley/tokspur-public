//! Port of `native/tests/test_gguf.cpp` (one module per C++ check group), the Rust-only file-reading and redaction checks, and the ignored differential runs against the C++ reader (`rust/tools/diff/gguf.sh`).
//!
//! Every module is test-only code; `cfg(test)` (always set for an integration test) lets clippy's `allow-*-in-tests` settings cover their helpers too.

#![forbid(unsafe_code)]

#[cfg(test)]
mod differential;
#[cfg(test)]
mod files;
#[cfg(test)]
mod metadata;
#[cfg(test)]
mod parsing;
#[cfg(test)]
mod reading;
#[cfg(test)]
mod redaction;
#[cfg(test)]
mod seeds;
#[cfg(test)]
mod summary;
#[cfg(test)]
mod support;
#[cfg(test)]
mod tensors;
#[cfg(test)]
mod typed_metadata;
