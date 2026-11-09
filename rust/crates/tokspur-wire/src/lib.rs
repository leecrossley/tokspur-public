//! Worker wire admission and the owned worker, ported one to one from `native/src/serve_verify8.cpp` and `native/src/owned_worker.cpp`.
//!
//! Cargo features mirror the C++ build macros (see `Cargo.toml`).

#![forbid(unsafe_code)]

pub mod qsa;
#[cfg(feature = "verify8")]
pub mod verify8;

pub mod mtp_wire;
pub mod worker;
