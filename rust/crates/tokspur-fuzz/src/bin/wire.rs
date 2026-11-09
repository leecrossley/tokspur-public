//! Fuzz target: worker readiness, private V8 and branch8 reply admission and the QSA dispatch formula on arbitrary JSON (`tokspur_fuzz::wire::check`). Needs the `verify8` feature or a later level; `native/rust.mk rust-fuzz` builds it once per level.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| tokspur_fuzz::wire::check(data));
