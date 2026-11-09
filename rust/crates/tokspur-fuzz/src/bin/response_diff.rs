//! Differential fuzz target: native/src/response.cpp against tokspur-text's `ResponseParser` on the same manifest, limits and feeds, snapshot after snapshot (`tokspur_fuzz::response::check_differential`). Any difference aborts the run.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| tokspur_fuzz::response::check_differential(data));
