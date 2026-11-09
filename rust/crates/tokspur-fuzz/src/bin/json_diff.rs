//! Differential fuzz target: native/src/json.cpp against tokspur-json on the same bytes and limits (`tokspur_fuzz::json::check_differential`). Any difference in acceptance, refusal message, dump bytes or number accessors aborts the run.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| tokspur_fuzz::json::check_differential(data));
