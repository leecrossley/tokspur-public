//! Differential fuzz target: native/src/gguf.cpp against tokspur-gguf on the same bytes, through the gguf_driver summary of every accessor (`tokspur_fuzz::gguf::check_differential`). Any difference aborts the run.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| tokspur_fuzz::gguf::check_differential(data));
