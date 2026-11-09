//! Fuzz target: `ResponseParser` fed arbitrary chunks, then `finish`, with the stream and split-invariance properties of `tokspur_fuzz::response::check`.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| tokspur_fuzz::response::check(data));
