//! Fuzz target: `Gguf::from_bytes` on arbitrary bytes, with the layout and accessor properties of `tokspur_fuzz::gguf::check`.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| tokspur_fuzz::gguf::check(data));
