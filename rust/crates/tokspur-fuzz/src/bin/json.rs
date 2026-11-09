//! Fuzz target: `parse_json_limited` on arbitrary bytes, with the dump, re-parse and exact-limit properties of `tokspur_fuzz::json::check`.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| tokspur_fuzz::json::check(data));
