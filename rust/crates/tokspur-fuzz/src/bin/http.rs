//! Fuzz target: HTTP request framing (`tokspur-server` `Connection:: read_request`) over a socket pair, written whole and at arbitrary split points (`tokspur_fuzz::http::check`).

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| tokspur_fuzz::http::check(data));
