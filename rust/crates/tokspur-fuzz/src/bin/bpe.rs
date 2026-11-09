//! Fuzz target: byte-level BPE (`ByteBpe` with a small fixed vocabulary) against a reference encoder, with the decode invariants of `tokspur_fuzz::bpe::check`.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| tokspur_fuzz::bpe::check(data));
