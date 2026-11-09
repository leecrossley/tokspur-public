//! The C++ oracle: calls into the unchanged native/src parsers through the narrow shim in `cpp/oracle.cpp`, which `build.rs` compiles and links.
//!
//! This is the only module in the crate allowed `unsafe` (see the lint table in `Cargo.toml`). Every entry point follows one contract: the shim reads exactly the input span it is given, catches every C++ exception before returning (nothing unwinds across the boundary), stores the encoded outcome in a thread-local buffer and returns a pointer to it with its length. The buffer stays valid until the next oracle call on the same thread; each wrapper copies it out before returning. The encodings are documented in the shim and reproduced from the Rust port by [`crate::encode`].

#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use tokspur_json::JsonLimits;

// SAFETY: these signatures match the `extern "C"` definitions in cpp/oracle.cpp: `size_t` is `usize` and `uint8_t` is `u8` on aarch64-apple-darwin, and every pointer is to plain bytes or one `size_t`.
unsafe extern "C" {
    fn tokspur_oracle_json(
        input: *const u8,
        input_size: usize,
        bytes: usize,
        nodes: usize,
        depth: usize,
        size: *mut usize,
    ) -> *const u8;
    fn tokspur_oracle_gguf(input: *const u8, input_size: usize, size: *mut usize) -> *const u8;
    fn tokspur_oracle_response(input: *const u8, input_size: usize, size: *mut usize) -> *const u8;
}

/// Copies one outcome out of the shim's thread-local buffer.
///
/// # Safety
///
/// `data` and `size` must be the pointer and length an oracle entry point has just returned on this thread, with no oracle call in between.
unsafe fn take(data: *const u8, size: usize) -> Vec<u8> {
    if data.is_null() {
        return Vec::new();
    }
    // SAFETY: per this function's contract, `data` points to `size` initialized bytes of the shim's thread-local `std::string`, which no other code touches until the next oracle call on this thread; the bytes are copied out before this function returns.
    unsafe { std::slice::from_raw_parts(data, size) }.to_vec()
}

/// `parse_json(input, limits)` in C++: the refusal, or the dump and every number's accessors (see `json_case` in the shim).
pub fn json(input: &[u8], limits: JsonLimits) -> Vec<u8> {
    let mut size = 0;
    // SAFETY: `input` is a live slice whose pointer and length are passed together (the shim reads nothing for an empty slice) and `size` is a writable `usize`. The shim catches every exception, so the call returns normally.
    let data = unsafe {
        tokspur_oracle_json(
            input.as_ptr(),
            input.len(),
            limits.bytes,
            limits.nodes,
            limits.depth,
            &mut size,
        )
    };
    // SAFETY: `data` and `size` were just returned by an oracle call on this thread.
    unsafe { take(data, size) }
}

/// `Gguf(std::span)` in C++ and the gguf_driver summary of every accessor.
pub fn gguf(input: &[u8]) -> Vec<u8> {
    let mut size = 0;
    // SAFETY: as in `json`: a live slice with its own length, a writable `usize`, and a shim that never lets an exception escape.
    let data = unsafe { tokspur_oracle_gguf(input.as_ptr(), input.len(), &mut size) };
    // SAFETY: `data` and `size` were just returned by an oracle call on this thread.
    unsafe { take(data, size) }
}

/// One encoded `ResponseParser` case run by the C++ parser: the constructor's refusal, or a snapshot after every operation.
pub fn response(case: &[u8]) -> Vec<u8> {
    let mut size = 0;
    // SAFETY: as in `json`: a live slice with its own length, a writable `usize`, and a shim that never lets an exception escape.
    let data = unsafe { tokspur_oracle_response(case.as_ptr(), case.len(), &mut size) };
    // SAFETY: `data` and `size` were just returned by an oracle call on this thread.
    unsafe { take(data, size) }
}
