//! The outcome encoding the C++ shim (`cpp/oracle.cpp`) writes, produced here from the Rust port, and the helpers that print a mismatch.
//!
//! Little-endian `u8`, `u32` and `u64`; a byte string is a `u32` length followed by the bytes.

#![forbid(unsafe_code)]

/// The shim's `Writer`.
#[derive(Default)]
pub struct Writer {
    pub out: Vec<u8>,
}

impl Writer {
    pub fn u8(&mut self, value: u8) {
        self.out.push(value);
    }

    pub fn u32(&mut self, value: u32) {
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn u64(&mut self, value: u64) {
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    /// A length-prefixed byte string. Fuzz inputs are far below 4 GiB.
    pub fn bytes(&mut self, value: &[u8]) {
        let Ok(size) = u32::try_from(value.len()) else {
            panic!("encoded field of {} bytes exceeds u32", value.len());
        };
        self.u32(size);
        self.out.extend_from_slice(value);
    }

    /// A count or size as `u32` (`static_cast<uint32_t>` of a small size).
    pub fn len(&mut self, value: usize) {
        let Ok(value) = u32::try_from(value) else {
            panic!("encoded count {value} exceeds u32");
        };
        self.u32(value);
    }

    /// A `size_t` as `u64`.
    pub fn size(&mut self, value: usize) {
        self.u64(value as u64);
    }
}

/// Printable form of synthetic fuzz bytes for a failure message: ASCII as is, everything else as `\xNN`, truncated after `limit` bytes.
pub fn escape(bytes: &[u8], limit: usize) -> String {
    let mut text = String::new();
    for &byte in bytes.iter().take(limit) {
        match byte {
            b'\\' => text.push_str("\\\\"),
            b'\n' => text.push_str("\\n"),
            b'\r' => text.push_str("\\r"),
            b'\t' => text.push_str("\\t"),
            0x20..=0x7e => text.push(char::from(byte)),
            _ => text.push_str(&format!("\\x{byte:02x}")),
        }
    }
    if bytes.len() > limit {
        text.push_str(&format!("... ({} bytes)", bytes.len()));
    }
    text
}

/// Panics with both encodings around their first difference unless they are identical. `what` names the comparison; `input` is the synthetic case.
pub fn require_identical(what: &str, input: &[u8], cpp: &[u8], rust: &[u8]) {
    if cpp == rust {
        return;
    }
    let first = cpp
        .iter()
        .zip(rust)
        .position(|(a, b)| a != b)
        .unwrap_or(cpp.len().min(rust.len()));
    let start = first.saturating_sub(48);
    panic!(
        "{what}: C++ and Rust differ at byte {first} (C++ {} bytes, Rust {} bytes)\n\
         input: {}\n\
         C++  [{start}..]: {}\n\
         Rust [{start}..]: {}",
        cpp.len(),
        rust.len(),
        escape(input, 512),
        escape(&cpp[start..], 160),
        escape(&rust[start..], 160),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_matches_the_shim_layout() {
        let mut out = Writer::default();
        out.u8(7);
        out.u32(0x0102_0304);
        out.u64(5);
        out.bytes(b"ab");
        assert_eq!(
            out.out,
            [
                7, 4, 3, 2, 1, 5, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, b'a', b'b'
            ]
        );
    }

    #[test]
    fn escape_keeps_ascii_and_marks_the_rest() {
        assert_eq!(escape(b"a\\\n\x00\xff", 16), "a\\\\\\n\\x00\\xff");
        assert_eq!(escape(b"abcdef", 2), "ab... (6 bytes)");
    }

    #[test]
    #[should_panic(expected = "differ at byte 1")]
    fn identical_encodings_are_required() {
        require_identical("case", b"x", b"ab", b"ac");
    }
}
