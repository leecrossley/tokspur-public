//! Chunked inputs for the streaming targets (HTTP framing, response parser).
//!
//! A target reads a separator byte from its input and splits the rest on it: the chunks are fed or written one by one, and their concatenation is the whole stream. libFuzzer moves a split point by inserting or deleting one separator, and choosing another separator byte keeps every byte value available to the stream.

#![forbid(unsafe_code)]

/// The chunks of `data` between `separator` bytes, empty chunks included.
pub fn chunks(data: &[u8], separator: u8) -> Vec<&[u8]> {
    data.split(|&byte| byte == separator).collect()
}

/// Where each chunk ends in the concatenated stream.
pub fn boundaries(chunks: &[&[u8]]) -> Vec<usize> {
    chunks
        .iter()
        .scan(0, |end, chunk| {
            *end += chunk.len();
            Some(*end)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_split_on_every_separator() {
        assert_eq!(
            chunks(b"ab|c||d", b'|'),
            [b"ab".as_slice(), b"c", b"", b"d"]
        );
        assert_eq!(chunks(b"", b'|'), [b"".as_slice()]);
        assert_eq!(boundaries(&chunks(b"ab|c||d", b'|')), [2, 3, 3, 4]);
    }
}
