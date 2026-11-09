//! UTF-8 scalar codec from `native/src/unicode.cpp` (`unicode_scalars`, `scalar_utf8`). It lives here because JSON validates every string with it; `tokspur-text` re-exports it next to NFC normalization.

use std::fmt;

/// Byte and scalar bound shared by every Unicode routine (`work_limit`).
pub const WORK_LIMIT: usize = 8 * 1024 * 1024;

/// The C++ `FormatError` raised by the scalar codec. `Display` is the exact C++ message: it can reach HTTP error bodies and response-parser errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnicodeError {
    /// `unicode_scalars` refused its input.
    Decode(&'static str),
    /// `scalar_utf8` was given a surrogate or a value above U+10FFFF.
    Encode,
}

impl fmt::Display for UnicodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(reason) => formatter.write_str(reason),
            Self::Encode => formatter.write_str("invalid Unicode scalar"),
        }
    }
}

impl std::error::Error for UnicodeError {}

/// Decodes strict UTF-8 into scalars, refusing overlong forms, surrogates and values above U+10FFFF.
pub fn unicode_scalars(utf8: &[u8]) -> Result<Vec<u32>, UnicodeError> {
    if utf8.len() > WORK_LIMIT {
        return Err(UnicodeError::Decode("Unicode input exceeds limit"));
    }
    let mut output = Vec::with_capacity(utf8.len());
    let mut i = 0;
    while i < utf8.len() {
        let first = utf8[i];
        i += 1;
        if first < 128 {
            output.push(u32::from(first));
            continue;
        }
        let tail = match first {
            0xc2..=0xdf => 1,
            0xe0..=0xef => 2,
            0xf0..=0xf4 => 3,
            _ => 0,
        };
        if tail == 0 || tail > utf8.len() - i {
            return Err(UnicodeError::Decode("invalid UTF-8 input"));
        }
        let mut point = u32::from(first) & ((1u32 << (6 - tail)) - 1);
        for _ in 0..tail {
            let next = utf8[i];
            i += 1;
            if next & 0xc0 != 0x80 {
                return Err(UnicodeError::Decode("invalid UTF-8 continuation"));
            }
            point = (point << 6) | u32::from(next & 63);
        }
        if (tail == 1 && point < 0x80)
            || (tail == 2 && point < 0x800)
            || (tail == 3 && point < 0x10000)
            || point > 0x10ffff
            || (0xd800..=0xdfff).contains(&point)
        {
            return Err(UnicodeError::Decode("non-scalar or overlong UTF-8"));
        }
        output.push(point);
    }
    Ok(output)
}

/// Encodes one scalar as UTF-8.
pub fn scalar_utf8(point: u32) -> Result<Vec<u8>, UnicodeError> {
    let mut output = Vec::with_capacity(4);
    push_scalar_utf8(&mut output, point)?;
    Ok(output)
}

/// Appends one scalar's UTF-8 bytes, as `output += scalar_utf8(point)` does.
pub fn push_scalar_utf8(output: &mut Vec<u8>, point: u32) -> Result<(), UnicodeError> {
    if point > 0x10ffff || (0xd800..=0xdfff).contains(&point) {
        return Err(UnicodeError::Encode);
    }
    if point < 0x80 {
        output.push(point as u8);
        return Ok(());
    }
    let tail: u32 = if point < 0x800 {
        1
    } else if point < 0x10000 {
        2
    } else {
        3
    };
    let prefix: u32 = match tail {
        1 => 0xc0,
        2 => 0xe0,
        _ => 0xf0,
    };
    output.push((prefix | (point >> (tail * 6))) as u8);
    for remaining in (1..=tail).rev() {
        output.push((0x80 | ((point >> ((remaining - 1) * 6)) & 63)) as u8);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_refuses_each_malformed_class_with_its_message() {
        let cases: [(&[u8], &str); 5] = [
            (b"\x80", "invalid UTF-8 input"),
            (b"\xc3", "invalid UTF-8 input"),
            (b"\xc3\x28", "invalid UTF-8 continuation"),
            (b"\xe0\x80\x80", "non-scalar or overlong UTF-8"),
            (b"\xed\xa0\x80", "non-scalar or overlong UTF-8"),
        ];
        for (bytes, message) in cases {
            assert_eq!(unicode_scalars(bytes).unwrap_err().to_string(), message);
        }
    }

    #[test]
    fn encode_matches_std_for_every_scalar_boundary() {
        for point in [
            0, 0x7f, 0x80, 0x7ff, 0x800, 0xd7ff, 0xe000, 0xffff, 0x10000, 0x10ffff,
        ] {
            let expected = char::from_u32(point).unwrap().to_string().into_bytes();
            assert_eq!(scalar_utf8(point).unwrap(), expected);
            assert_eq!(unicode_scalars(&expected).unwrap(), vec![point]);
        }
        assert_eq!(scalar_utf8(0xd800), Err(UnicodeError::Encode));
        assert_eq!(scalar_utf8(0x110000), Err(UnicodeError::Encode));
    }

    #[test]
    fn decode_refuses_input_over_the_work_limit() {
        let input = vec![b'a'; WORK_LIMIT + 1];
        assert_eq!(
            unicode_scalars(&input),
            Err(UnicodeError::Decode("Unicode input exceeds limit"))
        );
    }
}
