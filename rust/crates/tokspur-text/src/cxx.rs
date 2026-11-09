//! C++ standard-library behaviour that the chat and response ports rely on: `std::string_view::find` over bytes, and libc++'s `std::from_chars` and `std::to_chars` for `double`.

/// `std::string_view::find(needle)`: the first byte offset of `needle`.
pub(crate) fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    find_from(haystack, needle, 0)
}

/// `std::string_view::find(needle, from)`; `None` when `from` is past the end.
pub(crate) fn find_from(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    let tail = haystack.get(from..)?;
    if needle.is_empty() {
        return Some(from);
    }
    tail.windows(needle.len())
        .position(|window| window == needle)
        .map(|found| found + from)
}

/// The outcome of `std::from_chars(first, last, double&)` in its default `chars_format::general` mode, as libc++ (Apple clang 21) reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FromChars {
    /// `ec == std::errc{}`, the whole text was consumed and the value is finite.
    Value(f64),
    /// `ec == std::errc::result_out_of_range`: the decimal overflows, or it is nonzero and rounds to zero. Subnormal results are in range.
    OutOfRange,
    /// Every other outcome: `errc::invalid_argument`, a parse that stops before the end, or an `inf`/`nan` spelling. Both C++ call sites refuse these with the same message whichever of them occurred.
    Invalid,
}

/// Parses `text` exactly as libc++ `std::from_chars` does for the two call sites. Rust's `f64` parser rounds correctly, as libc++ does, and returns infinity on overflow and a signed zero on underflow, where libc++ reports `result_out_of_range`; this maps those back.
pub(crate) fn from_chars_f64(text: &str) -> FromChars {
    if !general_decimal(text.as_bytes()) {
        return FromChars::Invalid;
    }
    let Ok(value) = text.parse::<f64>() else {
        return FromChars::Invalid;
    };
    if value.is_infinite() {
        return FromChars::OutOfRange;
    }
    if value == 0.0 && nonzero_significand(text.as_bytes()) {
        return FromChars::OutOfRange;
    }
    FromChars::Value(value)
}

/// The whole of `text` matches the `chars_format::general` decimal subject sequence: `-?(digits(.digits?)?|.digits)([eE][+-]?digits)?`.
fn general_decimal(text: &[u8]) -> bool {
    let mut at = usize::from(text.first() == Some(&b'-'));
    let digits = |from: usize| {
        text[from..]
            .iter()
            .take_while(|c| c.is_ascii_digit())
            .count()
    };
    let whole = digits(at);
    at += whole;
    let mut fraction = 0;
    if text.get(at) == Some(&b'.') {
        at += 1;
        fraction = digits(at);
        at += fraction;
    }
    if whole + fraction == 0 {
        return false;
    }
    if matches!(text.get(at), Some(b'e' | b'E')) {
        at += 1;
        if matches!(text.get(at), Some(b'+' | b'-')) {
            at += 1;
        }
        let exponent = digits(at);
        if exponent == 0 {
            return false;
        }
        at += exponent;
    }
    at == text.len()
}

fn nonzero_significand(text: &[u8]) -> bool {
    text.iter()
        .take_while(|&&c| c != b'e' && c != b'E')
        .any(|c| matches!(c, b'1'..=b'9'))
}

/// `std::to_chars(first, last, value, std::chars_format::scientific)` for a finite `value > 0`: libc++'s Ryu output, the shortest round-trip digits as `d[.ddd]e±XX` with at least two exponent digits.
///
/// Rust's `{:e}` yields the same shortest, closest digits except when the value lies exactly halfway between two shortest candidates: Rust rounds that tie up, Ryu to the even digit. The midpoint test is exact.
pub(crate) fn to_chars_scientific(value: f64) -> Vec<u8> {
    let shortest = format!("{value:e}");
    let shortest = shortest.as_bytes();
    let separator = shortest
        .iter()
        .position(|&c| c == b'e')
        .unwrap_or(shortest.len());
    let mut digits: Vec<u8> = shortest[..separator]
        .iter()
        .copied()
        .filter(|&c| c != b'.')
        .collect();
    let mut exponent: i32 = 0;
    let mut negative = false;
    for &c in shortest.get(separator + 1..).unwrap_or_default() {
        if c == b'-' {
            negative = true;
        } else {
            exponent = exponent * 10 + i32::from(c - b'0');
        }
    }
    if negative {
        exponent = -exponent;
    }
    let length = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    if let Some(even) = even_tie_partner(value, &digits, exponent - (length - 1)) {
        digits = even;
    }
    let Some((&first, rest)) = digits.split_first() else {
        return shortest.to_vec();
    };
    let mut output = vec![first];
    if !rest.is_empty() {
        output.push(b'.');
        output.extend_from_slice(rest);
    }
    output.extend_from_slice(if exponent < 0 { b"e-" } else { b"e+" });
    let magnitude = exponent.unsigned_abs();
    if magnitude < 10 {
        output.push(b'0');
    }
    output.extend_from_slice(magnitude.to_string().as_bytes());
    output
}

/// The even neighbour of `digits` (scaled by `10^q`) when `value` is exactly their midpoint and the neighbour also parses back to `value`.
fn even_tie_partner(value: f64, digits: &[u8], q: i32) -> Option<Vec<u8>> {
    if digits.len() > 17 || digits.last().is_none_or(|&c| (c - b'0').is_multiple_of(2)) {
        return None;
    }
    let significand = digits
        .iter()
        .try_fold(0u64, |total, &c| Some(total * 10 + u64::from(c - b'0')))?;
    let floor = 10u64.pow(u32::try_from(digits.len() - 1).ok()?);
    for partner in [significand - 1, significand + 1] {
        if partner < floor || partner >= floor * 10 || !midpoint(value, significand.min(partner), q)
        {
            continue;
        }
        let candidate = partner.to_string().into_bytes();
        let text = format!("{partner}e{q}");
        if text
            .parse::<f64>()
            .is_ok_and(|parsed| parsed.to_bits() == value.to_bits())
        {
            return Some(candidate);
        }
    }
    None
}

/// Whether `value == (2 * low + 1) * 10^q / 2` exactly.
fn midpoint(value: f64, low: u64, q: i32) -> bool {
    // value = m * 2^k exactly; the midpoint is n * 5^q * 2^(q - 1).
    let bits = value.to_bits();
    let field = i32::try_from((bits >> 52) & 0x7ff).unwrap_or(0);
    let fraction = bits & ((1u64 << 52) - 1);
    let (m, k) = if field == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), field - 1075)
    };
    let n = 2 * low + 1;
    let shift = k.min(q - 1);
    let five = q.min(0);
    let scale = |exponent: i32| u32::try_from(exponent).unwrap_or(0);
    let left = Big::new(m)
        .times_pow5(scale(-five))
        .times_pow2(scale(k - shift));
    let right = Big::new(n)
        .times_pow5(scale(q - five))
        .times_pow2(scale(q - 1 - shift));
    left == right
}

/// A little-endian unsigned integer, just big enough for [`midpoint`].
#[derive(PartialEq, Eq)]
struct Big(Vec<u32>);

impl Big {
    fn new(value: u64) -> Self {
        let mut big = Self(vec![value as u32, (value >> 32) as u32]);
        big.trim();
        big
    }

    fn trim(&mut self) {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }

    fn times_small(mut self, factor: u32) -> Self {
        let mut carry = 0u64;
        for limb in &mut self.0 {
            let product = u64::from(*limb) * u64::from(factor) + carry;
            *limb = product as u32;
            carry = product >> 32;
        }
        if carry != 0 {
            self.0.push(carry as u32);
        }
        self
    }

    fn times_pow5(mut self, mut exponent: u32) -> Self {
        while exponent >= 13 {
            self = self.times_small(1_220_703_125);
            exponent -= 13;
        }
        self.times_small(5u32.pow(exponent))
    }

    fn times_pow2(mut self, exponent: u32) -> Self {
        let (limbs, bits) = ((exponent / 32) as usize, exponent % 32);
        if bits != 0 {
            let mut carry = 0u32;
            for limb in &mut self.0 {
                let shifted = (*limb << bits) | carry;
                carry = *limb >> (32 - bits);
                *limb = shifted;
            }
            if carry != 0 {
                self.0.push(carry);
            }
        }
        if !self.0.is_empty() {
            self.0.splice(0..0, std::iter::repeat_n(0, limbs));
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_matches_string_view_find() {
        assert_eq!(find(b"abcabc", b"bc"), Some(1));
        assert_eq!(find_from(b"abcabc", b"bc", 2), Some(4));
        assert_eq!(find_from(b"abc", b"c", 3), None);
        assert_eq!(find_from(b"abc", b"c", 9), None);
        assert_eq!(find(b"ab", b"abc"), None);
        assert_eq!(find(b"ab", b""), Some(0));
    }

    #[test]
    fn from_chars_reports_libcxx_ranges() {
        // Observed with Apple clang 21 libc++: subnormals are in range, a nonzero decimal that rounds to zero and overflow are out of range.
        assert_eq!(from_chars_f64("5e-324"), FromChars::Value(5e-324));
        assert_eq!(
            from_chars_f64("2.4703282292062328e-324"),
            FromChars::Value(5e-324)
        );
        assert_eq!(
            from_chars_f64("2.4703282292062327e-324"),
            FromChars::OutOfRange
        );
        assert_eq!(from_chars_f64("1e-400"), FromChars::OutOfRange);
        assert_eq!(from_chars_f64("-1e-400"), FromChars::OutOfRange);
        assert_eq!(from_chars_f64("0e999999"), FromChars::Value(0.0));
        assert_eq!(
            from_chars_f64("1.7976931348623158e308"),
            FromChars::Value(f64::MAX)
        );
        assert_eq!(
            from_chars_f64("1.7976931348623159e308"),
            FromChars::OutOfRange
        );
        assert_eq!(from_chars_f64("1e309"), FromChars::OutOfRange);
        assert!(matches!(from_chars_f64("-0"), FromChars::Value(zero) if zero.is_sign_negative()));
    }

    #[test]
    fn to_chars_matches_libcxx_scientific_shortest() {
        for (value, expected) in [
            (125.0, "1.25e+02"),
            (5e-324, "5e-324"),
            (1e16, "1e+16"),
            (1e-5, "1e-05"),
            (0.1, "1e-01"),
            (1.0, "1e+00"),
            (f64::MAX, "1.7976931348623157e+308"),
            (2.2250738585072014e-308, "2.2250738585072014e-308"),
            (123_456_789_012_345_680.0, "1.2345678901234568e+17"),
        ] {
            assert_eq!(to_chars_scientific(value), expected.as_bytes(), "{value:e}");
        }
    }

    #[test]
    fn to_chars_breaks_exact_ties_to_even_like_ryu() {
        // 3353589096455369/4 = 838397274113842.25 exactly: "…842.2" and "…842.3" both round-trip; libc++ and Python print the even digit.
        let tie = 3_353_589_096_455_369.0 / 4.0;
        assert_eq!(format!("{tie:e}"), "8.383972741138423e14");
        assert_eq!(to_chars_scientific(tie), b"8.383972741138422e+14");
        // 539281086611173/16 = 33705067913198.3125: the 17th digit ties.
        assert_eq!(
            to_chars_scientific(539_281_086_611_173.0 / 16.0),
            b"3.3705067913198312e+13"
        );
        // Ties already on an even digit, and near-ties, stay as Rust prints them.
        assert_eq!(to_chars_scientific(0.125), b"1.25e-01");
        assert_eq!(
            to_chars_scientific(6_707_178_192_910_739.0 / 8.0),
            b"8.383972741138424e+14"
        );
    }

    #[test]
    fn midpoint_is_exact_across_the_exponent_range() {
        assert!(midpoint(2.5, 2, 0));
        assert!(!midpoint(2.5, 1, 0));
        assert!(midpoint(0.25, 2, -1));
        // 5e19 = 5^20 * 2^19 is exactly representable: half of 10^20.
        assert!(midpoint(5e19, 0, 20));
        assert!(!midpoint(1e300, 0, 300));
        assert!(!midpoint(5e-324, 4, -324));
    }

    #[test]
    fn from_chars_refuses_partial_and_nonfinite_spellings() {
        for text in [
            "", "-", ".", "1e", "1e+", "+1", "0x10", "inf", "nan", "1 ", " 1", "1.5.2",
        ] {
            assert_eq!(from_chars_f64(text), FromChars::Invalid, "{text:?}");
        }
        assert_eq!(from_chars_f64(".5"), FromChars::Value(0.5));
        assert_eq!(from_chars_f64("5."), FromChars::Value(5.0));
    }
}
