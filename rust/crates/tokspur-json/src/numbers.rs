//! Exact equivalents of the libc++ `std::from_chars`/`std::to_chars` behaviour the C++ protocol code relies on for JSON number lexemes.

/// `std::from_chars(first, last, double)` over a whole JSON number lexeme, returning `None` where the C++ checks `ec != std::errc{}` or a partial parse would fail.
///
/// libc++ reports `result_out_of_range` when a nonzero decimal rounds to zero or overflows to infinity; Rust's parser returns `0.0`/`inf` silently, so both cases are refused here explicitly. Both parsers round correctly to nearest-even, so every accepted value is bit-identical.
pub fn from_chars_f64(lexeme: &str) -> Option<f64> {
    let bytes = lexeme.as_bytes();
    if !is_json_number(bytes) {
        return None;
    }
    let value: f64 = lexeme.parse().ok()?;
    if value.is_infinite() {
        return None;
    }
    let significand = bytes.iter().take_while(|&&c| c != b'e' && c != b'E');
    let nonzero = significand.into_iter().any(|&c| matches!(c, b'1'..=b'9'));
    if value == 0.0 && nonzero {
        return None;
    }
    Some(value)
}

/// `std::from_chars(first, last, uint32_t)` over the whole text: ASCII digits only, no sign, refusing overflow.
pub fn from_chars_u32(text: &str) -> Option<u32> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    bytes.iter().try_fold(0u32, |value, digit| {
        value.checked_mul(10)?.checked_add(u32::from(digit - b'0'))
    })
}

/// `std::from_chars(first, last, int64_t)` over the whole text: an optional `-` then ASCII digits, refusing overflow.
pub fn from_chars_i64(text: &str) -> Option<i64> {
    let (negative, digits) = match text.as_bytes() {
        [b'-', rest @ ..] => (true, rest),
        all => (false, all),
    };
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    // Accumulate negatively so i64::MIN parses, as from_chars accepts it.
    let magnitude = digits.iter().try_fold(0i64, |value, digit| {
        value.checked_mul(10)?.checked_sub(i64::from(digit - b'0'))
    })?;
    if negative {
        Some(magnitude)
    } else {
        magnitude.checked_neg()
    }
}

/// The strict JSON number grammar `parse_json` admits. `Number` lexemes always satisfy it; checking keeps `from_chars_f64` total on arbitrary text.
fn is_json_number(bytes: &[u8]) -> bool {
    let mut i = 0;
    let digits_from = |mut i: usize| {
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        (i, i > start)
    };
    if bytes.get(i) == Some(&b'-') {
        i += 1;
    }
    match bytes.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => i = digits_from(i).0,
        _ => return false,
    }
    if bytes.get(i) == Some(&b'.') {
        let (next, any) = digits_from(i + 1);
        if !any {
            return false;
        }
        i = next;
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let (next, any) = digits_from(i);
        if !any {
            return false;
        }
        i = next;
    }
    i == bytes.len()
}

/// `std::to_chars(first, last, double)` with no format or precision: the shortest round-trip digits, printed in `%f` or `%e` style, whichever is shorter, preferring `%f` on a tie. `%e` exponents have at least two digits.
///
/// The digits come from Rust's shortest round-trip formatting, which, like libc++'s Ryu, selects the closest shortest candidate. When `%f` wins for an integer value, libc++ prints the exact integer (not the shortest digits padded with zeros); so does this. Checked against a libc++ corpus by `tests/numbers_differential.rs`.
pub fn to_chars_shortest(value: f64) -> Option<String> {
    if !value.is_finite() {
        return None;
    }
    let negative = value.is_sign_negative();
    let (digits, exponent) = shortest_digits(value.abs())?;
    let count = i64::try_from(digits.len()).ok()?;
    let sign = if negative { "-" } else { "" };

    let exponent_digits = exponent.unsigned_abs().to_string();
    let exponent_text = if exponent_digits.len() < 2 {
        format!("0{exponent_digits}")
    } else {
        exponent_digits
    };
    let scientific_len =
        count + i64::from(count > 1) + 2 + i64::try_from(exponent_text.len()).ok()?;
    let fixed_len = if exponent >= 0 {
        if count <= exponent + 1 {
            exponent + 1
        } else {
            count + 1
        }
    } else {
        2 + (-exponent - 1) + count
    };

    if fixed_len <= scientific_len {
        let fixed = if exponent >= 0 {
            let integer_len = usize::try_from(exponent + 1).ok()?;
            if digits.len() <= integer_len {
                // The value is an integer here (below 2^53 the shortest digits would otherwise need a fraction to round-trip). libc++ prints its exact value, which differs from zero padding at or above 2^53. Fixed only wins below 1e23, so u128 holds it exactly.
                (value.abs() as u128).to_string()
            } else {
                format!("{}.{}", &digits[..integer_len], &digits[integer_len..])
            }
        } else {
            let zeros = usize::try_from(-exponent - 1).ok()?;
            format!("0.{}{digits}", "0".repeat(zeros))
        };
        Some(format!("{sign}{fixed}"))
    } else {
        let (first, rest) = digits.split_at(1);
        let point = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        Some(format!(
            "{sign}{first}{point}e{exponent_sign}{exponent_text}"
        ))
    }
}

/// The shortest significand digits (no trailing zeros) and decimal exponent that parse back to `value`, choosing the closest such decimal and breaking exact ties to an even last digit, as libc++'s Ryu does.
///
/// Rust's `{:.N e}` rounds the exact binary value half-to-even, so the first precision whose nearest decimal round-trips is the libc++ answer. At a power of two the lower round-trip gap is half the upper one, so the nearest decimal can fall just outside while the next one up round-trips; that neighbour is tried too.
fn shortest_digits(value: f64) -> Option<(String, i64)> {
    if value == 0.0 {
        return Some(("0".to_owned(), 0));
    }
    for precision in 0..17 {
        let nearest = format!("{value:.precision$e}");
        let (digits, exponent) = split_scientific(&nearest)?;
        if round_trips(&digits, exponent, value) {
            return Some(strip_zeros(digits, exponent));
        }
        let (up, up_exponent) = increment(&digits, exponent);
        if round_trips(&up, up_exponent, value) {
            return Some(strip_zeros(up, up_exponent));
        }
    }
    let (digits, exponent) = split_scientific(&format!("{value:.16e}"))?;
    Some(strip_zeros(digits, exponent))
}

fn split_scientific(text: &str) -> Option<(String, i64)> {
    let (mantissa, exponent) = text.split_once('e')?;
    Some((
        mantissa.chars().filter(char::is_ascii_digit).collect(),
        exponent.parse().ok()?,
    ))
}

fn round_trips(digits: &str, exponent: i64, value: f64) -> bool {
    let (first, rest) = digits.split_at(1);
    format!("{first}.{rest}0e{exponent}")
        .parse::<f64>()
        .is_ok_and(|parsed| parsed == value)
}

/// Adds one unit in the last digit, carrying into a new leading digit.
fn increment(digits: &str, exponent: i64) -> (String, i64) {
    let mut bytes = digits.as_bytes().to_vec();
    for byte in bytes.iter_mut().rev() {
        if *byte == b'9' {
            *byte = b'0';
        } else {
            *byte += 1;
            return (bytes.into_iter().map(char::from).collect(), exponent);
        }
    }
    bytes.insert(0, b'1');
    bytes.pop();
    (bytes.into_iter().map(char::from).collect(), exponent + 1)
}

fn strip_zeros(mut digits: String, exponent: i64) -> (String, i64) {
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    (digits, exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_chars_breaks_exact_ties_to_an_even_digit() {
        // 870865041176563.25 is exactly halfway between two 16-digit candidates; libc++ prints .2, Rust's `{:e}` alone prints .3.
        assert_eq!(
            to_chars_shortest(f64::from_bits(0x4308_c060_8b69_bf9a)).as_deref(),
            Some("870865041176563.2")
        );
    }

    #[test]
    fn to_chars_prints_exact_integers_at_or_above_two_to_the_53() {
        assert_eq!(
            to_chars_shortest(f64::from_bits(0x4400_1fdd_3812_f7eb)).as_deref(),
            Some("37180494777312698368")
        );
    }

    #[test]
    fn from_chars_refuses_underflow_and_overflow_like_libcxx() {
        // Observed with Apple clang 21 libc++ on this host (build/rust-diff/fromchars).
        assert_eq!(from_chars_f64("1e-400"), None);
        assert_eq!(from_chars_f64("-1e-400"), None);
        assert_eq!(from_chars_f64("2.4703282292062327e-324"), None);
        assert_eq!(from_chars_f64("2.4703282292062328e-324"), Some(5e-324));
        assert_eq!(from_chars_f64("1e309"), None);
        assert_eq!(from_chars_f64("1.7976931348623159e308"), None);
        assert_eq!(from_chars_f64("1.7976931348623158e308"), Some(f64::MAX));
        assert_eq!(from_chars_f64("1e-320"), Some(1e-320));
    }

    #[test]
    fn from_chars_accepts_every_zero_spelling() {
        for text in ["0", "-0", "0e5", "0e-999999", "0.000", "-0.0E+0"] {
            assert_eq!(from_chars_f64(text), Some(0.0), "{text}");
        }
        assert!(from_chars_f64("-0").unwrap().is_sign_negative());
    }

    #[test]
    fn from_chars_refuses_text_outside_the_json_grammar() {
        for text in [
            "", "-", "+1", "1e", "1.", ".5", "inf", "nan", "01", "1e+", "0x10",
        ] {
            assert_eq!(from_chars_f64(text), None, "{text}");
        }
    }

    #[test]
    fn integer_parsers_follow_from_chars_ranges() {
        assert_eq!(from_chars_u32("4294967295"), Some(u32::MAX));
        assert_eq!(from_chars_u32("4294967296"), None);
        assert_eq!(from_chars_u32("+1"), None);
        assert_eq!(from_chars_u32("-0"), None);
        assert_eq!(from_chars_i64("-9223372036854775808"), Some(i64::MIN));
        assert_eq!(from_chars_i64("9223372036854775808"), None);
        assert_eq!(from_chars_i64("+5"), None);
    }

    #[test]
    fn to_chars_matches_libcxx_shortest_layouts() {
        let cases = [
            (0.0, "0"),
            (-0.0, "-0"),
            (1.0, "1"),
            (0.7, "0.7"),
            (100.0, "100"),
            (1e-5, "1e-05"),
            (1e-4, "1e-04"),
            (1e-3, "0.001"),
            (0.123, "0.123"),
            (1e5, "1e+05"),
            (1e4, "10000"),
            (123456.0, "123456"),
            (1.5e-300, "1.5e-300"),
            (0.1 + 0.2, "0.30000000000000004"),
            (5e-324, "5e-324"),
            (1e100, "1e+100"),
        ];
        for (value, expected) in cases {
            assert_eq!(
                to_chars_shortest(value).as_deref(),
                Some(expected),
                "{value:e}"
            );
        }
        assert_eq!(to_chars_shortest(f64::NAN), None);
    }
}
