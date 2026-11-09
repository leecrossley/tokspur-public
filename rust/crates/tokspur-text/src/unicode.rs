//! Unicode 9 NFC from `native/src/unicode.cpp`.
//!
//! The scalar codec (`unicode_scalars`, `scalar_utf8`) lives in `tokspur-json`, which validates every JSON string with it, and is re-exported here unchanged. `normalize_nfc9` keeps the C++ work limits, its stable sort per run of non-starters and its composition rule, byte for byte.

use std::fmt;

pub use tokspur_json::{UnicodeError, WORK_LIMIT, push_scalar_utf8, scalar_utf8, unicode_scalars};

/// `tokspur::unicode9_data`: the tables `build.rs` generates from `native/data/unicode9.inc`, the include the C++ compiles, so both read the same data. The build fails unless each table is in the sorted order the binary searches below rely on.
mod unicode9_data {
    /// `ClassRange`: scalars `first..=last` share canonical combining class `value`. Scalars outside every range have class 0.
    pub(super) struct ClassRange {
        pub(super) first: u32,
        pub(super) last: u32,
        pub(super) value: u8,
    }

    /// `Decomposition`: the canonical decomposition of `point`; `second` is 0 for a singleton.
    pub(super) struct Decomposition {
        pub(super) point: u32,
        pub(super) first: u32,
        pub(super) second: u32,
    }

    /// `Composition`: the primary composite `point` of the scalars in `pair`, keyed as `(first << 21) | second`.
    pub(super) struct Composition {
        pub(super) pair: u64,
        pub(super) point: u32,
    }

    include!(concat!(env!("OUT_DIR"), "/unicode9.rs"));
}

const HANGUL_START: u32 = 0xac00;
const HANGUL_COUNT: u32 = 19 * 21 * 28;

/// The C++ `FormatError`s `normalize_nfc9` raises. `Display` is the exact C++ message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizeError {
    /// The scalar codec refused: `unicode_scalars` on the input, or `scalar_utf8` on a composed scalar (unreachable with the pinned data). Transparent, as the C++ lets the codec's `FormatError` propagate.
    Scalar(UnicodeError),
    /// `decompose` passed its recursion depth or the scalar work limit.
    Decompose,
    /// The normalized UTF-8 would pass the byte work limit.
    Encode,
}

impl fmt::Display for NormalizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(error) => error.fmt(formatter),
            Self::Decompose => formatter.write_str("Unicode normalization work limit exceeded"),
            Self::Encode => formatter.write_str("normalized UTF-8 exceeds limit"),
        }
    }
}

impl std::error::Error for NormalizeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scalar(error) => Some(error),
            Self::Decompose | Self::Encode => None,
        }
    }
}

impl From<UnicodeError> for NormalizeError {
    fn from(error: UnicodeError) -> Self {
        Self::Scalar(error)
    }
}

/// `combining`: the canonical combining class of `point`.
fn combining(point: u32) -> u8 {
    let table = &unicode9_data::COMBINING;
    // `std::lower_bound` by `range.last < point`.
    let found = table.partition_point(|range| range.last < point);
    match table.get(found) {
        Some(range) if range.first <= point => range.value,
        _ => 0,
    }
}

/// `decompose`: appends the full canonical decomposition of `point`.
fn decompose(point: u32, output: &mut Vec<u32>, depth: u32) -> Result<(), NormalizeError> {
    if depth > 32 || output.len() >= WORK_LIMIT {
        return Err(NormalizeError::Decompose);
    }
    if (HANGUL_START..HANGUL_START + HANGUL_COUNT).contains(&point) {
        let index = point - HANGUL_START;
        output.push(0x1100 + index / (21 * 28));
        output.push(0x1161 + (index / 28) % 21);
        if !index.is_multiple_of(28) {
            output.push(0x11a7 + index % 28);
        }
        return Ok(());
    }
    let table = &unicode9_data::DECOMPOSITION;
    // `std::lower_bound` by `item.point < point`.
    let found = table.partition_point(|item| item.point < point);
    let Some(item) = table.get(found).filter(|item| item.point == point) else {
        output.push(point);
        return Ok(());
    };
    decompose(item.first, output, depth + 1)?;
    if item.second != 0 {
        decompose(item.second, output, depth + 1)?;
    }
    Ok(())
}

/// `compose`: the primary composite of the pair, or 0 when there is none.
fn compose(first_point: u32, second_point: u32) -> u32 {
    if (0x1100..0x1100 + 19).contains(&first_point) && (0x1161..0x1161 + 21).contains(&second_point)
    {
        return HANGUL_START + ((first_point - 0x1100) * 21 + second_point - 0x1161) * 28;
    }
    if (HANGUL_START..HANGUL_START + HANGUL_COUNT).contains(&first_point)
        && (first_point - HANGUL_START).is_multiple_of(28)
        && second_point > 0x11a7
        && second_point < 0x11a7 + 28
    {
        return first_point + second_point - 0x11a7;
    }
    let key = (u64::from(first_point) << 21) | u64::from(second_point);
    let table = &unicode9_data::COMPOSITION;
    // `std::lower_bound` by `item.pair < key`.
    let found = table.partition_point(|item| item.pair < key);
    match table.get(found) {
        Some(item) if item.pair == key => item.point,
        _ => 0,
    }
}

/// `normalize_nfc9`: canonical decomposition, canonical ordering and canonical composition with the Unicode 9 tables.
///
/// Composition follows the C++ rule exactly: a scalar composes with the last starter when `previous < current || previous == 0`, where `previous` is the class of the last scalar appended (not of one composed away).
pub fn normalize_nfc9(utf8: &[u8]) -> Result<Vec<u8>, NormalizeError> {
    let points = unicode_scalars(utf8)?;
    let mut decomposed = Vec::with_capacity(points.len());
    for point in points {
        decompose(point, &mut decomposed, 0)?;
    }
    // Stable sorting per non-starter run avoids quadratic insertion sorting on adversarial long sequences of combining marks.
    let mut cursor = 0;
    while cursor < decomposed.len() {
        if combining(decomposed[cursor]) == 0 {
            cursor += 1;
        }
        let start = cursor;
        while cursor < decomposed.len() && combining(decomposed[cursor]) != 0 {
            cursor += 1;
        }
        // `std::stable_sort` by class; `sort_by_key` is stable too, and a stable sort has exactly one result.
        decomposed[start..cursor].sort_by_key(|&point| combining(point));
    }
    let mut composed: Vec<u32> = Vec::with_capacity(decomposed.len());
    // `work_limit` doubles as "no starter yet", as in the C++.
    let mut starter = WORK_LIMIT;
    let mut previous: u8 = 0;
    for &point in &decomposed {
        let current = combining(point);
        let combination = if starter != WORK_LIMIT && (previous < current || previous == 0) {
            compose(composed[starter], point)
        } else {
            0
        };
        if combination != 0 {
            composed[starter] = combination;
        } else {
            if current == 0 {
                starter = composed.len();
            }
            composed.push(point);
            previous = current;
        }
    }
    let mut output = Vec::new();
    let mut bytes = Vec::with_capacity(4);
    for point in composed {
        bytes.clear();
        push_scalar_utf8(&mut bytes, point)?;
        if bytes.len() > WORK_LIMIT - output.len() {
            return Err(NormalizeError::Encode);
        }
        output.extend_from_slice(&bytes);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The C++ include, read independently of `build.rs`.
    const INCLUDE: &str = include_str!("../../../../native/data/unicode9.inc");

    fn number(field: &str) -> u64 {
        match field.strip_prefix("0x") {
            Some(hex) => u64::from_str_radix(hex.trim_end_matches("ULL"), 16).unwrap(),
            None => field.parse().unwrap(),
        }
    }

    /// Every `    {...},` entry of the named `inline constexpr` table.
    fn include_entries(declaration: &str) -> Vec<Vec<u64>> {
        let mut inside = false;
        let mut entries = Vec::new();
        for line in INCLUDE.lines() {
            if line.starts_with("inline constexpr ") {
                inside = line == declaration;
            } else if line == "};" {
                inside = false;
            } else if inside {
                let entry = line
                    .strip_prefix("    {")
                    .unwrap()
                    .strip_suffix("},")
                    .unwrap();
                entries.push(entry.split(", ").map(number).collect());
            }
        }
        entries
    }

    #[test]
    fn tables_hold_exactly_the_cpp_include_entries() {
        let combining = include_entries("inline constexpr ClassRange combining[] = {");
        let decomposition = include_entries("inline constexpr Decomposition decomposition[] = {");
        let composition = include_entries("inline constexpr Composition composition[] = {");
        // Lines 12-344, 348-2407 and 2411-3350 of the include.
        assert_eq!(
            (combining.len(), decomposition.len(), composition.len()),
            (333, 2060, 940)
        );
        assert_eq!(unicode9_data::COMBINING.len(), combining.len());
        assert_eq!(unicode9_data::DECOMPOSITION.len(), decomposition.len());
        assert_eq!(unicode9_data::COMPOSITION.len(), composition.len());
        for (range, entry) in unicode9_data::COMBINING.iter().zip(&combining) {
            assert_eq!(
                vec![
                    u64::from(range.first),
                    u64::from(range.last),
                    u64::from(range.value)
                ],
                *entry
            );
        }
        for (item, entry) in unicode9_data::DECOMPOSITION.iter().zip(&decomposition) {
            assert_eq!(
                vec![
                    u64::from(item.point),
                    u64::from(item.first),
                    u64::from(item.second)
                ],
                *entry
            );
        }
        for (item, entry) in unicode9_data::COMPOSITION.iter().zip(&composition) {
            assert_eq!(vec![item.pair, u64::from(item.point)], *entry);
        }
    }

    #[test]
    fn combining_reads_every_range_edge() {
        for range in &unicode9_data::COMBINING {
            let before = range.first - 1;
            let inside_previous = unicode9_data::COMBINING
                .iter()
                .any(|other| (other.first..=other.last).contains(&before));
            if !inside_previous {
                assert_eq!(combining(before), 0, "U+{before:04X}");
            }
            assert_eq!(combining(range.first), range.value);
            assert_eq!(combining(range.last), range.value);
        }
        assert_eq!(combining(0), 0);
        assert_eq!(combining(0x41), 0);
        assert_eq!(combining(0x300), 230);
        assert_eq!(combining(0x34f), 0, "U+034F is a gap between ranges");
        assert_eq!(combining(0x1e94a), 7);
        assert_eq!(combining(0x1e94b), 0);
        assert_eq!(combining(0x10ffff), 0);
    }

    #[test]
    fn decompose_expands_hangul_and_recurses_through_the_table() {
        let mut output = Vec::new();
        decompose(0xac00, &mut output, 0).unwrap();
        assert_eq!(output, [0x1100, 0x1161]);
        output.clear();
        decompose(0xd7a3, &mut output, 0).unwrap();
        assert_eq!(output, [0x1112, 0x1175, 0x11c2]);
        output.clear();
        // U+1F82 -> U+1F02 U+0345 -> U+1F00 U+0300 U+0345 -> U+03B1 U+0313 U+0300 U+0345.
        decompose(0x1f82, &mut output, 0).unwrap();
        assert_eq!(output, [0x3b1, 0x313, 0x300, 0x345]);
        output.clear();
        decompose(0x212b, &mut output, 0).unwrap();
        assert_eq!(output, [0x41, 0x30a]);
        output.clear();
        decompose(0x41, &mut output, 0).unwrap();
        assert_eq!(output, [0x41]);
    }

    #[test]
    fn decompose_refuses_past_its_depth_or_the_work_limit() {
        let mut output = Vec::new();
        assert_eq!(
            decompose(0x41, &mut output, 33),
            Err(NormalizeError::Decompose)
        );
        assert_eq!(decompose(0x41, &mut output, 32), Ok(()));
        let mut full = vec![0u32; WORK_LIMIT];
        assert_eq!(
            decompose(0x41, &mut full, 0),
            Err(NormalizeError::Decompose)
        );
        full.pop();
        // The bound is checked before appending, so a Hangul syllable can still take the vector two past the limit, as in the C++.
        assert_eq!(decompose(0xac01, &mut full, 0), Ok(()));
        assert_eq!(full.len(), WORK_LIMIT + 2);
    }

    #[test]
    fn compose_covers_hangul_and_the_table() {
        assert_eq!(compose(0x1100, 0x1161), 0xac00);
        assert_eq!(compose(0x1112, 0x1175), 0xd788);
        assert_eq!(
            compose(0x1113, 0x1161),
            0,
            "U+1113 is not a modern leading jamo"
        );
        assert_eq!(compose(0x1100, 0x1176), 0);
        assert_eq!(compose(0xac00, 0x11a8), 0xac01);
        assert_eq!(compose(0xac00, 0x11a7), 0, "U+11A7 is not a trailing jamo");
        assert_eq!(compose(0xac00, 0x11c3), 0);
        assert_eq!(
            compose(0xac01, 0x11a8),
            0,
            "only LV syllables take a trailing jamo"
        );
        assert_eq!(compose(0x41, 0x30a), 0xc5);
        assert_eq!(compose(0x3b1, 0x313), 0x1f00);
        assert_eq!(
            compose(0x915, 0x93c),
            0,
            "U+0958 is a composition exclusion"
        );
        assert_eq!(compose(0x41, 0x41), 0);
        for item in &unicode9_data::COMPOSITION {
            let (first, second) = ((item.pair >> 21) as u32, (item.pair & 0x1f_ffff) as u32);
            assert_eq!(compose(first, second), item.point);
        }
    }

    #[test]
    fn errors_print_the_cpp_messages() {
        assert_eq!(
            NormalizeError::Decompose.to_string(),
            "Unicode normalization work limit exceeded"
        );
        assert_eq!(
            NormalizeError::Encode.to_string(),
            "normalized UTF-8 exceeds limit"
        );
        assert_eq!(
            NormalizeError::Scalar(UnicodeError::Decode("invalid UTF-8 input")).to_string(),
            "invalid UTF-8 input"
        );
        assert_eq!(
            NormalizeError::from(UnicodeError::Encode).to_string(),
            "invalid Unicode scalar"
        );
    }

    #[test]
    fn input_over_the_work_limit_is_refused_before_normalizing() {
        let long = vec![b'a'; WORK_LIMIT + 1];
        assert_eq!(
            normalize_nfc9(&long),
            Err(NormalizeError::Scalar(UnicodeError::Decode(
                "Unicode input exceeds limit"
            )))
        );
    }

    #[test]
    fn ascii_at_the_work_limit_normalizes_to_itself() {
        let full = vec![b'a'; WORK_LIMIT];
        assert_eq!(normalize_nfc9(&full).unwrap(), full);
    }

    #[test]
    fn expanding_decompositions_trip_the_scalar_limit() {
        // U+1F82 (3 bytes) decomposes to 4 scalars.
        let expanding = "\u{1f82}".repeat(WORK_LIMIT / 3);
        assert_eq!(
            normalize_nfc9(expanding.as_bytes()),
            Err(NormalizeError::Decompose)
        );
    }

    #[test]
    fn growing_output_trips_the_byte_limit() {
        // U+0958 (3 bytes) stays U+0915 U+093C (6 bytes): an exclusion.
        let doubling = "\u{958}".repeat(WORK_LIMIT / 3);
        assert_eq!(
            normalize_nfc9(doubling.as_bytes()),
            Err(NormalizeError::Encode)
        );
    }
}
