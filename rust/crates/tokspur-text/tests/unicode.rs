//! `native/tests/test_unicode.cpp`, check for check. Where the C++ `rejected` only asks for some `FormatError`, these tests also pin the exact message. Non-ASCII literals are spelled as escapes of the C++ source's UTF-8 bytes.

use tokspur_text::unicode::{
    NormalizeError, UnicodeError, normalize_nfc9, scalar_utf8, unicode_scalars,
};

fn nfc(text: &str) -> Result<Vec<u8>, NormalizeError> {
    normalize_nfc9(text.as_bytes())
}

#[test]
fn combining_accent_composes() {
    // "NFC combining accent"
    assert_eq!(nfc("cafe\u{301}").unwrap(), "caf\u{e9}".as_bytes());
}

#[test]
fn singleton_decompositions_apply() {
    // "NFC singleton decompositions": ANGSTROM SIGN and OHM SIGN.
    assert_eq!(nfc("\u{212b}\u{2126}").unwrap(), "\u{c5}\u{3a9}".as_bytes());
}

#[test]
fn compatibility_forms_are_left_alone() {
    // "must not do compatibility normalization"
    assert_eq!(
        nfc("\u{2460} \u{ff11}").unwrap(),
        "\u{2460} \u{ff11}".as_bytes()
    );
}

#[test]
fn excluded_composites_stay_decomposed() {
    // "NFC excluded composites"
    assert_eq!(
        nfc("\u{340}\u{341}\u{344}").unwrap(),
        "\u{300}\u{301}\u{308}\u{301}".as_bytes()
    );
}

#[test]
fn marks_unassigned_in_unicode9_keep_class_zero() {
    // "Unicode 9 unknown combining class": U+0EBA is unassigned in Unicode 9, so U+0EB8 (class 118) is not reordered past it.
    assert_eq!(nfc("\u{eb8}\u{eba}").unwrap(), "\u{eb8}\u{eba}".as_bytes());
}

#[test]
fn later_unicode_classes_are_not_inherited() {
    // "do not inherit later Unicode classes"
    assert_eq!(
        nfc("\u{10f48}\u{10f4b}").unwrap(),
        "\u{10f48}\u{10f4b}".as_bytes()
    );
}

#[test]
fn hangul_jamo_compose_into_a_syllable() {
    // "Hangul composition"
    assert_eq!(
        nfc("\u{1100}\u{1161}\u{11a8}").unwrap(),
        "\u{ac01}".as_bytes()
    );
}

#[test]
fn leading_nonstarters_are_canonically_ordered() {
    // "leading nonstarter ordering"
    assert_eq!(nfc("\u{301}\u{323}").unwrap(), "\u{323}\u{301}".as_bytes());
}

#[test]
fn embedded_nul_survives() {
    // "embedded NUL"
    assert_eq!(normalize_nfc9(b"\0a\0").unwrap(), b"\0a\0");
}

#[test]
fn every_scalar_round_trips_through_utf8() {
    // "Unicode scalar UTF-8 round trip", for all 1,112,064 scalars.
    for point in (0..0xd800).chain(0xe000..0x11_0000) {
        let bytes = scalar_utf8(point).unwrap();
        assert_eq!(unicode_scalars(&bytes).unwrap(), [point], "U+{point:04X}");
    }
}

#[test]
fn invalid_utf8_is_refused_with_the_codec_message() {
    let cases: [(&[u8], &str); 6] = [
        (b"\xc0\x80", "invalid UTF-8 input"),
        (b"\xe0\x80\x80", "non-scalar or overlong UTF-8"),
        (b"\xed\xa0\x80", "non-scalar or overlong UTF-8"),
        (b"\xf4\x90\x80\x80", "non-scalar or overlong UTF-8"),
        (b"\xc2", "invalid UTF-8 input"),
        (b"\x80", "invalid UTF-8 input"),
    ];
    for (bytes, message) in cases {
        let error = normalize_nfc9(bytes).unwrap_err();
        assert_eq!(error, NormalizeError::Scalar(UnicodeError::Decode(message)));
        assert_eq!(error.to_string(), message);
    }
}

#[test]
fn scalar_utf8_refuses_surrogates_and_values_past_the_last_scalar() {
    for point in [0xd800, 0x11_0000] {
        let error = scalar_utf8(point).unwrap_err();
        assert_eq!(error, UnicodeError::Encode);
        assert_eq!(error.to_string(), "invalid Unicode scalar");
    }
}

#[test]
fn long_combining_sequence_is_idempotent() {
    // "long combining sequence idempotence"
    let mut long_marks = String::from("a");
    for _ in 0..10_000 {
        long_marks.push_str("\u{301}\u{323}");
    }
    let normalized = normalize_nfc9(long_marks.as_bytes()).unwrap();
    assert_eq!(normalize_nfc9(&normalized).unwrap(), normalized);
    // The stable sort moves every U+0323 (class 220) before every U+0301 (230); the first U+0323 composes with the starter.
    let mut expected = String::from("\u{1ea1}");
    expected.push_str(&"\u{323}".repeat(9_999));
    expected.push_str(&"\u{301}".repeat(10_000));
    assert_eq!(normalized, expected.as_bytes());
}

/// `from_hex`: whitespace-separated hexadecimal scalars, each parsed as `std::from_chars(..., 16)` does (digits only, no sign or prefix). `None` is the C++ "invalid normalization test scalar".
fn from_hex(value: &[u8]) -> Option<Vec<u8>> {
    let mut output = Vec::new();
    // `std::istringstream >> std::string` splits on C `isspace`.
    let words =
        value.split(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r'));
    for hex in words.filter(|word| !word.is_empty()) {
        if !hex.iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        let point = u32::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok()?;
        output.extend_from_slice(&scalar_utf8(point).ok()?);
    }
    Some(output)
}

/// `standard_tests`: every row of the official Unicode 9.0.0 `NormalizationTest.txt` (18,722 rows). Set `TOKSPUR_NORMALIZATION_TEST` to its path; the file is never downloaded or committed.
#[test]
#[ignore = "needs TOKSPUR_NORMALIZATION_TEST=/path/to/NormalizationTest.txt (Unicode 9.0.0)"]
fn uax15_normalization_test_rows_hold() {
    let path = std::env::var_os("TOKSPUR_NORMALIZATION_TEST")
        .expect("set TOKSPUR_NORMALIZATION_TEST to the Unicode 9.0.0 NormalizationTest.txt");
    let file = std::fs::read(path).expect("cannot open Unicode normalization tests");
    let mut cases = 0;
    // `std::getline` splits on '\n' only; a final empty segment is no line.
    let mut lines: Vec<&[u8]> = file.split(|&byte| byte == b'\n').collect();
    if lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    for (index, line) in lines.into_iter().enumerate() {
        let row = index + 1;
        let line = match line.iter().position(|&byte| byte == b'#') {
            Some(comment) => &line[..comment],
            None => line,
        };
        if line.is_empty() || line[0] == b'@' {
            continue;
        }
        let mut columns: Vec<Vec<u8>> = Vec::with_capacity(5);
        let mut cursor = 0;
        for _ in 0..5 {
            let end = line[cursor..]
                .iter()
                .position(|&byte| byte == b';')
                .unwrap_or_else(|| panic!("invalid normalization test row {row}"));
            columns.push(
                from_hex(&line[cursor..cursor + end]).expect("invalid normalization test scalar"),
            );
            cursor += end + 1;
        }
        let normalize = |column: usize| normalize_nfc9(&columns[column]).unwrap();
        assert_eq!(
            normalize(0),
            columns[1],
            "UAX15 source -> NFC failed, row {row}"
        );
        assert_eq!(
            normalize(1),
            columns[1],
            "UAX15 NFC idempotence failed, row {row}"
        );
        assert_eq!(
            normalize(2),
            columns[1],
            "UAX15 NFD -> NFC failed, row {row}"
        );
        assert_eq!(
            normalize(3),
            columns[3],
            "UAX15 NFKC -> NFC failed, row {row}"
        );
        assert_eq!(
            normalize(4),
            columns[3],
            "UAX15 NFKD -> NFC failed, row {row}"
        );
        cases += 1;
    }
    assert_eq!(
        cases, 18_722,
        "normative rows in NormalizationTest-9.0.0.txt"
    );
    println!("normative rows: {cases}");
}
