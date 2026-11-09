//! JSON targets: properties of `tokspur-json` on its own ([`check`]) and the byte-for-byte comparison with the unchanged `native/src/json.cpp` ([`check_differential`]).

#![forbid(unsafe_code)]

use tokspur_json::{
    Json, JsonError, JsonLimits, from_chars_f64, parse_json, parse_json_limited, to_chars_shortest,
};

#[cfg(feature = "cpp-oracle")]
use crate::encode::require_identical;
use crate::encode::{Writer, escape};
#[cfg(feature = "cpp-oracle")]
use crate::oracle;

/// Every message `parse_json` and `dump` can raise: json.cpp and the unicode.cpp checks it calls.
pub(crate) const MESSAGES: [&str; 26] = [
    "JSON input byte limit exceeded",
    "trailing JSON data",
    "truncated JSON",
    "invalid JSON literal",
    "invalid JSON Unicode escape",
    "unescaped JSON control character",
    "missing JSON low surrogate",
    "invalid JSON low surrogate",
    "unpaired JSON low surrogate",
    "invalid JSON escape",
    "invalid JSON number",
    "missing JSON number digits",
    "JSON structural limit exceeded",
    "expected JSON array delimiter",
    "expected JSON object key",
    "duplicate JSON object key",
    "expected JSON object colon",
    "expected JSON object delimiter",
    "invalid JSON value",
    "JSON output byte limit exceeded",
    "JSON output nesting limit exceeded",
    "Unicode input exceeds limit",
    "invalid UTF-8 input",
    "invalid UTF-8 continuation",
    "non-scalar or overlong UTF-8",
    "invalid Unicode scalar",
];

const STRUCTURAL: JsonError = JsonError::Parse("JSON structural limit exceeded");
const INPUT_BYTES: JsonError = JsonError::Parse("JSON input byte limit exceeded");

fn known(error: &JsonError, input: &[u8]) {
    let message = error.to_string();
    assert!(
        MESSAGES.contains(&message.as_str()),
        "unknown JSON error {message:?} for {}",
        escape(input, 512)
    );
}

/// The number of values and the deepest nesting level (the root is 0), as the parser's limits count them.
fn shape(value: &Json, depth: usize, nodes: &mut usize, deepest: &mut usize) {
    *nodes += 1;
    *deepest = (*deepest).max(depth);
    match value {
        Json::Array(items) => {
            for item in items {
                shape(item, depth + 1, nodes, deepest);
            }
        }
        Json::Object(fields) => {
            for (_, item) in fields {
                shape(item, depth + 1, nodes, deepest);
            }
        }
        Json::Null | Json::Bool(_) | Json::Number(_) | Json::String(_) => {}
    }
}

/// `unsigned_integer` on every number agrees with an independent reading of the lexeme: decimal digits only, within `u64`.
fn check_numbers(value: &Json, input: &[u8]) {
    match value {
        Json::Number(number) => {
            let lexeme = number.lexeme();
            let expected = if lexeme.bytes().all(|c| c.is_ascii_digit()) {
                lexeme.parse::<u64>().ok()
            } else {
                None
            };
            let range = JsonError::Access("JSON unsigned integer out of range or nonintegral");
            assert_eq!(
                value.unsigned_integer(),
                expected.ok_or(range),
                "unsigned_integer of {lexeme:?} in {}",
                escape(input, 512)
            );
        }
        Json::Array(items) => items.iter().for_each(|item| check_numbers(item, input)),
        Json::Object(fields) => fields
            .iter()
            .for_each(|(_, item)| check_numbers(item, input)),
        Json::Null | Json::Bool(_) | Json::String(_) => {}
    }
}

/// `parse_json` on arbitrary bytes. A refusal must carry a known message and stay a refusal under tighter limits. An accepted document must dump, the dump must re-parse to an equal value and dump identically again, the dump can at most triple the input (a two-byte escape becomes `\u00XX`), and the document must parse under exactly its own byte, node and depth counts but fail with the exact limit message one below any of them.
pub fn check(data: &[u8]) {
    let value = match parse_json(data) {
        Ok(value) => value,
        Err(error) => {
            known(&error, data);
            let tight = parse_json_limited(data, JsonLimits::new(data.len(), 1, 0));
            assert!(
                tight.is_err(),
                "tighter limits accepted {}",
                escape(data, 512)
            );
            return;
        }
    };
    let dumped = match value.dump() {
        Ok(dumped) => dumped,
        Err(error) => panic!(
            "parsed value does not dump ({error}): {}",
            escape(data, 512)
        ),
    };
    match parse_json(&dumped) {
        Ok(again) => {
            assert!(
                again == value,
                "dump re-parses to another value: {}",
                escape(data, 512)
            );
            assert_eq!(
                again.dump().as_deref(),
                Ok(dumped.as_slice()),
                "dump is not stable"
            );
        }
        Err(error) => panic!(
            "dump {} does not re-parse ({error}) for {}",
            escape(&dumped, 512),
            escape(data, 512)
        ),
    }
    assert!(
        dumped.len() <= 3 * data.len(),
        "dump grew {} bytes to {}",
        data.len(),
        dumped.len()
    );
    check_numbers(&value, data);

    let (mut nodes, mut depth) = (0, 0);
    shape(&value, 0, &mut nodes, &mut depth);
    let exact = JsonLimits::new(data.len(), nodes, depth);
    let accepted = parse_json_limited(data, exact);
    assert!(
        accepted.as_ref() == Ok(&value),
        "exact limits {exact:?} refused {}: {:?}",
        escape(data, 512),
        accepted.err()
    );
    let fewer_nodes = JsonLimits::new(data.len(), nodes - 1, depth);
    assert_eq!(
        parse_json_limited(data, fewer_nodes).err(),
        Some(STRUCTURAL),
        "{fewer_nodes:?} for {}",
        escape(data, 512)
    );
    if depth > 0 {
        let shallower = JsonLimits::new(data.len(), nodes, depth - 1);
        assert_eq!(
            parse_json_limited(data, shallower).err(),
            Some(STRUCTURAL),
            "{shallower:?} for {}",
            escape(data, 512)
        );
    }
    let shorter = JsonLimits::new(data.len() - 1, nodes, depth);
    assert_eq!(
        parse_json_limited(data, shorter).err(),
        Some(INPUT_BYTES),
        "{shorter:?} for {}",
        escape(data, 512)
    );
}

/// The shim's `numbers`: every number in document order with its lexeme, `unsigned_integer`, and the `std::from_chars`/`std::to_chars` results `from_chars_f64` and `to_chars_shortest` reproduce.
fn numbers(out: &mut Writer, value: &Json) {
    match value {
        Json::Number(number) => {
            let lexeme = number.lexeme();
            out.u8(b'n');
            out.bytes(lexeme.as_bytes());
            match value.unsigned_integer() {
                Ok(integer) => {
                    out.u8(b'O');
                    out.u64(integer);
                }
                Err(error) => {
                    out.u8(b'E');
                    out.bytes(error.to_string().as_bytes());
                }
            }
            let Some(parsed) = from_chars_f64(lexeme) else {
                out.u8(b'N');
                return;
            };
            out.u8(b'O');
            out.u64(parsed.to_bits());
            match to_chars_shortest(parsed) {
                Some(text) => {
                    out.u8(b'O');
                    out.bytes(text.as_bytes());
                }
                None => out.u8(b'E'),
            }
        }
        Json::Array(items) => items.iter().for_each(|item| numbers(out, item)),
        Json::Object(fields) => fields.iter().for_each(|(_, item)| numbers(out, item)),
        Json::Null | Json::Bool(_) | Json::String(_) => {}
    }
}

/// The shim's `json_case`, from the Rust port.
pub fn outcome(input: &[u8], limits: JsonLimits) -> Vec<u8> {
    let mut out = Writer::default();
    match parse_json_limited(input, limits) {
        Err(error) => {
            out.u8(b'E');
            out.bytes(error.to_string().as_bytes());
        }
        Ok(value) => {
            out.u8(b'O');
            match value.dump() {
                Ok(dumped) => {
                    out.u8(b'O');
                    out.bytes(&dumped);
                }
                Err(error) => {
                    out.u8(b'E');
                    out.bytes(error.to_string().as_bytes());
                }
            }
            numbers(&mut out, &value);
        }
    }
    out.out
}

/// The limits each input is compared under: the defaults, and tight ones derived from its length so that libFuzzer reaches the structural limits.
pub fn limit_sets(data: &[u8]) -> [JsonLimits; 2] {
    let size = data.len();
    [
        JsonLimits::default(),
        JsonLimits::new(size, 1 + size % 17, size % 5),
    ]
}

/// The same bytes through json.cpp and tokspur-json: acceptance, the exact refusal message, the dump bytes, and every number's `unsigned_integer`, `std::from_chars` and `std::to_chars` results must be identical.
#[cfg(feature = "cpp-oracle")]
pub fn check_differential(data: &[u8]) {
    for limits in limit_sets(data) {
        let cpp = oracle::json(data, limits);
        let rust = outcome(data, limits);
        require_identical(&format!("parse_json {limits:?}"), data, &cpp, &rust);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCUMENTS: [&[u8]; 12] = [
        // `\x5c` is the JSON escape's backslash.
        b"{\"text\":\"\x5cufeffusing caf\x5cu00e9\x5cu0000\",\"ids\":[0,1,4294967295],\"flag\":true,\"nothing\":null}",
        b"\"\x5cud83e\x5cudd89\"",
        br#""\"\\\/\b\f\n\r\t""#,
        b" \n [true, false, null, -1.25e+20] \t",
        b"18446744073709551615",
        b"18446744073709551616",
        b"[-0, 0.1, 1e400, 1e-400, 5e-324, 123456789012345678901234567890]",
        b"[[[[[]]]]]",
        b"{\"x\":1,\"x\":2}",
        b"\"\xed\xa0\x80\"",
        b"[1,]",
        b"",
    ];

    #[test]
    fn properties_hold_for_the_json_test_documents() {
        for document in DOCUMENTS {
            check(document);
        }
    }

    #[cfg(feature = "cpp-oracle")]
    #[test]
    fn the_cpp_oracle_agrees_on_the_json_test_documents() {
        for document in DOCUMENTS {
            check_differential(document);
        }
    }

    #[test]
    fn outcome_encodes_numbers_in_document_order() {
        let encoded = outcome(b"[7,{\"a\":-1.5}]", JsonLimits::default());
        let mut expected = Writer::default();
        expected.u8(b'O');
        expected.u8(b'O');
        expected.bytes(b"[7,{\"a\":-1.5}]");
        for (lexeme, integer, bits, text) in [
            ("7", Ok(7), 7f64.to_bits(), "7"),
            ("-1.5", Err(()), (-1.5f64).to_bits(), "-1.5"),
        ] {
            expected.u8(b'n');
            expected.bytes(lexeme.as_bytes());
            match integer {
                Ok(value) => {
                    expected.u8(b'O');
                    expected.u64(value);
                }
                Err(()) => {
                    expected.u8(b'E');
                    expected.bytes(b"JSON unsigned integer out of range or nonintegral");
                }
            }
            expected.u8(b'O');
            expected.u64(bits);
            expected.u8(b'O');
            expected.bytes(text.as_bytes());
        }
        assert_eq!(encoded, expected.out);
    }
}
