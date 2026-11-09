//! `typed_metadata()` in `native/tests/test_gguf.cpp` (37 checks).

use tokspur_gguf::Gguf;

use crate::support::{Bytes, integer, key, keys_fixture, number, refusal, replace};

fn floats() -> Bytes {
    keys_fixture(&[
        key("eps", 6, integer(u64::from(1e-6f32.to_bits()), 4)),
        key("base", 12, integer(10_000_000.0f64.to_bits(), 8)),
        key("nan", 6, integer(0x7fc0_0000, 4)),
        key("inf", 12, integer(0x7ff0_0000_0000_0000, 8)),
    ])
}

#[test]
fn floating_values_keep_their_precision() {
    let bytes = floats();
    let file = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.floating_value("eps").unwrap(),
        f64::from(1e-6f32),
        "float metadata precision"
    );
    assert_eq!(
        file.floating_value("base").unwrap(),
        10_000_000.0,
        "double metadata precision"
    );
}

#[test]
fn floating_value_refuses_nonfinite_and_missing_items() {
    let bytes = floats();
    let file = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(
        refusal(file.floating_value("nan")),
        "nonfinite GGUF metadata"
    );
    assert_eq!(
        refusal(file.floating_value("inf")),
        "nonfinite GGUF metadata"
    );
    assert_eq!(
        refusal(file.floating_value("missing")),
        "GGUF metadata is not floating point"
    );
}

#[test]
fn unsigned_array_refuses_a_float() {
    let bytes = floats();
    let file = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(
        refusal(file.unsigned_array("eps")),
        "GGUF metadata is not an integer array"
    );
}

#[test]
fn every_integer_array_type_reads_back_and_signed_ones_refuse_negatives() {
    let types: [(u64, usize); 8] = [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (5, 4),
        (10, 8),
        (11, 8),
    ];
    for (r#type, width) in types {
        let mut values = Bytes::new();
        number(&mut values, r#type, 4);
        number(&mut values, 3, 8);
        number(&mut values, 0, width);
        number(&mut values, 1, width);
        number(&mut values, 42, width);
        let bytes = keys_fixture(&[key("a", 9, values.clone())]);
        let array = Gguf::from_bytes(&bytes).unwrap();
        assert_eq!(
            array.unsigned_array("a").unwrap(),
            [0, 1, 42],
            "typed integer array {type}"
        );
        assert_eq!(
            refusal(array.floating_value("a")),
            "GGUF metadata is not floating point"
        );
        assert_eq!(
            refusal(array.unsigned_array("missing")),
            "GGUF metadata is not an integer array"
        );
        if matches!(r#type, 1 | 3 | 5 | 11) {
            replace(&mut values, 12, u64::MAX, width);
            let bytes = keys_fixture(&[key("a", 9, values)]);
            let negative = Gguf::from_bytes(&bytes).unwrap();
            assert_eq!(
                refusal(negative.unsigned_array("a")),
                "negative GGUF array count",
                "{type}"
            );
        }
    }
}

#[test]
fn unsigned_arrays_keep_the_full_width() {
    let mut values = Bytes::new();
    number(&mut values, 10, 4);
    number(&mut values, 1, 8);
    number(&mut values, u64::MAX, 8);
    let bytes = keys_fixture(&[key("a", 9, values)]);
    let maximum = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(
        maximum.unsigned_array("a").unwrap()[0],
        u64::MAX,
        "full-width unsigned array"
    );
}

#[test]
fn unsigned_array_refuses_noninteger_elements() {
    let mut values = Bytes::new();
    number(&mut values, 6, 4);
    number(&mut values, 0, 8);
    let bytes = keys_fixture(&[key("a", 9, values)]);
    let noninteger = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(
        refusal(noninteger.unsigned_array("a")),
        "GGUF array has noninteger elements"
    );
}

#[test]
fn empty_integer_arrays_read_back_empty() {
    let mut values = Bytes::new();
    number(&mut values, 5, 4);
    number(&mut values, 0, 8);
    let bytes = keys_fixture(&[key("a", 9, values)]);
    let empty = Gguf::from_bytes(&bytes).unwrap();
    assert!(
        empty.unsigned_array("a").unwrap().is_empty(),
        "empty integer array"
    );
}
