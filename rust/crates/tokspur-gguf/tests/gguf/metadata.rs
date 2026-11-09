//! `metadata()` in `native/tests/test_gguf.cpp` (34 checks).

use tokspur_gguf::Gguf;

use crate::support::{
    Bytes, Data, fixture, integer, key, keys_fixture, number, refusal, replace, string, text,
};

fn model_bytes() -> Bytes {
    keys_fixture(&[
        key("general.architecture", 8, text("toy")),
        key("test.count", 2, integer(42, 2)),
        key("test.negative", 5, integer(0xffff_ffff, 4)),
        key("test.bool", 7, integer(1, 1)),
    ])
}

#[test]
fn typed_scalars_read_back() {
    let bytes = model_bytes();
    let model = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(
        model.string_value("general.architecture").unwrap(),
        b"toy",
        "string metadata"
    );
    assert_eq!(
        model.unsigned_value("test.count").unwrap(),
        42,
        "unsigned metadata"
    );
}

#[test]
fn unsigned_value_refuses_negative_boolean_and_missing_items() {
    let bytes = model_bytes();
    let model = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(
        refusal(model.unsigned_value("test.negative")),
        "negative GGUF count"
    );
    assert_eq!(
        refusal(model.unsigned_value("test.bool")),
        "GGUF metadata is not an integer"
    );
    assert_eq!(
        refusal(model.unsigned_value("test.missing")),
        "missing GGUF metadata test.missing"
    );
}

#[test]
fn string_value_refuses_an_integer() {
    let bytes = model_bytes();
    let model = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(
        refusal(model.string_value("test.count")),
        "GGUF metadata is not a string: test.count"
    );
}

#[test]
fn find_returns_nothing_for_a_missing_key() {
    let bytes = model_bytes();
    let model = Gguf::from_bytes(&bytes).unwrap();
    assert!(model.find("not.found").is_none(), "missing metadata lookup");
}

#[test]
fn malformed_keys_are_refused() {
    for name in ["", "Bad", ".bad", "bad.", "bad..key", "bad-key"] {
        let data = keys_fixture(&[key(name, 0, integer(0, 1))]);
        assert_eq!(
            refusal(Gguf::from_bytes(&data)),
            "invalid GGUF metadata key",
            "{name:?}"
        );
    }
}

#[test]
fn duplicate_keys_are_refused() {
    let bytes = keys_fixture(&[key("a", 0, integer(0, 1)), key("a", 0, integer(1, 1))]);
    assert_eq!(
        refusal(Gguf::from_bytes(&bytes)),
        "duplicate GGUF metadata key"
    );
}

#[test]
fn booleans_other_than_zero_or_one_are_refused() {
    let bytes = keys_fixture(&[key("a", 7, integer(2, 1))]);
    assert_eq!(refusal(Gguf::from_bytes(&bytes)), "invalid GGUF boolean");
}

#[test]
fn unknown_metadata_types_are_refused() {
    let bytes = keys_fixture(&[key("a", 13, Bytes::new())]);
    assert_eq!(
        refusal(Gguf::from_bytes(&bytes)),
        "unknown GGUF metadata type"
    );
}

#[test]
fn invalid_utf8_strings_are_refused() {
    let invalid: [&[u8]; 6] = [
        b"\xc0\x80",
        b"\xed\xa0\x80",
        b"\xf4\x90\x80\x80",
        b"\xe0\x80\x80",
        b"\xc2",
        b"\xf5\x80\x80\x80",
    ];
    for value in invalid {
        let bytes = keys_fixture(&[key("a", 8, text(value))]);
        assert_eq!(
            refusal(Gguf::from_bytes(&bytes)),
            "invalid UTF-8 in GGUF string",
            "{value:02x?}"
        );
    }
}

#[test]
fn valid_unicode_strings_read_back() {
    let bytes = keys_fixture(&[key("a", 8, text("α🦉"))]);
    let utf8 = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(
        utf8.string_value("a").unwrap(),
        "α🦉".as_bytes(),
        "valid Unicode rejected"
    );
}

fn string_array() -> Bytes {
    let mut array = Bytes::new();
    number(&mut array, 8, 4);
    number(&mut array, 2, 8);
    string(&mut array, b"first");
    string(&mut array, b"second");
    array
}

#[test]
fn string_arrays_keep_their_descriptor_and_borrow_their_strings() {
    let bytes = keys_fixture(&[key("a", 9, string_array())]);
    let strings = Gguf::from_bytes(&bytes).unwrap();
    let item = strings.find("a").unwrap();
    assert!(
        item.elements == 2 && item.element_type == 8,
        "array descriptor"
    );
    assert_eq!(
        strings.string_array("a").unwrap(),
        [b"first".as_slice(), b"second".as_slice()],
        "borrowed string array"
    );
}

#[test]
fn string_array_refuses_scalars_and_missing_items() {
    let bytes = model_bytes();
    let model = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(
        refusal(model.string_array("test.count")),
        "GGUF metadata is not a string array"
    );
    assert_eq!(
        refusal(model.string_array("missing")),
        "GGUF metadata is not a string array"
    );
}

#[test]
fn array_counts_over_the_limit_are_refused() {
    let mut array = string_array();
    replace(&mut array, 4, u64::MAX, 8);
    let bytes = keys_fixture(&[key("a", 9, array)]);
    assert_eq!(refusal(Gguf::from_bytes(&bytes)), "invalid GGUF array");
}

#[test]
fn boolean_array_elements_are_validated() {
    let mut array = Bytes::new();
    number(&mut array, 7, 4);
    number(&mut array, 1, 8);
    number(&mut array, 2, 1);
    let bytes = keys_fixture(&[key("a", 9, array)]);
    assert_eq!(refusal(Gguf::from_bytes(&bytes)), "invalid GGUF boolean");
}

#[test]
fn arrays_nested_past_eight_levels_are_refused() {
    let mut array = Bytes::new();
    for _ in 0..10 {
        number(&mut array, 9, 4);
        number(&mut array, 1, 8);
    }
    number(&mut array, 0, 4);
    number(&mut array, 0, 8);
    let bytes = keys_fixture(&[key("a", 9, array)]);
    assert_eq!(
        refusal(Gguf::from_bytes(&bytes)),
        "GGUF array nesting exceeds limit"
    );
}

#[test]
fn unsupported_alignments_are_refused() {
    for alignment in [0, 3, 65537] {
        let bytes = keys_fixture(&[key("general.alignment", 4, integer(alignment, 4))]);
        assert_eq!(
            refusal(Gguf::from_bytes(&bytes)),
            "unsupported GGUF alignment",
            "{alignment}"
        );
    }
}

#[test]
fn explicit_alignment_places_the_data() {
    let bytes = fixture(
        &[key("general.alignment", 4, integer(64, 4))],
        &[Data::default()],
        64,
    );
    let aligned = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(aligned.data_offset() % 64, 0, "explicit alignment");
}
