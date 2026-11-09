//! `tensors()` in `native/tests/test_gguf.cpp` (26 checks).

use tokspur_gguf::{Gguf, checked_add, checked_mul};

use crate::support::{Data, data, default_fixture, fixture, refusal, replace};

#[test]
fn tensor_offset_overflow_is_refused() {
    let mut huge_offset = default_fixture();
    replace(&mut huge_offset, 54, 0u64.wrapping_sub(32), 8);
    assert_eq!(
        refusal(Gguf::from_bytes(&huge_offset)),
        "GGUF offset overflow"
    );
}

#[test]
fn malformed_tensor_infos_are_refused() {
    let cases = [
        (data("", &[4], 0, 0, 16), "empty or duplicate tensor name"),
        (data("a", &[], 0, 0, 16), "unsupported GGUF tensor rank"),
        (
            data("a", &[1, 1, 1, 1, 1], 0, 0, 16),
            "unsupported GGUF tensor rank",
        ),
        (data("a", &[0], 0, 0, 16), "zero tensor dimension"),
        (
            data("a", &[3], 12, 0, 16),
            "quantized tensor row is not block-aligned",
        ),
        (
            data("a", &[1], 999, 0, 16),
            "unsupported GGUF tensor encoding 999",
        ),
        (data("a", &[4], 0, 1, 16), "unaligned GGUF tensor offset"),
        (data("a", &[5], 0, 0, 16), "truncated GGUF tensor data"),
        (data("a", &[u64::MAX, 2], 0, 0, 0), "GGUF shape overflow"),
        (data("a b", &[4], 0, 0, 16), "non-ASCII tensor name"),
    ];
    for (bad, message) in cases {
        let bytes = fixture(&[], &[bad], 32);
        assert_eq!(refusal(Gguf::from_bytes(&bytes)), message);
    }
}

#[test]
fn duplicate_tensor_names_are_refused() {
    let bytes = fixture(&[], &[Data::default(), Data::default()], 32);
    assert_eq!(
        refusal(Gguf::from_bytes(&bytes)),
        "empty or duplicate tensor name"
    );
}

#[test]
fn overlapping_tensors_are_refused() {
    let bytes = fixture(&[], &[Data::default(), data("second", &[4], 0, 0, 16)], 32);
    assert_eq!(
        refusal(Gguf::from_bytes(&bytes)),
        "overlapping GGUF tensors"
    );
}

#[test]
fn separated_tensors_are_accepted() {
    let bytes = fixture(&[], &[Data::default(), data("second", &[4], 0, 32, 16)], 32);
    let separated = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(separated.tensors().len(), 2, "separated tensors");
}

#[test]
fn block_encodings_give_their_byte_counts() {
    // (encoding, elements per block, bytes per block)
    let blocks: [[u64; 3]; 9] = [
        [0, 1, 4],
        [1, 1, 2],
        [8, 32, 34],
        [12, 256, 144],
        [13, 256, 176],
        [14, 256, 210],
        [20, 32, 18],
        [23, 256, 136],
        [30, 1, 2],
    ];
    for block in blocks {
        let tensor = data(
            "weight",
            &[block[1], 3, 2],
            block[0] as u32,
            0,
            block[2] * 6,
        );
        let bytes = fixture(&[], &[tensor], 32);
        let file = Gguf::from_bytes(&bytes).unwrap();
        assert_eq!(
            file.tensors()[0].bytes,
            block[2] * 6,
            "block byte count {block:?}"
        );
    }
}

#[test]
fn checked_arithmetic_refuses_overflow() {
    assert_eq!(refusal(checked_add(u64::MAX, 1)), "GGUF offset overflow");
    assert_eq!(refusal(checked_mul(u64::MAX, 2)), "GGUF shape overflow");
    assert_eq!(checked_mul(0, u64::MAX).unwrap(), 0, "zero multiplication");
}
