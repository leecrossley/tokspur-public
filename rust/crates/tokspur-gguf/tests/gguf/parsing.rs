//! `parsing()` in `native/tests/test_gguf.cpp` (89 checks).

use tokspur_gguf::{Gguf, ModelFiles};

use crate::support::{
    Data, Temporary, default_fixture, fixture, integer, key, refusal, replace, split,
};

#[test]
fn metadata_only_shards_accept_optional_padding() {
    let temporary = Temporary::new();
    for alignment in [32, 64] {
        let mut keys = split(0, 2, 1);
        keys.push(key("general.alignment", 4, integer(alignment, 4)));
        let padded = fixture(&keys, &[], alignment);
        let metadata_end = 24
            + keys
                .iter()
                .map(|k| 8 + k.name.len() + 4 + k.value.len())
                .sum::<usize>();
        assert!(metadata_end < padded.len());
        for length in metadata_end..=padded.len() {
            let file = Gguf::from_bytes(&padded[..length]).unwrap();
            assert!(file.tensors().is_empty());
            assert_eq!(file.data_offset(), length as u64);
            assert_eq!(file.unsigned_value("split.tensors.count").unwrap(), 1);
        }
        let metadata_path =
            temporary.write(&format!("metadata-{alignment}"), &padded[..metadata_end]);
        let file = Gguf::from_path(&metadata_path).unwrap();
        assert!(file.tensors().is_empty());
        assert_eq!(file.data_offset(), metadata_end as u64);
        let payload_path = temporary.write(
            &format!("tensor-{alignment}"),
            &fixture(&split(1, 2, 1), &[Data::default()], 32),
        );
        let model = ModelFiles::new(&[&payload_path, &metadata_path]).unwrap();
        assert_eq!(model.tensors().len(), 1);
        let mut payload = Vec::new();
        model.tensor_bytes("weight", &mut payload).unwrap();
        assert_eq!(payload.len(), 16);
    }
    let mut bare = fixture(&[], &[], 32);
    bare.truncate(24);
    let file = Gguf::from_bytes(&bare).unwrap();
    assert!(file.tensors().is_empty());
    assert_eq!(file.data_offset(), 24);
}

#[test]
fn metadata_only_shards_still_validate_metadata_and_alignment() {
    let temporary = Temporary::new();
    let keys = split(0, 2, 1);
    let padded = fixture(&keys, &[], 32);
    let metadata_end = 24
        + keys
            .iter()
            .map(|k| 8 + k.name.len() + 4 + k.value.len())
            .sum::<usize>();
    for length in 0..metadata_end {
        assert!(
            Gguf::from_bytes(&padded[..length]).is_err(),
            "length {length}"
        );
    }
    let truncated = temporary.write("truncated-metadata", &padded[..metadata_end - 1]);
    assert!(Gguf::from_path(truncated).is_err());
    for alignment in [0, 3, 65537] {
        let bytes = fixture(
            &[key("general.alignment", 4, integer(alignment, 4))],
            &[],
            32,
        );
        assert_eq!(
            refusal(Gguf::from_bytes(&bytes)),
            "unsupported GGUF alignment"
        );
    }
}

#[test]
fn tensor_shards_still_require_padding() {
    let temporary = Temporary::new();
    for alignment in [32, 64] {
        let mut keys = split(0, 2, 1);
        keys.push(key("general.alignment", 4, integer(alignment, 4)));
        let padded = fixture(&keys, &[Data::default()], alignment);
        let file = Gguf::from_bytes(&padded).unwrap();
        let short_header = &padded[..file.data_offset() as usize - 1];
        assert_eq!(
            refusal(Gguf::from_bytes(short_header)),
            "truncated GGUF header padding"
        );
        let path = temporary.write(&format!("short-{alignment}"), short_header);
        assert_eq!(
            refusal(Gguf::from_path(path)),
            "truncated GGUF header padding"
        );
    }
}

#[test]
fn valid_fixture_has_one_four_element_tensor_of_sixteen_bytes() {
    let bytes = default_fixture();
    let valid = Gguf::from_bytes(&bytes).unwrap();
    assert_eq!(valid.tensors().len(), 1, "missing tensor");
    assert_eq!(valid.tensors()[0].elements, 4, "element count");
    let mut span = Vec::new();
    valid.tensor_bytes(&valid.tensors()[0], &mut span).unwrap();
    assert_eq!(span.len(), 16, "tensor span");
}

#[test]
fn copied_tensor_does_not_belong_to_the_file() {
    let bytes = default_fixture();
    let valid = Gguf::from_bytes(&bytes).unwrap();
    let copied_tensor = valid.tensors()[0].clone();
    let mut span = Vec::new();
    assert_eq!(
        refusal(valid.tensor_bytes(&copied_tensor, &mut span)),
        "tensor does not belong to this GGUF"
    );
}

#[test]
fn every_truncation_is_refused() {
    // Header to 24, tensor info to 62, padding to 64, payload to 80.
    let bytes = default_fixture();
    assert_eq!(bytes.len(), 80);
    for length in 0..bytes.len() {
        let expected = match length {
            0..24 => "truncated or oversized GGUF header",
            24..57 => "GGUF counts exceed available file bytes",
            57..62 => "truncated or oversized GGUF header",
            62..64 => "truncated GGUF header padding",
            _ => "truncated GGUF tensor data",
        };
        assert_eq!(
            refusal(Gguf::from_bytes(&bytes[..length])),
            expected,
            "length {length}"
        );
    }
}

#[test]
fn wrong_magic_is_refused() {
    let mut corrupt = default_fixture();
    corrupt[0] = 0;
    assert_eq!(
        refusal(Gguf::from_bytes(&corrupt)),
        "expected little-endian GGUF v3"
    );
}

#[test]
fn other_versions_and_byte_orders_are_refused() {
    let mut corrupt = default_fixture();
    replace(&mut corrupt, 4, 2, 4);
    assert_eq!(
        refusal(Gguf::from_bytes(&corrupt)),
        "expected little-endian GGUF v3"
    );
    replace(&mut corrupt, 4, 0x0300_0000, 4);
    assert_eq!(
        refusal(Gguf::from_bytes(&corrupt)),
        "expected little-endian GGUF v3"
    );
}

#[test]
fn record_counts_over_the_limit_are_refused() {
    let mut corrupt = default_fixture();
    replace(&mut corrupt, 8, 100_001, 8);
    assert_eq!(
        refusal(Gguf::from_bytes(&corrupt)),
        "GGUF record limit exceeded"
    );
    replace(&mut corrupt, 8, 1, 8);
    replace(&mut corrupt, 16, u64::MAX, 8);
    assert_eq!(
        refusal(Gguf::from_bytes(&corrupt)),
        "GGUF record limit exceeded"
    );
}
