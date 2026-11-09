//! `Debug` never prints weight bytes or tokenizer strings.

use tokspur_gguf::{Gguf, GgufError, MappedFile, ModelFiles};

use crate::support::{Bytes, Temporary, data, fixture, key, number, split, string, text};

const SECRET_TOKEN: &str = "secret-token-🦉";
const WEIGHT: u8 = 0xa7;

fn tokenizer_fixture(keys: Vec<crate::support::Key>) -> Bytes {
    let mut tokens = Bytes::new();
    number(&mut tokens, 8, 4);
    number(&mut tokens, 2, 8);
    string(&mut tokens, SECRET_TOKEN.as_bytes());
    string(&mut tokens, b"secret merge");
    let mut keys = keys;
    keys.push(key("tokenizer.ggml.tokens", 9, tokens));
    let mut bytes = fixture(&keys, &[data("first", &[4], 0, 0, 16)], 32);
    let length = bytes.len();
    bytes[length - 16..].fill(WEIGHT);
    bytes
}

fn assert_redacted(printed: &str) {
    assert!(!printed.contains("secret"), "{printed}");
    assert!(!printed.contains("🦉"), "{printed}");
    // Tensor bytes would print as a run of WEIGHT (167) values.
    assert!(!printed.contains("167, 167"), "{printed}");
}

#[test]
fn gguf_debug_prints_counts_only() {
    let bytes = tokenizer_fixture(vec![key("general.architecture", 8, text("toy"))]);
    let file = Gguf::from_bytes(&bytes).unwrap();
    let printed = format!("{file:?}");
    assert_eq!(
        printed,
        format!(
            "Gguf {{ mapped: false, tensors: 1, metadata: 2, data_offset: {}, file_size: {}, .. }}",
            file.data_offset(),
            bytes.len()
        )
    );
    assert_redacted(&printed);
    assert_redacted(&format!("{file:#?}"));
    assert_redacted(&format!("{:?}", file.metadata()));
    assert_redacted(&format!("{:?}", file.tensors()));
}

#[test]
fn model_files_debug_prints_counts_only() {
    let temporary = Temporary::new();
    let p0 = temporary.write("part0.gguf", &tokenizer_fixture(split(0, 2, 2)));
    let p1 = temporary.write(
        "part1.gguf",
        &fixture(&split(1, 2, 2), &[data("second", &[4], 0, 0, 16)], 32),
    );
    let model = ModelFiles::new(&[&p0, &p1]).unwrap();
    let printed = format!("{model:?}");
    assert_eq!(printed, "ModelFiles { shards: 2, tensors: 2, .. }");
    assert_redacted(&format!("{model:#?}"));
    assert_redacted(&format!("{:?}", model.shards()));
    assert_redacted(&format!("{:?}", model.tensors()));
    let mapped = MappedFile::new(&p0).unwrap();
    assert_redacted(&format!("{mapped:?}"));
}

#[test]
fn tensor_and_error_debug_show_no_payload() {
    let bytes = tokenizer_fixture(Vec::new());
    let file = Gguf::from_bytes(&bytes).unwrap();
    let tensor = &file.tensors()[0];
    assert_eq!(
        format!("{tensor:?}"),
        format!(
            "Tensor {{ name: \"first\", dimensions: [4], type: 0, offset: {}, bytes: 16, elements: 4 }}",
            tensor.offset
        )
    );
    let error: GgufError = match file.string_value("tokenizer.ggml.tokens") {
        Ok(_) => panic!("a string array is not a string"),
        Err(error) => error,
    };
    assert_redacted(&format!("{error:?}"));
}
