//! `NativeText`'s hand-written `Debug` prints counts only. The fixture is the smallest tokenizer the loader accepts: the 256 byte-level symbols and one special marker, written under Cargo's integration-test temporary directory.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fmt::Write as _;
use std::path::PathBuf;

use tokspur_server::native_text::NativeText;
use tokspur_text::sha256::{hex, sha256};

/// The one split pattern `NativeText` accepts (`SUPPORTED_PATTERN`).
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const MARKER: &str = "<|private|>";

/// The byte-level (GPT-2) spelling of one byte.
fn byte_symbol(byte: u8) -> char {
    let printable = |b: u8| (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
    if printable(byte) {
        char::from(byte)
    } else {
        let offset = (0..byte).filter(|&b| !printable(b)).count();
        char::from_u32(256 + u32::try_from(offset).unwrap()).unwrap()
    }
}

fn json_string(text: &str) -> String {
    let mut quoted = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            _ => quoted.push(c),
        }
    }
    quoted.push('"');
    quoted
}

fn tokenizer_json() -> String {
    let byte_level = |prefix: bool| {
        format!(
            r#"{{"type":"ByteLevel","add_prefix_space":{prefix},"trim_offsets":{prefix},"use_regex":{prefix}}}"#
        )
    };
    let mut vocabulary = String::new();
    for byte in 0..=u8::MAX {
        write!(
            vocabulary,
            "{}:{byte},",
            json_string(&byte_symbol(byte).to_string())
        )
        .unwrap();
    }
    write!(vocabulary, "{}:256", json_string(MARKER)).unwrap();
    format!(
        concat!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"#,
            r#""added_tokens":[{{"id":256,"content":{marker},"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}],"#,
            r#""normalizer":{{"type":"NFC"}},"#,
            r#""pre_tokenizer":{{"type":"Sequence","pretokenizers":[{{"type":"Split","pattern":{{"Regex":{pattern}}},"behavior":"Isolated","invert":false}},{split_level}]}},"#,
            r#""post_processor":{split_level},"decoder":{decoder},"#,
            r#""model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{{vocabulary}}},"merges":[]}}}}"#
        ),
        marker = json_string(MARKER),
        pattern = json_string(PATTERN),
        split_level = byte_level(false),
        decoder = byte_level(true),
        vocabulary = vocabulary,
    )
}

#[test]
fn native_text_debug_shows_counts_without_vocabulary_or_markers() {
    // Arrange
    let text = tokenizer_json();
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("native-text-debug-{}.json", std::process::id()));
    std::fs::write(&path, &text).unwrap();
    let pin = hex(&sha256(text.as_bytes()));

    // Act
    let loaded = NativeText::new(&path, &pin);
    std::fs::remove_file(&path).unwrap();
    let rendered = format!("{:?}", loaded.unwrap());

    // Assert
    assert_eq!(
        rendered,
        "NativeText { vocabulary_size: 257, merge_count: 0, .. }"
    );
    assert!(!rendered.contains("private"));
}
