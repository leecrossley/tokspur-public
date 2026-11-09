//! Derives the pinned tokenizer and chat template from verified GGUF metadata.

use std::path::{Path, PathBuf};

use tokspur_gguf::Gguf;
use tokspur_json::Json;
use tokspur_text::sha256::{hex, sha256};

use crate::error::CliError;
use crate::state::{private_dir, write_private_bytes};

pub(crate) const SHARD1_SHA256: &str =
    "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082";
pub(crate) const DERIVED_DIR: &str = "derived-text";
const TOKENIZER_PIN: &str = tokspur_text::assets::TOKENIZER_SHA256;
const TEMPLATE_PIN: &str = tokspur_text::assets::TEMPLATE_SHA256;
const SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn byte_level(on: bool) -> Json {
    Json::new_object([
        ("type", Json::from("ByteLevel")),
        ("add_prefix_space", Json::from(on)),
        ("trim_offsets", Json::from(on)),
        ("use_regex", Json::from(on)),
    ])
}

fn refuse(error: impl std::fmt::Display) -> CliError {
    CliError::Validation(format!("cannot derive text assets: {error}"))
}

fn tokenizer_json(gguf: &Gguf<'_>) -> Result<Vec<u8>, CliError> {
    let tokens = gguf.string_array("tokenizer.ggml.tokens").map_err(refuse)?;
    let kinds = gguf
        .unsigned_array("tokenizer.ggml.token_type")
        .map_err(refuse)?;
    if tokens.len() != kinds.len() || tokens.is_empty() || tokens.len() > 248320 {
        return Err(refuse("token vocabulary and type counts differ"));
    }
    let merges = gguf.string_array("tokenizer.ggml.merges").map_err(refuse)?;
    let added: Vec<Json> = tokens
        .iter()
        .zip(&kinds)
        .enumerate()
        .filter(|(_, (_, kind))| matches!(**kind, 3 | 4))
        .map(|(id, (token, kind))| {
            Json::new_object([
                ("id", Json::integer(id as u64)),
                ("content", Json::from(*token)),
                ("single_word", Json::from(false)),
                ("lstrip", Json::from(false)),
                ("rstrip", Json::from(false)),
                ("normalized", Json::from(false)),
                ("special", Json::from(*kind == 3)),
            ])
        })
        .collect();
    let vocabulary: Vec<(Vec<u8>, Json)> = tokens
        .iter()
        .enumerate()
        .map(|(id, token)| (token.to_vec(), Json::integer(id as u64)))
        .collect();
    let mut rows = Vec::with_capacity(merges.len());
    for merge in &merges {
        let space = merge
            .iter()
            .position(|c| *c == b' ')
            .ok_or_else(|| refuse("merge without a separator"))?;
        rows.push(Json::Array(vec![
            Json::from(&merge[..space]),
            Json::from(&merge[space + 1..]),
        ]));
    }
    Json::new_object([
        ("version", Json::from("1.0")),
        ("truncation", Json::Null),
        ("padding", Json::Null),
        ("added_tokens", Json::Array(added)),
        (
            "normalizer",
            Json::new_object([("type", Json::from("NFC"))]),
        ),
        (
            "pre_tokenizer",
            Json::new_object([
                ("type", Json::from("Sequence")),
                (
                    "pretokenizers",
                    Json::Array(vec![
                        Json::new_object([
                            ("type", Json::from("Split")),
                            ("pattern", Json::new_object([("Regex", Json::from(SPLIT))])),
                            ("behavior", Json::from("Isolated")),
                            ("invert", Json::from(false)),
                        ]),
                        byte_level(false),
                    ]),
                ),
            ]),
        ),
        ("post_processor", byte_level(false)),
        ("decoder", byte_level(true)),
        (
            "model",
            Json::new_object([
                ("type", Json::from("BPE")),
                ("dropout", Json::Null),
                ("unk_token", Json::Null),
                ("continuing_subword_prefix", Json::Null),
                ("end_of_word_suffix", Json::Null),
                ("fuse_unk", Json::from(false)),
                ("byte_fallback", Json::from(false)),
                ("ignore_merges", Json::from(false)),
                ("vocab", Json::Object(vocabulary)),
                ("merges", Json::Array(rows)),
            ]),
        ),
    ])
    .dump()
    .map_err(refuse)
}

/// Writes both assets under `state/gguf/derived-text`, refusing anything that does not match its pin.
pub(crate) fn derive(tokspur_base: &Path, shard: &Path) -> Result<PathBuf, CliError> {
    let gguf = Gguf::from_path(shard).map_err(refuse)?;
    let tokenizer = tokenizer_json(&gguf)?;
    let template = gguf
        .string_value("tokenizer.chat_template")
        .map_err(refuse)?;
    if hex(&sha256(&tokenizer)) != TOKENIZER_PIN || hex(&sha256(template)) != TEMPLATE_PIN {
        return Err(refuse("derived bytes differ from the pins"));
    }
    private_dir(tokspur_base, Path::new("state/gguf"))?;
    let directory = private_dir(tokspur_base, Path::new("state/gguf/derived-text"))?;
    for (name, bytes) in [
        ("tokenizer.json", tokenizer.as_slice()),
        ("chat_template.jinja", template),
    ] {
        write_private_bytes(
            tokspur_base,
            &Path::new("state/gguf/derived-text").join(name),
            bytes,
        )?;
    }
    Ok(directory)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn string(out: &mut Vec<u8>, value: &[u8]) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value);
    }

    fn metadata(kinds: &[u32], merge: &[u8]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&3u64.to_le_bytes());
        for (key, values) in [
            (
                b"tokenizer.ggml.tokens".as_slice(),
                vec![b"a".as_slice(), b"b", b"<|im_end|>"],
            ),
            (b"tokenizer.ggml.merges", vec![merge]),
        ] {
            string(&mut out, key);
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                string(&mut out, value);
            }
        }
        string(&mut out, b"tokenizer.ggml.token_type");
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&(kinds.len() as u64).to_le_bytes());
        for kind in kinds {
            out.extend_from_slice(&kind.to_le_bytes());
        }
        out
    }

    #[test]
    fn derives_bpe_added_tokens_and_refuses_invalid_metadata() {
        let bytes = metadata(&[1, 1, 3], b"a b");
        let model = Gguf::from_bytes(&bytes).unwrap();
        let encoded = tokenizer_json(&model).unwrap();
        let value = tokspur_json::parse_json(&encoded).unwrap();
        assert_eq!(
            value
                .find("added_tokens")
                .unwrap()
                .unwrap()
                .array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            value
                .find("model")
                .unwrap()
                .unwrap()
                .find("vocab")
                .unwrap()
                .unwrap()
                .find("<|im_end|>")
                .unwrap()
                .unwrap()
                .unsigned_integer()
                .unwrap(),
            2
        );
        for bytes in [metadata(&[1, 3], b"a b"), metadata(&[1, 1, 3], b"ab")] {
            assert!(tokenizer_json(&Gguf::from_bytes(&bytes).unwrap()).is_err());
        }
    }

    #[test]
    fn verified_public_metadata_matches_both_pins() {
        let Some(path) = std::env::var_os("TOKSPUR_TEST_TEXT_METADATA") else {
            return;
        };
        let model = Gguf::from_path(path).unwrap();
        assert_eq!(
            hex(&sha256(&tokenizer_json(&model).unwrap())),
            TOKENIZER_PIN
        );
        assert_eq!(
            hex(&sha256(
                model.string_value("tokenizer.chat_template").unwrap()
            )),
            TEMPLATE_PIN
        );
    }
}
