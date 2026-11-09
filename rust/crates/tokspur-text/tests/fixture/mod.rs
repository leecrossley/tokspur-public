//! The C++ protocol test's tokenizer fixture, shared by the protocol and CLI tests: `ByteBpe` over 256 byte singletons plus `<|im_end|>`, `<|endoftext|>` and `<|im_start|>` (IDs 256-258), no merges, and an identity `TextProcessor`. `ByteBpe` itself is ported separately, so this reproduces only that fixture's behaviour. `rust/tools/diff/chat_response.sh` runs the same cases against the real C++ `ByteBpe` fixture.

// Each test target uses a subset of this shared fixture.
#![allow(dead_code)]

use std::fmt;
use std::path::Path;

use tokspur_text::response_cli::ResponseTokenizer;

pub const MARKERS: [&[u8]; 3] = [b"<|im_end|>", b"<|endoftext|>", b"<|im_start|>"];

/// The fixture's `FormatError`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureError {
    /// `ByteBpe::decode_bytes` was given an ID outside the vocabulary.
    Decode,
    /// The CLI stub loader does not know the requested path.
    Load,
}

impl fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Decode => "token ID outside vocabulary",
            Self::Load => "stub tokenizer missing",
        })
    }
}

impl std::error::Error for FixtureError {}

/// `added: false` keeps the three markers as ordinary vocabulary entries, so no merge can produce them and they encode byte by byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fixture {
    pub added: bool,
}

impl Fixture {
    pub const VOCABULARY: u32 = 259;
    pub const STANDARD: Fixture = Fixture { added: true };

    /// Added tokens match first, left to right; every other byte is its own ID.
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut i = 0;
        while i < text.len() {
            match MARKERS
                .iter()
                .position(|marker| self.added && text[i..].starts_with(marker))
            {
                Some(index) => {
                    ids.push(256 + index as u32);
                    i += MARKERS[index].len();
                }
                None => {
                    ids.push(u32::from(text[i]));
                    i += 1;
                }
            }
        }
        ids
    }

    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, FixtureError> {
        let mut output = Vec::new();
        for &id in ids {
            match id {
                0..=255 => output.push(id as u8),
                256..=258 => output.extend_from_slice(MARKERS[(id - 256) as usize]),
                _ => return Err(FixtureError::Decode),
            }
        }
        Ok(output)
    }

    /// `ByteBpe::decode_text`: `replace_invalid_utf8` over the decoded bytes. `from_utf8_lossy` replaces the inputs used in these tests identically.
    pub fn decode_text(&self, ids: &[u32]) -> Result<Vec<u8>, FixtureError> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?)
            .into_owned()
            .into_bytes())
    }
}

impl ResponseTokenizer for Fixture {
    type Error = FixtureError;

    fn encode(&self, text: &[u8]) -> Result<Vec<u32>, FixtureError> {
        Ok(Fixture::encode(self, text))
    }

    fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, FixtureError> {
        Fixture::decode_bytes(self, ids)
    }

    fn vocabulary_size(&self) -> usize {
        Self::VOCABULARY as usize
    }
}

/// The CLI stand-in for `NativeText`'s constructor: `fixture.json` is the standard fixture, `plain.json` has no added markers, anything else fails.
pub fn load(path: &Path, _pin: &str) -> Result<Fixture, FixtureError> {
    match path.file_name().and_then(|name| name.to_str()) {
        Some("fixture.json") => Ok(Fixture::STANDARD),
        Some("plain.json") => Ok(Fixture { added: false }),
        _ => Err(FixtureError::Load),
    }
}
