//! The Rust stand-in for the C++ fixture tokenizer: `ByteBpe` over 256 byte singletons plus `<|im_end|>`, `<|endoftext|>` and `<|im_start|>` (IDs 256-258), no merges, identity `TextProcessor`. The C++ side uses the real `ByteBpe` (`native/src/tokenizer.cpp`), so every protocol and CLI case also checks this stand-in against it.

use std::fmt;
use std::path::Path;

use tokspur_text::response_cli::ResponseTokenizer;

pub const MARKERS: [&[u8]; 3] = [b"<|im_end|>", b"<|endoftext|>", b"<|im_start|>"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureError {
    Decode,
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

#[derive(Debug, Clone, Copy)]
pub struct Fixture {
    added: bool,
}

pub const FIXTURE: Fixture = Fixture { added: true };

impl Fixture {
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut i = 0;
        while i < text.len() {
            match MARKERS
                .iter()
                .position(|marker| self.added && text[i..].starts_with(marker))
            {
                Some(index) => {
                    ids.push(256 + u32::try_from(index).unwrap_or(0));
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
                0..=255 => output.push(u8::try_from(id).unwrap_or(0)),
                256..=258 => {
                    output.extend_from_slice(MARKERS[usize::try_from(id - 256).unwrap_or(0)])
                }
                _ => return Err(FixtureError::Decode),
            }
        }
        Ok(output)
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
        259
    }
}

/// `cpp/native_text_stub.cpp`'s `NativeText` constructor.
pub fn load(path: &Path, _pin: &str) -> Result<Fixture, FixtureError> {
    match path.file_name().and_then(|name| name.to_str()) {
        Some("fixture.json") => Ok(FIXTURE),
        Some("plain.json") => Ok(Fixture { added: false }),
        _ => Err(FixtureError::Load),
    }
}
