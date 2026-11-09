//! Text layer: Unicode 9 NFC, byte-level BPE, chat rendering and the response parser/protocol, ported one to one from `native/src`.

#![forbid(unsafe_code)]

pub mod assets;
pub mod chat;
mod cxx;
pub mod response;
pub mod response_cli;
pub mod response_protocol;
pub mod sha256;
pub mod tokenizer;
pub mod unicode;
