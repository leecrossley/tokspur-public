//! Owned HTTP/SSE serving layer, ported one to one from `native/src/http.cpp`, `serve_protocol.cpp`, `serve.cpp` and the server half of `text.mm`.

#![forbid(unsafe_code)]

pub mod admission;
pub mod http;
pub mod native_text;
pub mod serve;
pub mod session;
