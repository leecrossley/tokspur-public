//! Accumulated-response adapter, matched with `native/src/response_protocol.cpp`.
//!
//! The caller supplies generated assistant IDs, the renderer’s initial thinking mode, a typed tool manifest and the worker’s termination fields. Only a matching final terminal ID and stop metadata allow completion and publication of tool calls. Callers own correlation, epochs and tool side effects. Invalid envelopes, bytes or frames return an error, and the CLI returns an empty-call error envelope before starting a fresh parser for the next request.

use std::fmt;

use tokspur_json::{Json, JsonError, Object};

use crate::response::{
    ParameterSpec, ParameterType, ResponseError, ResponseKind, ResponseLimits, ResponseParser,
    ResponseStatus, ToolSpec,
};

/// `response_request_bytes`: request-line and decoded-response bound.
pub const RESPONSE_REQUEST_BYTES: usize = 1024 * 1024;
/// `response_maximum_ids`.
pub const RESPONSE_MAXIMUM_IDS: usize = 16384;
/// `response_maximum_accumulated_ids`: HTTP accumulates multiple bounded native packets. The CLI default stays fixed.
pub const RESPONSE_MAXIMUM_ACCUMULATED_IDS: usize = 32768;
/// `response_output_bytes`: response-envelope `dump` bound.
pub const RESPONSE_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

/// `ResponseAdmission`: derived by the native CLI from the exact pinned tokenizer, never admitted directly from an untrusted JSON `stop_id`. These are tokenizer facts (the CLI publishes them in its ready line), not conversation token IDs, so `Debug` is derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseAdmission {
    pub vocabulary: u32,
    pub im_end: u32,
    pub end_of_text: u32,
    pub im_start: u32,
}

impl ResponseAdmission {
    pub fn is_stop(self, id: u32) -> bool {
        id == self.im_end || id == self.end_of_text
    }
}

/// `ResponseDecoder` (`std::function<std::string(std::span<const uint32_t>)>`): the exact represented bytes of generated IDs, including incomplete UTF-8 and special markers. The production decoder is `ByteBpe::decode_bytes`.
pub type ResponseDecoder<'a, E> = dyn Fn(&[u32]) -> Result<Vec<u8>, E> + 'a;

/// The C++ `FormatError` raised by [`parse_native_response`]. `Display` is the exact C++ `what()` text; `E` is the decoder's error.
#[derive(Clone, PartialEq, Eq)]
pub enum ResponseProtocolError<E> {
    /// The envelope, admission or decoded bytes were refused.
    Admit(&'static str),
    /// A typed JSON accessor found another kind or an out-of-range integer.
    Access(JsonError),
    /// `ResponseParser::new` refused the manifest.
    Configure(ResponseError),
    /// The decoder refused the IDs.
    Decode(E),
}

impl<E> fmt::Debug for ResponseProtocolError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admit(reason) => formatter.debug_tuple("Admit").field(reason).finish(),
            Self::Access(error) => formatter.debug_tuple("Access").field(error).finish(),
            Self::Configure(error) => formatter.debug_tuple("Configure").field(error).finish(),
            Self::Decode(_) => formatter.write_str("Decode([redacted])"),
        }
    }
}

impl<E: fmt::Display> fmt::Display for ResponseProtocolError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admit(reason) => formatter.write_str(reason),
            Self::Access(error) => error.fmt(formatter),
            Self::Configure(error) => error.fmt(formatter),
            Self::Decode(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ResponseProtocolError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admit(_) => None,
            Self::Access(error) => Some(error),
            Self::Configure(error) => Some(error),
            Self::Decode(error) => Some(error),
        }
    }
}

fn require<E>(ok: bool, message: &'static str) -> Result<(), ResponseProtocolError<E>> {
    if ok {
        Ok(())
    } else {
        Err(ResponseProtocolError::Admit(message))
    }
}

fn field<'a, E>(object: &'a Json, name: &str) -> Result<&'a Json, ResponseProtocolError<E>> {
    match object.find(name).map_err(ResponseProtocolError::Access)? {
        Some(value) => Ok(value),
        None => Err(ResponseProtocolError::Admit(
            "missing explicit response field",
        )),
    }
}

fn keys<E>(object: &Json, allowed: &[&[u8]]) -> Result<(), ResponseProtocolError<E>> {
    for (key, _) in object.object().map_err(ResponseProtocolError::Access)? {
        require(
            allowed.contains(&key.as_slice()),
            "unknown response protocol field",
        )?;
    }
    Ok(())
}

fn tools<E>(value: &Json) -> Result<Vec<ToolSpec>, ResponseProtocolError<E>> {
    let access = ResponseProtocolError::Access;
    let manifest = value.array().map_err(access)?;
    require(manifest.len() <= 64, "response tool manifest limit")?;
    let mut result = Vec::with_capacity(manifest.len());
    for entry in manifest {
        keys(entry, &[b"name", b"parameters"])?;
        let mut tool = ToolSpec::new(field(entry, "name")?.string().map_err(access)?, Vec::new());
        let parameters = field(entry, "parameters")?.array().map_err(access)?;
        require(parameters.len() <= 32, "response parameter manifest limit")?;
        tool.parameters.reserve(parameters.len());
        for parameter in parameters {
            keys(parameter, &[b"name", b"type", b"required"])?;
            let kind = match field(parameter, "type")?.string().map_err(access)? {
                b"string" => ParameterType::RawString,
                b"boolean" => ParameterType::Boolean,
                b"number" => ParameterType::Number,
                b"object" => ParameterType::Object,
                b"array" => ParameterType::Array,
                b"null" => ParameterType::NullValue,
                b"any" => ParameterType::JsonValue,
                _ => {
                    return Err(ResponseProtocolError::Admit(
                        "unsupported response parameter type",
                    ));
                }
            };
            // A braced initializer: name before required, left to right.
            let name = field(parameter, "name")?.string().map_err(access)?;
            let required = field(parameter, "required")?.boolean().map_err(access)?;
            tool.parameters
                .push(ParameterSpec::new(name, kind, required));
        }
        result.push(tool);
    }
    Ok(result)
}

fn result(
    status: &[u8],
    finish: &[u8],
    complete: bool,
    content: Vec<u8>,
    thinking: Vec<u8>,
    calls: Vec<Json>,
) -> Object {
    vec![
        (b"event".to_vec(), Json::from("response")),
        (b"status".to_vec(), Json::from(status)),
        (b"finish_reason".to_vec(), Json::from(finish)),
        (b"complete".to_vec(), Json::Bool(complete)),
        (b"content".to_vec(), Json::String(content)),
        (b"reasoning_content".to_vec(), Json::String(thinking)),
        (b"tool_calls".to_vec(), Json::Array(calls)),
        (b"tools_executed".to_vec(), Json::Bool(false)),
    ]
}

/// The empty-call error envelope; `message` is cut to its first 2048 bytes.
pub fn native_response_error(message: impl AsRef<[u8]>) -> Json {
    let message = message.as_ref();
    let mut fields = result(
        b"error",
        b"error",
        false,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    fields.push((
        b"message".to_vec(),
        Json::String(message[..message.len().min(2048)].to_vec()),
    ));
    Json::Object(fields)
}

/// Admits one accumulated response envelope and returns its result envelope.
///
/// `decode` is `None` only to reproduce the C++ empty-`std::function` refusal.
pub fn parse_native_response<E>(
    data: &Json,
    admission: ResponseAdmission,
    decode: Option<&ResponseDecoder<'_, E>>,
    maximum_ids: usize,
) -> Result<Json, ResponseProtocolError<E>> {
    let access = ResponseProtocolError::Access;
    require(
        admission.vocabulary != 0
            && admission.vocabulary <= 1_048_576
            && admission.im_end < admission.vocabulary
            && admission.end_of_text < admission.vocabulary
            && admission.im_start < admission.vocabulary
            && admission.im_end != admission.end_of_text
            && admission.im_end != admission.im_start
            && admission.end_of_text != admission.im_start,
        "invalid tokenizer-derived response admission",
    )?;
    keys(
        data,
        &[
            b"ids",
            b"tools",
            b"initial_thinking",
            b"finish_reason",
            b"stop_id",
            b"executor_parameters",
        ],
    )?;
    let executor_parameters = match data.find("executor_parameters").map_err(access)? {
        Some(value) => value.boolean().map_err(access)?,
        None => false,
    };
    require(
        maximum_ids != 0 && maximum_ids <= RESPONSE_MAXIMUM_ACCUMULATED_IDS,
        "invalid response ID admission bound",
    )?;
    let input = field(data, "ids")?.array().map_err(access)?;
    require(input.len() <= maximum_ids, "response ID count limit")?;
    let finish = field(data, "finish_reason")?.string().map_err(access)?;
    require(
        finish == b"stop" || finish == b"length" || finish == b"error",
        "unknown response finish reason",
    )?;
    let stopped = finish == b"stop";
    let stop = field(data, "stop_id")?;
    if stopped {
        require(
            !stop.is_null()
                && (stop.unsigned_integer().map_err(access)? == u64::from(admission.im_end)
                    || stop.unsigned_integer().map_err(access)?
                        == u64::from(admission.end_of_text)),
            "response stop ID is not an admitted terminal token",
        )?;
    } else {
        require(stop.is_null(), "non-stop response must have null stop_id")?;
    }
    let mut ids = Vec::with_capacity(input.len());
    for (i, item) in input.iter().enumerate() {
        let id = item.unsigned_integer().map_err(access)?;
        require(
            id < u64::from(admission.vocabulary),
            "response token outside vocabulary",
        )?;
        require(
            id != u64::from(admission.im_start),
            "unexpected response terminal or role marker",
        )?;
        require(
            !admission.is_stop(id as u32)
                || (stopped
                    && i + 1 == input.len()
                    && id == stop.unsigned_integer().map_err(access)?),
            "earlier or unacknowledged response terminal token",
        )?;
        // Lossless: `id` is below a 32-bit vocabulary size.
        ids.push(id as u32);
    }
    if stopped {
        let expected_stop = stop.unsigned_integer().map_err(access)?;
        require(
            ids.last().copied().map(u64::from) == Some(expected_stop),
            "stopped response lacks exact final stop token",
        )?;
    }
    // Clang evaluates the C++ constructor arguments left to right.
    let manifest = tools(field(data, "tools")?)?;
    let initial_thinking = field(data, "initial_thinking")?.boolean().map_err(access)?;
    let mut parser = ResponseParser::new_with_executor_parameters(
        manifest,
        initial_thinking,
        ResponseLimits::default(),
        executor_parameters,
    )
    .map_err(ResponseProtocolError::Configure)?
    .with_token_framing();
    if finish == b"error" {
        return Ok(native_response_error(
            "target reported an error; response is not executable",
        ));
    }
    let Some(decode) = decode else {
        return Err(ResponseProtocolError::Admit("missing raw response decoder"));
    };
    let mut bytes = decode(&ids).map_err(ResponseProtocolError::Decode)?;
    require(
        bytes.len() <= RESPONSE_REQUEST_BYTES,
        "decoded response byte limit",
    )?;
    if stopped {
        let marker = if stop.unsigned_integer().map_err(access)? == u64::from(admission.im_end) {
            b"<|im_end|>".as_slice()
        } else {
            b"<|endoftext|>"
        };
        require(
            bytes.ends_with(marker),
            "response decoder/terminal identity mismatch",
        )?;
        bytes.truncate(bytes.len() - marker.len());
    }
    require(
        parser.feed(&bytes) != ResponseStatus::Error,
        "invalid response bytes or frames",
    )?;
    if stopped {
        require(
            parser.finish() == ResponseStatus::Complete,
            "incomplete or invalid stopped response",
        )?;
    }
    let mut content = Vec::new();
    let mut thinking = Vec::new();
    let mut calls = Vec::new();
    for record in parser.records() {
        match record.kind {
            ResponseKind::Text => content.extend_from_slice(&record.text),
            ResponseKind::Thinking => thinking.extend_from_slice(&record.text),
            ResponseKind::Tool => {
                require(
                    stopped && parser.status() == ResponseStatus::Complete,
                    "tool record before terminal admission",
                )?;
                calls.push(Json::new_object([
                    ("ordinal", Json::integer(record.ordinal as u64)),
                    ("type", Json::from("function")),
                    (
                        "function",
                        Json::new_object([
                            ("name", Json::String(record.name.clone())),
                            ("arguments", record.arguments.clone()),
                        ]),
                    ),
                ]));
            }
        }
    }
    let status: &[u8] = if stopped { b"complete" } else { b"incomplete" };
    Ok(Json::Object(result(
        status, finish, stopped, content, thinking, calls,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::error::Error;
    use tokspur_json::UnicodeError;

    #[test]
    fn errors_display_the_exact_cpp_messages_and_retain_sources() {
        let cases: [(ResponseProtocolError<UnicodeError>, &str, bool); 4] = [
            (
                ResponseProtocolError::Admit("response ID count limit"),
                "response ID count limit",
                false,
            ),
            (
                ResponseProtocolError::Access(JsonError::Access("expected JSON array")),
                "expected JSON array",
                true,
            ),
            (
                ResponseProtocolError::Configure(ResponseError::Name("unsafe response name")),
                "unsafe response name",
                true,
            ),
            (
                ResponseProtocolError::Decode(UnicodeError::Decode("invalid UTF-8 input")),
                "invalid UTF-8 input",
                true,
            ),
        ];
        for (error, message, has_source) in cases {
            assert_eq!(error.to_string(), message);
            assert_eq!(error.source().is_some(), has_source);
            let debug = format!("{error:?}");
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn error_message_is_cut_at_2048_bytes_like_substr() {
        // substr cuts bytes, not scalars: a split sequence then fails dump, exactly as the C++ envelope would.
        let message = [vec![b'a'; 2047], "é".as_bytes().to_vec()].concat();
        let envelope = native_response_error(&message);
        let Some(Json::String(kept)) = envelope.find("message").unwrap() else {
            panic!("message field")
        };
        assert_eq!(kept.len(), 2048);
        assert_eq!(kept.last(), Some(&0xc3));
        assert_eq!(
            envelope.dump().unwrap_err().to_string(),
            "invalid UTF-8 input"
        );
        assert_eq!(
            native_response_error("").find("message").unwrap(),
            Some(&Json::from(""))
        );
    }

    #[test]
    fn upstream_errors_skip_decoding_but_not_manifest_validation() {
        let admission = ResponseAdmission {
            vocabulary: 259,
            im_end: 256,
            end_of_text: 257,
            im_start: 258,
        };
        let request = |tools: &str| {
            tokspur_json::parse_json(
                format!(
                    r#"{{"ids":[1],"tools":{tools},"initial_thinking":false,"finish_reason":"error","stop_id":null}}"#
                )
                .as_bytes(),
            )
            .unwrap()
        };
        let never =
            |_: &[u32]| -> Result<Vec<u8>, Infallible> { panic!("decoded an upstream error") };
        let output = parse_native_response(
            &request("[]"),
            admission,
            Some(&never),
            RESPONSE_MAXIMUM_IDS,
        )
        .unwrap();
        assert_eq!(output.find("status").unwrap(), Some(&Json::from("error")));
        let refused = parse_native_response(
            &request(r#"[{"name":"a b","parameters":[]}]"#),
            admission,
            Some(&never),
            RESPONSE_MAXIMUM_IDS,
        );
        assert_eq!(
            refused.unwrap_err().to_string(),
            "unsupported response function name"
        );
    }
}
