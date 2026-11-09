//! Cases, their wire encoding for `cpp/text_driver.cpp`, and the Rust side's outcome bytes, written exactly as the C++ driver writes them.

use std::cell::Cell;

use tokspur_json::{Json, parse_json};
use tokspur_text::chat::{
    ChatRenderer, ChatStrings, chat_trim, extract_flash_next_literals, template_json,
};
use tokspur_text::response::{
    ParameterType, ResponseKind, ResponseLimits, ResponseParser, ResponseStatus, ToolSpec,
};
use tokspur_text::response_protocol::{
    RESPONSE_MAXIMUM_IDS, RESPONSE_REQUEST_BYTES, ResponseAdmission, ResponseDecoder,
    parse_native_response,
};

use crate::fixture::{FIXTURE, FixtureError};

/// Little-endian integers and u32-length-prefixed byte strings.
#[derive(Default)]
pub struct Wire {
    pub bytes: Vec<u8>,
}

impl Wire {
    pub fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub fn len(&mut self, value: usize) {
        self.u32(u32::try_from(value).unwrap_or(u32::MAX));
    }

    pub fn bytes(&mut self, value: impl AsRef<[u8]>) {
        let value = value.as_ref();
        self.len(value.len());
        self.bytes.extend_from_slice(value);
    }
}

pub enum Op {
    Feed(Vec<u8>),
    Finish,
}

/// The protocol decoder the driver installs (`ResponseDecoder`).
#[derive(Clone, Copy)]
pub enum Decoder {
    /// The fixture `ByteBpe::decode_bytes`.
    Fixture = 0,
    /// An empty `std::function` (`None`).
    Missing = 1,
    /// Always `wrong terminal`.
    WrongTerminal = 2,
    /// One byte over `response_request_bytes`.
    Oversized = 3,
}

pub enum Case {
    TemplateJson {
        json: Vec<u8>,
    },
    Render {
        strings: [Vec<u8>; 4],
        request: Vec<u8>,
    },
    Extract {
        source: Vec<u8>,
    },
    Trim {
        text: Vec<u8>,
    },
    Parser {
        tools: Vec<ToolSpec>,
        thinking: bool,
        limits: ResponseLimits,
        ops: Vec<Op>,
    },
    Protocol {
        admission: ResponseAdmission,
        envelope: Vec<u8>,
        decoder: Decoder,
    },
    /// A constructed value (any string bytes, any depth), pre-encoded.
    TemplateTree {
        tree: Vec<u8>,
    },
    RenderTree {
        strings: [Vec<u8>; 4],
        tree: Vec<u8>,
    },
}

/// Decodes the driver's constructed-value encoding (`tree` in the driver). The encoder in `generate.rs` writes only well-formed trees.
pub fn decode_tree(input: &mut &[u8]) -> Json {
    fn take<'a>(input: &mut &'a [u8], count: usize) -> &'a [u8] {
        let (head, tail) = input.split_at(count.min(input.len()));
        *input = tail;
        head
    }
    fn count(input: &mut &[u8]) -> usize {
        let head = take(input, 4);
        let mut bytes = [0u8; 4];
        bytes[..head.len()].copy_from_slice(head);
        usize::try_from(u32::from_le_bytes(bytes)).unwrap_or(0)
    }
    fn bytes(input: &mut &[u8]) -> Vec<u8> {
        let size = count(input);
        take(input, size).to_vec()
    }
    let tag = take(input, 1).first().copied().unwrap_or(0);
    match tag {
        1 => Json::Bool(take(input, 1).first().is_some_and(|&flag| flag != 0)),
        2 => parse_json(&bytes(input)).unwrap_or(Json::Null),
        3 => Json::String(bytes(input)),
        4 => {
            let items = count(input);
            Json::Array((0..items).map(|_| decode_tree(input)).collect())
        }
        5 => {
            let fields = count(input);
            Json::Object(
                (0..fields)
                    .map(|_| {
                        let key = bytes(input);
                        (key, decode_tree(input))
                    })
                    .collect(),
            )
        }
        _ => Json::Null,
    }
}

pub struct Labeled {
    pub label: String,
    pub case: Case,
}

fn parameter_type(value: ParameterType) -> u8 {
    match value {
        ParameterType::RawString => 0,
        ParameterType::Boolean => 1,
        ParameterType::Number => 2,
        ParameterType::Object => 3,
        ParameterType::Array => 4,
        ParameterType::NullValue => 5,
    }
}

fn status(value: ResponseStatus) -> u8 {
    match value {
        ResponseStatus::Incomplete => 0,
        ResponseStatus::Complete => 1,
        ResponseStatus::Error => 2,
    }
}

fn kind(value: ResponseKind) -> u8 {
    match value {
        ResponseKind::Text => 0,
        ResponseKind::Thinking => 1,
        ResponseKind::Tool => 2,
    }
}

fn usize_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn dump_or_error(out: &mut Wire, value: &Json) {
    match value.dump() {
        Ok(text) => {
            out.u8(0);
            out.bytes(text);
        }
        Err(error) => {
            out.u8(1);
            out.bytes(error.to_string());
        }
    }
}

impl Case {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::TemplateJson { .. } => "template_json",
            Self::Render { .. } => "render",
            Self::Extract { .. } => "extract",
            Self::Trim { .. } => "chat_trim",
            Self::Parser { .. } => "parser",
            Self::Protocol { .. } => "protocol",
            Self::TemplateTree { .. } => "template_json tree",
            Self::RenderTree { .. } => "render tree",
        }
    }

    /// The case as the C++ driver reads it.
    pub fn encode(&self, out: &mut Wire) {
        match self {
            Self::TemplateJson { json } => {
                out.u8(1);
                out.bytes(json);
            }
            Self::Render { strings, request } => {
                out.u8(2);
                for value in strings {
                    out.bytes(value);
                }
                out.bytes(request);
            }
            Self::Extract { source } => {
                out.u8(3);
                out.bytes(source);
            }
            Self::Trim { text } => {
                out.u8(4);
                out.bytes(text);
            }
            Self::Parser {
                tools,
                thinking,
                limits,
                ops,
            } => {
                out.u8(5);
                out.len(tools.len());
                for tool in tools {
                    out.bytes(&tool.name);
                    out.len(tool.parameters.len());
                    for parameter in &tool.parameters {
                        out.bytes(&parameter.name);
                        out.u8(parameter_type(parameter.r#type));
                        out.u8(u8::from(parameter.required));
                    }
                }
                out.u8(u8::from(*thinking));
                for limit in [
                    limits.bytes,
                    limits.call_bytes,
                    limits.records,
                    limits.calls,
                    limits.json_nodes,
                    limits.json_depth,
                ] {
                    out.u64(usize_u64(limit));
                }
                out.len(ops.len());
                for op in ops {
                    match op {
                        Op::Feed(bytes) => {
                            out.u8(0);
                            out.bytes(bytes);
                        }
                        Op::Finish => out.u8(1),
                    }
                }
            }
            Self::Protocol {
                admission,
                envelope,
                decoder,
            } => {
                out.u8(6);
                for value in [
                    admission.vocabulary,
                    admission.im_end,
                    admission.end_of_text,
                    admission.im_start,
                ] {
                    out.u32(value);
                }
                out.bytes(envelope);
                out.u8(*decoder as u8);
            }
            Self::TemplateTree { tree } => {
                out.u8(7);
                out.bytes.extend_from_slice(tree);
            }
            Self::RenderTree { strings, tree } => {
                out.u8(8);
                for value in strings {
                    out.bytes(value);
                }
                out.bytes.extend_from_slice(tree);
            }
        }
    }

    /// The Rust port's outcome, in the driver's outcome encoding.
    pub fn evaluate(&self) -> Vec<u8> {
        let mut out = Wire::default();
        match self {
            Self::TemplateJson { json } => match parse_json(json) {
                Err(error) => {
                    out.u8(b'P');
                    out.bytes(error.to_string());
                }
                Ok(value) => match template_json(&value) {
                    Ok(rendered) => {
                        out.u8(b'O');
                        out.bytes(rendered);
                    }
                    Err(error) => {
                        out.u8(b'E');
                        out.bytes(error.to_string());
                    }
                },
            },
            Self::Render { strings, request } => render(&mut out, strings, request),
            Self::Extract { source } => match extract_flash_next_literals(source) {
                Ok(strings) => {
                    out.u8(b'O');
                    out.bytes(&strings.reasoning_xhigh);
                    out.bytes(&strings.reasoning_low);
                    out.bytes(&strings.tools_header);
                    out.bytes(&strings.tool_instructions);
                }
                Err(error) => {
                    out.u8(b'E');
                    out.bytes(error.to_string());
                }
            },
            Self::Trim { text } => match chat_trim(text) {
                Ok(trimmed) => {
                    out.u8(b'O');
                    out.bytes(trimmed);
                }
                Err(error) => {
                    out.u8(b'E');
                    out.bytes(error.to_string());
                }
            },
            Self::Parser {
                tools,
                thinking,
                limits,
                ops,
            } => match ResponseParser::new(tools.clone(), *thinking, *limits) {
                Err(error) => {
                    out.u8(b'C');
                    out.bytes(error.to_string());
                }
                Ok(mut parser) => {
                    out.u8(b'O');
                    for op in ops {
                        let returned = match op {
                            Op::Feed(bytes) => parser.feed(bytes),
                            Op::Finish => parser.finish(),
                        };
                        snapshot(&mut out, returned, &parser);
                    }
                }
            },
            Self::Protocol {
                admission,
                envelope,
                decoder,
            } => protocol(&mut out, *admission, envelope, *decoder),
            Self::TemplateTree { tree } => {
                match template_json(&decode_tree(&mut tree.as_slice())) {
                    Ok(rendered) => {
                        out.u8(b'O');
                        out.bytes(rendered);
                    }
                    Err(error) => {
                        out.u8(b'E');
                        out.bytes(error.to_string());
                    }
                }
            }
            Self::RenderTree { strings, tree } => {
                let [
                    reasoning_xhigh,
                    reasoning_low,
                    tools_header,
                    tool_instructions,
                ] = strings.clone();
                let strings = ChatStrings {
                    reasoning_xhigh,
                    reasoning_low,
                    tools_header,
                    tool_instructions,
                };
                match ChatRenderer::new(strings)
                    .map(|renderer| renderer.render(&decode_tree(&mut tree.as_slice())))
                {
                    Ok(Ok(rendered)) => {
                        out.u8(b'O');
                        out.bytes(&rendered.prompt);
                        dump_or_error(&mut out, &rendered.normalized_messages);
                        out.u64(usize_u64(rendered.parsed_tool_arguments));
                        out.u64(usize_u64(rendered.hoisted_system_messages));
                    }
                    Ok(Err(error)) | Err(error) => {
                        out.u8(b'E');
                        out.bytes(error.to_string());
                    }
                }
            }
        }
        out.bytes
    }
}

fn render(out: &mut Wire, strings: &[Vec<u8>; 4], request: &[u8]) {
    let request = match parse_json(request) {
        Ok(request) => request,
        Err(error) => {
            out.u8(b'P');
            out.bytes(error.to_string());
            return;
        }
    };
    let [
        reasoning_xhigh,
        reasoning_low,
        tools_header,
        tool_instructions,
    ] = strings.clone();
    let renderer = match ChatRenderer::new(ChatStrings {
        reasoning_xhigh,
        reasoning_low,
        tools_header,
        tool_instructions,
    }) {
        Ok(renderer) => renderer,
        Err(error) => {
            out.u8(b'C');
            out.bytes(error.to_string());
            return;
        }
    };
    match renderer.render(&request) {
        Ok(rendered) => {
            out.u8(b'O');
            out.bytes(&rendered.prompt);
            dump_or_error(out, &rendered.normalized_messages);
            out.u64(usize_u64(rendered.parsed_tool_arguments));
            out.u64(usize_u64(rendered.hoisted_system_messages));
        }
        Err(error) => {
            out.u8(b'E');
            out.bytes(error.to_string());
        }
    }
}

fn snapshot(out: &mut Wire, returned: ResponseStatus, parser: &ResponseParser) {
    out.u8(status(returned));
    out.u8(status(parser.status()));
    out.bytes(parser.error());
    out.u64(usize_u64(parser.bytes_received()));
    out.len(parser.records().len());
    for record in parser.records() {
        out.u8(kind(record.kind));
        out.bytes(&record.text);
        out.u64(usize_u64(record.ordinal));
        out.bytes(&record.name);
        dump_or_error(out, &record.arguments);
    }
}

fn protocol(out: &mut Wire, admission: ResponseAdmission, envelope: &[u8], decoder: Decoder) {
    let envelope = match parse_json(envelope) {
        Ok(envelope) => envelope,
        Err(error) => {
            out.u8(b'P');
            out.bytes(error.to_string());
            return;
        }
    };
    let calls = Cell::new(0u64);
    let fixture = |ids: &[u32]| {
        calls.set(calls.get() + 1);
        FIXTURE.decode_bytes(ids)
    };
    let wrong = |_: &[u32]| {
        calls.set(calls.get() + 1);
        Ok::<_, FixtureError>(b"wrong terminal".to_vec())
    };
    let oversized = |_: &[u32]| {
        calls.set(calls.get() + 1);
        Ok::<_, FixtureError>(vec![b'x'; RESPONSE_REQUEST_BYTES + 1])
    };
    let decode: Option<&ResponseDecoder<'_, FixtureError>> = match decoder {
        Decoder::Fixture => Some(&fixture),
        Decoder::Missing => None,
        Decoder::WrongTerminal => Some(&wrong),
        Decoder::Oversized => Some(&oversized),
    };
    match parse_native_response(&envelope, admission, decode, RESPONSE_MAXIMUM_IDS) {
        Ok(result) => {
            out.u8(b'O');
            dump_or_error(out, &result);
        }
        Err(error) => {
            out.u8(b'E');
            out.bytes(error.to_string());
        }
    }
    out.u64(calls.get());
}
