//! Request, tool-schema and worker-readiness admission from `native/src/serve_protocol.cpp` (everything except `ServeSession`). CPU only; no tool is ever executed here.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use std::sync::Arc;
use std::time::SystemTimeError;
use tokspur_json::{
    Json, JsonError, JsonLimits, Kind, from_chars_f64, from_chars_i64, parse_json,
    parse_json_limited, to_chars_shortest,
};

use tokspur_text::response::{ParameterSpec, ParameterType, ResponseError, ToolSpec};
use tokspur_wire::mtp_wire::MtpWireError;

/// Policy governing whether tool parameter `enum` membership is enforced at the server or deferred to the agent executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolEnumValidation {
    #[default]
    Strict,
    Executor,
}

/// Parses `--tool-enum-validation` values (`strict` | `executor`).
pub fn parse_tool_enum_validation(value: &str) -> Result<ToolEnumValidation> {
    match value {
        "strict" => Ok(ToolEnumValidation::Strict),
        "executor" => Ok(ToolEnumValidation::Executor),
        _ => Err(AdmissionError::Refused(
            "tool-enum-validation must be strict or executor",
        )),
    }
}

/// Canonical short name for `ToolEnumValidation`.
pub const fn tool_enum_validation_name(value: ToolEnumValidation) -> &'static str {
    match value {
        ToolEnumValidation::Strict => "strict",
        ToolEnumValidation::Executor => "executor",
    }
}

/// Canonical startup policy identifier for `ToolEnumValidation`.
pub const fn tool_enum_validation_policy(value: ToolEnumValidation) -> &'static str {
    match value {
        ToolEnumValidation::Strict => "strict-server-enums-v1",
        ToolEnumValidation::Executor => "original-agent-executor-enums-v1",
    }
}

/// Policy governing whether tool argument schema properties/required/types are enforced at the server or deferred to the agent executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolArgValidation {
    #[default]
    Strict,
    Executor,
}

/// Parses `--tool-arg-validation` values (`strict` | `executor`).
pub fn parse_tool_arg_validation(value: &str) -> Result<ToolArgValidation> {
    match value {
        "strict" => Ok(ToolArgValidation::Strict),
        "executor" => Ok(ToolArgValidation::Executor),
        _ => Err(AdmissionError::Refused(
            "tool-arg-validation must be strict or executor",
        )),
    }
}

/// Canonical short name for `ToolArgValidation`.
pub const fn tool_arg_validation_name(value: ToolArgValidation) -> &'static str {
    match value {
        ToolArgValidation::Strict => "strict",
        ToolArgValidation::Executor => "executor",
    }
}

/// Canonical startup policy identifier for `ToolArgValidation`.
pub const fn tool_arg_validation_policy(value: ToolArgValidation) -> &'static str {
    match value {
        ToolArgValidation::Strict => "strict-server-args-v1",
        ToolArgValidation::Executor => "executor-deferred-args-v1",
    }
}

/// The C++ `FormatError` raised by serving admission.
#[derive(Clone)]
pub enum AdmissionError {
    /// `require`: a request, schema or readiness check refused.
    Refused(&'static str),
    /// A JSON accessor, parse or dump refused a value.
    Json(JsonError),
    /// A response parameter marker check refused a value.
    Response(ResponseError),
    /// System clock before UNIX epoch when building an API timestamp.
    Clock(SystemTimeError),
    /// Persisting a private prefix trace record failed.
    TracePersist(Arc<std::io::Error>),
    /// MTP wire validation refused inside a session.
    Mtp(MtpWireError),
}

impl PartialEq for AdmissionError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Refused(left), Self::Refused(right)) => left == right,
            (Self::Json(left), Self::Json(right)) => left == right,
            (Self::Response(left), Self::Response(right)) => left == right,
            (Self::Clock(left), Self::Clock(right)) => left.duration() == right.duration(),
            (Self::TracePersist(left), Self::TracePersist(right)) => {
                left.kind() == right.kind() && left.raw_os_error() == right.raw_os_error()
            }
            (Self::Mtp(left), Self::Mtp(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for AdmissionError {}

impl fmt::Debug for AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(reason) => formatter.debug_tuple("Refused").field(reason).finish(),
            Self::Json(error) => formatter.debug_tuple("Json").field(error).finish(),
            Self::Response(error) => formatter.debug_tuple("Response").field(error).finish(),
            Self::Clock(_) => formatter.write_str("Clock([redacted])"),
            Self::TracePersist(_) => formatter.write_str("TracePersist([redacted])"),
            Self::Mtp(error) => formatter.debug_tuple("Mtp").field(error).finish(),
        }
    }
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(reason) => formatter.write_str(reason),
            Self::Json(error) => error.fmt(formatter),
            Self::Response(error) => error.fmt(formatter),
            Self::Clock(_) => formatter.write_str("invalid API clock"),
            Self::TracePersist(_) => formatter.write_str("cannot persist private prefix trace"),
            Self::Mtp(MtpWireError::Refused(reason)) => formatter.write_str(reason),
            Self::Mtp(MtpWireError::MissingField(_)) => formatter.write_str("missing field"),
            Self::Mtp(MtpWireError::Json(error)) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Response(error) => Some(error),
            Self::Clock(error) => Some(error),
            Self::TracePersist(error) => Some(error.as_ref()),
            Self::Mtp(error) => Some(error),
            Self::Refused(_) => None,
        }
    }
}

impl From<JsonError> for AdmissionError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

impl From<ResponseError> for AdmissionError {
    fn from(error: ResponseError) -> Self {
        Self::Response(error)
    }
}

type Result<T> = std::result::Result<T, AdmissionError>;

pub(crate) fn require(ok: bool, why: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(AdmissionError::Refused(why))
    }
}

pub(crate) fn field(value: &Json, key: impl AsRef<[u8]>) -> Result<&Json> {
    value
        .find(key)?
        .ok_or(AdmissionError::Refused("missing required API field"))
}

fn keys(value: &Json, allowed: &[&str]) -> Result<()> {
    for (key, _) in value.object()? {
        require(
            allowed.iter().any(|name| name.as_bytes() == key.as_slice()),
            "unsupported API/schema field",
        )?;
    }
    Ok(())
}

pub(crate) fn number(value: &Json, low: f64, high: f64) -> Result<f64> {
    let text = value.number_lexeme()?;
    match from_chars_f64(text) {
        Some(result) if result.is_finite() && result >= low && result <= high => Ok(result),
        _ => Err(AdmissionError::Refused(
            "API number outside supported range",
        )),
    }
}

/// `real(double)`: `std::to_chars` then the owned parser, so the lexeme is exactly the C++ one.
pub(crate) fn real(value: f64) -> Result<Json> {
    let text =
        to_chars_shortest(value).ok_or(AdmissionError::Refused("cannot encode API number"))?;
    Ok(parse_json(text.as_bytes())?)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Decimal {
    negative: bool,
    digits: Vec<u8>,
    exponent: i64,
}

fn decimal(value: &Json) -> Result<Decimal> {
    let text = value.number_lexeme()?.as_bytes();
    require(text.len() <= 128, "tool numeric lexeme bound")?;
    let mut i = usize::from(text.first() == Some(&b'-'));
    let negative = i != 0;
    let mut digits = Vec::new();
    let mut fraction: i64 = 0;
    let mut after_point = false;
    while i < text.len() && text[i] != b'e' && text[i] != b'E' {
        if text[i] == b'.' {
            after_point = true;
        } else {
            digits.push(text[i]);
            if after_point {
                fraction += 1;
            }
        }
        i += 1;
    }
    let mut exponent: i64 = 0;
    if i < text.len() {
        i += 1;
        let mut minus = false;
        if matches!(text.get(i), Some(b'-' | b'+')) {
            minus = text[i] == b'-';
            i += 1;
        }
        let rest = std::str::from_utf8(&text[i..]).ok();
        match rest.and_then(from_chars_i64) {
            Some(parsed) if parsed <= 1_000_000 => exponent = parsed,
            _ => return Err(AdmissionError::Refused("tool numeric exponent bound")),
        }
        if minus {
            exponent = -exponent;
        }
    }
    exponent -= fraction;
    let Some(first) = digits.iter().position(|&c| c != b'0') else {
        return Ok(Decimal {
            negative: false,
            digits: b"0".to_vec(),
            exponent: 0,
        });
    };
    digits.drain(..first);
    while digits.last() == Some(&b'0') {
        digits.pop();
        exponent += 1;
    }
    Ok(Decimal {
        negative,
        digits,
        exponent,
    })
}

fn compare(left: &Decimal, right: &Decimal) -> i32 {
    if left.digits == b"0" && right.digits == b"0" {
        return 0;
    }
    if left.negative != right.negative {
        return if left.negative { -1 } else { 1 };
    }
    let mut order = 0;
    if left.digits == b"0" {
        order = -1;
    } else if right.digits == b"0" {
        order = 1;
    } else {
        let l = left.digits.len() as i64 + left.exponent;
        let r = right.digits.len() as i64 + right.exponent;
        if l != r {
            order = if l < r { -1 } else { 1 };
        } else {
            for i in 0..left.digits.len().max(right.digits.len()) {
                let a = left.digits.get(i).copied().unwrap_or(b'0');
                let b = right.digits.get(i).copied().unwrap_or(b'0');
                if a != b {
                    order = if a < b { -1 } else { 1 };
                    break;
                }
            }
        }
    }
    if left.negative { -order } else { order }
}

fn safe_name(name: &[u8], function: bool) -> Result<()> {
    require(
        !name.is_empty() && name.len() <= if function { 64 } else { 1024 },
        "tool name bound",
    )?;
    for &c in name {
        require(
            (32..127).contains(&c) && c != b'<' && c != b'>',
            "unsupported tool name",
        )?;
        if function {
            require(
                c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.',
                "unsupported function name",
            )?;
        }
    }
    Ok(())
}

fn schema_type(schema: &Json) -> Result<&[u8]> {
    if schema.find("anyOf")?.is_some() {
        return Ok(b"any");
    }
    let value = field(schema, "type")?.string()?;
    require(
        matches!(
            value,
            b"string" | b"boolean" | b"number" | b"integer" | b"object" | b"array" | b"null"
        ),
        "unsupported tool schema type",
    )?;
    Ok(value)
}

fn schema_admit(schema: &Json, depth: usize, nodes: &mut usize) -> Result<()> {
    require(
        depth <= 16 && {
            *nodes += 1;
            *nodes <= 4096
        },
        "tool schema complexity bound",
    )?;
    keys(
        schema,
        &[
            "type",
            "anyOf",
            "title",
            "default",
            "examples",
            "format",
            "$schema",
            "description",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "enum",
            "minimum",
            "exclusiveMinimum",
            "maximum",
        ],
    )?;
    let kind = schema_type(schema)?;
    if let Some(dialect) = schema.find("$schema")? {
        require(
            depth == 0 && dialect.string()? == b"https://json-schema.org/draft/2020-12/schema",
            "unsupported tool schema dialect",
        )?;
    }
    if let Some(description) = schema.find("description")? {
        require(
            description.string()?.len() <= 16384,
            "schema description bound",
        )?;
    }
    for name in ["title", "format"] {
        if let Some(value) = schema.find(name)? {
            require(value.string()?.len() <= 16384, "schema annotation bound")?;
        }
    }
    if let Some(examples) = schema.find("examples")? {
        require(examples.array()?.len() <= 128, "schema examples bound")?;
    }
    if kind == b"any" {
        keys(
            schema,
            &[
                "anyOf",
                "$schema",
                "title",
                "description",
                "default",
                "examples",
                "format",
            ],
        )?;
        let alternatives = field(schema, "anyOf")?.array()?;
        require(
            !alternatives.is_empty() && alternatives.len() <= 8,
            "schema union bound",
        )?;
        for branch in alternatives {
            schema_admit(branch, depth + 1, nodes)?;
        }
        return Ok(());
    }
    if kind == b"object" {
        let properties = schema.find("properties")?;
        if let Some(properties) = properties {
            require(properties.object()?.len() <= 32, "too many tool parameters")?;
            for (name, child) in properties.object()? {
                safe_name(name, false)?;
                schema_admit(child, depth + 1, nodes)?;
            }
        }
        if let Some(additional) = schema.find("additionalProperties")? {
            require(
                !additional.boolean()?,
                "additional tool properties are unsupported",
            )?;
        }
        if let Some(required) = schema.find("required")? {
            let mut seen = BTreeSet::new();
            for item in required.array()? {
                let name = item.string()?;
                let present = match properties {
                    Some(properties) => properties.find(name)?.is_some(),
                    None => false,
                };
                require(
                    present && seen.insert(name),
                    "invalid required tool parameter",
                )?;
            }
        }
    } else {
        require(
            schema.find("properties")?.is_none()
                && schema.find("required")?.is_none()
                && schema.find("additionalProperties")?.is_none(),
            "object schema keyword on other type",
        )?;
    }
    if kind == b"array" {
        schema_admit(field(schema, "items")?, depth + 1, nodes)?;
    } else {
        require(
            schema.find("items")?.is_none(),
            "array schema keyword on other type",
        )?;
    }
    if let Some(values) = schema.find("enum")? {
        require(
            kind != b"object"
                && kind != b"array"
                && !values.array()?.is_empty()
                && values.array()?.len() <= 128,
            "unsupported enum",
        )?;
        // Admission checks enum members against the same type, without enum.
        let base = Json::Object(
            schema
                .object()?
                .iter()
                .filter(|(key, _)| key.as_slice() != b"enum")
                .cloned()
                .collect(),
        );
        for value in values.array()? {
            validate_tool_value(&base, value)?;
        }
    }
    if schema.find("minimum")?.is_some()
        || schema.find("maximum")?.is_some()
        || schema.find("exclusiveMinimum")?.is_some()
    {
        require(
            kind == b"integer" || kind == b"number",
            "numeric bound on other schema type",
        )?;
        if let Some(low) = schema.find("minimum")? {
            number(low, -1e300, 1e300)?;
            decimal(low)?;
        }
        if let Some(high) = schema.find("maximum")? {
            number(high, -1e300, 1e300)?;
            decimal(high)?;
        }
        if let Some(low) = schema.find("exclusiveMinimum")? {
            number(low, -1e300, 1e300)?;
            decimal(low)?;
            if let Some(high) = schema.find("maximum")? {
                require(
                    compare(&decimal(low)?, &decimal(high)?) < 0,
                    "inverted schema bounds",
                )?;
            }
        }
        if let (Some(low), Some(high)) = (schema.find("minimum")?, schema.find("maximum")?) {
            require(
                compare(&decimal(low)?, &decimal(high)?) <= 0,
                "inverted schema bounds",
            )?;
        }
    }
    Ok(())
}

fn parameter_type(kind: &[u8]) -> ParameterType {
    match kind {
        b"string" => ParameterType::RawString,
        b"boolean" => ParameterType::Boolean,
        b"number" | b"integer" => ParameterType::Number,
        b"object" => ParameterType::Object,
        b"array" => ParameterType::Array,
        b"any" => ParameterType::JsonValue,
        _ => ParameterType::NullValue,
    }
}

fn validate_executor_string(raw: &[u8]) -> Result<()> {
    require(!raw.contains(&0), "NUL byte inside tool parameter")
}

fn validate_executor_node(value: &Json, depth: usize, nodes: &mut usize) -> Result<()> {
    *nodes += 1;
    require(
        depth <= 32 && *nodes <= 16384,
        "executor tool argument complexity bound",
    )?;
    match value.kind() {
        Kind::String => validate_executor_string(value.string()?),
        Kind::Number => {
            number(value, -1e300, 1e300)?;
            Ok(())
        }
        Kind::Array => {
            require(value.array()?.len() <= 4096, "tool array bound")?;
            for item in value.array()? {
                validate_executor_node(item, depth + 1, nodes)?;
            }
            Ok(())
        }
        Kind::Object => {
            require(value.object()?.len() <= 256, "too many tool parameters")?;
            let mut seen: BTreeSet<&[u8]> = BTreeSet::new();
            for (key, item) in value.object()? {
                if depth == 0 {
                    safe_name(key, false)?;
                }
                require(
                    seen.insert(key.as_slice()),
                    "duplicate executor tool property",
                )?;
                validate_executor_string(key)?;
                validate_executor_node(item, depth + 1, nodes)?;
            }
            Ok(())
        }
        Kind::Boolean | Kind::Null => Ok(()),
    }
}

/// Validates executor-deferred tool arguments with bounded depth/size, finite numbers and representable strings.
pub fn validate_executor_tool_arguments(value: &Json) -> Result<()> {
    require(
        value.kind() == Kind::Object,
        "executor tool arguments must be a JSON object",
    )?;
    let mut nodes = 0;
    validate_executor_node(value, 0, &mut nodes)
}

fn thinking_history(message: &Json, role: &[u8]) -> Result<()> {
    let Some(blocks) = message.find("thinking_blocks")? else {
        return Ok(());
    };
    require(
        role == b"assistant",
        "thinking blocks require assistant history",
    )?;
    require(blocks.array()?.len() <= 4096, "thinking block count bound")?;
    let mut text = Vec::new();
    for block in blocks.array()? {
        keys(block, &["type", "thinking", "signature"])?;
        require(
            field(block, "type")?.string()? == b"thinking",
            "unsupported thinking block type",
        )?;
        if let Some(signature) = block.find("signature")? {
            require(
                signature.string()?.is_empty(),
                "signed thinking blocks are unsupported",
            )?;
        }
        let part = field(block, "thinking")?.string()?;
        require(
            part.len() <= 1024 * 1024 - text.len(),
            "thinking history bound",
        )?;
        text.extend_from_slice(part);
    }
    let reasoning = match message.find("reasoning_content")? {
        Some(value) if !value.is_null() => value.string()?,
        _ => b"",
    };
    require(
        text == reasoning,
        "thinking history differs from reasoning content",
    )
}

fn history(
    messages: &Json,
    tools: &[Vec<u8>],
    schemas: &[Json],
    tool_enums: ToolEnumValidation,
    tool_args: ToolArgValidation,
) -> Result<()> {
    require(
        !messages.array()?.is_empty() && messages.array()?.len() <= 4096,
        "API history bound",
    )?;
    let mut pending: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    let mut used: BTreeSet<Vec<u8>> = BTreeSet::new();
    for message in messages.array()? {
        keys(
            message,
            &[
                "role",
                "content",
                "reasoning_content",
                "thinking_blocks",
                "tool_calls",
                "tool_call_id",
                "name",
            ],
        )?;
        let role = field(message, "role")?.string()?;
        require(
            matches!(
                role,
                b"system" | b"developer" | b"user" | b"assistant" | b"tool"
            ),
            "unsupported API message role",
        )?;
        thinking_history(message, role)?;
        if let Some(content) = message.find("content")? {
            if content.kind() == Kind::Array {
                require(content.array()?.len() <= 4096, "text content part bound")?;
                for part in content.array()? {
                    keys(part, &["type", "text"])?;
                    require(
                        field(part, "type")?.string()? == b"text",
                        "only text content parts supported",
                    )?;
                    field(part, "text")?.string()?;
                }
            } else {
                require(
                    content.is_null() || content.kind() == Kind::String,
                    "unsupported message content",
                )?;
            }
        }
        if role == b"tool" {
            require(
                message.find("tool_calls")?.is_none()
                    && message.find("reasoning_content")?.is_none(),
                "invalid tool result fields",
            )?;
            require(
                !field(message, "content")?.is_null(),
                "tool result requires text content",
            )?;
            let id = field(message, "tool_call_id")?.string()?;
            let Some(found) = pending.get(id).cloned() else {
                return Err(AdmissionError::Refused(
                    "unsolicited or duplicate tool result",
                ));
            };
            if let Some(name) = message.find("name")? {
                require(
                    name.string()? == found.as_slice(),
                    "tool result name/ID disagreement",
                )?;
            }
            pending.remove(id);
        } else {
            require(
                pending.is_empty(),
                "missing tool results before next message",
            )?;
            require(
                message.find("tool_call_id")?.is_none() && message.find("name")?.is_none(),
                "tool identity field outside tool result",
            )?;
            if role != b"assistant" {
                require(
                    !field(message, "content")?.is_null(),
                    "message requires text content",
                )?;
            }
            if let Some(reasoning) = message.find("reasoning_content")? {
                require(
                    role == b"assistant"
                        && (reasoning.is_null() || reasoning.kind() == Kind::String),
                    "invalid reasoning history",
                )?;
            }
            if let Some(calls) = message.find("tool_calls")? {
                require(
                    role == b"assistant"
                        && !calls.array()?.is_empty()
                        && calls.array()?.len() <= 128,
                    "invalid assistant tool history",
                )?;
                for call in calls.array()? {
                    keys(call, &["id", "type", "function", "index"])?;
                    if let Some(index) = call.find("index")? {
                        require(
                            index.unsigned_integer()? < 128,
                            "historical tool index bound",
                        )?;
                    }
                    require(
                        field(call, "type")?.string()? == b"function",
                        "unsupported historical call type",
                    )?;
                    let id = field(call, "id")?.string()?.to_vec();
                    require(
                        !id.is_empty() && id.len() <= 128 && used.insert(id.clone()),
                        "invalid/duplicate historical tool ID",
                    )?;
                    let function = field(call, "function")?;
                    keys(function, &["name", "arguments"])?;
                    let name = field(function, "name")?.string()?;
                    let found = tools.iter().position(|tool| tool.as_slice() == name);
                    let Some(index) = found else {
                        return Err(AdmissionError::Refused(
                            "historical function absent from admitted tools",
                        ));
                    };
                    let arguments = field(function, "arguments")?;
                    require(
                        arguments.kind() == Kind::Object || arguments.kind() == Kind::String,
                        "invalid historical tool arguments",
                    )?;
                    let parsed;
                    let value = if arguments.kind() == Kind::String {
                        // The server emits `arguments` from a call of at most 64 KiB, and escaping can grow each byte to six (`\u000a`). Accept everything this server can emit, so a client that echoes it verbatim is not refused on its next turn.
                        parsed = parse_json_limited(
                            arguments.string()?,
                            JsonLimits::new(6 * 65536, 65536, 33),
                        )?;
                        &parsed
                    } else {
                        arguments
                    };
                    if tool_args == ToolArgValidation::Executor {
                        validate_executor_tool_arguments(value)?;
                    } else {
                        validate_tool_value_with_policy(&schemas[index], value, tool_enums)?;
                    }
                    pending.entry(id).or_insert_with(|| name.to_vec());
                }
            }
        }
    }
    require(
        pending.is_empty(),
        "request ends with unanswered tool calls",
    )
}

/// The response parser's tool manifest, as JSON (`parser_manifest`).
pub(crate) fn parser_manifest(tools: &[ToolSpec]) -> Json {
    const TYPES: [&str; 7] = [
        "string", "boolean", "number", "object", "array", "null", "any",
    ];
    Json::Array(
        tools
            .iter()
            .map(|tool| {
                let parameters = tool
                    .parameters
                    .iter()
                    .map(|p| {
                        Json::new_object([
                            ("name", Json::from(p.name.clone())),
                            ("type", Json::from(TYPES[p.r#type as usize])),
                            ("required", Json::Bool(p.required)),
                        ])
                    })
                    .collect();
                Json::new_object([
                    ("name", Json::from(tool.name.clone())),
                    ("parameters", Json::Array(parameters)),
                ])
            })
            .collect(),
    )
}

/// One SSE `chat.completion.chunk`.
pub(crate) fn chunk(id: &[u8], alias: &[u8], created: u64, delta: Json, finish: Json) -> Json {
    Json::new_object([
        ("id", Json::from(id)),
        ("object", Json::from("chat.completion.chunk")),
        ("created", Json::integer(created)),
        ("model", Json::from(alias)),
        (
            "choices",
            Json::Array(vec![Json::new_object([
                ("index", Json::integer(0)),
                ("delta", delta),
                ("finish_reason", finish),
            ])]),
        ),
    ])
}

pub(crate) fn worker_execution(value: &Json, selected: &str) -> Result<()> {
    require(
        selected == "direct" || selected == "recurrent-replay",
        "unsupported serving startup execution mode",
    )?;
    require(
        field(value, "small_gemv_column_splits")?.unsigned_integer()? == 16
            && field(value, "block_gemv_formats")?.unsigned_integer()? == 3
            && field(value, "stack_execution")?.string()? == selected.as_bytes(),
        "worker execution policy changed",
    )?;
    if selected == "recurrent-replay" {
        require(
            field(value, "replay_prepared")?.boolean()?
                && field(value, "recurrent_replay_captures")?.unsigned_integer()? == 24
                && field(value, "mapped_weight_region_limit_bytes")?.unsigned_integer()? == 4 << 30,
            "owned recurrent replay preparation/capture/region identity mismatch",
        )?;
    }
    Ok(())
}

pub(crate) fn broker_storage(value: &Json, ready: &Json) -> Result<()> {
    let slots = field(ready, "checkpoint_slots")?.unsigned_integer()?;
    let budget = field(ready, "checkpoint_combined_budget_bytes")?.unsigned_integer()?;
    require(
        slots != 0 && slots <= 4 && budget != 0 && budget <= (512 << 20),
        "invalid serving broker storage preparation",
    )?;
    require(
        field(value, "checkpoint_slots")?.unsigned_integer()? == slots
            && field(value, "checkpoint_combined_budget_bytes")?.unsigned_integer()? == budget,
        "serving broker configuration changed",
    )?;
    let mut total: u64 = 0;
    for name in [
        "checkpoint_host_bytes",
        "checkpoint_store_host_bytes",
        "checkpoint_gpu_bytes",
    ] {
        let bytes = field(value, name)?.unsigned_integer()?;
        require(
            bytes != 0
                && bytes <= budget.wrapping_sub(total)
                && bytes == field(ready, name)?.unsigned_integer()?,
            "serving broker storage charge changed",
        )?;
        total = total.wrapping_add(bytes);
    }
    Ok(())
}

/// CPU-only startup admission for the fixed owned target policy. Every field is mandatory and type-checked; absent metadata never implies the direct path.
pub fn validate_worker_ready(
    ready: &Json,
    stack_execution: &str,
    checkpoint_slots: u32,
    checkpoint_budget: u64,
) -> Result<()> {
    require(
        field(ready, "event")?.string()? == b"ready"
            && field(ready, "maximum_generated_tokens")?.unsigned_integer()? == 4096,
        "owned worker readiness/execution identity mismatch",
    )?;
    worker_execution(ready, stack_execution)?;
    require(
        checkpoint_slots <= 4
            && checkpoint_budget <= (512 << 20)
            && (checkpoint_slots != 0) == (checkpoint_budget != 0),
        "invalid serving broker preparation",
    )?;
    if checkpoint_slots != 0 {
        require(
            stack_execution == "direct"
                && field(ready, "checkpoint_broker")?.boolean()?
                && field(ready, "broker_epoch")?.unsigned_integer()? != 0
                && field(ready, "checkpoint_slots")?.unsigned_integer()?
                    == u64::from(checkpoint_slots)
                && field(ready, "checkpoint_combined_budget_bytes")?.unsigned_integer()?
                    == checkpoint_budget,
            "owned broker readiness identity mismatch",
        )?;
        broker_storage(ready, ready)?;
    } else if let Some(enabled) = ready.find("checkpoint_broker")? {
        require(!enabled.boolean()?, "unexpected serving broker capability")?;
    }
    Ok(())
}

/// Validates the supported schema subset recursively under strict `enum` enforcement.
pub fn validate_tool_value(schema: &Json, value: &Json) -> Result<()> {
    validate_tool_value_with_policy(schema, value, ToolEnumValidation::Strict)
}

/// Validates the supported schema subset recursively under the selected `ToolEnumValidation` policy.
pub fn validate_tool_value_with_policy(
    schema: &Json,
    value: &Json,
    tool_enums: ToolEnumValidation,
) -> Result<()> {
    let kind = schema_type(schema)?;
    if kind == b"any" {
        return require(
            field(schema, "anyOf")?
                .array()?
                .iter()
                .any(|branch| validate_tool_value_with_policy(branch, value, tool_enums).is_ok()),
            "tool value matches no schema alternative",
        );
    }
    match kind {
        b"string" => {
            require(value.kind() == Kind::String, "tool value is not string")?;
            if tool_enums == ToolEnumValidation::Executor {
                validate_executor_string(value.string()?)?;
            }
        }
        b"boolean" => require(value.kind() == Kind::Boolean, "tool value is not boolean")?,
        b"null" => require(value.is_null(), "tool value is not null")?,
        b"number" | b"integer" => {
            number(value, -1e300, 1e300)?;
            let n = decimal(value)?;
            let safe = decimal(&parse_json(b"9007199254740991")?)?;
            let mut magnitude = n.clone();
            magnitude.negative = false;
            require(
                kind != b"integer"
                    || ((n.digits == b"0" || n.exponent >= 0)
                        && compare(&magnitude, &safe) <= 0
                        && !value.number_lexeme()?.contains(['.', 'e', 'E'])),
                "tool integer must use exact safe-integer syntax",
            )?;
            if let Some(minimum) = schema.find("minimum")? {
                require(
                    compare(&n, &decimal(minimum)?) >= 0,
                    "tool value below minimum",
                )?;
            }
            if let Some(maximum) = schema.find("maximum")? {
                require(
                    compare(&n, &decimal(maximum)?) <= 0,
                    "tool value above maximum",
                )?;
            }
            if let Some(minimum) = schema.find("exclusiveMinimum")? {
                require(
                    compare(&n, &decimal(minimum)?) > 0,
                    "tool value at or below exclusive minimum",
                )?;
            }
        }
        b"array" => {
            require(value.array()?.len() <= 4096, "tool array bound")?;
            for item in value.array()? {
                validate_tool_value_with_policy(field(schema, "items")?, item, tool_enums)?;
            }
        }
        _ => {
            let properties = schema.find("properties")?;
            for (name, item) in value.object()? {
                let child = match properties {
                    Some(properties) => properties.find(name)?,
                    None => None,
                };
                let Some(child) = child else {
                    return Err(AdmissionError::Refused("unknown tool property"));
                };
                validate_tool_value_with_policy(child, item, tool_enums)?;
            }
            if let Some(required) = schema.find("required")? {
                for name in required.array()? {
                    require(
                        value.find(name.string()?)?.is_some(),
                        "missing required tool property",
                    )?;
                }
            }
        }
    }
    if tool_enums == ToolEnumValidation::Strict
        && let Some(values) = schema.find("enum")?
    {
        let mut found = false;
        for candidate in values.array()? {
            if candidate.kind() == Kind::Number && value.kind() == Kind::Number {
                found |= compare(&decimal(candidate)?, &decimal(value)?) == 0;
            } else {
                found |= candidate.dump()? == value.dump()?;
            }
        }
        require(found, "tool value outside enum")?;
    }
    Ok(())
}

/// Checks announced controller sampling modes before rendering or worker submission.
pub fn validate_calibrated_sampling(sampling: &Json, ready: &Json) -> Result<()> {
    if ready.kind() != Kind::Object
        || ready
            .find("native_measured_controller")?
            .is_none_or(|value| value != &Json::Bool(true))
    {
        return Ok(());
    }
    let temperature = number(field(sampling, "temperature")?, 0.0, 100.0)?;
    let mode: &[u8] = if temperature == 0.0 {
        b"greedy".as_slice()
    } else {
        require(
            temperature == 1.0 && number(field(sampling, "min_p")?, 0.0, 1.0)? == 0.0,
            "sampling policy is not calibrated by measured profile",
        )?;
        let top_k = field(sampling, "top_k")?.unsigned_integer()?;
        let top_p = number(field(sampling, "top_p")?, 0.0, 1.0)?;
        if top_k == 20 && top_p == 0.95 {
            b"coding"
        } else {
            require(
                top_p == 1.0 && matches!(top_k, 0 | 20),
                "sampling policy is not calibrated by measured profile",
            )?;
            if top_k == 20 { b"topk" } else { b"full" }
        }
    };
    let modes = field(ready, "native_controller_sampling_modes")?.array()?;
    require(
        !modes.is_empty()
            && modes.len() <= 4
            && modes
                .iter()
                .any(|value| value.string().is_ok_and(|value| value == mode)),
        "sampling mode is not calibrated by measured profile",
    )
}

/// `AdmittedChat`.
#[derive(Clone)]
pub struct AdmittedChat {
    pub render: Json,
    pub sampling: Json,
    pub manifest: Vec<ToolSpec>,
    pub schemas: Vec<Json>,
    pub names: Vec<Vec<u8>>,
    pub seed: Option<u64>,
    pub tool_choice: Vec<u8>,
    pub named_tool_choice: bool,
    pub single_tool_call: bool,
    pub stream: bool,
    pub include_usage: bool,
    pub thinking: bool,
    pub maximum_tokens: u32,
}

impl fmt::Debug for AdmittedChat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Messages, tool schemas and names stay out of debug output.
        formatter
            .debug_struct("AdmittedChat")
            .field("tools", &self.names.len())
            .field("stream", &self.stream)
            .field("thinking", &self.thinking)
            .field("maximum_tokens", &self.maximum_tokens)
            .finish_non_exhaustive()
    }
}

/// `admit_chat`: validates one OpenAI-style request without side effects under strict `enum` enforcement.
pub fn admit_chat(request: &Json, alias: &[u8]) -> Result<AdmittedChat> {
    admit_chat_with_policy(request, alias, ToolEnumValidation::Strict)
}

/// `admit_chat`: validates one OpenAI-style request without side effects under the selected `ToolEnumValidation` policy.
pub fn admit_chat_with_policy(
    request: &Json,
    alias: &[u8],
    tool_enums: ToolEnumValidation,
) -> Result<AdmittedChat> {
    admit_chat_with_policies(request, alias, tool_enums, ToolArgValidation::Strict)
}

/// `admit_chat`: validates one OpenAI-style request without side effects under the selected `ToolEnumValidation` and `ToolArgValidation` policies.
pub fn admit_chat_with_policies(
    request: &Json,
    alias: &[u8],
    tool_enums: ToolEnumValidation,
    tool_args: ToolArgValidation,
) -> Result<AdmittedChat> {
    request.dump_limited(1024 * 1024)?;
    keys(
        request,
        &[
            "model",
            "messages",
            "n",
            "parallel_tool_calls",
            "stream",
            "stream_options",
            "keep_alive",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "tools",
            "tool_choice",
            "enable_thinking",
            "preserve_thinking",
            "reasoning_effort",
            "chat_template_kwargs",
            "presence_penalty",
            "repetition_penalty",
        ],
    )?;
    require(
        field(request, "model")?.string()? == alias,
        "unknown model alias",
    )?;
    let mut out = AdmittedChat {
        render: Json::Null,
        sampling: Json::Null,
        manifest: Vec::new(),
        schemas: Vec::new(),
        names: Vec::new(),
        seed: None,
        tool_choice: Vec::new(),
        named_tool_choice: false,
        single_tool_call: false,
        stream: false,
        include_usage: false,
        thinking: true,
        maximum_tokens: 8192,
    };
    if let Some(n) = request.find("n")? {
        require(n.unsigned_integer()? == 1, "only n=1 is supported")?;
    }
    if let Some(parallel) = request.find("parallel_tool_calls")? {
        out.single_tool_call = !parallel.boolean()?;
    }
    if let Some(p) = request.find("stream")? {
        out.stream = p.boolean()?;
    }
    if let Some(options) = request.find("stream_options")? {
        require(out.stream, "stream_options requires streaming")?;
        keys(options, &["include_usage"])?;
        if let Some(include) = options.find("include_usage")? {
            out.include_usage = include.boolean()?;
        }
    }
    // Numeric lifetime hints do not change the owned worker's supervised lifetime.
    if let Some(keep) = request.find("keep_alive")? {
        if keep.kind() == Kind::String {
            let value = keep.string()?;
            let numeric = value
                .strip_suffix(b"ms")
                .or_else(|| value.strip_suffix(b"s"))
                .or_else(|| value.strip_suffix(b"m"))
                .or_else(|| value.strip_suffix(b"h"))
                .unwrap_or(value);
            require(
                !value.is_empty()
                    && value.len() <= 64
                    && (value == b"-1"
                        || (!numeric.is_empty()
                            && numeric.iter().all(|c| c.is_ascii_digit() || *c == b'.')
                            && numeric.iter().filter(|c| **c == b'.').count() <= 1
                            && numeric.iter().any(u8::is_ascii_digit))),
                "invalid keep_alive duration hint",
            )?;
        } else {
            number(keep, -1.0, 1e12)?;
        }
    }
    let template_options = request.find("chat_template_kwargs")?;
    if let Some(template) = template_options {
        keys(
            template,
            &["enable_thinking", "preserve_thinking", "reasoning_effort"],
        )?;
    }
    let mut canonical_options = Vec::new();
    for name in ["enable_thinking", "preserve_thinking", "reasoning_effort"] {
        let outer = request.find(name)?;
        let inner = match template_options {
            Some(template) => template.find(name)?,
            None => None,
        };
        if name == "reasoning_effort" {
            if let Some(p) = outer {
                p.string()?;
            }
            if let Some(p) = inner {
                p.string()?;
            }
        } else {
            if let Some(p) = outer {
                p.boolean()?;
            }
            if let Some(p) = inner {
                p.boolean()?;
            }
        }
        if let (Some(a), Some(b)) = (outer, inner) {
            require(a.dump()? == b.dump()?, "conflicting chat template options")?;
        }
        if let Some(selected) = outer.or(inner) {
            canonical_options.push((name.as_bytes().to_vec(), selected.clone()));
        }
    }
    let options = Json::Object(canonical_options);
    if let Some(p) = options.find("enable_thinking")? {
        out.thinking = p.boolean()?;
    }
    if let Some(p) = options.find("reasoning_effort")? {
        let effort = p.string()?;
        require(
            effort == b"xhigh"
                || effort == b"high"
                || effort == b"medium"
                || effort == b"low"
                || (!out.thinking && effort == b"none"),
            "unsupported reasoning effort",
        )?;
    }
    require(
        !(request.find("max_tokens")?.is_some()
            && request.find("max_completion_tokens")?.is_some()),
        "provide only one output limit",
    )?;
    let limit = match request.find("max_tokens")? {
        Some(limit) => Some(limit),
        None => request.find("max_completion_tokens")?,
    };
    if let Some(limit) = limit {
        let n = limit.unsigned_integer()?;
        require(n != 0 && n <= 32768, "output limit must be 1..32768")?;
        out.maximum_tokens = n as u32;
    }
    for name in ["presence_penalty", "repetition_penalty"] {
        if let Some(p) = request.find(name)? {
            let neutral = parse_json(if name == "presence_penalty" {
                b"0"
            } else {
                b"1"
            })?;
            require(
                compare(&decimal(p)?, &decimal(&neutral)?) == 0,
                "non-neutral sampling penalty is unsupported",
            )?;
        }
    }
    let temperature = match request.find("temperature")? {
        Some(value) => number(value, 0.0, 100.0)?,
        None => 0.0,
    };
    let top_p = match request.find("top_p")? {
        Some(value) => number(value, 0.0, 1.0)?,
        None => 1.0,
    };
    require(top_p > 0.0, "top_p must be positive")?;
    let min_p = match request.find("min_p")? {
        Some(value) => number(value, 0.0, 1.0)?,
        None => 0.0,
    };
    let top_k = match request.find("top_k")? {
        Some(value) => value.unsigned_integer()?,
        None => 0,
    };
    require(top_k <= 248_320, "top_k bound")?;
    out.sampling = Json::new_object([
        ("temperature", real(temperature)?),
        ("top_k", Json::integer(top_k)),
        ("top_p", real(top_p)?),
        ("min_p", real(min_p)?),
    ]);
    if let Some(seed) = request.find("seed")? {
        out.seed = Some(seed.unsigned_integer()?);
    }
    let mut declarations: Vec<Json> = Vec::new();
    let mut names: BTreeSet<Vec<u8>> = BTreeSet::new();
    if let Some(tools) = request.find("tools")? {
        require(tools.array()?.len() <= 64, "API tool limit")?;
        for declaration in tools.array()? {
            keys(declaration, &["type", "function"])?;
            require(
                field(declaration, "type")?.string()? == b"function",
                "unsupported tool type",
            )?;
            let function = field(declaration, "function")?;
            keys(function, &["name", "description", "parameters", "strict"])?;
            let name = field(function, "name")?.string()?.to_vec();
            safe_name(&name, true)?;
            require(names.insert(name.clone()), "duplicate tool name")?;
            if let Some(description) = function.find("description")? {
                require(
                    description.string()?.len() <= 16384,
                    "tool description bound",
                )?;
            }
            if let Some(strict) = function.find("strict")? {
                strict.boolean()?;
            }
            let schema = match function.find("parameters")? {
                Some(parameters) => parameters.clone(),
                None => parse_json(
                    br#"{"type":"object","properties":{},"additionalProperties":false}"#,
                )?,
            };
            let mut nodes = 0;
            schema_admit(&schema, 0, &mut nodes)?;
            require(
                schema_type(&schema)? == b"object",
                "function parameters must be object",
            )?;
            let mut manifest = ToolSpec {
                name: name.clone(),
                parameters: Vec::new(),
            };
            let mut required: BTreeSet<Vec<u8>> = BTreeSet::new();
            if let Some(p) = schema.find("required")? {
                for value in p.array()? {
                    required.insert(value.string()?.to_vec());
                }
            }
            if let Some(p) = schema.find("properties")? {
                for (key, value) in p.object()? {
                    manifest.parameters.push(ParameterSpec {
                        name: key.clone(),
                        r#type: parameter_type(schema_type(value)?),
                        required: required.contains(key),
                    });
                }
            }
            out.manifest.push(manifest);
            out.schemas.push(schema);
            out.names.push(name);
            declarations.push(declaration.clone());
        }
    }
    out.tool_choice = b"auto".to_vec();
    if let Some(choice) = request.find("tool_choice")? {
        if choice.kind() == Kind::String {
            out.tool_choice = choice.string()?.to_vec();
            require(
                matches!(out.tool_choice.as_slice(), b"auto" | b"none" | b"required"),
                "unsupported tool_choice",
            )?;
        } else {
            out.named_tool_choice = true;
            keys(choice, &["type", "function"])?;
            require(
                field(choice, "type")?.string()? == b"function",
                "unsupported named tool_choice",
            )?;
            let function = field(choice, "function")?;
            keys(function, &["name"])?;
            out.tool_choice = field(function, "name")?.string()?.to_vec();
            require(
                names.contains(&out.tool_choice),
                "unknown named tool_choice",
            )?;
        }
    }
    let choice = out.tool_choice.clone();
    require(
        choice == b"auto" || choice == b"none" || !declarations.is_empty(),
        "tool choice requires tools",
    )?;
    let messages = field(request, "messages")?;
    history(messages, &out.names, &out.schemas, tool_enums, tool_args)?;
    let mut rendered_messages = messages.array()?.clone();
    if out.named_tool_choice || choice == b"required" {
        let instruction = if !out.named_tool_choice {
            b"For this response, call at least one of the declared tools. Do not answer without a tool call.".to_vec()
        } else {
            let mut text = b"For this response, call only the declared function named ".to_vec();
            text.extend_from_slice(&choice);
            text.extend_from_slice(b". Do not answer without that tool call.");
            text
        };
        rendered_messages.insert(
            0,
            Json::new_object([
                ("role", Json::from("system")),
                ("content", Json::String(instruction)),
            ]),
        );
    }
    if !out.named_tool_choice && choice == b"none" {
        declarations.clear();
        out.manifest.clear();
        out.schemas.clear();
        out.names.clear();
    } else if out.named_tool_choice {
        let index = out
            .names
            .iter()
            .position(|name| *name == choice)
            .unwrap_or(out.names.len());
        let (Some(declaration), Some(manifest), Some(schema), Some(name)) = (
            declarations.get(index).cloned(),
            out.manifest.get(index).cloned(),
            out.schemas.get(index).cloned(),
            out.names.get(index).cloned(),
        ) else {
            return Err(AdmissionError::Refused("unknown named tool_choice"));
        };
        declarations = vec![declaration];
        out.manifest = vec![manifest];
        out.schemas = vec![schema];
        out.names = vec![name];
    }
    let mut render = vec![
        (b"messages".to_vec(), Json::Array(rendered_messages)),
        (b"tools".to_vec(), Json::Array(declarations)),
        (b"enable_thinking".to_vec(), Json::Bool(out.thinking)),
    ];
    for name in ["preserve_thinking", "reasoning_effort"] {
        if let Some(p) = options.find(name)? {
            render.push((name.as_bytes().to_vec(), p.clone()));
        }
    }
    out.render = Json::Object(render);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexeme(text: &str) -> Json {
        parse_json(text.as_bytes()).unwrap()
    }

    #[test]
    fn decimal_normalizes_zeros_signs_and_exponents() {
        let d = decimal(&lexeme("-12.3400e2")).unwrap();
        assert_eq!(
            d,
            Decimal {
                negative: true,
                digits: b"1234".to_vec(),
                exponent: 0
            }
        );
        assert_eq!(
            decimal(&lexeme("-0.000")).unwrap(),
            Decimal {
                negative: false,
                digits: b"0".to_vec(),
                exponent: 0
            }
        );
        assert_eq!(
            decimal(&lexeme("1e1000001")),
            Err(AdmissionError::Refused("tool numeric exponent bound"))
        );
    }

    #[test]
    fn compare_orders_by_magnitude_then_digits() {
        let parse = |text: &str| decimal(&lexeme(text)).unwrap();
        assert_eq!(compare(&parse("10"), &parse("9.99")), 1);
        assert_eq!(compare(&parse("-10"), &parse("-9.99")), -1);
        assert_eq!(compare(&parse("1.50"), &parse("15e-1")), 0);
        assert_eq!(compare(&parse("0"), &parse("-0.0")), 0);
        assert_eq!(compare(&parse("-1"), &parse("0")), -1);
    }

    #[test]
    fn opencode_schema_dialect_and_exclusive_bound_are_checked() {
        let schema = lexeme(
            r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"timeout":{"type":"number","exclusiveMinimum":0,"maximum":60}},"required":["timeout"]}"#,
        );
        schema_admit(&schema, 0, &mut 0).unwrap();
        for value in ["0.0001", "1", "60"] {
            let value = lexeme(&format!(r#"{{"timeout":{value}}}"#));
            validate_tool_value(&schema, &value).unwrap();
        }
        for value in ["0", "-0.0", "-1", "60.01"] {
            let value = lexeme(&format!(r#"{{"timeout":{value}}}"#));
            assert!(validate_tool_value(&schema, &value).is_err());
        }
        for text in [
            r#"{"type":"number","exclusiveMinimum":true}"#,
            r#"{"type":"string","exclusiveMinimum":0}"#,
            r#"{"type":"number","exclusiveMinimum":1,"maximum":1}"#,
            r#"{"type":"object","$schema":"https://example.invalid/schema"}"#,
            r#"{"type":"object","$schema":12}"#,
            r#"{"type":"object","$schema":"https://json-schema.org/draft/2020-12/schema","$ref":"remote"}"#,
        ] {
            assert!(schema_admit(&lexeme(text), 0, &mut 0).is_err());
        }
        assert!(schema_admit(&schema, 1, &mut 0).is_err());
    }

    #[test]
    fn sampling_values_keep_the_cpp_to_chars_lexemes() {
        let admitted = admit_chat(
            &lexeme(r#"{"model":"m","messages":[{"role":"user","content":"x"}],"temperature":0.00001,"top_p":0.7}"#),
            b"m",
        )
        .unwrap();
        assert_eq!(
            admitted.sampling.dump().unwrap(),
            br#"{"temperature":1e-05,"top_k":0,"top_p":0.7,"min_p":0}"#
        );
    }

    #[test]
    fn admitted_chat_debug_counts_tools_without_messages_or_names() {
        // Arrange
        let request = lexeme(
            r#"{"model":"m","messages":[{"role":"user","content":"private prompt"}],"tools":[{"type":"function","function":{"name":"private_tool","parameters":{"type":"object","properties":{}}}}],"max_tokens":7}"#,
        );

        // Act
        let rendered = format!("{:?}", admit_chat(&request, b"m").unwrap());

        // Assert
        assert_eq!(
            rendered,
            "AdmittedChat { tools: 1, stream: false, thinking: true, maximum_tokens: 7, .. }"
        );
        assert!(!rendered.contains("private"));
    }

    #[test]
    fn executor_enum_policy_defers_membership_and_accepts_literal_markers() {
        assert_eq!(
            parse_tool_enum_validation("strict"),
            Ok(ToolEnumValidation::Strict)
        );
        assert_eq!(
            parse_tool_enum_validation("executor"),
            Ok(ToolEnumValidation::Executor)
        );
        assert_eq!(
            parse_tool_enum_validation("permissive"),
            Err(AdmissionError::Refused(
                "tool-enum-validation must be strict or executor"
            ))
        );
        assert_eq!(
            tool_enum_validation_policy(ToolEnumValidation::Strict),
            "strict-server-enums-v1"
        );
        assert_eq!(
            tool_enum_validation_policy(ToolEnumValidation::Executor),
            "original-agent-executor-enums-v1"
        );

        let schema = lexeme(
            r#"{"type":"object","properties":{"cmd":{"type":"string","enum":["cargo test"]}},"required":["cmd"],"additionalProperties":false}"#,
        );
        let outside = lexeme(r#"{"cmd":"cargo test --frozen"}"#);
        assert_eq!(
            validate_tool_value(&schema, &outside),
            Err(AdmissionError::Refused("tool value outside enum"))
        );
        assert_eq!(
            validate_tool_value_with_policy(&schema, &outside, ToolEnumValidation::Executor),
            Ok(())
        );

        for value in [
            r#"{"cmd":"echo <|im_start|>"}"#,
            r#"{"cmd":"echo <|endoftext|>"}"#,
        ] {
            assert_eq!(
                validate_tool_value_with_policy(
                    &schema,
                    &lexeme(value),
                    ToolEnumValidation::Executor
                ),
                Ok(())
            );
        }
        assert!(
            validate_tool_value_with_policy(
                &schema,
                &lexeme(r#"{"cmd":"\u0000"}"#),
                ToolEnumValidation::Executor
            )
            .is_err()
        );
    }
}
