//! Chat rendering for the pinned Flash-Next template, ported one to one from `native/src/chat.cpp`.
//!
//! [`ChatStrings`] are literal DATA read from the pinned model template, not executable template code. The rendering rules are independently authored and reproduce that template: message normalization, tool-argument parsing, system-message hoisting, reasoning history and the Python-`json.dumps`-like [`template_json`]. Every refusal carries the exact C++ `FormatError` text.

use std::fmt;

use tokspur_json::{
    Json, JsonError, Kind, UnicodeError, parse_json, push_scalar_utf8, unicode_scalars,
};

use crate::cxx::{FromChars, find, from_chars_f64, to_chars_scientific};

/// Byte bound for every rendered string (`output_limit`).
const OUTPUT_LIMIT: usize = 8 * 1024 * 1024;

/// The C++ `FormatError` raised while extracting literals or rendering. Each `Display` is the exact C++ `what()` text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatError {
    /// `extract_flash_next_literals` refused the template source.
    Extract(&'static str),
    /// `ChatRenderer::new`, `render` or `template_json` refused its input.
    Render(&'static str),
    /// A typed JSON accessor found another kind (`expected JSON object`, ...).
    Access(JsonError),
    /// `parse_json` refused a JSON-string tool-argument payload.
    Arguments(JsonError),
    /// Text is not valid, bounded UTF-8 (`unicode_scalars`/`scalar_utf8`).
    Text(UnicodeError),
}

impl fmt::Display for ChatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Extract(reason) | Self::Render(reason) => formatter.write_str(reason),
            Self::Access(error) | Self::Arguments(error) => error.fmt(formatter),
            Self::Text(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ChatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Access(error) | Self::Arguments(error) => Some(error),
            Self::Text(error) => Some(error),
            Self::Extract(_) | Self::Render(_) => None,
        }
    }
}

impl From<JsonError> for ChatError {
    /// Typed accessors only; `parse_json` failures map to [`ChatError::Arguments`].
    fn from(error: JsonError) -> Self {
        Self::Access(error)
    }
}

impl From<UnicodeError> for ChatError {
    fn from(error: UnicodeError) -> Self {
        Self::Text(error)
    }
}

fn require(condition: bool, message: &'static str) -> Result<(), ChatError> {
    if condition {
        Ok(())
    } else {
        Err(ChatError::Render(message))
    }
}

fn require_extract(condition: bool, message: &'static str) -> Result<(), ChatError> {
    if condition {
        Ok(())
    } else {
        Err(ChatError::Extract(message))
    }
}

/// `Builder`: an append-only string bounded by [`OUTPUT_LIMIT`].
struct Builder {
    value: Vec<u8>,
    literal_ranges: Vec<std::ops::Range<usize>>,
}

impl Builder {
    fn new() -> Self {
        Self {
            value: Vec::new(),
            literal_ranges: Vec::new(),
        }
    }

    fn add(&mut self, value: &[u8]) -> Result<(), ChatError> {
        if value.len() > OUTPUT_LIMIT - self.value.len() {
            return Err(ChatError::Render("rendered chat exceeds byte limit"));
        }
        self.value.extend_from_slice(value);
        Ok(())
    }

    fn finish(self) -> Vec<u8> {
        self.value
    }

    fn literal(&mut self, value: &[u8]) -> Result<(), ChatError> {
        let start = self.value.len();
        self.add(value)?;
        if start != self.value.len() {
            if let Some(last) = self
                .literal_ranges
                .last_mut()
                .filter(|last| last.end == start)
            {
                last.end = self.value.len();
            } else {
                self.literal_ranges.push(start..self.value.len());
            }
        }
        Ok(())
    }

    fn parameter(&mut self, value: &[u8]) -> Result<(), ChatError> {
        require(
            !value.windows(13).any(|window| window == b"\n</parameter>"),
            "raw tool argument contains parameter delimiter",
        )?;
        self.literal(value)
    }
}

/// Python-style `str.strip` whitespace used by the pinned template, not NFC.
fn whitespace(c: u32) -> bool {
    (9..=13).contains(&c)
        || (0x1c..=0x20).contains(&c)
        || c == 0x85
        || c == 0xa0
        || c == 0x1680
        || (0x2000..=0x200a).contains(&c)
        || c == 0x2028
        || c == 0x2029
        || c == 0x202f
        || c == 0x205f
        || c == 0x3000
}

fn option(request: &Json, name: &str, fallback: bool) -> Result<bool, ChatError> {
    match request.find(name)? {
        Some(value) => Ok(value.boolean()?),
        None => Ok(fallback),
    }
}

fn role(message: &Json) -> Result<&[u8], ChatError> {
    let Some(value) = message.find("role")? else {
        return Err(ChatError::Render("chat message lacks role"));
    };
    let name = value.string()?;
    if !matches!(
        name,
        b"system" | b"developer" | b"user" | b"assistant" | b"tool"
    ) {
        return Err(ChatError::Render("unexpected message role"));
    }
    Ok(name)
}

fn content(value: Option<&Json>) -> Result<Vec<u8>, ChatError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    if value.kind() == Kind::String {
        return Ok(value.string()?.to_vec());
    }
    if value.kind() != Kind::Array {
        return Err(ChatError::Render("unexpected text content type"));
    }
    let mut output = Builder::new();
    for item in value.array()? {
        let kind = item.find("type")?;
        // Short-circuits in the C++ order: `type` is read as a string last.
        let media = item.find("image")?.is_some()
            || item.find("image_url")?.is_some()
            || item.find("video")?.is_some()
            || match kind {
                Some(kind) => matches!(kind.string()?, b"image" | b"video"),
                None => false,
            };
        if media {
            return Err(ChatError::Render(
                "text-only chat does not accept images or video",
            ));
        }
        if let Some(kind) = kind {
            require(
                kind.string()? == b"text",
                "unsupported text content item type",
            )?;
        }
        let Some(text) = item.find("text")? else {
            return Err(ChatError::Render("unexpected text content item"));
        };
        output.add(text.string()?)?;
    }
    Ok(output.finish())
}

fn message_content(message: &Json) -> Result<Vec<u8>, ChatError> {
    Ok(chat_trim(&content(message.find("content")?)?)?)
}

/// Replaces the first field named `name`, or appends it.
fn replace(mut fields: tokspur_json::Object, name: &[u8], value: Json) -> Json {
    if let Some((_, item)) = fields.iter_mut().find(|(key, _)| key.as_slice() == name) {
        *item = value;
        return Json::Object(fields);
    }
    fields.push((name.to_vec(), value));
    Json::Object(fields)
}

fn normalize_arguments(message: &Json, parsed: &mut usize) -> Result<Json, ChatError> {
    let Some(calls) = message.find("tool_calls")?.filter(|calls| !calls.is_null()) else {
        return Ok(message.clone());
    };
    let mut result = Vec::new();
    require(
        calls.array()?.len() <= 128,
        "too many tool calls in one message",
    )?;
    for call in calls.array()? {
        let wrapped = call.find("function")?;
        let function = wrapped.unwrap_or(call);
        let payload = match function.find("arguments")? {
            Some(args) if args.kind() == Kind::String => {
                let text = args.string()?;
                if chat_trim(text)?.is_empty() {
                    None
                } else {
                    Some(text)
                }
            }
            _ => None,
        };
        let Some(payload) = payload else {
            result.push(call.clone());
            continue;
        };
        let object = parse_json(payload).map_err(ChatError::Arguments)?;
        object.object()?;
        let changed = replace(function.object()?.clone(), b"arguments", object);
        result.push(match wrapped {
            Some(_) => replace(call.object()?.clone(), b"function", changed),
            None => changed,
        });
        *parsed += 1;
    }
    Ok(replace(
        message.object()?.clone(),
        b"tool_calls",
        Json::Array(result),
    ))
}

fn safe_name(name: &[u8], function: bool) -> Result<(), ChatError> {
    require(
        !name.is_empty() && name.len() <= if function { 64 } else { 1024 },
        "invalid tool or parameter name length",
    )?;
    require(
        !name
            .iter()
            .any(|c| matches!(c, b'<' | b'>' | b'\r' | b'\n' | 0)),
        "unsafe tool or parameter name",
    )?;
    if function {
        for &c in name {
            require(
                c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.',
                "unsupported tool function name",
            )?;
        }
    }
    Ok(())
}

/// Python `float.__repr__` of the binary64 nearest to `lexeme`.
fn float_json(lexeme: &str) -> Result<Vec<u8>, ChatError> {
    let text = lexeme.as_bytes();
    let value = match from_chars_f64(lexeme) {
        FromChars::Value(value) => value,
        FromChars::Invalid => return Err(ChatError::Render("unsupported JSON number in chat")),
        FromChars::OutOfRange => {
            // A decimal too small for binary64 rounds to signed zero. Overflow is rejected, rather than emitting a non-JSON Infinity into tool schemas.
            let negative = text.first() == Some(&b'-');
            let mut cursor = usize::from(negative);
            let mut before_point: i64 = 0;
            let mut first_nonzero: i64 = -1;
            let mut digits_seen: i64 = 0;
            let mut fraction = false;
            while cursor < text.len() && text[cursor] != b'e' && text[cursor] != b'E' {
                if text[cursor] == b'.' {
                    fraction = true;
                    cursor += 1;
                    continue;
                }
                if !fraction {
                    before_point += 1;
                }
                if first_nonzero < 0 && text[cursor] != b'0' {
                    first_nonzero = digits_seen;
                }
                digits_seen += 1;
                cursor += 1;
            }
            let mut exponent: i64 = 0;
            let mut negative_exponent = false;
            if cursor < text.len() {
                cursor += 1;
                if let Some(&sign @ (b'-' | b'+')) = text.get(cursor) {
                    negative_exponent = sign == b'-';
                    cursor += 1;
                }
                for &digit in &text[cursor..] {
                    exponent = (exponent * 10 + i64::from(digit) - i64::from(b'0')).min(1_000_000);
                }
            }
            exponent = (if negative_exponent {
                -exponent
            } else {
                exponent
            }) + before_point
                - first_nonzero
                - 1;
            if first_nonzero < 0 || exponent <= -324 {
                if negative { -0.0 } else { 0.0 }
            } else {
                return Err(ChatError::Render("nonfinite JSON number in chat"));
            }
        }
    };
    let negative = value.is_sign_negative();
    if value == 0.0 {
        return Ok(if negative {
            b"-0.0".to_vec()
        } else {
            b"0.0".to_vec()
        });
    }
    // `std::to_chars` into a 128-byte buffer cannot fail for a double, so the C++ "cannot format chat number" refusal has no Rust counterpart.
    let scientific = to_chars_scientific(value.abs());
    let Some(separator) = scientific.iter().position(|&c| c == b'e') else {
        return Err(ChatError::Render("missing scientific exponent"));
    };
    let mantissa = &scientific[..separator];
    let mut at = separator + 1;
    // `to_chars` always writes the exponent sign.
    let exponent_negative = scientific.get(at) == Some(&b'-');
    at += 1;
    let mut exponent: i32 = 0;
    for &digit in scientific.get(at..).unwrap_or_default() {
        exponent = exponent * 10 + i32::from(digit) - i32::from(b'0');
    }
    if exponent_negative {
        exponent = -exponent;
    }
    let mut output = if negative { b"-".to_vec() } else { Vec::new() };
    if !(-4..16).contains(&exponent) {
        output.extend_from_slice(mantissa);
        output.extend_from_slice(if exponent < 0 { b"e-" } else { b"e+" });
        let magnitude = exponent.unsigned_abs();
        if magnitude < 10 {
            output.push(b'0');
        }
        output.extend_from_slice(magnitude.to_string().as_bytes());
    } else {
        let digits: Vec<u8> = mantissa.iter().copied().filter(|&c| c != b'.').collect();
        let position = exponent + 1;
        if position <= 0 {
            output.extend_from_slice(b"0.");
            output.extend(std::iter::repeat_n(b'0', position.unsigned_abs() as usize));
            output.extend_from_slice(&digits);
        } else if position as usize >= digits.len() {
            output.extend_from_slice(&digits);
            output.extend(std::iter::repeat_n(b'0', position as usize - digits.len()));
            output.extend_from_slice(b".0");
        } else {
            let (whole, fraction) = digits.split_at(position as usize);
            output.extend_from_slice(whole);
            output.push(b'.');
            output.extend_from_slice(fraction);
        }
    }
    Ok(output)
}

fn literals(source: &[u8]) -> Result<Vec<Vec<u8>>, ChatError> {
    require_extract(
        source.len() <= 64 * 1024,
        "template source exceeds profile limit",
    )?;
    let mut result = Vec::new();
    let mut i = 0;
    while i < source.len() {
        if source[i..].starts_with(b"{#") {
            let Some(end) = find(&source[i + 2..], b"#}") else {
                return Err(ChatError::Extract("unterminated template comment"));
            };
            i += 2 + end + 2;
            continue;
        }
        let quote = source[i];
        i += 1;
        if quote != b'\'' && quote != b'"' {
            continue;
        }
        let mut value = Vec::new();
        let mut closed = false;
        while i < source.len() {
            let c = source[i];
            i += 1;
            if c == quote {
                closed = true;
                break;
            }
            if c != b'\\' {
                value.push(c);
                continue;
            }
            if i == source.len() {
                return Err(ChatError::Extract("truncated template literal"));
            }
            let escaped = source[i];
            i += 1;
            value.push(match escaped {
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                b'\\' => b'\\',
                b'\'' => b'\'',
                b'"' => b'"',
                _ => return Err(ChatError::Extract("unsupported template literal escape")),
            });
        }
        require_extract(closed, "unterminated template literal")?;
        result.push(value);
    }
    Ok(result)
}

/// Literal DATA read from the pinned model template. `Debug` prints only lengths: the strings are model assets.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct ChatStrings {
    pub reasoning_xhigh: Vec<u8>,
    pub reasoning_low: Vec<u8>,
    pub tools_header: Vec<u8>,
    pub tool_instructions: Vec<u8>,
}

impl fmt::Debug for ChatStrings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatStrings")
            .field("reasoning_xhigh", &Redacted(self.reasoning_xhigh.len()))
            .field("reasoning_low", &Redacted(self.reasoning_low.len()))
            .field("tools_header", &Redacted(self.tools_header.len()))
            .field("tool_instructions", &Redacted(self.tool_instructions.len()))
            .finish()
    }
}

/// Prints `<N bytes>` in place of a byte string.
struct Redacted(usize);

impl fmt::Debug for Redacted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "<{} bytes>", self.0)
    }
}

/// Selects the four renderer literals from the pinned template source.
pub fn extract_flash_next_literals(pinned_template: &[u8]) -> Result<ChatStrings, ChatError> {
    let values = literals(pinned_template)?;
    let select = |prefix: &[u8]| -> Result<Vec<u8>, ChatError> {
        let mut result = Vec::new();
        let mut matches = 0usize;
        for value in &values {
            if value.starts_with(prefix) {
                result.clone_from(value);
                matches += 1;
            }
        }
        require_extract(matches == 1, "pinned template literal missing or ambiguous")?;
        Ok(result)
    };
    Ok(ChatStrings {
        reasoning_xhigh: select(b"Reasoning effort is set to xhigh.")?,
        reasoning_low: select(b"Reasoning effort is set to low.")?,
        tools_header: select(b"# Tools\n\n")?,
        tool_instructions: select(b"\n\nIf you choose to call a function ONLY reply")?,
    })
}

/// Python `json.dumps(value)` as the pinned template renders it: `", "` and `": "` separators, insertion order, raw non-ASCII, `float.__repr__`.
pub fn template_json(value: &Json) -> Result<Vec<u8>, ChatError> {
    let mut output = Builder::new();
    template_visit(&mut output, value, 0)?;
    Ok(output.finish())
}

fn template_quoted(output: &mut Builder, text: &[u8]) -> Result<(), ChatError> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    unicode_scalars(text)?;
    output.add(b"\"")?;
    for &c in text {
        match c {
            b'"' => output.add(b"\\\"")?,
            b'\\' => output.add(b"\\\\")?,
            0x08 => output.add(b"\\b")?,
            0x0c => output.add(b"\\f")?,
            b'\n' => output.add(b"\\n")?,
            b'\r' => output.add(b"\\r")?,
            b'\t' => output.add(b"\\t")?,
            c if c < 32 => {
                let escaped = [
                    b'\\',
                    b'u',
                    b'0',
                    b'0',
                    HEX[usize::from(c >> 4)],
                    HEX[usize::from(c & 15)],
                ];
                output.add(&escaped)?;
            }
            c => output.add(&[c])?,
        }
    }
    output.add(b"\"")
}

fn template_visit(output: &mut Builder, item: &Json, depth: usize) -> Result<(), ChatError> {
    require(depth <= 64, "chat JSON exceeds nesting limit")?;
    match item {
        Json::Null => output.add(b"null"),
        Json::Bool(flag) => output.add(if *flag { b"true".as_slice() } else { b"false" }),
        Json::Number(number) => {
            let text = number.lexeme();
            if text.bytes().any(|c| matches!(c, b'.' | b'e' | b'E')) {
                output.add(&float_json(text)?)
            } else {
                require(text.len() <= 4300, "chat integer exceeds digit limit")?;
                output.add(if text == "-0" {
                    b"0".as_slice()
                } else {
                    text.as_bytes()
                })
            }
        }
        Json::String(text) => template_quoted(output, text),
        Json::Array(items) => {
            output.add(b"[")?;
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    output.add(b", ")?;
                }
                template_visit(output, item, depth + 1)?;
            }
            output.add(b"]")
        }
        Json::Object(fields) => {
            output.add(b"{")?;
            for (i, (name, item)) in fields.iter().enumerate() {
                if i > 0 {
                    output.add(b", ")?;
                }
                template_quoted(output, name)?;
                output.add(b": ")?;
                template_visit(output, item, depth + 1)?;
            }
            output.add(b"}")
        }
    }
}

/// Python `str.strip()`: drops leading and trailing Unicode whitespace.
pub fn chat_trim(value: &[u8]) -> Result<Vec<u8>, UnicodeError> {
    let points = unicode_scalars(value)?;
    let mut start = 0;
    let mut end = points.len();
    while start < end && whitespace(points[start]) {
        start += 1;
    }
    while end > start && whitespace(points[end - 1]) {
        end -= 1;
    }
    let mut output = Vec::new();
    for &point in &points[start..end] {
        push_scalar_utf8(&mut output, point)?;
    }
    Ok(output)
}

/// A rendered prompt. `Debug` prints only sizes and counters: the prompt and messages are conversation text.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct RenderedChat {
    pub prompt: Vec<u8>,
    pub normalized_messages: Json,
    pub parsed_tool_arguments: usize,
    pub hoisted_system_messages: usize,
    /// Renderer-owned byte ranges where client spellings must not select added tokens.
    pub literal_ranges: Vec<std::ops::Range<usize>>,
}

impl fmt::Debug for RenderedChat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RenderedChat")
            .field("prompt", &Redacted(self.prompt.len()))
            .field("normalized_messages", &self.normalized_messages)
            .field("parsed_tool_arguments", &self.parsed_tool_arguments)
            .field("hoisted_system_messages", &self.hoisted_system_messages)
            .finish()
    }
}

/// `ChatRenderer`: renders chat requests with the pinned template's literals.
#[derive(Clone, PartialEq, Eq)]
pub struct ChatRenderer {
    strings: ChatStrings,
}

impl fmt::Debug for ChatRenderer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatRenderer")
            .field("strings", &self.strings)
            .finish()
    }
}

impl ChatRenderer {
    pub fn new(strings: ChatStrings) -> Result<Self, ChatError> {
        require(
            !strings.reasoning_xhigh.is_empty()
                && !strings.reasoning_low.is_empty()
                && !strings.tools_header.is_empty()
                && !strings.tool_instructions.is_empty(),
            "chat profile lacks model literals",
        )?;
        Ok(Self { strings })
    }

    pub fn render(&self, request: &Json) -> Result<RenderedChat, ChatError> {
        for (name, _) in request.object()? {
            require(
                matches!(
                    name.as_slice(),
                    b"messages"
                        | b"tools"
                        | b"enable_thinking"
                        | b"preserve_thinking"
                        | b"add_generation_prompt"
                        | b"reasoning_effort"
                        | b"late_system_policy"
                        | b"parse_tool_arguments"
                ),
                "unknown chat request field",
            )?;
        }
        let Some(raw_messages) = request.find("messages")? else {
            return Err(ChatError::Render("no messages provided"));
        };
        let mut messages = raw_messages.array()?.clone();
        require(
            !messages.is_empty() && messages.len() <= 4096,
            "chat message count is empty or exceeds limit",
        )?;
        let mut result = RenderedChat::default();
        let parse_arguments = option(request, "parse_tool_arguments", true)?;
        for message in &mut messages {
            let assistant = role(message)? == b"assistant";
            message_content(message)?;
            if parse_arguments && assistant {
                *message = normalize_arguments(message, &mut result.parsed_tool_arguments)?;
            }
        }
        let mut late_policy: &[u8] = b"reject";
        if let Some(value) = request.find("late_system_policy")? {
            late_policy = value.string()?;
        }
        require(
            late_policy == b"reject" || late_policy == b"hoist",
            "unsupported late system message policy",
        )?;
        if late_policy == b"hoist" {
            let mut leading = Vec::new();
            let mut rest = Vec::new();
            let mut seen_non_system = false;
            for message in messages {
                if matches!(role(&message)?, b"system" | b"developer") {
                    if seen_non_system {
                        result.hoisted_system_messages += 1;
                    }
                    leading.push(message);
                } else {
                    seen_non_system = true;
                    rest.push(message);
                }
            }
            leading.extend(rest);
            messages = leading;
        }
        let thinking = option(request, "enable_thinking", true)?;
        let preserve = option(request, "preserve_thinking", true)?;
        let mut instructions: &[u8] = b"";
        if thinking {
            let mut effort: &[u8] = b"xhigh";
            if let Some(value) = request.find("reasoning_effort")? {
                effort = value.string()?;
            }
            if effort == b"high" {
                effort = b"xhigh";
            }
            require(
                effort == b"xhigh" || effort == b"medium" || effort == b"low",
                "unexpected reasoning effort",
            )?;
            if effort == b"xhigh" {
                instructions = &self.strings.reasoning_xhigh;
            } else if effort == b"low" {
                instructions = &self.strings.reasoning_low;
            }
        }
        let mut leading_count = 0;
        let mut system = Builder::new();
        let mut has_system = false;
        while leading_count < messages.len()
            && matches!(role(&messages[leading_count])?, b"system" | b"developer")
        {
            let text = message_content(&messages[leading_count])?;
            leading_count += 1;
            if !text.is_empty() {
                if has_system {
                    system.add(b"\n")?;
                }
                system.add(&text)?;
                has_system = true;
            }
        }
        let merged_system = system.finish();
        let mut tools = None;
        if let Some(value) = request.find("tools")?.filter(|value| !value.is_null()) {
            tools = Some(value.array()?);
        }
        if let Some(tools) = tools {
            require(tools.len() <= 512, "too many tools")?;
        }
        let mut output = Builder::new();
        match tools {
            Some(tools) if !tools.is_empty() => {
                output.add(b"<|im_start|>system\n")?;
                if !instructions.is_empty() {
                    output.add(instructions)?;
                    output.add(b"\n\n")?;
                }
                output.add(&self.strings.tools_header)?;
                for tool in tools {
                    tool.object()?;
                    output.add(b"\n")?;
                    output.literal(&template_json(tool)?)?;
                }
                output.add(b"\n</tools>")?;
                output.add(&self.strings.tool_instructions)?;
                if !merged_system.is_empty() {
                    output.add(b"\n\n")?;
                    output.literal(&merged_system)?;
                }
                output.add(b"<|im_end|>\n")?;
            }
            _ if !merged_system.is_empty() || !instructions.is_empty() => {
                output.add(b"<|im_start|>system\n")?;
                if !instructions.is_empty() {
                    output.add(instructions)?;
                    if !merged_system.is_empty() {
                        output.add(b"\n\n")?;
                    }
                }
                output.literal(&merged_system)?;
                output.add(b"<|im_end|>\n")?;
            }
            _ => {}
        }
        let mut last_query = messages.len() - 1;
        for i in (0..messages.len()).rev() {
            if role(&messages[i])? != b"user" {
                continue;
            }
            last_query = i;
            break;
        }
        for i in leading_count..messages.len() {
            let message = &messages[i];
            let who = role(message)?;
            let text = message_content(message)?;
            require(
                who != b"system" && who != b"developer",
                "system message must be at the beginning or explicitly hoisted",
            )?;
            if who == b"user" {
                output.add(b"<|im_start|>user\n")?;
                output.literal(&text)?;
                output.add(b"<|im_end|>\n")?;
            } else if who == b"assistant" {
                output.add(b"<|im_start|>assistant\n")?;
                if preserve || i > last_query {
                    let mut reasoning = Vec::new();
                    if let Some(value) = message
                        .find("reasoning_content")?
                        .filter(|value| !value.is_null())
                    {
                        reasoning = chat_trim(value.string()?)?;
                    }
                    output.add(b"<think>\n")?;
                    output.literal(&reasoning)?;
                    output.add(b"\n</think>\n\n")?;
                }
                output.literal(&text)?;
                if let Some(value) = message.find("tool_calls")?.filter(|value| !value.is_null()) {
                    let calls = value.array()?;
                    require(calls.len() <= 128, "too many tool calls")?;
                    for (call_index, call) in calls.iter().enumerate() {
                        let function = call.find("function")?.unwrap_or(call);
                        let name = function.find("name")?.filter(|name| !name.is_null());
                        let Some(name) = name else {
                            return Err(ChatError::Render("tool call lacks function name"));
                        };
                        safe_name(name.string()?, true)?;
                        output.add(if call_index != 0 {
                            b"\n<tool_call>\n<function=".as_slice()
                        } else if !text.is_empty() {
                            b"\n\n<tool_call>\n<function="
                        } else {
                            b"<tool_call>\n<function="
                        })?;
                        output.add(name.string()?)?;
                        output.add(b">\n")?;
                        if let Some(args) =
                            function.find("arguments")?.filter(|args| !args.is_null())
                        {
                            if args.kind() == Kind::Object {
                                require(args.object()?.len() <= 256, "too many tool parameters")?;
                                for (key, argument) in args.object()? {
                                    safe_name(key, false)?;
                                    output.add(b"<parameter=")?;
                                    output.add(key)?;
                                    output.add(b">\n")?;
                                    if argument.kind() == Kind::String {
                                        output.parameter(argument.string()?)?;
                                    } else {
                                        output.parameter(&template_json(argument)?)?;
                                    }
                                    output.add(b"\n</parameter>\n")?;
                                }
                            } else if args.kind() == Kind::String {
                                require(
                                    chat_trim(args.string()?)?.is_empty(),
                                    "parse nonempty JSON-string tool arguments before raw template rendering",
                                )?;
                            } else {
                                return Err(ChatError::Render(
                                    "tool arguments must be an object or empty string",
                                ));
                            }
                        }
                        output.add(b"</function>\n</tool_call>")?;
                    }
                }
                output.add(b"<|im_end|>\n")?;
            } else if who == b"tool" {
                // Preserve the pinned formatter's exact treatment of history fragments. The pinned template checks loop.previtem, not loop.first; a leading tool fragment therefore has no user opener. Tool-call ID/sequence admission is a separate gate.
                if i != 0 && role(&messages[i - 1])? != b"tool" {
                    output.add(b"<|im_start|>user")?;
                }
                output.add(b"\n<tool_response>\n")?;
                output.literal(&text)?;
                output.add(b"\n</tool_response>")?;
                if i + 1 == messages.len() || role(&messages[i + 1])? != b"tool" {
                    output.add(b"<|im_end|>\n")?;
                }
            }
        }
        if option(request, "add_generation_prompt", true)? {
            output.add(b"<|im_start|>assistant\n")?;
            output.add(if thinking {
                b"<think>\n".as_slice()
            } else {
                b"<think>\n\n</think>\n\n"
            })?;
        }
        result.literal_ranges = std::mem::take(&mut output.literal_ranges);
        result.prompt = output.finish();
        result.normalized_messages = Json::Array(messages);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Json {
        parse_json(text.as_bytes()).unwrap()
    }

    fn renderer() -> ChatRenderer {
        ChatRenderer::new(ChatStrings {
            reasoning_xhigh: b"XHIGH".to_vec(),
            reasoning_low: b"LOW".to_vec(),
            tools_header: b"# Tools".to_vec(),
            tool_instructions: b"INSTRUCTIONS".to_vec(),
        })
        .unwrap()
    }

    fn deep(levels: usize) -> Json {
        (0..levels).fold(Json::Null, |inner, _| Json::Array(vec![inner]))
    }

    #[test]
    fn debug_redacts_prompts_messages_and_template_literals() {
        let request = parse(r#"{"messages":[{"role":"user","content":"secret question"}]}"#);
        let rendered = renderer().render(&request).unwrap();
        assert_eq!(
            format!("{rendered:?}"),
            format!(
                "RenderedChat {{ prompt: <{} bytes>, normalized_messages: Json::Array(1 items), \
                 parsed_tool_arguments: 0, hoisted_system_messages: 0 }}",
                rendered.prompt.len()
            )
        );
        assert_eq!(
            format!("{:?}", renderer()),
            "ChatRenderer { strings: ChatStrings { reasoning_xhigh: <5 bytes>, \
             reasoning_low: <3 bytes>, tools_header: <7 bytes>, tool_instructions: <12 bytes> } }"
        );
        assert!(!format!("{rendered:#?}").contains("secret"));
        assert!(!format!("{:#?}", renderer()).contains("XHIGH"));
    }

    #[test]
    fn errors_display_the_exact_cpp_messages_and_stay_transparent() {
        use std::error::Error;
        let cases = [
            (
                ChatError::Extract("unterminated template literal"),
                "unterminated template literal",
                false,
            ),
            (
                ChatError::Render("no messages provided"),
                "no messages provided",
                false,
            ),
            (
                ChatError::Access(JsonError::Access("expected JSON object")),
                "expected JSON object",
                true,
            ),
            (
                ChatError::Arguments(JsonError::Parse("trailing JSON data")),
                "trailing JSON data",
                true,
            ),
            (
                ChatError::Text(UnicodeError::Decode("invalid UTF-8 input")),
                "invalid UTF-8 input",
                true,
            ),
        ];
        for (error, message, has_source) in cases {
            assert_eq!(error.to_string(), message);
            assert_eq!(error.source().is_some(), has_source);
        }
    }

    #[test]
    fn template_json_refuses_constructed_values_parsing_cannot_produce() {
        // parse_json bounds depth at 64, so only constructed values go deeper.
        assert!(template_json(&deep(64)).is_ok());
        assert_eq!(
            template_json(&deep(65)),
            Err(ChatError::Render("chat JSON exceeds nesting limit"))
        );
        let invalid = Json::String(b"\xc3".to_vec());
        assert_eq!(
            template_json(&invalid),
            Err(ChatError::Text(UnicodeError::Decode("invalid UTF-8 input")))
        );
        let key = Json::Object(vec![(b"\xed\xa0\x80".to_vec(), Json::Null)]);
        assert_eq!(
            template_json(&key).unwrap_err().to_string(),
            "non-scalar or overlong UTF-8"
        );
        // Unlike dump, template_json does not police duplicate keys.
        let duplicate = Json::Object(vec![
            (b"a".to_vec(), Json::Null),
            (b"a".to_vec(), Json::Bool(true)),
        ]);
        assert_eq!(
            template_json(&duplicate).unwrap(),
            br#"{"a": null, "a": true}"#
        );
    }

    #[test]
    fn template_json_bounds_integer_digits_and_output_bytes() {
        let digits = |count: usize| parse(&"9".repeat(count));
        assert_eq!(template_json(&digits(4300)).unwrap().len(), 4300);
        assert_eq!(
            template_json(&digits(4301)),
            Err(ChatError::Render("chat integer exceeds digit limit"))
        );
        // The sign counts towards the 4300 bytes, as the C++ size() does.
        let negative = parse(&format!("-{}", "9".repeat(4300)));
        assert_eq!(
            template_json(&negative),
            Err(ChatError::Render("chat integer exceeds digit limit"))
        );
        // Each control byte becomes six output bytes.
        let at_limit = Json::String(vec![b'\x01'; OUTPUT_LIMIT / 6]);
        assert_eq!(template_json(&at_limit).unwrap().len(), OUTPUT_LIMIT);
        let over = Json::String(vec![b'\x01'; OUTPUT_LIMIT / 6 + 1]);
        assert_eq!(
            template_json(&over),
            Err(ChatError::Render("rendered chat exceeds byte limit"))
        );
    }

    #[test]
    fn float_json_reproduces_the_out_of_range_fallback() {
        for (lexeme, expected) in [
            ("0.0001e-320", "0.0"),
            ("-0.000000e-400", "-0.0"),
            ("0e99999999", "0.0"),
            ("100.0", "100.0"),
            ("1e21", "1e+21"),
            ("1.5e-7", "1.5e-07"),
            ("123.456e-2", "1.23456"),
            ("9999999999999999.0", "1e+16"),
            ("1e-99999999999999999999", "0.0"),
        ] {
            assert_eq!(float_json(lexeme).unwrap(), expected.as_bytes(), "{lexeme}");
        }
        for lexeme in ["1e309", "-1e400", "1e99999999999999999999", "123456e305"] {
            assert_eq!(
                float_json(lexeme),
                Err(ChatError::Render("nonfinite JSON number in chat"))
            );
        }
        // Spellings no JSON lexeme can take still map to the C++ refusal.
        for lexeme in ["inf", "nan", "1e", "0x1p3"] {
            assert_eq!(
                float_json(lexeme),
                Err(ChatError::Render("unsupported JSON number in chat"))
            );
        }
    }

    #[test]
    fn literals_skip_comments_and_refuse_unterminated_forms() {
        assert_eq!(
            literals(b"{# 'x' #}'a'\"b\"").unwrap(),
            [b"a".to_vec(), b"b".to_vec()]
        );
        assert_eq!(literals(b"'a\\'b'").unwrap(), [b"a'b".to_vec()]);
        let refusals: [(&[u8], &str); 5] = [
            (b"{#", "unterminated template comment"),
            (b"{#}", "unterminated template comment"),
            (b"'a\\", "truncated template literal"),
            (b"'a", "unterminated template literal"),
            (b"'\\x'", "unsupported template literal escape"),
        ];
        for (source, message) in refusals {
            assert_eq!(literals(source), Err(ChatError::Extract(message)));
        }
        let oversized = vec![b' '; 64 * 1024 + 1];
        assert_eq!(
            literals(&oversized),
            Err(ChatError::Extract("template source exceeds profile limit"))
        );
    }

    #[test]
    fn render_checks_fields_in_the_cpp_order() {
        let render = |text: &str| {
            renderer()
                .render(&parse(text))
                .map(|rendered| rendered.prompt)
        };
        // Unknown fields are refused before messages are even looked up.
        assert_eq!(
            render(r#"{"extra":1}"#),
            Err(ChatError::Render("unknown chat request field"))
        );
        assert_eq!(
            render(r#"[]"#),
            Err(ChatError::Access(JsonError::Access("expected JSON object")))
        );
        // Options are typed: null is not a boolean.
        assert_eq!(
            render(r#"{"messages":[{"role":"user","content":"x"}],"enable_thinking":null}"#),
            Err(ChatError::Access(JsonError::Access(
                "expected JSON boolean"
            )))
        );
        // reasoning_effort is not read at all when thinking is disabled.
        let disabled = r#"{"messages":[{"role":"user","content":"x"}],"enable_thinking":false,"reasoning_effort":5}"#;
        assert!(render(disabled).is_ok());
        // add_generation_prompt is read last, after every message rendered.
        assert_eq!(
            render(
                r#"{"messages":[{"role":"user","content":"x"},{"role":"system","content":"y"}],"add_generation_prompt":5}"#
            ),
            Err(ChatError::Render(
                "system message must be at the beginning or explicitly hoisted"
            ))
        );
    }

    #[test]
    fn constructed_parameter_keys_reach_the_prompt_unvalidated() {
        // As in C++, only safe_name's byte rules apply to parameter keys.
        let arguments = Json::Object(vec![(b"k\xff".to_vec(), Json::Bool(true))]);
        let call = Json::new_object([(
            "function",
            Json::new_object([("name", Json::from("f")), ("arguments", arguments)]),
        )]);
        let message = Json::new_object([
            ("role", Json::from("assistant")),
            ("content", Json::from("")),
            ("tool_calls", Json::Array(vec![call])),
        ]);
        let request = Json::new_object([("messages", Json::Array(vec![message]))]);
        let prompt = renderer().render(&request).unwrap().prompt;
        let needle = b"<parameter=k\xff>\ntrue\n";
        assert!(prompt.windows(needle.len()).any(|window| window == needle));
    }

    #[test]
    fn safe_names_follow_the_template_alphabet() {
        assert!(safe_name(b"edit.file-2_x", true).is_ok());
        assert!(safe_name("日本".as_bytes(), false).is_ok());
        for (name, function, message) in [
            (
                b"".as_slice(),
                true,
                "invalid tool or parameter name length",
            ),
            (&[b'a'; 65], true, "invalid tool or parameter name length"),
            (
                &[b'a'; 1025],
                false,
                "invalid tool or parameter name length",
            ),
            (b"a\0b", false, "unsafe tool or parameter name"),
            (b"a b", true, "unsupported tool function name"),
        ] {
            assert_eq!(safe_name(name, function), Err(ChatError::Render(message)));
        }
    }
}
