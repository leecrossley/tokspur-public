//! Owned JSON, ported one to one from `native/src/json.cpp`.
//!
//! Strings and object keys hold bytes, as the C++ `std::string` does: parsed values are always valid UTF-8, while constructed values are validated only when dumped. Numbers keep their exact lexeme, objects keep insertion order and refuse duplicate keys, and `dump` escapes exactly as the C++ does. These properties are part of the worker wire and HTTP contract.

#![forbid(unsafe_code)]

mod numbers;
mod utf8;

use std::collections::BTreeSet;
use std::fmt;

pub use numbers::{from_chars_f64, from_chars_i64, from_chars_u32, to_chars_shortest};
pub use utf8::{UnicodeError, WORK_LIMIT, push_scalar_utf8, scalar_utf8, unicode_scalars};

/// Default `dump` bound (`Json::dump(size_t byte_limit = 64 MiB)`).
pub const DEFAULT_DUMP_LIMIT: usize = 64 * 1024 * 1024;

/// Nesting bound `dump` enforces regardless of how the value was built.
const DUMP_DEPTH_LIMIT: usize = 64;

/// The C++ `FormatError` raised by JSON. `Display` is the exact C++ message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonError {
    /// `parse_json` refused the document.
    Parse(&'static str),
    /// A parsed or dumped string is not valid, bounded UTF-8.
    Text(UnicodeError),
    /// A typed accessor found another kind, or an integer out of range.
    Access(&'static str),
    /// `Json::dump` refused the value.
    Dump(&'static str),
}

impl fmt::Display for JsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(reason) | Self::Access(reason) | Self::Dump(reason) => {
                formatter.write_str(reason)
            }
            Self::Text(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for JsonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Text(error) => Some(error),
            Self::Parse(_) | Self::Access(_) | Self::Dump(_) => None,
        }
    }
}

impl From<UnicodeError> for JsonError {
    fn from(error: UnicodeError) -> Self {
        Self::Text(error)
    }
}

/// `JsonLimits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonLimits {
    pub bytes: usize,
    pub nodes: usize,
    pub depth: usize,
}

impl JsonLimits {
    pub const fn new(bytes: usize, nodes: usize, depth: usize) -> Self {
        Self {
            bytes,
            nodes,
            depth,
        }
    }
}

impl Default for JsonLimits {
    fn default() -> Self {
        Self::new(8 * 1024 * 1024, 1_000_000, 64)
    }
}

/// `Json::Kind`, in the C++ variant order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Null,
    Boolean,
    Number,
    String,
    Array,
    Object,
}

/// A number's exact source lexeme. Only the parser and [`Json::integer`] create one, as only `JsonParser` and `Json::integer` can in C++.
#[derive(Clone, PartialEq, Eq)]
pub struct Number {
    lexeme: String,
}

impl Number {
    pub fn lexeme(&self) -> &str {
        &self.lexeme
    }
}

pub type Array = Vec<Json>;
pub type Object = Vec<(Vec<u8>, Json)>;

/// One JSON value. `Debug` is hand-written and redacts every payload: values carry prompts, tool arguments and token IDs.
#[derive(Clone, PartialEq, Eq, Default)]
pub enum Json {
    #[default]
    Null,
    Bool(bool),
    Number(Number),
    String(Vec<u8>),
    Array(Array),
    Object(Object),
}

impl fmt::Debug for Json {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => formatter.write_str("Json::Null"),
            Self::Bool(_) => formatter.write_str("Json::Bool(..)"),
            Self::Number(_) => formatter.write_str("Json::Number(..)"),
            Self::String(bytes) => write!(formatter, "Json::String({} bytes)", bytes.len()),
            Self::Array(items) => write!(formatter, "Json::Array({} items)", items.len()),
            Self::Object(fields) => write!(formatter, "Json::Object({} fields)", fields.len()),
        }
    }
}

impl From<bool> for Json {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<&str> for Json {
    fn from(value: &str) -> Self {
        Self::String(value.as_bytes().to_vec())
    }
}

impl From<String> for Json {
    fn from(value: String) -> Self {
        Self::String(value.into_bytes())
    }
}

impl From<Vec<u8>> for Json {
    fn from(value: Vec<u8>) -> Self {
        Self::String(value)
    }
}

impl From<&[u8]> for Json {
    fn from(value: &[u8]) -> Self {
        Self::String(value.to_vec())
    }
}

impl From<Array> for Json {
    fn from(value: Array) -> Self {
        Self::Array(value)
    }
}

impl From<Object> for Json {
    fn from(value: Object) -> Self {
        Self::Object(value)
    }
}

impl Json {
    /// `Json::integer`: the canonical decimal lexeme of `value`.
    pub fn integer(value: u64) -> Self {
        Self::Number(Number {
            lexeme: value.to_string(),
        })
    }

    /// Builds an object from ordered fields, like `Json(Json::Object{...})`.
    pub fn new_object<K, I>(fields: I) -> Self
    where
        K: Into<Vec<u8>>,
        I: IntoIterator<Item = (K, Json)>,
    {
        Self::Object(
            fields
                .into_iter()
                .map(|(name, value)| (name.into(), value))
                .collect(),
        )
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub fn kind(&self) -> Kind {
        match self {
            Self::Null => Kind::Null,
            Self::Bool(_) => Kind::Boolean,
            Self::Number(_) => Kind::Number,
            Self::String(_) => Kind::String,
            Self::Array(_) => Kind::Array,
            Self::Object(_) => Kind::Object,
        }
    }

    pub fn number_lexeme(&self) -> Result<&str, JsonError> {
        match self {
            Self::Number(number) => Ok(&number.lexeme),
            _ => Err(JsonError::Access("expected JSON number")),
        }
    }

    pub fn boolean(&self) -> Result<bool, JsonError> {
        match self {
            Self::Bool(value) => Ok(*value),
            _ => Err(JsonError::Access("expected JSON boolean")),
        }
    }

    /// The string's bytes (`std::string_view` in C++).
    pub fn string(&self) -> Result<&[u8], JsonError> {
        match self {
            Self::String(bytes) => Ok(bytes),
            _ => Err(JsonError::Access("expected JSON string")),
        }
    }

    /// `std::from_chars<uint64_t>` over the whole lexeme: digits only.
    pub fn unsigned_integer(&self) -> Result<u64, JsonError> {
        let Self::Number(number) = self else {
            return Err(JsonError::Access("expected JSON unsigned integer"));
        };
        let range = JsonError::Access("JSON unsigned integer out of range or nonintegral");
        let text = number.lexeme.as_bytes();
        if text.is_empty() || !text.iter().all(u8::is_ascii_digit) {
            return Err(range);
        }
        text.iter()
            .try_fold(0u64, |value, digit| {
                value.checked_mul(10)?.checked_add(u64::from(digit - b'0'))
            })
            .ok_or(range)
    }

    pub fn array(&self) -> Result<&Array, JsonError> {
        match self {
            Self::Array(items) => Ok(items),
            _ => Err(JsonError::Access("expected JSON array")),
        }
    }

    pub fn object(&self) -> Result<&Object, JsonError> {
        match self {
            Self::Object(fields) => Ok(fields),
            _ => Err(JsonError::Access("expected JSON object")),
        }
    }

    /// The first field named `key`; refuses non-objects.
    pub fn find(&self, key: impl AsRef<[u8]>) -> Result<Option<&Json>, JsonError> {
        let key = key.as_ref();
        Ok(self
            .object()?
            .iter()
            .find(|(name, _)| name.as_slice() == key)
            .map(|(_, value)| value))
    }

    /// `dump()` with the default 64 MiB bound.
    pub fn dump(&self) -> Result<Vec<u8>, JsonError> {
        self.dump_limited(DEFAULT_DUMP_LIMIT)
    }

    /// `dump(byte_limit)`: compact output, refusing invalid UTF-8, duplicate keys, nesting beyond 64 and output beyond `byte_limit`.
    pub fn dump_limited(&self, byte_limit: usize) -> Result<Vec<u8>, JsonError> {
        let mut writer = Writer {
            output: Vec::new(),
            byte_limit,
        };
        writer.visit(self, 0)?;
        Ok(writer.output)
    }
}

struct Writer {
    output: Vec<u8>,
    byte_limit: usize,
}

impl Writer {
    fn append(&mut self, text: &[u8]) -> Result<(), JsonError> {
        if text.len() > self.byte_limit - self.output.len() {
            return Err(JsonError::Dump("JSON output byte limit exceeded"));
        }
        self.output.extend_from_slice(text);
        Ok(())
    }

    fn quoted(&mut self, text: &[u8]) -> Result<(), JsonError> {
        unicode_scalars(text)?;
        self.append(b"\"")?;
        for &c in text {
            if c == b'"' || c == b'\\' {
                self.append(&[b'\\', c])?;
            } else if c < 0x20 {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                let escaped = [
                    b'\\',
                    b'u',
                    b'0',
                    b'0',
                    HEX[usize::from(c >> 4)],
                    HEX[usize::from(c & 15)],
                ];
                self.append(&escaped)?;
            } else {
                self.append(&[c])?;
            }
        }
        self.append(b"\"")
    }

    fn visit(&mut self, value: &Json, depth: usize) -> Result<(), JsonError> {
        if depth > DUMP_DEPTH_LIMIT {
            return Err(JsonError::Dump("JSON output nesting limit exceeded"));
        }
        match value {
            Json::Null => self.append(b"null"),
            Json::Bool(flag) => self.append(if *flag { b"true" } else { b"false" }),
            Json::Number(number) => self.append(number.lexeme.as_bytes()),
            Json::String(text) => self.quoted(text),
            Json::Array(items) => {
                self.append(b"[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        self.append(b",")?;
                    }
                    self.visit(item, depth + 1)?;
                }
                self.append(b"]")
            }
            Json::Object(fields) => {
                self.append(b"{")?;
                let mut names = BTreeSet::new();
                for (i, (name, item)) in fields.iter().enumerate() {
                    if !names.insert(name.as_slice()) {
                        return Err(JsonError::Dump("duplicate JSON output key"));
                    }
                    if i > 0 {
                        self.append(b",")?;
                    }
                    self.quoted(name)?;
                    self.append(b":")?;
                    self.visit(item, depth + 1)?;
                }
                self.append(b"}")
            }
        }
    }
}

/// `parse_json` with explicit limits.
pub fn parse_json_limited(input: &[u8], limits: JsonLimits) -> Result<Json, JsonError> {
    if input.len() > limits.bytes {
        return Err(JsonError::Parse("JSON input byte limit exceeded"));
    }
    let mut parser = Parser {
        input,
        limits,
        position: 0,
        nodes: 0,
    };
    let result = parser.value(0)?;
    parser.whitespace();
    if parser.position != input.len() {
        return Err(JsonError::Parse("trailing JSON data"));
    }
    Ok(result)
}

/// `parse_json` with the default limits.
pub fn parse_json(input: &[u8]) -> Result<Json, JsonError> {
    parse_json_limited(input, JsonLimits::default())
}

struct Parser<'a> {
    input: &'a [u8],
    limits: JsonLimits,
    position: usize,
    nodes: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.input.get(self.position).copied()
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.position += 1;
        }
    }

    fn take(&mut self) -> Result<u8, JsonError> {
        let c = self.peek().ok_or(JsonError::Parse("truncated JSON"))?;
        self.position += 1;
        Ok(c)
    }

    fn consume(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn literal(&mut self, tail: &[u8]) -> Result<(), JsonError> {
        if !self.input[self.position..].starts_with(tail) {
            return Err(JsonError::Parse("invalid JSON literal"));
        }
        self.position += tail.len();
        Ok(())
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let mut value = 0;
        for _ in 0..4 {
            let digit = match self.take()? {
                c @ b'0'..=b'9' => c - b'0',
                c @ b'a'..=b'f' => c - b'a' + 10,
                c @ b'A'..=b'F' => c - b'A' + 10,
                _ => return Err(JsonError::Parse("invalid JSON Unicode escape")),
            };
            value = (value << 4) | u32::from(digit);
        }
        Ok(value)
    }

    fn text(&mut self) -> Result<Vec<u8>, JsonError> {
        let mut output = Vec::new();
        loop {
            let c = self.take()?;
            if c == b'"' {
                unicode_scalars(&output)?;
                return Ok(output);
            }
            if c < 0x20 {
                return Err(JsonError::Parse("unescaped JSON control character"));
            }
            if c != b'\\' {
                output.push(c);
                continue;
            }
            match self.take()? {
                b'"' => output.push(b'"'),
                b'\\' => output.push(b'\\'),
                b'/' => output.push(b'/'),
                b'b' => output.push(0x08),
                b'f' => output.push(0x0c),
                b'n' => output.push(b'\n'),
                b'r' => output.push(b'\r'),
                b't' => output.push(b'\t'),
                b'u' => {
                    let mut point = self.hex4()?;
                    if (0xd800..=0xdbff).contains(&point) {
                        if self.take()? != b'\\' || self.take()? != b'u' {
                            return Err(JsonError::Parse("missing JSON low surrogate"));
                        }
                        let low = self.hex4()?;
                        if !(0xdc00..=0xdfff).contains(&low) {
                            return Err(JsonError::Parse("invalid JSON low surrogate"));
                        }
                        point = 0x10000 + ((point - 0xd800) << 10) + low - 0xdc00;
                    } else if (0xdc00..=0xdfff).contains(&point) {
                        return Err(JsonError::Parse("unpaired JSON low surrogate"));
                    }
                    push_scalar_utf8(&mut output, point)?;
                }
                _ => return Err(JsonError::Parse("invalid JSON escape")),
            }
        }
    }

    fn digits(&mut self) -> Result<(), JsonError> {
        let before = self.position;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.position += 1;
        }
        if before == self.position {
            return Err(JsonError::Parse("missing JSON number digits"));
        }
        Ok(())
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.position;
        self.consume(b'-');
        if !self.consume(b'0') {
            if !matches!(self.peek(), Some(b'1'..=b'9')) {
                return Err(JsonError::Parse("invalid JSON number"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
        }
        if self.consume(b'.') {
            self.digits()?;
        }
        if self.consume(b'e') || self.consume(b'E') {
            if !self.consume(b'+') {
                self.consume(b'-');
            }
            self.digits()?;
        }
        // The grammar above admits only ASCII, so this is a byte-for-byte copy.
        let lexeme = self.input[start..self.position]
            .iter()
            .copied()
            .map(char::from)
            .collect();
        Ok(Json::Number(Number { lexeme }))
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.nodes += 1;
        if self.nodes > self.limits.nodes || depth > self.limits.depth {
            return Err(JsonError::Parse("JSON structural limit exceeded"));
        }
        self.whitespace();
        match self.take()? {
            b'n' => {
                self.literal(b"ull")?;
                Ok(Json::Null)
            }
            b't' => {
                self.literal(b"rue")?;
                Ok(Json::Bool(true))
            }
            b'f' => {
                self.literal(b"alse")?;
                Ok(Json::Bool(false))
            }
            b'"' => Ok(Json::String(self.text()?)),
            b'[' => {
                let mut result = Array::new();
                self.whitespace();
                if self.consume(b']') {
                    return Ok(Json::Array(result));
                }
                loop {
                    result.push(self.value(depth + 1)?);
                    self.whitespace();
                    if self.consume(b']') {
                        break;
                    }
                    if !self.consume(b',') {
                        return Err(JsonError::Parse("expected JSON array delimiter"));
                    }
                }
                Ok(Json::Array(result))
            }
            b'{' => {
                let mut result = Object::new();
                let mut names = BTreeSet::new();
                self.whitespace();
                if self.consume(b'}') {
                    return Ok(Json::Object(result));
                }
                loop {
                    self.whitespace();
                    if !self.consume(b'"') {
                        return Err(JsonError::Parse("expected JSON object key"));
                    }
                    let key = self.text()?;
                    if !names.insert(key.clone()) {
                        return Err(JsonError::Parse("duplicate JSON object key"));
                    }
                    self.whitespace();
                    if !self.consume(b':') {
                        return Err(JsonError::Parse("expected JSON object colon"));
                    }
                    let value = self.value(depth + 1)?;
                    result.push((key, value));
                    self.whitespace();
                    if self.consume(b'}') {
                        break;
                    }
                    if !self.consume(b',') {
                        return Err(JsonError::Parse("expected JSON object delimiter"));
                    }
                }
                Ok(Json::Object(result))
            }
            c if c == b'-' || c.is_ascii_digit() => {
                self.position -= 1;
                self.number()
            }
            _ => Err(JsonError::Parse("invalid JSON value")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_every_payload() {
        let value = Json::new_object([
            ("prompt", Json::from("secret prompt text")),
            (
                "ids",
                Json::Array(vec![Json::integer(151_644), Json::Bool(true)]),
            ),
        ]);
        let printed = format!("{value:?}");
        assert_eq!(printed, "Json::Object(2 fields)");
        let nested = format!("{:?}", value.find("ids").unwrap().unwrap());
        assert_eq!(nested, "Json::Array(2 items)");
        assert!(!format!("{:?}", Json::integer(151_644)).contains("151644"));
        assert!(!format!("{:?}", Json::from("secret")).contains("secret"));
    }

    #[test]
    fn kind_follows_the_cpp_variant_order() {
        let kinds: Vec<Kind> = [
            Json::Null,
            Json::Bool(false),
            Json::integer(0),
            Json::from(""),
            Json::Array(Vec::new()),
            Json::Object(Vec::new()),
        ]
        .iter()
        .map(Json::kind)
        .collect();
        assert_eq!(
            kinds,
            [
                Kind::Null,
                Kind::Boolean,
                Kind::Number,
                Kind::String,
                Kind::Array,
                Kind::Object
            ]
        );
    }

    #[test]
    fn errors_display_the_exact_cpp_messages() {
        assert_eq!(
            parse_json(b"[1,]").unwrap_err().to_string(),
            "invalid JSON value"
        );
        assert_eq!(
            parse_json(b"\"\xc0\x80\"").unwrap_err().to_string(),
            "invalid UTF-8 input"
        );
        assert_eq!(
            Json::Null.string().unwrap_err().to_string(),
            "expected JSON string"
        );
        assert_eq!(
            Json::from(b"\x80".as_slice())
                .dump()
                .unwrap_err()
                .to_string(),
            "invalid UTF-8 input"
        );
    }

    #[test]
    fn json_text_error_retains_immediate_unicode_source() {
        use std::error::Error;
        let err = parse_json(b"\"\xc0\x80\"").unwrap_err();
        let source = err
            .source()
            .expect("JsonError::Text must retain UnicodeError");
        assert_eq!(source.to_string(), "invalid UTF-8 input");
    }
}
