//! `NativeText`, ported from `native/src/text.mm`: the pinned tokenizer JSON, byte-level BPE, and the Foundation `NSRegularExpression` split.
//!
//! The C++ parses the tokenizer JSON with `NSJSONSerialization`; this port uses the owned parser. The SHA-256 pin is checked first, so only the one pinned file ever reaches parsing, and both parsers produce the same model data from it (vocabulary order does not matter: entries are indexed by ID).

use std::fmt;
use std::path::Path;

use tokspur_gguf::{GgufError, MappedFile, ModelFiles};
use tokspur_json::{
    Json, JsonError, JsonLimits, Kind, UnicodeError, from_chars_f64, parse_json_limited,
    unicode_scalars,
};
use tokspur_sys::foundation::{Regex, RegexError, Utf16String};
use tokspur_text::sha256::{Sha256, hex, sha256};
use tokspur_text::tokenizer::{
    AddedToken, BpeData, ByteBpe, TextProcessor, TokenizerError, byte_level_decode,
};
use tokspur_text::unicode::{NormalizeError, normalize_nfc9, push_scalar_utf8};

const TEXT_LIMIT: usize = 8 * 1024 * 1024;
const SUPPORTED_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The C++ `FormatError` raised by the macOS text adapter.
pub enum TextError {
    /// `require`: the asset or a split refused.
    Refused(&'static str),
    /// Input or scalar bytes were not valid UTF-8.
    Utf8(UnicodeError),
    /// Creating a Foundation UTF-16 string failed.
    Utf16String(RegexError),
    /// Compiling the declared split regex failed.
    SplitCompile(RegexError),
    /// Parsing the tokenizer JSON object failed.
    ModelParse(JsonError),
    /// NFC normalization refused the segment.
    Normalize(NormalizeError),
    /// The tokenizer refused data, text or IDs.
    Tokenizer(TokenizerError<Box<TextError>>),
    /// Reading the pinned file or GGUF shards failed.
    Gguf(GgufError),
    /// A JSON accessor refused (`parse_json` in `text_cli`).
    Json(JsonError),
}

impl fmt::Debug for TextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(reason) => formatter.debug_tuple("Refused").field(reason).finish(),
            Self::Utf8(error) => formatter.debug_tuple("Utf8").field(error).finish(),
            Self::Utf16String(error) => formatter.debug_tuple("Utf16String").field(error).finish(),
            Self::SplitCompile(error) => {
                formatter.debug_tuple("SplitCompile").field(error).finish()
            }
            Self::ModelParse(error) => formatter.debug_tuple("ModelParse").field(error).finish(),
            Self::Normalize(error) => formatter.debug_tuple("Normalize").field(error).finish(),
            Self::Tokenizer(_) => formatter.write_str("Tokenizer([redacted])"),
            Self::Gguf(error) => formatter.debug_tuple("Gguf").field(error).finish(),
            Self::Json(error) => formatter.debug_tuple("Json").field(error).finish(),
        }
    }
}

impl fmt::Display for TextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(reason) => formatter.write_str(reason),
            Self::Utf8(_) | Self::Utf16String(_) => {
                formatter.write_str("expected valid UTF-8 text")
            }
            Self::SplitCompile(_) => formatter.write_str("cannot compile declared split pattern"),
            Self::ModelParse(_) => formatter.write_str("invalid JSON object"),
            Self::Normalize(error) => error.fmt(formatter),
            Self::Tokenizer(error) => error.fmt(formatter),
            Self::Gguf(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Refused(_) => None,
            Self::Utf8(error) => Some(error),
            Self::Utf16String(error) | Self::SplitCompile(error) => Some(error),
            Self::ModelParse(error) | Self::Json(error) => Some(error),
            Self::Normalize(error) => Some(error),
            Self::Tokenizer(error) => Some(error),
            Self::Gguf(error) => Some(error),
        }
    }
}

impl From<NormalizeError> for TextError {
    fn from(error: NormalizeError) -> Self {
        Self::Normalize(error)
    }
}

impl From<GgufError> for TextError {
    fn from(error: GgufError) -> Self {
        Self::Gguf(error)
    }
}

impl From<JsonError> for TextError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

impl From<TokenizerError> for TextError {
    fn from(error: TokenizerError) -> Self {
        Self::Tokenizer(match error {
            TokenizerError::ByteLevelDecode(reason) => TokenizerError::ByteLevelDecode(reason),
            TokenizerError::New(reason) => TokenizerError::New(reason),
            TokenizerError::EncodePiece(reason) => TokenizerError::EncodePiece(reason),
            TokenizerError::Encode(reason) => TokenizerError::Encode(reason),
            TokenizerError::DecodeBytes(reason) => TokenizerError::DecodeBytes(reason),
            TokenizerError::Pieces(never) => match never {},
        })
    }
}

type Result<T> = std::result::Result<T, TextError>;

fn require(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(TextError::Refused(message))
    }
}

/// `ns()`: literal UTF-16 units (a leading U+FEFF stays text).
fn utf16(bytes: &[u8]) -> Result<Vec<u16>> {
    let points = unicode_scalars(bytes).map_err(TextError::Utf8)?;
    let mut units = Vec::with_capacity(points.len());
    for point in points {
        if point < 0x10000 {
            units.push(point as u16);
        } else {
            let point = point - 0x10000;
            units.push((0xd800 + (point >> 10)) as u16);
            units.push((0xdc00 + (point & 1023)) as u16);
        }
    }
    Ok(units)
}

/// `utf8(NSString*)`: UTF-16 units to UTF-8, refusing unpaired surrogates.
fn utf8(units: &[u16]) -> Result<Vec<u8>> {
    require(units.len() <= 16 * 1024 * 1024, "JSON string exceeds limit")?;
    let mut output = Vec::with_capacity(units.len());
    let mut i = 0;
    while i < units.len() {
        let mut point = u32::from(units[i]);
        if (0xd800..=0xdbff).contains(&point) {
            i += 1;
            if i == units.len() || !(0xdc00..=0xdfff).contains(&units[i]) {
                return Err(TextError::Refused(
                    "JSON string contains unpaired surrogate",
                ));
            }
            point = 0x10000 + ((point - 0xd800) << 10) + u32::from(units[i]) - 0xdc00;
        } else if (0xdc00..=0xdfff).contains(&point) {
            return Err(TextError::Refused(
                "JSON string contains unpaired surrogate",
            ));
        }
        push_scalar_utf8(&mut output, point).map_err(TextError::Utf8)?;
        i += 1;
    }
    Ok(output)
}

/// `utf8(id)` for a JSON value: a string, length-checked in UTF-16 units.
fn string_of(value: Option<&Json>) -> Result<Vec<u8>> {
    let Some(Json::String(bytes)) = value else {
        return Err(TextError::Refused("expected JSON string"));
    };
    utf8(&utf16(bytes)?)
}

fn dictionary(value: Option<&Json>) -> Result<&Json> {
    match value {
        Some(object @ Json::Object(_)) => Ok(object),
        _ => Err(TextError::Refused("expected JSON object")),
    }
}

fn array(value: Option<&Json>) -> Result<&[Json]> {
    match value {
        Some(Json::Array(items)) => Ok(items),
        _ => Err(TextError::Refused("expected JSON array")),
    }
}

fn boolean(value: Option<&Json>) -> Result<bool> {
    match value {
        Some(Json::Bool(flag)) => Ok(*flag),
        _ => Err(TextError::Refused("expected JSON boolean")),
    }
}

/// `integer(id)`: an `NSNumber` (not a boolean) whose double value is a finite integer in `0..=0xffffffff`.
fn integer(value: Option<&Json>) -> Result<u32> {
    let Some(Json::Number(number)) = value else {
        return Err(TextError::Refused("expected unsigned JSON integer"));
    };
    let parsed = from_chars_f64(number.lexeme()).unwrap_or(f64::NAN);
    if !parsed.is_finite()
        || parsed < 0.0
        || parsed > f64::from(u32::MAX)
        || parsed.floor() != parsed
    {
        return Err(TextError::Refused("JSON integer is out of range"));
    }
    Ok(parsed as u32)
}

fn null_or_missing(value: Option<&Json>) -> Result<()> {
    require(
        matches!(value, None | Some(Json::Null)),
        "unsupported non-null tokenizer setting",
    )
}

fn get<'a>(object: &'a Json, key: &str) -> Option<&'a Json> {
    match object {
        Json::Object(fields) => fields
            .iter()
            .find(|(name, _)| name.as_slice() == key.as_bytes())
            .map(|(_, v)| v),
        _ => None,
    }
}

/// `string_fingerprint`: SHA-256 over little-endian u64 lengths and bytes.
fn string_fingerprint<S: AsRef<[u8]>>(values: &[S]) -> String {
    let mut context = Sha256::default();
    context.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        let value = value.as_ref();
        context.update(&(value.len() as u64).to_le_bytes());
        context.update(value);
    }
    hex(&context.finalize())
}

struct Parsed {
    data: BpeData,
    pattern: Vec<u8>,
    vocabulary_fingerprint: String,
    merge_fingerprint: String,
}

fn parse_model(root: &Json) -> Result<Parsed> {
    require(
        string_of(get(root, "version"))? == b"1.0",
        "unsupported tokenizer JSON version",
    )?;
    null_or_missing(get(root, "truncation"))?;
    null_or_missing(get(root, "padding"))?;
    require(
        string_of(get(dictionary(get(root, "normalizer"))?, "type"))? == b"NFC",
        "unsupported tokenizer normalizer",
    )?;
    let sequence = dictionary(get(root, "pre_tokenizer"))?;
    require(
        string_of(get(sequence, "type"))? == b"Sequence",
        "expected declared tokenizer sequence",
    )?;
    let steps = array(get(sequence, "pretokenizers"))?;
    require(steps.len() == 2, "unsupported tokenizer sequence length")?;
    let split = dictionary(steps.first())?;
    require(
        string_of(get(split, "type"))? == b"Split"
            && string_of(get(split, "behavior"))? == b"Isolated"
            && !boolean(get(split, "invert"))?,
        "unsupported split settings",
    )?;
    let pattern = string_of(get(dictionary(get(split, "pattern"))?, "Regex"))?;
    let byte_level = |settings: &Json, prefix: bool, trim: bool, regex: bool| -> Result<()> {
        require(
            string_of(get(settings, "type"))? == b"ByteLevel"
                && boolean(get(settings, "add_prefix_space"))? == prefix
                && boolean(get(settings, "trim_offsets"))? == trim
                && boolean(get(settings, "use_regex"))? == regex,
            "unsupported byte-level settings",
        )
    };
    byte_level(dictionary(steps.get(1))?, false, false, false)?;
    byte_level(
        dictionary(get(root, "post_processor"))?,
        false,
        false,
        false,
    )?;
    byte_level(dictionary(get(root, "decoder"))?, true, true, true)?;
    let model = dictionary(get(root, "model"))?;
    require(
        string_of(get(model, "type"))? == b"BPE",
        "unsupported tokenizer model",
    )?;
    for name in [
        "dropout",
        "unk_token",
        "continuing_subword_prefix",
        "end_of_word_suffix",
    ] {
        null_or_missing(get(model, name))?;
    }
    for name in ["fuse_unk", "byte_fallback", "ignore_merges"] {
        require(!boolean(get(model, name))?, "unsupported BPE option")?;
    }
    let mut data = BpeData::default();
    let mut added = std::collections::BTreeMap::new();
    let markers = array(get(root, "added_tokens"))?;
    require(markers.len() <= 4096, "added token count exceeds limit")?;
    for item in markers {
        let token = dictionary(Some(item))?;
        for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
            require(
                !boolean(get(token, flag))?,
                "unsupported added-token matching flags",
            )?;
        }
        let id = integer(get(token, "id"))?;
        let content = string_of(get(token, "content"))?;
        require(
            added.insert(id, content.clone()).is_none(),
            "duplicate added token ID",
        )?;
        data.added.push(AddedToken {
            id,
            content,
            special: boolean(get(token, "special"))?,
        });
    }
    let Json::Object(vocabulary) = dictionary(get(model, "vocab"))? else {
        return Err(TextError::Refused("expected JSON object"));
    };
    require(
        vocabulary.len() >= 256 && vocabulary.len() <= 1_000_000,
        "tokenizer vocabulary count exceeds limit",
    )?;
    data.vocabulary = vec![Vec::new(); vocabulary.len()];
    let mut spellings: Vec<Vec<u8>> = vec![Vec::new(); vocabulary.len()];
    let mut seen = vec![false; vocabulary.len()];
    for (spelling, value) in vocabulary {
        let id = integer(Some(value))? as usize;
        require(
            id < vocabulary.len() && !seen[id],
            "noncontiguous or duplicate vocabulary ID",
        )?;
        seen[id] = true;
        let encoded = utf8(&utf16(spelling)?)?;
        spellings[id] = encoded.clone();
        if let Some(marker) = added.get(&(id as u32)) {
            require(*marker == encoded, "added token differs from vocabulary")?;
            data.vocabulary[id] = encoded;
        } else {
            data.vocabulary[id] = byte_level_decode(&encoded)?;
        }
    }
    let vocabulary_fingerprint = string_fingerprint(&spellings);
    spellings.clear();
    let merges = array(get(model, "merges"))?;
    require(merges.len() <= 1_000_000, "merge count exceeds limit")?;
    data.merges.reserve(merges.len());
    for item in merges {
        let parts = array(Some(item))?;
        require(parts.len() == 2, "unsupported merge representation")?;
        let left = string_of(parts.first())?;
        let right = string_of(parts.get(1))?;
        data.merges
            .push((byte_level_decode(&left)?, byte_level_decode(&right)?));
        let mut joined = left;
        joined.push(b' ');
        joined.extend_from_slice(&right);
        spellings.push(joined);
    }
    let merge_fingerprint = string_fingerprint(&spellings);
    Ok(Parsed {
        data,
        pattern,
        vocabulary_fingerprint,
        merge_fingerprint,
    })
}

/// `AppleProcessor`: NFC, then the declared split through `NSRegularExpression`.
struct AppleProcessor {
    regex: Regex,
}

impl AppleProcessor {
    fn new(pattern: &[u8]) -> Result<Self> {
        require(
            pattern == SUPPORTED_PATTERN.as_bytes(),
            "unsupported tokenizer split pattern",
        )?;
        let compiled = Utf16String::new(&utf16(pattern)?)
            .and_then(|units| Regex::new(&units))
            .map_err(TextError::SplitCompile)?;
        Ok(Self { regex: compiled })
    }
}

impl TextProcessor for AppleProcessor {
    type Error = Box<TextError>;

    fn pieces(&self, segment: &[u8]) -> std::result::Result<Vec<Vec<u8>>, Box<TextError>> {
        self.split(segment).map_err(Box::new)
    }
}

impl AppleProcessor {
    fn split(&self, segment: &[u8]) -> Result<Vec<Vec<u8>>> {
        require(
            segment.len() <= TEXT_LIMIT,
            "normalizer input exceeds limit",
        )?;
        let units = utf16(&normalize_nfc9(segment)?)?;
        let normalized = Utf16String::new(&units).map_err(TextError::Utf16String)?;
        let mut output = Vec::new();
        let mut last = 0;
        let mut exceeded = false;
        let mut failure = None;
        self.regex
            .enumerate_matches(&normalized, &mut |location, length| {
                if output.len() >= 1_000_000 {
                    exceeded = true;
                    return false;
                }
                let pieces = (|| -> Result<()> {
                    if location > last {
                        output.push(utf8(&units[last..location])?);
                    }
                    if length != 0 {
                        output.push(utf8(&units[location..location + length])?);
                    }
                    Ok(())
                })();
                if let Err(error) = pieces {
                    failure = Some(error);
                    return false;
                }
                last = location + length;
                true
            });
        if let Some(error) = failure {
            return Err(error);
        }
        require(!exceeded, "pretokenizer piece limit exceeded")?;
        if last < units.len() {
            output.push(utf8(&units[last..])?);
        }
        Ok(output)
    }
}

/// The macOS tokenizer adapter (`NativeText`).
pub struct NativeText {
    bpe: ByteBpe,
    processor: AppleProcessor,
    vocabulary_fingerprint: String,
    merge_fingerprint: String,
}

impl fmt::Debug for NativeText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeText")
            .field("vocabulary_size", &self.bpe.vocabulary_size())
            .field("merge_count", &self.bpe.merge_count())
            .finish_non_exhaustive()
    }
}

impl NativeText {
    pub fn new(tokenizer_json: &Path, expected_sha256: &str) -> Result<Self> {
        require(
            expected_sha256.len() == 64
                && expected_sha256
                    .bytes()
                    .all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')),
            "tokenizer asset requires a lowercase SHA256 pin",
        )?;
        let file = MappedFile::new(tokenizer_json)?;
        require(
            file.size() <= 64 * 1024 * 1024,
            "tokenizer JSON exceeds limit",
        )?;
        let bytes = file.bytes()?;
        require(
            hex(&sha256(&bytes)) == expected_sha256,
            "tokenizer JSON hash differs from supplied pin",
        )?;
        require(
            bytes.len() <= 64 * 1024 * 1024,
            "JSON document exceeds limit",
        )?;
        let root = parse_json_limited(&bytes, JsonLimits::new(64 * 1024 * 1024, usize::MAX, 64))
            .map_err(TextError::ModelParse)?;
        require(root.kind() == Kind::Object, "expected JSON object")?;
        let parsed = parse_model(&root)?;
        let processor = AppleProcessor::new(&parsed.pattern)?;
        let bpe = ByteBpe::new(parsed.data)?;
        file.assert_unchanged()?;
        Ok(Self {
            bpe,
            processor,
            vocabulary_fingerprint: parsed.vocabulary_fingerprint,
            merge_fingerprint: parsed.merge_fingerprint,
        })
    }

    pub fn encode(&self, text: &[u8]) -> Result<Vec<u32>> {
        self.bpe
            .encode(text, &self.processor)
            .map_err(|error| match error {
                TokenizerError::Pieces(inner) => *inner,
                TokenizerError::ByteLevelDecode(reason) => {
                    TokenizerError::ByteLevelDecode(reason).into()
                }
                TokenizerError::New(reason) => TokenizerError::New(reason).into(),
                TokenizerError::EncodePiece(reason) => TokenizerError::EncodePiece(reason).into(),
                TokenizerError::Encode(reason) => TokenizerError::Encode(reason).into(),
                TokenizerError::DecodeBytes(reason) => TokenizerError::DecodeBytes(reason).into(),
            })
    }

    pub fn encode_rendered(&self, rendered: &tokspur_text::chat::RenderedChat) -> Result<Vec<u32>> {
        self.bpe
            .encode_with_literals(&rendered.prompt, &self.processor, &rendered.literal_ranges)
            .map_err(TextError::Tokenizer)
    }

    /// Exact represented bytes, including incomplete UTF-8 and markers.
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Result<Vec<u8>> {
        Ok(self.bpe.decode_bytes(ids, skip_special)?)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<Vec<u8>> {
        Ok(self.bpe.decode_text(ids, skip_special)?)
    }

    /// The ID of the added token whose content is exactly `content`, if there is one.
    pub fn added_token(&self, content: &[u8]) -> Option<u32> {
        self.bpe.added_token(content)
    }

    pub fn vocabulary_size(&self) -> usize {
        self.bpe.vocabulary_size()
    }

    pub fn merge_count(&self) -> usize {
        self.bpe.merge_count()
    }

    /// Compares the JSON tokenizer with the GGUF tokenizer metadata.
    pub fn verify_gguf(&self, model: &ModelFiles) -> Result<()> {
        let mut found = 0;
        for shard in model.shards() {
            if shard.find("tokenizer.ggml.tokens").is_none() {
                continue;
            }
            found += 1;
            let tokens = shard.string_array("tokenizer.ggml.tokens")?;
            let merges = shard.string_array("tokenizer.ggml.merges")?;
            require(
                tokens.len() == self.vocabulary_size() && merges.len() == self.merge_count(),
                "JSON/GGUF tokenizer counts differ",
            )?;
            require(
                string_fingerprint(&tokens) == self.vocabulary_fingerprint,
                "JSON vocabulary differs from GGUF",
            )?;
            require(
                string_fingerprint(&merges) == self.merge_fingerprint,
                "JSON merge ranks differ from GGUF",
            )?;
        }
        require(found > 0, "GGUF shard set lacks tokenizer metadata")?;
        Ok(model.assert_unchanged()?)
    }
}

/// One input line for `text_cli`: drains an oversized line instead of failing, so the caller can report it and keep serving.
fn read_text_line(
    input: &mut impl std::io::BufRead,
    output: &mut Vec<u8>,
    oversized: &mut bool,
) -> std::io::Result<bool> {
    output.clear();
    *oversized = false;
    let mut any = false;
    let mut byte = [0u8; 1];
    loop {
        if input.read(&mut byte)? == 0 {
            return Ok(any);
        }
        any = true;
        if byte[0] == b'\n' {
            return Ok(true);
        }
        if output.len() >= TEXT_LIMIT {
            *oversized = true;
        } else {
            output.push(byte[0]);
        }
    }
}

/// Failures `text_cli` reports through `main` (`tokspur-native: ...`).
pub enum TextCliError {
    /// Startup refused the options or assets.
    Startup(TextError),
    /// Startup could not load the pinned chat template or renderer.
    Chat(Box<dyn std::error::Error + Send + Sync>),
    /// Reading requests or writing replies failed.
    Stream(std::io::Error),
}

impl fmt::Debug for TextCliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Startup(error) => formatter.debug_tuple("Startup").field(error).finish(),
            Self::Chat(_) => formatter.write_str("Chat([redacted])"),
            Self::Stream(_) => formatter.write_str("Stream([redacted])"),
        }
    }
}

impl fmt::Display for TextCliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Startup(error) => error.fmt(formatter),
            Self::Chat(error) => error.fmt(formatter),
            Self::Stream(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TextCliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Startup(error) => Some(error),
            Self::Chat(error) => Some(error.as_ref()),
            Self::Stream(error) => Some(error),
        }
    }
}

impl From<TextError> for TextCliError {
    fn from(error: TextError) -> Self {
        Self::Startup(error)
    }
}

impl From<GgufError> for TextCliError {
    fn from(error: GgufError) -> Self {
        Self::Startup(TextError::Gguf(error))
    }
}

const SUPPORTED_TEMPLATE_PIN: &str = tokspur_text::assets::TEMPLATE_SHA256;

fn write_json(
    output: &mut impl std::io::Write,
    value: &Json,
) -> std::result::Result<(), TextCliError> {
    let mut line = value
        .dump()
        .map_err(|error| TextCliError::Startup(error.into()))?;
    line.push(b'\n');
    output
        .write_all(&line)
        .and_then(|()| output.flush())
        .map_err(TextCliError::Stream)
}

/// `text_cli` (`tokspur-native text|chat`): one JSON request per line, one JSON reply per line; per-line refusals become `{"error": ...}` and exit 1.
pub fn text_cli(
    args: &[std::ffi::OsString],
    input: &mut impl std::io::BufRead,
    output: &mut impl std::io::Write,
) -> std::result::Result<i32, TextCliError> {
    use std::os::unix::ffi::OsStrExt;
    let chat_mode = args.get(1).is_some_and(|mode| mode == "chat");
    let mut path = std::path::PathBuf::new();
    let mut pin = String::new();
    let mut template_path = std::path::PathBuf::new();
    let mut template_pin = String::new();
    let mut shards: Vec<std::path::PathBuf> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        require(i + 1 < args.len(), "missing text option value")?;
        let flag = args[i].as_bytes();
        let value = &args[i + 1];
        let text = value.to_str().unwrap_or("\u{fffd}");
        match flag {
            b"--tokenizer" if path.as_os_str().is_empty() => path = value.into(),
            b"--sha256" if pin.is_empty() => pin = text.to_owned(),
            b"--template" if template_path.as_os_str().is_empty() => template_path = value.into(),
            b"--template-sha256" if template_pin.is_empty() => template_pin = text.to_owned(),
            b"--gguf" => shards.push(value.into()),
            _ => return Err(TextError::Refused("unknown or duplicate text option").into()),
        }
        i += 2;
    }
    require(
        !path.as_os_str().is_empty() && !pin.is_empty(),
        "text requires --tokenizer PATH --sha256 PIN",
    )?;
    let mut renderer = None;
    if chat_mode {
        require(
            !template_path.as_os_str().is_empty() && template_pin == SUPPORTED_TEMPLATE_PIN,
            "chat requires the supported pinned template",
        )?;
        let file = MappedFile::new(&template_path)?;
        require(file.size() <= 64 * 1024, "chat template exceeds limit")?;
        let bytes = file.bytes()?;
        require(
            hex(&sha256(&bytes)) == template_pin,
            "chat template differs from its pin",
        )?;
        let strings = tokspur_text::chat::extract_flash_next_literals(&bytes)
            .map_err(|error| TextCliError::Chat(Box::new(error)))?;
        renderer = Some(
            tokspur_text::chat::ChatRenderer::new(strings)
                .map_err(|error| TextCliError::Chat(Box::new(error)))?,
        );
        file.assert_unchanged()?;
    } else {
        require(
            template_path.as_os_str().is_empty() && template_pin.is_empty(),
            "template options require chat mode",
        )?;
    }
    let tokenizer = NativeText::new(&path, &pin)?;
    if !shards.is_empty() {
        tokenizer.verify_gguf(&ModelFiles::new(&shards)?)?;
    }
    let mut request = Vec::new();
    let mut oversized = false;
    let mut failed = false;
    while read_text_line(input, &mut request, &mut oversized).map_err(TextCliError::Stream)? {
        match text_reply(&tokenizer, renderer.as_ref(), &request, oversized) {
            Ok(reply) => write_json(output, &reply)?,
            Err(error) => {
                failed = true;
                write_json(
                    output,
                    &Json::new_object([("error", Json::from(error.to_string()))]),
                )?;
            }
        }
    }
    Ok(i32::from(failed))
}

/// One `text_cli` request (the C++ per-line `try`).
fn text_reply(
    tokenizer: &NativeText,
    renderer: Option<&tokspur_text::chat::ChatRenderer>,
    request: &[u8],
    oversized: bool,
) -> std::result::Result<Json, Box<dyn std::error::Error + Send + Sync>> {
    require(!oversized, "text request line exceeds limit")?;
    let object = tokspur_json::parse_json(request)?;
    let ids_json =
        |ids: &[u32]| Json::Array(ids.iter().map(|&id| Json::integer(u64::from(id))).collect());
    if let Some(renderer) = renderer {
        let rendered = renderer.render(&object)?;
        let ids = tokenizer.encode_rendered(&rendered)?;
        return Ok(Json::new_object([
            ("prompt", Json::String(rendered.prompt)),
            ("ids", ids_json(&ids)),
            ("normalized_messages", rendered.normalized_messages),
            (
                "parsed_tool_arguments",
                Json::integer(rendered.parsed_tool_arguments as u64),
            ),
            (
                "hoisted_system_messages",
                Json::integer(rendered.hoisted_system_messages as u64),
            ),
        ]));
    }
    for (key, _) in object.object()? {
        require(
            matches!(key.as_slice(), b"text" | b"ids" | b"skip_special_tokens"),
            "unknown text request field",
        )?;
    }
    let text = object.find("text")?;
    let input_ids = object.find("ids")?;
    require(
        text.is_some() != input_ids.is_some(),
        "provide exactly one of text or ids",
    )?;
    let skip = match object.find("skip_special_tokens")? {
        Some(option) => option.boolean()?,
        None => false,
    };
    let ids = match (text, input_ids) {
        (Some(text), _) => tokenizer.encode(text.string()?)?,
        (None, Some(input)) => {
            let input = input.array()?;
            require(input.len() <= 1_000_000, "decode token count exceeds limit")?;
            let mut ids = Vec::with_capacity(input.len());
            for value in input {
                let id = value.unsigned_integer()?;
                require(id <= 0xffff_ffff, "token ID exceeds range")?;
                ids.push(id as u32);
            }
            ids
        }
        (None, None) => Vec::new(),
    };
    let decoded = tokenizer.decode(&ids, skip)?;
    Ok(Json::new_object([
        ("ids", ids_json(&ids)),
        ("text", Json::String(decoded)),
        (
            "vocabulary_size",
            Json::integer(tokenizer.vocabulary_size() as u64),
        ),
        ("merge_count", Json::integer(tokenizer.merge_count() as u64)),
    ]))
}

impl tokspur_text::response_cli::ResponseTokenizer for NativeText {
    type Error = TextError;

    fn encode(&self, text: &[u8]) -> Result<Vec<u32>> {
        NativeText::encode(self, text)
    }

    fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        NativeText::decode_bytes(self, ids, false)
    }

    fn vocabulary_size(&self) -> usize {
        NativeText::vocabulary_size(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_isolates_the_pinned_pattern_pieces() {
        let processor = AppleProcessor::new(SUPPORTED_PATTERN.as_bytes()).unwrap();
        let pieces = processor.split(b"Hi, it's 42 ok\n\n  end").unwrap();
        let expected: [&[u8]; 11] = [
            b"Hi", b",", b" it", b"'s", b" ", b"4", b"2", b" ok", b"\n\n", b" ", b" end",
        ];
        assert_eq!(pieces, expected);
    }

    #[test]
    fn split_refuses_past_the_piece_limit() {
        let processor = AppleProcessor::new(SUPPORTED_PATTERN.as_bytes()).unwrap();
        // Each digit is a piece of its own.
        assert_eq!(
            processor.split(&[b'7'; 1_000_000]).unwrap().len(),
            1_000_000
        );
        assert!(matches!(
            processor.split(&[b'7'; 1_000_001]),
            Err(TextError::Refused("pretokenizer piece limit exceeded"))
        ));
    }
}
