//! The GGUF reader the server needs, ported one to one from `native/src/gguf.cpp` and `native/include/tokspur/gguf.hpp`.
//!
//! The server builds [`ModelFiles`] from its shard paths and compares the tokenizer arrays with the pinned tokenizer JSON, so this crate must accept and refuse exactly the shard sets the C++ does. Every limit, ordering rule and check follows the C++ line by line, and every error's `Display` is the exact C++ `what()`.
//!
//! One deliberate difference: safe Rust cannot `mmap`. [`MappedFile`] keeps the C++ open flags, checks and `fstat` snapshot but reads with positioned reads. [`Gguf::from_path`] reads only the header region the parser touches (the cursor still bounds every read by the whole file's size), and [`Gguf::tensor_bytes`] is a positioned read into a caller buffer. The zero-copy tensor path the engine needs belongs to phase B and is out of scope here.

#![forbid(unsafe_code)]

mod file;

use std::borrow::Cow;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io;
use std::path::Path;

use file::Header;
pub use file::MappedFile;

/// `header_limit`: the cursor never reads past this offset.
const HEADER_LIMIT: u64 = 256 * 1024 * 1024;
/// `record_limit`: tensor and metadata record counts.
const RECORD_LIMIT: u64 = 100_000;
/// `array_limit`: metadata array elements.
const ARRAY_LIMIT: u64 = 16 * 1024 * 1024;

const TRUNCATED_HEADER: &str = "truncated or oversized GGUF header";
const GENERAL_ALIGNMENT: &[u8] = b"general.alignment";

/// Fixed element widths by metadata type; zero for bool, string and array, whose elements are validated one by one.
const ARRAY_WIDTHS: [u64; 13] = [1, 1, 2, 2, 4, 4, 4, 0, 0, 0, 8, 8, 8];
/// Smallest encoding of one value of each metadata type (a bool is a byte, a string its length, an array its type and count). Read-ahead hint only.
const MINIMUM_VALUE: [u64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 8, 12, 8, 8, 8];

/// The C++ `FormatError` raised by the GGUF reader. `Display` is the exact C++ `what()`; the server reports it on its refusal line.
pub enum GgufError {
    /// `MappedFile`: `open` refused the path.
    Open(io::Error),
    /// `MappedFile`: `fstat` failed (`Some`), or the path is not a nonempty regular file (`None`).
    Inspect(Option<io::Error>),
    /// `MappedFile`: bytes could not be made available. The C++ raises this when `mmap` fails; see [`MappedFile`] for when this port does.
    Map(io::Error),
    /// `assert_unchanged`: `fstat` failed (`Some`), or the size, device, inode or a timestamp differs from the snapshot (`None`).
    Unchanged(Option<io::Error>),
    /// `Gguf::parse` or its cursor refused the header or a metadata record.
    Header(&'static str),
    /// `checked_add` or `checked_mul` overflowed.
    Arithmetic(&'static str),
    /// `encoding` does not know the tensor encoding.
    Encoding(u32),
    /// A typed metadata accessor refused the item.
    Metadata(&'static str),
    /// A keyed metadata accessor refused the item; the C++ message ends with the key. `Display` is exact for UTF-8 keys (every key the server and the GGUF format use); other bytes print as U+FFFD.
    Lookup { reason: &'static str, key: Vec<u8> },
    /// Tensor-info validation, tensor data bounds or a tensor lookup failed.
    Tensor(&'static str),
    /// `ModelFiles` refused the shard set.
    Shards(&'static str),
}

impl fmt::Debug for GgufError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(_) => formatter.write_str("Open([redacted])"),
            Self::Inspect(Some(_)) => formatter.write_str("Inspect(Some([redacted]))"),
            Self::Inspect(None) => formatter.write_str("Inspect(None)"),
            Self::Map(_) => formatter.write_str("Map([redacted])"),
            Self::Unchanged(Some(_)) => formatter.write_str("Unchanged(Some([redacted]))"),
            Self::Unchanged(None) => formatter.write_str("Unchanged(None)"),
            Self::Header(reason) => formatter.debug_tuple("Header").field(reason).finish(),
            Self::Arithmetic(reason) => formatter.debug_tuple("Arithmetic").field(reason).finish(),
            Self::Encoding(id) => formatter.debug_tuple("Encoding").field(id).finish(),
            Self::Metadata(reason) => formatter.debug_tuple("Metadata").field(reason).finish(),
            Self::Lookup { reason, key } => formatter
                .debug_struct("Lookup")
                .field("reason", reason)
                .field("key_bytes", &key.len())
                .finish(),
            Self::Tensor(reason) => formatter.debug_tuple("Tensor").field(reason).finish(),
            Self::Shards(reason) => formatter.debug_tuple("Shards").field(reason).finish(),
        }
    }
}

impl fmt::Display for GgufError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(_) => formatter.write_str("cannot open weight file"),
            Self::Inspect(_) => formatter.write_str("weight file must be a nonempty regular file"),
            Self::Map(_) => formatter.write_str("cannot map weight file"),
            Self::Unchanged(_) => formatter.write_str("weight file changed since mapping"),
            Self::Header(reason)
            | Self::Arithmetic(reason)
            | Self::Metadata(reason)
            | Self::Tensor(reason)
            | Self::Shards(reason) => formatter.write_str(reason),
            Self::Encoding(id) => write!(formatter, "unsupported GGUF tensor encoding {id}"),
            Self::Lookup { reason, key } => {
                formatter.write_str(reason)?;
                formatter.write_str(&String::from_utf8_lossy(key))
            }
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open(error) | Self::Map(error) => Some(error),
            Self::Inspect(error) | Self::Unchanged(error) => error
                .as_ref()
                .map(|error| error as &(dyn std::error::Error + 'static)),
            Self::Header(_)
            | Self::Arithmetic(_)
            | Self::Encoding(_)
            | Self::Metadata(_)
            | Self::Lookup { .. }
            | Self::Tensor(_)
            | Self::Shards(_) => None,
        }
    }
}

/// `Encoding`: an encoded tensor block. Dimensions and offsets are in the on-disk order: the first dimension is contiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Encoding {
    pub id: u32,
    pub name: &'static str,
    pub elements_per_block: u32,
    pub bytes_per_block: u32,
}

const fn layout(
    id: u32,
    name: &'static str,
    elements_per_block: u32,
    bytes_per_block: u32,
) -> Encoding {
    Encoding {
        id,
        name,
        elements_per_block,
        bytes_per_block,
    }
}

// Encoded block sizes, not in-memory C struct sizes. There is no external engine ABI.
static LAYOUTS: [Encoding; 20] = [
    layout(0, "F32", 1, 4),
    layout(1, "F16", 1, 2),
    layout(2, "Q4_0", 32, 18),
    layout(3, "Q4_1", 32, 20),
    layout(6, "Q5_0", 32, 22),
    layout(7, "Q5_1", 32, 24),
    layout(8, "Q8_0", 32, 34),
    layout(10, "Q2_K", 256, 84),
    layout(11, "Q3_K", 256, 110),
    layout(12, "Q4_K", 256, 144),
    layout(13, "Q5_K", 256, 176),
    layout(14, "Q6_K", 256, 210),
    layout(20, "IQ4_NL", 32, 18),
    layout(23, "IQ4_XS", 256, 136),
    layout(24, "I8", 1, 1),
    layout(25, "I16", 1, 2),
    layout(26, "I32", 1, 4),
    layout(27, "I64", 1, 8),
    layout(28, "F64", 1, 8),
    layout(30, "BF16", 1, 2),
];

/// `encoding(id)`.
pub fn encoding(id: u32) -> Result<&'static Encoding, GgufError> {
    LAYOUTS
        .iter()
        .find(|item| item.id == id)
        .ok_or(GgufError::Encoding(id))
}

/// `checked_add`.
pub fn checked_add(a: u64, b: u64) -> Result<u64, GgufError> {
    if a > u64::MAX - b {
        return Err(GgufError::Arithmetic("GGUF offset overflow"));
    }
    Ok(a + b)
}

/// `checked_mul`.
pub fn checked_mul(a: u64, b: u64) -> Result<u64, GgufError> {
    if a != 0 && b > u64::MAX / a {
        return Err(GgufError::Arithmetic("GGUF shape overflow"));
    }
    Ok(a * b)
}

/// `valid_utf8`: strict UTF-8 (no overlong forms, surrogates or values above U+10FFFF).
fn valid_utf8(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i < bytes.len() {
        let first = u32::from(bytes[i]);
        i += 1;
        if first < 0x80 {
            continue;
        }
        let count: u32 = if first >= 0xf0 {
            3
        } else if first >= 0xe0 {
            2
        } else {
            1
        };
        if !(0xc2..=0xf4).contains(&first) || count as usize > bytes.len() - i {
            return false;
        }
        let mut point = first & ((1u32 << (6 - count)) - 1);
        for _ in 0..count {
            let next = u32::from(bytes[i]);
            i += 1;
            if next & 0xc0 != 0x80 {
                return false;
            }
            point = (point << 6) | (next & 0x3f);
        }
        if (count == 1 && point < 0x80)
            || (count == 2 && point < 0x800)
            || (count == 3 && point < 0x10000)
            || point > 0x10ffff
            || (0xd800..=0xdfff).contains(&point)
        {
            return false;
        }
    }
    true
}

/// `valid_key`: dot-separated, nonempty `[a-z0-9_]` components.
fn valid_key(key: &[u8]) -> bool {
    if key.first().is_none_or(|&c| c == b'.') || key.last() == Some(&b'.') {
        return false;
    }
    let mut previous = b'.';
    for &c in key {
        if !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_' || c == b'.') {
            return false;
        }
        if c == b'.' && previous == b'.' {
            return false;
        }
        previous = c;
    }
    true
}

/// Where a cursor reads (`Cursor::bytes_` in C++).
enum Source<'s> {
    /// Bytes in memory: borrowed input, or a parsed file's header region. `size` is the whole input's size (`bytes_.size()`), which bounds reads.
    Memory { bytes: &'s [u8], size: u64 },
    /// A GGUF file's header region, read as the parser advances.
    File(&'s mut Header),
}

impl Source<'_> {
    fn size(&self) -> u64 {
        match self {
            Self::Memory { size, .. } => *size,
            Self::File(header) => header.size(),
        }
    }

    fn loaded(&self) -> &[u8] {
        match self {
            Self::Memory { bytes, .. } => bytes,
            Self::File(header) => header.loaded(),
        }
    }

    /// Bytes `[start, end)`; the cursor has checked `end` against `size()` and the header limit. A parsed file's header is only read at spans its parse already validated, so a miss reports as the cursor's bound would.
    fn bytes(&mut self, start: u64, end: u64) -> Result<&[u8], GgufError> {
        match self {
            // Both bounds are at most the header limit, so the casts are exact.
            Self::Memory { bytes, .. } => bytes
                .get(start as usize..end as usize)
                .ok_or(GgufError::Header(TRUNCATED_HEADER)),
            Self::File(header) => header.bytes(start, end),
        }
    }
}

/// `Cursor`: bounded little-endian reads over a GGUF header.
struct Cursor<'s> {
    bytes: Source<'s>,
    position: u64,
}

impl<'s> Cursor<'s> {
    fn new(bytes: Source<'s>, start: u64) -> Self {
        Self {
            bytes,
            position: start,
        }
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn loaded(&self) -> &[u8] {
        self.bytes.loaded()
    }

    /// Read-ahead hint for a file: a successful parse requests at least `end` bytes. Validation never depends on it.
    fn floor(&mut self, end: u64) {
        if let Source::File(header) = &mut self.bytes {
            header.floor(end);
        }
    }

    fn take(&mut self, n: u64) -> Result<&[u8], GgufError> {
        let end = checked_add(self.position, n)?;
        if end > self.bytes.size() || end > HEADER_LIMIT {
            return Err(GgufError::Header(TRUNCATED_HEADER));
        }
        let result = self.bytes.bytes(self.position, end)?;
        self.position = end;
        Ok(result)
    }

    fn number(&mut self, width: u64) -> Result<u64, GgufError> {
        let view = self.take(width)?;
        Ok(view.iter().enumerate().fold(0, |result, (i, &byte)| {
            result | (u64::from(byte) << (8 * i))
        }))
    }

    fn string(&mut self, limit: u64) -> Result<&[u8], GgufError> {
        let length = self.number(8)?;
        if length > limit {
            return Err(GgufError::Header("GGUF string exceeds limit"));
        }
        let view = self.take(length)?;
        if !valid_utf8(view) {
            return Err(GgufError::Header("invalid UTF-8 in GGUF string"));
        }
        Ok(view)
    }

    fn value(&mut self, value_type: u32, depth: u32) -> Result<Metadata, GgufError> {
        if depth > 8 {
            return Err(GgufError::Header("GGUF array nesting exceeds limit"));
        }
        let mut result = Metadata {
            r#type: value_type,
            offset: self.position,
            ..Metadata::default()
        };
        match value_type {
            0 | 1 => {
                self.take(1)?;
            }
            2 | 3 => {
                self.take(2)?;
            }
            4..=6 => {
                self.take(4)?;
            }
            10..=12 => {
                self.take(8)?;
            }
            7 => {
                if self.number(1)? > 1 {
                    return Err(GgufError::Header("invalid GGUF boolean"));
                }
            }
            8 => {
                self.string(HEADER_LIMIT)?;
            }
            9 => {
                // Four bytes: the cast is exact.
                result.element_type = self.number(4)? as u32;
                result.elements = self.number(8)?;
                if result.element_type > 12 || result.elements > ARRAY_LIMIT {
                    return Err(GgufError::Header("invalid GGUF array"));
                }
                let element_type = result.element_type as usize;
                if ARRAY_WIDTHS[element_type] != 0 {
                    self.take(checked_mul(result.elements, ARRAY_WIDTHS[element_type])?)?;
                } else {
                    for i in 0..result.elements {
                        let remaining =
                            (result.elements - i).saturating_mul(MINIMUM_VALUE[element_type]);
                        self.floor(self.position.saturating_add(remaining));
                        self.value(result.element_type, depth + 1)?;
                    }
                }
            }
            _ => return Err(GgufError::Header("unknown GGUF metadata type")),
        }
        result.end = self.position;
        Ok(result)
    }
}

/// `Metadata`: one validated key's value span. `offset` is the value's first byte; for arrays `element_type` and `elements` repeat the array header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Metadata {
    pub r#type: u32,
    pub offset: u64,
    pub end: u64,
    pub element_type: u32,
    pub elements: u64,
}

/// `Tensor`: one validated tensor info.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct Tensor {
    /// Printable ASCII (0x21 to 0x7e), at most 64 bytes.
    pub name: Vec<u8>,
    pub dimensions: Vec<u64>,
    pub r#type: u32,
    /// Absolute file offset, never relative to tensor_data.
    pub offset: u64,
    pub bytes: u64,
    pub elements: u64,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("name", &String::from_utf8_lossy(&self.name))
            .field("dimensions", &self.dimensions)
            .field("type", &self.r#type)
            .field("offset", &self.offset)
            .field("bytes", &self.bytes)
            .field("elements", &self.elements)
            .finish()
    }
}

/// `TensorLocation`: a tensor's shard and its index in that shard's tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorLocation {
    pub shard: usize,
    pub index: usize,
}

/// What `Gguf::parse` produces.
struct Parsed {
    metadata: BTreeMap<Vec<u8>, Metadata>,
    tensors: Vec<Tensor>,
    data_offset: u64,
}

/// `Gguf::parse`.
fn parse(bytes: Source<'_>) -> Result<Parsed, GgufError> {
    let size = bytes.size();
    let mut reader = Cursor::new(bytes, 0);
    // Read-ahead hint: magic, version and both counts.
    reader.floor(24);
    if reader.number(4)? != 0x4655_4747 || reader.number(4)? != 3 {
        return Err(GgufError::Header("expected little-endian GGUF v3"));
    }
    let tensor_count = reader.number(8)?;
    let metadata_count = reader.number(8)?;
    if tensor_count > RECORD_LIMIT || metadata_count > RECORD_LIMIT {
        return Err(GgufError::Header("GGUF record limit exceeded"));
    }
    let minimum_records = checked_add(
        checked_mul(tensor_count, 33)?,
        checked_mul(metadata_count, 14)?,
    )?;
    if minimum_records > size - reader.position() {
        return Err(GgufError::Header("GGUF counts exceed available file bytes"));
    }
    let mut metadata = BTreeMap::new();
    for i in 0..metadata_count {
        // Read-ahead hint: every remaining record needs 14 bytes, every tensor 33.
        reader.floor(reader.position() + (metadata_count - i) * 14 + tensor_count * 33);
        let key = reader.string(65535)?.to_vec();
        if !valid_key(&key) {
            return Err(GgufError::Header("invalid GGUF metadata key"));
        }
        // Four bytes: the cast is exact.
        let value_type = reader.number(4)? as u32;
        let value = reader.value(value_type, 0)?;
        match metadata.entry(key) {
            Entry::Vacant(slot) => {
                slot.insert(value);
            }
            Entry::Occupied(_) => return Err(GgufError::Header("duplicate GGUF metadata key")),
        }
    }
    let alignment = if metadata.contains_key(GENERAL_ALIGNMENT) {
        let bytes = Source::Memory {
            bytes: reader.loaded(),
            size,
        };
        unsigned_value(bytes, &metadata, GENERAL_ALIGNMENT)?
    } else {
        32
    };
    if alignment == 0 || alignment > 65536 || alignment & (alignment - 1) != 0 {
        return Err(GgufError::Header("unsupported GGUF alignment"));
    }
    // At most the record limit, so the cast is exact.
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    let mut names = BTreeSet::new();
    for i in 0..tensor_count {
        reader.floor(reader.position() + (tensor_count - i) * 33);
        let mut tensor = Tensor {
            name: reader.string(64)?.to_vec(),
            ..Tensor::default()
        };
        if tensor.name.is_empty() || !names.insert(tensor.name.clone()) {
            return Err(GgufError::Tensor("empty or duplicate tensor name"));
        }
        if tensor.name.iter().any(|c| !(0x21..=0x7e).contains(c)) {
            return Err(GgufError::Tensor("non-ASCII tensor name"));
        }
        let dimensions = reader.number(4)?;
        if dimensions == 0 || dimensions > 4 {
            return Err(GgufError::Tensor("unsupported GGUF tensor rank"));
        }
        tensor.elements = 1;
        for _ in 0..dimensions {
            let dimension = reader.number(8)?;
            if dimension == 0 {
                return Err(GgufError::Tensor("zero tensor dimension"));
            }
            tensor.dimensions.push(dimension);
            tensor.elements = checked_mul(tensor.elements, dimension)?;
        }
        // Four bytes: the cast is exact.
        tensor.r#type = reader.number(4)? as u32;
        let layout = encoding(tensor.r#type)?;
        let elements_per_block = u64::from(layout.elements_per_block);
        if !tensor.dimensions[0].is_multiple_of(elements_per_block) {
            return Err(GgufError::Tensor(
                "quantized tensor row is not block-aligned",
            ));
        }
        tensor.bytes = checked_mul(
            tensor.elements / elements_per_block,
            u64::from(layout.bytes_per_block),
        )?;
        tensor.offset = reader.number(8)?;
        if !tensor.offset.is_multiple_of(alignment) {
            return Err(GgufError::Tensor("unaligned GGUF tensor offset"));
        }
        tensors.push(tensor);
    }
    let mut data_offset = checked_add(reader.position(), alignment - 1)? & !(alignment - 1);
    // Metadata-only shards have no tensor data section requiring alignment padding.
    if tensor_count == 0 {
        data_offset = data_offset.min(size);
    }
    if data_offset > size {
        return Err(GgufError::Header("truncated GGUF header padding"));
    }
    let mut intervals = Vec::with_capacity(tensors.len());
    for tensor in &mut tensors {
        tensor.offset = checked_add(data_offset, tensor.offset)?;
        let end = checked_add(tensor.offset, tensor.bytes)?;
        if end > size {
            return Err(GgufError::Tensor("truncated GGUF tensor data"));
        }
        intervals.push((tensor.offset, end));
    }
    intervals.sort_unstable();
    if intervals.windows(2).any(|pair| pair[1].0 < pair[0].1) {
        return Err(GgufError::Tensor("overlapping GGUF tensors"));
    }
    Ok(Parsed {
        metadata,
        tensors,
        data_offset,
    })
}

/// `Gguf::unsigned_value`, shared with `parse`, which reads `general.alignment` before the header is complete.
fn unsigned_value(
    bytes: Source<'_>,
    metadata: &BTreeMap<Vec<u8>, Metadata>,
    key: &[u8],
) -> Result<u64, GgufError> {
    let Some(item) = metadata.get(key) else {
        return Err(GgufError::Lookup {
            reason: "missing GGUF metadata ",
            key: key.to_vec(),
        });
    };
    let (width, signed_value) = match item.r#type {
        0 => (1, false),
        1 => (1, true),
        2 => (2, false),
        3 => (2, true),
        4 => (4, false),
        5 => (4, true),
        10 => (8, false),
        11 => (8, true),
        _ => return Err(GgufError::Metadata("GGUF metadata is not an integer")),
    };
    let mut reader = Cursor::new(bytes, item.offset);
    let result = reader.number(width)?;
    if signed_value && result >> (width * 8 - 1) != 0 {
        return Err(GgufError::Metadata("negative GGUF count"));
    }
    Ok(result)
}

/// `Gguf`: one validated GGUF v3 file or borrowed byte span.
///
/// For a file, only the header region the parse read is held in memory; [`file_size`](Self::file_size) is still the whole file's size. `Debug` prints counts only: the header holds tokenizer strings.
pub struct Gguf<'a> {
    mapping: Option<MappedFile>, // Zero or one; makes borrowed test input explicit.
    data: Cow<'a, [u8]>,
    size: u64,
    metadata: BTreeMap<Vec<u8>, Metadata>,
    tensors: Vec<Tensor>,
    data_offset: u64,
}

impl fmt::Debug for Gguf<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gguf")
            .field("mapped", &self.mapping.is_some())
            .field("tensors", &self.tensors.len())
            .field("metadata", &self.metadata.len())
            .field("data_offset", &self.data_offset)
            .field("file_size", &self.size)
            .finish_non_exhaustive()
    }
}

impl Gguf<'static> {
    /// `Gguf(path)`: opens the file, reads and validates its header region, then checks that the file is unchanged.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let mut header = Header::new(MappedFile::new(path)?);
        let parsed = parse(Source::File(&mut header))?;
        let size = header.size();
        let (file, data) = header.into_parts();
        let gguf = Self::assemble(Some(file), Cow::Owned(data), size, parsed);
        gguf.assert_unchanged()?;
        Ok(gguf)
    }
}

impl<'a> Gguf<'a> {
    /// `Gguf(std::span)`: borrowed bytes, useful for model-free tests.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, GgufError> {
        let size = bytes.len() as u64;
        let parsed = parse(Source::Memory { bytes, size })?;
        Ok(Self::assemble(None, Cow::Borrowed(bytes), size, parsed))
    }

    fn assemble(
        mapping: Option<MappedFile>,
        data: Cow<'a, [u8]>,
        size: u64,
        parsed: Parsed,
    ) -> Self {
        Self {
            mapping,
            data,
            size,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            data_offset: parsed.data_offset,
        }
    }

    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// Validated keys in byte order (`std::map<std::string, ...>`).
    pub fn metadata(&self) -> &BTreeMap<Vec<u8>, Metadata> {
        &self.metadata
    }

    pub fn find(&self, key: impl AsRef<[u8]>) -> Option<&Metadata> {
        self.metadata.get(key.as_ref())
    }

    pub fn unsigned_value(&self, key: impl AsRef<[u8]>) -> Result<u64, GgufError> {
        unsigned_value(self.memory(), &self.metadata, key.as_ref())
    }

    pub fn floating_value(&self, key: impl AsRef<[u8]>) -> Result<f64, GgufError> {
        let item = match self.find(key) {
            Some(item) if item.r#type == 6 || item.r#type == 12 => item,
            _ => return Err(GgufError::Metadata("GGUF metadata is not floating point")),
        };
        self.assert_unchanged()?;
        let mut reader = self.cursor(item.offset);
        let value = if item.r#type == 6 {
            // Four bytes: the cast is exact.
            f64::from(f32::from_bits(reader.number(4)? as u32))
        } else {
            f64::from_bits(reader.number(8)?)
        };
        if !value.is_finite() {
            return Err(GgufError::Metadata("nonfinite GGUF metadata"));
        }
        self.assert_unchanged()?;
        Ok(value)
    }

    pub fn unsigned_array(&self, key: impl AsRef<[u8]>) -> Result<Vec<u64>, GgufError> {
        let item = match self.find(key) {
            Some(item) if item.r#type == 9 => item,
            _ => return Err(GgufError::Metadata("GGUF metadata is not an integer array")),
        };
        let (width, signed_value) = match item.element_type {
            0 => (1, false),
            1 => (1, true),
            2 => (2, false),
            3 => (2, true),
            4 => (4, false),
            5 => (4, true),
            10 => (8, false),
            11 => (8, true),
            _ => return Err(GgufError::Metadata("GGUF array has noninteger elements")),
        };
        self.assert_unchanged()?;
        let mut reader = self.cursor(item.offset);
        if reader.number(4)? != u64::from(item.element_type) || reader.number(8)? != item.elements {
            return Err(GgufError::Metadata("GGUF array changed after validation"));
        }
        // At most the array limit, so the cast is exact.
        let mut values = Vec::with_capacity(item.elements as usize);
        for _ in 0..item.elements {
            let value = reader.number(width)?;
            if signed_value && value >> (width * 8 - 1) != 0 {
                return Err(GgufError::Metadata("negative GGUF array count"));
            }
            values.push(value);
        }
        if reader.position() != item.end {
            return Err(GgufError::Metadata("GGUF integer array span mismatch"));
        }
        self.assert_unchanged()?;
        Ok(values)
    }

    pub fn string_value(&self, key: impl AsRef<[u8]>) -> Result<&[u8], GgufError> {
        let key = key.as_ref();
        let item = match self.find(key) {
            Some(item) if item.r#type == 8 => item,
            _ => {
                return Err(GgufError::Lookup {
                    reason: "GGUF metadata is not a string: ",
                    key: key.to_vec(),
                });
            }
        };
        let mut reader = self.cursor(item.offset);
        self.string_at(&mut reader)
    }

    pub fn string_array(&self, key: impl AsRef<[u8]>) -> Result<Vec<&[u8]>, GgufError> {
        let item = match self.find(key) {
            Some(item) if item.r#type == 9 && item.element_type == 8 => item,
            _ => return Err(GgufError::Metadata("GGUF metadata is not a string array")),
        };
        self.assert_unchanged()?;
        let mut reader = self.cursor(item.offset);
        if reader.number(4)? != 8 || reader.number(8)? != item.elements {
            return Err(GgufError::Metadata("GGUF array changed after validation"));
        }
        // At most the array limit, so the cast is exact.
        let mut result = Vec::with_capacity(item.elements as usize);
        for _ in 0..item.elements {
            result.push(self.string_at(&mut reader)?);
        }
        if reader.position() != item.end {
            return Err(GgufError::Metadata("GGUF string array span mismatch"));
        }
        self.assert_unchanged()?;
        Ok(result)
    }

    /// `tensor_bytes(tensor)`, as a copy into `buffer` (cleared first): a positioned read for a file, a copy of the span for borrowed input. The C++ returns a span of its mapping; the zero-copy path the engine needs is phase B and out of scope for this port.
    pub fn tensor_bytes(&self, tensor: &Tensor, buffer: &mut Vec<u8>) -> Result<(), GgufError> {
        self.tensor_range_bytes(tensor, 0, tensor.bytes, buffer)
    }

    /// Positioned read of `[sub_offset, sub_offset + length)` within `tensor` into `buffer` (cleared first), allowing bounded streaming without loading an entire large tensor at once.
    pub fn tensor_range_bytes(
        &self,
        tensor: &Tensor,
        sub_offset: u64,
        length: u64,
        buffer: &mut Vec<u8>,
    ) -> Result<(), GgufError> {
        // Verify membership, not just caller-supplied offsets into a large file.
        if !self.tensors.iter().any(|item| std::ptr::eq(item, tensor)) {
            return Err(GgufError::Tensor("tensor does not belong to this GGUF"));
        }
        let sub_end = checked_add(sub_offset, length)?;
        if sub_end > tensor.bytes {
            return Err(GgufError::Tensor("tensor slice exceeds tensor byte length"));
        }
        let file_offset = checked_add(tensor.offset, sub_offset)?;
        self.assert_unchanged()?;
        buffer.clear();
        match &self.mapping {
            Some(file) => file.read_into(file_offset, length, buffer),
            None => {
                // Parsing bounded every tensor by the borrowed input's size.
                let view = usize::try_from(file_offset)
                    .ok()
                    .zip(usize::try_from(length).ok())
                    .and_then(|(start, count)| self.data.get(start..start.checked_add(count)?))
                    .ok_or(GgufError::Tensor("truncated GGUF tensor data"))?;
                buffer.extend_from_slice(view);
                Ok(())
            }
        }
    }

    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// The whole input's size (`data_.size()`), not the bytes held in memory.
    pub fn file_size(&self) -> u64 {
        self.size
    }

    /// Bytes of the input held in memory: all of borrowed input, or the header region read from a file. No C++ counterpart (it maps the file).
    pub fn loaded_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn assert_unchanged(&self) -> Result<(), GgufError> {
        self.mapping
            .iter()
            .try_for_each(|item| item.assert_unchanged())
    }

    fn memory(&self) -> Source<'_> {
        Source::Memory {
            bytes: &self.data,
            size: self.size,
        }
    }

    fn cursor(&self, start: u64) -> Cursor<'_> {
        Cursor::new(self.memory(), start)
    }

    /// `Cursor::string` with the view borrowed from `data_`, as the C++ `std::string_view` is, rather than from the cursor.
    fn string_at(&self, reader: &mut Cursor<'_>) -> Result<&[u8], GgufError> {
        let length = reader.string(HEADER_LIMIT)?.len();
        // The cursor has just read `length` bytes ending here, so the span is in `data_` and the casts are exact.
        let end = reader.position() as usize;
        end.checked_sub(length)
            .and_then(|start| self.data.get(start..end))
            .ok_or(GgufError::Header(TRUNCATED_HEADER))
    }
}

/// `ModelFiles`: a complete, consistent set of GGUF shards.
pub struct ModelFiles {
    shards: Vec<Gguf<'static>>,
    tensors: BTreeMap<Vec<u8>, TensorLocation>,
    architecture: Vec<u8>,
}

impl fmt::Debug for ModelFiles {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelFiles")
            .field("shards", &self.shards.len())
            .field("tensors", &self.tensors.len())
            .finish_non_exhaustive()
    }
}

impl ModelFiles {
    /// `ModelFiles(paths)`: opens every shard in order, then checks split metadata, architecture and tensor names across the set.
    pub fn new<P: AsRef<Path>>(paths: &[P]) -> Result<Self, GgufError> {
        if paths.is_empty() || paths.len() > 1024 {
            return Err(GgufError::Shards("expected 1 to 1024 GGUF shards"));
        }
        let mut shards = Vec::with_capacity(paths.len());
        for path in paths {
            shards.push(Gguf::from_path(path)?);
        }
        let mut tensors = BTreeMap::new();
        let mut architecture: Vec<u8> = Vec::new();
        let mut numbers = BTreeSet::new();
        let mut declared_total = 0;
        let mut split = false;
        for (s, shard) in shards.iter().enumerate() {
            let has_split = shard.find("split.count").is_some()
                || shard.find("split.no").is_some()
                || shard.find("split.tensors.count").is_some();
            if s == 0 {
                split = has_split;
            }
            if has_split != split || (!split && paths.len() != 1) {
                return Err(GgufError::Shards("inconsistent split-file metadata"));
            }
            if split {
                let count = shard.unsigned_value("split.count")?;
                let number = shard.unsigned_value("split.no")?;
                let total = shard.unsigned_value("split.tensors.count")?;
                if count != paths.len() as u64
                    || number >= count
                    || !numbers.insert(number)
                    || (s != 0 && declared_total != total)
                {
                    return Err(GgufError::Shards("incomplete or mismatched GGUF shard set"));
                }
                declared_total = total;
            }
            if shard.find("general.architecture").is_some() {
                let value = shard.string_value("general.architecture")?;
                if !architecture.is_empty() && architecture != value {
                    return Err(GgufError::Shards("shard architecture mismatch"));
                }
                architecture = value.to_vec();
            }
            for (t, tensor) in shard.tensors().iter().enumerate() {
                match tensors.entry(tensor.name.clone()) {
                    Entry::Vacant(slot) => {
                        slot.insert(TensorLocation { shard: s, index: t });
                    }
                    Entry::Occupied(_) => {
                        return Err(GgufError::Shards("duplicate tensor across GGUF shards"));
                    }
                }
            }
        }
        if architecture.is_empty() {
            return Err(GgufError::Shards("GGUF model has no architecture"));
        }
        if split && declared_total != tensors.len() as u64 {
            return Err(GgufError::Shards("GGUF split tensor count mismatch"));
        }
        Ok(Self {
            shards,
            tensors,
            architecture,
        })
    }

    pub fn shards(&self) -> &[Gguf<'static>] {
        &self.shards
    }

    /// Tensor names in byte order (`std::map<std::string, ...>`).
    pub fn tensors(&self) -> &BTreeMap<Vec<u8>, TensorLocation> {
        &self.tensors
    }

    /// `tensor_bytes(name)`, as a copy into `buffer`; see [`Gguf::tensor_bytes`].
    pub fn tensor_bytes(
        &self,
        name: impl AsRef<[u8]>,
        buffer: &mut Vec<u8>,
    ) -> Result<(), GgufError> {
        let (shard, tensor) = self.locate(name.as_ref())?;
        shard.tensor_bytes(tensor, buffer)
    }

    /// Positioned read of `[sub_offset, sub_offset + length)` within `name` into `buffer`; see [`Gguf::tensor_range_bytes`].
    pub fn tensor_range_bytes(
        &self,
        name: impl AsRef<[u8]>,
        sub_offset: u64,
        length: u64,
        buffer: &mut Vec<u8>,
    ) -> Result<(), GgufError> {
        let (shard, tensor) = self.locate(name.as_ref())?;
        shard.tensor_range_bytes(tensor, sub_offset, length, buffer)
    }

    pub fn tensor(&self, name: impl AsRef<[u8]>) -> Result<&Tensor, GgufError> {
        Ok(self.locate(name.as_ref())?.1)
    }

    pub fn architecture(&self) -> &[u8] {
        &self.architecture
    }

    pub fn assert_unchanged(&self) -> Result<(), GgufError> {
        self.shards.iter().try_for_each(Gguf::assert_unchanged)
    }

    fn locate(&self, name: &[u8]) -> Result<(&Gguf<'static>, &Tensor), GgufError> {
        let unknown = GgufError::Tensor("unknown tensor");
        let item = self.tensors.get(name).ok_or(unknown)?;
        // `new` recorded both indices, so these lookups cannot miss.
        let shard = self
            .shards
            .get(item.shard)
            .ok_or(GgufError::Tensor("unknown tensor"))?;
        let tensor = shard
            .tensors()
            .get(item.index)
            .ok_or(GgufError::Tensor("unknown tensor"))?;
        Ok((shard, tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokspur_testkit::Mt19937;

    fn header(tensors: u64, metadata: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&tensors.to_le_bytes());
        bytes.extend_from_slice(&metadata.to_le_bytes());
        bytes
    }

    #[test]
    fn valid_utf8_matches_the_standard_decoder() {
        let mut random = Mt19937::new(0x6767_7566);
        for _ in 0..20_000 {
            let length = random.next_u32() % 8;
            let bytes: Vec<u8> = (0..length)
                .map(|_| {
                    // Bias towards lead and continuation bytes.
                    let byte = random.next_u32() as u8;
                    if random.next_u32().is_multiple_of(2) {
                        0x80 | (byte & 0x7f)
                    } else {
                        byte
                    }
                })
                .collect();
            assert_eq!(
                valid_utf8(&bytes),
                std::str::from_utf8(&bytes).is_ok(),
                "{bytes:02x?}"
            );
        }
        for point in [
            0x7fu32, 0x80, 0x7ff, 0x800, 0xd7ff, 0xe000, 0xffff, 0x1_0000, 0x10_ffff,
        ] {
            let text = char::from_u32(point).unwrap().to_string();
            assert!(valid_utf8(text.as_bytes()), "{point:x}");
        }
    }

    #[test]
    fn valid_key_follows_the_cpp_grammar() {
        for key in ["a", "general.alignment", "a.b_c.9", "_", "0.0"] {
            assert!(valid_key(key.as_bytes()), "{key}");
        }
        for key in ["", ".", "a.", ".a", "a..b", "A", "a-b", "a b", "é"] {
            assert!(!valid_key(key.as_bytes()), "{key}");
        }
        assert!(!valid_key(b"a\x80"));
    }

    #[test]
    fn cursor_refuses_reads_past_the_header_limit() {
        // A cursor over a pretend 300 MiB input: the size check passes, the header limit refuses, and no byte is read.
        let bytes = [0u8; 8];
        let mut reader = Cursor::new(
            Source::Memory {
                bytes: &bytes,
                size: 300 << 20,
            },
            HEADER_LIMIT - 4,
        );
        assert_eq!(
            reader.take(5).unwrap_err().to_string(),
            "truncated or oversized GGUF header"
        );
        // `checked_add` runs before the bounds check.
        let mut reader = Cursor::new(
            Source::Memory {
                bytes: &bytes,
                size: 8,
            },
            0,
        );
        assert_eq!(
            reader.take(u64::MAX).unwrap_err().to_string(),
            "truncated or oversized GGUF header"
        );
        let mut reader = Cursor::new(
            Source::Memory {
                bytes: &bytes,
                size: 8,
            },
            1,
        );
        assert_eq!(
            reader.take(u64::MAX).unwrap_err().to_string(),
            "GGUF offset overflow"
        );
        assert_eq!(reader.position(), 1);
    }

    #[test]
    fn cursor_checks_string_limits_before_reading() {
        let mut bytes = (65u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&[b'a'; 65]);
        let mut reader = Cursor::new(
            Source::Memory {
                bytes: &bytes,
                size: bytes.len() as u64,
            },
            0,
        );
        assert_eq!(
            reader.string(64).unwrap_err().to_string(),
            "GGUF string exceeds limit"
        );
        let mut reader = Cursor::new(
            Source::Memory {
                bytes: &bytes,
                size: bytes.len() as u64,
            },
            0,
        );
        assert_eq!(reader.string(65).unwrap(), [b'a'; 65].as_slice());
    }

    #[test]
    fn parse_reads_alignment_before_the_header_is_complete() {
        let mut bytes = header(0, 1);
        bytes.extend_from_slice(&17u64.to_le_bytes());
        bytes.extend_from_slice(b"general.alignment");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.resize(64, 0);
        let gguf = Gguf::from_bytes(&bytes).unwrap();
        assert_eq!(gguf.data_offset(), 64);
        assert_eq!(gguf.unsigned_value("general.alignment").unwrap(), 64);
    }

    #[test]
    fn encoding_table_is_the_cpp_table() {
        let ids: Vec<u32> = LAYOUTS.iter().map(|item| item.id).collect();
        assert_eq!(
            ids,
            [
                0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23, 24, 25, 26, 27, 28, 30
            ]
        );
        assert_eq!(
            *encoding(23).unwrap(),
            Encoding {
                id: 23,
                name: "IQ4_XS",
                elements_per_block: 256,
                bytes_per_block: 136
            }
        );
        for id in [4, 5, 9, 15, 19, 21, 22, 29, 31, u32::MAX] {
            assert_eq!(
                encoding(id).unwrap_err().to_string(),
                format!("unsupported GGUF tensor encoding {id}")
            );
        }
    }

    #[test]
    fn every_error_displays_the_cpp_message() {
        let cases = [
            (
                GgufError::Open(io::ErrorKind::NotFound.into()),
                "cannot open weight file",
            ),
            (
                GgufError::Inspect(None),
                "weight file must be a nonempty regular file",
            ),
            (
                GgufError::Map(io::ErrorKind::UnexpectedEof.into()),
                "cannot map weight file",
            ),
            (
                GgufError::Unchanged(None),
                "weight file changed since mapping",
            ),
            (GgufError::Encoding(4), "unsupported GGUF tensor encoding 4"),
            (
                GgufError::Lookup {
                    reason: "missing GGUF metadata ",
                    key: b"split.count".to_vec(),
                },
                "missing GGUF metadata split.count",
            ),
            (
                GgufError::Lookup {
                    reason: "GGUF metadata is not a string: ",
                    key: b"general.architecture".to_vec(),
                },
                "GGUF metadata is not a string: general.architecture",
            ),
        ];
        for (error, message) in cases {
            assert_eq!(error.to_string(), message);
        }
    }

    #[test]
    fn errors_keep_their_io_source() {
        use std::error::Error;
        let error = GgufError::Open(io::Error::new(io::ErrorKind::NotFound, "/secret/path.gguf"));
        assert_eq!(error.source().unwrap().to_string(), "/secret/path.gguf");
        assert!(!format!("{error:?}").contains("secret"));
        let lookup = GgufError::Lookup {
            reason: "missing GGUF metadata ",
            key: b"secret.metadata.key".to_vec(),
        };
        assert!(!format!("{lookup:?}").contains("secret.metadata.key"));
        assert!(GgufError::Inspect(None).source().is_none());
        assert!(GgufError::Header(TRUNCATED_HEADER).source().is_none());
    }

    #[test]
    fn non_utf8_lookup_keys_print_replacement_characters() {
        // The one Display that cannot be byte-identical: the C++ appends raw key bytes. Keys in a GGUF are ASCII, so only a caller can supply one.
        let bytes = {
            let mut bytes = header(0, 0);
            bytes.resize(32, 0);
            bytes
        };
        let gguf = Gguf::from_bytes(&bytes).unwrap();
        assert_eq!(
            gguf.unsigned_value(b"bad\xff").unwrap_err().to_string(),
            "missing GGUF metadata bad\u{fffd}"
        );
    }
}
