//! Dispatch traces: one JSON record per line, as written by the C++ host's dispatch-graph build (`MetalGraph::configure_dispatch_trace`) and, later, by the phase B encoder. Only sizes, ranges and SHA-256 digests appear; never buffer bytes.

use std::collections::BTreeMap;
use std::fmt;

use tokspur_json::{Json, JsonError, JsonLimits, parse_json_limited};

/// Why a trace could not be read.
pub enum TraceError {
    /// A line is not valid JSON within the line limits.
    Json { line: usize, source: JsonError },
    /// A record is missing a field, has the wrong kind or breaks the begin/dispatch/submit structure.
    Malformed { line: usize, reason: &'static str },
}

impl fmt::Debug for TraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json { line, source } => formatter
                .debug_struct("Json")
                .field("line", line)
                .field("source", source)
                .finish(),
            Self::Malformed { line, reason } => formatter
                .debug_struct("Malformed")
                .field("line", line)
                .field("reason", reason)
                .finish(),
        }
    }
}

impl fmt::Display for TraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json { line, .. } => write!(formatter, "trace line {line} is not valid JSON"),
            Self::Malformed { line, reason } => write!(formatter, "trace line {line}: {reason}"),
        }
    }
}

impl std::error::Error for TraceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json { source, .. } => Some(source),
            Self::Malformed { .. } => None,
        }
    }
}

/// One allocation, in slot order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocation {
    pub slot: u64,
    pub label: String,
    pub kind: String,
    pub bytes: u64,
    pub immutable: bool,
    /// Initial contents; absent for mapped weights.
    pub sha256: Option<String>,
}

/// How a kernel binding touches its range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Access {
    Read,
    Write,
}

/// A byte range of one allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Range {
    pub slot: u64,
    pub offset: u64,
    pub bytes: u64,
}

impl Range {
    pub fn overlaps(&self, other: &Range) -> bool {
        self.slot == other.slot
            && self.offset < other.offset.saturating_add(other.bytes)
            && other.offset < self.offset.saturating_add(self.bytes)
    }
}

/// What a binding index refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Target {
    /// The shared error flag every kernel may OR into (`buffer(7)`).
    ErrorFlag,
    Range(Range, Access),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Binding {
    pub index: u64,
    pub target: Target,
}

/// One `setBytes`: its index, length and SHA-256.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Constant {
    pub index: u64,
    pub bytes: u64,
    pub sha256: String,
}

/// One compute dispatch with every argument that affects it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dispatch {
    /// `function:constant[:threads]`, the host's pipeline cache key.
    pub pipeline: String,
    pub aligned: bool,
    pub grid: [u64; 3],
    pub group: [u64; 3],
    /// Sorted by index; binding order within a dispatch is not observable.
    pub bindings: Vec<Binding>,
    /// Sorted by index.
    pub constants: Vec<Constant>,
}

impl Dispatch {
    /// The kernel function name, the pipeline key before its first `:`.
    pub fn function(&self) -> &str {
        self.pipeline.split(':').next().unwrap_or("")
    }

    /// A canonical text of every argument, for matching dispatches.
    pub fn signature(&self) -> String {
        let mut text = format!(
            "{}|{}|{:?}|{:?}",
            self.pipeline, self.aligned, self.grid, self.group
        );
        for binding in &self.bindings {
            match binding.target {
                Target::ErrorFlag => text.push_str(&format!("|b{}=flag", binding.index)),
                Target::Range(range, access) => text.push_str(&format!(
                    "|b{}={}:{}+{}:{:?}",
                    binding.index, range.slot, range.offset, range.bytes, access
                )),
            }
        }
        for constant in &self.constants {
            text.push_str(&format!(
                "|c{}={}:{}",
                constant.index, constant.bytes, constant.sha256
            ));
        }
        text
    }
}

/// Inside a batch: a dispatch, or a barrier ordering everything before it ahead of everything after it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    Dispatch(Dispatch),
    Barrier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Completed; digests of every live mutable allocation afterwards.
    Submitted {
        dispatches: u64,
        states: Vec<(u64, String)>,
    },
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    pub items: Vec<Item>,
    pub outcome: Outcome,
}

impl Batch {
    pub fn dispatches(&self) -> impl Iterator<Item = &Dispatch> {
        self.items.iter().filter_map(|item| match item {
            Item::Dispatch(dispatch) => Some(dispatch),
            Item::Barrier => None,
        })
    }

    pub fn barriers(&self) -> usize {
        self.items
            .iter()
            .filter(|item| matches!(item, Item::Barrier))
            .count()
    }
}

/// Host-side events between batches, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Allocation(Allocation),
    Release(u64),
    HostWrite { range: Range, sha256: String },
    HostRead { range: Range, sha256: String },
    HostCopy(Vec<(Range, Range, String)>),
    Batch(Batch),
}

/// Reflection of one pipeline's buffer bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pipeline {
    pub function: String,
    /// `(index, access)` with access `read`, `write` or `read_write`.
    pub bindings: Vec<(u64, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trace {
    pub producer: String,
    pub pipelines: BTreeMap<String, Pipeline>,
    pub events: Vec<Event>,
}

impl Trace {
    pub fn batches(&self) -> impl Iterator<Item = &Batch> {
        self.events.iter().filter_map(|event| match event {
            Event::Batch(batch) => Some(batch),
            _ => None,
        })
    }
}

const LINE_LIMITS: JsonLimits = JsonLimits::new(4 * 1024 * 1024, 1 << 20, 16);

struct Record<'a> {
    line: usize,
    json: &'a Json,
}

impl<'a> Record<'a> {
    fn malformed(&self, reason: &'static str) -> TraceError {
        TraceError::Malformed {
            line: self.line,
            reason,
        }
    }

    fn field(&self, name: &'static str) -> Result<&'a Json, TraceError> {
        match self.json.find(name) {
            Ok(Some(value)) => Ok(value),
            Ok(None) => Err(self.malformed("missing field")),
            Err(_) => Err(self.malformed("record is not an object")),
        }
    }

    fn text(&self, name: &'static str) -> Result<String, TraceError> {
        let value = self.field(name)?;
        let bytes = value
            .string()
            .map_err(|_| self.malformed("expected a string field"))?;
        String::from_utf8(bytes.to_vec()).map_err(|_| self.malformed("expected UTF-8 text"))
    }

    fn number(&self, name: &'static str) -> Result<u64, TraceError> {
        number(self.field(name)?).ok_or_else(|| self.malformed("expected an unsigned integer"))
    }

    fn flag(&self, name: &'static str) -> Result<bool, TraceError> {
        self.field(name)?
            .boolean()
            .map_err(|_| self.malformed("expected a boolean field"))
    }

    fn range(&self) -> Result<Range, TraceError> {
        Ok(Range {
            slot: self.number("slot")?,
            offset: self.number("offset")?,
            bytes: self.number("bytes")?,
        })
    }

    fn triple(&self, name: &'static str) -> Result<[u64; 3], TraceError> {
        let values = self
            .field(name)?
            .array()
            .map_err(|_| self.malformed("expected a size triple"))?;
        let mut size = [0; 3];
        if values.len() != 3 {
            return Err(self.malformed("expected a size triple"));
        }
        for (slot, value) in size.iter_mut().zip(values) {
            *slot = number(value).ok_or_else(|| self.malformed("expected a size triple"))?;
        }
        Ok(size)
    }

    fn nested(&self, json: &'a Json) -> Record<'a> {
        Record {
            line: self.line,
            json,
        }
    }

    fn list(&self, name: &'static str) -> Result<&'a [Json], TraceError> {
        self.field(name)?
            .array()
            .map(Vec::as_slice)
            .map_err(|_| self.malformed("expected an array field"))
    }
}

fn number(value: &Json) -> Option<u64> {
    value.unsigned_integer().ok()
}

fn binding(record: &Record<'_>) -> Result<Binding, TraceError> {
    let index = record.number("index")?;
    if let Ok(Some(buffer)) = record.json.find("buffer") {
        return match buffer.string() {
            Ok(b"error_flag") => Ok(Binding {
                index,
                target: Target::ErrorFlag,
            }),
            _ => Err(record.malformed("unknown named buffer")),
        };
    }
    let access = match record.text("access")?.as_str() {
        "read" => Access::Read,
        "write" => Access::Write,
        _ => return Err(record.malformed("unknown binding access")),
    };
    Ok(Binding {
        index,
        target: Target::Range(record.range()?, access),
    })
}

fn dispatch(record: &Record<'_>) -> Result<Dispatch, TraceError> {
    let mut bindings = record
        .list("bindings")?
        .iter()
        .map(|item| binding(&record.nested(item)))
        .collect::<Result<Vec<_>, _>>()?;
    bindings.sort();
    if bindings
        .windows(2)
        .any(|pair| pair[0].index == pair[1].index)
    {
        return Err(record.malformed("binding index bound twice"));
    }
    let mut constants = record
        .list("constants")?
        .iter()
        .map(|item| {
            let item = record.nested(item);
            Ok(Constant {
                index: item.number("index")?,
                bytes: item.number("bytes")?,
                sha256: item.text("sha256")?,
            })
        })
        .collect::<Result<Vec<_>, TraceError>>()?;
    constants.sort();
    if constants
        .windows(2)
        .any(|pair| pair[0].index == pair[1].index)
    {
        return Err(record.malformed("constant index set twice"));
    }
    Ok(Dispatch {
        pipeline: record.text("pipeline")?,
        aligned: record.flag("aligned")?,
        grid: record.triple("grid")?,
        group: record.triple("group")?,
        bindings,
        constants,
    })
}

/// Parses a whole trace. The first record must be the `trace` header.
pub fn parse_trace(text: &[u8]) -> Result<Trace, TraceError> {
    let mut trace = Trace {
        producer: String::new(),
        pipelines: BTreeMap::new(),
        events: Vec::new(),
    };
    let mut open: Option<Vec<Item>> = None;
    let mut header = false;
    for (index, line) in text.split(|&byte| byte == b'\n').enumerate() {
        if line.is_empty() {
            continue;
        }
        let number = index + 1;
        let json = parse_json_limited(line, LINE_LIMITS).map_err(|source| TraceError::Json {
            line: number,
            source,
        })?;
        let record = Record {
            line: number,
            json: &json,
        };
        let event = record.text("event")?;
        if !header {
            if event != "trace" || record.number("schema")? != 1 {
                return Err(record.malformed("expected a schema 1 trace header first"));
            }
            trace.producer = record.text("producer")?;
            header = true;
            continue;
        }
        let in_batch = open.is_some();
        match (event.as_str(), in_batch) {
            ("pipeline", _) => {
                let bindings = record
                    .list("bindings")?
                    .iter()
                    .map(|item| {
                        let item = record.nested(item);
                        Ok((item.number("index")?, item.text("access")?))
                    })
                    .collect::<Result<Vec<_>, TraceError>>()?;
                trace.pipelines.insert(
                    record.text("key")?,
                    Pipeline {
                        function: record.text("function")?,
                        bindings,
                    },
                );
            }
            ("begin", false) => open = Some(Vec::new()),
            ("dispatch", true) => {
                let item = Item::Dispatch(dispatch(&record)?);
                open.get_or_insert_with(Vec::new).push(item);
            }
            ("barrier", true) => open.get_or_insert_with(Vec::new).push(Item::Barrier),
            ("submit", true) => {
                let items = open.take().unwrap_or_default();
                trace.events.push(Event::Batch(Batch {
                    items,
                    outcome: Outcome::Submitted {
                        dispatches: record.number("dispatches")?,
                        states: Vec::new(),
                    },
                }));
            }
            ("abort", true) => {
                let items = open.take().unwrap_or_default();
                trace.events.push(Event::Batch(Batch {
                    items,
                    outcome: Outcome::Aborted,
                }));
            }
            ("state", false) => {
                let entry = (record.number("slot")?, record.text("sha256")?);
                match trace.events.last_mut() {
                    Some(Event::Batch(Batch {
                        outcome: Outcome::Submitted { states, .. },
                        ..
                    })) => states.push(entry),
                    _ => return Err(record.malformed("state digest outside a submitted batch")),
                }
            }
            ("allocation", false) => {
                let sha256 = match record.field("sha256")? {
                    value if value.is_null() => None,
                    _ => Some(record.text("sha256")?),
                };
                trace.events.push(Event::Allocation(Allocation {
                    slot: record.number("slot")?,
                    label: record.text("label")?,
                    kind: record.text("kind")?,
                    bytes: record.number("bytes")?,
                    immutable: record.flag("immutable")?,
                    sha256,
                }));
            }
            ("release", false) => trace.events.push(Event::Release(record.number("slot")?)),
            ("host_write", false) => trace.events.push(Event::HostWrite {
                range: record.range()?,
                sha256: record.text("sha256")?,
            }),
            ("host_byte_write", false) => {
                let storage = record.nested(record.field("storage")?).range()?;
                let offset = record.number("offset")?;
                let bytes = record.number("bytes")?;
                if storage.offset.checked_add(storage.bytes).is_none()
                    || offset > storage.bytes
                    || bytes > storage.bytes - offset
                {
                    return Err(record.malformed("host byte write exceeds storage"));
                }
                let offset = storage
                    .offset
                    .checked_add(offset)
                    .ok_or_else(|| record.malformed("host byte write offset overflow"))?;
                // Byte uploads use a relative offset within a checked view.
                // Canonicalise to the existing host event so equivalence still
                // compares the exact modified range and its payload digest.
                trace.events.push(Event::HostWrite {
                    range: Range {
                        slot: storage.slot,
                        offset,
                        bytes,
                    },
                    sha256: record.text("sha256")?,
                });
            }
            ("host_read", false) => trace.events.push(Event::HostRead {
                range: record.range()?,
                sha256: record.text("sha256")?,
            }),
            ("host_copy", false) => {
                let copies = record
                    .list("copies")?
                    .iter()
                    .map(|item| {
                        let item = record.nested(item);
                        let source = item.nested(item.field("source")?).range()?;
                        let destination = item.nested(item.field("destination")?).range()?;
                        Ok((source, destination, item.text("sha256")?))
                    })
                    .collect::<Result<Vec<_>, TraceError>>()?;
                trace.events.push(Event::HostCopy(copies));
            }
            _ => return Err(record.malformed("unexpected record here")),
        }
    }
    if !header {
        return Err(TraceError::Malformed {
            line: 0,
            reason: "empty trace",
        });
    }
    if open.is_some() {
        return Err(TraceError::Malformed {
            line: 0,
            reason: "trace ends inside a batch",
        });
    }
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = r#"{"event":"trace","schema":1,"producer":"test","device":"none"}"#;

    #[test]
    fn byte_uploads_preserve_effective_range_and_digest_equivalence() {
        let reference = parse_trace(
            [
                HEADER,
                r#"{"event":"host_write","slot":3,"offset":18,"bytes":3,"sha256":"aa"}"#,
                r#"{"event":"host_read","slot":3,"offset":18,"bytes":3,"sha256":"aa"}"#,
            ]
            .join("\n")
            .as_bytes(),
        )
        .unwrap();
        let upload = r#"{"event":"host_byte_write","storage":{"slot":3,"offset":16,"bytes":16},"offset":2,"bytes":3,"sha256":"aa"}"#;
        let readback = r#"{"event":"host_read","slot":3,"offset":18,"bytes":3,"sha256":"aa"}"#;
        let candidate = parse_trace([HEADER, upload, readback].join("\n").as_bytes()).unwrap();
        assert_eq!(candidate.events, reference.events);
        assert!(crate::check::check(&reference, &candidate).passed());

        for changed in [
            upload.replace(r#""offset":2"#, r#""offset":4"#),
            upload.replace(r#""sha256":"aa""#, r#""sha256":"bb""#),
        ] {
            let candidate =
                parse_trace([HEADER, &changed, readback].join("\n").as_bytes()).unwrap();
            assert!(!crate::check::check(&reference, &candidate).passed());
        }
        let changed = readback.replace(r#""sha256":"aa""#, r#""sha256":"bb""#);
        let candidate = parse_trace([HEADER, upload, &changed].join("\n").as_bytes()).unwrap();
        assert!(!crate::check::check(&reference, &candidate).passed());
    }

    #[test]
    fn byte_uploads_refuse_out_of_view_and_overflowing_ranges() {
        for upload in [
            r#"{"event":"host_byte_write","storage":{"slot":0,"offset":16,"bytes":8},"offset":9,"bytes":0,"sha256":"aa"}"#,
            r#"{"event":"host_byte_write","storage":{"slot":0,"offset":16,"bytes":8},"offset":7,"bytes":2,"sha256":"aa"}"#,
            r#"{"event":"host_byte_write","storage":{"slot":0,"offset":18446744073709551615,"bytes":1},"offset":0,"bytes":1,"sha256":"aa"}"#,
        ] {
            let error = parse_trace([HEADER, upload].join("\n").as_bytes()).unwrap_err();
            assert_eq!(
                error.to_string(),
                "trace line 2: host byte write exceeds storage"
            );
        }
    }

    #[test]
    fn parses_a_batch_with_sorted_bindings_and_states() {
        // Arrange
        let text = [
            HEADER,
            r#"{"event":"allocation","slot":0,"label":"x","kind":"zeros","bytes":16,"immutable":false,"sha256":"aa"}"#,
            r#"{"event":"begin","batch":0}"#,
            r#"{"event":"dispatch","pipeline":"k:4294967295","aligned":true,"grid":[1,1,1],"group":[128,1,1],"bindings":[{"index":7,"buffer":"error_flag"},{"index":0,"slot":0,"offset":0,"bytes":16,"access":"write"}],"constants":[{"index":3,"bytes":4,"sha256":"bb"}]}"#,
            r#"{"event":"barrier","scope":"buffers"}"#,
            r#"{"event":"submit","batch":0,"dispatches":1}"#,
            r#"{"event":"state","slot":0,"sha256":"cc"}"#,
        ]
        .join("\n");

        // Act
        let trace = parse_trace(text.as_bytes()).unwrap();

        // Assert
        let batch = trace.batches().next().unwrap();
        let dispatch = batch.dispatches().next().unwrap();
        assert_eq!(dispatch.function(), "k");
        assert_eq!(dispatch.bindings[0].index, 0);
        assert_eq!(dispatch.bindings[1].target, Target::ErrorFlag);
        assert_eq!(batch.barriers(), 1);
        assert_eq!(
            batch.outcome,
            Outcome::Submitted {
                dispatches: 1,
                states: vec![(0, "cc".to_owned())]
            }
        );
    }

    #[test]
    fn refuses_a_dispatch_outside_a_batch() {
        // Arrange
        let text = [
            HEADER,
            r#"{"event":"dispatch","pipeline":"k","aligned":true,"grid":[1,1,1],"group":[1,1,1],"bindings":[],"constants":[]}"#,
        ]
        .join("\n");

        // Act
        let error = parse_trace(text.as_bytes()).unwrap_err();

        // Assert
        assert_eq!(error.to_string(), "trace line 2: unexpected record here");
    }

    #[test]
    fn refuses_a_trace_without_its_header() {
        // Act
        let error = parse_trace(br#"{"event":"begin","batch":0}"#).unwrap_err();

        // Assert
        assert_eq!(
            error.to_string(),
            "trace line 1: expected a schema 1 trace header first"
        );
    }
}
