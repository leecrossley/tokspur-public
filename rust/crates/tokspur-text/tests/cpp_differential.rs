//! Differential check of this port against the real C++.
//!
//! `rust/tools/diff/text.sh` compiles `rust/tools/diff/text_diff.cpp` together with `native/src/unicode.cpp`, `tokenizer.cpp` and `gguf.cpp`, unchanged, then runs this ignored test with `TOKSPUR_TEXT_DIFF_DRIVER` (the driver) and `TOKSPUR_TEXT_DIFF_DIR` (an output directory under `build/`). The test generates a deterministic case file (`Mt19937`, fixed seeds), has the C++ driver and this crate evaluate the same file, and requires every result line, error messages included, to be byte-identical. The case language is described at the top of `text_diff.cpp`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use tokspur_testkit::Mt19937;
use tokspur_text::sha256::{self, Sha256};
use tokspur_text::tokenizer::{
    AddedToken, BpeData, ByteBpe, TextProcessor, byte_level_decode, byte_level_encode,
    replace_invalid_utf8,
};
use tokspur_text::unicode::{NormalizeError, normalize_nfc9, scalar_utf8, unicode_scalars};

type Harness<T> = Result<T, String>;

const INCLUDE: &str = include_str!("../../../../native/data/unicode9.inc");

// Case encoding, shared by the generator and the Rust evaluator.

fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    output
}

fn x(bytes: &[u8]) -> String {
    format!("x{}", to_hex(bytes))
}

fn repeat(unit: &[u8], count: usize) -> String {
    format!("x{}*{count}", to_hex(unit))
}

fn ids_text(values: &[u32]) -> String {
    let mut output = String::from("i");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}

fn decimal(text: &str) -> Harness<u64> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("invalid decimal field {text:?}"));
    }
    text.parse()
        .map_err(|_| format!("invalid decimal field {text:?}"))
}

fn count(text: &str) -> Harness<usize> {
    usize::try_from(decimal(text)?).map_err(|_| format!("count {text} exceeds usize"))
}

fn decimals(body: &str) -> Harness<Vec<u64>> {
    if body.is_empty() {
        return Ok(Vec::new());
    }
    body.split(',').map(decimal).collect()
}

/// `"<prefix><body>[*<repeat>]"`.
fn repeated(field: &str, prefix: char) -> Harness<(&str, usize)> {
    let body = field
        .strip_prefix(prefix)
        .ok_or_else(|| format!("expected a {prefix} field, found {field:?}"))?;
    match body.split_once('*') {
        Some((unit, times)) => Ok((unit, count(times)?)),
        None => Ok((body, 1)),
    }
}

fn hex_digit(digit: u8) -> Harness<u8> {
    match digit {
        b'0'..=b'9' => Ok(digit - b'0'),
        b'a'..=b'f' => Ok(digit - b'a' + 10),
        _ => Err(format!("invalid hexadecimal digit {digit:#04x}")),
    }
}

fn bytes_field(field: &str) -> Harness<Vec<u8>> {
    let mut output = Vec::new();
    for segment in field.split('+') {
        let (body, times) = repeated(segment, 'x')?;
        if body.len() % 2 != 0 {
            return Err("odd hexadecimal length".into());
        }
        let mut unit = Vec::with_capacity(body.len() / 2);
        for pair in body.as_bytes().chunks(2) {
            unit.push(hex_digit(pair[0])? * 16 + hex_digit(pair[1])?);
        }
        output.extend_from_slice(&unit.repeat(times));
    }
    Ok(output)
}

fn ids_field(field: &str) -> Harness<Vec<u32>> {
    let mut output = Vec::new();
    for segment in field.split('+') {
        let (body, times) = repeated(segment, 'i')?;
        let mut unit = Vec::new();
        for value in decimals(body)? {
            unit.push(u32::try_from(value).map_err(|_| "token ID exceeds uint32_t".to_string())?);
        }
        output.extend_from_slice(&unit.repeat(times));
    }
    Ok(output)
}

fn flag(field: &str) -> Harness<bool> {
    match field {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("invalid flag {field:?}")),
    }
}

fn want(fields: &[&str], expected: usize) -> Harness<()> {
    if fields.len() == expected {
        Ok(())
    } else {
        Err(format!("wrong number of fields in {:?}", fields.first()))
    }
}

// The driver's test processors, implemented identically.

enum Processor {
    Identity,
    Bytes,
    Split,
    Empties,
    Copies(usize),
    Grow(usize),
    Refuse(u8),
    Nfc,
}

#[derive(Debug)]
enum ProcessorError {
    Refuse,
    Normalize(NormalizeError),
}

impl fmt::Display for ProcessorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refuse => formatter.write_str("text processor refused segment"),
            Self::Normalize(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ProcessorError {}

impl Processor {
    fn parse(spec: &str) -> Harness<Self> {
        Ok(match spec {
            "identity" => Self::Identity,
            "bytes" => Self::Bytes,
            "split" => Self::Split,
            "empties" => Self::Empties,
            "nfc" => Self::Nfc,
            _ => {
                if let Some(times) = spec.strip_prefix("copies") {
                    Self::Copies(count(times)?)
                } else if let Some(times) = spec.strip_prefix("grow") {
                    Self::Grow(count(times)?)
                } else if let Some(byte) = spec.strip_prefix("refuse")
                    && byte.len() == 2
                {
                    let byte = byte.as_bytes();
                    Self::Refuse(hex_digit(byte[0])? * 16 + hex_digit(byte[1])?)
                } else {
                    return Err(format!("unknown processor {spec:?}"));
                }
            }
        })
    }
}

impl TextProcessor for Processor {
    type Error = ProcessorError;

    fn pieces(&self, segment: &[u8]) -> Result<Vec<Vec<u8>>, ProcessorError> {
        Ok(match self {
            Self::Identity => vec![segment.to_vec()],
            Self::Bytes => segment.iter().map(|&byte| vec![byte]).collect(),
            Self::Split => {
                // A new piece starts at every space after the first byte.
                let mut output = Vec::new();
                let mut start = 0;
                for i in 1..segment.len() {
                    if segment[i] == b' ' {
                        output.push(segment[start..i].to_vec());
                        start = i;
                    }
                }
                output.push(segment[start..].to_vec());
                output
            }
            Self::Empties => vec![Vec::new(), segment.to_vec(), Vec::new()],
            Self::Copies(times) => vec![segment.to_vec(); *times],
            Self::Grow(times) => vec![segment.repeat(*times)],
            Self::Refuse(byte) => {
                if segment.contains(byte) {
                    return Err(ProcessorError::Refuse);
                }
                vec![segment.to_vec()]
            }
            Self::Nfc => vec![normalize_nfc9(segment).map_err(ProcessorError::Normalize)?],
        })
    }
}

// The Rust side of the driver.

fn outcome<E: fmt::Display>(result: Result<String, E>) -> String {
    match result {
        Ok(payload) => format!("ok {payload}"),
        Err(error) => format!("err {error}"),
    }
}

fn sha(bytes: &[u8], chunks: &str) -> Harness<String> {
    let sizes = chunks.strip_prefix('c').ok_or("expected chunk sizes")?;
    let once = sha256::sha256(bytes);
    let mut context = Sha256::new();
    let mut offset = 0;
    for size in decimals(sizes)? {
        let take = usize::try_from(size)
            .unwrap_or(usize::MAX)
            .min(bytes.len() - offset);
        context.update(&bytes[offset..offset + take]);
        offset += take;
    }
    context.update(&bytes[offset..]);
    Ok(format!(
        "ok {} {}",
        sha256::hex(&once),
        sha256::hex(&context.finalize())
    ))
}

fn evaluate(fields: &[&str], model: Option<&ByteBpe>) -> Harness<String> {
    Ok(match fields[0] {
        "nfc" => {
            want(fields, 2)?;
            outcome(normalize_nfc9(&bytes_field(fields[1])?).map(|output| x(&output)))
        }
        "scalars" => {
            want(fields, 2)?;
            outcome(unicode_scalars(&bytes_field(fields[1])?).map(|points| ids_text(&points)))
        }
        "scalar" => {
            want(fields, 2)?;
            let value =
                u32::try_from(decimal(fields[1])?).map_err(|_| "scalar exceeds uint32_t")?;
            outcome(scalar_utf8(value).map(|output| x(&output)))
        }
        "ble" => {
            want(fields, 2)?;
            format!("ok {}", x(&byte_level_encode(&bytes_field(fields[1])?)))
        }
        "bld" => {
            want(fields, 2)?;
            outcome(byte_level_decode(&bytes_field(fields[1])?).map(|output| x(&output)))
        }
        "riu" => {
            want(fields, 2)?;
            format!("ok {}", x(&replace_invalid_utf8(&bytes_field(fields[1])?)))
        }
        "sha" => {
            want(fields, 3)?;
            sha(&bytes_field(fields[1])?, fields[2])?
        }
        "piece" | "encode" | "decode" | "text" => {
            let Some(model) = model else {
                return Ok("nomodel".into());
            };
            match fields[0] {
                "piece" => {
                    want(fields, 2)?;
                    outcome(
                        model
                            .encode_piece(&bytes_field(fields[1])?)
                            .map(|ids| ids_text(&ids)),
                    )
                }
                "encode" => {
                    want(fields, 3)?;
                    let processor = Processor::parse(fields[1])?;
                    let text = bytes_field(fields[2])?;
                    outcome(model.encode(&text, &processor).map(|ids| ids_text(&ids)))
                }
                operation => {
                    want(fields, 3)?;
                    let skip = flag(fields[1])?;
                    let ids = ids_field(fields[2])?;
                    let decoded = if operation == "decode" {
                        model.decode_bytes(&ids, skip)
                    } else {
                        model.decode_text(&ids, skip)
                    };
                    outcome(decoded.map(|output| x(&output)))
                }
            }
        }
        other => return Err(format!("unknown operation {other:?}")),
    })
}

/// Evaluates a case file the way `text_diff.cpp`'s `main` does.
fn evaluate_file(cases: &str) -> Harness<String> {
    let mut output = String::new();
    // `std::getline` lines: a final newline does not start another line.
    if cases.is_empty() {
        return Ok(output);
    }
    let body = cases.strip_suffix('\n').unwrap_or(cases);
    let mut lines = body.split('\n');
    let mut model: Option<ByteBpe> = None;
    while let Some(line) = lines.next() {
        let fields: Vec<&str> = line.split(' ').collect();
        if fields[0] != "model" {
            output.push_str(&evaluate(&fields, model.as_ref())?);
            output.push('\n');
            continue;
        }
        want(&fields, 4)?;
        let (vocabulary, merges, added) = (count(fields[1])?, count(fields[2])?, count(fields[3])?);
        let mut next = || lines.next().ok_or("truncated model block");
        let mut data = BpeData::default();
        for _ in 0..vocabulary {
            data.vocabulary.push(bytes_field(next()?)?);
        }
        for _ in 0..merges {
            let parts: Vec<&str> = next()?.split(' ').collect();
            want(&parts, 2)?;
            data.merges
                .push((bytes_field(parts[0])?, bytes_field(parts[1])?));
        }
        for _ in 0..added {
            let parts: Vec<&str> = next()?.split(' ').collect();
            want(&parts, 3)?;
            let id = u32::try_from(decimal(parts[0])?).map_err(|_| "token ID exceeds uint32_t")?;
            let content = bytes_field(parts[2])?;
            data.added.push(AddedToken {
                id,
                content,
                special: flag(parts[1])?,
            });
        }
        let result = match ByteBpe::new(data) {
            Ok(built) => {
                let line = format!("ok {} {}", built.vocabulary_size(), built.merge_count());
                model = Some(built);
                line
            }
            Err(error) => format!("err {error}"),
        };
        output.push_str(&result);
        output.push('\n');
    }
    Ok(output)
}

// Deterministic case generation.

/// Scalars drawn from the Unicode 9 tables, read from the C++ include.
struct Pools {
    marks: Vec<(u32, u32)>,
    decomposable: Vec<u32>,
    parts: Vec<u32>,
    pairs: Vec<(u32, u32)>,
    composites: Vec<u32>,
}

fn pools() -> Harness<Pools> {
    let mut pools = Pools {
        marks: Vec::new(),
        decomposable: Vec::new(),
        parts: Vec::new(),
        pairs: Vec::new(),
        composites: Vec::new(),
    };
    let mut table = "";
    for line in INCLUDE.lines() {
        if let Some(declaration) = line.strip_prefix("inline constexpr ") {
            table = declaration;
            continue;
        }
        if line == "};" {
            table = "";
            continue;
        }
        let Some(entry) = line
            .strip_prefix("    {")
            .and_then(|rest| rest.strip_suffix("},"))
        else {
            continue;
        };
        let mut numbers = Vec::new();
        for field in entry.split(", ") {
            let value = match field.strip_prefix("0x") {
                Some(digits) => u64::from_str_radix(digits.trim_end_matches("ULL"), 16),
                None => field.parse(),
            };
            numbers.push(value.map_err(|_| format!("bad include field {field:?}"))?);
        }
        let point = |index: usize| -> Harness<u32> {
            u32::try_from(numbers[index] & 0x1f_ffff).map_err(|_| "bad scalar".to_string())
        };
        if table.starts_with("ClassRange") {
            pools.marks.push((point(0)?, point(1)?));
        } else if table.starts_with("Decomposition") {
            pools.decomposable.push(point(0)?);
            pools.parts.push(point(1)?);
            if numbers[2] != 0 {
                pools.parts.push(point(2)?);
            }
        } else if table.starts_with("Composition") {
            let pair = numbers[0];
            let first = u32::try_from(pair >> 21).map_err(|_| "bad pair".to_string())?;
            pools.pairs.push((first, point(0)?));
            pools.composites.push(point(1)?);
        }
    }
    if pools.marks.len() != 333 || pools.decomposable.len() != 2060 || pools.pairs.len() != 940 {
        return Err("unexpected unicode9.inc table sizes".into());
    }
    Ok(pools)
}

/// Scalars the C++ tests and UAX #15 single out: exclusions, singletons, non-starter decompositions, Indic and Kana compositions with starter seconds, astral decompositions and scalar-range edges.
const TRICKY: &[u32] = &[
    0x0340, 0x0341, 0x0343, 0x0344, 0x0374, 0x037e, 0x0387, 0x0958, 0x095f, 0x09dc, 0x0a33, 0x0f43,
    0x0f73, 0x0f75, 0x0f77, 0x0f79, 0x0f81, 0x1f71, 0x1fee, 0x2126, 0x212a, 0x212b, 0x2adc, 0xfb1d,
    0xfb1f, 0xfb2a, 0xfb4e, 0xf900, 0xfa10, 0x2f800, 0x2fa1d, 0x1d15e, 0x1d160, 0x1d164, 0x1d1bb,
    0x1d1c0, 0x0eb8, 0x0eba, 0x10f48, 0x10f4b, 0x0b3e, 0x0b47, 0x0b4b, 0x0cc2, 0x0cc6, 0x0ccb,
    0x0cd5, 0x0dcf, 0x0dd9, 0x0dda, 0x1025, 0x1026, 0x102e, 0x1b05, 0x1b06, 0x1b35, 0x3099, 0x309a,
    0x304b, 0x30ab, 0x11127, 0x11131, 0x11132, 0x1133e, 0x11347, 0x1134b, 0x114b9, 0x114ba,
    0x114bb, 0x115af, 0x115b8, 0x115ba, 0x0323, 0x0307, 0x1e0a, 0x1e9b, 0x1ea1, 0x00c5, 0x0345,
    0x0313, 0x0301, 0x0300, 0x0308, 0x05b0, 0x05bc, 0x0e48, 0xfffd, 0xfeff, 0xe000, 0xd7ff,
    0x10ffff, 0x0000, 0x007f, 0x0080, 0x07ff, 0x0800, 0xffff, 0x10000,
];

struct Generator {
    random: Mt19937,
    lines: Vec<String>,
    ops: Vec<&'static str>,
}

impl Generator {
    fn new(seed: u32) -> Self {
        Self {
            random: Mt19937::new(seed),
            lines: Vec::new(),
            ops: Vec::new(),
        }
    }

    fn below(&mut self, bound: usize) -> usize {
        self.random.next_u32() as usize % bound
    }

    fn one_in(&mut self, odds: usize) -> bool {
        self.below(odds) == 0
    }

    fn byte(&mut self) -> u8 {
        self.random.next_u32() as u8
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }

    fn case(&mut self, op: &'static str, line: String) {
        self.lines.push(line);
        self.ops.push(op);
    }

    fn model(&mut self, spec: &ModelSpec) {
        self.lines.push(format!(
            "model {} {} {}",
            spec.vocabulary.len(),
            spec.merges.len(),
            spec.added.len()
        ));
        for token in &spec.vocabulary {
            self.lines.push(x(token));
        }
        for (left, right) in &spec.merges {
            self.lines.push(format!("{} {}", x(left), x(right)));
        }
        for (id, special, content) in &spec.added {
            self.lines
                .push(format!("{id} {} {}", u8::from(*special), x(content)));
        }
        self.ops.push("model");
    }

    fn mark(&mut self, pools: &Pools, out: &mut Vec<u32>) {
        let (first, last) = self.pick(&pools.marks);
        out.push(first + self.below((last - first + 1) as usize) as u32);
    }

    fn hangul(&mut self, out: &mut Vec<u32>) {
        match self.below(6) {
            0 => out.push(0x1100 + self.below(21) as u32),
            1 => out.push(0x1160 + self.below(23) as u32),
            2 => out.push(0x11a7 + self.below(29) as u32),
            3 => out.push(0xac00 + 28 * self.below(399) as u32),
            4 => out.push(0xac00 + self.below(11_172 + 2) as u32),
            _ => {
                out.push(0x1100 + self.below(19) as u32);
                out.push(0x1161 + self.below(21) as u32);
                if self.one_in(2) {
                    out.push(0x11a8 + self.below(27) as u32);
                }
            }
        }
    }

    fn scalar(&mut self, pools: &Pools, out: &mut Vec<u32>) {
        match self.below(22) {
            0..=3 => out.push(self.below(128) as u32),
            4..=6 => self.mark(pools, out),
            7..=8 => out.push(self.pick(&pools.decomposable)),
            9..=10 => {
                let (first, second) = self.pick(&pools.pairs);
                out.push(first);
                if self.one_in(3) {
                    self.mark(pools, out);
                }
                out.push(second);
            }
            11..=12 => self.hangul(out),
            13 => out.push(self.pick(&pools.composites)),
            14 => out.push(self.pick(&pools.parts)),
            15..=16 => out.push(self.pick(TRICKY)),
            17 => {
                let base = match self.below(3) {
                    0 => self.pick(&pools.decomposable),
                    1 => self.pick(&pools.pairs).0,
                    _ => u32::from(b'a') + self.below(26) as u32,
                };
                out.push(base);
                for _ in 0..1 + self.below(6) {
                    if self.one_in(2) {
                        self.mark(pools, out);
                    } else {
                        out.push(self.pick(&pools.pairs).1);
                    }
                }
            }
            18 => loop {
                let point = self.below(0x1_0000) as u32;
                if !(0xd800..=0xdfff).contains(&point) {
                    out.push(point);
                    break;
                }
            },
            19 => out.push(0x1_0000 + self.below(0x10_0000) as u32),
            20 => out.push(self.pick(&pools.pairs).1),
            _ => {
                for _ in 0..1 + self.below(12) {
                    self.mark(pools, out);
                }
            }
        }
    }

    /// One valid multi-byte UTF-8 scalar.
    fn valid_utf8(&mut self) -> Vec<u8> {
        let point = match self.below(3) {
            0 => 0x80 + self.below(0x780) as u32,
            1 => loop {
                let point = 0x800 + self.below(0xf800) as u32;
                if !(0xd800..=0xdfff).contains(&point) {
                    break point;
                }
            },
            _ => 0x1_0000 + self.below(0x10_0000) as u32,
        };
        char::from_u32(point).map_or_else(Vec::new, |c| c.to_string().into_bytes())
    }

    /// Bytes shaped like UTF-8, valid or broken in every way the decoders distinguish.
    fn utf8ish(&mut self, out: &mut Vec<u8>) {
        match self.below(12) {
            0..=2 => out.push(self.below(128) as u8),
            3 => out.push(self.byte()),
            4 => out.push(0x80 + self.below(64) as u8),
            5 => {
                let bytes = self.valid_utf8();
                out.extend(bytes);
            }
            6 => {
                let bytes = self.valid_utf8();
                let keep = 1 + self.below(bytes.len().saturating_sub(1).max(1));
                out.extend_from_slice(&bytes[..keep.min(bytes.len())]);
            }
            7 => {
                out.push(0xc0 + self.below(64) as u8);
                for _ in 0..self.below(4) {
                    out.push(0x80 + self.below(64) as u8);
                }
            }
            8 => out.extend([
                0xed,
                0xa0 + self.below(32) as u8,
                0x80 + self.below(64) as u8,
            ]),
            9 => match self.below(3) {
                0 => out.extend([0xc0 + self.below(2) as u8, 0x80 + self.below(64) as u8]),
                1 => out.extend([
                    0xe0,
                    0x80 + self.below(32) as u8,
                    0x80 + self.below(64) as u8,
                ]),
                _ => out.extend([
                    0xf0,
                    0x80 + self.below(16) as u8,
                    0x80 + self.below(64) as u8,
                    0x80 + self.below(64) as u8,
                ]),
            },
            10 => {
                if self.one_in(2) {
                    out.extend([0xf4, 0x90 + self.below(48) as u8]);
                } else {
                    out.push(0xf5 + self.below(11) as u8);
                }
                for _ in 0..self.below(3) {
                    out.push(0x80 + self.below(64) as u8);
                }
            }
            _ => {
                out.push(self.pick(&[0xe0, 0xed, 0xf0, 0xf4]));
                for _ in 0..1 + self.below(3) {
                    let next = if self.one_in(4) {
                        self.byte()
                    } else {
                        0x80 + self.below(64) as u8
                    };
                    out.push(next);
                }
            }
        }
    }

    fn utf8ish_string(&mut self, units: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..units {
            self.utf8ish(&mut out);
        }
        out
    }

    fn random_bytes(&mut self, length: usize) -> Vec<u8> {
        (0..length).map(|_| self.byte()).collect()
    }
}

fn utf8(points: &[u32]) -> Vec<u8> {
    let mut text = String::new();
    for &point in points {
        if let Some(c) = char::from_u32(point) {
            text.push(c);
        }
    }
    text.into_bytes()
}

#[derive(Clone)]
struct ModelSpec {
    vocabulary: Vec<Vec<u8>>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    added: Vec<(u32, bool, Vec<u8>)>,
}

impl ModelSpec {
    fn fixture() -> Self {
        let mut vocabulary: Vec<Vec<u8>> = (0..=u8::MAX).map(|byte| vec![byte]).collect();
        for word in [
            "ab", "aa", "aba", "abab", "aaaa", " a", "<X>", "<X>!", "<Y>", "zz",
        ] {
            vocabulary.push(word.as_bytes().to_vec());
        }
        let merges = [
            ("a", "b"),
            ("a", "a"),
            ("ab", "a"),
            ("ab", "ab"),
            ("aa", "aa"),
            (" ", "a"),
        ]
        .map(|(left, right)| (left.as_bytes().to_vec(), right.as_bytes().to_vec()))
        .to_vec();
        let added = [(262, true, "<X>"), (263, true, "<X>!"), (264, false, "<Y>")]
            .map(|(id, special, content)| (id, special, content.as_bytes().to_vec()))
            .to_vec();
        Self {
            vocabulary,
            merges,
            added,
        }
    }

    fn added_contents(&self) -> Vec<Vec<u8>> {
        self.added
            .iter()
            .map(|(_, _, content)| content.clone())
            .collect()
    }

    fn ordinary_ids(&self) -> Vec<usize> {
        let added: BTreeSet<u32> = self.added.iter().map(|(id, _, _)| *id).collect();
        (0..self.vocabulary.len())
            .filter(|&id| !added.contains(&(id as u32)))
            .collect()
    }
}

/// A random valid model: 256 shuffled singletons, merges built over a small alphabet (some ranked before their parts exist), extra unreachable tokens and added tokens that may repeat ordinary spellings or prefix each other.
fn synthetic(generator: &mut Generator, merge_target: usize) -> (ModelSpec, Vec<u8>) {
    let size = 2 + generator.below(6);
    let mut alphabet = Vec::new();
    while alphabet.len() < size {
        let byte = if generator.one_in(3) {
            generator.byte()
        } else {
            b'a' + generator.below(6) as u8
        };
        if !alphabet.contains(&byte) {
            alphabet.push(byte);
        }
    }
    let mut tokens: Vec<Vec<u8>> = (0..=u8::MAX).map(|byte| vec![byte]).collect();
    let mut known: BTreeSet<Vec<u8>> = tokens.iter().cloned().collect();
    let mut parts: Vec<Vec<u8>> = alphabet.iter().map(|&byte| vec![byte]).collect();
    let mut pairs = BTreeSet::new();
    let mut merges = Vec::new();
    let target = 1 + generator.below(merge_target);
    let mut attempts = 0;
    while merges.len() < target && attempts < target * 20 {
        attempts += 1;
        let left = parts[generator.below(parts.len())].clone();
        let right = parts[generator.below(parts.len())].clone();
        if left.len() + right.len() > 48 || !pairs.insert((left.clone(), right.clone())) {
            continue;
        }
        let joined = [left.as_slice(), right.as_slice()].concat();
        if known.insert(joined.clone()) {
            tokens.push(joined.clone());
            parts.push(joined);
        }
        merges.push((left, right));
    }
    for _ in 0..generator.below(merges.len() / 2 + 1) {
        let (i, j) = (generator.below(merges.len()), generator.below(merges.len()));
        merges.swap(i, j);
    }
    for _ in 0..generator.below(5) {
        let length = 2 + generator.below(5);
        let token: Vec<u8> = (0..length).map(|_| generator.pick(&alphabet)).collect();
        if known.insert(token.clone()) {
            tokens.push(token);
        }
    }
    let mut added_contents: Vec<Vec<u8>> = Vec::new();
    for _ in 0..generator.below(7) {
        let letter = generator.pick(&alphabet);
        let content = match generator.below(6) {
            0 => [b"<|".as_slice(), &[letter], b"|>"].concat(),
            1 => {
                let mut content = vec![b'<'];
                for _ in 0..1 + generator.below(3) {
                    content.push(generator.pick(&alphabet));
                }
                content
            }
            2 => (0..1 + generator.below(4))
                .map(|_| generator.pick(&alphabet))
                .collect(),
            3 => vec![letter],
            4 => {
                let length = 1 + generator.below(4);
                generator.random_bytes(length)
            }
            _ => b"<|im_start|>".to_vec(),
        };
        if !added_contents.contains(&content) {
            added_contents.push(content);
        }
    }
    let mut entries: Vec<(Vec<u8>, Option<bool>)> =
        tokens.into_iter().map(|token| (token, None)).collect();
    for content in added_contents {
        let special = generator.one_in(2);
        entries.push((content, Some(special)));
    }
    for i in (1..entries.len()).rev() {
        let j = generator.below(i + 1);
        entries.swap(i, j);
    }
    let mut added: Vec<(u32, bool, Vec<u8>)> = entries
        .iter()
        .enumerate()
        .filter_map(|(id, (content, special))| {
            special.map(|special| (id as u32, special, content.clone()))
        })
        .collect();
    for i in (1..added.len()).rev() {
        let j = generator.below(i + 1);
        added.swap(i, j);
    }
    let vocabulary = entries.into_iter().map(|(token, _)| token).collect();
    (
        ModelSpec {
            vocabulary,
            merges,
            added,
        },
        alphabet,
    )
}

/// Text over the model's alphabet with added-token spellings, their prefixes and stray bytes mixed in.
fn model_text(generator: &mut Generator, spec: &ModelSpec, alphabet: &[u8]) -> Vec<u8> {
    let contents = spec.added_contents();
    let mut text = Vec::new();
    for _ in 0..generator.below(12) {
        match generator.below(8) {
            0..=3 => {
                for _ in 0..1 + generator.below(6) {
                    text.push(generator.pick(alphabet));
                }
            }
            4 if !contents.is_empty() => {
                text.extend_from_slice(&contents[generator.below(contents.len())])
            }
            5 if !contents.is_empty() => {
                let content = &contents[generator.below(contents.len())];
                text.extend_from_slice(&content[..generator.below(content.len() + 1)]);
            }
            6 => text.push(b' '),
            _ => text.push(generator.byte()),
        }
    }
    text
}

const PROCESSORS: &[&str] = &[
    "identity", "identity", "identity", "bytes", "split", "empties", "copies0", "copies2",
    "copies3", "grow2", "refuse61", "refuse3c", "refuse20", "nfc", "nfc",
];

/// One mutation of a valid model, aimed at a constructor refusal.
fn mutate(generator: &mut Generator, spec: &mut ModelSpec) {
    let length = spec.vocabulary.len();
    if length == 0 {
        return;
    }
    match generator.below(19) {
        0 => spec.vocabulary.truncate(generator.below(256)),
        1 => {
            let ordinary = spec.ordinary_ids();
            if !ordinary.is_empty() {
                let (i, j) = (generator.pick(&ordinary), generator.pick(&ordinary));
                spec.vocabulary[i] = spec.vocabulary[j].clone();
            }
        }
        2 => {
            let ordinary = spec.ordinary_ids();
            let single: Vec<usize> = ordinary
                .into_iter()
                .filter(|&id| spec.vocabulary[id].len() == 1)
                .collect();
            if !single.is_empty() {
                let id = generator.pick(&single);
                spec.vocabulary[id] = format!("no-singleton-{id}").into_bytes();
            }
        }
        3 => {
            let id = generator.below(length);
            spec.vocabulary[id].clear();
        }
        4 => {
            let id = generator.below(length);
            spec.vocabulary[id] = vec![b'x'; 4096 + generator.below(2)];
        }
        5 if !spec.merges.is_empty() => {
            let merge = spec.merges[generator.below(spec.merges.len())].clone();
            let at = generator.below(spec.merges.len() + 1);
            spec.merges.insert(at, merge);
        }
        6 => {
            let part = vec![generator.byte()];
            spec.merges.push((b"\xff\xfe-missing".to_vec(), part));
        }
        7 => {
            let at = generator.below(spec.merges.len() + 1);
            let part = vec![generator.byte()];
            let merge = if generator.one_in(2) {
                (Vec::new(), part)
            } else {
                (part, Vec::new())
            };
            spec.merges.insert(at, merge);
        }
        8 => {
            let (left, right) = (vec![generator.byte()], vec![generator.byte()]);
            spec.merges.push((left, right));
        }
        9 => {
            let (left, right) = (vec![b'L'; 2048], vec![b'R'; 2048 + generator.below(2)]);
            spec.vocabulary.push(left.clone());
            spec.vocabulary.push(right.clone());
            spec.vocabulary
                .push([left.as_slice(), right.as_slice()].concat());
            spec.merges.push((left, right));
        }
        10 => {
            let id = (length + generator.below(3)) as u32;
            spec.added.push((id, true, b"<oob>".to_vec()));
        }
        11 if !spec.added.is_empty() => {
            let at = generator.below(spec.added.len());
            spec.added[at].2.push(b'#');
        }
        12 if !spec.added.is_empty() => {
            let at = generator.below(spec.added.len());
            let id = spec.added[at].0 as usize;
            spec.added[at].2.clear();
            if generator.one_in(2)
                && let Some(token) = spec.vocabulary.get_mut(id)
            {
                token.clear();
            }
        }
        13 if !spec.added.is_empty() => {
            let at = generator.below(spec.added.len());
            let id = spec.added[at].0 as usize;
            if let Some(token) = spec.vocabulary.get_mut(id) {
                *token = vec![b'<'; 4097];
            }
            spec.added[at].2 = vec![b'<'; 4097];
        }
        14 if !spec.added.is_empty() => {
            let marker = spec.added[generator.below(spec.added.len())].clone();
            spec.added.push(marker);
        }
        15 if !spec.added.is_empty() => {
            let content = spec.added[generator.below(spec.added.len())].2.clone();
            spec.added.push((length as u32, false, content.clone()));
            spec.vocabulary.push(content);
        }
        16 => {
            for _ in spec.added.len()..4097 {
                spec.added.push((0, false, vec![0]));
            }
        }
        17 if !spec.added.is_empty() => {
            let content = spec.added[generator.below(spec.added.len())].2.clone();
            spec.merges.push((content, b"a".to_vec()));
        }
        _ => {
            let id = generator.below(length) as u32;
            let content = vec![generator.byte(), generator.byte()];
            spec.added.push((id, generator.one_in(2), content));
        }
    }
}

fn generate() -> Harness<Generator> {
    let pools = pools()?;
    let mut g = Generator::new(0x7465_7874);

    // The C++ suites' own inputs, so their `rejected` checks get exact messages.
    for (input, op) in [
        (&b"cafe\xcc\x81"[..], "nfc"),
        ("\u{212b}\u{2126}".as_bytes(), "nfc"),
        ("\u{2460} \u{ff11}".as_bytes(), "nfc"),
        ("\u{340}\u{341}\u{344}".as_bytes(), "nfc"),
        ("\u{eb8}\u{eba}".as_bytes(), "nfc"),
        ("\u{10f48}\u{10f4b}".as_bytes(), "nfc"),
        ("\u{1100}\u{1161}\u{11a8}".as_bytes(), "nfc"),
        ("\u{301}\u{323}".as_bytes(), "nfc"),
        (b"\0a\0", "nfc"),
        (b"\xc0\x80", "nfc"),
        (b"\xe0\x80\x80", "nfc"),
        (b"\xed\xa0\x80", "nfc"),
        (b"\xf4\x90\x80\x80", "nfc"),
        (b"\xc2", "nfc"),
        (b"\x80", "nfc"),
        (b" ", "bld"),
        ("\u{1f989}".as_bytes(), "bld"),
        (b"\xc0\x80", "bld"),
        (b"\xc4", "bld"),
        (b"\xc4\x20", "bld"),
        ("\u{120}Hello".as_bytes(), "bld"),
        (b" \n\0", "ble"),
        (b"", "riu"),
        ("caf\u{e9}\u{1f989}".as_bytes(), "riu"),
        (b"\0abc", "riu"),
        (b"\xc0\x80", "riu"),
        (b"\xe1\x80A", "riu"),
        (b"\xf1\x80\x80", "riu"),
        (b"\xed\xa0\x80", "riu"),
        (b"\xf4\x90\x80\x80", "riu"),
    ] {
        g.case(op, format!("{op} {}", x(input)));
    }
    let every: Vec<u8> = (0..=u8::MAX).collect();
    g.case("ble", format!("ble {}", x(&every)));
    g.case("bld", format!("bld {}", x(&byte_level_encode(&every))));
    for point in [0xd800u32, 0x11_0000] {
        g.case("scalar", format!("scalar {point}"));
    }
    let mut long_marks = b"a".to_vec();
    for _ in 0..10_000 {
        long_marks.extend_from_slice("\u{301}\u{323}".as_bytes());
    }
    g.case("nfc", format!("nfc {}", x(&long_marks)));

    // Random NFC inputs: mostly valid scalar sequences, some broken UTF-8.
    for _ in 0..200_000 {
        let input = if g.one_in(10) {
            let units = g.below(10);
            g.utf8ish_string(units)
        } else {
            let mut points = Vec::new();
            for _ in 0..g.below(25) {
                g.scalar(&pools, &mut points);
            }
            utf8(&points)
        };
        g.case("nfc", format!("nfc {}", x(&input)));
    }
    // Every tricky scalar alone, and after a starter and before a mark.
    for &point in TRICKY {
        g.case("nfc", format!("nfc {}", x(&utf8(&[point]))));
        g.case("nfc", format!("nfc {}", x(&utf8(&[0x61, point, 0x301]))));
    }
    // NFC work limits.
    let limit = 8 * 1024 * 1024;
    g.case("nfc", format!("nfc {}", repeat(b"a", limit)));
    g.case("nfc", format!("nfc {}", repeat(b"a", limit + 1)));
    g.case(
        "nfc",
        format!("nfc {}", repeat("\u{1f82}".as_bytes(), limit / 3)),
    );
    g.case(
        "nfc",
        format!("nfc {}", repeat("\u{958}".as_bytes(), limit / 3)),
    );
    // Two U+1F82 (8 scalars), then ASCII up to one below the scalar limit: the final Hangul syllable still appends 3, two past the limit.
    let edge = format!(
        "{}+{}",
        repeat("\u{1f82}".as_bytes(), 2),
        repeat(b"a", limit - 9)
    );
    g.case("nfc", format!("nfc {edge}+{}", x("\u{ac01}".as_bytes())));
    let past = format!(
        "{}+{}",
        repeat("\u{1f82}".as_bytes(), 2),
        repeat(b"a", limit - 10)
    );
    g.case(
        "nfc",
        format!("nfc {past}+{}+x62", x("\u{ac01}".as_bytes())),
    );
    g.case(
        "nfc",
        format!("nfc {}", repeat("\u{301}\u{323}".as_bytes(), 262_144)),
    );
    g.case(
        "nfc",
        format!(
            "nfc x61+{}",
            repeat("\u{301}\u{323}\u{5b0}".as_bytes(), 100_000)
        ),
    );

    // The scalar codec directly.
    for _ in 0..10_000 {
        let units = g.below(10);
        let input = g.utf8ish_string(units);
        g.case("scalars", format!("scalars {}", x(&input)));
    }
    for point in [
        0u32,
        0x7f,
        0x80,
        0x7ff,
        0x800,
        0xd7ff,
        0xd800,
        0xdfff,
        0xe000,
        0xffff,
        0x1_0000,
        0x10_ffff,
        0x11_0000,
        u32::MAX,
    ] {
        g.case("scalar", format!("scalar {point}"));
    }
    for _ in 0..5_000 {
        let point = match g.below(3) {
            0 => g.below(0x12_0000) as u32,
            1 => 0xd700 + g.below(0x900) as u32,
            _ => g.random.next_u32(),
        };
        g.case("scalar", format!("scalar {point}"));
    }

    // Byte alphabet and lossy UTF-8.
    for _ in 0..30_000 {
        let length = g.below(48);
        let input = g.random_bytes(length);
        g.case("ble", format!("ble {}", x(&input)));
    }
    for _ in 0..80_000 {
        let spelling = match g.below(7) {
            0..=2 => {
                let length = g.below(24);
                byte_level_encode(&g.random_bytes(length))
            }
            3..=4 => {
                let length = 1 + g.below(16);
                let mut spelling = byte_level_encode(&g.random_bytes(length));
                let at = g.below(spelling.len());
                match g.below(4) {
                    0 => spelling[at] = g.byte(),
                    1 => spelling.insert(at, g.byte()),
                    2 => {
                        spelling.remove(at);
                    }
                    _ => spelling.truncate(at),
                }
                spelling
            }
            5 => (0..g.below(12))
                .map(|_| match g.below(4) {
                    0 => 0xc0 + g.below(8) as u8,
                    1 => 0x80 + g.below(64) as u8,
                    2 => g.below(128) as u8,
                    _ => g.byte(),
                })
                .collect(),
            _ => {
                let length = g.below(12);
                g.random_bytes(length)
            }
        };
        g.case("bld", format!("bld {}", x(&spelling)));
    }
    for _ in 0..80_000 {
        let units = g.below(16);
        let input = g.utf8ish_string(units);
        g.case("riu", format!("riu {}", x(&input)));
    }

    // SHA-256 against CommonCrypto, one-shot and in random chunks.
    for length in 0..=300 {
        let input = g.random_bytes(length);
        g.case("sha", format!("sha {} c", x(&input)));
    }
    for _ in 0..3_000 {
        let length = g.below(5_000);
        let input = g.random_bytes(length);
        let chunks: Vec<String> = (0..g.below(8)).map(|_| g.below(200).to_string()).collect();
        g.case("sha", format!("sha {} c{}", x(&input), chunks.join(",")));
    }
    g.case(
        "sha",
        format!("sha {} c1,63,64,65,1000", repeat(b"a", 1_000_000)),
    );

    // The C++ fixture model, its checks, its 5,000 oracle inputs and more.
    let fixture = ModelSpec::fixture();
    g.model(&fixture);
    for piece in [&b""[..], b"abab", b"aaaaa", b"zz", b"<X>"] {
        g.case("piece", format!("piece {}", x(piece)));
    }
    for text in [&b"ab<X>!aa<Y>"[..], b"<X", b"<X><X>"] {
        g.case("encode", format!("encode identity {}", x(text)));
    }
    g.case("decode", "decode 0 i256,263,257,264".into());
    g.case("decode", "decode 1 i256,263,257,264".into());
    g.case("decode", "decode 0 i266".into());
    let mut oracle = Mt19937::new(0x6270_6531);
    for run in 0..5000 {
        let mut input = Vec::new();
        for _ in 0..oracle.next_u32() % 129 {
            input.push(if run % 2 == 1 {
                b"ab "[(oracle.next_u32() % 3) as usize]
            } else {
                (oracle.next_u32() & 255) as u8
            });
        }
        g.case("piece", format!("piece {}", x(&input)));
    }
    let fixture_alphabet = b"ab <X>!Yz";
    for _ in 0..20_000 {
        let text = model_text(&mut g, &fixture, fixture_alphabet);
        let processor = g.pick(PROCESSORS);
        g.case("encode", format!("encode {processor} {}", x(&text)));
    }
    for _ in 0..10_000 {
        let ids: Vec<u32> = (0..g.below(16)).map(|_| g.below(270) as u32).collect();
        let (op, skip) = (g.pick(&["decode", "text"]), g.below(2));
        g.case(op, format!("{op} {skip} {}", ids_text(&ids)));
    }

    // Token limits with the fixture.
    let mib = 1024 * 1024;
    for (line, op) in [
        (format!("piece {}", repeat(b"a", mib + 1)), "piece"),
        (format!("piece {}", repeat(b"a", mib)), "piece"),
        (format!("piece {}", repeat(b"z", 1_000_001)), "piece"),
        (format!("piece {}", repeat(b"z", 1_000_000)), "piece"),
        (
            format!("encode identity {}", repeat(b"a", limit + 1)),
            "encode",
        ),
        (
            format!("encode identity {}+x3c583e", repeat(b"z", 1_000_000)),
            "encode",
        ),
        (
            format!("encode identity {}+x3c583e", repeat(b"z", 999_999)),
            "encode",
        ),
        (
            format!("encode identity {0}+x3c583e+{0}", repeat(b"z", 600_000)),
            "encode",
        ),
        (
            format!(
                "encode identity {}+x3c583e+{}",
                repeat(b"z", 600_000),
                repeat(b"z", 399_999)
            ),
            "encode",
        ),
        (format!("encode grow{} x61", limit + 1), "encode"),
        (format!("encode grow{limit} x61"), "encode"),
        (format!("encode copies9 {}", repeat(b"a", mib)), "encode"),
        (
            format!("encode bytes {}", repeat(b"a", 1_000_001)),
            "encode",
        ),
        (
            format!("encode refuse7a {}+x3c583e", repeat(b"a", 1000)),
            "encode",
        ),
        (
            format!("encode refuse61 {}+x3c583e", repeat(b"a", 1000)),
            "encode",
        ),
        (format!("encode nfc {}", repeat(b"\xc0", 3)), "encode"),
        (
            format!("encode nfc {}", repeat("\u{1f82}".as_bytes(), limit / 3)),
            "encode",
        ),
        ("decode 0 i97*1000001".to_string(), "decode"),
        ("decode 0 i97*1000000".to_string(), "decode"),
        ("text 0 i97*1000001".to_string(), "text"),
        ("text 1 i262*1000000".to_string(), "text"),
    ] {
        g.case(op, line);
    }
    // A 4,096-byte token reaches the 8 MiB decode limit in 2,048 IDs.
    let mut big = ModelSpec::fixture();
    big.vocabulary.push(vec![b'q'; 4096]);
    g.model(&big);
    g.case("decode", "decode 0 i266*2048".into());
    g.case("decode", "decode 0 i266*2048+i97".into());
    g.case("decode", "decode 1 i262*9+i266*2048+i262".into());
    g.case("text", "text 0 i266*2048+i97".into());

    // Random valid models and their operations.
    for _ in 0..60 {
        let (spec, alphabet) = synthetic(&mut g, 300);
        g.model(&spec);
        let size = spec.vocabulary.len();
        for _ in 0..300 {
            let length = g.below(64);
            let piece: Vec<u8> = (0..length)
                .map(|_| {
                    if g.one_in(16) {
                        g.byte()
                    } else {
                        g.pick(&alphabet)
                    }
                })
                .collect();
            g.case("piece", format!("piece {}", x(&piece)));
        }
        for _ in 0..300 {
            let text = model_text(&mut g, &spec, &alphabet);
            let processor = g.pick(PROCESSORS);
            g.case("encode", format!("encode {processor} {}", x(&text)));
        }
        for _ in 0..300 {
            let ids: Vec<u32> = (0..g.below(16)).map(|_| g.below(size + 3) as u32).collect();
            let (op, skip) = (g.pick(&["decode", "text"]), g.below(2));
            g.case(op, format!("{op} {skip} {}", ids_text(&ids)));
        }
    }

    // Constructor refusals: one or two mutations of small valid models.
    for _ in 0..2_000 {
        let (mut spec, _) = synthetic(&mut g, 24);
        mutate(&mut g, &mut spec);
        if g.one_in(3) {
            mutate(&mut g, &mut spec);
        }
        g.model(&spec);
    }
    // Trie node limit: fifteen 4,096-byte markers fit, sixteen do not.
    for markers in [15usize, 16] {
        let mut spec = ModelSpec::fixture();
        spec.added.clear();
        for marker in 0..markers {
            let mut content = vec![b'#'; 4096];
            content[0] = b'A' + marker as u8;
            spec.added
                .push((spec.vocabulary.len() as u32, false, content.clone()));
            spec.vocabulary.push(content);
        }
        g.model(&spec);
    }
    // 4,096 added tokens fit; 4,097 do not.
    for markers in [4096usize, 4097] {
        let mut spec = ModelSpec::fixture();
        spec.added.clear();
        for marker in 0..markers {
            let content = format!("<{marker:04}>").into_bytes();
            spec.added.push((
                spec.vocabulary.len() as u32,
                marker % 2 == 0,
                content.clone(),
            ));
            spec.vocabulary.push(content);
        }
        g.model(&spec);
    }
    // Vocabulary bytes: 256 singletons plus 16,383 tokens of 4,096 bytes fit in 64 MiB; 16,384 do not.
    for tokens in [16_383u32, 16_384] {
        g.lines.push(format!("model {} 0 0", 256 + tokens));
        for byte in 0..=u8::MAX {
            g.lines.push(x(&[byte]));
        }
        for token in 0..tokens {
            g.lines.push(format!(
                "{}+{}",
                x(&token.to_be_bytes()),
                repeat(b".", 4092)
            ));
        }
        g.ops.push("model");
    }
    // Vocabulary and merge counts: 1,000,000 fit; 1,000,001 do not.
    for tokens in [1_000_000u32, 1_000_001] {
        g.lines.push(format!("model {tokens} 0 0"));
        for byte in 0..=u8::MAX {
            g.lines.push(x(&[byte]));
        }
        for token in 256..tokens {
            g.lines.push(x(&token.to_be_bytes()[1..]));
        }
        g.ops.push("model");
    }
    g.lines.push("model 256 1000001 0".into());
    for byte in 0..=u8::MAX {
        g.lines.push(x(&[byte]));
    }
    for _ in 0..1_000_001 {
        g.lines.push("x61 x61".into());
    }
    g.ops.push("model");
    Ok(g)
}

fn result_kind(line: &str) -> &str {
    line.split(' ').next().unwrap_or("")
}

fn shorten(line: &str) -> String {
    match line.get(..160) {
        Some(head) if line.len() > 160 => format!("{head}... ({} bytes)", line.len()),
        _ => line.to_string(),
    }
}

#[test]
#[ignore = "needs the C++ driver; run rust/tools/diff/text.sh"]
fn rust_matches_cpp_on_generated_cases() {
    let driver = std::env::var_os("TOKSPUR_TEXT_DIFF_DRIVER")
        .expect("set TOKSPUR_TEXT_DIFF_DRIVER to the compiled text_diff.cpp");
    let directory = PathBuf::from(
        std::env::var_os("TOKSPUR_TEXT_DIFF_DIR").expect("set TOKSPUR_TEXT_DIFF_DIR"),
    );
    let generator = generate().unwrap();
    let mut cases = generator.lines.join("\n");
    cases.push('\n');
    let cases_path = directory.join("cases.txt");
    fs::write(&cases_path, &cases).unwrap();

    let cpp_path = directory.join("cpp.txt");
    let status = Command::new(&driver)
        .stdin(fs::File::open(&cases_path).unwrap())
        .stdout(fs::File::create(&cpp_path).unwrap())
        .status()
        .unwrap();
    assert!(status.success(), "C++ driver failed: {status}");
    let rust = evaluate_file(&cases).unwrap();
    fs::write(directory.join("rust.txt"), &rust).unwrap();
    let cpp = fs::read_to_string(&cpp_path).unwrap();

    let cpp_lines: Vec<&str> = cpp.split_terminator('\n').collect();
    let rust_lines: Vec<&str> = rust.split_terminator('\n').collect();
    let ops = &generator.ops;
    let mut per_op: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    let mut messages: BTreeMap<&str, usize> = BTreeMap::new();
    let mut differences = Vec::new();
    for (index, (cpp_line, rust_line)) in cpp_lines.iter().zip(&rust_lines).enumerate() {
        let op = ops.get(index).copied().unwrap_or("?");
        *per_op
            .entry(op)
            .or_default()
            .entry(result_kind(cpp_line))
            .or_default() += 1;
        if let Some(message) = cpp_line.strip_prefix("err ") {
            *messages.entry(message).or_default() += 1;
        }
        if cpp_line != rust_line {
            differences.push(index);
        }
    }
    let mut summary = format!(
        "cases: {} generated, {} C++ results, {} Rust results, {} compared, {} different\n",
        ops.len(),
        cpp_lines.len(),
        rust_lines.len(),
        cpp_lines.len().min(rust_lines.len()),
        differences.len()
    );
    for (op, kinds) in &per_op {
        let kinds: Vec<String> = kinds
            .iter()
            .map(|(kind, n)| format!("{kind} {n}"))
            .collect();
        summary.push_str(&format!("  {op}: {}\n", kinds.join(", ")));
    }
    summary.push_str("refusals seen (C++ message: count):\n");
    for (message, n) in &messages {
        summary.push_str(&format!("  {message}: {n}\n"));
    }
    for &index in differences.iter().take(20) {
        summary.push_str(&format!(
            "difference at case {} ({}):\n  C++:  {}\n  Rust: {}\n",
            index + 1,
            ops.get(index).copied().unwrap_or("?"),
            shorten(cpp_lines[index]),
            shorten(rust_lines[index])
        ));
    }
    fs::write(directory.join("summary.txt"), &summary).unwrap();
    println!("{summary}");
    assert_eq!(cpp_lines.len(), ops.len(), "C++ result count");
    assert_eq!(rust_lines.len(), ops.len(), "Rust result count");
    assert!(
        differences.is_empty(),
        "{} differences; see summary.txt",
        differences.len()
    );
}

#[test]
fn case_language_round_trips() {
    // The harness itself, without the C++ driver.
    assert_eq!(bytes_field("x").unwrap(), b"");
    assert_eq!(bytes_field("x6162*3+x").unwrap(), b"ababab");
    assert_eq!(bytes_field("x61*0+x62").unwrap(), b"b");
    assert!(
        bytes_field("x6").is_err() && bytes_field("x6G").is_err() && bytes_field("61").is_err()
    );
    assert_eq!(ids_field("i").unwrap(), Vec::<u32>::new());
    assert_eq!(ids_field("i1,2*2+i3").unwrap(), [1, 2, 1, 2, 3]);
    assert!(ids_field("i4294967296").is_err() && decimal("+1").is_err());
    let processor = Processor::Split;
    assert_eq!(
        processor.pieces(b"a  b c").unwrap(),
        [b"a".to_vec(), b" ".to_vec(), b" b".to_vec(), b" c".to_vec()]
    );
    let refused = Processor::parse("refuse62")
        .unwrap()
        .pieces(b"abc")
        .unwrap_err();
    assert_eq!(refused.to_string(), "text processor refused segment");
    let cases = "nfc x63616665cc81\nmodel 1 0 0\nx61\npiece x61\nsha x616263 c1,1\n";
    assert_eq!(
        evaluate_file(cases).unwrap(),
        "ok x636166c3a9\nerr unsupported BPE vocabulary size\nnomodel\n\
         ok ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad \
         ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
    );
}
