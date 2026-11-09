//! Generates the Unicode 9 normalization tables from `native/data/unicode9.inc`, the generated data the C++ `#include`s, so the C++ and Rust read one source. The include is the only input; there is no network access. The parser accepts exactly the format `scripts/generate_unicode9.py` writes and fails the build on anything else, including tables out of the sorted order the binary searches rely on.

use std::env;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// The C++ include, relative to this crate's manifest directory.
const SOURCE: &str = "../../../native/data/unicode9.inc";

/// Why generation failed. `Debug` prints the `Display` text because a build script reports a returned error with `{:?}`.
enum BuildError {
    /// A variable Cargo sets for build scripts is missing.
    Locate(&'static str),
    /// The include could not be read as UTF-8 text.
    Read { path: PathBuf, source: io::Error },
    /// The include deviates from the generator's exact format.
    Parse { line: usize, reason: &'static str },
    /// Formatting the generated Rust failed.
    Render(fmt::Error),
    /// The generated tables or Cargo's directives could not be written.
    Write { path: PathBuf, source: io::Error },
}

impl fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Locate(variable) => write!(formatter, "Cargo did not set {variable}"),
            Self::Read { path, source } => {
                write!(formatter, "cannot read {}: {source}", path.display())
            }
            Self::Parse { line, reason } => write!(formatter, "{SOURCE}:{line}: {reason}"),
            Self::Render(_) => formatter.write_str("cannot format the generated tables"),
            Self::Write { path, source } => {
                write!(formatter, "cannot write {}: {source}", path.display())
            }
        }
    }
}

impl fmt::Debug for BuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } | Self::Write { source, .. } => Some(source),
            Self::Render(source) => Some(source),
            Self::Locate(_) | Self::Parse { .. } => None,
        }
    }
}

/// `unicode9_data::ClassRange`.
struct ClassRange {
    first: u32,
    last: u32,
    value: u8,
}

/// `unicode9_data::Decomposition`.
struct Decomposition {
    point: u32,
    first: u32,
    second: u32,
}

/// `unicode9_data::Composition`.
struct Composition {
    pair: u64,
    point: u32,
}

struct Tables {
    /// The include's leading `//` lines (generator, licence and source pins).
    notice: Vec<String>,
    combining: Vec<ClassRange>,
    decomposition: Vec<Decomposition>,
    composition: Vec<Composition>,
}

/// Numbered lines of the include; `Parse` errors name the line.
struct Lines<'a> {
    lines: std::iter::Peekable<std::iter::Enumerate<std::str::Split<'a, char>>>,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.split('\n').enumerate().peekable(),
        }
    }

    fn next(&mut self) -> Result<(usize, &'a str), BuildError> {
        match self.lines.next() {
            Some((index, line)) => Ok((index + 1, line)),
            None => Err(BuildError::Parse {
                line: 0,
                reason: "unexpected end of file",
            }),
        }
    }

    fn expect(&mut self, wanted: &str) -> Result<(), BuildError> {
        let (line, text) = self.next()?;
        if text == wanted {
            Ok(())
        } else {
            Err(BuildError::Parse {
                line,
                reason: "line differs from the generator's format",
            })
        }
    }

    /// Entry lines (`    {...},`) up to the closing `};`, with the braces and trailing comma removed.
    fn entries(&mut self) -> Result<Vec<(usize, &'a str)>, BuildError> {
        let mut entries = Vec::new();
        loop {
            let (line, text) = self.next()?;
            if text == "};" {
                return Ok(entries);
            }
            match text
                .strip_prefix("    {")
                .and_then(|rest| rest.strip_suffix("},"))
            {
                Some(inner) => entries.push((line, inner)),
                None => {
                    return Err(BuildError::Parse {
                        line,
                        reason: "malformed table entry",
                    });
                }
            }
        }
    }
}

/// Parses `0x<lowercase hex>` exactly as the generator's `0x{:x}` writes it.
fn hex(line: usize, field: &str) -> Result<u64, BuildError> {
    let malformed = BuildError::Parse {
        line,
        reason: "malformed hexadecimal field",
    };
    let Some(digits) = field.strip_prefix("0x") else {
        return Err(malformed);
    };
    if digits.is_empty()
        || !digits
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(malformed);
    }
    let Ok(value) = u64::from_str_radix(digits, 16) else {
        return Err(malformed);
    };
    // Canonical spelling only: no leading zeros.
    if format!("{value:x}") != digits {
        return Err(malformed);
    }
    Ok(value)
}

fn hex32(line: usize, field: &str) -> Result<u32, BuildError> {
    u32::try_from(hex(line, field)?).map_err(|_| BuildError::Parse {
        line,
        reason: "value exceeds uint32_t",
    })
}

/// Parses a decimal `uint8_t` exactly as the generator's `{}` writes it.
fn decimal8(line: usize, field: &str) -> Result<u8, BuildError> {
    let malformed = BuildError::Parse {
        line,
        reason: "malformed decimal field",
    };
    if field.is_empty() || !field.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(malformed);
    }
    let Ok(value) = field.parse::<u8>() else {
        return Err(malformed);
    };
    if value.to_string() != field {
        return Err(malformed);
    }
    Ok(value)
}

fn split<const N: usize>(line: usize, inner: &str) -> Result<[&str; N], BuildError> {
    let fields: Vec<&str> = inner.split(", ").collect();
    <[&str; N]>::try_from(fields).map_err(|_| BuildError::Parse {
        line,
        reason: "wrong number of fields",
    })
}

fn parse(text: &str) -> Result<Tables, BuildError> {
    let mut lines = Lines::new(text);
    let mut notice = Vec::new();
    while let Some((_, line)) = lines.lines.peek() {
        if !line.starts_with("// ") {
            break;
        }
        notice.push((*line).to_owned());
        lines.next()?;
    }
    if notice.is_empty() {
        return Err(BuildError::Parse {
            line: 1,
            reason: "missing generator notice",
        });
    }
    lines.expect("#pragma once")?;
    lines.expect("#include <cstdint>")?;
    lines.expect("namespace tokspur::unicode9_data {")?;

    lines.expect("struct ClassRange { uint32_t first; uint32_t last; uint8_t value; };")?;
    lines.expect("inline constexpr ClassRange combining[] = {")?;
    let mut combining: Vec<ClassRange> = Vec::new();
    for (line, inner) in lines.entries()? {
        let [first, last, value] = split::<3>(line, inner)?;
        let range = ClassRange {
            first: hex32(line, first)?,
            last: hex32(line, last)?,
            value: decimal8(line, value)?,
        };
        // `combining` binary-searches by `last`, then checks `first`: the ranges must be ordered and disjoint for that to find the one range.
        let ordered = combining
            .last()
            .is_none_or(|previous| previous.last < range.first);
        if range.first > range.last || !ordered {
            return Err(BuildError::Parse {
                line,
                reason: "class ranges are not ordered and disjoint",
            });
        }
        combining.push(range);
    }

    lines.expect("struct Decomposition { uint32_t point; uint32_t first; uint32_t second; };")?;
    lines.expect("inline constexpr Decomposition decomposition[] = {")?;
    let mut decomposition: Vec<Decomposition> = Vec::new();
    for (line, inner) in lines.entries()? {
        let [point, first, second] = split::<3>(line, inner)?;
        let item = Decomposition {
            point: hex32(line, point)?,
            first: hex32(line, first)?,
            second: hex32(line, second)?,
        };
        if decomposition
            .last()
            .is_some_and(|previous| previous.point >= item.point)
        {
            return Err(BuildError::Parse {
                line,
                reason: "decompositions are not strictly ordered",
            });
        }
        decomposition.push(item);
    }

    lines.expect("struct Composition { uint64_t pair; uint32_t point; };")?;
    lines.expect("inline constexpr Composition composition[] = {")?;
    let mut composition: Vec<Composition> = Vec::new();
    for (line, inner) in lines.entries()? {
        let [pair, point] = split::<2>(line, inner)?;
        let Some(pair) = pair.strip_suffix("ULL") else {
            return Err(BuildError::Parse {
                line,
                reason: "composition key lacks its ULL suffix",
            });
        };
        let item = Composition {
            pair: hex(line, pair)?,
            point: hex32(line, point)?,
        };
        if composition
            .last()
            .is_some_and(|previous| previous.pair >= item.pair)
        {
            return Err(BuildError::Parse {
                line,
                reason: "compositions are not strictly ordered",
            });
        }
        composition.push(item);
    }

    lines.expect("} // namespace tokspur::unicode9_data")?;
    lines.expect("")?;
    if let Some((index, _)) = lines.lines.next() {
        return Err(BuildError::Parse {
            line: index + 1,
            reason: "text after the namespace",
        });
    }
    Ok(Tables {
        notice,
        combining,
        decomposition,
        composition,
    })
}

/// Writes the tables as Rust statics with the include's field names, in the include's order, preceded by its notice.
fn render(tables: &Tables) -> Result<String, fmt::Error> {
    use std::fmt::Write as _;
    let mut output = String::new();
    writeln!(
        output,
        "// Generated by tokspur-text/build.rs from native/data/unicode9.inc. Do not edit."
    )?;
    for line in &tables.notice {
        writeln!(output, "{line}")?;
    }
    writeln!(
        output,
        "pub(super) static COMBINING: [ClassRange; {}] = [",
        tables.combining.len()
    )?;
    for range in &tables.combining {
        writeln!(
            output,
            "    ClassRange {{ first: {:#x}, last: {:#x}, value: {} }},",
            range.first, range.last, range.value
        )?;
    }
    writeln!(output, "];")?;
    writeln!(
        output,
        "pub(super) static DECOMPOSITION: [Decomposition; {}] = [",
        tables.decomposition.len()
    )?;
    for item in &tables.decomposition {
        writeln!(
            output,
            "    Decomposition {{ point: {:#x}, first: {:#x}, second: {:#x} }},",
            item.point, item.first, item.second
        )?;
    }
    writeln!(output, "];")?;
    writeln!(
        output,
        "pub(super) static COMPOSITION: [Composition; {}] = [",
        tables.composition.len()
    )?;
    for item in &tables.composition {
        writeln!(
            output,
            "    Composition {{ pair: {:#x}, point: {:#x} }},",
            item.pair, item.point
        )?;
    }
    writeln!(output, "];")?;
    Ok(output)
}

fn main() -> Result<(), BuildError> {
    let manifest =
        env::var_os("CARGO_MANIFEST_DIR").ok_or(BuildError::Locate("CARGO_MANIFEST_DIR"))?;
    let out_dir = env::var_os("OUT_DIR").ok_or(BuildError::Locate("OUT_DIR"))?;
    let source = Path::new(&manifest).join(SOURCE);
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "cargo::rerun-if-changed={}", source.display()).map_err(|error| {
        BuildError::Write {
            path: PathBuf::from("<stdout>"),
            source: error,
        }
    })?;
    let text = fs::read_to_string(&source).map_err(|error| BuildError::Read {
        path: source.clone(),
        source: error,
    })?;
    let tables = parse(&text)?;
    let rendered = render(&tables).map_err(BuildError::Render)?;
    let target = Path::new(&out_dir).join("unicode9.rs");
    fs::write(&target, rendered).map_err(|error| BuildError::Write {
        path: target.clone(),
        source: error,
    })
}
