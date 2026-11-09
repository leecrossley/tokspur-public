//! Differential runs against the C++ reader. `rust/tools/diff/gguf.sh` builds `gguf-driver` (`rust/tools/diff/gguf_driver.cpp` with `native/src/gguf.cpp`) into `build/rust-diff/` and runs these ignored tests with its paths in the environment. Every case is summarised by both readers (`summary.rs` and the driver) and compared line for line; outputs land in `$TOKSPUR_GGUF_DIFF_OUT`.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::Write as _;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use tokspur_gguf::{Gguf, MappedFile, ModelFiles};

use crate::summary;
use crate::support::{
    Bytes, Data, Key, data, default_fixture, fixture, fixture_header, fuzz_seeds, integer, key,
    keys_fixture, mutations, number, replace, split, string, text,
};

const DRIVER: &str = "TOKSPUR_GGUF_DIFF_DRIVER";
const OUT: &str = "TOKSPUR_GGUF_DIFF_OUT";
const CPP_SEEDS: &str = "TOKSPUR_GGUF_DIFF_CPP_SEEDS";
const CPP_FUZZ: &str = "TOKSPUR_GGUF_DIFF_CPP_FUZZ";
const REAL_SHARDS: &str = "TOKSPUR_GGUF_REAL_SHARDS";

/// `fuzz_gguf` (standalone) runs this many mutation rounds.
const MUTATION_ROUNDS: usize = 50_000;
/// Mutated inputs also written as files, to compare the file reader.
const MUTATION_FILES: usize = 1_000;
const HEADER_LIMIT: u64 = 256 * 1024 * 1024;

fn env_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name}; run rust/tools/diff/gguf.sh"))
}

fn push(text: &mut String, line: String) {
    text.push_str(&line);
    text.push('\n');
}

struct Cases {
    labels: Vec<String>,
    inputs: Vec<Bytes>,
}

impl Cases {
    fn new() -> Self {
        Self {
            labels: Vec::new(),
            inputs: Vec::new(),
        }
    }

    fn add(&mut self, label: impl Into<String>, bytes: Bytes) {
        self.labels.push(label.into());
        self.inputs.push(bytes);
    }
}

fn header(tensors: u64, metadata: u64) -> Bytes {
    let mut bytes = Bytes::new();
    number(&mut bytes, 0x4655_4747, 4);
    number(&mut bytes, 3, 4);
    number(&mut bytes, tensors, 8);
    number(&mut bytes, metadata, 8);
    bytes
}

fn array(element_type: u64, elements: u64, values: &[u8]) -> Bytes {
    let mut bytes = Bytes::new();
    number(&mut bytes, element_type, 4);
    number(&mut bytes, elements, 8);
    bytes.extend_from_slice(values);
    bytes
}

fn strings(values: &[&[u8]]) -> Bytes {
    let mut bytes = array(8, values.len() as u64, &[]);
    for value in values {
        string(&mut bytes, value);
    }
    bytes
}

fn nested(levels: usize) -> Bytes {
    let mut bytes = Bytes::new();
    for _ in 0..levels {
        number(&mut bytes, 9, 4);
        number(&mut bytes, 1, 8);
    }
    number(&mut bytes, 0, 4);
    number(&mut bytes, 0, 8);
    bytes
}

fn tokenizer_keys() -> Vec<Key> {
    vec![
        key("general.architecture", 8, text("toy")),
        key(
            "tokenizer.ggml.tokens",
            9,
            strings(&[b"<|im_start|>", "Ġthe".as_bytes(), "🦉".as_bytes(), b""]),
        ),
        key(
            "tokenizer.ggml.merges",
            9,
            strings(&["Ġ t".as_bytes(), b"h e"]),
        ),
        key(
            "tokenizer.ggml.token_type",
            9,
            array(5, 2, &[1, 0, 0, 0, 3, 0, 0, 0]),
        ),
    ]
}

/// Every input `test_gguf.cpp` parses, then edge fixtures around each limit.
fn fixture_cases(cases: &mut Cases) {
    let keys = split(0, 2, 1);
    let metadata_only = fixture(&keys, &[], 32);
    for length in 0..=metadata_only.len() {
        cases.add(
            format!("parsing/metadata-only-{length}"),
            metadata_only[..length].to_vec(),
        );
    }
    let base = default_fixture();
    cases.add("parsing/valid", base.clone());
    let mut corrupt = base.clone();
    corrupt[0] = 0;
    cases.add("parsing/magic", corrupt);
    let mut corrupt = base.clone();
    replace(&mut corrupt, 4, 2, 4);
    cases.add("parsing/version-2", corrupt.clone());
    replace(&mut corrupt, 4, 0x0300_0000, 4);
    cases.add("parsing/version-byte-order", corrupt);
    let mut corrupt = base.clone();
    replace(&mut corrupt, 8, 100_001, 8);
    cases.add("parsing/tensor-count", corrupt.clone());
    replace(&mut corrupt, 8, 1, 8);
    replace(&mut corrupt, 16, u64::MAX, 8);
    cases.add("parsing/metadata-count", corrupt);

    cases.add(
        "metadata/model",
        keys_fixture(&[
            key("general.architecture", 8, text("toy")),
            key("test.count", 2, integer(42, 2)),
            key("test.negative", 5, integer(0xffff_ffff, 4)),
            key("test.bool", 7, integer(1, 1)),
        ]),
    );
    for name in ["", "Bad", ".bad", "bad.", "bad..key", "bad-key"] {
        cases.add(
            format!("metadata/key-{name:?}"),
            keys_fixture(&[key(name, 0, integer(0, 1))]),
        );
    }
    cases.add(
        "metadata/duplicate",
        keys_fixture(&[key("a", 0, integer(0, 1)), key("a", 0, integer(1, 1))]),
    );
    cases.add(
        "metadata/bool-2",
        keys_fixture(&[key("a", 7, integer(2, 1))]),
    );
    cases.add(
        "metadata/type-13",
        keys_fixture(&[key("a", 13, Bytes::new())]),
    );
    let invalid: [&[u8]; 6] = [
        b"\xc0\x80",
        b"\xed\xa0\x80",
        b"\xf4\x90\x80\x80",
        b"\xe0\x80\x80",
        b"\xc2",
        b"\xf5\x80\x80\x80",
    ];
    for (i, value) in invalid.iter().enumerate() {
        cases.add(
            format!("metadata/utf8-{i}"),
            keys_fixture(&[key("a", 8, text(value))]),
        );
    }
    cases.add(
        "metadata/unicode",
        keys_fixture(&[key("a", 8, text("α🦉"))]),
    );
    let first_second = strings(&[b"first", b"second"]);
    cases.add(
        "metadata/string-array",
        keys_fixture(&[key("a", 9, first_second.clone())]),
    );
    let mut huge = first_second;
    replace(&mut huge, 4, u64::MAX, 8);
    cases.add("metadata/array-count", keys_fixture(&[key("a", 9, huge)]));
    cases.add(
        "metadata/bool-array",
        keys_fixture(&[key("a", 9, array(7, 1, &[2]))]),
    );
    cases.add(
        "metadata/nesting-10",
        keys_fixture(&[key("a", 9, nested(10))]),
    );
    for alignment in [0, 3, 65537] {
        cases.add(
            format!("metadata/alignment-{alignment}"),
            keys_fixture(&[key("general.alignment", 4, integer(alignment, 4))]),
        );
    }
    cases.add(
        "metadata/alignment-64",
        fixture(
            &[key("general.alignment", 4, integer(64, 4))],
            &[Data::default()],
            64,
        ),
    );

    let mut huge_offset = base.clone();
    replace(&mut huge_offset, 54, 0u64.wrapping_sub(32), 8);
    cases.add("tensors/huge-offset", huge_offset);
    let bad = [
        data("", &[4], 0, 0, 16),
        data("a", &[], 0, 0, 16),
        data("a", &[1, 1, 1, 1, 1], 0, 0, 16),
        data("a", &[0], 0, 0, 16),
        data("a", &[3], 12, 0, 16),
        data("a", &[1], 999, 0, 16),
        data("a", &[4], 0, 1, 16),
        data("a", &[5], 0, 0, 16),
        data("a", &[u64::MAX, 2], 0, 0, 0),
        data("a b", &[4], 0, 0, 16),
    ];
    for (i, tensor) in bad.into_iter().enumerate() {
        cases.add(format!("tensors/bad-{i}"), fixture(&[], &[tensor], 32));
    }
    cases.add(
        "tensors/duplicate",
        fixture(&[], &[Data::default(), Data::default()], 32),
    );
    cases.add(
        "tensors/overlap",
        fixture(&[], &[Data::default(), data("second", &[4], 0, 0, 16)], 32),
    );
    cases.add(
        "tensors/separated",
        fixture(&[], &[Data::default(), data("second", &[4], 0, 32, 16)], 32),
    );
    for [id, elements, bytes] in [
        [0, 1, 4],
        [1, 1, 2],
        [8, 32, 34],
        [12, 256, 144],
        [13, 256, 176],
        [14, 256, 210],
        [20, 32, 18],
        [23, 256, 136],
        [30, 1, 2],
    ] {
        cases.add(
            format!("tensors/block-{id}"),
            fixture(
                &[],
                &[data("weight", &[elements, 3, 2], id as u32, 0, bytes * 6)],
                32,
            ),
        );
    }

    cases.add(
        "typed/floats",
        keys_fixture(&[
            key("eps", 6, integer(u64::from(1e-6f32.to_bits()), 4)),
            key("base", 12, integer(10_000_000.0f64.to_bits(), 8)),
            key("nan", 6, integer(0x7fc0_0000, 4)),
            key("inf", 12, integer(0x7ff0_0000_0000_0000, 8)),
        ]),
    );
    for (r#type, width) in [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (5, 4),
        (10, 8),
        (11, 8),
    ] {
        let mut values = array(r#type, 3, &[]);
        number(&mut values, 0, width);
        number(&mut values, 1, width);
        number(&mut values, 42, width);
        cases.add(
            format!("typed/array-{type}"),
            keys_fixture(&[key("a", 9, values.clone())]),
        );
        if matches!(r#type, 1 | 3 | 5 | 11) {
            replace(&mut values, 12, u64::MAX, width);
            cases.add(
                format!("typed/negative-{type}"),
                keys_fixture(&[key("a", 9, values)]),
            );
        }
    }
    cases.add(
        "typed/full-width",
        keys_fixture(&[key("a", 9, array(10, 1, &u64::MAX.to_le_bytes()))]),
    );
    cases.add(
        "typed/noninteger",
        keys_fixture(&[key("a", 9, array(6, 0, &[]))]),
    );
    cases.add(
        "typed/empty",
        keys_fixture(&[key("a", 9, array(5, 0, &[]))]),
    );

    for (label, keys, name) in [
        ("files/part0", split(0, 2, 2), "first"),
        ("files/part1", split(1, 2, 2), "second"),
        ("files/bad", split(1, 2, 3), "second"),
        ("files/duplicate", split(1, 2, 2), "first"),
    ] {
        cases.add(label, fixture(&keys, &[data(name, &[4], 0, 0, 16)], 32));
    }
    cases.add(
        "files/plain",
        keys_fixture(&[key("general.architecture", 8, text("toy"))]),
    );

    edge_cases(cases);
}

fn edge_cases(cases: &mut Cases) {
    // general.alignment in every scalar type.
    for (r#type, value) in [
        (0, integer(64, 1)),
        (1, integer(64, 1)),
        (2, integer(64, 2)),
        (3, integer(64, 2)),
        (4, integer(64, 4)),
        (5, integer(64, 4)),
        (10, integer(64, 8)),
        (11, integer(64, 8)),
        (1, integer(0x80, 1)),
        (5, integer(0xffff_ffff, 4)),
        (11, integer(u64::MAX, 8)),
        (6, integer(u64::from(64.0f32.to_bits()), 4)),
        (12, integer(64.0f64.to_bits(), 8)),
        (7, integer(1, 1)),
        (8, text("64")),
        (9, array(4, 1, &[64, 0, 0, 0])),
    ] {
        cases.add(
            format!("edge/alignment-type-{type}"),
            fixture(
                &[key("general.alignment", r#type, value)],
                &[Data::default()],
                64,
            ),
        );
    }
    for alignment in [1, 2, 128, 65536] {
        cases.add(
            format!("edge/alignment-{alignment}"),
            fixture(
                &[key("general.alignment", 4, integer(alignment, 4))],
                &[Data::default()],
                alignment,
            ),
        );
    }
    cases.add(
        "edge/alignment-128-padded-to-32",
        fixture(
            &[key("general.alignment", 4, integer(128, 4))],
            &[Data::default()],
            32,
        ),
    );
    cases.add(
        "edge/alignment-duplicate",
        keys_fixture(&[
            key("general.alignment", 4, integer(32, 4)),
            key("general.alignment", 4, integer(32, 4)),
        ]),
    );
    cases.add(
        "edge/unaligned-for-64",
        fixture(
            &[key("general.alignment", 4, integer(64, 4))],
            &[data("weight", &[4], 0, 32, 16)],
            64,
        ),
    );
    cases.add("edge/nesting-8", keys_fixture(&[key("a", 9, nested(8))]));
    cases.add("edge/nesting-9", keys_fixture(&[key("a", 9, nested(9))]));
    cases.add(
        "edge/bool-array",
        keys_fixture(&[key("a", 9, array(7, 3, &[0, 1, 1]))]),
    );
    cases.add(
        "edge/empty-strings",
        keys_fixture(&[key("a", 9, strings(&[]))]),
    );
    let mut inner = Bytes::new();
    for _ in 0..2 {
        inner.extend_from_slice(&strings(&[b"x"]));
    }
    cases.add(
        "edge/array-of-string-arrays",
        keys_fixture(&[key("a", 9, array(9, 2, &inner))]),
    );
    cases.add(
        "edge/empty-nested-arrays",
        keys_fixture(&[key("a", 9, array(9, 0, &[]))]),
    );
    cases.add(
        "edge/f64-array",
        keys_fixture(&[key("a", 9, array(12, 2, &[0; 16]))]),
    );
    cases.add(
        "edge/array-type-13",
        keys_fixture(&[key("a", 9, array(13, 0, &[]))]),
    );
    cases.add(
        "edge/key-65535",
        keys_fixture(&[key(vec![b'a'; 65535], 0, integer(0, 1))]),
    );
    cases.add(
        "edge/key-65536",
        keys_fixture(&[key(vec![b'a'; 65536], 0, integer(0, 1))]),
    );
    cases.add(
        "edge/key-grammar",
        keys_fixture(&[
            key("a_1.b2", 0, integer(0, 1)),
            key("_", 0, integer(0, 1)),
            key("0.9", 0, integer(0, 1)),
        ]),
    );
    cases.add(
        "edge/value-type-max",
        keys_fixture(&[key("a", u32::MAX, Bytes::new())]),
    );
    cases.add(
        "edge/strings",
        keys_fixture(&[
            key("nul", 8, text(b"a\0b")),
            key("max", 8, text("\u{10ffff}")),
            key("empty", 8, text("")),
        ]),
    );
    for (label, name) in [
        ("edge/name-64", vec![b'n'; 64]),
        ("edge/name-65", vec![b'n'; 65]),
        ("edge/name-bang", b"!".to_vec()),
        ("edge/name-tilde", b"~".to_vec()),
        ("edge/name-del", b"\x7f".to_vec()),
        ("edge/name-latin", "é".as_bytes().to_vec()),
        ("edge/name-invalid", b"\xff".to_vec()),
    ] {
        cases.add(label, fixture(&[], &[data(name, &[4], 0, 0, 16)], 32));
    }
    cases.add(
        "edge/rank-4",
        fixture(&[], &[data("w", &[1, 2, 3, 4], 0, 0, 96)], 32),
    );
    for id in [
        0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23, 24, 25, 26, 27, 28, 30,
    ] {
        let (elements, bytes) = match id {
            2 | 3 | 6 | 7 | 8 | 20 => (32, 34),
            10..=14 | 23 => (256, 210),
            _ => (1, 8),
        };
        cases.add(
            format!("edge/encoding-{id}"),
            fixture(&[], &[data("w", &[elements], id, 0, bytes)], 32),
        );
    }
    for id in [4, 5, 9, 15, 19, 21, 22, 29, 31, u32::MAX] {
        cases.add(
            format!("edge/unknown-encoding-{id}"),
            fixture(&[], &[data("w", &[1], id, 0, 16)], 32),
        );
    }
    cases.add(
        "edge/row-33",
        fixture(&[], &[data("w", &[33], 8, 0, 64)], 32),
    );
    cases.add(
        "edge/adjacent",
        fixture(
            &[],
            &[data("a", &[8], 0, 0, 32), data("b", &[8], 0, 32, 32)],
            32,
        ),
    );
    cases.add(
        "edge/overlap-by-four",
        fixture(
            &[],
            &[data("a", &[9], 0, 0, 36), data("b", &[8], 0, 32, 32)],
            32,
        ),
    );
    cases.add(
        "edge/unsorted",
        fixture(
            &[],
            &[
                data("c", &[4], 0, 64, 16),
                data("a", &[4], 0, 0, 16),
                data("b", &[4], 0, 32, 16),
            ],
            32,
        ),
    );
    cases.add(
        "edge/data-one-short",
        fixture(&[], &[data("w", &[4], 0, 0, 15)], 32),
    );
    let mut empty = header(0, 0);
    cases.add("edge/no-records-24", empty.clone());
    empty.resize(32, 0);
    cases.add("edge/no-records-32", empty);
    cases.add(
        "edge/one-record",
        fixture(&[key("a", 0, integer(0, 1))], &[], 32),
    );
    let mut counts = header(1, 0);
    counts.resize(57, 0);
    cases.add("edge/counts-exact", counts.clone());
    counts.truncate(56);
    cases.add("edge/counts-short", counts);
    cases.add(
        "edge/floats",
        keys_fixture(&[
            key("negative-zero", 6, integer(0x8000_0000, 4)),
            key("subnormal", 6, integer(1, 4)),
            key("max", 12, integer(f64::MAX.to_bits(), 8)),
            key("payload-nan", 6, integer(0x7fc0_0001, 4)),
            key("negative-inf", 12, integer(0xfff0_0000_0000_0000, 8)),
            key("inf", 6, integer(0x7f80_0000, 4)),
        ]),
    );
    cases.add(
        "edge/integers",
        keys_fixture(&[
            key("u64", 10, integer(u64::MAX, 8)),
            key("i64", 11, integer(u64::MAX, 8)),
            key("i8-max", 1, integer(0x7f, 1)),
            key("i8-min", 1, integer(0x80, 1)),
            key("i16-min", 3, integer(0x8000, 2)),
            key("u8", 0, integer(255, 1)),
        ]),
    );
    cases.add("edge/tokenizer", keys_fixture(&tokenizer_keys()));
}

/// Every prefix, byte rewrite and field rewrite of `bytes`.
fn sweeps(cases: &mut Cases, label: &str, bytes: &[u8]) {
    for length in 0..bytes.len() {
        cases.add(format!("{label}/prefix-{length}"), bytes[..length].to_vec());
    }
    for offset in 0..bytes.len() {
        let original = bytes[offset];
        for value in [0, 255, original.wrapping_add(1), original ^ 0x80] {
            let mut changed = bytes.to_vec();
            changed[offset] = value;
            cases.add(format!("{label}/byte-{offset}-{value}"), changed);
        }
        for value in [u64::MAX, 1 << 63, 1 << 32, u64::MAX - 31] {
            if offset + 8 <= bytes.len() {
                let mut changed = bytes.to_vec();
                replace(&mut changed, offset, value, 8);
                cases.add(format!("{label}/u64-{offset}-{value:x}"), changed);
            }
        }
        for value in [0xffff_ffff, 0x8000_0000, 13] {
            if offset + 4 <= bytes.len() {
                let mut changed = bytes.to_vec();
                replace(&mut changed, offset, value, 4);
                cases.add(format!("{label}/u32-{offset}-{value:x}"), changed);
            }
        }
    }
}

fn synthetic_cases() -> Cases {
    let mut cases = Cases::new();
    fixture_cases(&mut cases);
    for (i, seed) in fuzz_seeds().iter().enumerate() {
        sweeps(&mut cases, &format!("seed-{i}"), seed);
    }
    sweeps(&mut cases, "tokenizer", &keys_fixture(&tokenizer_keys()));
    cases
}

fn write_pack(path: &Path, inputs: &[Bytes]) {
    let mut pack = Vec::new();
    for input in inputs {
        pack.extend_from_slice(&(input.len() as u64).to_le_bytes());
        pack.extend_from_slice(input);
    }
    std::fs::write(path, pack).unwrap();
}

fn write_list(path: &Path, lines: &[String]) {
    let mut text = String::new();
    for line in lines {
        assert!(!line.contains('\n'));
        text.push_str(line);
        text.push('\n');
    }
    std::fs::write(path, text).unwrap();
}

fn path_text(path: &Path) -> String {
    path.to_str().unwrap().to_owned()
}

fn driver(driver: &Path, arguments: &[&OsStr]) -> String {
    let output = Command::new(driver).args(arguments).output().unwrap();
    assert!(
        output.status.success(),
        "driver failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// Splits a summary into its cases (each starting with `case N`).
fn split_cases(text: &str) -> Vec<&str> {
    let mut starts: Vec<usize> = text
        .match_indices("case ")
        .map(|(index, _)| index)
        .filter(|&index| index == 0 || text.as_bytes()[index - 1] == b'\n')
        .collect();
    starts.push(text.len());
    starts
        .windows(2)
        .map(|pair| &text[pair[0]..pair[1]])
        .collect()
}

#[derive(Default)]
struct Tally {
    cases: usize,
    accepted: usize,
    refused: BTreeMap<String, usize>,
    differences: Vec<String>,
}

fn compare(out: &Path, mode: &str, labels: &[String], rust: &str, cpp: &str) -> Tally {
    std::fs::write(out.join(format!("rust-{mode}.txt")), rust).unwrap();
    std::fs::write(out.join(format!("cpp-{mode}.txt")), cpp).unwrap();
    let rust_cases = split_cases(rust);
    let cpp_cases = split_cases(cpp);
    let mut tally = Tally {
        cases: labels.len(),
        ..Tally::default()
    };
    if rust_cases.len() != labels.len() || cpp_cases.len() != labels.len() {
        tally.differences.push(format!(
            "{mode}: {} cases, Rust printed {}, C++ printed {}",
            labels.len(),
            rust_cases.len(),
            cpp_cases.len()
        ));
    }
    for (i, (ours, theirs)) in rust_cases.iter().zip(&cpp_cases).enumerate() {
        let first = ours.lines().nth(1).unwrap_or("");
        if let Some(message) = first.strip_prefix("error ") {
            *tally.refused.entry(message.to_owned()).or_default() += 1;
        } else if first.starts_with("ok ") {
            tally.accepted += 1;
        }
        if ours != theirs {
            let line = ours
                .lines()
                .zip(theirs.lines())
                .find(|(a, b)| a != b)
                .map_or_else(
                    || "(different line counts)".to_owned(),
                    |(a, b)| format!("Rust `{a}`, C++ `{b}`"),
                );
            tally
                .differences
                .push(format!("{mode} case {i} ({}): {line}", labels[i]));
        }
    }
    tally
}

/// `try_input` in `fuzz_gguf.cpp`: parse, read every tensor, read every string.
fn still_valid(input: &[u8]) -> bool {
    let Ok(file) = Gguf::from_bytes(input) else {
        return false;
    };
    let mut buffer = Vec::new();
    file.tensors()
        .iter()
        .all(|tensor| file.tensor_bytes(tensor, &mut buffer).is_ok())
        && file
            .metadata()
            .iter()
            .filter(|(_, item)| item.r#type == 8)
            .all(|(key, _)| file.string_value(key).is_ok())
}

/// Writes `header`, then extends the file to `size` without writing the rest.
fn sparse(path: &Path, header: &[u8], size: u64) {
    let mut file = File::create(path).unwrap();
    file.write_all(header).unwrap();
    file.set_len(size).unwrap();
}

/// Files near the header, array and record limits and past 4 GiB. Sparse, so they cost no disk space; removed after the run.
fn sparse_cases(directory: &Path) -> Vec<(String, PathBuf)> {
    std::fs::create_dir_all(directory).unwrap();
    let mut cases = Vec::new();
    let mut add = |label: &str, header: Bytes, size: u64| {
        let path = directory.join(format!("{label}.gguf"));
        sparse(&path, &header, size);
        cases.push((format!("sparse/{label}"), path));
    };
    let (bytes, payload) = fixture_header(&[], &[data("weight", &[1 << 28], 0, 0, 1 << 30)], 32);
    let size = bytes.len() as u64 + payload;
    add("tensor-1gib", bytes, size);
    let (bytes, payload) = fixture_header(
        &[],
        &[
            data("near", &[4], 0, 0, 16),
            data("far", &[4], 0, 1 << 32, 16),
        ],
        32,
    );
    let size = bytes.len() as u64 + payload;
    add("tensor-past-4gib", bytes, size);
    // A string value of `length` zero bytes under key "a", after a 24-byte header, 9-byte key, 4-byte type and 8-byte length: 45 bytes.
    let long_string = |tensors: u64, length: u64| {
        let mut bytes = header(tensors, 1);
        string(&mut bytes, b"a");
        number(&mut bytes, 8, 4);
        number(&mut bytes, length, 8);
        bytes
    };
    add(
        "string-to-limit",
        long_string(0, HEADER_LIMIT - 45),
        HEADER_LIMIT,
    );
    add(
        "string-past-limit",
        long_string(0, HEADER_LIMIT - 44),
        HEADER_LIMIT + 64,
    );
    add(
        "string-over-limit",
        long_string(0, HEADER_LIMIT + 1),
        HEADER_LIMIT + 64,
    );
    add(
        "tensor-after-limit",
        long_string(1, HEADER_LIMIT - 45),
        HEADER_LIMIT + (1 << 20),
    );
    let mut bytes = header(0, 1);
    string(&mut bytes, b"a");
    number(&mut bytes, 9, 4);
    bytes.extend_from_slice(&array(0, 16 << 20, &[]));
    let start = bytes.len() as u64;
    add(
        "array-at-limit",
        bytes.clone(),
        (start + (16 << 20)).next_multiple_of(32),
    );
    replace(&mut bytes, 41, (16 << 20) + 1, 8);
    add("array-over-limit", bytes, start + (17 << 20));
    add("tensor-records-zeroed", header(100_000, 0), 4 << 20);
    add("metadata-records-zeroed", header(0, 100_000), 4 << 20);
    add("records-exceed-file", header(100_000, 0), 1 << 20);
    cases
}

fn model_cases(directory: &Path) -> Vec<(String, Vec<PathBuf>)> {
    std::fs::create_dir_all(directory).unwrap();
    let write = |name: &str, bytes: Bytes| {
        let path = directory.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    };
    let shard = |name: &str, keys: &[Key], tensor: &str| {
        write(name, fixture(keys, &[data(tensor, &[4], 0, 0, 16)], 32))
    };
    let p0 = shard("part0.gguf", &split(0, 2, 2), "first");
    let p1 = shard("part1.gguf", &split(1, 2, 2), "second");
    let plain = write(
        "plain.gguf",
        keys_fixture(&[key("general.architecture", 8, text("toy"))]),
    );
    let plain2 = write(
        "plain2.gguf",
        fixture(
            &[key("general.architecture", 8, text("toy"))],
            &[data("other", &[4], 0, 0, 16)],
            32,
        ),
    );
    let bad = shard("bad.gguf", &split(1, 2, 3), "second");
    let mut other_keys = split(1, 2, 2);
    other_keys[0].value = text("other");
    let other = shard("other.gguf", &other_keys, "second");
    let duplicate = shard("duplicate.gguf", &split(1, 2, 2), "first");
    let anonymous = write("anonymous.gguf", default_fixture());
    let mut tokenized = split(0, 4, 4);
    tokenized.extend(tokenizer_keys().into_iter().skip(1));
    let q0 = shard("q0.gguf", &tokenized, "t0");
    let q1 = shard("q1.gguf", &split(1, 4, 4), "t1");
    let q2 = shard("q2.gguf", &split(2, 4, 4), "t2");
    let q3 = shard("q3.gguf", &split(3, 4, 4), "t3");
    let r0 = shard("r0.gguf", &split(0, 3, 3), "r0");
    let r1 = shard("r1.gguf", &split(1, 3, 3), "r1");
    let r2 = shard("r2.gguf", &split(2, 3, 3), "r2");
    let mut keys = split(1, 2, 2);
    keys[2] = key("split.count", 8, text("2"));
    let count_string = shard("count-string.gguf", &keys, "second");
    let mut keys = split(1, 2, 2);
    keys[1] = key("split.no", 3, integer(0xffff, 2));
    let number_negative = shard("number-negative.gguf", &keys, "second");
    let only_number = shard(
        "only-number.gguf",
        &[
            key("general.architecture", 8, text("test")),
            key("split.no", 2, integer(1, 2)),
        ],
        "second",
    );
    let declared_three = [
        shard("d0.gguf", &split(0, 2, 3), "d0"),
        shard("d1.gguf", &split(1, 2, 3), "d1"),
    ];
    let number_two = shard("number-two.gguf", &split(2, 2, 2), "second");
    let single_split = shard("single-split.gguf", &split(0, 1, 1), "only");
    let mut empty_keys = split(0, 2, 2);
    empty_keys[0].value = text("");
    let empty_architecture = shard("empty-architecture.gguf", &empty_keys, "first");
    let mut wrong_keys = split(1, 2, 2);
    wrong_keys[0] = key("general.architecture", 4, integer(1, 4));
    let wrong_architecture = shard("wrong-architecture.gguf", &wrong_keys, "second");
    let no_architecture = shard("no-architecture.gguf", &split(1, 2, 2)[1..], "second");
    let mut tokens_only = split(0, 2, 2);
    tokens_only.push(tokenizer_keys()[1].clone());
    let tokens_only = shard("tokens-only.gguf", &tokens_only, "first");
    let mut tokens_wrong = split(0, 2, 2);
    tokens_wrong.push(key("tokenizer.ggml.tokens", 9, array(4, 1, &[1, 0, 0, 0])));
    let tokens_wrong = shard("tokens-wrong.gguf", &tokens_wrong, "first");
    let broken = write("broken.gguf", default_fixture()[..30].to_vec());
    let link = directory.join("link0.gguf");
    if !link.exists() {
        symlink(&p0, &link).unwrap();
    }
    let missing = directory.join("missing.gguf");
    let mut cases = vec![
        ("valid-reversed", vec![p1.clone(), p0.clone()]),
        ("valid-ordered", vec![p0.clone(), p1.clone()]),
        ("empty", vec![]),
        ("duplicate-path", vec![p0.clone(), p0.clone()]),
        ("incomplete", vec![p0.clone()]),
        ("mixed", vec![p0.clone(), plain.clone()]),
        ("mixed-plain-first", vec![plain.clone(), p0.clone()]),
        ("total-mismatch", vec![p0.clone(), bad]),
        ("architecture-mismatch", vec![p0.clone(), other]),
        ("duplicate-tensor", vec![p0.clone(), duplicate]),
        ("plain", vec![plain.clone()]),
        ("two-plain", vec![plain.clone(), plain2]),
        ("anonymous", vec![anonymous]),
        ("four", vec![q2.clone(), q0.clone(), q3.clone(), q1.clone()]),
        ("four-missing-one", vec![q0.clone(), q1.clone(), q2.clone()]),
        ("three", vec![r0, r1, r2]),
        ("count-string", vec![p0.clone(), count_string]),
        ("number-negative", vec![p0.clone(), number_negative]),
        ("only-number", vec![p0.clone(), only_number]),
        ("declared-three", declared_three.to_vec()),
        ("number-two", vec![p0.clone(), number_two]),
        ("single-split", vec![single_split]),
        (
            "empty-architecture-first",
            vec![empty_architecture.clone(), p1.clone()],
        ),
        (
            "empty-architecture-second",
            vec![p1.clone(), empty_architecture],
        ),
        ("wrong-architecture", vec![p0.clone(), wrong_architecture]),
        ("architecture-in-one", vec![p0.clone(), no_architecture]),
        ("tokens-only", vec![tokens_only, p1.clone()]),
        ("tokens-wrong", vec![tokens_wrong, p1.clone()]),
        ("broken-shard", vec![p0.clone(), broken]),
        ("missing-shard", vec![p0.clone(), missing.clone()]),
        ("directory-shard", vec![p0.clone(), directory.to_path_buf()]),
        ("symlinked", vec![link, p1.clone()]),
        ("too-many", vec![missing.clone(); 1025]),
        ("at-limit", vec![missing; 1024]),
    ];
    cases.push(("tokenized-reversed", vec![q3, q2, q1, q0]));
    cases
        .into_iter()
        .map(|(label, paths)| (format!("models/{label}"), paths))
        .collect()
}

fn mapped_cases(directory: &Path) -> Vec<(String, PathBuf)> {
    std::fs::create_dir_all(directory).unwrap();
    let regular = directory.join("regular.bin");
    std::fs::write(&regular, b"tokenizer bytes").unwrap();
    let empty = directory.join("empty.bin");
    std::fs::write(&empty, b"").unwrap();
    let locked = directory.join("locked.bin");
    std::fs::write(&locked, b"locked").unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    let fifo = directory.join("fifo");
    if !fifo.exists() {
        assert!(
            Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
    }
    let mut links = Vec::new();
    for (name, target) in [
        ("link-regular", regular.clone()),
        ("link-directory", directory.to_path_buf()),
        ("link-dangling", directory.join("absent")),
    ] {
        let link = directory.join(name);
        if !link.exists() && std::fs::symlink_metadata(&link).is_err() {
            symlink(&target, &link).unwrap();
        }
        links.push((name, link));
    }
    let mut cases = vec![
        ("regular", regular.clone()),
        ("empty", empty),
        ("locked", locked),
        ("fifo", fifo),
        ("directory", directory.to_path_buf()),
        ("missing", directory.join("missing.bin")),
        ("dev-null", PathBuf::from("/dev/null")),
        ("empty-path", PathBuf::new()),
        ("file-as-directory", regular.join("inner")),
    ];
    cases.extend(links);
    cases
        .into_iter()
        .map(|(label, path)| (format!("mapped/{label}"), path))
        .collect()
}

#[test]
#[ignore = "needs the C++ driver; run rust/tools/diff/gguf.sh"]
fn matches_the_cpp_reader() {
    let driver_path = env_path(DRIVER);
    let out = env_path(OUT);
    let cases_root = out.join("cases");
    if cases_root.exists() {
        std::fs::remove_dir_all(&cases_root).unwrap();
    }
    std::fs::create_dir_all(cases_root.join("files")).unwrap();
    let mut report = String::new();
    let mut differences = Vec::new();
    let started = Instant::now();

    // Seeds: the Rust corpus must be byte-identical to the C++ one.
    let cpp_seeds = env_path(CPP_SEEDS);
    for (i, seed) in fuzz_seeds().iter().enumerate() {
        let theirs = std::fs::read(cpp_seeds.join(format!("seed-{i}"))).unwrap();
        if &theirs != seed {
            differences.push(format!("seed-{i} differs from the C++ corpus"));
        }
    }

    // In-memory inputs: Gguf::from_bytes against Gguf(std::span).
    let synthetic = synthetic_cases();
    let seeds = fuzz_seeds();
    let mutated = mutations(&seeds, MUTATION_ROUNDS);
    let mut memory = Cases::new();
    for (label, input) in synthetic.labels.iter().zip(&synthetic.inputs) {
        memory.add(label.clone(), input.clone());
    }
    for (i, input) in mutated.iter().enumerate() {
        memory.add(format!("mutation-{i}"), input.clone());
    }
    let pack = out.join("cases/memory.pack");
    write_pack(&pack, &memory.inputs);
    let mut rust = String::new();
    for (i, input) in memory.inputs.iter().enumerate() {
        push(&mut rust, format!("case {i}"));
        summary::gguf(&mut rust, Gguf::from_bytes(input));
    }
    let cpp = driver(&driver_path, &[OsStr::new("pack"), pack.as_os_str()]);
    let tally = compare(&out, "memory", &memory.labels, &rust, &cpp);
    report_tally(&mut report, "memory (Gguf::from_bytes)", &tally);
    differences.extend(tally.differences);

    // The fuzz_gguf acceptance count over the same 50,000 mutations.
    let accepted = mutated.iter().filter(|input| still_valid(input)).count();
    let cpp_fuzz = std::fs::read_to_string(env_path(CPP_FUZZ)).unwrap();
    let cpp_accepted: usize = cpp_fuzz
        .split("still valid: ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|count| count.parse().ok())
        .unwrap();
    push(
        &mut report,
        format!(
            "fuzz_gguf mutations: {MUTATION_ROUNDS} rounds, still valid: Rust {accepted}, C++ {cpp_accepted}"
        ),
    );
    if accepted != cpp_accepted {
        differences.push(format!(
            "fuzz acceptance: Rust {accepted}, C++ {cpp_accepted}"
        ));
    }

    // Files: Gguf::from_path against Gguf(path).
    let mut file_labels = Vec::new();
    let mut file_paths = Vec::new();
    let file_inputs = synthetic.labels.iter().zip(&synthetic.inputs).chain(
        memory.labels[synthetic.labels.len()..]
            .iter()
            .zip(&mutated)
            .take(MUTATION_FILES),
    );
    for (i, (label, input)) in file_inputs.enumerate() {
        let path = cases_root.join(format!("files/{i:05}.gguf"));
        std::fs::write(&path, input).unwrap();
        file_labels.push(label.clone());
        file_paths.push(path);
    }
    for (label, path) in sparse_cases(&cases_root.join("sparse")) {
        file_labels.push(label);
        file_paths.push(path);
    }
    let list = out.join("cases/files.txt");
    write_list(
        &list,
        &file_paths
            .iter()
            .map(|path| path_text(path))
            .collect::<Vec<_>>(),
    );
    let mut rust = String::new();
    let mut loaded = Vec::new();
    for (i, path) in file_paths.iter().enumerate() {
        push(&mut rust, format!("case {i}"));
        let result = Gguf::from_path(path);
        if let Ok(file) = &result {
            loaded.push((
                file_labels[i].clone(),
                file.loaded_bytes(),
                file.file_size(),
            ));
        }
        summary::gguf(&mut rust, result);
    }
    let cpp = driver(&driver_path, &[OsStr::new("files"), list.as_os_str()]);
    let tally = compare(&out, "files", &file_labels, &rust, &cpp);
    report_tally(&mut report, "files (Gguf::from_path)", &tally);
    differences.extend(tally.differences);
    for (label, loaded, size) in loaded
        .iter()
        .filter(|(label, ..)| label.starts_with("sparse/"))
    {
        push(
            &mut report,
            format!("  {label}: read {loaded} of {size} bytes"),
        );
    }

    // Shard sets: ModelFiles::new against ModelFiles(paths).
    let models = model_cases(&cases_root.join("models"));
    let labels: Vec<String> = models.iter().map(|(label, _)| label.clone()).collect();
    let lines: Vec<String> = models
        .iter()
        .map(|(_, paths)| {
            paths
                .iter()
                .map(|path| path_text(path))
                .collect::<Vec<_>>()
                .join("\t")
        })
        .collect();
    let list = out.join("cases/models.txt");
    write_list(&list, &lines);
    let mut rust = String::new();
    for (i, (_, paths)) in models.iter().enumerate() {
        push(&mut rust, format!("case {i}"));
        summary::model(&mut rust, ModelFiles::new(paths));
    }
    let cpp = driver(&driver_path, &[OsStr::new("models"), list.as_os_str()]);
    let tally = compare(&out, "models", &labels, &rust, &cpp);
    report_tally(&mut report, "models (ModelFiles::new)", &tally);
    differences.extend(tally.differences);

    // Paths: MappedFile::new against MappedFile(path).
    let mapped = mapped_cases(&cases_root.join("mapped"));
    let labels: Vec<String> = mapped.iter().map(|(label, _)| label.clone()).collect();
    let list = out.join("cases/mapped.txt");
    write_list(
        &list,
        &mapped
            .iter()
            .map(|(_, path)| path_text(path))
            .collect::<Vec<_>>(),
    );
    let mut rust = String::new();
    for (i, (_, path)) in mapped.iter().enumerate() {
        push(&mut rust, format!("case {i}"));
        summary::mapped(&mut rust, MappedFile::new(path));
    }
    let cpp = driver(&driver_path, &[OsStr::new("mapped"), list.as_os_str()]);
    let tally = compare(&out, "mapped", &labels, &rust, &cpp);
    report_tally(&mut report, "mapped (MappedFile::new)", &tally);
    differences.extend(tally.differences);

    // Sparse files report gigabytes of apparent size; remove them.
    std::fs::remove_dir_all(cases_root.join("sparse")).unwrap();
    push(
        &mut report,
        format!(
            "differences: {} (elapsed {:.1?})",
            differences.len(),
            started.elapsed()
        ),
    );
    for difference in differences.iter().take(50) {
        push(&mut report, format!("  {difference}"));
    }
    std::fs::write(out.join("report.txt"), &report).unwrap();
    println!("{report}");
    assert!(differences.is_empty(), "{report}");
}

fn report_tally(report: &mut String, title: &str, tally: &Tally) {
    let refused: usize = tally.refused.values().sum();
    push(
        report,
        format!(
            "{title}: {} cases, {} accepted, {refused} refused, {} differences",
            tally.cases,
            tally.accepted,
            tally.differences.len()
        ),
    );
    let mut messages: Vec<(&String, &usize)> = tally.refused.iter().collect();
    messages.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (message, count) in messages {
        push(report, format!("  {count:>6}  {message}"));
    }
}

#[test]
#[ignore = "reads real shard headers; run rust/tools/diff/gguf.sh --real DIR"]
fn real_shard_headers_match_the_cpp_reader() {
    let driver_path = env_path(DRIVER);
    let out = env_path(OUT);
    let directory = env_path(REAL_SHARDS);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "gguf")
        })
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no .gguf files in {}",
        directory.display()
    );

    let started = Instant::now();
    let model = ModelFiles::new(&paths).unwrap();
    let elapsed = started.elapsed();
    let mut report = format!(
        "real shards: {} files, Rust ModelFiles::new {elapsed:.2?}\n",
        paths.len()
    );
    for (path, shard) in paths.iter().zip(model.shards()) {
        push(
            &mut report,
            format!(
                "  {}: read {} of {} bytes (data_offset {}, {} tensors, {} keys)",
                path.file_name().unwrap().to_string_lossy(),
                shard.loaded_bytes(),
                shard.file_size(),
                shard.data_offset(),
                shard.tensors().len(),
                shard.metadata().len()
            ),
        );
    }
    let mut rust = String::from("case 0\n");
    summary::fingerprints(&mut rust, Ok(model));
    let mut arguments = vec![OsStr::new("fingerprints")];
    arguments.extend(paths.iter().map(|path| path.as_os_str()));
    let started = Instant::now();
    let cpp = driver(&driver_path, &arguments);
    push(&mut report, format!("C++ driver {:.2?}", started.elapsed()));
    let labels = vec!["real".to_owned()];
    let tally = compare(&out, "real", &labels, &rust, &cpp);
    for line in rust.lines().filter(|line| line.starts_with("tokenizer ")) {
        push(&mut report, format!("  {line}"));
    }
    push(
        &mut report,
        format!("differences: {}", tally.differences.len()),
    );
    for difference in &tally.differences {
        push(&mut report, format!("  {difference}"));
    }
    std::fs::write(out.join("real-shards.txt"), &report).unwrap();
    println!("{report}");
    assert!(tally.differences.is_empty(), "{report}");
}
