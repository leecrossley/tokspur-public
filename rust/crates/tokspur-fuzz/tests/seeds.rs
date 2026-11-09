//! Seed corpora and libFuzzer dictionaries for the fuzz targets, written by an ignored test, as `tokspur-gguf`'s `seeds::write_fuzz_seeds` writes the GGUF corpus (which `native/rust.mk rust-fuzz` uses for the two GGUF targets):
//!
//! ```text
//! TOKSPUR_FUZZ_SEEDS=DIR cargo test -p tokspur-fuzz --test seeds -- --ignored
//! ```
//!
//! writes `DIR/corpus/<target>/seed-NNNN` and `DIR/dict/<target>.dict`. No seed is written by hand: each one is a string or byte-string literal of the repository's existing Rust tests, extracted mechanically, wrapped in the target's input header where it has one. Dictionary entries are the short, space-free literals of the parsers' own sources (markers, field names, keywords). Everything goes under `build/`, which Git ignores.

#![forbid(unsafe_code)]
// Test code: the writer's helpers fail the test on any IO error.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use tokspur_fuzz::response::manifest;

/// The bytes of one cooked (escaped) literal starting after its opening quote; returns them and the index after the closing quote.
fn cooked(source: &[u8], mut i: usize) -> (Vec<u8>, usize) {
    let mut out = Vec::new();
    while i < source.len() {
        match source[i] {
            b'"' => return (out, i + 1),
            b'\\' if i + 1 < source.len() => {
                let escape = source[i + 1];
                i += 2;
                match escape {
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'0' => out.push(0),
                    b'x' => {
                        let digits = source.get(i..i + 2).unwrap_or_default();
                        let text = std::str::from_utf8(digits).unwrap_or_default();
                        if let Ok(byte) = u8::from_str_radix(text, 16) {
                            out.push(byte);
                        }
                        i += 2;
                    }
                    b'u' => {
                        let close = source[i..].iter().position(|&c| c == b'}').unwrap_or(0);
                        let digits = source.get(i + 1..i + close).unwrap_or_default();
                        let text = std::str::from_utf8(digits).unwrap_or_default();
                        let point = u32::from_str_radix(text, 16).ok().and_then(char::from_u32);
                        if let Some(point) = point {
                            out.extend_from_slice(point.encode_utf8(&mut [0; 4]).as_bytes());
                        }
                        i += close + 1;
                    }
                    b'\n' | b'\r' => {
                        while i < source.len() && source[i].is_ascii_whitespace() {
                            i += 1;
                        }
                    }
                    other => out.push(other),
                }
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    (out, i)
}

/// Skips a character literal (`'x'`, `'\''`, `'日'`) or a lifetime or label starting at the quote at `i`.
fn skip_quote(source: &[u8], i: usize) -> usize {
    if source.get(i + 1) == Some(&b'\\') {
        let close = source[i + 3..].iter().position(|&c| c == b'\'');
        return close.map_or(source.len(), |close| i + 4 + close);
    }
    let width = match source.get(i + 1) {
        Some(&lead) if lead >= 0xf0 => 4,
        Some(&lead) if lead >= 0xe0 => 3,
        Some(&lead) if lead >= 0xc0 => 2,
        _ => 1,
    };
    if source.get(i + 1 + width) == Some(&b'\'') {
        i + 2 + width
    } else {
        i + 1
    }
}

/// Every string, byte-string and raw-string literal in Rust `source`, as bytes, skipping comments and character literals.
fn literals(source: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < source.len() {
        let c = source[i];
        if source[i..].starts_with(b"//") {
            i += source[i..]
                .iter()
                .position(|&c| c == b'\n')
                .unwrap_or(source.len() - i);
        } else if source[i..].starts_with(b"/*") {
            let mut depth = 0;
            while i < source.len() {
                if source[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if source[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
        } else if c == b'\'' {
            i = skip_quote(source, i);
        } else if c == b'"' {
            let (literal, next) = cooked(source, i + 1);
            out.push(literal);
            i = next;
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < source.len() && (source[i].is_ascii_alphanumeric() || source[i] == b'_') {
                i += 1;
            }
            let prefix = &source[start..i];
            match (prefix, source.get(i)) {
                (b"r" | b"br", Some(b'"' | b'#')) => {
                    let hashes = source[i..].iter().take_while(|&&c| c == b'#').count();
                    i += hashes;
                    if source.get(i) != Some(&b'"') {
                        continue;
                    }
                    let mut close = b"\"".to_vec();
                    close.extend(std::iter::repeat_n(b'#', hashes));
                    let body = i + 1;
                    let end = source[body..]
                        .windows(close.len())
                        .position(|window| window == close.as_slice())
                        .map_or(source.len(), |end| body + end);
                    out.push(source[body..end].to_vec());
                    i = (end + close.len()).min(source.len());
                }
                (b"b", Some(b'"')) => {
                    let (literal, next) = cooked(source, i + 1);
                    out.push(literal);
                    i = next;
                }
                (b"b", Some(b'\'')) => i = skip_quote(source, i),
                _ => {}
            }
        } else {
            i += 1;
        }
    }
    out
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// The distinct literals of the given repository files, in first-seen order.
fn literals_of(files: &[&str]) -> Vec<Vec<u8>> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for file in files {
        let path = repository().join(file);
        let source = fs::read(&path).unwrap_or_else(|error| {
            panic!("cannot read {}: {error}", path.display());
        });
        for literal in literals(&source) {
            if !literal.is_empty() && seen.insert(literal.clone()) {
                out.push(literal);
            }
        }
    }
    out
}

/// Replaces the `seed-*` files of one corpus directory.
fn write_corpus(root: &Path, target: &str, seeds: &[Vec<u8>]) {
    let directory = root.join("corpus").join(target);
    fs::create_dir_all(&directory).unwrap();
    for entry in fs::read_dir(&directory).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if name.starts_with("seed-") {
            fs::remove_file(&path).unwrap();
        }
    }
    let unique: BTreeSet<&Vec<u8>> = seeds.iter().collect();
    for (i, seed) in unique.iter().enumerate() {
        fs::write(directory.join(format!("seed-{i:04}")), seed).unwrap();
    }
}

/// libFuzzer dictionary syntax: one quoted entry per line.
fn dictionary_entry(token: &[u8]) -> String {
    let mut line = String::from("\"");
    for &byte in token {
        match byte {
            b'"' => line.push_str("\\\""),
            b'\\' => line.push_str("\\\\"),
            0x20..=0x7e => line.push(char::from(byte)),
            _ => line.push_str(&format!("\\x{byte:02X}")),
        }
    }
    line.push('"');
    line
}

fn write_dictionary(root: &Path, target: &str, tokens: &BTreeSet<Vec<u8>>) {
    let directory = root.join("dict");
    fs::create_dir_all(&directory).unwrap();
    let mut text = String::new();
    for token in tokens {
        text.push_str(&dictionary_entry(token));
        text.push('\n');
    }
    fs::write(directory.join(format!("{target}.dict")), text).unwrap();
}

/// Short, space-free literals of the parsers' sources: markers, field names and keywords rather than messages.
fn tokens_of(files: &[&str], longest: usize) -> BTreeSet<Vec<u8>> {
    literals_of(files)
        .into_iter()
        .filter(|token| token.len() <= longest && !token.contains(&b' '))
        .collect()
}

fn with_header(header: &[u8], seeds: &[Vec<u8>]) -> Vec<Vec<u8>> {
    seeds
        .iter()
        .map(|seed| [header, seed.as_slice()].concat())
        .collect()
}

const WIRE_LEVELS: [&str; 4] = ["verify8", "branch8", "qsa-stage-v8", "fast-math"];

#[test]
#[ignore = "writes corpora and dictionaries into $TOKSPUR_FUZZ_SEEDS"]
fn write_seed_corpora() {
    let root = PathBuf::from(
        std::env::var_os("TOKSPUR_FUZZ_SEEDS")
            .unwrap_or_else(|| panic!("set TOKSPUR_FUZZ_SEEDS to the output directory")),
    );

    let json = literals_of(&[
        "rust/crates/tokspur-json/tests/json.rs",
        "rust/crates/tokspur-json/src/lib.rs",
        "rust/crates/tokspur-text/tests/chat.rs",
        "rust/crates/tokspur-text/tests/response_protocol.rs",
        "rust/crates/tokspur-server/tests/serve_protocol.rs",
    ]);
    write_corpus(&root, "json", &json);
    write_corpus(&root, "json_diff", &json);
    let mut json_tokens = tokens_of(
        &[
            "rust/crates/tokspur-json/src/lib.rs",
            "rust/crates/tokspur-json/src/utf8.rs",
        ],
        16,
    );
    for token in [
        "true", "false", "null", "\\u", "\\ud800", "\\udc00", "\\\\", "\\\"", "e+", "E-", "-0.",
        "{\"", "\":", "[]", "{}",
    ] {
        json_tokens.insert(token.as_bytes().to_vec());
    }
    write_dictionary(&root, "json", &json_tokens);
    write_dictionary(&root, "json_diff", &json_tokens);

    // Header: flags 0 (default limits), separator 0xff (no split); and the same with small, valid limits.
    let requests = literals_of(&["rust/crates/tokspur-server/tests/http.rs"]);
    let mut http = with_header(&[0, 0xff], &requests);
    http.extend(with_header(&[1, 0xff, 255, 255, 9, 255, 255], &requests));
    write_corpus(&root, "http", &http);
    let mut http_tokens = tokens_of(&["rust/crates/tokspur-server/src/http.rs"], 24);
    for token in [
        "\r\n",
        "\r\n\r\n",
        "Host: a",
        "Content-Length: ",
        "HTTP/1.1",
    ] {
        http_tokens.insert(token.as_bytes().to_vec());
    }
    write_dictionary(&root, "http", &http_tokens);

    // Seed both byte and token framing, in text and thinking modes, with separator 0xff.
    let streams = literals_of(&[
        "rust/crates/tokspur-text/tests/response.rs",
        "rust/crates/tokspur-text/tests/response_cli.rs",
        "rust/crates/tokspur-text/tests/literal_markers.rs",
        "rust/crates/tokspur-text/src/response.rs",
    ]);
    let mut response = with_header(&[0, 0xff], &streams);
    response.extend(with_header(&[1, 0xff], &streams));
    response.extend(with_header(&[32, 0xff], &streams));
    response.extend(with_header(&[33, 0xff], &streams));
    write_corpus(&root, "response", &response);
    write_corpus(&root, "response_diff", &response);
    let mut response_tokens = tokens_of(&["rust/crates/tokspur-text/src/response.rs"], 24);
    for choice in 0..4 {
        for tool in manifest(choice) {
            response_tokens.insert(tool.name.clone());
            for parameter in tool.parameters {
                response_tokens.insert(parameter.name);
            }
        }
    }
    write_dictionary(&root, "response", &response_tokens);
    write_dictionary(&root, "response_diff", &response_tokens);

    let texts = literals_of(&[
        "rust/crates/tokspur-text/tests/tokenizer.rs",
        "rust/crates/tokspur-text/src/tokenizer.rs",
    ]);
    write_corpus(&root, "bpe", &texts);
    let bpe_tokens: BTreeSet<Vec<u8>> = tokspur_fuzz::bpe::vocabulary()[256..]
        .iter()
        .cloned()
        .collect();
    write_dictionary(&root, "bpe", &bpe_tokens);

    // Every literal of the private-wire tests that is a JSON document, alone and all together in one array.
    let documents: Vec<Vec<u8>> = literals_of(&[
        "rust/crates/tokspur-server/tests/http_private/verify8.rs",
        "rust/crates/tokspur-server/tests/http_private/branch8.rs",
        "rust/crates/tokspur-server/tests/http_private/fast_math.rs",
        "rust/crates/tokspur-wire/tests/owned_worker.rs",
        "rust/crates/tokspur-server/tests/serve_protocol.rs",
    ])
    .into_iter()
    .filter(|literal| tokspur_json::parse_json(literal).is_ok())
    .collect();
    let mut wire = documents.clone();
    let objects: Vec<&[u8]> = documents
        .iter()
        .filter(|document| document.trim_ascii_start().starts_with(b"{"))
        .map(Vec::as_slice)
        .collect();
    wire.push([b"[".as_slice(), &objects.join(b",".as_slice()), b"]"].concat());
    let mut wire_tokens = BTreeSet::new();
    for token in tokens_of(
        &[
            "rust/crates/tokspur-wire/src/verify8.rs",
            "rust/crates/tokspur-wire/src/worker.rs",
            "rust/crates/tokspur-server/src/admission.rs",
        ],
        48,
    ) {
        wire_tokens.insert([b"\"".as_slice(), &token, b"\":"].concat());
        wire_tokens.insert(token);
    }
    for level in WIRE_LEVELS {
        write_corpus(&root, &format!("wire-{level}"), &wire);
        write_dictionary(&root, &format!("wire-{level}"), &wire_tokens);
    }
}

#[test]
fn literals_are_extracted_unescaped_and_comments_skipped() {
    let source = concat!(
        "// \"comment\"\n",
        "let a = \"x\\ny\\x41\\\"\";\n",
        "let b = b\"\\xff\\\\\";\n",
        "let c = r#\"raw \"inner\" text\"#;\n",
        "let d = br\"bytes\";\n",
        "let e = '\"'; let f = b'\\''; fn g<'a>(_: &'a str) {}\n",
        "let r#type = \"continued \\\n        line\";\n",
        "/* \"block\" /* nested */ */ let h = \"after\";\n",
    );
    let found = literals(source.as_bytes());
    let expected: [&[u8]; 6] = [
        b"x\nyA\"",
        b"\xff\\",
        b"raw \"inner\" text",
        b"bytes",
        b"continued line",
        b"after",
    ];
    assert_eq!(found, expected);
}

#[test]
fn dictionary_entries_escape_quotes_and_bytes() {
    assert_eq!(dictionary_entry(b"a\"b\\\x01"), "\"a\\\"b\\\\\\x01\"");
}
