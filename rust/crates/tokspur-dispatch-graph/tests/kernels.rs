//! The Rust and private C++ flag-reader lists must match the kernels that load the error flag in the current MSL sources.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::path::Path;

use tokspur_dispatch_graph::graph::FLAG_READERS;

/// `(kernel name, body)` for every `kernel void name(...) { ... }`.
fn kernels(source: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut rest = source;
    while let Some(at) = rest.find("kernel void ") {
        let after = &rest[at + "kernel void ".len()..];
        let name: String = after
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let open = after.find('{').unwrap();
        let mut depth = 0;
        let mut end = open;
        for (offset, c) in after[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + offset;
                        break;
                    }
                }
                _ => {}
            }
        }
        found.push((name, after[open..=end].to_owned()));
        rest = &after[end..];
    }
    found
}

#[test]
fn flag_readers_are_exactly_the_kernels_that_load_the_error_flag() {
    // Arrange
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../native/kernels");
    let mut readers = BTreeSet::new();
    let mut total = 0;

    // Act
    for entry in std::fs::read_dir(&directory).unwrap() {
        let path = entry.unwrap().path();
        if path
            .extension()
            .is_some_and(|extension| extension == "metal")
        {
            for (name, body) in kernels(&std::fs::read_to_string(&path).unwrap()) {
                total += 1;
                if body.contains("atomic_load_explicit(error") {
                    readers.insert(name);
                }
            }
        }
    }

    // Assert
    assert!(total > 20, "scanned only {total} kernels");
    let expected: BTreeSet<String> = FLAG_READERS.iter().map(|name| (*name).to_owned()).collect();
    assert_eq!(
        expected.len(),
        FLAG_READERS.len(),
        "duplicate Rust flag reader"
    );
    assert_eq!(readers, expected);
    let host = std::fs::read_to_string(directory.join("../src/graph.mm")).unwrap();
    let table = host
        .split_once("flag_reading_kernels{")
        .expect("private C++ flag-reader table")
        .1
        .split_once("};")
        .expect("end of private C++ flag-reader table")
        .0;
    let (production_table, diagnostic_table) = table
        .split_once("#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC")
        .expect("separate bounded diagnostic flag-reader table");
    let diagnostic_table = diagnostic_table.split_once("#endif").unwrap().0;
    let names: Vec<String> = production_table
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_owned)
        .collect();
    let cpp: BTreeSet<String> = names.iter().cloned().collect();
    assert_eq!(cpp.len(), names.len(), "duplicate C++ flag reader");
    assert_eq!(cpp, expected, "C++ and Rust dependency rules must agree");

    // Diagnostic shaders are linked only into carrier-diagnostic. Keep their
    // additional dependency rules out of both production scheduler tables.
    let mut diagnostic_readers = BTreeSet::new();
    for entry in std::fs::read_dir(directory.join("../diagnostics")).unwrap() {
        let path = entry.unwrap().path();
        if path
            .extension()
            .is_some_and(|extension| extension == "metal")
        {
            for (name, body) in kernels(&std::fs::read_to_string(path).unwrap()) {
                if body.contains("atomic_load_explicit(error") {
                    diagnostic_readers.insert(name);
                }
            }
        }
    }
    let diagnostic_names: Vec<String> = diagnostic_table
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_owned)
        .collect();
    let diagnostic_cpp: BTreeSet<String> = diagnostic_names.iter().cloned().collect();
    assert!(!diagnostic_readers.is_empty());
    assert_eq!(diagnostic_cpp.len(), diagnostic_names.len());
    assert!(diagnostic_cpp.is_disjoint(&expected));
    assert_eq!(diagnostic_cpp, diagnostic_readers);
}
