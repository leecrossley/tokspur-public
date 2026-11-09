//! `files()` in `native/tests/test_gguf.cpp` (17 checks).

use std::io::Write;
use std::path::PathBuf;

use tokspur_gguf::{MappedFile, ModelFiles};

use crate::support::{Temporary, data, fixture, key, keys_fixture, refusal, split, text};

struct Shards {
    temporary: Temporary,
    p0: PathBuf,
    p1: PathBuf,
}

fn shards() -> Shards {
    let temporary = Temporary::new();
    let p0 = temporary.write(
        "part0.gguf",
        &fixture(&split(0, 2, 2), &[data("first", &[4], 0, 0, 16)], 32),
    );
    let p1 = temporary.write(
        "part1.gguf",
        &fixture(&split(1, 2, 2), &[data("second", &[4], 0, 0, 16)], 32),
    );
    Shards { temporary, p0, p1 }
}

fn plain(temporary: &Temporary) -> PathBuf {
    temporary.write(
        "plain.gguf",
        &keys_fixture(&[key("general.architecture", 8, text("toy"))]),
    )
}

#[test]
fn split_set_in_any_order_loads() {
    let set = shards();
    let valid = ModelFiles::new(&[&set.p1, &set.p0]).unwrap();
    assert_eq!(valid.architecture(), b"test", "split architecture");
    assert_eq!(valid.tensors().len(), 2, "split tensors");
    let mut bytes = Vec::new();
    valid.tensor_bytes("first", &mut bytes).unwrap();
    assert_eq!(bytes.len(), 16, "split tensor lookup");
}

#[test]
fn unknown_tensor_names_are_refused() {
    let set = shards();
    let valid = ModelFiles::new(&[&set.p1, &set.p0]).unwrap();
    assert_eq!(refusal(valid.tensor("missing")), "unknown tensor");
    let mut bytes = Vec::new();
    assert_eq!(
        refusal(valid.tensor_bytes("missing", &mut bytes)),
        "unknown tensor"
    );
}

#[test]
fn empty_duplicate_and_incomplete_sets_are_refused() {
    let set = shards();
    let empty: [PathBuf; 0] = [];
    assert_eq!(
        refusal(ModelFiles::new(&empty)),
        "expected 1 to 1024 GGUF shards"
    );
    assert_eq!(
        refusal(ModelFiles::new(&[&set.p0, &set.p0])),
        "incomplete or mismatched GGUF shard set"
    );
    assert_eq!(
        refusal(ModelFiles::new(&[&set.p0])),
        "incomplete or mismatched GGUF shard set"
    );
}

#[test]
fn mixed_split_and_plain_files_are_refused() {
    let set = shards();
    let plain = plain(&set.temporary);
    assert_eq!(
        refusal(ModelFiles::new(&[&set.p0, &plain])),
        "inconsistent split-file metadata"
    );
}

#[test]
fn mismatched_declared_totals_are_refused() {
    let set = shards();
    let bad = set.temporary.write(
        "bad.gguf",
        &fixture(&split(1, 2, 3), &[data("second", &[4], 0, 0, 16)], 32),
    );
    assert_eq!(
        refusal(ModelFiles::new(&[&set.p0, &bad])),
        "incomplete or mismatched GGUF shard set"
    );
}

#[test]
fn mismatched_architectures_are_refused() {
    let set = shards();
    let mut architecture_keys = split(1, 2, 2);
    architecture_keys[0].value = text("other");
    let other = set.temporary.write(
        "other.gguf",
        &fixture(&architecture_keys, &[data("second", &[4], 0, 0, 16)], 32),
    );
    assert_eq!(
        refusal(ModelFiles::new(&[&set.p0, &other])),
        "shard architecture mismatch"
    );
}

#[test]
fn duplicate_tensors_across_shards_are_refused() {
    let set = shards();
    let duplicate_name = set.temporary.write(
        "duplicate.gguf",
        &fixture(&split(1, 2, 2), &[data("first", &[4], 0, 0, 16)], 32),
    );
    assert_eq!(
        refusal(ModelFiles::new(&[&set.p0, &duplicate_name])),
        "duplicate tensor across GGUF shards"
    );
}

#[test]
fn mapped_file_refuses_empty_files_directories_and_missing_paths() {
    let temporary = Temporary::new();
    let empty = temporary.write("empty.gguf", &[]);
    assert_eq!(
        refusal(MappedFile::new(&empty)),
        "weight file must be a nonempty regular file"
    );
    assert_eq!(
        refusal(MappedFile::new(&temporary.root)),
        "weight file must be a nonempty regular file"
    );
    assert_eq!(
        refusal(MappedFile::new(temporary.root.join("missing"))),
        "cannot open weight file"
    );
}

#[test]
fn moved_mapping_keeps_its_bytes_and_detects_a_change() {
    let temporary = Temporary::new();
    let plain = plain(&temporary);
    let mapped = MappedFile::new(&plain).unwrap();
    let moved = mapped;
    assert!(moved.size() > 16, "move mapping");
    assert_eq!(moved.bytes().unwrap(), std::fs::read(&plain).unwrap());
    moved.assert_unchanged().unwrap();
    std::fs::OpenOptions::new()
        .append(true)
        .open(&plain)
        .unwrap()
        .write_all(b"x")
        .unwrap();
    assert_eq!(
        refusal(moved.assert_unchanged()),
        "weight file changed since mapping"
    );
}
