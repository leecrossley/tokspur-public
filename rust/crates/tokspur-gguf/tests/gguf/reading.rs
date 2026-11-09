//! File reading without `mmap`: the port reads only the header region, keeps the C++ open flags and checks, and calls `assert_unchanged` exactly where the C++ does. No C++ counterpart; the differential run compares the same cases with the C++ reader.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::Path;
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

use tokspur_gguf::{Gguf, MappedFile, ModelFiles};

use crate::support::{
    Bytes, Data, Temporary, data, default_fixture, fixture, fixture_header, integer, key,
    keys_fixture, number, refusal, split, string, text,
};

/// Writes `header` and extends the file to `size` without writing the rest, so a large payload costs no disk space.
fn sparse(path: &Path, header: &[u8], size: u64) {
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(header).unwrap();
    file.set_len(size).unwrap();
}

fn append(path: &Path) {
    OpenOptions::new()
        .append(true)
        .open(path)
        .unwrap()
        .write_all(b"x")
        .unwrap();
}

#[test]
fn file_reader_reads_only_the_header_region() {
    // The default fixture's tensor info ends at byte 62; its data starts at 64.
    let temporary = Temporary::new();
    let path = temporary.write("plain.gguf", &default_fixture());
    let file = Gguf::from_path(&path).unwrap();
    assert_eq!(file.loaded_bytes(), 62);
    assert_eq!(file.file_size(), 80);
    assert_eq!(file.data_offset(), 64);
}

#[test]
fn large_tensor_data_is_never_read() {
    let temporary = Temporary::new();
    let tensor = data("weight", &[1 << 28], 0, 0, 1 << 30);
    let (header, payload) = fixture_header(&[], &[tensor], 32);
    let path = temporary.root.join("large.gguf");
    sparse(&path, &header, header.len() as u64 + payload);
    let file = Gguf::from_path(&path).unwrap();
    assert_eq!(file.loaded_bytes(), 62);
    assert_eq!(file.file_size(), 64 + (1 << 30));
    assert_eq!(file.tensors()[0].bytes, 1 << 30);
}

#[test]
fn tensor_bytes_reads_beyond_four_gibibytes() {
    let temporary = Temporary::new();
    let far = 1u64 << 32;
    let tensors = [data("near", &[4], 0, 0, 16), data("far", &[4], 0, far, 16)];
    let (header, payload) = fixture_header(&[], &tensors, 32);
    let path = temporary.root.join("far.gguf");
    sparse(&path, &header, header.len() as u64 + payload);
    {
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        use std::os::unix::fs::FileExt;
        file.write_all_at(&[0xab; 16], header.len() as u64 + far)
            .unwrap();
        file.flush().unwrap();
    }
    let model = Gguf::from_path(&path).unwrap();
    let mut bytes = Vec::new();
    model.tensor_bytes(&model.tensors()[1], &mut bytes).unwrap();
    assert_eq!(model.tensors()[1].offset, header.len() as u64 + far);
    assert_eq!(bytes, [0xab; 16]);
    model.tensor_bytes(&model.tensors()[0], &mut bytes).unwrap();
    assert_eq!(bytes, [0; 16]);
}

#[test]
fn borrowed_and_file_readers_agree() {
    let mut array = Bytes::new();
    number(&mut array, 8, 4);
    number(&mut array, 3, 8);
    for token in ["<|im_start|>", "Ġthe", "🦉"] {
        string(&mut array, token.as_bytes());
    }
    let bytes = keys_fixture(&[
        key("general.architecture", 8, text("toy")),
        key("tokenizer.ggml.tokens", 9, array),
        key("eps", 6, integer(u64::from(1e-6f32.to_bits()), 4)),
    ]);
    let temporary = Temporary::new();
    let path = temporary.write("tokens.gguf", &bytes);
    let borrowed = Gguf::from_bytes(&bytes).unwrap();
    let file = Gguf::from_path(&path).unwrap();
    assert_eq!(borrowed.metadata(), file.metadata());
    assert_eq!(borrowed.tensors(), file.tensors());
    assert_eq!(
        borrowed.string_array("tokenizer.ggml.tokens").unwrap(),
        file.string_array("tokenizer.ggml.tokens").unwrap()
    );
    assert_eq!(
        file.string_array("tokenizer.ggml.tokens").unwrap()[2],
        "🦉".as_bytes()
    );
    assert_eq!(
        borrowed.floating_value("eps").unwrap(),
        file.floating_value("eps").unwrap()
    );
    assert!(file.loaded_bytes() < bytes.len());
}

#[test]
fn accessors_check_for_changes_where_the_cpp_does() {
    let mut array = Bytes::new();
    number(&mut array, 8, 4);
    number(&mut array, 1, 8);
    string(&mut array, b"token");
    let mut integers = Bytes::new();
    number(&mut integers, 4, 4);
    number(&mut integers, 1, 8);
    number(&mut integers, 7, 4);
    let bytes = keys_fixture(&[
        key("general.architecture", 8, text("toy")),
        key("count", 4, integer(3, 4)),
        key("eps", 6, integer(u64::from(1.5f32.to_bits()), 4)),
        key("ids", 9, integers),
        key("tokens", 9, array),
    ]);
    let temporary = Temporary::new();
    let path = temporary.write("changing.gguf", &bytes);
    let file = Gguf::from_path(&path).unwrap();
    append(&path);
    let changed = "weight file changed since mapping";
    assert_eq!(refusal(file.assert_unchanged()), changed);
    assert_eq!(refusal(file.floating_value("eps")), changed);
    assert_eq!(refusal(file.unsigned_array("ids")), changed);
    assert_eq!(refusal(file.string_array("tokens")), changed);
    let mut buffer = Vec::new();
    assert_eq!(
        refusal(file.tensor_bytes(&file.tensors()[0], &mut buffer)),
        changed
    );
    // unsigned_value and string_value read the parsed header without a check.
    assert_eq!(file.unsigned_value("count").unwrap(), 3);
    assert_eq!(file.string_value("general.architecture").unwrap(), b"toy");
    // A refused accessor still refuses its type before it checks the file.
    assert_eq!(
        refusal(file.string_array("count")),
        "GGUF metadata is not a string array"
    );
}

#[test]
fn model_files_detect_a_changed_shard() {
    let temporary = Temporary::new();
    let p0 = temporary.write(
        "part0.gguf",
        &fixture(&split(0, 2, 2), &[data("first", &[4], 0, 0, 16)], 32),
    );
    let p1 = temporary.write(
        "part1.gguf",
        &fixture(&split(1, 2, 2), &[data("second", &[4], 0, 0, 16)], 32),
    );
    let model = ModelFiles::new(&[&p0, &p1]).unwrap();
    model.assert_unchanged().unwrap();
    append(&p1);
    assert_eq!(
        refusal(model.assert_unchanged()),
        "weight file changed since mapping"
    );
}

#[test]
fn symlinks_are_followed() {
    let temporary = Temporary::new();
    let target = temporary.write("target.gguf", &default_fixture());
    let link = temporary.root.join("link.gguf");
    symlink(&target, &link).unwrap();
    let file = Gguf::from_path(&link).unwrap();
    assert_eq!(file.tensors()[0].name, b"weight");
    let dangling = temporary.root.join("dangling.gguf");
    symlink(temporary.root.join("absent"), &dangling).unwrap();
    assert_eq!(
        refusal(MappedFile::new(&dangling)),
        "cannot open weight file"
    );
    let directory = temporary.root.join("directory.gguf");
    symlink(&temporary.root, &directory).unwrap();
    assert_eq!(
        refusal(MappedFile::new(&directory)),
        "weight file must be a nonempty regular file"
    );
}

#[test]
fn fifo_is_refused_without_blocking() {
    let temporary = Temporary::new();
    let fifo = temporary.root.join("fifo.gguf");
    let status = Command::new("mkfifo").arg(&fifo).status().unwrap();
    assert!(status.success());
    // Without O_NONBLOCK, open() would wait for a writer forever.
    let (sender, receiver) = mpsc::channel();
    let path = fifo.clone();
    std::thread::spawn(move || {
        sender.send(MappedFile::new(&path).map(|_| ())).unwrap();
    });
    let result = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(
        refusal(result),
        "weight file must be a nonempty regular file"
    );
}

#[test]
fn devices_and_unreadable_files_are_refused() {
    assert_eq!(
        refusal(MappedFile::new("/dev/null")),
        "weight file must be a nonempty regular file"
    );
    assert_eq!(refusal(MappedFile::new("")), "cannot open weight file");
    let temporary = Temporary::new();
    let locked = temporary.write("locked.gguf", &default_fixture());
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    assert_eq!(refusal(MappedFile::new(&locked)), "cannot open weight file");
    let file_as_directory = temporary.root.join("locked.gguf/inner");
    assert_eq!(
        refusal(MappedFile::new(file_as_directory)),
        "cannot open weight file"
    );
}

#[test]
fn shard_errors_surface_in_path_order() {
    let temporary = Temporary::new();
    let valid = temporary.write("valid.gguf", &default_fixture());
    let broken = temporary.write("broken.gguf", &default_fixture()[..10]);
    let missing = temporary.root.join("missing.gguf");
    assert_eq!(
        refusal(ModelFiles::new(&[&valid, &broken, &missing])),
        "truncated or oversized GGUF header"
    );
    assert_eq!(
        refusal(ModelFiles::new(&[&valid, &missing, &broken])),
        "cannot open weight file"
    );
    let too_many = vec![missing.clone(); 1025];
    assert_eq!(
        refusal(ModelFiles::new(&too_many)),
        "expected 1 to 1024 GGUF shards"
    );
    let at_limit = vec![missing; 1024];
    assert_eq!(
        refusal(ModelFiles::new(&at_limit)),
        "cannot open weight file"
    );
}

#[test]
fn single_file_model_needs_an_architecture() {
    let temporary = Temporary::new();
    let anonymous = temporary.write("anonymous.gguf", &default_fixture());
    assert_eq!(
        refusal(ModelFiles::new(&[&anonymous])),
        "GGUF model has no architecture"
    );
    let named = temporary.write(
        "named.gguf",
        &fixture(
            &[key("general.architecture", 8, text("toy"))],
            &[Data::default()],
            32,
        ),
    );
    let model = ModelFiles::new(&[&named]).unwrap();
    assert_eq!(model.architecture(), b"toy");
    // 24-byte header, 43-byte key, 38-byte tensor info, padded to 128.
    assert_eq!(model.tensor("weight").unwrap().offset, 128);
}
