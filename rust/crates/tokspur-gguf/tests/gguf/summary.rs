//! The canonical case summary `rust/tools/diff/gguf_driver.cpp` prints, here printed from the Rust reader. Both sides must stay line-for-line identical.

use tokspur_gguf::{Gguf, GgufError, MappedFile, ModelFiles};

use crate::support::{FNV_BASIS, fnv, fnv_number, hex, string_fingerprint, strings_hash};

/// Tensors larger than this are summarised without reading their bytes.
pub const TENSOR_BYTES_LIMIT: u64 = 16 * 1024 * 1024;

fn push(out: &mut String, line: impl AsRef<str>) {
    out.push_str(line.as_ref());
    out.push('\n');
}

fn line_or_error<T>(
    label: &str,
    result: Result<T, GgufError>,
    ok: impl FnOnce(T) -> String,
) -> String {
    match result {
        Ok(value) => format!("{label} {}", ok(value)),
        Err(error) => format!("{label} error {error}"),
    }
}

/// One `Gguf` case: construction, every accessor on every key, every tensor.
pub fn gguf(out: &mut String, result: Result<Gguf<'_>, GgufError>) {
    let file = match result {
        Ok(file) => file,
        Err(error) => return push(out, format!("error {error}")),
    };
    push(
        out,
        format!(
            "ok tensors={} metadata={} data_offset={} file_size={}",
            file.tensors().len(),
            file.metadata().len(),
            file.data_offset(),
            file.file_size()
        ),
    );
    for (key, item) in file.metadata() {
        push(
            out,
            format!(
                "key {} type={} offset={} end={} element_type={} elements={}",
                String::from_utf8_lossy(key),
                item.r#type,
                item.offset,
                item.end,
                item.element_type,
                item.elements
            ),
        );
        push(
            out,
            line_or_error("unsigned", file.unsigned_value(key), |value| {
                value.to_string()
            }),
        );
        push(
            out,
            line_or_error("floating", file.floating_value(key), |value| {
                format!("{:016x}", value.to_bits())
            }),
        );
        push(
            out,
            line_or_error("unsigned_array", file.unsigned_array(key), |values| {
                let hash = values
                    .iter()
                    .fold(FNV_BASIS, |hash, &value| fnv_number(hash, value));
                format!("{} {hash:016x}", values.len())
            }),
        );
        push(
            out,
            line_or_error("string", file.string_value(key), |value| {
                format!("{} {:016x}", value.len(), fnv(FNV_BASIS, value))
            }),
        );
        push(
            out,
            line_or_error("string_array", file.string_array(key), |values| {
                format!("{} {:016x}", values.len(), strings_hash(&values))
            }),
        );
    }
    let mut buffer = Vec::new();
    for tensor in file.tensors() {
        let dimensions: Vec<String> = tensor.dimensions.iter().map(u64::to_string).collect();
        push(
            out,
            format!(
                "tensor {} type={} offset={} bytes={} elements={} dimensions={}",
                String::from_utf8_lossy(&tensor.name),
                tensor.r#type,
                tensor.offset,
                tensor.bytes,
                tensor.elements,
                dimensions.join(",")
            ),
        );
        if tensor.bytes > TENSOR_BYTES_LIMIT {
            push(out, "tensor_bytes skipped");
        } else {
            let result = file.tensor_bytes(tensor, &mut buffer);
            push(
                out,
                line_or_error("tensor_bytes", result, |()| {
                    format!("{} {:016x}", buffer.len(), fnv(FNV_BASIS, &buffer))
                }),
            );
        }
        let copy = tensor.clone();
        push(
            out,
            match file.tensor_bytes(&copy, &mut buffer) {
                Ok(()) => "copy accepted".to_owned(),
                Err(error) => format!("copy error {error}"),
            },
        );
    }
    push(out, unchanged(file.assert_unchanged()));
}

fn unchanged(result: Result<(), GgufError>) -> String {
    match result {
        Ok(()) => "unchanged ok".to_owned(),
        Err(error) => format!("unchanged error {error}"),
    }
}

/// One `ModelFiles` case, including what `NativeText::verify_gguf` reads.
pub fn model(out: &mut String, result: Result<ModelFiles, GgufError>) {
    let model = match result {
        Ok(model) => model,
        Err(error) => return push(out, format!("error {error}")),
    };
    push(
        out,
        format!(
            "ok shards={} tensors={} architecture={}",
            model.shards().len(),
            model.tensors().len(),
            hex(model.architecture())
        ),
    );
    let mut buffer = Vec::new();
    for (name, location) in model.tensors() {
        let name_text = String::from_utf8_lossy(name);
        push(
            out,
            format!(
                "location {name_text} shard={} index={}",
                location.shard, location.index
            ),
        );
        match model.tensor(name) {
            Err(error) => push(out, format!("tensor error {error}")),
            Ok(tensor) => {
                push(
                    out,
                    format!("tensor offset={} bytes={}", tensor.offset, tensor.bytes),
                );
                if tensor.bytes > TENSOR_BYTES_LIMIT {
                    push(out, "tensor_bytes skipped");
                } else {
                    match model.tensor_bytes(name, &mut buffer) {
                        Ok(()) => push(
                            out,
                            format!(
                                "tensor_bytes {} {:016x}",
                                buffer.len(),
                                fnv(FNV_BASIS, &buffer)
                            ),
                        ),
                        Err(error) => push(out, format!("tensor error {error}")),
                    }
                }
            }
        }
    }
    for (s, shard) in model.shards().iter().enumerate() {
        if shard.find("tokenizer.ggml.tokens").is_none() {
            continue;
        }
        let arrays = shard
            .string_array("tokenizer.ggml.tokens")
            .and_then(|tokens| Ok((tokens, shard.string_array("tokenizer.ggml.merges")?)));
        push(
            out,
            match arrays {
                Ok((tokens, merges)) => format!(
                    "tokenizer {s} tokens {} {:016x} merges {} {:016x}",
                    tokens.len(),
                    strings_hash(&tokens),
                    merges.len(),
                    strings_hash(&merges)
                ),
                Err(error) => format!("tokenizer {s} error {error}"),
            },
        );
    }
    push(
        out,
        match model.tensor("missing") {
            Ok(_) => "unknown accepted".to_owned(),
            Err(error) => format!("unknown error {error}"),
        },
    );
    push(
        out,
        match model.tensor_bytes("missing", &mut buffer) {
            Ok(()) => "unknown_bytes accepted".to_owned(),
            Err(error) => format!("unknown_bytes error {error}"),
        },
    );
    push(out, unchanged(model.assert_unchanged()));
}

/// One `MappedFile` case.
pub fn mapped(out: &mut String, result: Result<MappedFile, GgufError>) {
    let file = match result {
        Ok(file) => file,
        Err(error) => return push(out, format!("error {error}")),
    };
    match file.bytes() {
        Ok(bytes) => push(
            out,
            format!(
                "ok size={} bytes={:016x}",
                file.size(),
                fnv(FNV_BASIS, &bytes)
            ),
        ),
        Err(error) => push(out, format!("bytes error {error}")),
    }
    push(out, unchanged(file.assert_unchanged()));
}

/// The real shard set: headers and the tokenizer fingerprints the server compares with its tokenizer JSON. No tensor data is read.
pub fn fingerprints(out: &mut String, result: Result<ModelFiles, GgufError>) {
    let model = match result {
        Ok(model) => model,
        Err(error) => return push(out, format!("error {error}")),
    };
    push(
        out,
        format!(
            "ok shards={} tensors={} architecture={}",
            model.shards().len(),
            model.tensors().len(),
            hex(model.architecture())
        ),
    );
    for (s, shard) in model.shards().iter().enumerate() {
        push(
            out,
            format!(
                "shard {s} file_size={} data_offset={} tensors={} metadata={}",
                shard.file_size(),
                shard.data_offset(),
                shard.tensors().len(),
                shard.metadata().len()
            ),
        );
    }
    for (s, shard) in model.shards().iter().enumerate() {
        if shard.find("tokenizer.ggml.tokens").is_none() {
            continue;
        }
        let arrays = shard
            .string_array("tokenizer.ggml.tokens")
            .and_then(|tokens| Ok((tokens, shard.string_array("tokenizer.ggml.merges")?)));
        push(
            out,
            match arrays {
                Ok((tokens, merges)) => format!(
                    "tokenizer {s} tokens {} {} merges {} {}",
                    tokens.len(),
                    string_fingerprint(&tokens),
                    merges.len(),
                    string_fingerprint(&merges)
                ),
                Err(error) => format!("tokenizer {s} error {error}"),
            },
        );
    }
    push(out, unchanged(model.assert_unchanged()));
}
