//! GGUF targets: properties of `Gguf::from_bytes` on its own ([`check`]) and the comparison with the unchanged `native/src/gguf.cpp` ([`check_differential`]) through the canonical summary that `rust/tools/diff/gguf_driver.cpp` prints.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use tokspur_gguf::{Gguf, GgufError, encoding};

use crate::encode::escape;
#[cfg(feature = "cpp-oracle")]
use crate::encode::require_identical;
#[cfg(feature = "cpp-oracle")]
use crate::oracle;

const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// Tensors larger than this are summarised without reading their bytes.
const TENSOR_BYTES_LIMIT: u64 = 16 * 1024 * 1024;

fn fnv(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn fnv_number(hash: u64, value: u64) -> u64 {
    fnv(hash, &value.to_le_bytes())
}

fn strings_hash(values: &[&[u8]]) -> u64 {
    values.iter().fold(FNV_BASIS, |hash, value| {
        fnv(fnv_number(hash, value.len() as u64), value)
    })
}

fn line_or_error<T>(
    out: &mut String,
    label: &str,
    result: Result<T, GgufError>,
    ok: impl FnOnce(T) -> String,
) {
    let line = match result {
        Ok(value) => format!("{label} {}\n", ok(value)),
        Err(error) => format!("{label} error {error}\n"),
    };
    out.push_str(&line);
}

/// `gguf_summary` (rust/tools/diff/gguf_driver.cpp) from the Rust reader: the refusal, or every accessor on every key and every tensor. Keys and tensor names are validated ASCII, so the lossy conversions are exact.
pub fn summary(input: &[u8]) -> String {
    let file = match Gguf::from_bytes(input) {
        Ok(file) => file,
        Err(error) => return format!("error {error}\n"),
    };
    let mut out = format!(
        "ok tensors={} metadata={} data_offset={} file_size={}\n",
        file.tensors().len(),
        file.metadata().len(),
        file.data_offset(),
        file.file_size()
    );
    for (key, item) in file.metadata() {
        out.push_str(&format!(
            "key {} type={} offset={} end={} element_type={} elements={}\n",
            String::from_utf8_lossy(key),
            item.r#type,
            item.offset,
            item.end,
            item.element_type,
            item.elements
        ));
        line_or_error(&mut out, "unsigned", file.unsigned_value(key), |value| {
            value.to_string()
        });
        line_or_error(&mut out, "floating", file.floating_value(key), |value| {
            format!("{:016x}", value.to_bits())
        });
        line_or_error(
            &mut out,
            "unsigned_array",
            file.unsigned_array(key),
            |values| {
                let hash = values
                    .iter()
                    .fold(FNV_BASIS, |hash, &value| fnv_number(hash, value));
                format!("{} {hash:016x}", values.len())
            },
        );
        line_or_error(&mut out, "string", file.string_value(key), |value| {
            format!("{} {:016x}", value.len(), fnv(FNV_BASIS, value))
        });
        line_or_error(&mut out, "string_array", file.string_array(key), |values| {
            format!("{} {:016x}", values.len(), strings_hash(&values))
        });
    }
    let mut buffer = Vec::new();
    for tensor in file.tensors() {
        let dimensions: Vec<String> = tensor.dimensions.iter().map(u64::to_string).collect();
        out.push_str(&format!(
            "tensor {} type={} offset={} bytes={} elements={} dimensions={}\n",
            String::from_utf8_lossy(&tensor.name),
            tensor.r#type,
            tensor.offset,
            tensor.bytes,
            tensor.elements,
            dimensions.join(",")
        ));
        if tensor.bytes > TENSOR_BYTES_LIMIT {
            out.push_str("tensor_bytes skipped\n");
        } else {
            let result = file.tensor_bytes(tensor, &mut buffer);
            let hash = fnv(FNV_BASIS, &buffer);
            line_or_error(&mut out, "tensor_bytes", result, |()| {
                format!("{} {hash:016x}", buffer.len())
            });
        }
        let copy = tensor.clone();
        match file.tensor_bytes(&copy, &mut buffer) {
            Ok(()) => out.push_str("copy accepted\n"),
            Err(error) => out.push_str(&format!("copy error {error}\n")),
        }
    }
    match file.assert_unchanged() {
        Ok(()) => out.push_str("unchanged ok\n"),
        Err(error) => out.push_str(&format!("unchanged error {error}\n")),
    }
    out
}

/// `Gguf(std::span)` in C++ and `Gguf::from_bytes` must summarise the same bytes identically: refusal message, metadata spans and every accessor's value or message, tensor layout and tensor bytes.
#[cfg(feature = "cpp-oracle")]
pub fn check_differential(data: &[u8]) {
    let cpp = oracle::gguf(data);
    let rust = summary(data);
    require_identical("Gguf summary", data, &cpp, rust.as_bytes());
}

/// Every message `Gguf::from_bytes` and the accessors can raise (gguf.cpp).
const MESSAGES: [&str; 26] = [
    "truncated or oversized GGUF header",
    "GGUF string exceeds limit",
    "invalid UTF-8 in GGUF string",
    "GGUF array nesting exceeds limit",
    "invalid GGUF boolean",
    "invalid GGUF array",
    "unknown GGUF metadata type",
    "GGUF offset overflow",
    "GGUF shape overflow",
    "expected little-endian GGUF v3",
    "GGUF record limit exceeded",
    "GGUF counts exceed available file bytes",
    "invalid GGUF metadata key",
    "duplicate GGUF metadata key",
    "unsupported GGUF alignment",
    "empty or duplicate tensor name",
    "non-ASCII tensor name",
    "unsupported GGUF tensor rank",
    "zero tensor dimension",
    "quantized tensor row is not block-aligned",
    "unaligned GGUF tensor offset",
    "truncated GGUF header padding",
    "truncated GGUF tensor data",
    "overlapping GGUF tensors",
    "GGUF metadata is not an integer",
    "negative GGUF count",
];

fn known(error: &GgufError, input: &[u8]) {
    let message = error.to_string();
    let known = MESSAGES.contains(&message.as_str())
        || message
            .strip_prefix("unsupported GGUF tensor encoding ")
            .is_some_and(|id| id.parse::<u32>().is_ok());
    assert!(
        known,
        "unknown GGUF error {message:?} for {}",
        escape(input, 256)
    );
}

fn valid_key(key: &[u8]) -> bool {
    !key.is_empty()
        && key[0] != b'.'
        && key[key.len() - 1] != b'.'
        && !key.windows(2).any(|pair| pair == b"..")
        && key
            .iter()
            .all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_' || c == b'.')
}

/// `Gguf::from_bytes` on arbitrary bytes. A refusal carries a known message. Accepted files have valid header keys and value spans. Files with tensors have an aligned data offset and tensor offsets, encoded sizes and nonoverlapping bytes within the file. `tensor_bytes` returns those exact bytes and refuses a copy of a tensor. Appending bytes preserves metadata and tensor records, while a metadata-only file can gain padding before its nominal data offset.
pub fn check(data: &[u8]) {
    let file = match Gguf::from_bytes(data) {
        Ok(file) => file,
        Err(error) => {
            known(&error, data);
            return;
        }
    };
    let input = || escape(data, 256);
    let size = data.len() as u64;
    assert_eq!(file.file_size(), size);
    let alignment = match file.find("general.alignment") {
        Some(_) => match file.unsigned_value("general.alignment") {
            Ok(value) => value,
            Err(error) => panic!("accepted alignment unreadable ({error}): {}", input()),
        },
        None => 32,
    };
    assert!(
        alignment.is_power_of_two() && alignment <= 65536,
        "alignment {alignment}: {}",
        input()
    );
    let data_offset = file.data_offset();
    assert!(
        file.tensors().is_empty() || (data_offset <= size && data_offset.is_multiple_of(alignment)),
        "data offset {data_offset}: {}",
        input()
    );
    for (key, item) in file.metadata() {
        assert!(valid_key(key), "invalid key accepted: {}", input());
        assert!(
            item.offset < item.end && item.end <= data_offset && item.r#type <= 12,
            "metadata span {item:?}: {}",
            input()
        );
        // Accessors refuse or answer; neither may panic.
        let string = file.string_value(key);
        assert_eq!(
            string.is_ok(),
            item.r#type == 8,
            "string_value: {}",
            input()
        );
        if let Ok(text) = string {
            assert!(
                std::str::from_utf8(text).is_ok(),
                "non-UTF-8 string: {}",
                input()
            );
        }
        if file.floating_value(key).is_ok() {
            assert!(matches!(item.r#type, 6 | 12), "floating_value: {}", input());
        }
        if file.unsigned_value(key).is_ok() {
            assert!(
                matches!(item.r#type, 0..=5 | 10 | 11),
                "unsigned_value: {}",
                input()
            );
        }
        if let Ok(values) = file.unsigned_array(key) {
            assert_eq!(
                values.len() as u64,
                item.elements,
                "unsigned_array: {}",
                input()
            );
        }
        if let Ok(values) = file.string_array(key) {
            assert_eq!(
                values.len() as u64,
                item.elements,
                "string_array: {}",
                input()
            );
        }
    }
    let mut names = BTreeSet::new();
    let mut intervals = Vec::new();
    let mut buffer = Vec::new();
    for tensor in file.tensors() {
        assert!(
            !tensor.name.is_empty()
                && tensor.name.len() <= 64
                && tensor.name.iter().all(|c| (0x21..=0x7e).contains(c))
                && names.insert(tensor.name.clone()),
            "tensor name: {}",
            input()
        );
        let layout = match encoding(tensor.r#type) {
            Ok(layout) => layout,
            Err(error) => panic!("accepted tensor encoding refused ({error}): {}", input()),
        };
        let per_block = u64::from(layout.elements_per_block);
        let elements = tensor
            .dimensions
            .iter()
            .try_fold(1u64, |product, &dimension| product.checked_mul(dimension));
        assert!(
            (1..=4).contains(&tensor.dimensions.len())
                && tensor.dimensions.iter().all(|&dimension| dimension != 0)
                && elements == Some(tensor.elements)
                && tensor.dimensions[0].is_multiple_of(per_block)
                && Some(tensor.bytes)
                    == (tensor.elements / per_block).checked_mul(u64::from(layout.bytes_per_block)),
            "tensor shape {tensor:?}: {}",
            input()
        );
        let end = tensor.offset + tensor.bytes;
        assert!(
            tensor.offset >= data_offset
                && (tensor.offset - data_offset).is_multiple_of(alignment)
                && end <= size,
            "tensor span {tensor:?}: {}",
            input()
        );
        intervals.push((tensor.offset, end));
        match file.tensor_bytes(tensor, &mut buffer) {
            Ok(()) => assert!(
                buffer.as_slice() == &data[tensor.offset as usize..end as usize],
                "tensor_bytes differ from the input span: {}",
                input()
            ),
            Err(error) => panic!(
                "tensor_bytes refused an accepted tensor ({error}): {}",
                input()
            ),
        }
        let copy = tensor.clone();
        assert_eq!(
            file.tensor_bytes(&copy, &mut buffer)
                .map_err(|error| error.to_string()),
            Err("tensor does not belong to this GGUF".to_owned()),
            "{}",
            input()
        );
    }
    intervals.sort_unstable();
    assert!(
        intervals.windows(2).all(|pair| pair[0].1 <= pair[1].0),
        "overlapping tensors accepted: {}",
        input()
    );

    let mut longer = data.to_vec();
    longer.extend_from_slice(&[0xa5; 37]);
    match Gguf::from_bytes(&longer) {
        Ok(extended) => assert!(
            extended.metadata() == file.metadata()
                && extended.tensors() == file.tensors()
                && (file.tensors().is_empty() || extended.data_offset() == data_offset),
            "appending bytes changed the records: {}",
            input()
        ),
        Err(error) => panic!(
            "appending bytes made the file invalid ({error}): {}",
            input()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small valid file: two metadata records (a string and an integer array) and one F32 tensor, laid out as the GGUF tests do.
    fn sample() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        let key = |bytes: &mut Vec<u8>, name: &[u8]| {
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name);
        };
        key(&mut bytes, b"general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        key(&mut bytes, b"qwen35moe");
        key(&mut bytes, b"tokenizer.ids");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&9u32.to_le_bytes());
        key(&mut bytes, b"w");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        bytes
    }

    #[test]
    fn a_valid_file_satisfies_every_property_and_matches_the_cpp() {
        let file = sample();
        assert!(summary(&file).starts_with("ok tensors=1 metadata=2"));
        check(&file);
        #[cfg(feature = "cpp-oracle")]
        check_differential(&file);
    }

    #[cfg(feature = "cpp-oracle")]
    #[test]
    fn every_truncation_is_refused_alike_by_both_readers() {
        let file = sample();
        for end in 0..file.len() {
            check(&file[..end]);
            check_differential(&file[..end]);
        }
    }
}

#[cfg(test)]
mod metadata_only_tests {
    #[test]
    fn metadata_only_gguf_needs_no_tensor_padding() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        super::check(&bytes);
    }
}
