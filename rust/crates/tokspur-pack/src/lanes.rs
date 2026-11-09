//! SIMD-group lane packing and eligibility rules ported one to one from `native/src/expert_lanes.cpp` and `native/src/lane_sidecar.cpp`.

use tokspur_gguf::checked_mul;

use crate::PackError;

const Q4K_BLOCK: u32 = 144;
const Q4K_HEADER: u32 = 16;
const Q4K_ELEMENTS: u32 = 256;
const Q4K_RUN: u32 = 4;

const Q51_BLOCK: u32 = 24;
const Q51_HEADER: u32 = 8;
const Q51_ELEMENTS: u32 = 32;
const Q51_RUN: u32 = 16;

const Q8_BLOCK: u32 = 34;
const Q8_HEADER: u32 = 2;
const Q8_ELEMENTS: u32 = 32;
const Q8_RUN: u32 = 16;

/// Canonical row geometry for `Q4_K`, `Q5_1` or `Q8_0` SIMD-lane packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLayout {
    pub r#type: u32,
    pub columns: u32,
    pub blocks: u32,
    pub meta_bytes: u32,
    pub row_bytes: u64,
}

/// Computes canonical row geometry for `type` (`7`, `8` or `12`) and `columns`.
pub fn row_layout(r#type: u32, columns: u32) -> Result<RowLayout, PackError> {
    match r#type {
        12 => {
            if columns == 0 || !columns.is_multiple_of(Q4K_ELEMENTS) || columns > 65_536 {
                return Err(PackError::Format(
                    "Q4_K lane layout needs whole 256-element blocks",
                ));
            }
            let blocks = columns / Q4K_ELEMENTS;
            Ok(RowLayout {
                r#type,
                columns,
                blocks,
                meta_bytes: blocks * Q4K_HEADER,
                row_bytes: u64::from(blocks) * u64::from(Q4K_BLOCK),
            })
        }
        7 => {
            if columns == 0 || !columns.is_multiple_of(Q51_ELEMENTS) || columns > 65_536 {
                return Err(PackError::Format(
                    "Q5_1 lane layout needs whole 32-element blocks",
                ));
            }
            let blocks = columns / Q51_ELEMENTS;
            Ok(RowLayout {
                r#type,
                columns,
                blocks,
                meta_bytes: blocks * Q51_HEADER,
                row_bytes: u64::from(blocks) * u64::from(Q51_BLOCK),
            })
        }
        8 => {
            if columns == 0 || !columns.is_multiple_of(Q8_ELEMENTS) || columns > 65_536 {
                return Err(PackError::Format(
                    "Q8_0 lane layout needs whole 32-element blocks",
                ));
            }
            let blocks = columns / Q8_ELEMENTS;
            Ok(RowLayout {
                r#type,
                columns,
                blocks,
                meta_bytes: blocks * Q8_HEADER,
                row_bytes: u64::from(blocks) * u64::from(Q8_BLOCK),
            })
        }
        _ => Err(PackError::Format(
            "lane layout admits only Q4_K, Q5_1 and Q8_0",
        )),
    }
}

/// Byte offset inside a packed row where quant byte `index` of `block` is stored.
pub fn packed_quant_offset(layout: &RowLayout, block: u32, index: u32) -> Result<u64, PackError> {
    if block >= layout.blocks {
        return Err(PackError::Format("lane layout block bound"));
    }
    if layout.r#type == 12 {
        if index >= 128 {
            return Err(PackError::Format("Q4_K quant byte bound"));
        }
        let run = block / Q4K_RUN;
        let width = Q4K_RUN.min(layout.blocks - run * Q4K_RUN) * 4;
        let pair = index / 32;
        let lane = index % 32;
        return Ok(u64::from(layout.meta_bytes)
            + u64::from(run) * 512
            + u64::from(lane * width + (block % Q4K_RUN) * 4 + pair));
    }
    if layout.r#type == 8 {
        if index >= 32 {
            return Err(PackError::Format("Q8_0 quant byte bound"));
        }
        let run = block / Q8_RUN;
        let width = Q8_RUN.min(layout.blocks - run * Q8_RUN);
        return Ok(u64::from(layout.meta_bytes)
            + u64::from(run) * 512
            + u64::from(index * width + (block % Q8_RUN)));
    }
    if layout.r#type == 7 {
        if index >= 16 {
            return Err(PackError::Format("Q5_1 quant byte bound"));
        }
        let run = block / Q51_RUN;
        let width = Q51_RUN.min(layout.blocks - run * Q51_RUN);
        return Ok(u64::from(layout.meta_bytes)
            + u64::from(run) * 256
            + u64::from(index * width + (block % Q51_RUN)));
    }
    Err(PackError::Format(
        "lane layout admits only Q4_K, Q5_1 and Q8_0",
    ))
}

fn check_rows(layout: &RowLayout, a: &[u8], b: &[u8]) -> Result<(usize, usize, usize), PackError> {
    let expected = row_layout(layout.r#type, layout.columns)?;
    if *layout != expected {
        return Err(PackError::Format(
            "lane layout derived fields differ from canonical geometry",
        ));
    }
    let row_bytes = usize::try_from(layout.row_bytes)
        .map_err(|_| PackError::Format("lane layout row exceeds address space"))?;
    if a.is_empty() || a.len() != b.len() || !a.len().is_multiple_of(row_bytes) {
        return Err(PackError::Format(
            "lane layout spans must hold equal whole rows",
        ));
    }
    let (block_bytes, header_bytes) = match layout.r#type {
        12 => (Q4K_BLOCK as usize, Q4K_HEADER as usize),
        8 => (Q8_BLOCK as usize, Q8_HEADER as usize),
        7 => (Q51_BLOCK as usize, Q51_HEADER as usize),
        _ => {
            return Err(PackError::Format(
                "lane layout admits only Q4_K, Q5_1 and Q8_0",
            ));
        }
    };
    Ok((row_bytes, block_bytes, header_bytes))
}

/// Packs whole rows from `original` GGUF block order into `packed` SIMD-lane order.
pub fn pack_rows(layout: &RowLayout, original: &[u8], packed: &mut [u8]) -> Result<(), PackError> {
    let (row_bytes, block_bytes, header_bytes) = check_rows(layout, original, packed)?;
    let quant_bytes = block_bytes - header_bytes;
    for (in_row, out_row) in original
        .chunks_exact(row_bytes)
        .zip(packed.chunks_exact_mut(row_bytes))
    {
        for b in 0..layout.blocks {
            let block_start = (b as usize) * block_bytes;
            let meta_start = (b as usize) * header_bytes;
            out_row[meta_start..meta_start + header_bytes]
                .copy_from_slice(&in_row[block_start..block_start + header_bytes]);
            for i in 0..quant_bytes {
                let dst = packed_quant_offset(layout, b, i as u32)? as usize;
                out_row[dst] = in_row[block_start + header_bytes + i];
            }
        }
    }
    Ok(())
}

/// Unpacks whole rows from `packed` SIMD-lane order back into `original` GGUF block order.
pub fn unpack_rows(
    layout: &RowLayout,
    packed: &[u8],
    original: &mut [u8],
) -> Result<(), PackError> {
    let (row_bytes, block_bytes, header_bytes) = check_rows(layout, packed, original)?;
    let quant_bytes = block_bytes - header_bytes;
    for (in_row, out_row) in packed
        .chunks_exact(row_bytes)
        .zip(original.chunks_exact_mut(row_bytes))
    {
        for b in 0..layout.blocks {
            let block_start = (b as usize) * block_bytes;
            let meta_start = (b as usize) * header_bytes;
            out_row[block_start..block_start + header_bytes]
                .copy_from_slice(&in_row[meta_start..meta_start + header_bytes]);
            for i in 0..quant_bytes {
                let src = packed_quant_offset(layout, b, i as u32)? as usize;
                out_row[block_start + header_bytes + i] = in_row[src];
            }
        }
    }
    Ok(())
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Port of `lane_sidecar_eligible` (`native/src/lane_sidecar.cpp:184-201`).
pub fn lane_sidecar_eligible(name: &[u8], r#type: u32, dimensions: &[u64], bytes: u64) -> bool {
    if name.is_empty()
        || name.len() > 1024
        || contains_subslice(name, b"token_embd")
        || contains_subslice(name, b"ple_")
        || contains_subslice(name, b".ple.")
        || name.starts_with(b"ple.")
        || dimensions.len() < 2
        || dimensions[0] == 0
        || dimensions[0] > 65_536
        || dimensions[1] == 0
        || dimensions[1] > 1_000_000
    {
        return false;
    }
    let columns = dimensions[0];
    let q8 = r#type == 8
        && dimensions.len() == 2
        && columns.is_multiple_of(32)
        && (columns.is_multiple_of(256) || columns < 512);
    const PREFIX: &[u8] = b"blk.";
    const SUFFIX: &[u8] = b".ffn_down_exps.weight";
    let named = name.starts_with(PREFIX)
        && name.ends_with(SUFFIX)
        && name.len() > PREFIX.len() + SUFFIX.len()
        && name[PREFIX.len()..name.len() - SUFFIX.len()]
            .iter()
            .all(u8::is_ascii_digit);
    let q51 = r#type == 7
        && named
        && dimensions.len() == 3
        && columns == 640
        && dimensions[1] <= 65_536
        && dimensions[2] != 0
        && dimensions[2] <= 4096;
    if !q8 && !q51 {
        return false;
    }
    let Ok(columns_u32) = u32::try_from(columns) else {
        return false;
    };
    let Ok(layout) = row_layout(r#type, columns_u32) else {
        return false;
    };
    let mut expected = layout.row_bytes;
    for &dim in &dimensions[1..] {
        let Ok(product) = checked_mul(expected, dim) else {
            return false;
        };
        expected = product;
    }
    expected == bytes
}
