//! Canonical 16 KiB-aligned GGUF v3 header and manifest builder matching `native/src/derived_pack.cpp:72-137` byte-for-byte.

use tokspur_json::Json;

use crate::PackError;
use crate::plan::{
    MTP_SIDECAR_SHA256, PACK_ALIGNMENT, PACK_ARCHITECTURE, PACK_RECIPE, PACK_VERSION, PackPlan,
    SOURCE_ARCHITECTURE, TARGET_REVISION, aligned,
};

/// Maximum permitted manifest or header byte size (`max_header_bytes` in `derived_pack.cpp`).
pub const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;

const ZERO_DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &[u8]) {
    push_u64(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn push_metadata_string(out: &mut Vec<u8>, key: &[u8], value: &[u8]) {
    push_string(out, key);
    push_u32(out, 8);
    push_string(out, value);
}

fn push_metadata_u32(out: &mut Vec<u8>, key: &[u8], value: u32) {
    push_string(out, key);
    push_u32(out, 4);
    push_u32(out, value);
}

/// Serialises the `tokspur_pack.manifest` JSON document with exact C++ field ordering.
pub fn build_manifest(plan: &PackPlan) -> Result<Vec<u8>, PackError> {
    let mut tensors = Vec::with_capacity(plan.tensors.len());
    for t in &plan.tensors {
        let dims: Vec<Json> = t.dimensions.iter().copied().map(Json::integer).collect();
        let src_sha = if t.source_sha256.is_empty() {
            ZERO_DIGEST
        } else {
            t.source_sha256.as_str()
        };
        let pack_sha = if t.packed_sha256.is_empty() {
            ZERO_DIGEST
        } else {
            t.packed_sha256.as_str()
        };
        tensors.push(Json::new_object([
            ("name", Json::from(t.name.as_str())),
            ("type", Json::integer(u64::from(t.r#type))),
            ("layout", Json::from(t.layout.as_str())),
            ("dimensions", Json::Array(dims)),
            ("bytes", Json::integer(t.bytes)),
            ("offset", Json::integer(t.offset)),
            ("source_sha256", Json::from(src_sha)),
            ("packed_sha256", Json::from(pack_sha)),
        ]));
    }
    let root = Json::new_object([
        ("schema", Json::integer(u64::from(PACK_VERSION))),
        ("recipe", Json::from(PACK_RECIPE)),
        ("source_architecture", Json::from(SOURCE_ARCHITECTURE)),
        ("target_revision", Json::from(TARGET_REVISION)),
        ("mtp_sidecar_sha256", Json::from(MTP_SIDECAR_SHA256)),
        ("tensors", Json::Array(tensors)),
    ]);
    Ok(root.dump_limited(MAX_HEADER_BYTES)?)
}

/// Builds the 16 KiB-aligned GGUF v3 header containing the 5 metadata keys and all tensor descriptors.
pub fn build_header(plan: &PackPlan) -> Result<Vec<u8>, PackError> {
    let manifest = build_manifest(plan)?;
    let mut out = Vec::with_capacity(manifest.len().saturating_add(65_536));
    push_u32(&mut out, 0x4655_4747);
    push_u32(&mut out, 3);
    push_u64(&mut out, plan.tensors.len() as u64);
    push_u64(&mut out, 5);
    push_metadata_string(
        &mut out,
        b"general.architecture",
        PACK_ARCHITECTURE.as_bytes(),
    );
    push_metadata_u32(&mut out, b"general.alignment", PACK_ALIGNMENT as u32);
    push_metadata_u32(&mut out, b"tokspur_pack.version", PACK_VERSION);
    push_metadata_string(&mut out, b"tokspur_pack.recipe", PACK_RECIPE.as_bytes());
    push_metadata_string(&mut out, b"tokspur_pack.manifest", &manifest);

    for t in &plan.tensors {
        push_string(&mut out, t.name.as_bytes());
        push_u32(&mut out, t.dimensions.len() as u32);
        for &d in &t.dimensions {
            push_u64(&mut out, d);
        }
        push_u32(&mut out, t.r#type);
        push_u64(&mut out, t.offset);
    }

    if out.len() > MAX_HEADER_BYTES {
        return Err(PackError::Format("derived pack header exceeds bound"));
    }
    let padded = usize::try_from(aligned(out.len() as u64, PACK_ALIGNMENT)?)
        .map_err(|_| PackError::Format("derived pack header exceeds address space"))?;
    out.resize(padded, 0);
    Ok(out)
}
