//! Lossless derived weight pack planner and streaming builder (`tokspur-pack-v2`), ported from `native/src/derived_pack.cpp`, `native/src/expert_lanes.cpp`, `native/src/lane_sidecar.cpp` and `scripts/plan_pack_v2.py`.
//!
//! Enforces 16 KiB Metal page alignment, byte-exact inverse reconstruction proofs across row-aligned 4 MiB chunks, exclusive `RENAME_EXCL` (`0x4`) publication, and strict 50 GiB disk-reserve checks before preflight and before every chunk write without sparse `ftruncate` overcommit or existing-pack size credit. Worker-side `adopt_derived_pack` remains in C++.

#![forbid(unsafe_code)]

pub mod build;
pub mod header;
pub mod lanes;
pub mod plan;

use std::fmt;
use std::io;
use std::path::PathBuf;

use tokspur_gguf::GgufError;
use tokspur_json::JsonError;

pub use build::{
    build_derived_pack, build_derived_pack_with_callbacks, build_derived_pack_with_progress,
    build_derived_pack_with_space_query,
};
pub use header::{MAX_HEADER_BYTES, build_header, build_manifest};
pub use lanes::{
    RowLayout, lane_sidecar_eligible, pack_rows, packed_quant_offset, row_layout, unpack_rows,
};
pub use plan::{
    DEFAULT_CHUNK_BYTES, DEFAULT_RESERVE_BYTES, EXPECTED_GPU_TENSORS, MAX_PACK_BYTES,
    MAX_STREAMING_BUFFER_BYTES, MTP_SIDECAR_SHA256, PACK_ALIGNMENT, PACK_ARCHITECTURE, PACK_RECIPE,
    PACK_VERSION, PackPlan, PackTensor, SOURCE_ARCHITECTURE, StaticPackPlan, TARGET_REVISION,
    aligned, build_gpu_pack_census, check_disk_reserve, is_cpu_pinned_embedding,
    is_gpu_pack_tensor, plan_derived_pack, plan_static_pack, verify_budget, verify_gpu_pack_census,
};

/// Errors produced while planning or building a `tokspur-pack-v2` derived weight pack.
pub enum PackError {
    /// Underlying GGUF shard or tensor validation failed.
    Gguf(GgufError),
    /// Manifest or plan JSON serialisation failed.
    Json(JsonError),
    /// Pack invariant, geometry or byte-proof failure.
    Format(&'static str),
    /// Filesystem or descriptor operation failed.
    Io {
        context: &'static str,
        source: io::Error,
    },
    /// Output pack path already exists or is inaccessible.
    OutputExists,
    /// A caller stopped the build before sealing. The incomplete temporary is removed.
    Cancelled,
    /// A sealed, byte-verified candidate remains after exclusive publication failed.
    Publication {
        candidate: PathBuf,
        source: io::Error,
    },
    /// Available unprivileged disk space cannot satisfy the upcoming write or remaining pack allocation plus the required safety reserve.
    InsufficientDiskSpace {
        available_bytes: u64,
        required_bytes: Option<u64>,
        reserve_bytes: u64,
    },
}

impl fmt::Debug for PackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf(error) => formatter.debug_tuple("Gguf").field(error).finish(),
            Self::Json(error) => formatter.debug_tuple("Json").field(error).finish(),
            Self::Format(reason) => formatter.debug_tuple("Format").field(reason).finish(),
            Self::Io { context, .. } => formatter
                .debug_struct("Io")
                .field("context", context)
                .field("source", &"[redacted]")
                .finish(),
            Self::OutputExists => formatter.write_str("OutputExists"),
            Self::Cancelled => formatter.write_str("Cancelled"),
            Self::Publication { .. } => {
                formatter.write_str("Publication { candidate: [redacted], source: [redacted] }")
            }
            Self::InsufficientDiskSpace {
                available_bytes,
                required_bytes,
                reserve_bytes,
            } => formatter
                .debug_struct("InsufficientDiskSpace")
                .field("available_bytes", available_bytes)
                .field("required_bytes", required_bytes)
                .field("reserve_bytes", reserve_bytes)
                .finish(),
        }
    }
}

impl fmt::Display for PackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
            Self::Format(reason) => formatter.write_str(reason),
            Self::Io { context, .. } => formatter.write_str(context),
            Self::OutputExists => {
                formatter.write_str("derived pack output already exists or is inaccessible")
            }
            Self::Cancelled => {
                formatter.write_str("pack build cancelled (incomplete temporary removed)")
            }
            Self::Publication { candidate, .. } => write!(
                formatter,
                "exclusive pack publication failed. Sealed candidate retained at {}. Keep it on the same filesystem and rename it to an absent output path before full verification",
                candidate.display()
            ),
            Self::InsufficientDiskSpace {
                available_bytes,
                required_bytes: Some(required),
                ..
            } => write!(
                formatter,
                "insufficient disk space: available {available_bytes} bytes is below required {required} bytes"
            ),
            Self::InsufficientDiskSpace {
                available_bytes,
                required_bytes: None,
                reserve_bytes,
            } => write!(
                formatter,
                "insufficient disk space: available {available_bytes} bytes cannot satisfy overflowing allocation plus {reserve_bytes} byte reserve"
            ),
        }
    }
}

impl std::error::Error for PackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gguf(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Io { source, .. } | Self::Publication { source, .. } => Some(source),
            Self::Format(_)
            | Self::OutputExists
            | Self::Cancelled
            | Self::InsufficientDiskSpace { .. } => None,
        }
    }
}

impl From<GgufError> for PackError {
    fn from(error: GgufError) -> Self {
        Self::Gguf(error)
    }
}

impl From<JsonError> for PackError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}
