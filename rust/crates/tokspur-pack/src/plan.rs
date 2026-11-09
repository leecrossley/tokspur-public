//! Pack layout planning and 50 GiB disk-reserve admission ported from `native/src/derived_pack.cpp` and `scripts/plan_pack_v2.py`.

use std::collections::BTreeSet;
use std::fs::File;
use std::os::fd::AsFd;
use std::path::Path;

use tokspur_gguf::{ModelFiles, checked_add, checked_mul, encoding};
use tokspur_json::Json;
use tokspur_sys::fd;

use crate::PackError;
use crate::header::build_header;
use crate::lanes::lane_sidecar_eligible;

pub const PACK_ALIGNMENT: u64 = 16_384;
pub const PACK_VERSION: u32 = 2;
pub const PACK_RECIPE: &str = "tokspur-pack-v2";
pub const PACK_ARCHITECTURE: &str = "tokspur_pack";
pub const SOURCE_ARCHITECTURE: &str = "qwen4exp";
pub const TARGET_REVISION: &str = "766911a6b7369840a91dbcd95f9f997acaab6cd6";
pub const MTP_SIDECAR_SHA256: &str =
    "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6";
pub const EXPECTED_GPU_TENSORS: usize = 1254;
pub const DEFAULT_RESERVE_BYTES: u64 = 50 * 1024 * 1024 * 1024;
pub const DEFAULT_CHUNK_BYTES: u64 = 4_194_304;
pub const MAX_STREAMING_BUFFER_BYTES: u64 = 512 * 1024 * 1024;
pub const MAX_PACK_BYTES: u64 = 128_000_000_000;
const EXPECTED_LANES_V1_BYTES: u64 = 31_412_379_904;
const STATIC_HEADER_ESTIMATE: u64 = 512 * 1024;

/// One planned tensor inside `tokspur-pack-v2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackTensor {
    pub name: String,
    pub r#type: u32,
    pub layout: String,
    pub dimensions: Vec<u64>,
    pub bytes: u64,
    /// Aligned offset relative to `data_offset`.
    pub offset: u64,
    pub source_sha256: String,
    pub packed_sha256: String,
}

/// Complete derived pack plan matching `tokspur::derived_pack::PackPlan`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackPlan {
    pub tensors: Vec<PackTensor>,
    pub payload_bytes: u64,
    pub file_bytes: u64,
    pub data_offset: u64,
    pub recipe: String,
    pub version: u32,
    pub alignment: u32,
}

impl Default for PackPlan {
    fn default() -> Self {
        Self {
            tensors: Vec::new(),
            payload_bytes: 0,
            file_bytes: 0,
            data_offset: 0,
            recipe: PACK_RECIPE.to_owned(),
            version: PACK_VERSION,
            alignment: PACK_ALIGNMENT as u32,
        }
    }
}

/// Static census plan and strict filesystem reserve evaluation matching `scripts/plan_pack_v2.py`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticPackPlan {
    pub tensors: Vec<PackTensor>,
    pub payload_bytes: u64,
    pub planned_file_bytes: u64,
    pub peak_additional_allocation: u64,
    pub reserve_bytes: u64,
    pub available_space: u64,
    pub existing_pack_bytes: u64,
    pub total_capacity: u64,
    pub budget_passed: bool,
    pub budget_message: String,
    pub existing_lanes_preserved: bool,
}

impl StaticPackPlan {
    /// Serialises the summary JSON produced by `plan_pack_v2.py --json`.
    pub fn summary_json(&self) -> Result<Vec<u8>, PackError> {
        let doc = Json::new_object([
            ("schema", Json::integer(u64::from(PACK_VERSION))),
            ("recipe", Json::from(PACK_RECIPE)),
            ("architecture", Json::from(PACK_ARCHITECTURE)),
            ("source_architecture", Json::from(SOURCE_ARCHITECTURE)),
            ("target_revision", Json::from(TARGET_REVISION)),
            ("mtp_sidecar_sha256", Json::from(MTP_SIDECAR_SHA256)),
            ("alignment", Json::integer(PACK_ALIGNMENT)),
            ("tensor_count", Json::integer(self.tensors.len() as u64)),
            ("payload_bytes", Json::integer(self.payload_bytes)),
            ("planned_file_bytes", Json::integer(self.planned_file_bytes)),
            (
                "peak_additional_allocation",
                Json::integer(self.peak_additional_allocation),
            ),
            ("reserve_bytes", Json::integer(self.reserve_bytes)),
            ("available_space", Json::integer(self.available_space)),
            (
                "existing_pack_bytes",
                Json::integer(self.existing_pack_bytes),
            ),
            ("total_capacity", Json::integer(self.total_capacity)),
            ("budget_passed", Json::from(self.budget_passed)),
            ("budget_message", Json::from(self.budget_message.as_str())),
            (
                "existing_lanes_preserved",
                Json::from(self.existing_lanes_preserved),
            ),
            ("tensors", Json::integer(self.tensors.len() as u64)),
        ]);
        Ok(doc.dump()?)
    }
}

/// Aligns `n` upwards to `boundary`.
pub fn aligned(n: u64, boundary: u64) -> Result<u64, PackError> {
    if boundary == 0 {
        return Err(PackError::Format("zero alignment boundary"));
    }
    let added = checked_add(n, boundary - 1)?;
    Ok(added / boundary * boundary)
}

/// Returns whether `name` is a CPU-pinned embedding table kept outside the GPU pack.
pub fn is_cpu_pinned_embedding(name: &str) -> bool {
    matches!(
        name,
        "token_embd.weight" | "per_layer_token_embd.weight" | "per_layer_token_embd"
    )
}

/// Returns whether `name` belongs in the GPU derived weight pack.
pub fn is_gpu_pack_tensor(name: &str) -> bool {
    !is_cpu_pinned_embedding(name)
}

/// Verifies the full 1,254-tensor census invariants from `derived_pack.cpp:263-284`.
pub fn verify_gpu_pack_census(plan: &PackPlan) -> bool {
    if plan.tensors.len() != EXPECTED_GPU_TENSORS {
        return false;
    }
    let mut has_output = false;
    let mut has_output_hc = false;
    let mut has_gdn = false;
    let mut has_qsa_l3 = false;
    let mut has_ple_l1 = false;
    let mut mtp_count = 0usize;
    let mut seen = BTreeSet::new();

    for t in &plan.tensors {
        if !is_gpu_pack_tensor(&t.name) || !seen.insert(t.name.as_str()) {
            return false;
        }
        if t.name == "output.weight" {
            has_output = true;
        }
        if t.name == "output_hc_norm.weight" {
            has_output_hc = true;
        }
        if t.name == "blk.0.ssm_a" {
            has_gdn = true;
        }
        if t.name == "blk.3.attn_q.weight" {
            has_qsa_l3 = true;
        }
        if t.name == "blk.1.ple_key.weight" {
            has_ple_l1 = true;
        }
        if t.name.starts_with("blk.48.") {
            mtp_count += 1;
        }
    }
    has_output && has_output_hc && has_gdn && has_qsa_l3 && has_ple_l1 && mtp_count == 32
}

/// Overflow-safe disk reserve check matching `derived_pack.cpp:286-294`.
pub fn check_disk_reserve(
    available_bytes: u64,
    peak_additional_bytes: u64,
    reserve_bytes: u64,
) -> bool {
    if available_bytes == 0 {
        return false;
    }
    let Some(required) = peak_additional_bytes.checked_add(reserve_bytes) else {
        return false;
    };
    available_bytes >= required
}

/// Evaluates `available_space >= peak_additional + reserve_bytes` with detailed diagnostic text matching `plan_pack_v2.py`.
pub fn verify_budget(
    available_space: u64,
    peak_additional: u64,
    reserve_bytes: u64,
    total_capacity: Option<u64>,
) -> (bool, String) {
    if available_space == 0 {
        return (
            false,
            "zero or negative available space triggers immediate refusal".to_owned(),
        );
    }
    if let Some(total) = total_capacity
        && peak_additional > total
    {
        return (
            false,
            "peak allocation exceeds total disk capacity".to_owned(),
        );
    }
    let Some(required) = peak_additional.checked_add(reserve_bytes) else {
        return (
            false,
            "required peak + reserve overflows 64-bit byte counter".to_owned(),
        );
    };
    if available_space < required {
        let deficit = required - available_space;
        return (
            false,
            format!(
                "available space ({available_space} bytes) is below required peak + reserve ({required} bytes), deficit: {deficit} bytes"
            ),
        );
    }
    (true, "budget check passed".to_owned())
}

fn tensor_layout_name(name: &[u8], r#type: u32, dimensions: &[u64], bytes: u64) -> &'static str {
    if lane_sidecar_eligible(name, r#type, dimensions, bytes) {
        if r#type == 8 {
            return "q8_lanes_v1";
        }
        if r#type == 7 {
            return "q51_lanes_v1";
        }
    }
    "plain"
}

/// Plans `tokspur-pack-v2` from live `ModelFiles` and optional MTP sidecar (`derived_pack.cpp:296-356`).
pub fn plan_derived_pack(
    source: &ModelFiles,
    mtp_sidecar: Option<&ModelFiles>,
    byte_budget: u64,
) -> Result<PackPlan, PackError> {
    source.assert_unchanged()?;
    if source.architecture() == PACK_ARCHITECTURE.as_bytes() {
        return Err(PackError::Format("cannot repack an existing derived pack"));
    }

    let mut plan = PackPlan::default();
    let mut seen = BTreeSet::new();

    for name_bytes in source.tensors().keys() {
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| PackError::Format("non-UTF-8 tensor name"))?;
        if !is_gpu_pack_tensor(name) {
            continue;
        }
        let tensor = source.tensor(name_bytes)?;
        let layout =
            tensor_layout_name(name_bytes, tensor.r#type, &tensor.dimensions, tensor.bytes);
        if !seen.insert(name.to_owned()) {
            return Err(PackError::Format("duplicate tensor in source"));
        }
        plan.tensors.push(PackTensor {
            name: name.to_owned(),
            r#type: tensor.r#type,
            layout: layout.to_owned(),
            dimensions: tensor.dimensions.clone(),
            bytes: tensor.bytes,
            offset: 0,
            source_sha256: String::new(),
            packed_sha256: String::new(),
        });
        plan.payload_bytes = checked_add(plan.payload_bytes, tensor.bytes)?;
    }

    if let Some(mtp) = mtp_sidecar {
        mtp.assert_unchanged()?;
        for name_bytes in mtp.tensors().keys() {
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| PackError::Format("non-UTF-8 MTP tensor name"))?;
            if !is_gpu_pack_tensor(name) {
                continue;
            }
            let tensor = mtp.tensor(name_bytes)?;
            let layout =
                tensor_layout_name(name_bytes, tensor.r#type, &tensor.dimensions, tensor.bytes);
            if seen.insert(name.to_owned()) {
                plan.tensors.push(PackTensor {
                    name: name.to_owned(),
                    r#type: tensor.r#type,
                    layout: layout.to_owned(),
                    dimensions: tensor.dimensions.clone(),
                    bytes: tensor.bytes,
                    offset: 0,
                    source_sha256: String::new(),
                    packed_sha256: String::new(),
                });
                plan.payload_bytes = checked_add(plan.payload_bytes, tensor.bytes)?;
            }
        }
    }

    if plan.tensors.is_empty() {
        return Err(PackError::Format("empty derived pack tensor selection"));
    }

    let mut relative_offset = 0u64;
    for t in &mut plan.tensors {
        relative_offset = aligned(relative_offset, PACK_ALIGNMENT)?;
        t.offset = relative_offset;
        relative_offset = checked_add(relative_offset, t.bytes)?;
    }

    let initial_header = build_header(&plan)?;
    plan.data_offset = initial_header.len() as u64;
    plan.file_bytes = checked_add(plan.data_offset, relative_offset)?;
    if plan.file_bytes > byte_budget {
        return Err(PackError::Format(
            "planned derived pack exceeds byte budget",
        ));
    }

    source.assert_unchanged()?;
    if let Some(mtp) = mtp_sidecar {
        mtp.assert_unchanged()?;
    }
    Ok(plan)
}

fn compute_tensor_bytes(r#type: u32, dimensions: &[u64]) -> Result<u64, PackError> {
    let enc = encoding(r#type)?;
    let mut elements = 1u64;
    for &d in dimensions {
        elements = checked_mul(elements, d)?;
    }
    let block_elems = u64::from(enc.elements_per_block);
    if !elements.is_multiple_of(block_elems) {
        return Err(PackError::Format(
            "static census tensor is not block-aligned",
        ));
    }
    Ok(checked_mul(
        elements / block_elems,
        u64::from(enc.bytes_per_block),
    )?)
}

fn make_static_tensor(
    name: String,
    r#type: u32,
    dimensions: Vec<u64>,
) -> Result<PackTensor, PackError> {
    let bytes = compute_tensor_bytes(r#type, &dimensions)?;
    let layout = tensor_layout_name(name.as_bytes(), r#type, &dimensions, bytes).to_owned();
    Ok(PackTensor {
        name,
        r#type,
        layout,
        dimensions,
        bytes,
        offset: 0,
        source_sha256: String::new(),
        packed_sha256: String::new(),
    })
}

/// Builds the complete 1,254-tensor static inventory matching `scripts/plan_pack_v2.py:build_gpu_pack_census`.
pub fn build_gpu_pack_census() -> Result<Vec<PackTensor>, PackError> {
    let mut census = Vec::with_capacity(EXPECTED_GPU_TENSORS);
    for layer in 0..48u32 {
        let prefix = format!("blk.{layer}.");
        for pathway in ["hc_attn", "hc_ffn"] {
            let root = format!("{prefix}{pathway}");
            census.push(make_static_tensor(
                format!("{root}_norm.weight"),
                0,
                vec![10240],
            )?);
            census.push(make_static_tensor(
                format!("{root}_down.weight"),
                8,
                vec![10240, 320],
            )?);
            census.push(make_static_tensor(
                format!("{root}_up.weight"),
                8,
                vec![320, 10240],
            )?);
            census.push(make_static_tensor(
                format!("{root}_inject.weight"),
                0,
                vec![10240, 4],
            )?);
        }

        census.push(make_static_tensor(
            format!("{prefix}ffn_gate_inp.weight"),
            0,
            vec![2560, 512],
        )?);
        census.push(make_static_tensor(
            format!("{prefix}ffn_gate_inp_shexp.weight"),
            0,
            vec![2560],
        )?);

        let gate_up_type = if layer == 2 { 13 } else { 12 };
        let down_type = if matches!(layer, 2 | 4 | 30 | 46 | 47) {
            8
        } else {
            7
        };

        census.push(make_static_tensor(
            format!("{prefix}ffn_gate_exps.weight"),
            gate_up_type,
            vec![2560, 640, 512],
        )?);
        census.push(make_static_tensor(
            format!("{prefix}ffn_up_exps.weight"),
            gate_up_type,
            vec![2560, 640, 512],
        )?);
        census.push(make_static_tensor(
            format!("{prefix}ffn_down_exps.weight"),
            down_type,
            vec![640, 2560, 512],
        )?);
        census.push(make_static_tensor(
            format!("{prefix}ffn_gate_shexp.weight"),
            8,
            vec![2560, 640],
        )?);
        census.push(make_static_tensor(
            format!("{prefix}ffn_up_shexp.weight"),
            8,
            vec![2560, 640],
        )?);
        census.push(make_static_tensor(
            format!("{prefix}ffn_down_shexp.weight"),
            8,
            vec![640, 2560],
        )?);

        if layer % 4 == 3 {
            census.push(make_static_tensor(
                format!("{prefix}attn_q.weight"),
                8,
                vec![2560, 12288],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}attn_k.weight"),
                8,
                vec![2560, 512],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}attn_v.weight"),
                8,
                vec![2560, 512],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}attn_output.weight"),
                8,
                vec![6144, 2560],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}attn_q_norm.weight"),
                0,
                vec![256],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}attn_k_norm.weight"),
                0,
                vec![256],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}indexer.q_proj.weight"),
                30,
                vec![2560, 512],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}indexer.k_proj.weight"),
                30,
                vec![2560, 128],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}indexer.q_norm.weight"),
                0,
                vec![128],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}indexer.k_norm.weight"),
                0,
                vec![128],
            )?);
        } else {
            census.push(make_static_tensor(
                format!("{prefix}attn_qkv.weight"),
                8,
                vec![2560, 10240],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}attn_gate.weight"),
                8,
                vec![2560, 6144],
            )?);
            census.push(make_static_tensor(format!("{prefix}ssm_a"), 0, vec![48])?);
            census.push(make_static_tensor(
                format!("{prefix}ssm_dt.bias"),
                0,
                vec![48],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}ssm_alpha.weight"),
                0,
                vec![2560, 48],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}ssm_beta.weight"),
                0,
                vec![2560, 48],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}ssm_conv1d.weight"),
                0,
                vec![4, 10240],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}ssm_norm.weight"),
                0,
                vec![128],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}ssm_out.weight"),
                8,
                vec![6144, 2560],
            )?);
        }

        if layer == 1 {
            census.push(make_static_tensor(
                format!("{prefix}ple_key.weight"),
                8,
                vec![2560, 10240],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}ple_value.weight"),
                8,
                vec![2560, 2560],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}ple_norm_key.weight"),
                0,
                vec![10240],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}ple_norm_query.weight"),
                0,
                vec![10240],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}ple_norm_conv.weight"),
                0,
                vec![10240],
            )?);
            census.push(make_static_tensor(
                format!("{prefix}ple_conv1d.weight"),
                0,
                vec![4, 10240],
            )?);
        }
    }

    census.push(make_static_tensor(
        "output_hc_norm.weight".to_owned(),
        0,
        vec![10240],
    )?);
    census.push(make_static_tensor(
        "output_hc_down.weight".to_owned(),
        8,
        vec![10240, 320],
    )?);
    census.push(make_static_tensor(
        "output_hc_up.weight".to_owned(),
        8,
        vec![320, 10240],
    )?);
    census.push(make_static_tensor(
        "output.weight".to_owned(),
        8,
        vec![2560, 248_320],
    )?);

    let mtp_defs: &[(&str, &[u64], u32)] = &[
        ("attn_k.weight", &[2560, 512], 8),
        ("attn_k_norm.weight", &[256], 0),
        ("attn_output.weight", &[6144, 2560], 8),
        ("attn_q.weight", &[2560, 12288], 8),
        ("attn_q_norm.weight", &[256], 0),
        ("attn_v.weight", &[2560, 512], 8),
        ("ffn_down_exps.weight", &[640, 2560, 512], 8),
        ("ffn_down_shexp.weight", &[640, 2560], 8),
        ("ffn_gate_exps.weight", &[2560, 640, 512], 8),
        ("ffn_gate_inp.weight", &[2560, 512], 0),
        ("ffn_gate_inp_shexp.weight", &[2560], 0),
        ("ffn_gate_shexp.weight", &[2560, 640], 8),
        ("ffn_up_exps.weight", &[2560, 640, 512], 8),
        ("ffn_up_shexp.weight", &[2560, 640], 8),
        ("hc_attn_down.weight", &[10240, 320], 8),
        ("hc_attn_inject.weight", &[10240, 4], 8),
        ("hc_attn_norm.weight", &[10240], 0),
        ("hc_attn_up.weight", &[320, 10240], 8),
        ("hc_ffn_down.weight", &[10240, 320], 8),
        ("hc_ffn_inject.weight", &[10240, 4], 8),
        ("hc_ffn_norm.weight", &[10240], 0),
        ("hc_ffn_up.weight", &[320, 10240], 8),
        ("indexer.k_norm.weight", &[128], 0),
        ("indexer.k_proj.weight", &[2560, 128], 30),
        ("indexer.q_norm.weight", &[128], 0),
        ("indexer.q_proj.weight", &[2560, 512], 30),
        ("nextn.eh_proj.weight", &[5120, 2560], 8),
        ("nextn.enorm.weight", &[2560], 0),
        ("nextn.hc_head_down.weight", &[10240, 320], 8),
        ("nextn.hc_head_norm.weight", &[10240], 0),
        ("nextn.hc_head_up.weight", &[320, 10240], 8),
        ("nextn.hnorm.weight", &[10240], 0),
    ];
    for &(suffix, dims, r#type) in mtp_defs {
        census.push(make_static_tensor(
            format!("blk.48.{suffix}"),
            r#type,
            dims.to_vec(),
        )?);
    }

    Ok(census)
}

fn query_existing_ancestor_space(target_dir: &Path) -> Result<(u64, u64), PackError> {
    let mut resolved = target_dir;
    while !resolved.exists() {
        let Some(parent) = resolved.parent() else {
            break;
        };
        if parent == resolved {
            break;
        }
        resolved = parent;
    }
    let dir_file = File::open(resolved).map_err(|source| PackError::Io {
        context: "cannot open target directory ancestor for statvfs",
        source,
    })?;
    let space = fd::filesystem_space(dir_file.as_fd()).map_err(|source| PackError::Io {
        context: "statvfs on target directory failed",
        source,
    })?;
    Ok((space.available_bytes, space.total_bytes))
}

/// Computes the shard-free 1,254-tensor static pack plan and evaluates filesystem reserve strictly without existing-pack size credit.
pub fn plan_static_pack(
    target_dir: &Path,
    existing_pack_hint: Option<&Path>,
    existing_lanes_path: Option<&Path>,
    reserve_bytes: u64,
) -> Result<StaticPackPlan, PackError> {
    let (available, total) = query_existing_ancestor_space(target_dir)?;
    plan_static_pack_with_space(
        target_dir,
        existing_pack_hint,
        existing_lanes_path,
        reserve_bytes,
        available,
        total,
    )
}

/// Computes the shard-free static pack plan against explicit `(available_space, total_capacity)` figures.
pub fn plan_static_pack_with_space(
    target_dir: &Path,
    existing_pack_hint: Option<&Path>,
    existing_lanes_path: Option<&Path>,
    reserve_bytes: u64,
    available_space: u64,
    total_capacity: u64,
) -> Result<StaticPackPlan, PackError> {
    let mut tensors = build_gpu_pack_census()?;
    let data_offset = aligned(STATIC_HEADER_ESTIMATE, PACK_ALIGNMENT)?;
    let mut current_offset = 0u64;
    let mut payload_bytes = 0u64;
    for t in &mut tensors {
        current_offset = aligned(current_offset, PACK_ALIGNMENT)?;
        t.offset = current_offset;
        current_offset = checked_add(current_offset, t.bytes)?;
        payload_bytes = checked_add(payload_bytes, t.bytes)?;
    }
    let planned_file_bytes = checked_add(data_offset, current_offset)?;
    let peak_additional_allocation = checked_add(planned_file_bytes, MAX_STREAMING_BUFFER_BYTES)?;

    // Existing packs stay allocated throughout construction and RENAME_EXCL refuses replacement, so their size is informational only and never added to available_space.
    fn optional_file_len(path: &Path) -> Option<u64> {
        match std::fs::metadata(path) {
            Ok(meta) => Some(meta.len()),
            Err(_) => None,
        }
    }
    let candidate_asa = target_dir.join("tokspur-pack-v2.asa.gguf");
    let candidate_legacy = target_dir.join("tokspur-pack-v2.gguf");
    let existing_pack_bytes = existing_pack_hint
        .and_then(optional_file_len)
        .or_else(|| optional_file_len(&candidate_asa))
        .or_else(|| optional_file_len(&candidate_legacy))
        .unwrap_or(0);

    let (budget_passed, budget_message) = verify_budget(
        available_space,
        peak_additional_allocation,
        reserve_bytes,
        Some(total_capacity),
    );
    let existing_lanes_preserved = existing_lanes_path
        .and_then(|p| std::fs::metadata(p).ok())
        .is_some_and(|m| m.len() == EXPECTED_LANES_V1_BYTES);

    Ok(StaticPackPlan {
        tensors,
        payload_bytes,
        planned_file_bytes,
        peak_additional_allocation,
        reserve_bytes,
        available_space,
        existing_pack_bytes,
        total_capacity,
        budget_passed,
        budget_message,
        existing_lanes_preserved,
    })
}
