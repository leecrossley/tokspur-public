//! Integration tests for `tokspur-pack`: 1,254-tensor static census, 64.4 GB vs 138.9 GB existing-pack no-credit refusal, mid-stream per-chunk reserve refusal + cleanup, `RENAME_EXCL` (`0x4`) refusal on existing files and symlinks, error `source()` / redacting `Debug`, and byte-for-byte C++ `adopt_derived_pack` round-trip acceptance.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::error::Error;
use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::process::Command;

use tokspur_gguf::{Gguf, ModelFiles};
use tokspur_pack::plan::plan_static_pack_with_space;
use tokspur_pack::{
    DEFAULT_RESERVE_BYTES, EXPECTED_GPU_TENSORS, PACK_ALIGNMENT, PACK_ARCHITECTURE, PackError,
    PackPlan, build_derived_pack, build_derived_pack_with_callbacks,
    build_derived_pack_with_space_query, build_gpu_pack_census, lane_sidecar_eligible, pack_rows,
    plan_derived_pack, row_layout, unpack_rows, verify_gpu_pack_census,
};
use tokspur_sys::random_bytes;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(prefix: &str) -> Self {
        let mut rnd = [0u8; 8];
        random_bytes(&mut rnd);
        let tag: u64 = u64::from_le_bytes(rnd);
        let base = fs::canonicalize(std::env::temp_dir()).unwrap();
        let path = base.join(format!("{prefix}_{}_{tag:016x}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if let Ok(entries) = fs::read_dir(&self.path) {
            for entry in entries.flatten() {
                if let Ok(ft) = entry.file_type()
                    && !ft.is_symlink()
                {
                    let mode = if ft.is_dir() { 0o700 } else { 0o600 };
                    let _ignored =
                        fs::set_permissions(entry.path(), fs::Permissions::from_mode(mode));
                }
            }
        }
        let _ignored = fs::remove_dir_all(&self.path);
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    let r = v % a;
    if r == 0 { v } else { v + (a - r) }
}

fn push_u(out: &mut Vec<u8>, v: u64, width: usize) {
    out.extend_from_slice(&v.to_le_bytes()[..width]);
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    push_u(out, s.len() as u64, 8);
    out.extend_from_slice(s.as_bytes());
}

struct TensorSpec<'a> {
    name: &'a str,
    r#type: u32,
    dims: &'a [u64],
    payload: Vec<u8>,
}

fn make_q8_rows(columns: u32, rows: u32, seed: u8) -> Vec<u8> {
    let blocks = columns / 32;
    let mut out = Vec::with_capacity((rows * blocks * 34) as usize);
    for r in 0..rows {
        for b in 0..blocks {
            let scale = 0x3800u16 | (((r + b + u32::from(seed)) & 0xff) as u16);
            out.extend_from_slice(&scale.to_le_bytes());
            for q in 0..32u32 {
                out.push(((r * 13 + b * 7 + q * 3 + u32::from(seed)) & 0xff) as u8);
            }
        }
    }
    out
}

fn make_q51_rows(columns: u32, rows: u32, seed: u8) -> Vec<u8> {
    let blocks = columns / 32;
    let mut out = Vec::with_capacity((rows * blocks * 24) as usize);
    for r in 0..rows {
        for b in 0..blocks {
            let d = 0x3800u16 | (((r + b + u32::from(seed)) & 0xff) as u16);
            let m = 0x3400u16 | (((r * 3 + b + u32::from(seed)) & 0xff) as u16);
            out.extend_from_slice(&d.to_le_bytes());
            out.extend_from_slice(&m.to_le_bytes());
            for h in 0..4u32 {
                out.push(((r + b + h + u32::from(seed)) & 0xff) as u8);
            }
            for q in 0..16u32 {
                out.push(((r * 17 + b * 5 + q + u32::from(seed)) & 0xff) as u8);
            }
        }
    }
    out
}

fn make_f32(elements: usize, seed: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(elements * 4);
    for i in 0..elements {
        let v = (((i + usize::from(seed)) % 97) as f32 - 48.0) * 0.125;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn write_gguf_shard(
    path: &Path,
    arch: &str,
    specs: &[TensorSpec<'_>],
    split: Option<(u16, u16, u32)>,
) {
    let alignment = 32u64;
    let mut hdr = Vec::new();
    push_u(&mut hdr, 0x4655_4747, 4);
    push_u(&mut hdr, 3, 4);
    push_u(&mut hdr, specs.len() as u64, 8);
    push_u(&mut hdr, if split.is_some() { 5 } else { 2 }, 8);

    push_str(&mut hdr, "general.architecture");
    push_u(&mut hdr, 8, 4);
    push_str(&mut hdr, arch);

    push_str(&mut hdr, "general.alignment");
    push_u(&mut hdr, 4, 4);
    push_u(&mut hdr, alignment, 4);

    if let Some((no, count, total)) = split {
        push_str(&mut hdr, "split.no");
        push_u(&mut hdr, 2, 4);
        push_u(&mut hdr, u64::from(no), 2);
        push_str(&mut hdr, "split.count");
        push_u(&mut hdr, 2, 4);
        push_u(&mut hdr, u64::from(count), 2);
        push_str(&mut hdr, "split.tensors.count");
        push_u(&mut hdr, 4, 4);
        push_u(&mut hdr, u64::from(total), 4);
    }

    let mut offsets = Vec::with_capacity(specs.len());
    let mut rel = 0u64;
    for s in specs {
        rel = align_up(rel, alignment);
        offsets.push(rel);
        rel += s.payload.len() as u64;
    }
    for (s, &off) in specs.iter().zip(&offsets) {
        push_str(&mut hdr, s.name);
        push_u(&mut hdr, s.dims.len() as u64, 4);
        for &d in s.dims {
            push_u(&mut hdr, d, 8);
        }
        push_u(&mut hdr, u64::from(s.r#type), 4);
        push_u(&mut hdr, off, 8);
    }
    let data_offset = align_up(hdr.len() as u64, alignment) as usize;
    hdr.resize(data_offset + rel as usize, 0);
    for (s, &off) in specs.iter().zip(&offsets) {
        let start = data_offset + off as usize;
        hdr[start..start + s.payload.len()].copy_from_slice(&s.payload);
    }
    fs::write(path, hdr).unwrap();
}

#[test]
fn static_census_and_no_existing_pack_credit_contract() {
    let census = build_gpu_pack_census().unwrap();
    assert_eq!(census.len(), EXPECTED_GPU_TENSORS);
    let plan = PackPlan {
        tensors: census,
        ..PackPlan::default()
    };
    assert!(verify_gpu_pack_census(&plan));

    let dir = TempDir::new("tokspur_pack_static");
    let fake_existing = dir.path.join("tokspur-pack-v2.asa.gguf");
    {
        let f = fs::File::create(&fake_existing).unwrap();
        // Sparse 84.6 GB existing pack file: must NEVER be added to available_space.
        f.set_len(84_600_000_000).unwrap();
    }

    let sixty_four_point_four_gb = 64_400_000_000u64;
    let total_two_tb = 2_000_000_000_000u64;
    let refused = plan_static_pack_with_space(
        &dir.path,
        Some(&fake_existing),
        None,
        DEFAULT_RESERVE_BYTES,
        sixty_four_point_four_gb,
        total_two_tb,
    )
    .unwrap();
    assert!(!refused.budget_passed);
    assert_eq!(refused.available_space, sixty_four_point_four_gb);
    assert_eq!(refused.existing_pack_bytes, 84_600_000_000);
    assert!(refused.budget_message.contains("deficit"));
}

#[test]
fn lane_roundtrip_and_error_redaction() {
    for (ty, cols) in [(12u32, 512u32), (7, 640), (8, 512)] {
        let layout = row_layout(ty, cols).unwrap();
        let total = (layout.row_bytes * 3) as usize;
        let mut orig = vec![0u8; total];
        random_bytes(&mut orig);
        let mut packed = vec![0xaa; total];
        let mut restored = vec![0x55; total];
        pack_rows(&layout, &orig, &mut packed).unwrap();
        unpack_rows(&layout, &packed, &mut restored).unwrap();
        assert_eq!(orig, restored);
    }

    assert!(lane_sidecar_eligible(
        b"blk.0.ffn_down_exps.weight",
        7,
        &[640, 2560, 512],
        (640 * 2560 * 512 / 32) * 24,
    ));
    assert!(!lane_sidecar_eligible(
        b"blk.1.ple_key.weight",
        8,
        &[2560, 10240],
        (2560 * 10240 / 32) * 34,
    ));

    let err = PackError::Io {
        context: "statvfs on target directory failed",
        source: std::io::Error::other("/Users/secret/private-pack.gguf"),
    };
    assert!(err.source().is_some());
    let dbg = format!("{err:?}");
    assert!(!dbg.contains("secret"), "secret leaked into Debug: {dbg}");
}

#[test]
fn build_reserve_exclusivity_and_cpp_adopt_roundtrip() {
    let dir = TempDir::new("tokspur_pack_roundtrip");
    let src_dir = dir.path.join("src");
    let out_dir = dir.path.join("out");
    fs::create_dir_all(&src_dir).unwrap();
    fs::create_dir_all(&out_dir).unwrap();

    let shard0 = [
        TensorSpec {
            name: "token_embd.weight",
            r#type: 8,
            dims: &[256, 4],
            payload: make_q8_rows(256, 4, 1),
        },
        TensorSpec {
            name: "blk.0.attn_q.weight",
            r#type: 8,
            dims: &[256, 256],
            payload: make_q8_rows(256, 256, 2),
        },
        TensorSpec {
            name: "blk.0.ffn_down_exps.weight",
            r#type: 7,
            dims: &[640, 4, 2],
            payload: make_q51_rows(640, 8, 3),
        },
    ];
    let shard1 = [
        TensorSpec {
            name: "blk.1.ple_key.weight",
            r#type: 8,
            dims: &[256, 4],
            payload: make_q8_rows(256, 4, 4),
        },
        TensorSpec {
            name: "output_hc_norm.weight",
            r#type: 0,
            dims: &[128],
            payload: make_f32(128, 5),
        },
    ];
    let mtp_specs = [
        TensorSpec {
            name: "blk.48.attn_q.weight",
            r#type: 8,
            dims: &[256, 8],
            payload: make_q8_rows(256, 8, 6),
        },
        TensorSpec {
            name: "blk.48.ffn_gate_inp.weight",
            r#type: 0,
            dims: &[64, 2],
            payload: make_f32(128, 7),
        },
    ];

    let s0 = src_dir.join("s0.gguf");
    let s1 = src_dir.join("s1.gguf");
    let mtp_path = src_dir.join("mtp.gguf");
    let total_src = (shard0.len() + shard1.len()) as u32;
    write_gguf_shard(&s0, "qwen4exp", &shard0, Some((0, 2, total_src)));
    write_gguf_shard(&s1, "qwen4exp", &shard1, Some((1, 2, total_src)));
    write_gguf_shard(&mtp_path, "qwen4exp", &mtp_specs, None);

    let source = ModelFiles::new(&[&s0, &s1]).unwrap();
    let mtp = ModelFiles::new(&[&mtp_path]).unwrap();
    let planned = plan_derived_pack(&source, Some(&mtp), 128_000_000_000).unwrap();
    assert_eq!(planned.tensors.len(), 6);
    assert!(planned.data_offset.is_multiple_of(PACK_ALIGNMENT));

    // Mid-stream per-chunk reserve drop: preflight (call 1) and chunk 0 (call 2) pass, chunk 1 (call 3) drops into reserve -> aborts and unlinks temp file.
    let refused_path = out_dir.join("refused.gguf");
    let mut calls = 0usize;
    let err = build_derived_pack_with_space_query(
        &source,
        Some(&mtp),
        &refused_path,
        DEFAULT_RESERVE_BYTES,
        32_768,
        |_| {
            calls += 1;
            if calls <= 2 {
                Ok(planned.file_bytes + DEFAULT_RESERVE_BYTES + (1 << 20))
            } else {
                Ok(DEFAULT_RESERVE_BYTES + 512)
            }
        },
    )
    .unwrap_err();
    assert!(matches!(err, PackError::InsufficientDiskSpace { .. }));
    assert_eq!(calls, 3);
    assert_eq!(fs::read_dir(&out_dir).unwrap().count(), 0);

    let cancelled = build_derived_pack_with_callbacks(
        &source,
        Some(&mtp),
        &refused_path,
        0,
        32_768,
        |_| Ok(u64::MAX),
        |written, _| written == 0,
    )
    .unwrap_err();
    assert!(matches!(cancelled, PackError::Cancelled));
    assert_eq!(fs::read_dir(&out_dir).unwrap().count(), 0);

    // An output race at completion preserves the sealed candidate and the other writer's file.
    let mut inserted = false;
    let failed = build_derived_pack_with_callbacks(
        &source,
        Some(&mtp),
        &refused_path,
        0,
        32_768,
        |_| Ok(u64::MAX),
        |written, total| {
            if written == total && !inserted {
                fs::write(&refused_path, b"other writer").unwrap();
                inserted = true;
            }
            true
        },
    )
    .unwrap_err();
    let PackError::Publication { candidate, .. } = failed else {
        panic!("completed candidate was not retained");
    };
    assert_eq!(fs::read(&refused_path).unwrap(), b"other writer");
    assert_eq!(fs::metadata(&candidate).unwrap().len(), planned.file_bytes);
    assert_eq!(
        fs::metadata(&candidate).unwrap().permissions().mode() & 0o777,
        0o444
    );
    assert_eq!(Gguf::from_path(&candidate).unwrap().tensors().len(), 6);
    fs::remove_file(candidate).unwrap();
    fs::remove_file(&refused_path).unwrap();

    // Real streaming build with 32 KiB chunks
    let pack_path = out_dir.join("tokspur-pack-v2.asa.gguf");
    let built = build_derived_pack(&source, Some(&mtp), &pack_path, 1 << 20, 32_768).unwrap();
    assert_eq!(built.file_bytes, planned.file_bytes);
    let meta = fs::symlink_metadata(&pack_path).unwrap();
    assert_eq!(meta.len(), planned.file_bytes);
    assert_eq!(meta.permissions().mode() & 0o777, 0o444);

    // Parse back with Rust tokspur-gguf
    let parsed_pack = Gguf::from_path(&pack_path).unwrap();
    assert_eq!(
        parsed_pack.string_value("general.architecture").unwrap(),
        PACK_ARCHITECTURE.as_bytes()
    );
    assert_eq!(parsed_pack.tensors().len(), 6);

    // Exclusive publication refuses overwrite of existing regular file and dangling symlink
    let exists_err =
        build_derived_pack(&source, Some(&mtp), &pack_path, 1 << 20, 32_768).unwrap_err();
    assert!(matches!(exists_err, PackError::OutputExists));
    assert_eq!(fs::read_dir(&out_dir).unwrap().count(), 1);

    let sym = out_dir.join("dangling.gguf");
    symlink("missing-target", &sym).unwrap();
    let sym_err = build_derived_pack(&source, Some(&mtp), &sym, 1 << 20, 32_768).unwrap_err();
    assert!(matches!(sym_err, PackError::OutputExists));
    fs::remove_file(&sym).unwrap();

    // Verify byte-for-byte acceptance by C++ adopt_derived_pack oracle!
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let cpp_build_dir = dir.path.join("cpp-oracle-build");
    let oracle_bin = if let Ok(explicit) = std::env::var("TOKSPUR_TEST_DERIVED_PACK_BIN") {
        PathBuf::from(explicit)
    } else {
        let target = cpp_build_dir.join("test-derived_pack");
        let status = Command::new("make")
            .arg("-C")
            .arg(repo_root.join("native"))
            .arg("-j4")
            .arg(format!("BUILD={}", cpp_build_dir.display()))
            .arg("ENABLE_METAL=0")
            .arg(&target)
            .status()
            .unwrap();
        assert!(
            status.success(),
            "building C++ test-derived_pack oracle failed"
        );
        target
    };
    let output = Command::new(&oracle_bin)
        .arg("--verify-pack")
        .arg(&pack_path)
        .arg(&s0)
        .arg(&s1)
        .arg("--mtp")
        .arg(&mtp_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "C++ adopt_derived_pack rejected Rust pack: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("PASS verified pack tensors=6"),
        "unexpected oracle output: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}
