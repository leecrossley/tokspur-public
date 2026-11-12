//! Build and staged-package admission (`tokspur verify-build`): Mach-O system-link checks, separate `MTLB` `.metallib` verification, and source/compiler/receipt provenance verification.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs;
use std::os::fd::AsFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use tokspur_json::{Json, JsonLimits, parse_json_limited};
use tokspur_sys::effective_uid;
use tokspur_sys::fd::{
    FileStat, O_CLOEXEC, O_NOFOLLOW, O_RDONLY, fstat, lstat_at, openat, pread_exact,
};
use tokspur_text::sha256::{Sha256, hex};

use crate::error::CliError;
use crate::layout::{
    CheckoutLayout, LaunchTargets, RuntimeLayout, StagedLayout, ensure_no_symlink_components,
};
use crate::state::open_nofollow_dir;

const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_RECEIPT_BYTES: u64 = 4 * 1024 * 1024;
const MAX_SOURCE_BYTES: u64 = 16 * 1024 * 1024;
const MACHO_MAGIC_64_LE: [u8; 4] = [0xcf, 0xfa, 0xed, 0xfe];
const MACHO_CIGAM_64_BE: [u8; 4] = [0xfe, 0xed, 0xfa, 0xcf];
const FAT_MAGIC_BE: [u8; 4] = [0xca, 0xfe, 0xba, 0xbe];
const MTLB_MAGIC: [u8; 4] = *b"MTLB";
const FORBIDDEN_METALLIB_MARKERS: [&[u8]; 2] = [b"probe_unchecked", b"graph_capture_routes"];
const FORBIDDEN_WORKER_MARKERS: [&[u8]; 1] = [b"tokspur-carrier-diagnostic-v1"];

/// Opens `path` with `O_NOFOLLOW`, validates single-link regular-file ownership/mode bounds, and returns `(FileStat, sha256_hex, header_4_bytes)`.
pub fn inspect_and_hash_nofollow(
    path: &Path,
    max_bytes: u64,
    label: &str,
) -> Result<(FileStat, String, [u8; 4]), CliError> {
    let mut hasher = Sha256::new();
    let mut tail = Vec::new();
    let mut artifact_markers: &[&[u8]] = &[];
    let mut first = true;
    let mut forbidden = false;
    let (before, magic) = inspect_reading(path, max_bytes, label, true, &mut |bytes| {
        hasher.update(bytes);
        if first {
            first = false;
            if bytes.starts_with(&MTLB_MAGIC) {
                artifact_markers = &FORBIDDEN_METALLIB_MARKERS;
            } else if path.file_name().is_none_or(|name| name != "tokspur")
                && (bytes.starts_with(&MACHO_MAGIC_64_LE)
                    || bytes.starts_with(&MACHO_CIGAM_64_BE)
                    || bytes.starts_with(&FAT_MAGIC_BE))
            {
                artifact_markers = &FORBIDDEN_WORKER_MARKERS;
            }
        }
        if !artifact_markers.is_empty() {
            tail.extend_from_slice(bytes);
            forbidden |= artifact_markers
                .iter()
                .any(|marker| tail.windows(marker.len()).any(|w| w == *marker));
            let keep = artifact_markers
                .iter()
                .map(|marker| marker.len())
                .max()
                .unwrap_or(0);
            if tail.len() > keep {
                tail.drain(..tail.len() - keep);
            }
        }
    })?;
    if forbidden {
        return Err(CliError::Validation(
            "diagnostic or probe artefact refused by production verification".to_owned(),
        ));
    }
    Ok((before, hex(&hasher.finalize()), magic))
}

/// The checks of [`inspect_and_hash_nofollow`] without the digest, reading only the 4 header bytes, for a file whose content is hashed elsewhere or never compared.
pub fn inspect_nofollow(
    path: &Path,
    max_bytes: u64,
    label: &str,
) -> Result<(FileStat, [u8; 4]), CliError> {
    inspect_reading(path, max_bytes, label, false, &mut |_| {})
}

/// Opens and validates `path`, passes its bytes to `sink` (the whole file, or only the 4 header bytes) and confirms the file did not change meanwhile.
fn inspect_reading(
    path: &Path,
    max_bytes: u64,
    label: &str,
    whole: bool,
    sink: &mut dyn FnMut(&[u8]),
) -> Result<(FileStat, [u8; 4]), CliError> {
    ensure_no_symlink_components(path, label)?;
    let parent = path
        .parent()
        .ok_or_else(|| CliError::Validation(format!("{label} has no parent directory")))?;
    let name = path
        .file_name()
        .ok_or_else(|| CliError::Validation(format!("{label} has no filename")))?;
    let dir_fd = open_nofollow_dir(parent, label)?;
    let c_name = CString::new(name.as_bytes())
        .map_err(|_| CliError::Validation(format!("{label} filename contains NUL")))?;
    let file = openat(
        dir_fd.as_fd(),
        &c_name,
        O_RDONLY | O_NOFOLLOW | O_CLOEXEC,
        0,
    )
    .map_err(|source| CliError::Io {
        context: format!("cannot open {label}"),
        source,
    })?;
    let before = fstat(file.as_fd()).map_err(|source| CliError::Io {
        context: format!("cannot stat {label}"),
        source,
    })?;
    if !before.is_regular()
        || before.nlink != 1
        || before.size == 0
        || before.size > max_bytes
        || (before.mode & 0o6022) != 0
    {
        return Err(CliError::Validation(format!(
            "{label} must be a bounded, single-link regular file without setuid/group/world write bits"
        )));
    }

    let end = if whole {
        before.size
    } else {
        before.size.min(4)
    };
    let mut magic = [0u8; 4];
    // Most inventory entries are small source files, so the buffer is sized to the read up to 1 MiB.
    let mut buf = vec![0u8; end.min(1024 * 1024) as usize];
    let mut at = 0u64;
    while at < end {
        let take = (end - at).min(buf.len() as u64) as usize;
        pread_exact(file.as_fd(), &mut buf[..take], at).map_err(|source| CliError::Io {
            context: format!("failed reading {label}"),
            source,
        })?;
        if at == 0 && take >= 4 {
            magic.copy_from_slice(&buf[..4]);
        }
        sink(&buf[..take]);
        at += take as u64;
    }

    let after = fstat(file.as_fd()).map_err(|source| CliError::Io {
        context: format!("cannot re-stat {label}"),
        source,
    })?;
    let dir_now = lstat_at(dir_fd.as_fd(), &c_name).map_err(|source| CliError::Io {
        context: format!("cannot verify directory entry for {label}"),
        source,
    })?;
    if after != before || dir_now != before {
        return Err(CliError::Validation(format!(
            "{label} changed during verification"
        )));
    }
    Ok((before, magic))
}

fn macho_dylib_paths(bytes: &[u8]) -> Result<Vec<String>, CliError> {
    let malformed = || CliError::Validation("Malformed arm64 Mach-O load commands".to_owned());
    let word = |at: usize| -> Result<u32, CliError> {
        let end = at.checked_add(4).ok_or_else(malformed)?;
        let raw = <[u8; 4]>::try_from(bytes.get(at..end).ok_or_else(malformed)?)
            .map_err(|_| malformed())?;
        Ok(u32::from_le_bytes(raw))
    };
    if bytes.get(..4) != Some(MACHO_MAGIC_64_LE.as_slice())
        || word(4)? != 0x0100_000c
        || word(12)? != 2
    {
        return Err(malformed());
    }
    let commands = word(16)? as usize;
    let limit = 32usize
        .checked_add(word(20)? as usize)
        .filter(|n| *n <= bytes.len())
        .ok_or_else(malformed)?;
    if commands == 0 || commands > 65536 {
        return Err(malformed());
    }
    let mut at = 32usize;
    let mut dependencies = Vec::new();
    for _ in 0..commands {
        let command = word(at)?;
        let size = word(at.checked_add(4).ok_or_else(malformed)?)? as usize;
        let end = at
            .checked_add(size)
            .filter(|end| size >= 8 && size.is_multiple_of(8) && *end <= limit)
            .ok_or_else(malformed)?;
        if [0x0c, 0x8000_0018, 0x8000_001f, 0x20, 0x8000_0023].contains(&command) {
            if size < 24 {
                return Err(malformed());
            }
            let offset = word(at + 8)? as usize;
            if offset < 24 || offset >= size {
                return Err(malformed());
            }
            let raw = bytes.get(at + offset..end).ok_or_else(malformed)?;
            let nul = raw.iter().position(|c| *c == 0).ok_or_else(malformed)?;
            let path = std::str::from_utf8(&raw[..nul]).map_err(|_| malformed())?;
            if !(path.starts_with("/System/Library/") || path.starts_with("/usr/lib/"))
                || Path::new(path).components().any(|c| {
                    matches!(
                        c,
                        std::path::Component::ParentDir | std::path::Component::CurDir
                    )
                })
            {
                return Err(CliError::Validation(format!(
                    "Non-system executable dependency: {path}"
                )));
            }
            dependencies.push(path.to_owned());
        }
        if command == 0x27 {
            return Err(CliError::Validation(
                "Mach-O embeds a dynamic-linker environment".to_owned(),
            ));
        }
        at = end;
    }
    if at != limit || dependencies.is_empty() {
        return Err(malformed());
    }
    Ok(dependencies)
}

/// Reads arm64 executable load commands directly, with no developer-tool dependency.
pub fn audit_macho_dependencies(binary: &Path) -> Result<Vec<String>, CliError> {
    const COMMAND_BYTES: usize = 8 * 1024 * 1024 + 32;
    let mut header = Vec::new();
    let (info, _) = inspect_reading(
        binary,
        MAX_ARTIFACT_BYTES,
        "executable",
        true,
        &mut |block| {
            let count = (COMMAND_BYTES - header.len()).min(block.len());
            header.extend_from_slice(&block[..count]);
        },
    )?;
    if (info.mode & 0o111) == 0 {
        return Err(CliError::Validation(
            "Executable lacks execute permission".to_owned(),
        ));
    }
    macho_dylib_paths(&header)
}

/// Verifies `tokspur-owned.metallib` as an `O_NOFOLLOW` single-link regular file with `MTLB` magic, exact size and SHA-256.
pub fn verify_metallib(
    path: &Path,
    expected_sha256: &str,
    expected_size: u64,
    require_mode_0600: bool,
) -> Result<(), CliError> {
    let (st, sha, magic) = inspect_and_hash_nofollow(path, MAX_ARTIFACT_BYTES, "metallib")?;
    if st.size < 4 || magic != MTLB_MAGIC {
        return Err(CliError::Validation(
            "tokspur-owned.metallib lacks valid MTLB header magic".to_owned(),
        ));
    }
    if st.size != expected_size {
        return Err(CliError::Validation(
            "tokspur-owned.metallib byte size differs from manifest".to_owned(),
        ));
    }
    if require_mode_0600 && st.permissions() != 0o600 {
        return Err(CliError::Validation(
            "Staged tokspur-owned.metallib must have mode 0600".to_owned(),
        ));
    }
    if sha != expected_sha256 {
        return Err(CliError::Validation(
            "Built artifact changed or unrecorded: tokspur-owned.metallib".to_owned(),
        ));
    }
    Ok(())
}

/// Value of `native_source_scope` in `build-manifest.json`: `native_sources` pins the build inputs that [`is_native_build_input`] selects and nothing else under `native/`.
///
/// `scripts/bootstrap.py` and `scripts/stage_owned_native_package.py` carry the same rule, and `tests/test_native_build_inputs.py` compares all three. Bump this value in all three places with any change to the rule.
pub const NATIVE_SOURCE_SCOPE: &str = "build-inputs-v2";

const NATIVE_BUILD_INPUT_SUFFIXES: &[&str] = &[
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".m", ".mm", ".metal", ".mk",
    ".json", ".patch", ".LICENSE",
];

const NATIVE_BUILD_INPUT_NAMES: &[&str] = &[
    "Makefile",
    "GNUmakefile",
    "makefile",
    "LICENSE",
    "UNICODE-LICENSE.txt",
    "THIRD_PARTY_NOTICES",
];

/// Whether a file with this name under `native/` belongs to the pinned identity.
///
/// A name is pinned when it is listed or ends with a listed suffix, matched exactly and case-sensitively, so Markdown notes and every other unlisted file stay out of the fingerprint.
pub fn is_native_build_input(name: &str) -> bool {
    NATIVE_BUILD_INPUT_NAMES.contains(&name)
        || NATIVE_BUILD_INPUT_SUFFIXES
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

pub fn collect_native_sources(root: &Path) -> Result<BTreeMap<String, String>, CliError> {
    let mut out = BTreeMap::new();
    collect_tree_digests(
        root,
        &root.join("native"),
        &mut out,
        &[],
        Some(is_native_build_input),
    )?;
    if out.is_empty() {
        return Err(CliError::Validation(
            "Empty native source inventory".to_owned(),
        ));
    }
    Ok(out)
}

pub fn collect_rust_sources(root: &Path) -> Result<BTreeMap<String, String>, CliError> {
    let mut out = BTreeMap::new();
    let top_files = [
        "rust/Cargo.toml",
        "rust/Cargo.lock",
        "rust/rust-toolchain.toml",
        "rust/.cargo/config.toml",
        "rust/clippy.toml",
    ];
    for rel in top_files {
        let full = root.join(rel);
        if full.exists() {
            let (_st, sha, _magic) = inspect_and_hash_nofollow(&full, MAX_SOURCE_BYTES, rel)?;
            out.insert(rel.to_owned(), sha);
        }
    }
    collect_tree_digests(root, &root.join("rust/crates"), &mut out, &["target"], None)?;
    if out.is_empty() {
        return Err(CliError::Validation(
            "Empty Rust source inventory".to_owned(),
        ));
    }
    Ok(out)
}

/// Digests every regular file below `dir`, refusing symlinks and other entries everywhere. `keep`, when given, selects files by name before they are read.
fn collect_tree_digests(
    root: &Path,
    dir: &Path,
    out: &mut BTreeMap<String, String>,
    skip_dirs: &[&str],
    keep: Option<fn(&str) -> bool>,
) -> Result<(), CliError> {
    let dir_fd = open_nofollow_dir(dir, "source directory")?;
    drop(dir_fd);
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|source| CliError::Io {
            context: "cannot read source directory".to_owned(),
            source,
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| CliError::Io {
            context: "cannot read source directory entry".to_owned(),
            source,
        })?
        .into_iter()
        .map(|e| e.path())
        .collect();
    entries.sort();
    for child in entries {
        let name = child
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| CliError::Validation("non-UTF-8 source entry".to_owned()))?;
        if name == ".DS_Store" {
            continue;
        }
        let meta = fs::symlink_metadata(&child).map_err(|source| CliError::Io {
            context: "cannot stat source entry".to_owned(),
            source,
        })?;
        if meta.file_type().is_symlink() {
            return Err(CliError::Validation(
                "Unsafe symlink in source tree".to_owned(),
            ));
        }
        if meta.is_dir() {
            if skip_dirs.contains(&name) {
                continue;
            }
            collect_tree_digests(root, &child, out, skip_dirs, keep)?;
        } else if meta.is_file() {
            if keep.is_some_and(|keep| !keep(name)) {
                continue;
            }
            let rel = child
                .strip_prefix(root)
                .map_err(|_| CliError::Validation("source entry outside root".to_owned()))?
                .to_str()
                .ok_or_else(|| CliError::Validation("non-UTF-8 relative path".to_owned()))?
                .replace('\\', "/");
            let (_st, sha, _magic) = inspect_and_hash_nofollow(&child, MAX_SOURCE_BYTES, &rel)?;
            out.insert(rel, sha);
        } else {
            return Err(CliError::Validation(
                "Non-regular entry in source tree".to_owned(),
            ));
        }
    }
    Ok(())
}

fn json_map_to_btreemap(val: &Json, label: &str) -> Result<BTreeMap<String, String>, CliError> {
    let fields = val.object().map_err(|source| CliError::Json {
        context: "expected JSON object map",
        source,
    })?;
    let mut map = BTreeMap::new();
    for (k, v) in fields {
        let key = std::str::from_utf8(k)
            .map_err(|_| CliError::Validation(format!("non-UTF-8 key in {label}")))?;
        let val_s = std::str::from_utf8(v.string().map_err(|source| CliError::Json {
            context: "expected JSON string digest",
            source,
        })?)
        .map_err(|_| CliError::Validation(format!("non-UTF-8 digest in {label}")))?;
        map.insert(key.to_owned(), val_s.to_owned());
    }
    Ok(map)
}

fn verify_production_carrier_config(receipt: &Json) -> Result<(), CliError> {
    let invalid = |source| CliError::Json {
        context: "invalid native build configuration",
        source,
    };
    let configuration = match receipt.find("native_config").map_err(invalid)? {
        Some(config) => match config.find("NATIVE_CONFIG").map_err(invalid)? {
            Some(value) => Some(value.string().map_err(invalid)?),
            None => None,
        },
        None => None,
    };
    let flags = match receipt.find("tracked_cflags").map_err(invalid)? {
        Some(value) => Some(value.array().map_err(invalid)?),
        None => None,
    };
    if configuration != Some(b"carrier".as_slice())
        || !flags.is_some_and(|values| {
            values
                .first()
                .is_some_and(|v| v.string().is_ok_and(|v| v == b"carrier"))
                && values.iter().all(|v| {
                    v.string().is_ok_and(|v| {
                        !v.windows(b"CARRIER_DIAGNOSTIC".len())
                            .any(|s| s == b"CARRIER_DIAGNOSTIC")
                    })
                })
        })
    {
        return Err(CliError::Validation(
            "production verification requires the carrier build".to_owned(),
        ));
    }
    Ok(())
}

fn verify_checkout_build(
    checkout: &CheckoutLayout,
    receipt_path: &Path,
    targets: &LaunchTargets,
) -> Result<(), CliError> {
    inspect_nofollow(receipt_path, MAX_RECEIPT_BYTES, "build-manifest.json")?;
    let raw = fs::read(receipt_path).map_err(|source| CliError::Io {
        context: "cannot read build-manifest.json".to_owned(),
        source,
    })?;
    let receipt = parse_json_limited(
        &raw,
        JsonLimits::new(MAX_RECEIPT_BYTES as usize, 500_000, 64),
    )
    .map_err(|source| CliError::Json {
        context: "invalid build-manifest.json",
        source,
    })?;

    let schema = receipt
        .find("schema")
        .ok()
        .flatten()
        .and_then(|v| v.unsigned_integer().ok());
    if schema != Some(2) {
        return Err(CliError::Validation(
            "build-manifest.json schema 2 is required; run scripts/bootstrap.py".to_owned(),
        ));
    }
    let native_scope = receipt
        .find("native_source_scope")
        .ok()
        .flatten()
        .and_then(|v| v.string().ok());
    if native_scope != Some(NATIVE_SOURCE_SCOPE.as_bytes()) {
        return Err(CliError::Validation(format!(
            "build-manifest.json native_source_scope \"{NATIVE_SOURCE_SCOPE}\" is required (older manifests also pinned native/ documentation); run scripts/bootstrap.py"
        )));
    }
    if receipt
        .find("enable_metal")
        .ok()
        .flatten()
        .and_then(|v| v.boolean().ok())
        != Some(true)
    {
        return Err(CliError::Validation(
            "build-manifest.json must record an ENABLE_METAL=1 launch build".to_owned(),
        ));
    }
    if receipt
        .find("frozen")
        .ok()
        .flatten()
        .and_then(|v| v.boolean().ok())
        != Some(true)
    {
        return Err(CliError::Validation(
            "build-manifest.json must record frozen Cargo builds".to_owned(),
        ));
    }
    let fn_count = receipt
        .find("metallib_function_count")
        .ok()
        .flatten()
        .and_then(|v| v.unsigned_integer().ok())
        .unwrap_or(0);
    if fn_count == 0 {
        return Err(CliError::Validation(
            "build-manifest.json must record a positive metallib_function_count".to_owned(),
        ));
    }
    for str_field in [
        "git_head",
        "target_triple",
        "profile",
        "rustflags",
        "clang_version",
        "metal_version",
        "rustc_verbose",
        "cargo_version",
        "rust_toolchain_sha256",
        "cargo_lock_sha256",
        "cargo_config_sha256",
    ] {
        let ok = receipt
            .find(str_field)
            .ok()
            .flatten()
            .and_then(|v| v.string().ok())
            .is_some_and(|s| str_field == "rustflags" || !s.is_empty());
        if !ok {
            return Err(CliError::Validation(format!(
                "build-manifest.json missing or empty required field: {str_field}"
            )));
        }
    }
    if receipt
        .find("git_dirty")
        .ok()
        .flatten()
        .and_then(|v| v.boolean().ok())
        .is_none()
    {
        return Err(CliError::Validation(
            "build-manifest.json missing git_dirty boolean".to_owned(),
        ));
    }
    if receipt
        .find("native_config")
        .ok()
        .flatten()
        .and_then(|v| v.object().ok())
        .is_none()
        || receipt.find("tracked_cflags").ok().flatten().is_none()
        || receipt.find("tracked_metalflags").ok().flatten().is_none()
    {
        return Err(CliError::Validation(
            "build-manifest.json missing native/Metal build configuration".to_owned(),
        ));
    }
    verify_production_carrier_config(&receipt)?;
    let rust_targets = receipt.find("rust_targets").ok().flatten().ok_or_else(|| {
        CliError::Validation("build-manifest.json missing rust_targets".to_owned())
    })?;
    let server_feats: Vec<&[u8]> = rust_targets
        .find("tokspur-server")
        .ok()
        .flatten()
        .and_then(|v| v.array().ok())
        .map(|arr| arr.iter().filter_map(|x| x.string().ok()).collect())
        .unwrap_or_default();
    let cli_feats_ok = rust_targets
        .find("tokspur")
        .ok()
        .flatten()
        .and_then(|v| v.array().ok())
        .is_some_and(Vec::is_empty);
    if server_feats != vec![b"fast-math".as_slice()] || !cli_feats_ok {
        return Err(CliError::Validation(
            "build-manifest.json rust_targets must record tokspur-server=[\"fast-math\"] and tokspur=[]"
                .to_owned(),
        ));
    }

    let artifacts =
        receipt.find("artifacts").ok().flatten().ok_or_else(|| {
            CliError::Validation("build-manifest.json missing artifacts".to_owned())
        })?;

    let check_artifact = |path: &Path, is_metallib: bool| -> Result<(), CliError> {
        let rel = path
            .strip_prefix(&checkout.root)
            .map_err(|_| {
                CliError::Validation("Launch target lies outside repository root".to_owned())
            })?
            .to_str()
            .ok_or_else(|| CliError::Validation("non-UTF-8 launch target path".to_owned()))?
            .replace('\\', "/");
        let Some(entry) = artifacts.find(&rel).ok().flatten() else {
            let short = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(rel.as_str());
            return Err(CliError::Validation(format!(
                "Built artifact changed or unrecorded: {short}"
            )));
        };
        let expected_sha = std::str::from_utf8(
            entry
                .find("sha256")
                .ok()
                .flatten()
                .ok_or_else(|| CliError::Validation("artifact entry missing sha256".to_owned()))?
                .string()
                .map_err(|source| CliError::Json {
                    context: "artifact sha256 must be string",
                    source,
                })?,
        )
        .map_err(|_| CliError::Validation("non-UTF-8 artifact sha256".to_owned()))?;
        let expected_size = entry
            .find("size")
            .ok()
            .flatten()
            .ok_or_else(|| CliError::Validation("artifact entry missing size".to_owned()))?
            .unsigned_integer()
            .map_err(|source| CliError::Json {
                context: "artifact size must be integer",
                source,
            })?;

        let mode_str = std::str::from_utf8(
            entry
                .find("mode")
                .ok()
                .flatten()
                .ok_or_else(|| CliError::Validation("artifact entry missing mode".to_owned()))?
                .string()
                .map_err(|source| CliError::Json {
                    context: "artifact mode must be string",
                    source,
                })?,
        )
        .map_err(|_| CliError::Validation("non-UTF-8 artifact mode".to_owned()))?;
        let expected_mode = u16::from_str_radix(mode_str.strip_prefix("0o").unwrap_or(mode_str), 8)
            .map_err(|_| {
                CliError::Validation("invalid octal mode in build-manifest.json".to_owned())
            })?;

        if is_metallib {
            verify_metallib(path, expected_sha, expected_size, false)?;
            let (st, _magic) = inspect_nofollow(path, MAX_ARTIFACT_BYTES, &rel)?;
            if st.permissions() != expected_mode || (st.mode & 0o111) != 0 {
                return Err(CliError::Validation(
                    "tokspur-owned.metallib mode differs from manifest or has execute bit"
                        .to_owned(),
                ));
            }
            if receipt
                .find("system_dependencies")
                .ok()
                .flatten()
                .and_then(|m| m.find(&rel).ok().flatten())
                .is_some()
            {
                return Err(CliError::Validation(
                    "tokspur-owned.metallib must not appear in Mach-O system_dependencies"
                        .to_owned(),
                ));
            }
        } else {
            let (st, sha, _magic) = inspect_and_hash_nofollow(path, MAX_ARTIFACT_BYTES, &rel)?;
            if st.size != expected_size || sha != expected_sha || st.permissions() != expected_mode
            {
                let short = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(rel.as_str());
                return Err(CliError::Validation(format!(
                    "Built artifact changed or unrecorded: {short}"
                )));
            }
            let deps = audit_macho_dependencies(path)?;
            let sys_deps = receipt
                .find("system_dependencies")
                .ok()
                .flatten()
                .and_then(|m| m.find(&rel).ok().flatten())
                .and_then(|a| a.array().ok())
                .ok_or_else(|| {
                    CliError::Validation(format!(
                        "build-manifest.json missing Mach-O system_dependencies for {rel}"
                    ))
                })?;
            let recorded_deps: Vec<String> = sys_deps
                .iter()
                .filter_map(|v| v.string().ok())
                .filter_map(|b| std::str::from_utf8(b).ok())
                .map(str::to_owned)
                .collect();
            if deps != recorded_deps {
                return Err(CliError::Validation(format!(
                    "System library dependencies changed for {rel}"
                )));
            }
        }
        Ok(())
    };

    check_artifact(&targets.server_bin, false)?;
    check_artifact(&targets.worker_bin, false)?;
    check_artifact(&targets.metallib, true)?;
    let cli_rel = "build/rust-fast-math/release/tokspur";
    if artifacts.find(cli_rel).ok().flatten().is_some() {
        check_artifact(&checkout.root.join(cli_rel), false)?;
    }

    // Verify profile digests
    let recorded_profiles = json_map_to_btreemap(
        receipt
            .find("profile_digests")
            .ok()
            .flatten()
            .ok_or_else(|| {
                CliError::Validation("build-manifest.json missing profile_digests".to_owned())
            })?,
        "profile_digests",
    )?;
    for (rel, expected_sha) in &recorded_profiles {
        let (_st, actual_sha, _m) =
            inspect_and_hash_nofollow(&checkout.root.join(rel), MAX_SOURCE_BYTES, rel)?;
        if &actual_sha != expected_sha {
            return Err(CliError::Validation(format!("Profile drift: {rel}")));
        }
    }
    let policy_rel = targets
        .policy
        .strip_prefix(&checkout.root)
        .map_err(|_| CliError::Validation("Policy path lies outside repository root".to_owned()))?
        .to_str()
        .ok_or_else(|| CliError::Validation("non-UTF-8 policy path".to_owned()))?
        .replace('\\', "/");
    if !recorded_profiles.contains_key(&policy_rel) {
        return Err(CliError::Validation(format!(
            "Selected policy is unrecorded in build-manifest.json: {policy_rel}"
        )));
    }

    let recorded_native = json_map_to_btreemap(
        receipt
            .find("native_sources")
            .ok()
            .flatten()
            .ok_or_else(|| {
                CliError::Validation("build-manifest.json missing native_sources".to_owned())
            })?,
        "native_sources",
    )?;
    if collect_native_sources(&checkout.root)? != recorded_native {
        return Err(CliError::Validation(
            "Source drift in native/; re-run scripts/bootstrap.py".to_owned(),
        ));
    }

    let recorded_rust = json_map_to_btreemap(
        receipt.find("rust_sources").ok().flatten().ok_or_else(|| {
            CliError::Validation("build-manifest.json missing rust_sources".to_owned())
        })?,
        "rust_sources",
    )?;
    if collect_rust_sources(&checkout.root)? != recorded_rust {
        return Err(CliError::Validation(
            "Source drift in rust/; re-run scripts/bootstrap.py".to_owned(),
        ));
    }
    for (field, rel_key) in [
        ("rust_toolchain_sha256", "rust/rust-toolchain.toml"),
        ("cargo_lock_sha256", "rust/Cargo.lock"),
        ("cargo_config_sha256", "rust/.cargo/config.toml"),
    ] {
        let pin_sha = receipt
            .find(field)
            .ok()
            .flatten()
            .and_then(|v| v.string().ok())
            .and_then(|b| std::str::from_utf8(b).ok())
            .unwrap_or("");
        if recorded_rust.get(rel_key).map(String::as_str) != Some(pin_sha) {
            return Err(CliError::Validation(format!(
                "build-manifest.json {field} mismatches {rel_key}"
            )));
        }
    }

    let expected_clang = receipt
        .find("clang_version")
        .ok()
        .flatten()
        .and_then(|v| v.string().ok())
        .and_then(|b| std::str::from_utf8(b).ok())
        .unwrap_or("");
    let out = Command::new("/usr/bin/xcrun")
        .args(["clang++", "--version"])
        .output()
        .map_err(|source| CliError::Io {
            context: "cannot run xcrun clang++ --version".to_owned(),
            source,
        })?;
    let actual = String::from_utf8_lossy(&out.stdout);
    if actual.trim() != expected_clang.trim() {
        return Err(CliError::Validation(
            "Compiler differs from the build receipt; re-run bootstrap".to_owned(),
        ));
    }

    Ok(())
}

fn verify_staged_package(staged: &StagedLayout, targets: &LaunchTargets) -> Result<(), CliError> {
    ensure_no_symlink_components(&staged.package_dir, "staged package directory")?;
    let pkg_fd = open_nofollow_dir(&staged.package_dir, "staged package directory")?;
    let pkg_stat = fstat(pkg_fd.as_fd()).map_err(|source| CliError::Io {
        context: "cannot stat staged package directory".to_owned(),
        source,
    })?;
    if pkg_stat.uid != effective_uid() || pkg_stat.permissions() != 0o700 {
        return Err(CliError::Validation(
            "Staged package directory must be owned by current user with mode 0700".to_owned(),
        ));
    }

    let prov_path = staged.package_dir.join("provenance.json");
    let (prov_stat, _prov_magic) =
        inspect_nofollow(&prov_path, MAX_RECEIPT_BYTES, "provenance.json")?;
    if prov_stat.uid != effective_uid() || prov_stat.permissions() != 0o600 {
        return Err(CliError::Validation(
            "Staged provenance.json must be owned by current user with mode 0600".to_owned(),
        ));
    }

    let raw = fs::read(&prov_path).map_err(|source| CliError::Io {
        context: "cannot read provenance.json".to_owned(),
        source,
    })?;
    let prov = parse_json_limited(
        &raw,
        JsonLimits::new(MAX_RECEIPT_BYTES as usize, 500_000, 64),
    )
    .map_err(|source| CliError::Json {
        context: "invalid provenance.json",
        source,
    })?;
    if prov
        .find("schema")
        .ok()
        .flatten()
        .and_then(|v| v.unsigned_integer().ok())
        != Some(2)
        || prov
            .find("control_plane")
            .ok()
            .flatten()
            .and_then(|v| v.string().ok())
            != Some(b"rust".as_slice())
    {
        return Err(CliError::Validation(
            "Staged package requires provenance.json schema 2 with control_plane=\"rust\""
                .to_owned(),
        ));
    }

    let entries_obj = prov
        .find("entries")
        .ok()
        .flatten()
        .ok_or_else(|| CliError::Validation("provenance.json missing entries".to_owned()))?
        .object()
        .map_err(|source| CliError::Json {
            context: "provenance entries must be object",
            source,
        })?;

    let mut declared_names = BTreeSet::new();
    for (name_bytes, entry) in entries_obj {
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| CliError::Validation("non-UTF-8 staged entry name".to_owned()))?;
        if name.contains('/') || name == "." || name == ".." {
            return Err(CliError::Validation("unsafe staged entry name".to_owned()));
        }
        declared_names.insert(name.to_owned());

        let expected_sha = std::str::from_utf8(
            entry
                .find("sha256")
                .ok()
                .flatten()
                .ok_or_else(|| CliError::Validation("staged entry missing sha256".to_owned()))?
                .string()
                .map_err(|source| CliError::Json {
                    context: "staged entry sha256 must be string",
                    source,
                })?,
        )
        .map_err(|_| CliError::Validation("non-UTF-8 staged sha256".to_owned()))?;
        let expected_size = entry
            .find("size")
            .ok()
            .flatten()
            .ok_or_else(|| CliError::Validation("staged entry missing size".to_owned()))?
            .unsigned_integer()
            .map_err(|source| CliError::Json {
                context: "staged entry size must be integer",
                source,
            })?;
        let mode_str = std::str::from_utf8(
            entry
                .find("mode")
                .ok()
                .flatten()
                .ok_or_else(|| CliError::Validation("staged entry missing mode".to_owned()))?
                .string()
                .map_err(|source| CliError::Json {
                    context: "staged entry mode must be string",
                    source,
                })?,
        )
        .map_err(|_| CliError::Validation("non-UTF-8 staged mode".to_owned()))?;
        let trimmed_mode = mode_str.strip_prefix("0o").unwrap_or(mode_str);
        let expected_mode = u16::from_str_radix(trimmed_mode, 8).map_err(|_| {
            CliError::Validation("invalid octal mode in provenance.json".to_owned())
        })?;

        let entry_path = staged.package_dir.join(name);
        if name == "tokspur-owned.metallib" {
            verify_metallib(&entry_path, expected_sha, expected_size, true)?;
            if prov
                .find("system_dependencies")
                .ok()
                .flatten()
                .and_then(|m| m.find(name).ok().flatten())
                .is_some()
            {
                return Err(CliError::Validation(
                    "Staged tokspur-owned.metallib must not appear in Mach-O system_dependencies"
                        .to_owned(),
                ));
            }
        } else {
            let (st, sha, _m) = inspect_and_hash_nofollow(&entry_path, MAX_ARTIFACT_BYTES, name)?;
            if st.uid != effective_uid()
                || st.permissions() != expected_mode
                || st.size != expected_size
                || sha != expected_sha
            {
                return Err(CliError::Validation(format!(
                    "Staged entry ownership, mode, size or digest differs: {name}"
                )));
            }
            if expected_mode == 0o700 {
                let deps = audit_macho_dependencies(&entry_path)?;
                let sys_deps = prov
                    .find("system_dependencies")
                    .ok()
                    .flatten()
                    .and_then(|m| m.find(name).ok().flatten())
                    .and_then(|a| a.array().ok())
                    .ok_or_else(|| {
                        CliError::Validation(format!(
                            "Staged provenance.json missing Mach-O system_dependencies for {name}"
                        ))
                    })?;
                let recorded_deps: Vec<String> = sys_deps
                    .iter()
                    .filter_map(|v| v.string().ok())
                    .filter_map(|b| std::str::from_utf8(b).ok())
                    .map(str::to_owned)
                    .collect();
                if deps != recorded_deps {
                    return Err(CliError::Validation(format!(
                        "Staged system library dependencies changed for {name}"
                    )));
                }
            }
        }
    }

    // Refuse any undeclared file in package_dir
    let mut actual_names = BTreeSet::new();
    for dir_entry in fs::read_dir(&staged.package_dir).map_err(|source| CliError::Io {
        context: "cannot list staged package directory".to_owned(),
        source,
    })? {
        let dir_entry = dir_entry.map_err(|source| CliError::Io {
            context: "cannot read staged directory entry".to_owned(),
            source,
        })?;
        let name = dir_entry
            .file_name()
            .into_string()
            .map_err(|_| CliError::Validation("non-UTF-8 file in staged package".to_owned()))?;
        actual_names.insert(name);
    }
    let mut expected_all = declared_names.clone();
    expected_all.insert("provenance.json".to_owned());
    if actual_names != expected_all {
        return Err(CliError::Validation(
            "Staged package contains undeclared or missing files".to_owned(),
        ));
    }

    for required in [
        "tokspur",
        "tokspur-server",
        "tokspur-owned.metallib",
        "owned-target-experimental.json",
        "m5-max-128.json",
        "flash-next-assets.json",
    ] {
        if !declared_names.contains(required) {
            return Err(CliError::Validation(format!(
                "Staged package missing required entry: {required}"
            )));
        }
    }

    for target_path in [
        &targets.server_bin,
        &targets.worker_bin,
        &targets.metallib,
        &targets.policy,
    ] {
        let Some(name) = target_path.file_name().and_then(|s| s.to_str()) else {
            return Err(CliError::Validation(
                "invalid launch target filename".to_owned(),
            ));
        };
        if !declared_names.contains(name) || target_path.parent() != Some(&staged.package_dir) {
            return Err(CliError::Validation(format!(
                "Selected launch target is not declared in staged provenance.json: {name}"
            )));
        }
    }

    Ok(())
}

/// Runs `verify-build` against the active `RuntimeLayout` and exact launch targets.
pub fn verify_build(layout: &RuntimeLayout) -> Result<LaunchTargets, CliError> {
    let targets = layout.resolve_launch_targets()?;
    verify_targets(layout, &targets)?;
    Ok(targets)
}

/// Pack construction uses the utility executable even when serving uses the
/// scalar worker. Verify that exact executable and the adoption profile.
pub(crate) fn verify_pack_tool(layout: &RuntimeLayout) -> Result<PathBuf, CliError> {
    let mut targets = layout.resolve_launch_targets()?;
    targets.worker_bin.set_file_name("tokspur-native");
    targets.policy = layout.assets_profile_path();
    verify_targets(layout, &targets)?;
    Ok(targets.worker_bin)
}

fn verify_targets(layout: &RuntimeLayout, targets: &LaunchTargets) -> Result<(), CliError> {
    match layout {
        RuntimeLayout::Checkout(c) => {
            let receipt_path = layout
                .checkout_receipt_path()
                .ok_or_else(|| CliError::Validation("missing build-manifest path".to_owned()))?;
            verify_checkout_build(c, &receipt_path, targets)
        }
        RuntimeLayout::Staged(s) => verify_staged_package(s, targets),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_configuration_and_probe_artifacts_are_refused() {
        let valid = tokspur_json::parse_json(br#"{"native_config":{"NATIVE_CONFIG":"carrier"},"tracked_cflags":["carrier",":","flags"]}"#).unwrap();
        assert!(verify_production_carrier_config(&valid).is_ok());
        let diagnostic = tokspur_json::parse_json(br#"{"native_config":{"NATIVE_CONFIG":"carrier-diagnostic"},"tracked_cflags":["carrier-diagnostic",":","flags"]}"#).unwrap();
        assert!(verify_production_carrier_config(&diagnostic).is_err());
        let mut nonce = [0u8; 8];
        tokspur_sys::random_bytes(&mut nonce);
        let dir = fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("tokspur-marker-{}", hex(&nonce)));
        fs::create_dir(&dir).unwrap();
        for (name, bytes) in [
            ("tokspur-owned.metallib", b"MTLBprobe_unchecked".as_slice()),
            (
                "tokspur-owned.metallib",
                b"MTLBgraph_capture_routes".as_slice(),
            ),
            (
                "scalar-channel-target",
                b"\xcf\xfa\xed\xfetokspur-carrier-diagnostic-v1".as_slice(),
            ),
        ] {
            let file = dir.join(name);
            fs::write(&file, bytes).unwrap();
            assert!(inspect_and_hash_nofollow(&file, MAX_ARTIFACT_BYTES, "test artefact").is_err());
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn inspection_without_the_digest_reports_and_refuses_as_the_hashing_one() {
        let mut nonce = [0u8; 8];
        tokspur_sys::random_bytes(&mut nonce);
        let dir = fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("tokspur-cli-inspect-{}", hex(&nonce)));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("tokspur-owned.metallib");
        let mut bytes = MTLB_MAGIC.to_vec();
        bytes.resize(3 << 20, 7);
        fs::write(&file, &bytes).unwrap();

        let (hashed, digest, hashed_magic) =
            inspect_and_hash_nofollow(&file, MAX_ARTIFACT_BYTES, "metallib").unwrap();
        let (stat, magic) = inspect_nofollow(&file, MAX_ARTIFACT_BYTES, "metallib").unwrap();
        assert_eq!(stat, hashed);
        assert_eq!((magic, hashed_magic), (MTLB_MAGIC, MTLB_MAGIC));
        assert_eq!(digest, hex(&tokspur_text::sha256::sha256(&bytes)));

        // The size bound and a second link are refused without the digest as well.
        assert!(inspect_nofollow(&file, 1 << 20, "metallib").is_err());
        fs::hard_link(&file, dir.join("second-link")).unwrap();
        assert!(inspect_nofollow(&file, MAX_ARTIFACT_BYTES, "metallib").is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod macho_tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn executable(path: &[u8]) -> Vec<u8> {
        let size = (24 + path.len() + 1).next_multiple_of(8);
        let mut bytes = vec![0; 32 + size];
        for (at, value) in [
            (0, 0xfeed_facfu32),
            (4, 0x0100_000c),
            (12, 2),
            (16, 1),
            (20, size as u32),
            (32, 0xc),
            (36, size as u32),
            (40, 24),
        ] {
            bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes[56..56 + path.len()].copy_from_slice(path);
        bytes
    }

    #[test]
    fn reads_system_links_without_otool_and_refuses_injection() {
        assert_eq!(
            macho_dylib_paths(&executable(b"/usr/lib/libSystem.B.dylib")).unwrap(),
            ["/usr/lib/libSystem.B.dylib"]
        );
        for path in [
            b"@rpath/foreign.dylib".as_slice(),
            b"/usr/lib/../../tmp/foreign.dylib",
            b"/opt/homebrew/lib/foreign.dylib",
        ] {
            assert!(macho_dylib_paths(&executable(path)).is_err());
        }
        let valid = executable(b"/usr/lib/libSystem.B.dylib");
        for end in 0..valid.len() {
            assert!(macho_dylib_paths(&valid[..end]).is_err());
        }
        let mut injected = valid.clone();
        injected[32..36].copy_from_slice(&0x27u32.to_le_bytes());
        assert!(macho_dylib_paths(&injected).is_err());
        let mut huge = valid;
        huge[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(macho_dylib_paths(&huge).is_err());
    }
}
