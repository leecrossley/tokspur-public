//! Planar pack commands use the verified native builder and the same adopted assets as v2.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use tokspur_json::Json;
use tokspur_pack::MTP_SIDECAR_SHA256;

use crate::state::{check_adoption, format_pretty_sorted_json, read_json_file};
use crate::{CliError, PackFormat, RuntimeLayout};

pub(super) fn adopted_pack_sources(
    layout: &RuntimeLayout,
) -> Result<(Vec<PathBuf>, PathBuf), CliError> {
    let tokspur_base = layout.tokspur_base()?;
    let adoption = read_json_file(&tokspur_base.join("adoption.json"))?;
    let assets_profile = read_json_file(&layout.assets_profile_path())?;
    check_adoption(&adoption, &assets_profile)?;

    let shard_paths: Vec<PathBuf> = adoption
        .find("target")
        .ok()
        .flatten()
        .and_then(|t| t.find("assets").ok().flatten())
        .and_then(|a| a.array().ok())
        .ok_or_else(|| CliError::Validation("adoption missing target assets".to_owned()))?
        .iter()
        .filter_map(|a| a.find("path").ok().flatten())
        .filter_map(|v| v.string().ok())
        .filter_map(|b| std::str::from_utf8(b).ok())
        .map(PathBuf::from)
        .collect();

    let draft_assets = adoption
        .find("draft")
        .ok()
        .flatten()
        .and_then(|t| t.find("assets").ok().flatten())
        .and_then(|a| a.array().ok())
        .ok_or_else(|| CliError::Validation("adoption missing draft assets".to_owned()))?;
    let mtp_path = draft_assets
        .iter()
        .find(|a| {
            a.find("sha256")
                .ok()
                .flatten()
                .and_then(|v| v.string().ok())
                == Some(MTP_SIDECAR_SHA256.as_bytes())
        })
        .or_else(|| (draft_assets.len() == 1).then(|| &draft_assets[0]))
        .and_then(|a| a.find("path").ok().flatten())
        .and_then(|v| v.string().ok())
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(PathBuf::from)
        .ok_or_else(|| CliError::Validation("adopted draft missing MTP sidecar".to_owned()))?;

    Ok((shard_paths, mtp_path))
}

const MIB: u64 = 1 << 20;

fn v3_chunk_bytes(chunk_mib: u64) -> Result<u64, CliError> {
    if !(1..=64).contains(&chunk_mib) {
        return Err(CliError::Usage(
            "v3 --chunk-mib must be between 1 and 64".to_owned(),
        ));
    }
    Ok(chunk_mib * MIB)
}

fn reserve_bytes(gib: f64) -> u64 {
    (gib * (1_u64 << 30) as f64) as u64
}

/// The native tool emits one small JSON receipt on success and errors on stderr.
fn run_native(cmd: &mut Command) -> Result<Json, CliError> {
    use std::io::Read;

    let _signals = crate::serve::SignalGuard::install()?;
    if tokspur_sys::stop_requested() {
        return Err(CliError::Validation(
            "pack operation interrupted".to_owned(),
        ));
    }
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|source| CliError::Io {
            context: "cannot start verified pack tool".to_owned(),
            source,
        })?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| CliError::Validation("missing pack receipt pipe".to_owned()))?;
    // Drain stdout concurrently while monitoring the child. SIGTERM sent only to the CLI
    // must still unwind the native temporary file before the build releases its state lock.
    let bytes = std::thread::scope(|scope| -> Result<Vec<u8>, CliError> {
        let reader = scope.spawn(move || {
            let mut bytes = Vec::new();
            stdout.take(65537).read_to_end(&mut bytes)?;
            Ok::<_, std::io::Error>(bytes)
        });
        let mut forwarded = false;
        let status = loop {
            if let Some(status) = child.try_wait().map_err(|source| CliError::Io {
                context: "cannot wait for pack tool".to_owned(),
                source,
            })? {
                break status;
            }
            if tokspur_sys::stop_requested() && !forwarded {
                forwarded = true;
                if let Err(source) =
                    tokspur_sys::process::kill(child.id() as i32, tokspur_sys::SIGTERM)
                    && source.raw_os_error() != Some(tokspur_sys::ESRCH)
                {
                    return Err(CliError::Io {
                        context: "cannot interrupt pack tool".to_owned(),
                        source,
                    });
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        };
        if !status.success() || forwarded {
            return Err(CliError::Validation(format!(
                "native pack operation ended with {status}"
            )));
        }
        reader
            .join()
            .map_err(|_| CliError::Validation("pack receipt reader failed".to_owned()))?
            .map_err(|source| CliError::Io {
                context: "cannot read native pack receipt".to_owned(),
                source,
            })
    })?;
    let receipt =
        tokspur_json::parse_json_limited(&bytes, tokspur_json::JsonLimits::new(65536, 256, 8))
            .map_err(|source| CliError::Json {
                context: "invalid native pack receipt",
                source,
            })?;
    if receipt
        .find("recipe")
        .ok()
        .flatten()
        .and_then(|v| v.string().ok())
        != Some(b"tokspur-pack-v3")
        || receipt
            .find("version")
            .ok()
            .flatten()
            .and_then(|v| v.unsigned_integer().ok())
            != Some(3)
    {
        return Err(CliError::Validation(
            "unexpected native pack recipe or version".to_owned(),
        ));
    }
    Ok(receipt)
}

fn native_command(
    tool: &Path,
    operation: &str,
    sources: &(Vec<PathBuf>, PathBuf),
    output: &Path,
    reserve: u64,
) -> Command {
    let mut cmd = Command::new(tool);
    cmd.env_clear()
        .arg(operation)
        .arg("--output")
        .arg(output)
        .arg("--mtp")
        .arg(&sources.1)
        .arg("--reserve-bytes")
        .arg(reserve.to_string());
    for shard in &sources.0 {
        cmd.arg("--gguf").arg(shard);
    }
    cmd
}

fn number(receipt: &Json, key: &str) -> Result<u64, CliError> {
    receipt
        .find(key)
        .ok()
        .flatten()
        .and_then(|v| v.unsigned_integer().ok())
        .ok_or_else(|| CliError::Validation(format!("invalid {key} in native pack receipt")))
}

fn budget_gate(receipt: &Json, key: &str) -> Result<bool, CliError> {
    receipt
        .find(key)
        .ok()
        .flatten()
        .and_then(|v| v.boolean().ok())
        .ok_or_else(|| CliError::Validation(format!("invalid {key} in native pack plan")))
}

fn write_receipt(value: &Json, out: &mut dyn Write) -> Result<(), CliError> {
    out.write_all(&format_pretty_sorted_json(value)?)
        .map_err(|source| CliError::Io {
            context: "stdout write failed".to_owned(),
            source,
        })
}

pub(super) fn plan_v3(
    layout: &RuntimeLayout,
    target_dir: Option<&Path>,
    reserve_gb: f64,
    chunk_mib: u64,
    json: bool,
    check: bool,
    out: &mut dyn Write,
) -> Result<i32, CliError> {
    let chunk_bytes = v3_chunk_bytes(chunk_mib)?;
    let tool = crate::verify::verify_pack_tool(layout)?;
    let sources = adopted_pack_sources(layout)?;
    let dir = match target_dir {
        Some(dir) => std::path::absolute(dir).map_err(|source| CliError::Io {
            context: "cannot resolve pack plan directory".to_owned(),
            source,
        })?,
        None => layout.tokspur_base()?.join("derived"),
    };
    // The native planner queries an open directory. A future directory inherits
    // the containing volume, so planning never needs to create an output path.
    crate::layout::ensure_no_symlink_components(&dir, "pack plan directory")?;
    let mut volume_dir = dir.as_path();
    while !volume_dir.try_exists().map_err(|source| CliError::Io {
        context: "cannot inspect pack plan directory".to_owned(),
        source,
    })? {
        volume_dir = volume_dir.parent().ok_or_else(|| {
            CliError::Validation("pack plan directory has no existing ancestor".to_owned())
        })?;
    }
    let path = volume_dir.join(PackFormat::V3.filename());
    let reserve = reserve_bytes(reserve_gb);
    let mut cmd = native_command(&tool, "pack-v3-plan", &sources, &path, reserve);
    cmd.arg("--chunk-bytes").arg(chunk_bytes.to_string());
    let plan = run_native(&mut cmd)?;
    let tensors = number(&plan, "tensor_count")?;
    let payload = number(&plan, "payload_bytes")?;
    let bytes = number(&plan, "file_bytes")?;
    let available = number(&plan, "available_bytes")?;
    let buffer = number(&plan, "streaming_buffer_bytes")?;
    let ceiling = number(&plan, "streaming_ceiling_bytes")?;
    if number(&plan, "reserve_bytes")? != reserve {
        return Err(CliError::Validation(
            "native plan returned a different disk reserve".to_owned(),
        ));
    }
    let disk_ok = budget_gate(&plan, "disk_ok")?;
    let ram_ok = budget_gate(&plan, "ram_ok")?;
    if json {
        write_receipt(&plan, out)?;
    } else {
        writeln!(out, "=== Tokspur Pack Planner: tokspur-pack-v3 ===\nTensor count: {tensors}\nPayload bytes: {payload}\nPlanned file bytes: {bytes}\nRequired reserve: {reserve}\nAvailable disk space: {available}\nDisk budget: {}\nStreaming buffer: {buffer} bytes (limit {ceiling})\nRAM budget: {}",
            if disk_ok { "PASS" } else { "FAIL" }, if ram_ok { "PASS" } else { "FAIL" })
            .map_err(|source| CliError::Io { context: "stdout write failed".to_owned(), source })?;
    }
    Ok(i32::from(check && !(disk_ok && ram_ok)))
}

pub(super) fn build_v3(
    layout: &RuntimeLayout,
    output: Option<&Path>,
    reserve_gb: f64,
    chunk_mib: u64,
    out: &mut dyn Write,
) -> Result<i32, CliError> {
    let chunk_bytes = v3_chunk_bytes(chunk_mib)?;
    let tool = crate::verify::verify_pack_tool(layout)?;
    let sources = adopted_pack_sources(layout)?;
    let output = match output {
        Some(path) => std::path::absolute(path).map_err(|source| CliError::Io {
            context: "cannot resolve pack output".to_owned(),
            source,
        })?,
        None => crate::state::private_dir(&layout.tokspur_base()?, Path::new("derived"))?
            .join(PackFormat::V3.filename()),
    };
    let mut cmd = native_command(
        &tool,
        "pack-v3-build",
        &sources,
        &output,
        reserve_bytes(reserve_gb),
    );
    cmd.arg("--chunk-bytes").arg(chunk_bytes.to_string());
    let mut receipt = run_native(&mut cmd)?;
    if !matches!(
        receipt
            .find("status")
            .ok()
            .flatten()
            .and_then(|v| v.string().ok()),
        Some(b"built" | b"verified-existing")
    ) || !receipt
        .find("manifest_sha256")
        .ok()
        .flatten()
        .and_then(|v| v.string().ok())
        .is_some_and(|s| s.len() == 64 && s.iter().all(u8::is_ascii_hexdigit))
    {
        return Err(CliError::Validation(
            "invalid native pack build receipt".to_owned(),
        ));
    }
    for key in ["tensor_count", "payload_bytes", "file_bytes"] {
        number(&receipt, key)?;
    }
    if let Json::Object(fields) = &mut receipt {
        fields.push((
            b"output".to_vec(),
            Json::from(output.to_string_lossy().into_owned()),
        ));
    }
    write_receipt(&receipt, out)?;
    Ok(0)
}
