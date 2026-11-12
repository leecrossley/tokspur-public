//! Read-only pre-download comparison with the text layout in native/src/model.cpp.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use tokspur_json::{Json, JsonError, JsonLimits, parse_json_limited};

use crate::error::CliError;

// The config uses Transformers text field names; inspect_flash_next and
// flash_next_requirements own the GGUF metadata and tensor checks at load time.
const DIMENSIONS: &[(&str, u64)] = &[
    ("num_hidden_layers", 48),
    ("hidden_size", 2560),
    ("vocab_size", 248320),
    ("num_attention_heads", 24),
    ("num_key_value_heads", 2),
    ("head_dim", 256),
    ("num_experts", 512),
    ("num_experts_per_tok", 10),
    ("moe_intermediate_size", 640),
    ("shared_expert_intermediate_size", 640),
    ("full_attention_interval", 4),
    ("linear_conv_kernel_dim", 4),
    ("linear_num_key_heads", 16),
    ("linear_num_value_heads", 48),
    ("linear_key_head_dim", 128),
    ("linear_value_head_dim", 128),
    ("hc_count", 4),
    ("hc_lowrank", 320),
    ("indexer_n_heads", 4),
    ("indexer_kv_heads", 1),
    ("indexer_head_dim", 128),
    ("indexer_budget", 2048),
    ("indexer_compress_ratio", 4),
    ("ngram_size", 3),
    ("heads_per_ngram", 8),
    ("ple_conv_kernel_size", 4),
    ("ple_embed_dim", 2560),
];

const MAX_CONFIG_BYTES: usize = 4 << 20;

pub(super) fn check_file(path: &Path, out: &mut dyn Write) -> Result<i32, CliError> {
    let read_error = |source| CliError::Io {
        context: "cannot read model config".to_owned(),
        source,
    };
    let metadata = std::fs::metadata(path).map_err(read_error)?;
    if !metadata.is_file() {
        return Err(CliError::Validation(
            "model config must be a regular JSON file".to_owned(),
        ));
    }
    if metadata.len() > MAX_CONFIG_BYTES as u64 {
        return Err(CliError::Validation(
            "model config exceeds 4 MiB".to_owned(),
        ));
    }
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(read_error)?
        .take(MAX_CONFIG_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(read_error)?;
    let config = parse_json_limited(&bytes, JsonLimits::new(MAX_CONFIG_BYTES, 262144, 64))
        .map_err(|source| CliError::Json {
            context: "invalid model config",
            source,
        })?;
    let (status, report) = compare(&config).map_err(|source| CliError::Json {
        context: "invalid text config",
        source,
    })?;
    out.write_all(report.as_bytes())
        .map_err(|source| CliError::Io {
            context: "stdout write failed".to_owned(),
            source,
        })?;
    Ok(status)
}

fn compare(config: &Json) -> Result<(i32, String), JsonError> {
    let text = config.find("text_config")?.unwrap_or(config);
    let mut mismatches = Vec::new();
    let mut unknown = Vec::new();
    match text.find("model_type")? {
        Some(v)
            if v.string()
                .is_ok_and(|name| matches!(name, b"qwen4_exp" | b"qwen4_exp_text")) => {}
        Some(_) => unknown.push("model_type (unrecognized text architecture)"),
        None => unknown.push("model_type"),
    }
    let mut owners = vec![("text".to_owned(), text)];
    if !std::ptr::eq(text, config) {
        owners.push(("root".to_owned(), config));
        // Only compare overlapping text geometry. Wrappers can legitimately
        // have their own model_type, architecture names, and vision geometry.
        for field in DIMENSIONS.iter().map(|(name, _)| *name).chain([
            "max_position_embeddings",
            "ple_layer_ids",
            "layer_types",
        ]) {
            if let (Some(inner), Some(outer)) = (text.find(field)?, config.find(field)?)
                && inner != outer
            {
                mismatches.push(format!("root.{field} conflicts with text_config.{field}"));
            }
        }
    }
    for &(field, expected) in DIMENSIONS {
        if let Some(value) = text.find(field)? {
            match value.unsigned_integer() {
                Ok(actual) if actual == expected => {}
                Ok(actual) => {
                    mismatches.push(format!("{field} expected {expected}, found {actual}"))
                }
                Err(_) => mismatches.push(format!("{field} must be an unsigned integer")),
            }
        } else {
            unknown.push(field);
        }
    }
    match text.find("max_position_embeddings")? {
        Some(value) => match value.unsigned_integer() {
            Ok(65536..) => {}
            Ok(length) => mismatches.push(format!(
                "max_position_embeddings requires at least 65536, found {length}"
            )),
            Err(_) => {
                mismatches.push("max_position_embeddings must be an unsigned integer".to_owned())
            }
        },
        None => unknown.push("max_position_embeddings"),
    }
    // The reference Transformers config identifies PLE layer 2; the GGUF
    // layer indices inspected by the native runtime are zero-based.
    match text.find("ple_layer_ids")? {
        Some(value) => {
            if !value.array().is_ok_and(|layers| matches!(layers.as_slice(), [layer] if layer.unsigned_integer().ok() == Some(2))) {
                mismatches.push("ple_layer_ids expected [2]".to_owned());
            }
        }
        None => unknown.push("ple_layer_ids"),
    }
    match text.find("layer_types")? {
        Some(value) => {
            let matches_schedule = value.array().is_ok_and(|layers| {
                layers.len() == 48
                    && layers.iter().enumerate().all(|(index, layer)| {
                        let expected = if index % 4 == 3 {
                            b"full_attention".as_slice()
                        } else {
                            b"linear_attention".as_slice()
                        };
                        layer.string().ok() == Some(expected)
                    })
            });
            if !matches_schedule {
                mismatches.push("layer_types expected 48 layers with full attention every fourth layer and linear attention elsewhere".to_owned());
            }
        }
        None => unknown.push("layer_types"),
    }

    check_scalar(
        &owners,
        "rms_norm_eps",
        |v| v > 0.0 && v <= 0.01,
        "greater than 0 and at most 0.01",
        &mut mismatches,
        &mut unknown,
    )?;
    check_rope(&owners, &mut mismatches, &mut unknown)?;

    let status = if !mismatches.is_empty() {
        "incompatible with the current Flash-Next text layout"
    } else if !unknown.is_empty() {
        "insufficient evidence for a Flash-Next text layout match"
    } else {
        "checked fields match the current Flash-Next text layout"
    };
    let mut lines = vec![format!("Config preflight: {status}.")];
    let code = i32::from(!mismatches.is_empty() || !unknown.is_empty());
    lines.extend(mismatches);
    if !unknown.is_empty() {
        lines.push(format!("Unverified fields: {}", unknown.join(", ")));
    }
    lines.push(
        "Serving requires the pinned target and draft assets to pass adoption and GGUF validation."
            .to_owned(),
    );
    Ok((code, lines.join("\n") + "\n"))
}

// Native model.cpp consumes RMS epsilon and RoPE base from the GGUF metadata.
fn check_scalar(
    owners: &[(String, &Json)],
    field: &'static str,
    valid: fn(f64) -> bool,
    expected: &str,
    mismatches: &mut Vec<String>,
    unknown: &mut Vec<&'static str>,
) -> Result<(), JsonError> {
    let mut found = false;
    let mut previous = None;
    for (scope, owner) in owners {
        let Some(value) = owner.find(field)? else {
            continue;
        };
        found = true;
        let number = value
            .number_lexeme()
            .ok()
            .and_then(|s| s.parse::<f64>().ok());
        match number {
            Some(v) if v.is_finite() && valid(v) => {
                if previous.is_some_and(|earlier| earlier != v) {
                    mismatches.push(format!(
                        "{scope}.{field} conflicts with the other {field} declaration"
                    ));
                } else {
                    previous = Some(v);
                }
            }
            _ => mismatches.push(format!("{scope}.{field} expected {expected}")),
        }
    }
    if !found {
        unknown.push(field);
    }
    Ok(())
}

// Config writers can repeat RoPE settings at the root, in text_config and
// inside each object's rope_parameters or rope_scaling.
fn check_rope(
    owners: &[(String, &Json)],
    mismatches: &mut Vec<String>,
    unknown: &mut Vec<&'static str>,
) -> Result<(), JsonError> {
    let mut declarations = owners.to_vec();
    for (scope, owner) in owners {
        for key in ["rope_parameters", "rope_scaling"] {
            if let Some(params) = owner.find(key)?
                && !params.is_null()
            {
                if params.object().is_err() {
                    mismatches.push(format!("{scope}.{key} must be an object"));
                } else {
                    declarations.push((format!("{scope}.{key}"), params));
                }
            }
        }
    }
    check_scalar(
        &declarations,
        "partial_rotary_factor",
        |v| v == 0.25,
        "0.25 (64 rotary dimensions at head_dim 256)",
        mismatches,
        unknown,
    )?;
    check_scalar(
        &declarations,
        "rope_theta",
        |v| v > 1.0,
        "a finite number greater than 1",
        mismatches,
        unknown,
    )?;
    let mut found_sections = false;
    for (label, params) in declarations {
        if let Some(value) = params.find("mrope_section")? {
            found_sections = true;
            let valid = value.array().is_ok_and(|items| {
                let sections = items.iter().map(|v| v.unsigned_integer().ok());
                sections.clone().eq([Some(11), Some(11), Some(10)])
                    || sections.eq([Some(11), Some(11), Some(10), Some(0)])
            });
            if !valid {
                mismatches.push(format!(
                    "{label}.mrope_section expected [11, 11, 10] (optional zero padding)"
                ));
            }
        }
    }
    if !found_sections {
        unknown.push("mrope_section");
    }
    Ok(())
}
