#!/usr/bin/env python3
"""Measured pack planner for tokspur-pack-v2 lossless derived weight pack.

Enforces the 50 GB disk reserve rule (available_space >= peak_additional_allocation + 50 GB)
throughout streaming construction without touching or deleting existing lane sidecars.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Default format constants
PACK_FORMAT_RECIPE = "tokspur-pack-v2"
PACK_FORMAT_VERSION = 2
PACK_ARCHITECTURE = "tokspur_pack"
SOURCE_ARCHITECTURE = "qwen4exp"
SOURCE_REVISION = "766911a6b7369840a91dbcd95f9f997acaab6cd6"
MTP_SIDECAR_SHA256 = "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6"

PAGE_ALIGNMENT = 16384  # 16 KiB Metal buffer alignment
DEFAULT_RESERVE_BYTES = 50 * 1024 * 1024 * 1024  # 50 GiB safety reserve
DEFAULT_STREAMING_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB streaming chunk
MAX_STREAMING_BUFFER_BYTES = 512 * 1024 * 1024  # 512 MiB streaming memory bound

DEFAULT_PACK_DIR = ROOT / ".tokspur" / "derived" / "pack-v2"
DEFAULT_PACK_NAME = "tokspur-pack-v2.asa.gguf"
LEGACY_PACK_NAME = "tokspur-pack-v2.gguf"
EXISTING_LANES_PATH = (
    ROOT / ".tokspur" / "derived" / "lane-layout-v1" / "flash-next-lanes-v1.gguf"
)


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    tensor_type: int
    dimensions: list[int]
    bytes_count: int
    layout: str = "plain"


def align_offset(offset: int, alignment: int = PAGE_ALIGNMENT) -> int:
    """Align byte offset up to the specified boundary."""
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


def compute_tensor_bytes(tensor_type: int, dimensions: list[int]) -> int:
    """Compute exact byte count for supported tensor quantisation types."""
    elements = 1
    for d in dimensions:
        elements *= d

    # Quantisation encodings
    if tensor_type == 0:  # F32
        return elements * 4
    if tensor_type in (1, 30):  # F16, BF16
        return elements * 2
    if tensor_type == 7:  # Q5_1 (32 elements per 24 bytes)
        return (elements // 32) * 24
    if tensor_type == 8:  # Q8_0 (32 elements per 34 bytes)
        return (elements // 32) * 34
    if tensor_type == 12:  # Q4_K (256 elements per 144 bytes)
        return (elements // 256) * 144
    if tensor_type == 13:  # Q5_K (256 elements per 176 bytes)
        return (elements // 256) * 176
    if tensor_type == 20:  # IQ4_NL (32 elements per 18 bytes)
        return (elements // 32) * 18
    raise ValueError(f"Unsupported tensor type: {tensor_type}")


def build_gpu_pack_census() -> list[TensorDescriptor]:
    """Build the complete 1,254 GPU tensor inventory for tokspur-pack-v2.

    Excludes CPU embedding tables token_embd.weight and per_layer_token_embd.weight,
    which are retained in original GGUF shard mappings.
    """
    census: list[TensorDescriptor] = []

    # 1. 48 Target transformer layers (1,218 tensors)
    for layer in range(48):
        prefix = f"blk.{layer}."

        # Hyper-Connections (8 tensors per layer)
        for pathway in ("hc_attn", "hc_ffn"):
            root = f"{prefix}{pathway}"
            census.append(
                TensorDescriptor(f"{root}_norm.weight", 0, [10240], 10240 * 4)
            )
            census.append(
                TensorDescriptor(
                    f"{root}_down.weight", 8, [10240, 320], (10240 * 320 // 32) * 34
                )
            )
            census.append(
                TensorDescriptor(
                    f"{root}_up.weight", 8, [320, 10240], (320 * 10240 // 32) * 34
                )
            )
            census.append(
                TensorDescriptor(f"{root}_inject.weight", 0, [10240, 4], 10240 * 4 * 4)
            )

        # MoE FFN (8 tensors per layer)
        census.append(
            TensorDescriptor(
                f"{prefix}ffn_gate_inp.weight", 0, [2560, 512], 2560 * 512 * 4
            )
        )
        census.append(
            TensorDescriptor(f"{prefix}ffn_gate_inp_shexp.weight", 0, [2560], 2560 * 4)
        )
        # Expert tensors: UD-Q4_K_XL quantization census
        # Layer 2 uses Q5_K (13) for gate/up; other layers use Q4_K (12)
        if layer == 2:
            gate_up_type = 13
            gate_up_bytes = (2560 * 640 * 512 // 256) * 176
        else:
            gate_up_type = 12
            gate_up_bytes = (2560 * 640 * 512 // 256) * 144

        # Layers 2, 4, 30, 46, 47 use Q8_0 (8) down exps; other layers use Q5_1 (7)
        if layer in (2, 4, 30, 46, 47):
            down_type = 8
            down_bytes = (640 * 2560 * 512 // 32) * 34
            down_layout = "plain"
        else:
            down_type = 7
            down_bytes = (640 * 2560 * 512 // 32) * 24
            down_layout = "q51_lanes_v1"

        census.append(
            TensorDescriptor(
                f"{prefix}ffn_gate_exps.weight",
                gate_up_type,
                [2560, 640, 512],
                gate_up_bytes,
                layout="plain",
            )
        )
        census.append(
            TensorDescriptor(
                f"{prefix}ffn_up_exps.weight",
                gate_up_type,
                [2560, 640, 512],
                gate_up_bytes,
                layout="plain",
            )
        )
        census.append(
            TensorDescriptor(
                f"{prefix}ffn_down_exps.weight",
                down_type,
                [640, 2560, 512],
                down_bytes,
                layout=down_layout,
            )
        )
        census.append(
            TensorDescriptor(
                f"{prefix}ffn_gate_shexp.weight",
                8,
                [2560, 640],
                (2560 * 640 // 32) * 34,
                layout="plain",
            )
        )
        census.append(
            TensorDescriptor(
                f"{prefix}ffn_up_shexp.weight",
                8,
                [2560, 640],
                (2560 * 640 // 32) * 34,
                layout="plain",
            )
        )
        census.append(
            TensorDescriptor(
                f"{prefix}ffn_down_shexp.weight",
                8,
                [640, 2560],
                (640 * 2560 // 32) * 34,
                layout="plain",
            )
        )

        # Recurrence or Attention
        if layer % 4 == 3:
            # 12 QSA layers: 10 tensors per layer
            census.append(
                TensorDescriptor(
                    f"{prefix}attn_q.weight",
                    8,
                    [2560, 12288],
                    (2560 * 12288 // 32) * 34,
                )
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}attn_k.weight", 8, [2560, 512], (2560 * 512 // 32) * 34
                )
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}attn_v.weight", 8, [2560, 512], (2560 * 512 // 32) * 34
                )
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}attn_output.weight",
                    8,
                    [6144, 2560],
                    (6144 * 2560 // 32) * 34,
                )
            )
            census.append(
                TensorDescriptor(f"{prefix}attn_q_norm.weight", 0, [256], 256 * 4)
            )
            census.append(
                TensorDescriptor(f"{prefix}attn_k_norm.weight", 0, [256], 256 * 4)
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}indexer.q_proj.weight", 30, [2560, 512], 2560 * 512 * 2
                )
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}indexer.k_proj.weight", 30, [2560, 128], 2560 * 128 * 2
                )
            )
            census.append(
                TensorDescriptor(f"{prefix}indexer.q_norm.weight", 0, [128], 128 * 4)
            )
            census.append(
                TensorDescriptor(f"{prefix}indexer.k_norm.weight", 0, [128], 128 * 4)
            )
        else:
            # 36 GDN layers: 9 tensors per layer
            census.append(
                TensorDescriptor(
                    f"{prefix}attn_qkv.weight",
                    8,
                    [2560, 10240],
                    (2560 * 10240 // 32) * 34,
                )
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}attn_gate.weight",
                    8,
                    [2560, 6144],
                    (2560 * 6144 // 32) * 34,
                )
            )
            census.append(TensorDescriptor(f"{prefix}ssm_a", 0, [48], 48 * 4))
            census.append(TensorDescriptor(f"{prefix}ssm_dt.bias", 0, [48], 48 * 4))
            census.append(
                TensorDescriptor(
                    f"{prefix}ssm_alpha.weight", 0, [2560, 48], 2560 * 48 * 4
                )
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}ssm_beta.weight", 0, [2560, 48], 2560 * 48 * 4
                )
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}ssm_conv1d.weight", 0, [4, 10240], 4 * 10240 * 4
                )
            )
            census.append(
                TensorDescriptor(f"{prefix}ssm_norm.weight", 0, [128], 128 * 4)
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}ssm_out.weight", 8, [6144, 2560], (6144 * 2560 // 32) * 34
                )
            )

        # Layer 1 PLE projections: 6 tensors
        if layer == 1:
            census.append(
                TensorDescriptor(
                    f"{prefix}ple_key.weight",
                    8,
                    [2560, 10240],
                    (2560 * 10240 // 32) * 34,
                )
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}ple_value.weight",
                    8,
                    [2560, 2560],
                    (2560 * 2560 // 32) * 34,
                )
            )
            census.append(
                TensorDescriptor(f"{prefix}ple_norm_key.weight", 0, [10240], 10240 * 4)
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}ple_norm_query.weight", 0, [10240], 10240 * 4
                )
            )
            census.append(
                TensorDescriptor(f"{prefix}ple_norm_conv.weight", 0, [10240], 10240 * 4)
            )
            census.append(
                TensorDescriptor(
                    f"{prefix}ple_conv1d.weight", 0, [4, 10240], 4 * 10240 * 4
                )
            )

    # 2. Global output Hyper-Connections (3 tensors)
    census.append(TensorDescriptor("output_hc_norm.weight", 0, [10240], 10240 * 4))
    census.append(
        TensorDescriptor(
            "output_hc_down.weight", 8, [10240, 320], (10240 * 320 // 32) * 34
        )
    )
    census.append(
        TensorDescriptor(
            "output_hc_up.weight", 8, [320, 10240], (320 * 10240 // 32) * 34
        )
    )

    # 3. Global vocabulary output projection (1 tensor)
    census.append(
        TensorDescriptor("output.weight", 8, [2560, 248320], (2560 * 248320 // 32) * 34)
    )

    # 4. Native MTP sidecar block (32 tensors in blk.48.*)
    mtp_defs: list[tuple[str, list[int], int]] = [
        ("attn_k.weight", [2560, 512], 8),
        ("attn_k_norm.weight", [256], 0),
        ("attn_output.weight", [6144, 2560], 8),
        ("attn_q.weight", [2560, 12288], 8),
        ("attn_q_norm.weight", [256], 0),
        ("attn_v.weight", [2560, 512], 8),
        ("ffn_down_exps.weight", [640, 2560, 512], 8),
        ("ffn_down_shexp.weight", [640, 2560], 8),
        ("ffn_gate_exps.weight", [2560, 640, 512], 8),
        ("ffn_gate_inp.weight", [2560, 512], 0),
        ("ffn_gate_inp_shexp.weight", [2560], 0),
        ("ffn_gate_shexp.weight", [2560, 640], 8),
        ("ffn_up_exps.weight", [2560, 640, 512], 8),
        ("ffn_up_shexp.weight", [2560, 640], 8),
        ("hc_attn_down.weight", [10240, 320], 8),
        ("hc_attn_inject.weight", [10240, 4], 8),
        ("hc_attn_norm.weight", [10240], 0),
        ("hc_attn_up.weight", [320, 10240], 8),
        ("hc_ffn_down.weight", [10240, 320], 8),
        ("hc_ffn_inject.weight", [10240, 4], 8),
        ("hc_ffn_norm.weight", [10240], 0),
        ("hc_ffn_up.weight", [320, 10240], 8),
        ("indexer.k_norm.weight", [128], 0),
        ("indexer.k_proj.weight", [2560, 128], 30),
        ("indexer.q_norm.weight", [128], 0),
        ("indexer.q_proj.weight", [2560, 512], 30),
        ("nextn.eh_proj.weight", [5120, 2560], 8),
        ("nextn.enorm.weight", [2560], 0),
        ("nextn.hc_head_down.weight", [10240, 320], 8),
        ("nextn.hc_head_norm.weight", [10240], 0),
        ("nextn.hc_head_up.weight", [320, 10240], 8),
        ("nextn.hnorm.weight", [10240], 0),
    ]
    for name, dims, ttype in mtp_defs:
        b = compute_tensor_bytes(ttype, dims)
        census.append(TensorDescriptor(f"blk.48.{name}", ttype, dims, b))

    return census


def is_gpu_pack_tensor(name: str) -> bool:
    """Return True if tensor belongs in the GPU weight pack."""
    if "token_embd.weight" in name:
        return False
    if "per_layer_token_embd" in name:
        return False
    return True


def verify_budget(
    available_space: int,
    peak_additional: int,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    total_capacity: int | None = None,
) -> tuple[bool, str]:
    """Validate disk budget against the 50 GB safety margin rule."""
    if available_space <= 0:
        return False, "zero or negative available space triggers immediate refusal"
    if total_capacity is not None and peak_additional > total_capacity:
        return False, "peak allocation exceeds total disk capacity"
    required = peak_additional + reserve_bytes
    if available_space < required:
        deficit = required - available_space
        return (
            False,
            f"available space ({available_space} bytes) is below required "
            f"peak + reserve ({required} bytes), deficit: {deficit} bytes",
        )
    return True, "budget check passed"


def query_filesystem_space(path: Path) -> tuple[int, int]:
    """Query available bytes and total capacity bytes on filesystem backing path."""
    resolved = path if path.exists() else path.parent
    while not resolved.exists() and resolved != resolved.parent:
        resolved = resolved.parent
    stat = os.statvfs(str(resolved))
    available = stat.f_bavail * stat.f_frsize
    total = stat.f_blocks * stat.f_frsize
    return available, total


def plan_pack(
    target_dir: Path | None = None,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
) -> dict:
    """Compute complete Milestone 4 pack plan and evaluate disk budget."""
    pack_dir = target_dir or DEFAULT_PACK_DIR
    census = build_gpu_pack_census()

    # Calculate aligned layout
    header_estimate = 512 * 1024  # 512 KiB for GGUF header + manifest
    data_offset = align_offset(header_estimate, PAGE_ALIGNMENT)

    current_offset = 0
    tensor_records = []
    total_payload_bytes = 0

    for t in census:
        current_offset = align_offset(current_offset, PAGE_ALIGNMENT)
        tensor_records.append(
            {
                "name": t.name,
                "type": t.tensor_type,
                "dimensions": t.dimensions,
                "bytes": t.bytes_count,
                "offset": current_offset,
                "layout": t.layout,
            }
        )
        current_offset += t.bytes_count
        total_payload_bytes += t.bytes_count

    final_file_bytes = data_offset + current_offset
    # Peak additional allocation equals single temporary file plus staging buffer
    peak_additional = final_file_bytes + MAX_STREAMING_BUFFER_BYTES

    # Check filesystem without crediting any existing pack: existing packs stay allocated throughout construction and RENAME_EXCL refuses in-place replacement.
    available_space, total_capacity = query_filesystem_space(pack_dir)
    existing_pack_bytes = 0
    for candidate_pack in (
        ROOT / ".tokspur" / "derived" / DEFAULT_PACK_NAME,
        ROOT / ".tokspur" / "derived" / LEGACY_PACK_NAME,
        pack_dir / DEFAULT_PACK_NAME,
        pack_dir / LEGACY_PACK_NAME,
    ):
        if candidate_pack.exists():
            existing_pack_bytes = candidate_pack.stat().st_size
            break
    passed, reason = verify_budget(
        available_space, peak_additional, reserve_bytes, total_capacity
    )

    # Check existing lane sidecar
    existing_lanes_preserved = (
        EXISTING_LANES_PATH.exists()
        and EXISTING_LANES_PATH.stat().st_size == 31_412_379_904
    )

    plan = {
        "schema": PACK_FORMAT_VERSION,
        "recipe": PACK_FORMAT_RECIPE,
        "architecture": PACK_ARCHITECTURE,
        "source_architecture": SOURCE_ARCHITECTURE,
        "target_revision": SOURCE_REVISION,
        "mtp_sidecar_sha256": MTP_SIDECAR_SHA256,
        "alignment": PAGE_ALIGNMENT,
        "tensor_count": len(census),
        "payload_bytes": total_payload_bytes,
        "planned_file_bytes": final_file_bytes,
        "peak_additional_allocation": peak_additional,
        "reserve_bytes": reserve_bytes,
        "available_space": available_space,
        "existing_pack_bytes": existing_pack_bytes,
        "total_capacity": total_capacity,
        "budget_passed": passed,
        "budget_message": reason,
        "existing_lanes_preserved": existing_lanes_preserved,
        "tensors": tensor_records,
    }
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan tokspur-pack-v2 derived weight pack and verify disk budget."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_PACK_DIR,
        help="Target directory for derived pack storage.",
    )
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=50.0,
        help="Free disk space safety reserve in GB (default: 50.0).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output plan in JSON format.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Perform preflight budget check and exit 0 if passed, 1 if refused.",
    )
    args = parser.parse_args()

    reserve_bytes = int(args.reserve_gb * 1024 * 1024 * 1024)
    plan = plan_pack(args.target_dir, reserve_bytes)

    if args.json:
        # Output summary json without the full 1254 tensor details unless explicitly requested
        summary = dict(plan)
        summary["tensors"] = len(plan["tensors"])
        print(json.dumps(summary, indent=2))
    else:
        print(f"=== Tokspur Pack Planner: {plan['recipe']} ===")
        print(f"Tensor count:               {plan['tensor_count']}")
        print(
            f"Payload bytes:              {plan['payload_bytes']:,} bytes ({plan['payload_bytes'] / (1024**3):.2f} GiB)"
        )
        print(
            f"Planned file bytes:         {plan['planned_file_bytes']:,} bytes ({plan['planned_file_bytes'] / (1024**3):.2f} GiB)"
        )
        print(
            f"Peak additional allocation: {plan['peak_additional_allocation']:,} bytes ({plan['peak_additional_allocation'] / (1024**3):.2f} GiB)"
        )
        print(
            f"Required reserve:           {plan['reserve_bytes']:,} bytes ({plan['reserve_bytes'] / (1024**3):.2f} GiB)"
        )
        print(
            f"Available disk space:       {plan['available_space']:,} bytes ({plan['available_space'] / (1024**3):.2f} GiB)"
        )
        print(
            f"Budget check status:        {'PASS' if plan['budget_passed'] else 'FAIL'}"
        )
        print(f"Budget message:             {plan['budget_message']}")
        print(
            f"Existing lane sidecar:      {'PRESERVED' if plan['existing_lanes_preserved'] else 'NOT FOUND'}"
        )

    return 0 if plan["budget_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
