# SPDX-License-Identifier: Apache-2.0
"""Unit test suite for Milestone 4 measured pack planner (scripts/plan_pack_v2.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plan_pack_v2 import (  # noqa: E402
    DEFAULT_RESERVE_BYTES,
    EXISTING_LANES_PATH,
    MAX_STREAMING_BUFFER_BYTES,
    PAGE_ALIGNMENT,
    build_gpu_pack_census,
    is_gpu_pack_tensor,
    plan_pack,
    verify_budget,
)


def test_pack_v2_census_inventory_and_counts():
    """Verify pack census accounts for exactly 1,254 GPU tensors and excludes CPU embeddings."""
    census = build_gpu_pack_census()
    assert len(census) == 1254, f"Expected 1254 tensors, got {len(census)}"

    names = {t.name for t in census}

    # Verify exclusions: CPU embedding tables must NOT be in the pack
    assert "token_embd.weight" not in names
    assert "per_layer_token_embd.weight" not in names
    assert not is_gpu_pack_tensor("token_embd.weight")
    assert not is_gpu_pack_tensor("per_layer_token_embd.weight")

    # Verify 48 layers accounted for
    for layer in range(48):
        assert f"blk.{layer}.ffn_gate_inp.weight" in names
        assert f"blk.{layer}.hc_attn_down.weight" in names
        if layer % 4 == 3:
            assert f"blk.{layer}.attn_q.weight" in names
            assert f"blk.{layer}.indexer.q_proj.weight" in names
        else:
            assert f"blk.{layer}.attn_qkv.weight" in names
            assert f"blk.{layer}.ssm_a" in names

    # Verify layer-1 PLE projections
    assert "blk.1.ple_key.weight" in names
    assert "blk.1.ple_value.weight" in names
    assert "blk.1.ple_norm_key.weight" in names
    assert "blk.1.ple_norm_query.weight" in names
    assert "blk.1.ple_norm_conv.weight" in names
    assert "blk.1.ple_conv1d.weight" in names

    # Verify global output HC weights (3)
    assert "output_hc_norm.weight" in names
    assert "output_hc_down.weight" in names
    assert "output_hc_up.weight" in names

    # Verify global output projection (1)
    assert "output.weight" in names

    # Verify native MTP blk.48 sidecar block (32)
    mtp_tensors = [name for name in names if name.startswith("blk.48.")]
    assert len(mtp_tensors) == 32
    assert "blk.48.nextn.eh_proj.weight" in names
    assert "blk.48.attn_q.weight" in names


def test_pack_v2_16kib_page_alignment():
    """Verify all tensor offsets adhere strictly to 16 KiB (16,384 bytes) page alignment."""
    plan = plan_pack()
    assert plan["alignment"] == 16384
    for tensor in plan["tensors"]:
        offset = tensor["offset"]
        assert offset % PAGE_ALIGNMENT == 0, (
            f"Tensor {tensor['name']} offset {offset} is not 16 KiB aligned"
        )


def test_pack_v2_budget_passes_with_adequate_headroom():
    """Verify budget passes when available disk space satisfies the 50 GB reserve rule."""
    peak = 85 * 1024 * 1024 * 1024
    reserve = DEFAULT_RESERVE_BYTES
    available = peak + reserve + 10 * 1024 * 1024 * 1024  # 10 GiB surplus
    passed, msg = verify_budget(available, peak, reserve)
    assert passed is True
    assert "passed" in msg


def test_pack_v2_budget_fails_one_byte_below_reserve():
    """Verify construction is refused when available space is 1 byte below required reserve."""
    peak = 85 * 1024 * 1024 * 1024
    reserve = 50 * 1024**3
    required = peak + reserve
    available = required - 1
    passed, msg = verify_budget(available, peak, reserve)
    assert passed is False
    assert "below required" in msg


def test_pack_v2_budget_refuses_zero_or_negative_space():
    """Verify zero or negative available space triggers immediate refusal."""
    passed, msg = verify_budget(0, 1000)
    assert passed is False
    assert "zero or negative" in msg

    passed, msg = verify_budget(-100, 1000)
    assert passed is False
    assert "zero or negative" in msg


def test_pack_v2_budget_refuses_peak_exceeding_capacity():
    """Verify peak allocation exceeding total disk capacity triggers immediate refusal."""
    capacity = 500 * 1024**3
    peak = 600 * 1024**3
    passed, msg = verify_budget(
        available_space=400 * 1024**3, peak_additional=peak, total_capacity=capacity
    )
    assert passed is False
    assert "exceeds total disk capacity" in msg


def test_pack_v2_preserves_existing_lanes_sidecar(monkeypatch, tmp_path):
    """Verify existing lane sidecar flash-next-lanes-v1.gguf remains untouched and preserved."""
    import scripts.plan_pack_v2 as planner

    if EXISTING_LANES_PATH.exists():
        before = EXISTING_LANES_PATH.stat()
        assert before.st_size == 31_412_379_904, (
            f"Unexpected lane sidecar size: {before.st_size}"
        )
        plan = plan_pack()
        after = EXISTING_LANES_PATH.stat()
        assert plan["existing_lanes_preserved"] is True
        assert (after.st_size, after.st_mtime_ns) == (
            before.st_size,
            before.st_mtime_ns,
        )

    fake_lanes = tmp_path / "flash-next-lanes-v1.gguf"
    with fake_lanes.open("wb") as handle:
        handle.truncate(31_412_379_904)
    before_fake = fake_lanes.stat()
    monkeypatch.setattr(planner, "EXISTING_LANES_PATH", fake_lanes)

    present_plan = planner.plan_pack(target_dir=tmp_path / "pack-v2")
    after_fake = fake_lanes.stat()
    assert present_plan["existing_lanes_preserved"] is True
    assert (after_fake.st_size, after_fake.st_mtime_ns) == (
        before_fake.st_size,
        before_fake.st_mtime_ns,
    )

    fake_lanes.unlink()
    absent_plan = planner.plan_pack(target_dir=tmp_path / "pack-v2")
    assert absent_plan["existing_lanes_preserved"] is False


def test_pack_v2_streaming_memory_bound():
    """Verify bounded streaming memory does not exceed RAM threshold (512 MiB)."""
    assert MAX_STREAMING_BUFFER_BYTES <= 512 * 1024 * 1024


def test_pack_v2_cli_json_and_exit_codes():
    """Verify CLI interface outputs valid JSON and matches actual budget status."""
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plan_pack_v2.py"),
            "--json",
            "--reserve-gb",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    data = json.loads(proc.stdout)
    assert data["schema"] == 2
    assert data["recipe"] == "tokspur-pack-v2"
    assert data["tensor_count"] == 1254
    assert proc.returncode == (0 if data["budget_passed"] else 1)


def test_pack_v2_refuses_existing_pack_credit_64gb_free_vs_138gb_required(
    monkeypatch, tmp_path
):
    """Existing packs stay allocated during construction and must never count as free space."""
    import scripts.plan_pack_v2 as planner

    fake_pack = tmp_path / "tokspur-pack-v2.asa.gguf"
    fake_pack.write_bytes(b"0")
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    # Simulate 64.4 GB free on a 1 TB volume while an 84.6 GB existing pack sits in target_dir.
    monkeypatch.setattr(
        planner,
        "query_filesystem_space",
        lambda _path: (64_400_000_000, 1_000_000_000_000),
    )
    orig_stat = Path.stat

    class FakeStat:
        st_size = 84_631_363_584

    legacy_pack = tmp_path / "tokspur-pack-v2.gguf"

    def patched_stat(self, *args, **kwargs):
        if self in (fake_pack, legacy_pack):
            return FakeStat()
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", patched_stat)
    plan = planner.plan_pack(target_dir=tmp_path)
    assert plan["available_space"] == 64_400_000_000
    assert plan["existing_pack_bytes"] == 84_631_363_584
    assert plan["peak_additional_allocation"] + plan["reserve_bytes"] > 138_000_000_000
    assert plan["budget_passed"] is False
    assert "below required" in plan["budget_message"]

    # Legacy .gguf filename in target_dir is also recognised without crediting free space.
    fake_pack.unlink()
    legacy_pack.write_bytes(b"0")
    legacy_plan = planner.plan_pack(target_dir=tmp_path)
    assert legacy_plan["existing_pack_bytes"] == 84_631_363_584
    assert legacy_plan["budget_passed"] is False
