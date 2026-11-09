# SPDX-License-Identifier: Apache-2.0
"""Empirical adversarial stress test suite for Milestone 4 pack planner."""

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
    MAX_STREAMING_BUFFER_BYTES,
    PAGE_ALIGNMENT,
    align_offset,
    compute_tensor_bytes,
    plan_pack,
    verify_budget,
)


def test_adversarial_budget_deficit_and_surplus_boundaries():
    """Adversarially probe boundary conditions around exact budget margin."""
    peak = 85 * 1024 * 1024 * 1024
    reserve = DEFAULT_RESERVE_BYTES  # 50 GiB
    required = peak + reserve

    # 1 byte surplus -> must PASS
    passed, msg = verify_budget(
        available_space=required + 1, peak_additional=peak, reserve_bytes=reserve
    )
    assert passed is True
    assert "passed" in msg

    # Exact match (0 byte margin) -> must PASS
    passed, msg = verify_budget(
        available_space=required, peak_additional=peak, reserve_bytes=reserve
    )
    assert passed is True
    assert "passed" in msg

    # 1 byte deficit -> must FAIL with exact deficit reported
    passed, msg = verify_budget(
        available_space=required - 1, peak_additional=peak, reserve_bytes=reserve
    )
    assert passed is False
    assert "deficit: 1 bytes" in msg

    # 2 bytes deficit -> must FAIL with exact deficit reported
    passed, msg = verify_budget(
        available_space=required - 2, peak_additional=peak, reserve_bytes=reserve
    )
    assert passed is False
    assert "deficit: 2 bytes" in msg

    # 1 GiB deficit
    deficit_1gib = 1024 * 1024 * 1024
    passed, msg = verify_budget(
        available_space=required - deficit_1gib,
        peak_additional=peak,
        reserve_bytes=reserve,
    )
    assert passed is False
    assert f"deficit: {deficit_1gib} bytes" in msg


def test_adversarial_budget_negative_and_zero_available_space():
    """Verify immediate refusal on zero, negative, or degenerate available space."""
    # Zero available space
    passed, msg = verify_budget(available_space=0, peak_additional=1000)
    assert passed is False
    assert "zero or negative" in msg

    # Negative values
    for neg_val in (-1, -100, -1024 * 1024, -(2**31), -(2**63)):
        passed, msg = verify_budget(available_space=neg_val, peak_additional=1000)
        assert passed is False
        assert "zero or negative" in msg


def test_adversarial_budget_total_capacity_boundaries():
    """Stress test total disk capacity edge cases against peak additional allocation."""
    peak = 85 * 1024 * 1024 * 1024
    reserve = 50 * 1024 * 1024 * 1024
    adequate_available = peak + reserve + 10 * 1024 * 1024 * 1024

    # Peak exceeds total capacity by 1 byte -> must FAIL
    passed, msg = verify_budget(
        available_space=adequate_available,
        peak_additional=peak,
        reserve_bytes=reserve,
        total_capacity=peak - 1,
    )
    assert passed is False
    assert "exceeds total disk capacity" in msg

    # Peak strictly greater than total capacity -> must FAIL
    passed, msg = verify_budget(
        available_space=adequate_available,
        peak_additional=peak,
        reserve_bytes=reserve,
        total_capacity=peak // 2,
    )
    assert passed is False
    assert "exceeds total disk capacity" in msg

    # Total capacity exactly equals peak:
    # If total_capacity == peak, available space can never be >= peak + reserve (since reserve > 0)
    # Available cannot exceed total capacity on any physical drive
    passed, msg = verify_budget(
        available_space=peak,
        peak_additional=peak,
        reserve_bytes=reserve,
        total_capacity=peak,
    )
    assert passed is False
    assert "deficit" in msg

    # Total capacity with zero capacity
    passed, msg = verify_budget(
        available_space=1, peak_additional=100, total_capacity=0
    )
    assert passed is False
    assert "exceeds total disk capacity" in msg


def test_adversarial_budget_zero_reserve_and_custom_reserves():
    """Verify budget checks with zero reserve and custom extreme reserves."""
    peak = 50 * 1024 * 1024 * 1024

    # Zero reserve: available space must equal or exceed peak
    passed, _ = verify_budget(
        available_space=peak, peak_additional=peak, reserve_bytes=0
    )
    assert passed is True

    passed, msg = verify_budget(
        available_space=peak - 1, peak_additional=peak, reserve_bytes=0
    )
    assert passed is False
    assert "deficit: 1 bytes" in msg

    # Extreme reserve (1 Petabyte)
    huge_reserve = 1024 * 1024 * 1024 * 1024 * 1024
    passed, msg = verify_budget(
        available_space=100 * 1024 * 1024 * 1024,
        peak_additional=peak,
        reserve_bytes=huge_reserve,
    )
    assert passed is False
    assert "deficit" in msg


def test_adversarial_budget_extreme_64bit_values():
    """Verify arbitrary precision handling without overflow on extreme integer values."""
    huge_val = 2**64 - 1
    passed, msg = verify_budget(
        available_space=huge_val,
        peak_additional=100 * 1024**3,
        reserve_bytes=50 * 1024**3,
        total_capacity=huge_val,
    )
    assert passed is True
    assert "passed" in msg


def test_adversarial_alignment_and_quantisation_math():
    """Test alignment arithmetic and quantisation byte calculation with odd shapes."""
    # Align offset tests
    assert align_offset(0) == 0
    assert align_offset(1) == PAGE_ALIGNMENT
    assert align_offset(PAGE_ALIGNMENT - 1) == PAGE_ALIGNMENT
    assert align_offset(PAGE_ALIGNMENT) == PAGE_ALIGNMENT
    assert align_offset(PAGE_ALIGNMENT + 1) == PAGE_ALIGNMENT * 2
    assert align_offset(2**31 - 1) % PAGE_ALIGNMENT == 0
    assert align_offset(2**32 - 1) % PAGE_ALIGNMENT == 0
    assert align_offset(2**33 - 1) % PAGE_ALIGNMENT == 0

    # Quantisation byte calculation for supported types
    assert compute_tensor_bytes(0, [100]) == 400
    assert compute_tensor_bytes(1, [100]) == 200
    assert compute_tensor_bytes(30, [100]) == 200
    assert compute_tensor_bytes(7, [32]) == 24
    assert compute_tensor_bytes(8, [32]) == 34
    assert compute_tensor_bytes(12, [256]) == 144
    assert compute_tensor_bytes(13, [256]) == 176
    assert compute_tensor_bytes(20, [32]) == 18

    # Multi-dimensional calculation
    assert compute_tensor_bytes(8, [640, 2560]) == (640 * 2560 // 32) * 34


def test_adversarial_plan_tensors_invariants():
    """Verify non-overlapping intervals, monotonic offsets, and page alignment across all 1254 tensors."""
    plan = plan_pack()
    tensors = plan["tensors"]
    assert len(tensors) == 1254

    running_offset = 0
    for i, t in enumerate(tensors):
        name = t["name"]
        offset = t["offset"]
        bytes_count = t["bytes"]

        assert offset % PAGE_ALIGNMENT == 0, (
            f"Tensor {name} offset {offset} not 16 KiB aligned"
        )
        assert bytes_count > 0, f"Tensor {name} has zero or negative bytes"
        assert offset >= running_offset, (
            f"Tensor {name} overlaps with previous tensor (offset {offset} < {running_offset})"
        )

        running_offset = offset + bytes_count

    # Verify final file bytes exceeds last tensor extent
    assert plan["planned_file_bytes"] >= running_offset
    assert (
        plan["peak_additional_allocation"]
        == plan["planned_file_bytes"] + MAX_STREAMING_BUFFER_BYTES
    )


def test_adversarial_cli_exit_codes_and_flags(monkeypatch):
    """Verify CLI exit codes under controlled sufficient space and deficit conditions."""
    import scripts.plan_pack_v2 as planner

    cli_path = str(ROOT / "scripts" / "plan_pack_v2.py")

    # Controlled 200 GiB free space: must exit 0 with default 50 GiB reserve.
    monkeypatch.setattr(
        planner,
        "query_filesystem_space",
        lambda _path: (200 * 1024**3, 1024 * 1024**3),
    )
    monkeypatch.setattr(sys, "argv", [cli_path, "--check"])
    assert planner.main() == 0

    # Controlled 64.4 GB free space: must exit 1 even if an existing pack exists on disk.
    monkeypatch.setattr(
        planner,
        "query_filesystem_space",
        lambda _path: (64_400_000_000, 1024 * 1024**3),
    )
    monkeypatch.setattr(sys, "argv", [cli_path, "--check"])
    assert planner.main() == 1

    # Artificial impossible reserve (999,999 GB) via subprocess: must exit 1 (refusal).
    proc_fail = subprocess.run(
        [sys.executable, cli_path, "--check", "--reserve-gb", "999999"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_fail.returncode == 1, (
        f"Expected 1 on reserve deficit, got {proc_fail.returncode}"
    )

    # JSON output check via subprocess (exit code reflects actual uncredited free space).
    proc_json = subprocess.run(
        [sys.executable, cli_path, "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    data = json.loads(proc_json.stdout)
    assert proc_json.returncode == (0 if data["budget_passed"] else 1)
    assert "payload_bytes" in data
    assert "planned_file_bytes" in data
    assert "peak_additional_allocation" in data
    assert "existing_pack_bytes" in data
    assert "budget_passed" in data
