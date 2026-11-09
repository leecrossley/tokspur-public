"""The CPU candidate build selector is explicit and fingerprinted."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def make_stamp(build, selector=None, extra=()):
    command = [
        "make",
        "-s",
        "-C",
        str(ROOT / "native"),
        f"BUILD={build}",
        "ENABLE_METAL=0",
    ]
    if selector is not None:
        command.append(f"M5_CPU_CANDIDATES={selector}")
    command.extend(extra)
    command.append(str(build / ".cflags"))
    environment = dict(os.environ)
    environment.pop("M5_CPU_CANDIDATES", None)
    return subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=30
    )


def test_cpu_candidate_selector_changes_build_fingerprint_and_resets(tmp_path):
    build = tmp_path / "build"
    assert make_stamp(build).returncode == 0
    default = (build / ".cflags").read_text()
    assert "m5-cpu-candidates=0" in default
    assert "TOKSPUR_PRIVATE_M5_CPU_CANDIDATES" not in default
    assert make_stamp(build, "1").returncode == 0
    candidate = (build / ".cflags").read_text()
    assert "m5-cpu-candidates=1" in candidate
    assert candidate.count("-DTOKSPUR_PRIVATE_M5_CPU_CANDIDATES=1") == 1
    assert "-DTOKSPUR_PRIVATE_BOUNDED_TOPK=1" in candidate
    assert candidate != default
    assert make_stamp(build, "0").returncode == 0
    assert (build / ".cflags").read_text() == default


def test_cpu_candidate_selector_survives_explicit_cppflags(tmp_path):
    build = tmp_path / "build"
    assert make_stamp(build, "1", ("CPPFLAGS=-Iinclude",)).returncode == 0
    assert "-DTOKSPUR_PRIVATE_M5_CPU_CANDIDATES=1" in (build / ".cflags").read_text()


@pytest.mark.parametrize("selector", ["", "2", "on", "0 1"])
def test_cpu_candidate_selector_refuses_unsupported_values(tmp_path, selector):
    result = make_stamp(tmp_path / "build", selector)
    assert result.returncode != 0
    assert "expected 0 or 1" in result.stderr
