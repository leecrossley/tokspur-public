"""Separate native outputs must not mislabel a shared object cache's policy."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="requires the Apple toolchain")
@pytest.mark.parametrize(
    "control",
    [
        "VERIFY_NARROW_ROWS",
        "PACK_SHARED_MAPPING",
        "PACK_AUTH_RECLAIM",
        "CACHE_PRESSURE_RECLAIM",
        "MEMORY_PLAN_REPORT",
        "SHARED_PREFIX_EVICTION_COUNTERS",
    ],
)
def test_shared_objects_rebuild_on_a_b_a_policy_switch(tmp_path, control):
    if not shutil.which("make") or not shutil.which("xcrun"):
        pytest.skip("Apple build tools are unavailable")
    compiler = subprocess.run(
        ["xcrun", "--find", "clang++"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert Path(compiler).is_file()
    native = Path(__file__).resolve().parents[1] / "native"
    objects = tmp_path / "objects"
    target = objects / "measured_controller.o"
    stamps = []
    for arm, narrow, compile_expected in (
        ("a", 1, True),
        ("b", 0, True),
        ("a", 1, True),
        ("a", 1, False),
    ):
        build = tmp_path / arm
        result = subprocess.run(
            [
                "make",
                "-C",
                str(native),
                "CXX=xcrun --sdk macosx clang++",
                "ENABLE_METAL=0",
                "NATIVE_CONFIG=carrier",
                "Q8_IMPORT_ISOLATION=1",
                f"{control}={narrow}",
                f"BUILD={build}",
                f"OBJDIR={objects}",
                str(target),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (" -c " in result.stdout) == compile_expected
        stamp = (objects / ".cflags").read_bytes()
        assert stamp == (build / ".cflags").read_bytes()
        stamps.append(stamp)
    assert stamps[0] == stamps[2] == stamps[3] and stamps[0] != stamps[1]


def test_prefill_shader_and_library_follow_same_timestamp_policy_changes(tmp_path):
    native = Path(__file__).resolve().parents[1] / "native"
    shader = tmp_path / "graph.air"
    library = tmp_path / "tokspur-owned.metallib"
    stamp = tmp_path / ".metalflags"
    # Model the compiler and linker outputs so this dependency test runs on Linux too.
    xcrun = tmp_path / "xcrun"
    xcrun.write_text(
        '#!/bin/sh\nfor output in "$@"; do :; done\n'
        'if [ "$1" = metal ]; then echo "$*" > "$output"; else cp "$2" "$output"; fi\n'
    )
    xcrun.chmod(0o755)
    # None leaves a switch to the Makefile default, which batches.
    cases = [(None, None), (0, 0), (1, 0), (0, 1), (1, 1), (0, 0), (None, None)]
    for top10, mixture in cases:
        make = [
            "make",
            "-s",
            "-C",
            str(native),
            "ENABLE_METAL=1",
            f"BUILD={tmp_path}",
            f"ALL_AIR={shader}",
        ]
        if top10 is not None:
            make.append(f"PREFILL_BATCH_TOP10={top10}")
        if mixture is not None:
            make.append(f"PREFILL_BATCH_MIXTURE={mixture}")
        env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
        env.pop("PREFILL_BATCH_TOP10", None)
        env.pop("PREFILL_BATCH_MIXTURE", None)
        subprocess.run([*make, str(stamp)], check=True, env=env)
        if shader.exists():
            timestamp = stamp.stat().st_mtime_ns
            os.utime(shader, ns=(timestamp, timestamp))
        subprocess.run([*make, str(shader)], check=True, env=env)
        if library.exists():
            timestamp = shader.stat().st_mtime_ns
            os.utime(library, ns=(timestamp, timestamp))
        subprocess.run([*make, str(library)], check=True, env=env)
        compiled = shader.read_text()
        assert library.read_text() == compiled
        assert ("-DTOKSPUR_PRIVATE_PREFILL_BATCH_TOP10=1" in compiled) == (top10 != 0)
        assert ("-DTOKSPUR_PRIVATE_PREFILL_BATCH_MIXTURE=1" in compiled) == (
            mixture != 0
        )
