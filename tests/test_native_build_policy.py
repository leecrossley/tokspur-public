"""Separate native outputs must not mislabel a shared object cache's policy."""

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
