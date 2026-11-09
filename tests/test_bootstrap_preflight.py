# SPDX-License-Identifier: Apache-2.0
"""Bootstrap preflight: clear failures for missing toolchains, warnings for untested hosts."""

from __future__ import annotations

import importlib.util
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bootstrap():
    spec = importlib.util.spec_from_file_location(
        "bootstrap_under_test", ROOT / "scripts/bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed(stdout="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def fake_host(
    monkeypatch,
    bootstrap,
    *,
    metal=0,
    rustc="rustc {channel} (0000000 2026-10-13)",
    rustup=True,
    memory=128 * 1024**3,
    macos="27.0.1",
):
    channel = bootstrap.pinned_rust_channel()
    monkeypatch.setattr(bootstrap.sys, "platform", "darwin")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        bootstrap.platform, "mac_ver", lambda: (macos, ("", "", ""), "")
    )
    monkeypatch.setattr(
        bootstrap.shutil,
        "which",
        lambda name, path=None: "/bin/rustup" if rustup else None,
    )

    def run(command, **_):
        if command[:2] == ["xcrun", "metal"]:
            return completed(returncode=metal)
        if command[0] == "rustc":
            return completed(stdout=rustc.format(channel=channel) + "\n")
        raise AssertionError(command)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    monkeypatch.setattr(
        bootstrap.subprocess, "check_output", lambda command, **_: f"{memory}\n"
    )


def test_pinned_channel_matches_the_toolchain_file(bootstrap):
    pin = tomllib.loads((ROOT / "rust/rust-toolchain.toml").read_text())
    assert bootstrap.pinned_rust_channel() == pin["toolchain"]["channel"]


def test_macos_major_parses_product_versions(bootstrap):
    assert bootstrap.macos_major("27.0.1") == 27
    assert bootstrap.macos_major("26.4") == 26
    assert bootstrap.macos_major("") is None


def test_supported_host_passes_without_warnings(monkeypatch, bootstrap):
    fake_host(monkeypatch, bootstrap)
    assert bootstrap.preflight({}) == []


def test_missing_metal_toolchain_names_the_download(monkeypatch, bootstrap):
    fake_host(monkeypatch, bootstrap, metal=1)
    with pytest.raises(ValueError, match="downloadComponent MetalToolchain"):
        bootstrap.preflight({})


def test_missing_rustup_or_pinned_toolchain_names_the_install(monkeypatch, bootstrap):
    channel = bootstrap.pinned_rust_channel()
    fake_host(monkeypatch, bootstrap, rustup=False)
    with pytest.raises(ValueError, match="rustup.rs"):
        bootstrap.preflight({})
    fake_host(monkeypatch, bootstrap, rustc="rustc 1.80.0 (0000000 2024-07-21)")
    with pytest.raises(ValueError, match=f"rustup toolchain install {channel}"):
        bootstrap.preflight({})


def test_small_memory_and_older_macos_warn(monkeypatch, bootstrap):
    fake_host(monkeypatch, bootstrap, memory=64 * 1024**3, macos="26.4")
    warnings = bootstrap.preflight({})
    assert len(warnings) == 2
    assert "128 GB" in warnings[0]
    assert "macOS 27" in warnings[1]


def test_other_platforms_are_refused(monkeypatch, bootstrap):
    monkeypatch.setattr(bootstrap.sys, "platform", "linux")
    with pytest.raises(ValueError, match="Apple silicon"):
        bootstrap.preflight({})


def test_path_remapping_preserves_encoded_flags(monkeypatch, bootstrap, tmp_path):
    monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", "-C\x1fopt-level=1")
    monkeypatch.setenv("RUSTFLAGS", "--ignored-by-cargo")
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path / "source with spaces")
    env = bootstrap.build_env()
    flags = bootstrap.effective_rustflags(env)
    assert flags[:2] == ["-C", "opt-level=1"]
    assert flags[-1] == f"--remap-path-prefix={bootstrap.ROOT}=/tokspur"
    recorded = bootstrap.public_rustflags(env)
    assert str(Path.home()) not in recorded
    assert str(bootstrap.ROOT) not in recorded
    assert "$CHECKOUT=/tokspur" in recorded
    assert (
        bootstrap.effective_rustflags(
            {"CARGO_ENCODED_RUSTFLAGS": "", "RUSTFLAGS": "--ignored"}
        )
        == []
    )


def test_signing_refuses_remaining_home_paths(monkeypatch, bootstrap, tmp_path):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "EXECUTABLES", ("host",))
    (tmp_path / "host").write_bytes(b"generated fixture\0/Users/private-owner/source\0")
    calls = []
    monkeypatch.setattr(
        bootstrap, "run", lambda command, **kwargs: calls.append(command)
    )
    with pytest.raises(ValueError, match="still contains a home path"):
        bootstrap.codesign_executables("test identity", None, {})
    assert len(calls) == 1
    assert "--sign" not in calls[0]
