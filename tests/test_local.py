# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from tokspur import local

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "ROOT", tmp_path)
    return tmp_path


def test_private_state_permissions_and_no_symlinks(sandbox):
    state = local.private_dir(sandbox / ".tokspur/state")
    assert state.stat().st_mode & 0o777 == 0o700
    (state / "link").symlink_to(sandbox, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        local.private_dir(state / "link/elsewhere")
    with pytest.raises(ValueError):
        local.private_dir(sandbox / "outside")
    with pytest.raises(ValueError):
        local.private_dir(sandbox / ".tokspur/state/../../outside")


def test_atomic_private_json_refuses_link(sandbox):
    target = sandbox / ".tokspur/state/test.json"
    local.write_json(target, {"key": "private"})
    assert json.loads(target.read_text()) == {"key": "private"}
    assert target.stat().st_mode & 0o777 == 0o600
    link = target.parent / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        local.write_json(link, {})
    assert json.loads(target.read_text()) == {"key": "private"}


def test_hash_verified_in_place(tmp_path):
    asset = tmp_path / "weights"
    asset.write_bytes(b"test weight fixture")
    spec = {
        "size": asset.stat().st_size,
        "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
    }
    result = local.verify_asset(asset, spec)
    assert result["sha256"] == spec["sha256"]
    assert result["inode"] == asset.stat().st_ino
    with pytest.raises(ValueError, match="Checksum"):
        local.verify_asset(asset, {**spec, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="size"):
        local.verify_asset(asset, {**spec, "size": 0})


@pytest.mark.skipif(not local.IS_DARWIN, reason="requires the Apple volume API")
def test_volume_identity_matches_distinct_files_on_the_same_volume(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    a, b = local.identity(first), local.identity(second)
    assert a["inode"] != b["inode"]
    assert len(a["volume_uuid"]) == 32
    assert a["volume_uuid"] == b["volume_uuid"]


def test_adoption_remount_keeps_volume_and_file_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "_volume_uuid", lambda fd: "ab" * 16)
    data = {"schema": 1, "verification": "full-sha256"}
    profile = {}
    for group in ("target", "draft"):
        directory = tmp_path / group
        directory.mkdir()
        path = directory / "fixture"
        path.write_bytes(group.encode())
        spec = {
            "name": "fixture",
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        profile[group] = {"files": [spec]}
        data[group] = {
            "directory": str(directory),
            "assets": [local.verify_asset(path, spec)],
        }
    monkeypatch.setattr(local, "read_json", lambda path: profile)
    local.check_adoption(data)
    original = json.loads(json.dumps(data))
    asset = data["target"]["assets"][0]
    asset["device"] += 1
    local.check_adoption(data)
    for key in ("inode", "size", "mtime_ns", "ctime_ns"):
        asset[key] += 1
        with pytest.raises(ValueError, match="Asset changed"):
            local.check_adoption(data)
        asset[key] -= 1
    for uuid in (None, 0, "bad", "z" * 32, "0" * 32):
        asset["volume_uuid"] = uuid
        with pytest.raises(ValueError, match="volume UUID|Asset changed"):
            local.check_adoption(data)
    asset["volume_uuid"] = "ab" * 16
    for device in (-1, True, 0x80000000):
        asset["device"] = device
        with pytest.raises(ValueError, match="invalid recorded asset device"):
            local.check_adoption(data)
    legacy = original["target"]["assets"][0]
    del legacy["volume_uuid"]
    local.check_adoption(original)
    legacy["device"] += 1
    with pytest.raises(ValueError, match="Asset changed"):
        local.check_adoption(original)


def test_port_guard():
    with socket.socket() as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        assert not local.port_available(occupied.getsockname()[1])


def test_port_guard_sees_wildcard_listener():
    # macOS lets an exact 127.0.0.1 bind succeed beside a 0.0.0.0 listener, so
    # the bind probe alone would report this port as free.
    with socket.socket() as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("0.0.0.0", 0))
        occupied.listen()
        assert not local.port_available(occupied.getsockname()[1])


def test_port_guard_allows_reuse_after_connection_closes():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            connection, _ = listener.accept()
            connection.close()  # Server-side TIME_WAIT must not look occupied.
            assert client.recv(1) == b""
    assert local.port_available(port)


def test_inference_processes_include_owned_native_worker(monkeypatch):
    import psutil

    class Fake:
        def __init__(self, pid, name, cmdline):
            self.pid = pid
            self.info = {"name": name, "cmdline": cmdline}

    fakes = [
        Fake(10, None, ["/x/build/owned-native/tokspur-native", "target", "--policy"]),
        Fake(11, "omlx-server", None),  # cmdline unreadable; the name still counts
        Fake(12, "bash", ["bash", "-c", "echo tokspur-native"]),
        Fake(13, "Python", ["python", "-m", "pytest"]),
        Fake(14, "branch8-target", None),
        Fake(15, None, ["/x/private/branch8-target", "target"]),
        Fake(16, "stage8-target", None),
        Fake(17, None, ["/x/private/stage8-target", "target"]),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(fakes))
    assert {entry["pid"] for entry in local.inference_processes()} == {
        10,
        11,
        14,
        15,
        16,
        17,
    }


def test_scalar_identity_full_argv_unreadable_name_and_exact_darwin_alias(monkeypatch):
    from types import SimpleNamespace

    import psutil

    full = local.SCALAR_WORKER_NAME
    short = full[: local.DARWIN_MAXCOMLEN]
    cases = [
        (201, None, ["/private/candidate/" + full, "target"]),
        (202, full, None),
        (203, short, None),  # Another user's argv is unreadable.
        (204, short, []),
        (205, short, ["/usr/bin/unrelated"]),
        (206, short + "x", None),
        (207, short[:-1], None),
        (208, "unrelated", ["bash", "-c", "echo " + full]),
        (209, "unrelated", None),
        (210, full, ["/private/candidate/" + full]),  # self
    ]
    fakes = [
        SimpleNamespace(pid=pid, info={"name": name, "cmdline": argv})
        for pid, name, argv in cases
    ]
    monkeypatch.setattr(local, "IS_DARWIN", True)
    monkeypatch.setattr(local.os, "getpid", lambda: 210)
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(fakes))
    assert {row["pid"] for row in local.inference_processes()} == {201, 202, 203, 204}
    monkeypatch.setattr(local, "IS_DARWIN", False)
    assert {row["pid"] for row in local.inference_processes()} == {201, 202}


def test_scalar_identity_retains_permission_and_disappearance_handling(monkeypatch):
    import psutil

    class Unreadable:
        pid = 301

        @property
        def info(self):
            raise psutil.AccessDenied(self.pid)

    class Gone(Unreadable):
        @property
        def info(self):
            raise psutil.NoSuchProcess(self.pid)

    monkeypatch.setattr(
        psutil, "process_iter", lambda attrs=None: iter([Unreadable(), Gone()])
    )
    assert local.inference_processes() == []


def test_child_environment_drops_tokens_and_global_overrides(sandbox, monkeypatch):
    for key in (
        "HF_TOKEN",
        "HF_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(key, "do-not-inherit")
    env = local.isolated_environment()
    assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "do-not-inherit" not in env.values()


def test_state_lock_is_exclusive_and_released(sandbox):
    with local.state_lock():
        with pytest.raises(ValueError, match="owns this state"):
            with local.state_lock():
                pytest.fail("Two operations acquired the same state")
    with local.state_lock():
        pass


def test_state_lock_refuses_symlink(sandbox):
    state = local.private_dir(sandbox / ".tokspur/state")
    outside = sandbox / "outside"
    outside.write_text("unchanged")
    (state / "serve.lock").symlink_to(outside)
    with pytest.raises(OSError):
        with local.state_lock():
            pass
    assert outside.read_text() == "unchanged"


def test_profile_has_explicit_conservative_contract():
    p = local.profile()
    assert (p["context_tokens"], p["kv_format"], p["state_format"]) == (
        65536,
        "bf16",
        "fp32",
    )
    assert p["verify_rows"] == "all" and not p["persistent_cache"]
    assert p["resident_experts"] and p["queue_size"] == 1
    assert p["native_memory_gib"] < p["api_memory_ceiling_gib"] < 128
