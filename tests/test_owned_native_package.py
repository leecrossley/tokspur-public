"""Tiny filesystem/proof fakes only; never build or execute a staged artifact."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import socket
import stat
import subprocess
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def package(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Packaging must not launch processes or use the network")

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("system", "fork", "execv", "execve", "posix_spawn", "posix_spawnp"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    spec = importlib.util.spec_from_file_location(
        "native_package_under_test", ROOT / "scripts/stage_owned_native_package.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Read/write fault injection must not alter pytest's shared stdlib OS module.
    module.os = types.SimpleNamespace(**vars(os))
    return module


@pytest.fixture
def case(tmp_path, package):
    # Resolve pytest's own temporary prefix (macOS may spell it through /var).
    base = tmp_path.resolve()
    root, build = base / "owned source with spaces", base / "owned build with spaces"
    root.mkdir()
    build.mkdir()

    def write(path, data, mode=0o600):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)

    write(root / "native/src/owned.cpp", b"// inert independently owned fake\n")
    write(root / "native/Makefile", b"# fake; never execute\n")
    write(root / "native/PROVENANCE.md", b"Owned fake provenance\n")
    write(root / "native/data/UNICODE-LICENSE.txt", b"Unicode fake notice\n")
    write(root / "LICENSE", b"Fake licence for inert fixture only\n")
    for name in package.NOTICE_FILES:
        if not (root / name).exists():
            write(root / name, f"Inert fixture notice: {name}\n".encode())
    checker = b'CPU_SUITES = tuple(("gguf numeric").split())\n'
    checker += b'METAL_SUITES = tuple(("graph stack").split())\n'
    write(root / "scripts/check_owned_gpu_experiments.py", checker)
    protocol = root / "scripts/check_owned_target_protocol.py"
    write(protocol, b"# inert fake protocol checker\n")
    policy = root / package.POLICY
    write(policy, b'{"router":"softmax","normalize_selected":true}\n')
    names = package.ARTIFACTS + (
        "test-gguf",
        "test-numeric",
        "test-graph",
        "test-stack",
    )
    for name in names:
        write(build / name, ("inert fake " + name + "\n").encode(), 0o700)
    commands = [
        ["c++", "--version"],
        ["xcrun", "metal", "--version"],
        [str(build / "tokspur-native"), "host-info"],
        ["otool", "-L", str(build / "tokspur-native")],
        ["otool", "-L", str(build / "tokspur-server")],
        ["/fake/python with spaces", str(protocol)],
        [str(build / "test-gguf")],
        [str(build / "test-numeric")],
        [str(build / "test-graph"), str(build / "tokspur-owned.metallib")],
        [str(build / "test-stack"), str(build / "tokspur-owned.metallib")],
    ]
    proof = {
        "schema": 1,
        "pass": True,
        "system_only_component_link_check": True,
        "harness_sha256": digest(checker),
        "protocol_checker_sha256": digest(protocol.read_bytes()),
        "native_sources": {
            str(path.relative_to(root)): digest(path.read_bytes())
            for path in sorted((root / "native").rglob("*"))
            if path.is_file()
        },
        "artifacts": {name: digest((build / name).read_bytes()) for name in names},
        "commands": [
            {"argv": argv, "returncode": 0, "stdout": "fake", "stderr": ""}
            for argv in commands
        ],
    }
    for index, name in ((3, "tokspur-native"), (4, "tokspur-server")):
        proof["commands"][index]["stdout"] = (
            str(build / name)
            + ":\n"
            + "\t/usr/lib/libSystem.B.dylib "
            + "(compatibility version 1.0.0, current version 1351.0.0)\n"
        )
    proof["commands"][5]["stdout"] = (
        "Owned CPU target protocol checks passed: 245\n"
        "Owned CPU target stack-experiment protocol checks passed: 59\n"
    )
    receipt = base / "verified build receipt.json"
    args = {
        "source_root": root,
        "build": build,
        "receipt": receipt,
        "receipt_sha256": "",
        "policy": policy,
        "policy_sha256": digest(policy.read_bytes()),
        "output": base / "new private native package with spaces",
    }

    def save():
        raw = (json.dumps(proof) + "\n").encode()
        write(receipt, raw)
        args["receipt_sha256"] = digest(raw)

    save()
    return args, proof, save


def test_exact_five_file_private_model_free_stage(package, case):
    args, proof, _save = case
    result = package.stage(**args)
    output = args["output"]
    expected = set(package.ARTIFACTS) | {"owned-target-experimental.json"}
    assert set(path.name for path in output.iterdir()) == expected | {package.MANIFEST}
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    manifest_bytes = (output / package.MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    assert result["provenance_sha256"] == digest(manifest_bytes)
    assert manifest["proof"]["build_receipt_sha256"] == args["receipt_sha256"]
    assert manifest["proof"]["native_sources"] == proof["native_sources"]
    assert manifest["proof"]["artifacts"] == proof["artifacts"]
    assert manifest["proof"]["notices"]["LICENSE"].startswith("Fake licence")
    assert "Unicode fake notice" in str(manifest["proof"]["notices"])
    assert set(manifest["entries"]) == expected
    for name, entry in manifest["entries"].items():
        path = output / name
        assert entry["sha256"] == digest(path.read_bytes())
        assert entry["size"] == path.stat().st_size
        expected_mode = 0o700 if name in package.ARTIFACTS[:2] else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected_mode
        assert path.stat().st_nlink == 1
        source = args["policy"] if name.endswith(".json") else args["build"] / name
        assert path.read_bytes() == source.read_bytes()
        assert path.stat().st_ino != source.stat().st_ino
    assert str(args["receipt"]) not in manifest_bytes.decode()
    assert str(args["source_root"]) not in manifest_bytes.decode()
    assert not any(path.suffix == ".py" for path in output.iterdir())


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling"])
def test_never_merge_or_overwrite_existing_output(package, case, kind):
    args, _proof, _save = case
    output = args["output"]
    protected = output.parent / "protected"
    protected.mkdir()
    sentinel = protected / "keep"
    sentinel.write_bytes(b"unchanged")
    if kind == "directory":
        output.mkdir()
        (output / "keep").write_bytes(b"unchanged")
    elif kind == "file":
        output.write_bytes(b"unchanged")
    else:
        output.symlink_to(protected if kind == "symlink" else protected / "missing")
    with pytest.raises((RuntimeError, OSError)):
        package.stage(**args)
    assert sentinel.read_bytes() == b"unchanged"
    if kind == "directory":
        assert list(output.iterdir()) == [output / "keep"]
    elif kind == "file":
        assert output.read_bytes() == b"unchanged"
    else:
        assert output.is_symlink()


@pytest.mark.parametrize("name", ["source_root", "build", "receipt", "policy"])
def test_refuse_symlink_in_input_or_ancestor(package, case, name):
    args, _proof, _save = case
    original = args[name]
    link = original.parent / (original.name + " link")
    link.symlink_to(original, target_is_directory=original.is_dir())
    args[name] = link
    with pytest.raises((RuntimeError, OSError)):
        package.stage(**args)
    assert not args["output"].exists()


def test_refuse_symlinked_output_parent(package, case):
    args, _proof, _save = case
    link = args["output"].parent / "linked output parent"
    link.symlink_to(args["output"].parent, target_is_directory=True)
    args["output"] = link / "must not appear"
    with pytest.raises((RuntimeError, OSError)):
        package.stage(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "writable", "directory"])
def test_refuse_unsafe_artifact(package, case, kind):
    args, _proof, _save = case
    artifact = args["build"] / "tokspur-server"
    saved = artifact.with_name("saved fake server")
    artifact.rename(saved)
    if kind == "symlink":
        artifact.symlink_to(saved)
    elif kind == "hardlink":
        os.link(saved, artifact)
    elif kind == "writable":
        artifact.write_bytes(saved.read_bytes())
        artifact.chmod(0o777)
    else:
        artifact.mkdir()
    with pytest.raises((RuntimeError, OSError)):
        package.stage(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("field", ["receipt_sha256", "policy_sha256"])
@pytest.mark.parametrize("bad", ["", "f" * 64, "A" * 64, True, "0" * 63])
def test_explicit_digests_required(package, case, field, bad):
    args, _proof, _save = case
    args[field] = bad
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "failed",
        "bool_schema",
        "error",
        "no_link_flag",
        "source_hash",
        "source_missing",
        "source_extra",
        "artifact_hash",
        "artifact_missing",
        "artifact_extra",
        "producer_hash",
        "protocol_hash",
        "command_missing",
        "command_extra",
        "command_reordered",
        "command_returncode",
        "command_bool_returncode",
        "command_stderr_type",
        "command_wrong_artifact",
        "protocol_partial",
        "protocol_duplicate",
        "link_missing",
        "link_external",
        "link_relative",
        "link_wrong_header",
        "link_extra_line",
    ],
)
def test_refuse_incomplete_or_drifted_proof_before_output(package, case, mutation):
    args, proof, save = case
    if mutation == "failed":
        proof["pass"] = False
    elif mutation == "bool_schema":
        proof["schema"] = True
    elif mutation == "error":
        proof["error"] = "retained failure"
    elif mutation == "no_link_flag":
        proof["system_only_component_link_check"] = False
    elif mutation.startswith("source_"):
        if mutation == "source_hash":
            proof["native_sources"]["native/Makefile"] = "f" * 64
        elif mutation == "source_missing":
            del proof["native_sources"]["native/Makefile"]
        else:
            proof["native_sources"]["native/extra.cpp"] = "f" * 64
    elif mutation.startswith("artifact_"):
        if mutation == "artifact_hash":
            proof["artifacts"]["tokspur-native"] = "f" * 64
        elif mutation == "artifact_missing":
            del proof["artifacts"]["test-graph"]
        else:
            proof["artifacts"]["external-engine"] = "f" * 64
    elif mutation in ("producer_hash", "protocol_hash"):
        key = (
            "harness_sha256"
            if mutation == "producer_hash"
            else "protocol_checker_sha256"
        )
        proof[key] = "f" * 64
    elif mutation == "command_missing":
        proof["commands"].pop()
    elif mutation == "command_extra":
        proof["commands"].append(copy.deepcopy(proof["commands"][-1]))
    elif mutation == "command_reordered":
        proof["commands"][6:8] = reversed(proof["commands"][6:8])
    elif mutation == "command_returncode":
        proof["commands"][-1]["returncode"] = 1
    elif mutation == "command_bool_returncode":
        proof["commands"][-1]["returncode"] = False
    elif mutation == "command_stderr_type":
        proof["commands"][-1]["stderr"] = None
    elif mutation == "command_wrong_artifact":
        proof["commands"][-1]["argv"][0] = "/external/test-stack"
    elif mutation == "protocol_partial":
        proof["commands"][5]["stdout"] = (
            "Owned CPU target protocol checks passed: 245\n"
        )
    elif mutation == "protocol_duplicate":
        proof["commands"][5]["stdout"] *= 2
    elif mutation == "link_missing":
        proof["commands"][3]["stdout"] = str(args["build"] / "tokspur-native") + ":\n"
    elif mutation == "link_external":
        proof["commands"][3]["stdout"] = proof["commands"][3]["stdout"].replace(
            "/usr/lib/", "/opt/external/"
        )
    elif mutation == "link_relative":
        proof["commands"][3]["stdout"] = proof["commands"][3]["stdout"].replace(
            "/usr/lib/libSystem.B.dylib", "@rpath/engine.dylib"
        )
    elif mutation == "link_wrong_header":
        proof["commands"][3]["stdout"] = proof["commands"][3]["stdout"].replace(
            "tokspur-native:", "other-engine:"
        )
    else:
        proof["commands"][3]["stdout"] += "malformed unexplained line\n"
    save()
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("name", ["source", "artifact", "policy", "receipt"])
def test_midcopy_drift_retains_only_unpublished_failed_stage(
    package, case, monkeypatch, name
):
    args, _proof, _save = case
    original = package.copy_file
    changed = False

    def copy_then_drift(*values):
        nonlocal changed
        result = original(*values)
        if not changed:
            changed = True
            paths = {
                "source": args["source_root"] / "native/Makefile",
                "artifact": args["build"] / "test-stack",
                "policy": args["policy"],
                "receipt": args["receipt"],
            }
            with paths[name].open("ab") as stream:
                stream.write(b"drift")
        return result

    monkeypatch.setattr(package, "copy_file", copy_then_drift)
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert args["output"].is_dir()
    assert not (args["output"] / package.MANIFEST).exists()
    # Never merge or silently retry into the failed output.
    with pytest.raises((RuntimeError, OSError)):
        package.stage(**args)


def test_input_change_during_read_refused(package, case, monkeypatch):
    args, _proof, _save = case
    original = package.os.read
    path = args["receipt"]
    changed = False

    def read_then_change(descriptor, count):
        nonlocal changed
        data = original(descriptor, count)
        if data and not changed:
            changed = True
            with path.open("ab") as stream:
                stream.write(b" ")
        return data

    monkeypatch.setattr(package.os, "read", read_then_change)
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("entry", ["native/link", "native/subdir/link"])
def test_source_symlink_refused(package, case, entry):
    args, _proof, _save = case
    path = args["source_root"] / entry
    path.parent.mkdir(exist_ok=True)
    path.symlink_to(args["policy"])
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert not args["output"].exists()


def test_policy_must_match_declared_owned_source(package, case):
    args, _proof, _save = case
    different = args["output"].parent / "other policy.json"
    different.write_bytes(b'{"router":"other"}\n')
    different.chmod(0o600)
    args["policy"] = different
    args["policy_sha256"] = digest(different.read_bytes())
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert not args["output"].exists()


def test_byte_identical_explicit_policy_copy_is_allowed(package, case):
    args, _proof, _save = case
    separate = args["output"].parent / "explicit copied policy with spaces.json"
    separate.write_bytes(args["policy"].read_bytes())
    separate.chmod(0o600)
    args["policy"] = separate
    package.stage(**args)
    assert (args["output"] / separate.name).exists() is False
    assert (args["output"] / "owned-target-experimental.json").read_bytes() == (
        separate.read_bytes()
    )


def test_copy_handles_short_writes_without_losing_bytes(package, case, monkeypatch):
    args, _proof, _save = case
    original = package.os.write

    def short_write(descriptor, data):
        return original(descriptor, data[:7])

    monkeypatch.setattr(package.os, "write", short_write)
    result = package.stage(**args)
    assert result["entries"]["tokspur-native"]["sha256"] == digest(
        (args["build"] / "tokspur-native").read_bytes()
    )
    assert result["provenance_sha256"] == digest(
        (args["output"] / package.MANIFEST).read_bytes()
    )


def test_short_write_failure_does_not_publish_manifest(package, case, monkeypatch):
    args, _proof, _save = case
    monkeypatch.setattr(package.os, "write", lambda *_args: 0)
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert args["output"].is_dir()
    assert not (args["output"] / package.MANIFEST).exists()


def test_unexpected_entry_refused_before_manifest(package, case, monkeypatch):
    args, _proof, _save = case
    original = package.copy_file

    def copy_with_unexpected(*values):
        result = original(*values)
        (args["output"] / "must not ship a venv").mkdir(exist_ok=True)
        return result

    monkeypatch.setattr(package, "copy_file", copy_with_unexpected)
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert not (args["output"] / package.MANIFEST).exists()


@pytest.mark.parametrize("kind", ["directory", "payload"])
def test_staged_permissions_must_remain_private(package, case, monkeypatch, kind):
    args, _proof, _save = case
    original = package.copy_file

    def copy_then_chmod(*values):
        result = original(*values)
        path = args["output"]
        if kind == "payload":
            path = path / "tokspur-native"
        path.chmod(0o755)
        return result

    monkeypatch.setattr(package, "copy_file", copy_then_chmod)
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert not (args["output"] / package.MANIFEST).exists()


def test_read_bounds_and_empty_inputs(package, tmp_path):
    path = tmp_path.resolve() / "bounded tiny input"
    path.write_bytes(b"12345")
    path.chmod(0o600)
    with pytest.raises(RuntimeError):
        package.read_file(path, 4)
    assert package.read_file(path, 5) == b"12345"
    path.write_bytes(b"")
    with pytest.raises(RuntimeError):
        package.read_file(path, 5)


def test_fifo_input_refused_without_blocking(package, case):
    args, _proof, _save = case
    receipt = args["receipt"]
    receipt.rename(receipt.with_name("original receipt.json"))
    os.mkfifo(receipt, 0o600)
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize(
    "field", ["source_root", "build", "receipt", "policy", "output"]
)
def test_relative_paths_refused(package, case, field):
    args, _proof, _save = case
    args[field] = Path("relative path with spaces")
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("input_name", ["source_root", "build"])
def test_output_cannot_modify_source_or_build_tree(package, case, input_name):
    args, _proof, _save = case
    args["output"] = args[input_name] / "new package"
    with pytest.raises(RuntimeError):
        package.stage(**args)
    assert not args["output"].exists()


def test_duplicate_json_and_nonfinite_refused(package):
    for value in (b'{"pass":true,"pass":false}', b'{"value":NaN}'):
        with pytest.raises(RuntimeError):
            package.strict_json(value)


@pytest.mark.parametrize(
    "source",
    [
        "CPU_SUITES = tuple(read_external().split())\n"
        'METAL_SUITES=tuple(("a").split())',
        'CPU_SUITES = tuple(("a a").split())\nMETAL_SUITES=tuple(("b").split())',
        'CPU_SUITES = tuple(("../engine").split())\nMETAL_SUITES=tuple(("b").split())',
    ],
)
def test_proof_inventory_is_literal_unique_and_closed(package, source):
    with pytest.raises(RuntimeError):
        package.suites(source)


def test_current_producer_inventory_can_be_read_without_import(package):
    checker = (ROOT / "scripts/check_owned_gpu_experiments.py").read_bytes()
    cpu, metal = package.suites(checker)
    assert len(cpu) == 32
    assert len(metal) == 18
    assert {
        "derived_pack",
        "serve_protocol",
        "http",
        "owned_worker",
        "transcript_replay",
    } <= set(cpu)
    assert {"hc_q8_gemv", "q8_split_gemv"} <= set(metal)


@pytest.fixture
def rust_case(tmp_path, package):
    base = tmp_path.resolve()
    root = base / "rust owned source"
    build = base / "owned native build"
    rust_build = base / "rust fast math release"
    root.mkdir()
    build.mkdir()
    rust_build.mkdir()

    def write(path, data, mode=0o600):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)

    write(root / "native/src/owned.cpp", b"// native source\n")
    write(root / "native/Makefile", b"# makefile\n")
    write(root / "native/PROVENANCE.md", b"Owned provenance\n")
    write(root / "native/README.md", b"Owned notes\n")
    write(root / "native/data/UNICODE-LICENSE.txt", b"Unicode licence\n")
    write(root / "LICENSE", b"Apache-2.0\n")
    for name in package.NOTICE_FILES:
        if not (root / name).exists():
            write(root / name, f"Inert fixture notice: {name}\n".encode())
    write(root / "rust/Cargo.toml", b"[workspace]\n")
    write(root / "rust/Cargo.lock", b"version = 4\n")
    write(root / "rust/rust-toolchain.toml", b'[toolchain]\nchannel = "1.99.1"\n')
    write(root / "rust/.cargo/config.toml", b"[build]\n")
    write(root / "rust/crates/tokspur-cli/src/main.rs", b"fn main() {}\n")

    policy = root / package.POLICY
    hw = root / package.HARDWARE_PROFILE
    assets = root / package.ASSETS_PROFILE
    write(policy, b'{"schema":1}\n')
    write(hw, b'{"schema":1,"port":18081}\n')
    write(assets, b'{"schema":1,"target":{},"draft":{}}\n')

    macho_prefix = b"\xcf\xfa\xed\xfe\x0c\x00\x00\x01"
    write(rust_build / "tokspur", macho_prefix + b"fake-cli", 0o700)
    write(rust_build / "tokspur-server", macho_prefix + b"fake-srv", 0o700)
    write(build / "tokspur-native", macho_prefix + b"fake-worker", 0o700)
    write(build / "tokspur-owned.metallib", b"MTLB\x01\x00\x02\x00fake-metal", 0o600)

    def artifact_rec(p):
        b = p.read_bytes()
        return {
            "sha256": digest(b),
            "size": len(b),
            "mode": oct(stat.S_IMODE(p.stat().st_mode)),
        }

    manifest = {
        "schema": 2,
        "git_head": "45563e6" + "0" * 33,
        "git_dirty": False,
        "enable_metal": True,
        "frozen": True,
        "metallib_function_count": 16,
        "native_config": {"NATIVE_CONFIG": "carrier"},
        "tracked_cflags": ["carrier", ":", "test flags"],
        "tracked_metalflags": "carrier : metal4",
        "rust_targets": {"tokspur-server": ["fast-math"], "tokspur": []},
        "native_source_scope": package.NATIVE_SOURCE_SCOPE,
        # Build inputs only: the Markdown notes written above are not pinned.
        "native_sources": {
            rel: digest((root / rel).read_bytes())
            for rel in (
                "native/Makefile",
                "native/data/UNICODE-LICENSE.txt",
                "native/src/owned.cpp",
                *(
                    name
                    for name in package.NOTICE_FILES
                    if name.startswith("native/third_party/")
                ),
            )
        },
        "rust_sources": {
            "rust/Cargo.toml": digest((root / "rust/Cargo.toml").read_bytes()),
            "rust/Cargo.lock": digest((root / "rust/Cargo.lock").read_bytes()),
            "rust/rust-toolchain.toml": digest(
                (root / "rust/rust-toolchain.toml").read_bytes()
            ),
            "rust/.cargo/config.toml": digest(
                (root / "rust/.cargo/config.toml").read_bytes()
            ),
            "rust/crates/tokspur-cli/src/main.rs": digest(
                (root / "rust/crates/tokspur-cli/src/main.rs").read_bytes()
            ),
        },
        "profile_digests": {
            package.POLICY: digest(policy.read_bytes()),
            package.HARDWARE_PROFILE: digest(hw.read_bytes()),
            package.ASSETS_PROFILE: digest(assets.read_bytes()),
        },
        "artifacts": {
            "build/rust-fast-math/release/tokspur": artifact_rec(
                rust_build / "tokspur"
            ),
            "build/rust-fast-math/release/tokspur-server": artifact_rec(
                rust_build / "tokspur-server"
            ),
            "build/owned-native/tokspur-native": artifact_rec(build / "tokspur-native"),
            "build/owned-native/tokspur-owned.metallib": artifact_rec(
                build / "tokspur-owned.metallib"
            ),
        },
        "system_dependencies": {
            "build/rust-fast-math/release/tokspur": ["/usr/lib/libSystem.B.dylib"],
            "build/rust-fast-math/release/tokspur-server": [
                "/usr/lib/libSystem.B.dylib"
            ],
            "build/owned-native/tokspur-native": [
                "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
                "/usr/lib/libSystem.B.dylib",
            ],
        },
    }
    receipt = base / "build-manifest.json"
    args = {
        "source_root": root,
        "build": build,
        "rust_build": rust_build,
        "receipt": receipt,
        "receipt_sha256": "",
        "policy": policy,
        "policy_sha256": digest(policy.read_bytes()),
        "output": base / "staged-rust-package",
        "control_plane": "rust",
    }

    def save():
        raw = (json.dumps(manifest) + "\n").encode()
        write(receipt, raw)
        args["receipt_sha256"] = digest(raw)

    save()
    return args, manifest, save


def test_schema2_rust_control_plane_stages_self_contained_layout(package, rust_case):
    assert package.DEFAULT_CLI_CONTROL_PLANE == "rust"
    args, manifest, _save = rust_case
    res = package.stage(**args)
    output = args["output"]
    expected = {
        "tokspur",
        "tokspur-server",
        "tokspur-native",
        "tokspur-owned.metallib",
        "owned-target-experimental.json",
        "m5-max-128.json",
        "flash-next-assets.json",
    }
    assert set(p.name for p in output.iterdir()) == expected | {package.MANIFEST}
    prov = json.loads((output / package.MANIFEST).read_bytes())
    assert prov["schema"] == 2
    assert prov["control_plane"] == "rust"
    assert res["provenance_sha256"] == digest((output / package.MANIFEST).read_bytes())
    assert "tokspur-owned.metallib" not in prov["system_dependencies"]
    assert set(prov["system_dependencies"]) == {
        "tokspur",
        "tokspur-server",
        "tokspur-native",
    }
    assert prov["proof"]["rust_sources"] == manifest["rust_sources"]
    assert prov["proof"]["native_sources"] == manifest["native_sources"]
    assert prov["proof"]["native_source_scope"] == package.NATIVE_SOURCE_SCOPE
    for name in ("tokspur", "tokspur-server", "tokspur-native"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o700
    for name in expected - {"tokspur", "tokspur-server", "tokspur-native"}:
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600


def test_schema2_stage_embeds_every_distribution_notice(package, rust_case):
    args, _manifest, _save = rust_case
    package.stage(**args)
    notices = json.loads((args["output"] / package.MANIFEST).read_bytes())["proof"][
        "notices"
    ]
    assert notices == {
        name: (args["source_root"] / name).read_text() for name in package.NOTICE_FILES
    }


def test_actual_distribution_notices_leave_room_in_the_provenance_bound(package):
    root = Path(__file__).resolve().parents[1]
    notices = {
        name: (root / name).read_bytes().decode() for name in package.NOTICE_FILES
    }
    payload = json.dumps({"proof": {"notices": notices}}).encode()
    assert len(payload) < package.MAX_RECEIPT // 2


@pytest.mark.parametrize(
    "name",
    [
        "NOTICE",
        "native/third_party/omlx/LICENSE",
        "third_party/model-licences/Qwen3.8-Flash-Next/LICENSE",
    ],
)
def test_staging_refuses_a_missing_notice(package, rust_case, name):
    args, _manifest, _save = rust_case
    (args["source_root"] / name).unlink()
    with pytest.raises((OSError, RuntimeError)):
        package.stage(**args)
    assert not args["output"].exists()


def test_staging_refuses_notice_changes_during_copy(package, rust_case, monkeypatch):
    args, _manifest, _save = rust_case
    original = package.copy_file

    def changed(*positional, **keywords):
        result = original(*positional, **keywords)
        (args["source_root"] / "NOTICE").write_text("Changed during staging\n")
        return result

    monkeypatch.setattr(package, "copy_file", changed)
    with pytest.raises(RuntimeError, match="changed during staging"):
        package.stage(**args)
    assert not (args["output"] / package.MANIFEST).exists()


def test_schema2_documentation_edits_do_not_change_admission(package, rust_case):
    args, manifest, _save = rust_case
    root = args["source_root"]
    (root / "native/PROVENANCE.md").write_bytes(b"Edited provenance\n")
    (root / "native/README.md").write_bytes(b"Edited notes\n")
    (root / "native/NEW-NOTES.md").write_bytes(b"Added notes\n")
    (root / "native/src/scratch.cpp.orig").write_bytes(b"editor leftover\n")
    (root / "native/.DS_Store").write_bytes(b"finder state\n")
    package.stage(**args)
    prov = json.loads((args["output"] / package.MANIFEST).read_bytes())
    assert prov["proof"]["native_sources"] == manifest["native_sources"]
    # Staging reads each notice itself, so the packaged text is the edited one.
    assert prov["proof"]["notices"]["native/PROVENANCE.md"] == "Edited provenance\n"


@pytest.mark.parametrize("change", ["edit", "add", "remove", "licence"])
def test_schema2_build_input_changes_are_refused(package, rust_case, change):
    args, _manifest, _save = rust_case
    root = args["source_root"]
    if change == "edit":
        (root / "native/src/owned.cpp").write_bytes(b"// edited source\n")
    elif change == "add":
        (root / "native/src/added.mm").write_bytes(b"// added source\n")
    elif change == "remove":
        (root / "native/Makefile").unlink()
    else:
        (root / "native/data/UNICODE-LICENSE.txt").write_bytes(b"Edited licence\n")
    with pytest.raises(RuntimeError, match="Native source pins differ"):
        package.stage(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("scope", ["absent", "older", "other", "wrong_type"])
def test_schema2_refuses_manifest_from_another_native_scope(package, rust_case, scope):
    args, manifest, _save = rust_case
    if scope == "absent":
        del manifest["native_source_scope"]
    elif scope == "older":
        # A manifest recorded before the scope existed also pinned every note.
        del manifest["native_source_scope"]
        manifest["native_sources"] = {
            p.relative_to(args["source_root"]).as_posix(): digest(p.read_bytes())
            for p in sorted((args["source_root"] / "native").rglob("*"))
            if p.is_file()
        }
    elif scope == "other":
        manifest["native_source_scope"] = "everything-v0"
    else:
        manifest["native_source_scope"] = [package.NATIVE_SOURCE_SCOPE]
    raw = (json.dumps(manifest) + "\n").encode()
    args["receipt"].write_bytes(raw)
    args["receipt_sha256"] = digest(raw)
    with pytest.raises(RuntimeError, match="native_source_scope") as refused:
        package.stage(**args)
    assert "run scripts/bootstrap.py" in str(refused.value)
    assert not args["output"].exists()


def test_cpp_control_plane_proof_still_pins_every_native_file(package, case):
    # The older regression receipt hashes all of native/, notes included, and
    # keeps doing so: only the schema 2 build manifest pins build inputs alone.
    args, _proof, _save = case
    (args["source_root"] / "native/NOTES.md").write_bytes(b"Added after the proof\n")
    with pytest.raises(RuntimeError, match="Native source pins differ"):
        package.stage(**args)
    assert not args["output"].exists()


def test_schema2_refuses_metallib_in_otool_or_bad_mtlb_or_unrecorded_scalar(
    package, rust_case
):
    args, manifest, save = rust_case
    # 1. Metallib listed in system_dependencies must be refused
    bad = copy.deepcopy(manifest)
    bad["system_dependencies"]["build/owned-native/tokspur-owned.metallib"] = [
        "/usr/lib/libSystem.B.dylib"
    ]
    raw = (json.dumps(bad) + "\n").encode()
    args["receipt"].write_bytes(raw)
    args["receipt_sha256"] = digest(raw)
    with pytest.raises(RuntimeError, match="must not appear in Mach-O"):
        package.stage(**args)

    # 2. Unrecorded scalar-channel-target in build/ must be refused
    save()
    scalar = args["build"] / "scalar-channel-target"
    scalar.write_bytes(b"\xcf\xfa\xed\xfe\x00\x00\x00\x00")
    scalar.chmod(0o700)
    with pytest.raises(RuntimeError, match="scalar-channel-target"):
        package.stage(**args)
    scalar.unlink()

    # 3. Metallib lacking MTLB magic must be refused
    metallib = args["build"] / "tokspur-owned.metallib"
    metallib.write_bytes(b"NOPE\x01\x00\x02\x00fake-metal")
    manifest["artifacts"]["build/owned-native/tokspur-owned.metallib"]["sha256"] = (
        digest(metallib.read_bytes())
    )
    save()
    with pytest.raises(RuntimeError, match="MTLB"):
        package.stage(**args)


def test_diagnostic_and_probe_artefacts_are_not_production_inputs(package, tmp_path):
    package.admit_serving_config(
        {
            "native_config": {"NATIVE_CONFIG": "carrier"},
            "tracked_cflags": ["carrier", ":", "test flags"],
        }
    )
    with pytest.raises(RuntimeError, match="carrier build"):
        package.admit_serving_config(
            {
                "native_config": {"NATIVE_CONFIG": "carrier-diagnostic"},
                "tracked_cflags": ["carrier-diagnostic", ":", "test flags"],
            }
        )
    for name, data in [
        ("tokspur-native", b"tokspur-carrier-diagnostic-v1"),
        ("tokspur-owned.metallib", b"MTLBprobe_unchecked"),
        ("tokspur-owned.metallib", b"MTLBgraph_capture_routes"),
    ]:
        file = tmp_path / name
        file.write_bytes(data)
        with pytest.raises(RuntimeError, match="Diagnostic or probe"):
            package.file_digest(file)
