#!/usr/bin/env python3
"""Stage five pinned native-only files; never build, install, or launch them."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import stat
from pathlib import Path

ARTIFACTS = ("tokspur-native", "tokspur-server", "tokspur-owned.metallib")
POLICY = "profiles/owned-target-experimental.json"
HARDWARE_PROFILE = "profiles/m5-max-128.json"
ASSETS_PROFILE = "profiles/flash-next-assets.json"
MANIFEST = "provenance.json"
NOTICE_FILES = (
    "LICENSE",
    "NOTICE",
    "MODEL_LICENCES.md",
    "native/PROVENANCE.md",
    "native/third_party/sources.json",
    "native/data/UNICODE-LICENSE.txt",
    "native/third_party/llama-cpp/LICENSE",
    "native/third_party/mlx-vlm/LICENSE",
    "native/third_party/omlx/LICENSE",
    "native/third_party/m5-max-kernels/LICENSE",
    "native/third_party/splash/LICENSE",
    "native/third_party/splash/THIRD_PARTY_NOTICES",
    "third_party/model-licences/Qwen3.8-Flash-Next/LICENSE",
    "third_party/rust/COPYRIGHT-library.html",
)
DEFAULT_CLI_CONTROL_PLANE = "rust"
MACHO_MAGICS = (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe")
MTLB_MAGIC = b"MTLB"
MAX_FILE = 512 << 20
MAX_RECEIPT = 4 << 20
MAX_SOURCE = 16 << 20
MAX_SOURCES = 2048


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest_string(value):
    require(
        isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value),
        "Explicit lowercase SHA-256 required",
    )
    return value


def absolute(value):
    path = Path(value)
    require(
        path.is_absolute() and ".." not in path.parts and "\0" not in str(path),
        "Absolute path without parent traversal required",
    )
    return path


def identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
        value.st_nlink,
    )


@contextlib.contextmanager
def directory(path):
    """Walk descriptors, refusing symlinks in every component, including parents."""
    path = absolute(path)
    descriptors, links = [], []
    try:
        current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(current)
        for name in path.parts[1:]:
            child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            descriptors.append(child)
            opened = os.fstat(child)
            links.append((current, name, opened.st_dev, opened.st_ino))
            current = child

        def unchanged():
            for parent, name, device, inode in links:
                now = os.stat(name, dir_fd=parent, follow_symlinks=False)
                require(
                    stat.S_ISDIR(now.st_mode)
                    and (now.st_dev, now.st_ino) == (device, inode),
                    "Directory ancestry changed",
                )

        unchanged()
        yield current, unchanged
        unchanged()
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@contextlib.contextmanager
def opened_file(path, limit=MAX_FILE):
    path = absolute(path)
    with directory(path.parent) as (parent, parents_unchanged):
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            before = os.fstat(descriptor)
            require(
                stat.S_ISREG(before.st_mode)
                and before.st_nlink == 1
                and 0 < before.st_size <= limit
                and not before.st_mode & 0o6022,
                "Input must be bounded, regular, single-link, "
                "and not group/world writable",
            )

            def unchanged():
                parents_unchanged()
                require(
                    identity(os.fstat(descriptor)) == identity(before)
                    and identity(
                        os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                    )
                    == identity(before),
                    "Input changed while staging",
                )

            unchanged()
            yield descriptor, before, unchanged
            unchanged()
        finally:
            os.close(descriptor)


def read_file(path, limit=MAX_SOURCE):
    with opened_file(path, limit) as (descriptor, _before, unchanged):
        chunks, size = [], 0
        while chunk := os.read(descriptor, 1 << 20):
            size += len(chunk)
            require(size <= limit, "Input grew past its admission bound")
            chunks.append(chunk)
        unchanged()
        return b"".join(chunks)


def file_digest(path, limit=MAX_FILE, *, entry=None):
    with opened_file(path, limit) as (descriptor, before, unchanged):
        if entry is not None:
            require(
                before.st_uid == os.geteuid()
                and stat.S_IMODE(before.st_mode) == int(entry["mode"], 8)
                and before.st_size == entry["size"],
                "Staged entry ownership, mode, or size differs",
            )
        result, size, marker_tail = hashlib.sha256(), 0, b""
        while chunk := os.read(descriptor, 1 << 20):
            size += len(chunk)
            require(size <= limit, "Input grew past its admission bound")
            result.update(chunk)
            if path.name in ARTIFACTS or path.name.startswith("scalar-channel-"):
                serving_marker(marker_tail + chunk, path.suffix == ".metallib")
                marker_tail = chunk[-64:]
        unchanged()
        return result.hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_value):
        raise RuntimeError("Non-finite JSON value")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


# The build-input rule of scripts/bootstrap.py and rust/crates/tokspur-cli/src/verify.rs, copied here because this script stays standalone. A schema 2 build-manifest.json pins only these files under native/ and records NATIVE_SOURCE_SCOPE beside them.
# tests/test_native_build_inputs.py compares the three copies. Bump NATIVE_SOURCE_SCOPE in all three in the same change as any edit to a list.
NATIVE_SOURCE_SCOPE = "build-inputs-v2"
NATIVE_BUILD_INPUT_SUFFIXES = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
    ".m",
    ".mm",
    ".metal",
    ".mk",
    ".json",
    ".patch",
    ".LICENSE",
)
NATIVE_BUILD_INPUT_NAMES = (
    "Makefile",
    "GNUmakefile",
    "makefile",
    "LICENSE",
    "UNICODE-LICENSE.txt",
    "THIRD_PARTY_NOTICES",
)


def is_native_build_input(name):
    return name in NATIVE_BUILD_INPUT_NAMES or name.endswith(
        NATIVE_BUILD_INPUT_SUFFIXES
    )


def native_sources(root, *, build_inputs_only=False):
    """Digest regular files under native/.

    The default hashes every file, as the older regression receipts do. The schema 2 build manifest pins build inputs only, so admit_rust passes build_inputs_only. Symlinks and other entries are refused either way.
    """
    result = {}

    def visit(path):
        with directory(path) as (descriptor, unchanged):
            for name in sorted(os.listdir(descriptor)):
                value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                child = path / name
                if stat.S_ISDIR(value.st_mode):
                    visit(child)
                else:
                    require(stat.S_ISREG(value.st_mode), "Unsafe native source entry")
                    if build_inputs_only and not is_native_build_input(name):
                        continue
                    require(len(result) < MAX_SOURCES, "Native source inventory bound")
                    result[str(child.relative_to(root))] = file_digest(
                        child, MAX_SOURCE
                    )
            unchanged()

    visit(root / "native")
    require(result, "Empty native source inventory")
    return result


def rust_sources(root):
    result = {}
    for rel in (
        "rust/Cargo.toml",
        "rust/Cargo.lock",
        "rust/rust-toolchain.toml",
        "rust/.cargo/config.toml",
        "rust/clippy.toml",
    ):
        candidate = root / rel
        with directory(candidate.parent) as (parent_fd, parent_unchanged):
            if candidate.name in os.listdir(parent_fd):
                result[rel] = file_digest(candidate, MAX_SOURCE)
            parent_unchanged()

    def visit_crates(path):
        with directory(path) as (descriptor, unchanged):
            for name in sorted(os.listdir(descriptor)):
                if name == ".DS_Store":
                    continue
                value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                child = path / name
                if stat.S_ISDIR(value.st_mode):
                    if name == "target":
                        continue
                    visit_crates(child)
                else:
                    require(stat.S_ISREG(value.st_mode), "Unsafe Rust source entry")
                    require(len(result) < MAX_SOURCES, "Rust source inventory bound")
                    result[child.relative_to(root).as_posix()] = file_digest(
                        child, MAX_SOURCE
                    )
            unchanged()

    visit_crates(root / "rust/crates")
    require(result, "Empty Rust source inventory")
    return result


def suites(checker):
    """Read only the producer's two literal inventories; never import/execute it."""
    assignments = {
        node.targets[0].id: node.value
        for node in ast.parse(checker).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    result = []
    for name in ("CPU_SUITES", "METAL_SUITES"):
        node = assignments.get(name)
        require(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "tuple"
            and len(node.args) == 1
            and not node.keywords,
            "Unknown owned proof inventory syntax",
        )
        node = node.args[0]
        require(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "split"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
            and not node.args
            and not node.keywords,
            "Nonliteral owned proof inventory",
        )
        names = node.func.value.value.split()
        require(
            names and all(re.fullmatch("[a-z0-9_]+", item) for item in names),
            "Unsafe owned proof suite name",
        )
        result.append(tuple(names))
    require(len(set(sum(result, ()))) == len(sum(result, ())), "Duplicate suite")
    return tuple(result)


def system_links(output, binary):
    lines = output.splitlines()
    require(lines and lines[0] == str(binary) + ":", "Wrong link audit executable")
    dependencies = []
    for line in lines[1:]:
        match = re.fullmatch(
            r"\s+(/[^\s]+) \(compatibility version [^()]+, current version [^()]+\)",
            line,
        )
        require(match is not None, "Malformed link audit")
        path = match[1]
        require(
            path.startswith(("/System/Library/", "/usr/lib/"))
            and ".." not in Path(path).parts,
            "Non-system executable dependency",
        )
        dependencies.append(path)
    require(dependencies, "Empty link audit")
    return dependencies


def admit(root, build, receipt_path, receipt_sha256, policy, policy_sha256):
    raw = read_file(receipt_path, MAX_RECEIPT)
    require(
        sha256(raw) == digest_string(receipt_sha256), "Build receipt digest differs"
    )
    receipt = strict_json(raw)
    require(
        isinstance(receipt, dict)
        and type(receipt.get("schema")) is int
        and receipt["schema"] == 1
        and receipt.get("pass") is True
        and "error" not in receipt
        and receipt.get("system_only_component_link_check") is True,
        "Passing owned regression receipt required",
    )
    checker = read_file(root / "scripts/check_owned_gpu_experiments.py")
    protocol = read_file(root / "scripts/check_owned_target_protocol.py")
    require(sha256(checker) == receipt.get("harness_sha256"), "Proof producer changed")
    require(
        sha256(protocol) == receipt.get("protocol_checker_sha256"),
        "Protocol checker changed",
    )
    cpu, metal = suites(checker)
    names = ARTIFACTS + tuple("test-" + name for name in cpu + metal)
    sources = native_sources(root)
    require(sources == receipt.get("native_sources"), "Native source pins differ")
    artifacts = {name: file_digest(build / name) for name in names}
    require(artifacts == receipt.get("artifacts"), "Exact artifact inventory differs")
    commands = receipt.get("commands")
    require(
        isinstance(commands, list) and len(commands) == 6 + len(cpu) + len(metal),
        "Incomplete regression command inventory",
    )
    require(
        isinstance(commands[5], dict)
        and isinstance(commands[5].get("argv"), list)
        and len(commands[5]["argv"]) == 2
        and isinstance(commands[5]["argv"][0], str)
        and bool(commands[5]["argv"][0]),
        "Missing protocol interpreter identity",
    )
    expected = [
        ["c++", "--version"],
        ["xcrun", "metal", "--version"],
        [str(build / ARTIFACTS[0]), "host-info"],
        ["otool", "-L", str(build / ARTIFACTS[0])],
        ["otool", "-L", str(build / ARTIFACTS[1])],
        [commands[5]["argv"][0], str(root / "scripts/check_owned_target_protocol.py")],
        *[[str(build / ("test-" + name))] for name in cpu],
        *[[str(build / ("test-" + name)), str(build / ARTIFACTS[2])] for name in metal],
    ]
    for command, argv in zip(commands, expected, strict=True):
        require(
            isinstance(command, dict)
            and command.get("argv") == argv
            and type(command.get("returncode")) is int
            and command["returncode"] == 0
            and isinstance(command.get("stdout"), str)
            and isinstance(command.get("stderr"), str),
            "Missing, reordered, or failed regression command",
        )
    require(
        re.findall(
            r"^Owned CPU target protocol checks passed: (.*)$",
            commands[5]["stdout"],
            re.M,
        )
        == ["245"]
        and re.findall(
            r"^Owned CPU target stack-experiment protocol checks passed: (.*)$",
            commands[5]["stdout"],
            re.M,
        )
        == ["59"],
        "Incomplete target protocol proof",
    )
    links = {
        name: system_links(commands[index]["stdout"], build / name)
        for index, name in ((3, ARTIFACTS[0]), (4, ARTIFACTS[1]))
    }
    policy_bytes = read_file(policy)
    require(
        sha256(policy_bytes) == digest_string(policy_sha256)
        and policy_bytes == read_file(root / POLICY),
        "Only the explicitly pinned declared owned policy may be staged",
    )
    require(
        isinstance(strict_json(policy_bytes), dict), "Owned policy must be an object"
    )
    notices = {name: read_file(root / name).decode("utf-8") for name in NOTICE_FILES}
    return {
        "native_sources": sources,
        "artifacts": artifacts,
        "build_receipt_sha256": sha256(raw),
        "proof_producer_sha256": sha256(checker),
        "protocol_checker_sha256": sha256(protocol),
        "policy_sha256": sha256(policy_bytes),
        "system_dependencies": links,
        "notices": notices,
    }


def admit_serving_config(receipt):
    flags = receipt.get("tracked_cflags")
    require(
        isinstance(receipt.get("native_config"), dict)
        and receipt["native_config"].get("NATIVE_CONFIG") == "carrier"
        and isinstance(flags, list)
        and flags
        and flags[0] == "carrier"
        and all(
            isinstance(flag, str) and "CARRIER_DIAGNOSTIC" not in flag for flag in flags
        ),
        "Production staging requires the carrier build",
    )


def serving_marker(data, is_metallib):
    markers = (
        (b"probe_unchecked", b"graph_capture_routes")
        if is_metallib
        else (b"tokspur-carrier-diagnostic-v1",)
    )
    require(
        all(marker not in data for marker in markers),
        "Diagnostic or probe artefact refused by production staging",
    )


def inspect_artifact(path, *, is_metallib):
    with opened_file(path, MAX_FILE) as (descriptor, before, unchanged):
        require(before.st_size >= 4, "Artifact smaller than header magic")
        header, size, tail = b"", 0, b""
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            if not header:
                header = chunk[:4]
            size += len(chunk)
            require(size <= MAX_FILE, "Input grew past its admission bound")
            digest.update(chunk)
            if is_metallib or path.name != "tokspur":
                serving_marker(tail + chunk, is_metallib)
            tail = chunk[-64:]
        unchanged()
        if is_metallib:
            require(
                not (before.st_mode & 0o111) and header == MTLB_MAGIC,
                "tokspur-owned.metallib must be non-executable with MTLB header magic",
            )
        else:
            require(
                bool(before.st_mode & 0o111) and header[:4] in MACHO_MAGICS,
                "Executable must have execute bit and valid Mach-O magic",
            )
        return digest.hexdigest(), size


def check_system_dep_list(deps):
    require(
        isinstance(deps, list) and bool(deps),
        "Non-empty Mach-O system_dependencies list required",
    )
    for item in deps:
        require(
            isinstance(item, str)
            and item.startswith(("/System/Library/", "/usr/lib/"))
            and ".." not in Path(item).parts,
            "Non-system executable dependency",
        )
    return list(deps)


def admit_rust(
    root, build, rust_build, receipt_path, receipt_sha256, policy, policy_sha256
):
    raw = read_file(receipt_path, MAX_RECEIPT)
    require(
        sha256(raw) == digest_string(receipt_sha256), "Build receipt digest differs"
    )
    receipt = strict_json(raw)
    require(
        isinstance(receipt, dict)
        and type(receipt.get("schema")) is int
        and receipt["schema"] == 2
        and receipt.get("enable_metal") is True
        and receipt.get("frozen") is True
        and type(receipt.get("metallib_function_count")) is int
        and receipt["metallib_function_count"] >= 1,
        "Passing schema-2 Metal build-manifest receipt required",
    )
    require(
        receipt.get("native_source_scope") == NATIVE_SOURCE_SCOPE,
        f'build-manifest.json native_source_scope "{NATIVE_SOURCE_SCOPE}" is '
        "required (older manifests also pinned native/ documentation); "
        "run scripts/bootstrap.py",
    )
    admit_serving_config(receipt)
    nsrc = native_sources(root, build_inputs_only=True)
    rsrc = rust_sources(root)
    require(nsrc == receipt.get("native_sources"), "Native source pins differ")
    require(rsrc == receipt.get("rust_sources"), "Rust source pins differ")

    rust_targets = receipt.get("rust_targets")
    require(
        isinstance(rust_targets, dict)
        and rust_targets.get("tokspur-server") == ["fast-math"]
        and rust_targets.get("tokspur") == [],
        "build-manifest rust_targets mismatch",
    )

    rec_artifacts = receipt.get("artifacts")
    rec_sys_deps = receipt.get("system_dependencies")
    require(
        isinstance(rec_artifacts, dict) and isinstance(rec_sys_deps, dict),
        "build-manifest missing artifacts or system_dependencies",
    )
    metallib_rel = "build/owned-native/tokspur-owned.metallib"
    require(
        metallib_rel not in rec_sys_deps,
        "tokspur-owned.metallib must not appear in Mach-O system_dependencies",
    )

    required_files = [
        (
            "tokspur",
            rust_build / "tokspur",
            "build/rust-fast-math/release/tokspur",
            False,
        ),
        (
            "tokspur-server",
            rust_build / "tokspur-server",
            "build/rust-fast-math/release/tokspur-server",
            False,
        ),
        (
            "tokspur-native",
            build / "tokspur-native",
            "build/owned-native/tokspur-native",
            False,
        ),
        (
            "tokspur-owned.metallib",
            build / "tokspur-owned.metallib",
            metallib_rel,
            True,
        ),
    ]
    scalar_rel = "build/owned-native/scalar-channel-target"
    if scalar_rel in rec_artifacts:
        required_files.insert(
            3,
            (
                "scalar-channel-target",
                build / "scalar-channel-target",
                scalar_rel,
                False,
            ),
        )
    else:
        with directory(build) as (build_fd, build_unchanged):
            require(
                "scalar-channel-target" not in os.listdir(build_fd),
                "Unrecorded scalar-channel-target present in build directory",
            )
            build_unchanged()

    verified_artifacts = {}
    staged_links = {}
    for staged_name, full_path, rel_key, is_metallib in required_files:
        entry = rec_artifacts.get(rel_key)
        require(isinstance(entry, dict), f"Missing artifact entry: {rel_key}")
        actual_sha, actual_size = inspect_artifact(full_path, is_metallib=is_metallib)
        require(
            actual_sha == entry.get("sha256") and actual_size == entry.get("size"),
            f"Artifact digest or size differs: {staged_name}",
        )
        verified_artifacts[staged_name] = {
            "source": full_path,
            "sha256": actual_sha,
            "size": actual_size,
            "is_metallib": is_metallib,
        }
        if not is_metallib:
            staged_links[staged_name] = check_system_dep_list(rec_sys_deps.get(rel_key))

    policy_bytes = read_file(policy)
    require(
        sha256(policy_bytes) == digest_string(policy_sha256)
        and policy_bytes == read_file(root / POLICY),
        "Only the explicitly pinned declared owned policy may be staged",
    )
    require(
        isinstance(strict_json(policy_bytes), dict), "Owned policy must be an object"
    )

    rec_profiles = receipt.get("profile_digests")
    require(isinstance(rec_profiles, dict), "Missing profile_digests in receipt")
    hw_bytes = read_file(root / HARDWARE_PROFILE)
    assets_bytes = read_file(root / ASSETS_PROFILE)
    require(
        rec_profiles.get(POLICY) == sha256(policy_bytes)
        and rec_profiles.get(HARDWARE_PROFILE) == sha256(hw_bytes)
        and rec_profiles.get(ASSETS_PROFILE) == sha256(assets_bytes),
        "Profile digests differ from build receipt",
    )
    notices = {name: read_file(root / name).decode("utf-8") for name in NOTICE_FILES}
    return {
        "verified_artifacts": verified_artifacts,
        "system_dependencies": staged_links,
        "profiles": {
            Path(POLICY).name: (policy, sha256(policy_bytes)),
            Path(HARDWARE_PROFILE).name: (root / HARDWARE_PROFILE, sha256(hw_bytes)),
            Path(ASSETS_PROFILE).name: (root / ASSETS_PROFILE, sha256(assets_bytes)),
        },
        "proof_record": {
            "git_head": receipt.get("git_head"),
            "git_dirty": receipt.get("git_dirty"),
            "native_source_scope": NATIVE_SOURCE_SCOPE,
            "native_sources": nsrc,
            "rust_sources": rsrc,
            "rust_targets": rust_targets,
            "metallib_function_count": receipt["metallib_function_count"],
            "build_receipt_sha256": sha256(raw),
            "policy_sha256": sha256(policy_bytes),
            "notices": notices,
        },
    }


def copy_file(source, output, name, expected, mode):
    with opened_file(source) as (descriptor, before, unchanged):
        if mode == 0o700:
            require(before.st_mode & 0o111, "Owned executable lacks execute permission")
        destination = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode,
            dir_fd=output,
        )
        try:
            result, size = hashlib.sha256(), 0
            while chunk := os.read(descriptor, 1 << 20):
                size += len(chunk)
                require(size <= MAX_FILE, "Artifact grew past bound")
                result.update(chunk)
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(destination, remaining)
                    require(written > 0, "Short package write")
                    remaining = remaining[written:]
            unchanged()
            require(result.hexdigest() == expected, "Copied input digest differs")
            os.fchmod(destination, mode)
            os.fsync(destination)
        finally:
            os.close(destination)
    return {"sha256": expected, "size": size, "mode": oct(mode)}


def stage(
    *,
    source_root,
    build,
    receipt,
    receipt_sha256,
    policy,
    policy_sha256,
    output,
    control_plane="cpp",
    rust_build=None,
):
    require(
        control_plane in ("cpp", "rust"),
        "Unsupported control_plane (expected cpp or rust)",
    )
    root, build, receipt, policy, output = map(
        absolute, (source_root, build, receipt, policy, output)
    )
    resolved_rust_build = (
        absolute(rust_build)
        if rust_build is not None
        else (root / "build/rust-fast-math/release")
    )
    require(output.name not in ("", "/"), "New named output directory required")
    # Never allow creation inside an input tree or modification of the source proof.
    require(
        not output.is_relative_to(root)
        and not root.is_relative_to(output)
        and not output.is_relative_to(build)
        and not build.is_relative_to(output)
        and not receipt.is_relative_to(output)
        and not policy.is_relative_to(output)
        and (
            control_plane == "cpp"
            or (
                not output.is_relative_to(resolved_rust_build)
                and not resolved_rust_build.is_relative_to(output)
            )
        ),
        "Output must be separate from all input trees",
    )
    if control_plane == "cpp":
        proof = admit(root, build, receipt, receipt_sha256, policy, policy_sha256)
    else:
        rust_admitted = admit_rust(
            root,
            build,
            resolved_rust_build,
            receipt,
            receipt_sha256,
            policy,
            policy_sha256,
        )
    with directory(output.parent) as (parent, parent_unchanged):
        os.mkdir(output.name, mode=0o700, dir_fd=parent)
        created = os.stat(output.name, dir_fd=parent, follow_symlinks=False)
        with directory(output) as (destination, destination_unchanged):

            def private_output():
                current = os.fstat(destination)
                require(
                    (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino)
                    and current.st_uid == os.geteuid()
                    and stat.S_IMODE(current.st_mode) == 0o700,
                    "New output identity or private permissions changed",
                )

            private_output()
            entries = {}
            if control_plane == "cpp":
                for name in ARTIFACTS:
                    mode = 0o600 if name.endswith(".metallib") else 0o700
                    entries[name] = copy_file(
                        build / name, destination, name, proof["artifacts"][name], mode
                    )
                policy_name = Path(POLICY).name
                entries[policy_name] = copy_file(
                    policy, destination, policy_name, proof["policy_sha256"], 0o600
                )
                # Re-admit the receipt, every source/artifact, policy and notice after copy.
                require(
                    admit(root, build, receipt, receipt_sha256, policy, policy_sha256)
                    == proof,
                    "Build proof changed during staging",
                )
                manifest = {
                    "schema": 1,
                    "scope": (
                        "private model-free native staging; "
                        "not an installer or release qualification"
                    ),
                    "entries": entries,
                    "proof": proof,
                    "stage_script_sha256": file_digest(Path(__file__).absolute()),
                    "runtime": (
                        "Apple system libraries and owned Metal library; "
                        "no Python or external engine"
                    ),
                    "assets": (
                        "Separate verified original four GGUF shards and pinned "
                        "tokenizer/template required; no adoption or assets included"
                    ),
                    "launch": (
                        "Explicit manual native server launch only; "
                        "no install/startup hook, fallback, service, or tool executor"
                    ),
                    "qualification": (
                        "Recorded otool -L admission is not compiler attestation, "
                        "signing, reproducibility, or operational/model qualification"
                    ),
                }
            else:
                for name, meta in rust_admitted["verified_artifacts"].items():
                    mode = 0o600 if meta["is_metallib"] else 0o700
                    entries[name] = copy_file(
                        meta["source"], destination, name, meta["sha256"], mode
                    )
                for prof_name, (prof_path, prof_sha) in rust_admitted[
                    "profiles"
                ].items():
                    entries[prof_name] = copy_file(
                        prof_path, destination, prof_name, prof_sha, 0o600
                    )
                require(
                    admit_rust(
                        root,
                        build,
                        resolved_rust_build,
                        receipt,
                        receipt_sha256,
                        policy,
                        policy_sha256,
                    )
                    == rust_admitted,
                    "Build proof changed during staging",
                )
                manifest = {
                    "schema": 2,
                    "control_plane": "rust",
                    "scope": (
                        "private model-free Rust CLI/server + native Metal worker staging; "
                        "not an installer or release qualification"
                    ),
                    "entries": entries,
                    "system_dependencies": rust_admitted["system_dependencies"],
                    "proof": rust_admitted["proof_record"],
                    "stage_script_sha256": file_digest(Path(__file__).absolute()),
                    "runtime": (
                        "Apple system libraries, Rust control plane and owned Metal library; "
                        "no Python or external engine"
                    ),
                    "state_root": (
                        "Requires --state-root <abs-path> or TOKSPUR_STATE_ROOT "
                        "outside the immutable package directory"
                    ),
                }

            for name, entry in entries.items():
                require(
                    file_digest(output / name, entry=entry) == entry["sha256"],
                    "Staged entry digest differs",
                )
            require(
                set(os.listdir(destination)) == set(entries),
                "Unexpected entry before manifest publication",
            )
            private_output()
            payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            descriptor = os.open(
                MANIFEST,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=destination,
            )
            try:
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(descriptor, remaining)
                    require(written > 0, "Short manifest write")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            require(
                set(os.listdir(destination)) == set(entries) | {MANIFEST}
                and file_digest(
                    output / MANIFEST,
                    entry={"mode": "0o600", "size": len(payload)},
                )
                == sha256(payload),
                "Package contents changed before publication",
            )
            private_output()
            destination_unchanged()
            parent_unchanged()
            os.fsync(destination)
        os.fsync(parent)
    return {
        "output": str(output),
        "provenance_sha256": sha256(payload),
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "build", "receipt", "policy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("receipt-sha256", "policy-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--control-plane",
        choices=("cpp", "rust"),
        default=DEFAULT_CLI_CONTROL_PLANE,
    )
    parser.add_argument("--rust-build", type=Path, default=None)
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(stage(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
