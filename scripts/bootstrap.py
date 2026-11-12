#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build pinned sources; never download weights or start inference."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tokspur.local import write_json  # noqa: E402

MACHO_MAGICS = (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe")
MTLB_MAGIC = b"MTLB"


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    extra_dirs = [
        str(Path.home() / ".local/share/mise/shims"),
        "/opt/homebrew/bin",
        str(Path.home() / ".cargo/bin"),
    ]
    current = env.get("PATH", "")
    parts = [p for p in current.split(":") if p]
    for d in reversed(extra_dirs):
        if os.path.isdir(d) and d not in parts:
            parts.insert(0, d)
    env["PATH"] = ":".join(parts)
    flags = effective_rustflags(env)
    flags.extend(
        [
            f"--remap-path-prefix={Path.home()}=/build-user",
            f"--remap-path-prefix={ROOT}=/tokspur",
        ]
    )
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    env.pop("RUSTFLAGS", None)
    return env


def effective_rustflags(env: dict[str, str]) -> list[str]:
    encoded = env.get("CARGO_ENCODED_RUSTFLAGS")
    if encoded is not None:
        return encoded.split("\x1f") if encoded else []
    return env.get("RUSTFLAGS", "").split()


def public_rustflags(env: dict[str, str]) -> str:
    return (
        " ".join(effective_rustflags(env))
        .replace(str(ROOT), "$CHECKOUT")
        .replace(str(Path.home()), "$HOME")
    )


def run(
    command: list[str], cwd: Path = ROOT, env: dict[str, str] | None = None
) -> None:
    subprocess.run(command, cwd=cwd, env=env or build_env(), check=True)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# Build inputs under native/ that build-manifest.json pins. The manifest records NATIVE_SOURCE_SCOPE beside the digests, so a reader knows which rule produced them.
# A file is pinned when its name is listed or ends with a listed suffix, matched exactly and case-sensitively. Markdown notes and every other unlisted file are not pinned, so editing documentation cannot make `tokspur serve` refuse. Symlinks and non-regular entries are still refused everywhere under native/.
# scripts/stage_owned_native_package.py and rust/crates/tokspur-cli/src/verify.rs carry the same lists. tests/test_native_build_inputs.py compares all three, checks them against tests/fixtures/native-build-inputs.txt and requires every file under native/ to be either a build input or Markdown, so a new file type cannot silently leave the identity.
# Changing a list changes what the pinned source identity means: bump NATIVE_SOURCE_SCOPE here, in the staging script and in verify.rs in the same change.
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


def is_native_build_input(name: str) -> bool:
    """Whether a file with this name under native/ belongs to the pinned identity."""
    return name in NATIVE_BUILD_INPUT_NAMES or name.endswith(
        NATIVE_BUILD_INPUT_SUFFIXES
    )


def collect_tree_digests(
    base: Path,
    skip_dirs: set[str] | None = None,
    keep: Callable[[str], bool] | None = None,
) -> dict[str, str]:
    """Digest regular files below base; keep, when given, filters by file name."""
    skip = skip_dirs or set()
    out: dict[str, str] = {}
    for directory, dirnames, filenames in os.walk(base, followlinks=False):
        dir_path = Path(directory)
        kept = []
        for d in sorted(dirnames):
            if d == ".DS_Store":
                continue
            child_dir = dir_path / d
            if child_dir.is_symlink():
                raise ValueError(f"Unsafe symlink in source tree: {child_dir}")
            if d in skip:
                continue
            kept.append(d)
        dirnames[:] = kept
        for f in sorted(filenames):
            if f == ".DS_Store":
                continue
            child = dir_path / f
            st = child.lstat()
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                raise ValueError(f"Unsafe non-regular source entry: {child}")
            if keep is not None and not keep(f):
                continue
            rel = child.relative_to(ROOT).as_posix()
            out[rel] = sha(child)
    return dict(sorted(out.items()))


def collect_native_sources() -> dict[str, str]:
    res = collect_tree_digests(ROOT / "native", keep=is_native_build_input)
    if not res:
        raise ValueError("Empty native source inventory")
    return res


def collect_rust_sources() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in (
        "rust/Cargo.toml",
        "rust/Cargo.lock",
        "rust/rust-toolchain.toml",
        "rust/.cargo/config.toml",
        "rust/clippy.toml",
    ):
        p = ROOT / rel
        if p.exists():
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                raise ValueError(f"Unsafe Rust config file: {rel}")
            out[rel] = sha(p)
    out.update(collect_tree_digests(ROOT / "rust/crates", skip_dirs={"target"}))
    if not out:
        raise ValueError("Empty Rust source inventory")
    return dict(sorted(out.items()))


def audit_macho_executable(path: Path) -> list[str]:
    st = path.lstat()
    if (
        not stat.S_ISREG(st.st_mode)
        or st.st_nlink != 1
        or st.st_size < 4
        or (st.st_mode & 0o6022) != 0
        or (st.st_mode & 0o111) == 0
    ):
        raise ValueError(f"Unsafe or non-executable binary: {path}")
    with path.open("rb") as f:
        magic = f.read(4)
    if magic not in MACHO_MAGICS:
        raise ValueError(f"Artifact is not a Mach-O executable: {path}")
    out = subprocess.check_output(
        ["/usr/bin/otool", "-L", str(path)], text=True
    ).splitlines()
    if not out or out[0] != f"{path}:":
        raise ValueError(f"Unexpected otool -L header for {path}")
    deps: list[str] = []
    for line in out[1:]:
        trimmed = line.strip()
        if not trimmed:
            continue
        if " (compatibility version " not in trimmed or not trimmed.endswith(")"):
            raise ValueError(f"Malformed otool -L line for {path}: {line}")
        lib_path, rest = trimmed.split(" (compatibility version ", 1)
        if ", current version " not in rest:
            raise ValueError(f"Malformed otool -L version trailer for {path}: {line}")
        if not lib_path.startswith(("/System/Library/", "/usr/lib/")) or (
            ".." in Path(lib_path).parts
        ):
            raise ValueError(f"Non-system executable dependency in {path}: {lib_path}")
        deps.append(lib_path)
    if not deps:
        raise ValueError(f"Empty otool -L dependency list for {path}")
    return deps


def verify_metallib_and_count_functions(path: Path, env: dict[str, str]) -> int:
    st = path.lstat()
    if (
        not stat.S_ISREG(st.st_mode)
        or st.st_nlink != 1
        or st.st_size < 4
        or (st.st_mode & 0o6022) != 0
        or (st.st_mode & 0o111) != 0
    ):
        raise ValueError(
            f"Unsafe or executable metallib (expected non-executable regular file): {path}"
        )
    with path.open("rb") as f:
        magic = f.read(4)
    if magic != MTLB_MAGIC:
        raise ValueError(f"tokspur-owned.metallib lacks valid MTLB magic: {path}")
    objdump = subprocess.check_output(
        ["/usr/bin/xcrun", "metal-objdump", "-t", str(path)],
        text=True,
        env=env,
    )
    fn_lines = [line for line in objdump.splitlines() if "FUNCTION_LIST" in line]
    if not fn_lines:
        raise ValueError("metal-objdump -t returned zero FUNCTION_LIST Metal kernels")
    return len(fn_lines)


def install_scalar_channel_target() -> None:
    native_src = ROOT / "build/owned-native/tokspur-native"
    scalar_dst = ROOT / "build/owned-native/scalar-channel-target"
    scalar_tmp = ROOT / "build/owned-native/.scalar-channel-target.tmp"
    if not native_src.is_file() or native_src.is_symlink():
        raise ValueError(f"Expected regular Mach-O binary: {native_src}")
    if scalar_tmp.is_symlink() or scalar_tmp.exists():
        scalar_tmp.unlink()
    shutil.copyfile(native_src, scalar_tmp)
    os.chmod(scalar_tmp, 0o755)
    if scalar_dst.is_symlink():
        scalar_dst.unlink()
    os.replace(scalar_tmp, scalar_dst)


def install_launcher() -> None:
    venv_bin = ROOT / ".venv/bin"
    launcher_src = ROOT / "scripts/tokspur-launcher.sh"
    if venv_bin.is_dir() and launcher_src.is_file():
        dst = venv_bin / "tokspur"
        if dst.is_symlink():
            dst.unlink()
        shutil.copyfile(launcher_src, dst)
        os.chmod(dst, 0o755)


def record_build_manifest(env: dict[str, str], *, stripped: bool = False) -> dict:
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    clang_version = subprocess.check_output(
        ["/usr/bin/xcrun", "clang++", "--version"], text=True, env=env
    ).strip()
    metal_version = subprocess.check_output(
        ["/usr/bin/xcrun", "metal", "--version"],
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    ).strip()
    rustc_verbose = subprocess.check_output(
        ["rustc", "-vV"], cwd=ROOT / "rust", text=True, env=env
    ).strip()
    cargo_version = subprocess.check_output(
        ["cargo", "--version"], cwd=ROOT / "rust", text=True, env=env
    ).strip()

    target_triple = "aarch64-apple-darwin"
    for line in rustc_verbose.splitlines():
        if line.startswith("host: "):
            target_triple = line.split("host: ", 1)[1].strip()
            break

    cflags_path = ROOT / "build/owned-native/.cflags"
    metalflags_path = ROOT / "build/owned-native/.metalflags"
    tracked_cflags = (
        cflags_path.read_text(encoding="utf-8").split() if cflags_path.is_file() else []
    )
    tracked_metalflags = (
        metalflags_path.read_text(encoding="utf-8").split()
        if metalflags_path.is_file()
        else []
    )

    metallib_path = ROOT / "build/owned-native/tokspur-owned.metallib"
    metallib_fn_count = verify_metallib_and_count_functions(metallib_path, env)

    exec_rels = list(EXECUTABLES)

    artifacts: dict[str, dict[str, object]] = {}
    system_dependencies: dict[str, list[str]] = {}
    for rel in exec_rels:
        p = ROOT / rel
        deps = audit_macho_executable(p)
        st = p.lstat()
        artifacts[rel] = {
            "sha256": sha(p),
            "size": st.st_size,
            "mode": oct(stat.S_IMODE(st.st_mode)),
        }
        system_dependencies[rel] = deps

    mst = metallib_path.lstat()
    artifacts["build/owned-native/tokspur-owned.metallib"] = {
        "sha256": sha(metallib_path),
        "size": mst.st_size,
        "mode": oct(stat.S_IMODE(mst.st_mode)),
    }

    profile_digests = {}
    for rel in (
        "profiles/flash-next-assets.json",
        "profiles/m5-max-128.json",
        "profiles/owned-target-experimental.json",
    ):
        p = ROOT / rel
        if p.is_file():
            profile_digests[rel] = sha(p)

    native_src = collect_native_sources()
    rust_src = collect_rust_sources()

    return {
        "schema": 2,
        "git_head": git_head,
        "git_dirty": git_dirty,
        "enable_metal": True,
        "frozen": True,
        "metallib_function_count": metallib_fn_count,
        "target_triple": target_triple,
        "profile": "release",
        "rustflags": public_rustflags(env),
        "release_processing": {
            "rust_path_remap": "home-to-build-user-and-checkout-to-tokspur",
            "native_and_rust_debug_symbols_stripped": stripped,
        },
        "clang_version": clang_version,
        "metal_version": metal_version,
        "rustc_verbose": rustc_verbose,
        "cargo_version": cargo_version,
        "rust_toolchain_sha256": rust_src["rust/rust-toolchain.toml"],
        "cargo_lock_sha256": rust_src["rust/Cargo.lock"],
        "cargo_config_sha256": rust_src["rust/.cargo/config.toml"],
        "native_config": {
            "NATIVE_CONFIG": "carrier",
            "ENABLE_METAL": 1,
        },
        "tracked_cflags": tracked_cflags,
        "tracked_metalflags": tracked_metalflags,
        "rust_targets": {
            "tokspur-server": ["fast-math"],
            "tokspur": [],
        },
        "profile_digests": profile_digests,
        "native_source_scope": NATIVE_SOURCE_SCOPE,
        "native_sources": native_src,
        "rust_sources": rust_src,
        "artifacts": artifacts,
        "system_dependencies": system_dependencies,
    }


REQUIRED_MEMORY_BYTES = 128 * 1024**3
# The Mach-O executables that the build manifest records and a release signs.
EXECUTABLES = (
    "build/owned-native/tokspur-native",
    "build/owned-native/scalar-channel-target",
    "build/rust-fast-math/release/tokspur-server",
    "build/rust-fast-math/release/tokspur",
)
TESTED_MACOS_MAJOR = 27


def pinned_rust_channel() -> str:
    pin = tomllib.loads((ROOT / "rust/rust-toolchain.toml").read_text())
    return pin["toolchain"]["channel"]


def macos_major(version: str) -> int | None:
    head = version.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def preflight(env: dict[str, str]) -> list[str]:
    """Fail early on a missing toolchain; return warnings for untested hosts."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError("tokspur builds only on an Apple silicon Mac")
    metal = subprocess.run(
        ["xcrun", "metal", "--version"], env=env, capture_output=True, text=True
    )
    if metal.returncode != 0:
        raise ValueError(
            "The Metal compiler is not available. Install Xcode, then run "
            "`xcodebuild -downloadComponent MetalToolchain`"
        )
    channel = pinned_rust_channel()
    if shutil.which("rustup", path=env.get("PATH")) is None:
        raise ValueError(
            "rustup is not installed. Install it from https://rustup.rs, then run "
            f"`rustup toolchain install {channel}`"
        )
    rustc = subprocess.run(
        ["rustc", "--version"],
        cwd=ROOT / "rust",
        env=env,
        capture_output=True,
        text=True,
    )
    if rustc.returncode != 0 or not rustc.stdout.startswith(f"rustc {channel} "):
        raise ValueError(
            f"The pinned Rust {channel} toolchain is not active in rust/. Run "
            f"`rustup toolchain install {channel}`"
        )
    warnings = []
    memory = int(
        subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
    )
    if memory < REQUIRED_MEMORY_BYTES:
        warnings.append(
            "This Mac has less than 128 GB of unified memory, so `tokspur serve` "
            "will refuse to start."
        )
    major = macos_major(platform.mac_ver()[0])
    if major is None or major < TESTED_MACOS_MAJOR:
        warnings.append(
            f"tokspur is tested on macOS {TESTED_MACOS_MAJOR}, and its Metal 4 "
            f"kernels are built for macOS {TESTED_MACOS_MAJOR}.0."
        )
    return warnings


def codesign_executables(
    identity: str, keychain: str | None, env: dict[str, str]
) -> None:
    """Sign with the hardened runtime and a secure timestamp, as notarisation requires.

    Signing changes the files, so it runs before the build manifest records their digests.
    """
    for rel in EXECUTABLES:
        run(["xcrun", "strip", "-S", "-x", str(ROOT / rel)], env=env)
        if re.search(rb"/(?:Users|home)/[A-Za-z0-9_.-]+", (ROOT / rel).read_bytes()):
            raise ValueError("release executable still contains a home path")
        command = [
            "/usr/bin/codesign",
            "--force",
            "--timestamp",
            "--options",
            "runtime",
            "--sign",
            identity,
        ]
        if keychain:
            command += ["--keychain", keychain]
        run([*command, str(ROOT / rel)], env=env)
        run(
            [
                "/usr/bin/codesign",
                "--verify",
                "--strict",
                "--verbose=2",
                str(ROOT / rel),
            ],
            env=env,
        )


def fetch_rust_crates(env: dict[str, str]) -> None:
    """Fill the local Cargo cache from Cargo.lock so later builds stay offline."""
    run(
        ["cargo", "fetch", "--locked"],
        cwd=ROOT / "rust",
        env={**env, "CARGO_NET_OFFLINE": "false"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--install-venv", action="store_true")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python 3.12 or 3.13 for the optional test environment",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--codesign-identity",
        help="Developer ID identity to sign the executables with, for a release build",
    )
    parser.add_argument(
        "--codesign-keychain",
        help="Keychain file that holds the signing identity",
    )
    args = parser.parse_args()
    os.umask(0o077)
    if not 1 <= args.jobs <= 4:
        parser.error("Use 1 to 4 build jobs to leave room for desktop work")
    env = build_env()
    try:
        if args.verify_only:
            rust_cli = ROOT / "build/rust-fast-math/release/tokspur"
            if not rust_cli.is_file():
                raise ValueError(
                    f"Built artifact missing: {rust_cli}; run scripts/bootstrap.py"
                )
            run([str(rust_cli), "verify-build"], env=env)
            return

        if args.source_only:
            manifest = {
                "schema": 2,
                "native_source_scope": NATIVE_SOURCE_SCOPE,
                "native_sources": collect_native_sources(),
                "rust_sources": collect_rust_sources(),
            }
            write_json(ROOT / ".tokspur/receipts/source-manifest.json", manifest)
            print(
                "tokspur sources recorded. No model download, inference or service switch performed."
            )
            return

        warnings = preflight(env)
        for warning in warnings:
            print(f"bootstrap: warning: {warning}", file=sys.stderr)

        api = ROOT / ".venv/bin/python"
        if args.install_venv or not api.exists():
            version = subprocess.check_output(
                [
                    args.python,
                    "-c",
                    "import sys; print('.'.join(map(str, sys.version_info[:2])))",
                ],
                text=True,
            ).strip()
            if version not in ("3.12", "3.13"):
                raise ValueError(
                    "The pinned Python test environment requires Python 3.12 or 3.13"
                )
            if not api.exists():
                run([args.python, "-m", "venv", str(ROOT / ".venv")], env=env)
            # Every downloaded file must match requirements.lock, including the
            # setuptools build backend, so the editable install builds without
            # fetching anything.
            run(
                [
                    str(api),
                    "-m",
                    "pip",
                    "install",
                    "--require-hashes",
                    "--only-binary",
                    ":all:",
                    "--no-deps",
                    "-r",
                    str(ROOT / "requirements.lock"),
                ],
                env=env,
            )
            run(
                [
                    str(api),
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    "--no-deps",
                    "--no-index",
                    "-e",
                    str(ROOT),
                ],
                env=env,
            )

        run(
            ["nice", "-n", "10", "make", "-C", "native", f"-j{args.jobs}", "all"],
            env=env,
        )
        install_scalar_channel_target()
        fetch_rust_crates(env)
        run(
            [
                "nice",
                "-n",
                "10",
                "make",
                "-C",
                "native",
                "-f",
                "rust.mk",
                "rust-release",
            ],
            env=env,
        )
        install_launcher()
        if args.codesign_identity:
            codesign_executables(args.codesign_identity, args.codesign_keychain, env)

        manifest = record_build_manifest(env, stripped=bool(args.codesign_identity))
        write_json(ROOT / ".tokspur/receipts/build-manifest.json", manifest)
        run(
            [str(ROOT / "build/rust-fast-math/release/tokspur"), "verify-build"],
            env=env,
        )
        print(
            "tokspur prepared. No model download, inference or service switch performed."
        )
        print(
            "Next: add .venv/bin to your PATH, then run `tokspur doctor`, "
            "`tokspur adopt`, `tokspur configure` and `tokspur pack build` "
            "as the README describes."
        )
        for warning in warnings:
            print(f"bootstrap: warning: {warning}", file=sys.stderr)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"bootstrap: {error}\n")


if __name__ == "__main__":
    main()
