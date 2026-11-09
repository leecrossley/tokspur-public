#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package a built, signed checkout as a release archive and publish it as a draft GitHub release.

`package` stages the build that scripts/bootstrap.py recorded, through scripts/stage_owned_native_package.py, and adds licences, an installer pinned to the signing Team ID and a signed executable that authenticates the complete release inventory. `publish` creates a draft release for the tag and uploads the archive, its transport checksum and the pinned installer. Notarisation runs separately in the release workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = "macos-arm64"
POLICY = "profiles/owned-target-experimental.json"
RECEIPT = ".tokspur/receipts/build-manifest.json"
LICENCES = {
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
    "MODEL_LICENCES.md": "MODEL_LICENCES.md",
    "PROVENANCE.md": "native/PROVENANCE.md",
    "source-manifest.json": "native/third_party/sources.json",
    "licences/llama.cpp-LICENSE": "native/third_party/llama-cpp/LICENSE",
    "licences/UNICODE-LICENSE.txt": "native/data/UNICODE-LICENSE.txt",
    "licences/mlx-vlm-LICENSE": "native/third_party/mlx-vlm/LICENSE",
    "licences/omlx-LICENSE": "native/third_party/omlx/LICENSE",
    "licences/m5-max-kernels-LICENSE": "native/third_party/m5-max-kernels/LICENSE",
    "licences/splash-LICENSE": "native/third_party/splash/LICENSE",
    "licences/splash-THIRD_PARTY_NOTICES": "native/third_party/splash/THIRD_PARTY_NOTICES",
    "licences/Qwen3.8-Flash-Next-LICENSE": "third_party/model-licences/Qwen3.8-Flash-Next/LICENSE",
    "licences/rust-COPYRIGHT-library.html": "third_party/rust/COPYRIGHT-library.html",
}
TEAM_MARKER = "@TOKSPUR_TEAM_ID@"
RUNTIME_FILES = {
    "tokspur",
    "tokspur-server",
    "tokspur-native",
    "scalar-channel-target",
    "tokspur-owned.metallib",
    "provenance.json",
    "m5-max-128.json",
    "flash-next-assets.json",
    "owned-target-experimental.json",
}
API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"


def fail(message: str) -> None:
    raise ValueError(message)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_version(root: Path = ROOT) -> str:
    """The version Cargo and the Python launcher agree on."""
    cargo = tomllib.loads((root / "rust/Cargo.toml").read_text())
    rust = cargo["workspace"]["package"]["version"]
    python = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    launcher = (root / "src/tokspur/__init__.py").read_text()
    if rust != python or f'__version__ = "{rust}"' not in launcher:
        fail(
            "rust/Cargo.toml, pyproject.toml and src/tokspur/__init__.py "
            "must carry the same version"
        )
    return rust


def check_tag(tag: str, root: Path = ROOT) -> str:
    version = project_version(root)
    if tag != f"v{version}":
        fail(f"tag {tag} does not match the project version v{version}")
    return version


def archive_name(tag: str) -> str:
    return f"tokspur-{tag}-{PLATFORM}.tar.gz"


def team_id(signing_identity: str) -> str:
    """The Apple Team ID at the end of a Developer ID certificate name."""
    match = re.fullmatch(
        r"Developer ID Application: .+ \(([A-Z0-9]{10})\)", signing_identity
    )
    if not match:
        fail(
            "the signing identity must be a Developer ID Application name ending in (TEAMID)"
        )
    return match[1]


def render_installer(team: str, root: Path = ROOT) -> str:
    """install.sh with the Team ID pinned, so installs and updates check the signature."""
    source = (root / "scripts/install.sh").read_text()
    if source.count(TEAM_MARKER) != 1:
        fail("scripts/install.sh must contain the Team ID marker exactly once")
    return source.replace(TEAM_MARKER, team)


def stage_with_script(*, root: Path, output: Path) -> None:
    """Stage the recorded build with the Rust control plane."""
    spec = importlib.util.spec_from_file_location(
        "stage_owned_native_package", root / "scripts/stage_owned_native_package.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = root / RECEIPT
    policy = root / POLICY
    if not receipt.is_file():
        fail(f"{RECEIPT} is missing; run scripts/bootstrap.py first")
    module.stage(
        source_root=root,
        build=root / "build/owned-native",
        receipt=receipt,
        receipt_sha256=sha256_file(receipt),
        policy=policy,
        policy_sha256=sha256_file(policy),
        output=output,
        control_plane="rust",
        rust_build=root / "build/rust-fast-math/release",
    )


def tar_filter(member: tarfile.TarInfo) -> tarfile.TarInfo:
    # Extraction assigns the installing user. Keep the recorded modes, which verify-build checks.
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


def attest_release(release: Path, signing_identity: str) -> None:
    """Sign a fixed digest of the complete release file inventory."""
    paths = sorted(path for path in release.rglob("*") if path.is_file())
    lines = []
    for path in paths:
        relative = "./" + path.relative_to(release).as_posix()
        if path.is_symlink() or any(c in relative for c in "\n\r\\"):
            fail("release inventory refuses symlinks and control characters")
        lines.append(f"{sha256_file(path)}  {relative}\n")
    manifest = release / "RELEASE.sha256"
    manifest.write_text("".join(lines))
    manifest.chmod(0o644)
    digest = sha256_file(manifest)
    binary = release / "release-trust"
    with tempfile.TemporaryDirectory(prefix="tokspur-release-trust-") as temporary:
        source = Path(temporary) / "release-trust.c"
        source.write_text(
            '#include <stdio.h>\nint main(void) { return puts("'
            + digest
            + '") < 0; }\n'
        )
        subprocess.run(
            [
                "/usr/bin/xcrun",
                "clang",
                "-arch",
                "arm64",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
        )
    binary.chmod(0o755)
    command = [
        "/usr/bin/codesign",
        "--force",
        "--options",
        "runtime",
        "--timestamp",
        "--identifier",
        "com.tokspur.release-trust",
        "--sign",
        signing_identity,
    ]
    if keychain := os.environ.get("TOKSPUR_CODESIGN_KEYCHAIN"):
        command.extend(["--keychain", keychain])
    subprocess.run([*command, str(binary)], check=True)


def package(
    tag: str,
    output: Path,
    *,
    signing_identity: str,
    root: Path = ROOT,
    stage: Callable[..., None] = stage_with_script,
) -> dict[str, str]:
    """Write tokspur-<tag>/ with its pinned installer, and the archive, checksum and installer asset under output."""
    check_tag(tag, root)
    installer = render_installer(team_id(signing_identity), root)
    output = output.resolve()
    if output.is_relative_to(root.resolve()):
        fail("the output directory must be outside the checkout")
    output.mkdir(parents=True, exist_ok=True)
    release = output / f"tokspur-{tag}"
    if release.exists():
        fail(f"{release} already exists")
    release.mkdir(mode=0o755)
    stage(root=root, output=release / "package")
    if not (release / "package/provenance.json").is_file():
        fail("staging did not produce package/provenance.json")
    for path in (release / "package").iterdir():
        if path.name not in RUNTIME_FILES or path.is_symlink() or not path.is_file():
            fail("release package contains an unapproved asset")
        if path.name in {
            "tokspur",
            "tokspur-server",
            "tokspur-native",
            "scalar-channel-target",
        } and re.search(rb"/(?:Users|home)/[A-Za-z0-9_.-]+", path.read_bytes()):
            fail("release executable contains a home path")
    for name, source in LICENCES.items():
        destination = release / name
        destination.parent.mkdir(mode=0o755, exist_ok=True)
        shutil.copyfile(root / source, destination)
        destination.chmod(0o644)
    (release / "VERSION").write_text(tag + "\n")
    (release / "VERSION").chmod(0o644)
    # The installed copy runs later updates.
    for path in (release / "install.sh", output / "install.sh"):
        path.write_text(installer)
        path.chmod(0o755)

    attest_release(release, signing_identity)
    archive = output / archive_name(tag)
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(release, arcname=release.name, filter=tar_filter)
    digest = sha256_file(archive)
    checksum = output / (archive.name + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n")
    return {
        "release": str(release),
        "archive": str(archive),
        "checksum": str(checksum),
        "installer": str(output / "install.sh"),
        "sha256": digest,
    }


def verify_notarisation(
    tag: str,
    directory: Path,
    *,
    signing_identity: str,
    attempts: int = 40,
    delay: float = 15,
) -> dict:
    """Wait for Apple's tickets without running a release executable."""
    check_tag(tag)
    team = team_id(signing_identity)
    if not 1 <= attempts <= 40 or not 0 <= delay <= 15:
        fail(
            "notarisation verification permits 1 to 40 attempts, 0 to 15 seconds apart"
        )
    release = directory / f"tokspur-{tag}"
    names = (
        "release-trust",
        "package/tokspur",
        "package/tokspur-server",
        "package/tokspur-native",
        "package/scalar-channel-target",
    )
    requirement = (
        "anchor apple generic and "
        "certificate 1[field.1.2.840.113635.100.6.2.6] exists and "
        "certificate leaf[field.1.2.840.113635.100.6.1.13] exists and "
        f'certificate leaf[subject.OU] = "{team}" and notarized'
    )
    commands = []
    for name in names:
        path = release / name
        if not path.is_file() or path.is_symlink():
            fail(f"notarisation verification requires regular {name}")
        specific = requirement
        if name == "release-trust":
            specific += ' and identifier "com.tokspur.release-trust"'
        commands.append(
            [
                "/usr/bin/codesign",
                "--verify",
                "--strict",
                "--check-notarization",
                "-R",
                "=" + specific,
                str(path),
            ]
        )
    for attempt in range(1, attempts + 1):
        try:
            valid = all(
                subprocess.run(command, capture_output=True, timeout=30).returncode == 0
                for command in commands
            )
        except subprocess.TimeoutExpired:
            valid = False
        if valid:
            return {
                "tag": tag,
                "notarisation": "verified",
                "executables": len(names),
                "attempts": attempt,
            }
        if attempt < attempts:
            time.sleep(delay)
    fail(
        "Apple ticket, Developer ID chain, team or executable identifier verification failed"
    )


def github_request(
    url: str,
    token: str,
    *,
    data: bytes | None = None,
    content_type: str = "application/json",
    opener: Callable = urllib.request.urlopen,
) -> dict:
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with opener(request) as response:
        return json.loads(response.read() or b"{}")


def release_body(tag: str, repository: str, digest: str) -> str:
    """Install instructions that GitHub places above the generated release notes."""
    return (
        "Install on an Apple M5 Max with 128 GB of unified memory and macOS 27 "
        "or later:\n\n"
        "```sh\n"
        f"curl -fsSL https://github.com/{repository}/releases/download/{tag}/install.sh"
        f" | TOKSPUR_VERSION={tag} sh\n"
        "```\n\n"
        "The installer checks the archive against its SHA-256, "
        f"`{digest}`, requires the pinned Apple Developer ID signing team and authenticates every released file before verifying the package. Then add the model and "
        f"start the server as the [README](https://github.com/{repository}#add-the-model) "
        "describes.\n"
    )


def publish(
    tag: str,
    directory: Path,
    *,
    repository: str,
    token: str,
    root: Path = ROOT,
    opener: Callable = urllib.request.urlopen,
) -> dict:
    """Create a draft release for tag and upload the archive, checksum and installer."""
    check_tag(tag, root)
    archive = directory / archive_name(tag)
    assets = [
        (archive, "application/gzip"),
        (directory / (archive.name + ".sha256"), "text/plain"),
        (directory / "install.sh", "text/x-shellscript"),
    ]
    for path, _ in assets:
        if not path.is_file():
            fail(f"missing release asset: {path}")
    if TEAM_MARKER in (directory / "install.sh").read_text():
        fail("install.sh has no pinned Team ID; package with --signing-identity")
    expected = (directory / (archive.name + ".sha256")).read_text().split()[0]
    if sha256_file(archive) != expected:
        fail(f"{archive.name} does not match its checksum file")
    body = json.dumps(
        {
            "tag_name": tag,
            "name": f"tokspur {tag}",
            "body": release_body(tag, repository, expected),
            "draft": True,
            "generate_release_notes": True,
        }
    ).encode()
    release = github_request(
        f"{API}/repos/{repository}/releases", token, data=body, opener=opener
    )
    uploaded = []
    for path, content_type in assets:
        query = urllib.parse.urlencode({"name": path.name})
        github_request(
            f"{UPLOADS}/repos/{repository}/releases/{release['id']}/assets?{query}",
            token,
            data=path.read_bytes(),
            content_type=content_type,
            opener=opener,
        )
        uploaded.append(path.name)
    return {"release": release.get("html_url", ""), "assets": uploaded}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("package", help="stage and archive a recorded build")
    pack.add_argument("--tag", required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument(
        "--signing-identity",
        required=True,
        help="the Developer ID Application identity that signed the build",
    )
    pub = commands.add_parser("publish", help="create a draft GitHub release")
    pub.add_argument("--tag", required=True)
    pub.add_argument("--dir", type=Path, required=True)
    verify = commands.add_parser(
        "verify-notarisation", help="verify Apple tickets before publishing"
    )
    verify.add_argument("--tag", required=True)
    verify.add_argument("--dir", type=Path, required=True)
    verify.add_argument("--signing-identity", required=True)
    args = parser.parse_args()
    os.umask(0o022)
    try:
        if args.command == "package":
            result = package(
                args.tag, args.output, signing_identity=args.signing_identity
            )
        elif args.command == "verify-notarisation":
            result = verify_notarisation(
                args.tag, args.dir, signing_identity=args.signing_identity
            )
        else:
            repository = os.environ.get("GITHUB_REPOSITORY", "")
            token = os.environ.get("GITHUB_TOKEN", "")
            if not repository or not token:
                fail("GITHUB_REPOSITORY and GITHUB_TOKEN are required to publish")
            result = publish(args.tag, args.dir, repository=repository, token=token)
    except (OSError, ValueError) as error:
        print(f"release: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
