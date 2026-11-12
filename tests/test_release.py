# SPDX-License-Identifier: Apache-2.0
"""Release packaging and publishing, with a fake stager and a fake GitHub API."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import stat
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "rust/Cargo.toml").read_text())["workspace"]["package"][
    "version"
]
TAG = f"v{VERSION}"
IDENTITY = "Developer ID Application: Example Ltd (ABCDE12345)"


@pytest.fixture
def release():
    spec = importlib.util.spec_from_file_location(
        "release_under_test", ROOT / "scripts/release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.attest_release = fake_attest
    return module


def fake_attest(release, _identity):
    manifest = release / "RELEASE.sha256"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.relative_to(release).as_posix()}\n"
            for path in sorted(release.rglob("*"))
            if path.is_file()
        )
    )
    (release / "release-trust").write_text("signed test fixture\n")


def fake_stage(*, root, output):
    output.mkdir(mode=0o700)
    for name, mode in (("tokspur", 0o700), ("m5-max-128.json", 0o600)):
        (output / name).write_bytes(name.encode())
        (output / name).chmod(mode)
    (output / "provenance.json").write_text("{}\n")
    (output / "provenance.json").chmod(0o600)


def test_tag_must_match_the_project_version(release):
    assert release.check_tag(TAG) == VERSION
    with pytest.raises(ValueError, match="does not match"):
        release.check_tag("v0.0.0-not-this")


def test_release_checks_publication_sources_before_loading_the_signing_key():
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    command = '"$PYTHON" scripts/check_sources.py --publication'
    assert workflow.index(command) < workflow.index("Load the signing certificate")


def test_release_checks_apple_tickets_before_publishing():
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    command = '"$PYTHON" scripts/release.py verify-notarisation'
    assert workflow.index("notarytool submit") < workflow.index(command)
    assert workflow.index(command) < workflow.index("Publish a draft release")


def test_release_only_builds_tags_from_reviewed_main(tmp_path):
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    guard = 'git merge-base --is-ancestor HEAD "$GITHUB_SHA"'
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "fetch-depth: 0" in workflow
    assert workflow.index(guard) < workflow.index("Read the pinned Rust toolchain")
    assert workflow.index(guard) < workflow.index("Load the signing certificate")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    command = [
        "git",
        "-C",
        str(tmp_path),
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(
        [*command, "commit", "--allow-empty", "--quiet", "-m", "base"], check=True
    )
    base = subprocess.check_output([*command, "rev-parse", "HEAD"]).strip()
    subprocess.run(
        [*command, "commit", "--allow-empty", "--quiet", "-m", "main"], check=True
    )
    main = subprocess.check_output([*command, "rev-parse", "HEAD"]).strip()
    subprocess.run([*command, "checkout", "--quiet", "--detach", base], check=True)
    assert (
        subprocess.run(
            [*command, "merge-base", "--is-ancestor", "HEAD", main]
        ).returncode
        == 0
    )
    subprocess.run(
        [*command, "commit", "--allow-empty", "--quiet", "-m", "unreviewed tag"],
        check=True,
    )
    assert (
        subprocess.run(
            [*command, "merge-base", "--is-ancestor", "HEAD", main]
        ).returncode
        != 0
    )


def ticket_fixture(tmp_path):
    root = tmp_path / f"tokspur-{TAG}"
    for name in [
        "release-trust",
        "package/tokspur",
        "package/tokspur-server",
        "package/tokspur-native",
        "package/scalar-channel-target",
    ]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated ticket fixture")
    return root


def test_ticket_verification_retries_until_all_executables_pass(
    release, tmp_path, monkeypatch
):
    root = ticket_fixture(tmp_path)
    commands, delays = [], []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs == {"capture_output": True, "timeout": 30}
        return subprocess.CompletedProcess(command, 3 if len(commands) == 1 else 0)

    monkeypatch.setattr(release.subprocess, "run", run)
    monkeypatch.setattr(release.time, "sleep", delays.append)
    result = release.verify_notarisation(
        TAG, tmp_path, signing_identity=IDENTITY, attempts=2
    )
    assert result["attempts"] == 2
    assert result["executables"] == 5
    assert delays == [15]
    assert {Path(command[-1]).relative_to(root).as_posix() for command in commands} == {
        "release-trust",
        "package/tokspur",
        "package/tokspur-server",
        "package/tokspur-native",
        "package/scalar-channel-target",
    }
    for command in commands:
        assert command[:5] == [
            "/usr/bin/codesign",
            "--verify",
            "--strict",
            "--check-notarization",
            "-R",
        ]
        requirement = command[5]
        assert requirement.startswith("=anchor apple generic")
        assert (
            'certificate leaf[subject.OU] = "ABCDE12345" and notarized' in requirement
        )
        if command[-1].endswith("/release-trust"):
            assert 'identifier "com.tokspur.release-trust"' in requirement


@pytest.mark.parametrize("timeout", [False, True])
def test_ticket_verification_refuses_persistent_failures(
    release, tmp_path, monkeypatch, timeout
):
    ticket_fixture(tmp_path)
    calls, delays = [], []

    def run(command, **kwargs):
        calls.append(command)
        if timeout:
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 3)

    monkeypatch.setattr(release.subprocess, "run", run)
    monkeypatch.setattr(release.time, "sleep", delays.append)
    with pytest.raises(ValueError, match="verification failed"):
        release.verify_notarisation(
            TAG, tmp_path, signing_identity=IDENTITY, attempts=2, delay=0
        )
    assert len(calls) == 2
    assert delays == [0]


def test_ticket_verification_requires_every_executable(release, tmp_path, monkeypatch):
    root = ticket_fixture(tmp_path)
    (root / "package/tokspur-server").unlink()
    commands = []
    monkeypatch.setattr(
        release.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    with pytest.raises(ValueError, match="requires regular package/tokspur-server"):
        release.verify_notarisation(TAG, tmp_path, signing_identity=IDENTITY)
    assert commands == []


@pytest.mark.parametrize("name", ["weights.gguf", "tokenizer.json", "capture.jsonl"])
def test_runtime_release_refuses_model_assets_and_captures(release, tmp_path, name):
    def stage_with_private_asset(*, root, output):
        fake_stage(root=root, output=output)
        (output / name).write_bytes(b"generated private asset fixture")

    with pytest.raises(ValueError, match="unapproved asset"):
        release.package(
            TAG,
            tmp_path / "dist",
            signing_identity=IDENTITY,
            stage=stage_with_private_asset,
        )


def test_runtime_release_refuses_embedded_home_paths(release, tmp_path):
    def stage_with_path(*, root, output):
        fake_stage(root=root, output=output)
        (output / "tokspur").write_bytes(
            b"generated fixture\0/home/private-owner/source\0"
        )

    with pytest.raises(ValueError, match="contains a home path"):
        release.package(
            TAG, tmp_path / "dist", signing_identity=IDENTITY, stage=stage_with_path
        )


def test_release_selects_the_tag_when_a_branch_has_the_same_name(tmp_path):
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    selected = re.search(r"^\s+ref: (.+)$", workflow, re.MULTILINE).group(1)
    selected = selected.replace("${{ inputs.tag }}", "v0.0.0")
    assert selected == "refs/tags/v0.0.0"

    def git(*arguments):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                *arguments,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    git("init", "--quiet")
    path = tmp_path / "source.txt"
    path.write_text("tag source\n")
    git("add", path.name)
    git("commit", "--quiet", "-m", "tag fixture")
    expected = git("rev-parse", "HEAD")
    git("tag", "v0.0.0")
    path.write_text("branch source\n")
    git("commit", "--quiet", "-am", "branch fixture")
    git("branch", "v0.0.0")
    assert git("rev-parse", "refs/heads/v0.0.0") != expected
    assert git("rev-parse", selected) == expected


def test_package_writes_the_archive_licences_and_checksum(release, tmp_path):
    result = release.package(
        TAG, tmp_path / "dist", signing_identity=IDENTITY, stage=fake_stage
    )
    archive = Path(result["archive"])
    assert archive.name == f"tokspur-{TAG}-macos-arm64.tar.gz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert result["sha256"] == digest
    assert Path(result["checksum"]).read_text() == f"{digest}  {archive.name}\n"

    with tarfile.open(archive) as stream:
        members = {member.name: member for member in stream.getmembers()}
    top = f"tokspur-{TAG}"
    assert {
        f"{top}/package/tokspur",
        f"{top}/package/provenance.json",
        f"{top}/LICENSE",
        f"{top}/NOTICE",
        f"{top}/licences/llama.cpp-LICENSE",
        f"{top}/licences/UNICODE-LICENSE.txt",
        f"{top}/licences/omlx-LICENSE",
        f"{top}/licences/Qwen3.8-Flash-Next-LICENSE",
        f"{top}/MODEL_LICENCES.md",
        f"{top}/PROVENANCE.md",
        f"{top}/source-manifest.json",
        f"{top}/VERSION",
        f"{top}/install.sh",
    } <= set(members)
    assert stat.S_IMODE(members[f"{top}/install.sh"].mode) == 0o755
    installer = Path(result["installer"]).read_text()
    assert 'pinned_team_id="ABCDE12345"' in installer
    assert "@TOKSPUR_TEAM_ID@" not in installer
    # verify-build checks these modes after extraction, so the archive must keep them.
    assert stat.S_IMODE(members[f"{top}/package"].mode) == 0o700
    assert stat.S_IMODE(members[f"{top}/package/tokspur"].mode) == 0o700
    assert stat.S_IMODE(members[f"{top}/package/provenance.json"].mode) == 0o600
    assert all(m.uid == 0 and m.gid == 0 and not m.uname for m in members.values())


def test_team_id_comes_from_the_developer_id_name(release):
    assert release.team_id(IDENTITY) == "ABCDE12345"
    for bad in ("ABCDE12345", "Apple Development: Example (ABCDE12345)", ""):
        with pytest.raises(ValueError, match="Developer ID Application"):
            release.team_id(bad)


def test_publish_refuses_an_unpinned_installer(release, tmp_path):
    dist = tmp_path / "dist"
    release.package(TAG, dist, signing_identity=IDENTITY, stage=fake_stage)
    (dist / "install.sh").write_text((ROOT / "scripts/install.sh").read_text())
    with pytest.raises(ValueError, match="no pinned Team ID"):
        release.publish(TAG, dist, repository="o/r", token="t", opener=lambda _: None)


def test_package_refuses_an_output_inside_the_checkout(release):
    with pytest.raises(ValueError, match="outside the checkout"):
        release.package(
            TAG,
            ROOT / "build/release-test",
            signing_identity=IDENTITY,
            stage=fake_stage,
        )


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_publish_creates_a_draft_and_uploads_every_asset(release, tmp_path):
    dist = tmp_path / "dist"
    release.package(TAG, dist, signing_identity=IDENTITY, stage=fake_stage)
    requests = []

    def opener(request):
        requests.append(request)
        if request.full_url.endswith("/releases"):
            return FakeResponse(json.dumps({"id": 7, "html_url": "u"}).encode())
        return FakeResponse(b"{}")

    result = release.publish(
        TAG, dist, repository="owner/repo", token="t", opener=opener
    )
    create = requests[0]
    assert create.full_url == "https://api.github.com/repos/owner/repo/releases"
    assert json.loads(create.data)["draft"] is True
    assert json.loads(create.data)["tag_name"] == TAG
    notes = json.loads(create.data)["body"]
    digest = (dist / f"tokspur-{TAG}-macos-arm64.tar.gz.sha256").read_text().split()[0]
    assert f"releases/download/{TAG}/install.sh | TOKSPUR_VERSION={TAG} sh" in notes
    assert digest in notes
    assert create.get_header("Authorization") == "Bearer t"
    uploads = [r.full_url for r in requests[1:]]
    assert uploads == [
        f"https://uploads.github.com/repos/owner/repo/releases/7/assets?name={name}"
        for name in result["assets"]
    ]
    assert result["assets"] == [
        f"tokspur-{TAG}-macos-arm64.tar.gz",
        f"tokspur-{TAG}-macos-arm64.tar.gz.sha256",
        "install.sh",
    ]


def test_publish_refuses_an_archive_that_does_not_match_its_checksum(release, tmp_path):
    dist = tmp_path / "dist"
    result = release.package(TAG, dist, signing_identity=IDENTITY, stage=fake_stage)
    Path(result["archive"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        release.publish(TAG, dist, repository="o/r", token="t", opener=lambda _: None)


def test_bootstrap_signs_every_recorded_executable(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "bootstrap_under_test", ROOT / "scripts/bootstrap.py"
    )
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    for name in bootstrap.EXECUTABLES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated executable fixture")
    commands = []
    monkeypatch.setattr(bootstrap, "run", lambda command, **_: commands.append(command))
    bootstrap.codesign_executables("Developer ID Application: Example", "/k", {})
    signs = [c for c in commands if "--sign" in c]
    assert [Path(c[-1]).relative_to(tmp_path).as_posix() for c in signs] == list(
        bootstrap.EXECUTABLES
    )
    for command in signs:
        assert command[:7] == [
            "/usr/bin/codesign",
            "--force",
            "--timestamp",
            "--options",
            "runtime",
            "--sign",
            "Developer ID Application: Example",
        ]
        assert command[7:9] == ["--keychain", "/k"]
    assert len([c for c in commands if "--verify" in c]) == len(bootstrap.EXECUTABLES)
    assert len(commands) == 3 * len(bootstrap.EXECUTABLES)
    for index in range(0, len(commands), 3):
        assert commands[index][:4] == ["xcrun", "strip", "-S", "-x"]
        assert "--sign" in commands[index + 1]
        assert "--verify" in commands[index + 2]


def test_package_stages_a_recorded_build_with_the_real_staging_script(
    release, tmp_path
):
    """Run the real stage script over a fake but self-consistent Rust build."""
    root = tmp_path / "checkout"

    def write(rel, data, mode=0o600):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)
        return path

    for rel in (
        "LICENSE",
        "NOTICE",
        "MODEL_LICENCES.md",
        "native/Makefile",
        "native/src/owned.cpp",
        "native/PROVENANCE.md",
        "native/data/UNICODE-LICENSE.txt",
        "native/third_party/llama-cpp/LICENSE",
        "native/third_party/mlx-vlm/LICENSE",
        "native/third_party/omlx/LICENSE",
        "native/third_party/sources.json",
        "native/third_party/m5-max-kernels/LICENSE",
        "native/third_party/splash/LICENSE",
        "native/third_party/splash/THIRD_PARTY_NOTICES",
        "third_party/model-licences/Qwen3.8-Flash-Next/LICENSE",
        "third_party/rust/COPYRIGHT-library.html",
        "rust/Cargo.lock",
        "rust/rust-toolchain.toml",
        "rust/.cargo/config.toml",
        "rust/crates/tokspur-cli/src/main.rs",
    ):
        write(rel, f"{rel}\n".encode())
    write(
        "rust/Cargo.toml",
        f'[workspace.package]\nversion = "{VERSION}"\n'.encode(),
    )
    write("pyproject.toml", f'[project]\nversion = "{VERSION}"\n'.encode())
    write("src/tokspur/__init__.py", f'__version__ = "{VERSION}"\n'.encode())
    write("scripts/install.sh", (ROOT / "scripts/install.sh").read_bytes(), 0o755)
    stage_script = (ROOT / "scripts/stage_owned_native_package.py").read_bytes()
    write("scripts/stage_owned_native_package.py", stage_script, 0o755)
    for rel in (
        "profiles/owned-target-experimental.json",
        "profiles/m5-max-128.json",
        "profiles/flash-next-assets.json",
    ):
        write(rel, b'{"schema":1}\n')
    macho = b"\xcf\xfa\xed\xfe\x0c\x00\x00\x01"
    executables = {
        "build/rust-fast-math/release/tokspur": macho + b"cli",
        "build/rust-fast-math/release/tokspur-server": macho + b"server",
        "build/owned-native/tokspur-native": macho + b"worker",
        "build/owned-native/scalar-channel-target": macho + b"worker",
    }
    for rel, data in executables.items():
        write(rel, data, 0o700)
    write("build/owned-native/tokspur-owned.metallib", b"MTLBfake-metal", 0o600)

    spec = importlib.util.spec_from_file_location(
        "stage_for_test", root / "scripts/stage_owned_native_package.py"
    )
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)

    def record(rel):
        path = root / rel
        return {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
            "mode": oct(stat.S_IMODE(path.stat().st_mode)),
        }

    manifest = {
        "schema": 2,
        "git_head": "0" * 40,
        "git_dirty": False,
        "enable_metal": True,
        "native_config": {"NATIVE_CONFIG": "carrier"},
        "tracked_cflags": ["carrier", ":", "test flags"],
        "frozen": True,
        "metallib_function_count": 16,
        "rust_targets": {"tokspur-server": ["fast-math"], "tokspur": []},
        "native_source_scope": stage.NATIVE_SOURCE_SCOPE,
        "native_sources": stage.native_sources(root, build_inputs_only=True),
        "rust_sources": stage.rust_sources(root),
        "profile_digests": {
            rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
            for rel in (
                "profiles/owned-target-experimental.json",
                "profiles/m5-max-128.json",
                "profiles/flash-next-assets.json",
            )
        },
        "artifacts": {
            rel: record(rel)
            for rel in [*executables, "build/owned-native/tokspur-owned.metallib"]
        },
        "system_dependencies": {
            rel: ["/usr/lib/libSystem.B.dylib"] for rel in executables
        },
    }
    write(release.RECEIPT, json.dumps(manifest).encode())

    result = release.package(
        TAG, tmp_path / "dist", signing_identity=IDENTITY, root=root
    )
    package = Path(result["release"]) / "package"
    assert sorted(p.name for p in package.iterdir()) == [
        "flash-next-assets.json",
        "m5-max-128.json",
        "owned-target-experimental.json",
        "provenance.json",
        "scalar-channel-target",
        "tokspur",
        "tokspur-native",
        "tokspur-owned.metallib",
        "tokspur-server",
    ]
    provenance = json.loads((package / "provenance.json").read_text())
    assert provenance["schema"] == 2 and provenance["control_plane"] == "rust"
    assert set(release.LICENCES.values()) == set(provenance["proof"]["notices"])
    for destination, source in release.LICENCES.items():
        expected = (root / source).read_bytes()
        assert (Path(result["release"]) / destination).read_bytes() == expected
        assert provenance["proof"]["notices"][source].encode() == expected
    with tarfile.open(result["archive"]) as stream:
        names = {m.name for m in stream.getmembers()}
    assert f"tokspur-{TAG}/package/provenance.json" in names
    assert f"tokspur-{TAG}/licences/llama.cpp-LICENSE" in names
