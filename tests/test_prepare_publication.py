# SPDX-License-Identifier: Apache-2.0
"""Publication snapshots retain source notices while excluding private Git history."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def publisher():
    spec = importlib.util.spec_from_file_location(
        "publication", ROOT / "scripts/prepare_publication.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(path, *args):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Source owner",
            "-c",
            "user.email=private-owner@example.invalid",
            *args,
        ],
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "private-source"
    path.mkdir()
    git(path, "init", "--quiet", "--initial-branch=main")
    (path / "LICENSE").write_bytes(
        b"Copyright upstream contributors\r\nFull retained notice.\r\n"
    )
    (path / "private.txt").write_text("private-owner@example.invalid\n")
    git(path, "add", ".")
    git(path, "commit", "--quiet", "-m", "private fixture")
    (path / "private.txt").unlink()
    (path / "tool.sh").write_text("#!/bin/sh\nexit 0\n")
    (path / "tool.sh").chmod(0o755)
    git(path, "add", "--all")
    git(path, "commit", "--quiet", "-m", "source fixture")
    return path


def test_snapshot_excludes_old_objects_and_retains_exact_source(
    publisher, source, tmp_path
):
    old = git(source, "rev-parse", "HEAD^").decode().strip()
    result = publisher.prepare(source, tmp_path / "publication")
    snapshot = tmp_path / "publication/source"
    assert git(snapshot, "rev-list", "--all", "--count").strip() == b"1"
    assert b"private-owner@example.invalid" not in git(
        snapshot, "log", "--all", "--format=raw"
    )
    assert git(snapshot, "show", "HEAD:LICENSE") == git(source, "show", "HEAD:LICENSE")
    assert (snapshot / "tool.sh").stat().st_mode & 0o111
    assert (
        subprocess.run(
            ["git", "-C", str(snapshot), "cat-file", "-e", old], capture_output=True
        ).returncode
        != 0
    )
    assert result["excluded_contact_identities"] == 1
    assert (
        json.loads((snapshot / "PUBLICATION_SOURCE.json").read_text())["model_assets"]
        is False
    )
    clone = tmp_path / "from-bundle"
    subprocess.run(
        [
            "git",
            "-c",
            "init.defaultBranch=unrelated",
            "clone",
            "--quiet",
            str(tmp_path / "publication/source.bundle"),
            str(clone),
        ],
        check=True,
    )
    assert git(clone, "rev-parse", "HEAD").decode().strip() == result["snapshot_commit"]
    assert git(clone, "rev-list", "--all", "--count").strip() == b"1"


def test_snapshot_refuses_private_contact_in_current_content(
    publisher, source, tmp_path
):
    (source / "contact.txt").write_text("private-owner@example.invalid")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "contact fixture")
    with pytest.raises(ValueError, match="historical contact identity"):
        publisher.prepare(source, tmp_path / "publication")


@pytest.mark.parametrize(
    "name",
    [
        "weights.gguf",
        "draft.safetensors",
        "capture.jsonl",
        ".env",
        ".env.production",
        "private.p8",
        "build/evidence.txt",
    ],
)
def test_snapshot_refuses_private_assets(publisher, source, tmp_path, name):
    path = source / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("generated private fixture")
    git(source, "add", "--force", ".")
    git(source, "commit", "--quiet", "-m", "asset fixture")
    with pytest.raises(ValueError, match="refuses this tracked path"):
        publisher.prepare(source, tmp_path / "publication")


def test_snapshot_refuses_symlinks(publisher, source, tmp_path):
    (source / "link").symlink_to("LICENSE")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "link fixture")
    with pytest.raises(ValueError, match="refuses this tracked path"):
        publisher.prepare(source, tmp_path / "publication")


def test_export_attributes_cannot_omit_notices(publisher, source, tmp_path):
    (source / ".gitattributes").write_text("LICENSE export-ignore\n")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "attribute fixture")
    with pytest.raises(ValueError, match="omitted a tracked source file"):
        publisher.prepare(source, tmp_path / "publication")


def test_snapshot_never_overwrites_existing_output(publisher, source, tmp_path):
    output = tmp_path / "publication"
    output.mkdir()
    (output / "keep").write_text("existing work")
    with pytest.raises(ValueError, match="must be new"):
        publisher.prepare(source, output)
    assert (output / "keep").read_text() == "existing work"


def test_snapshot_refuses_control_characters_in_paths(publisher, source, tmp_path):
    (source / "bad\npath.txt").write_text("generated path fixture")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "path fixture")
    with pytest.raises(ValueError, match="refuses this tracked path"):
        publisher.prepare(source, tmp_path / "publication")


def test_git_attributes_cannot_change_notice_bytes(publisher, source, tmp_path):
    original = git(source, "rev-parse", "HEAD:LICENSE").decode().strip()
    (source / ".gitattributes").write_text("LICENSE text eol=lf\n")
    git(source, "add", ".gitattributes")
    git(source, "update-index", "--cacheinfo", f"100644,{original},LICENSE")
    git(source, "commit", "--quiet", "-m", "normalisation fixture")
    with pytest.raises(ValueError, match="Git changed a source or notice byte"):
        publisher.prepare(source, tmp_path / "publication")
