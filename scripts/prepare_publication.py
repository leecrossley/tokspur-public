#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a private source snapshot with fresh Git history and no model assets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

EMAIL = "807783+leecrossley@users.noreply.github.com"
PRIVATE_ROOTS = {
    ".agents",
    ".claude",
    ".codex",
    ".cursor",
    ".tokspur",
    "build",
    "node_modules",
    "target",
}
PRIVATE_SUFFIXES = {
    ".gguf",
    ".safetensors",
    ".jsonl",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".log",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".p12",
    ".p8",
    ".pem",
    ".key",
}


def git(root: Path, *arguments: str, env: dict | None = None) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *arguments],
        env=env,
        stderr=subprocess.PIPE,
    )


def service_identity(address: str) -> bool:
    local, _, domain = address.lower().rpartition("@")
    return (
        not address
        or "noreply" in domain
        or bool(re.fullmatch(r"(?:.*[+._-])?no[._-]?reply(?:[+._-].*)?", local))
    )


def prepare(source: Path, output: Path, *, revision: str = "HEAD") -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError(
            "publication output must be new and outside the source checkout"
        )
    commit = (
        git(
            source,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{revision}^{{commit}}",
        )
        .decode()
        .strip()
    )
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("publication requires a complete Git commit")
    tree = git(source, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    entries = []
    for record in git(source, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if not record:
            continue
        header, name_bytes = record.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = name_bytes.decode("utf-8")
        path = PurePosixPath(name)
        if (
            mode not in {"100644", "100755"}
            or kind != "blob"
            or path.is_absolute()
            or ".." in path.parts
            or any(part in PRIVATE_ROOTS for part in path.parts)
            or any(character in name for character in "\r\n\t\\")
            or path.name in {".env", "id_rsa", "id_ed25519"}
            or path.name.startswith(".env.")
            or path.suffix.lower() in PRIVATE_SUFFIXES
        ):
            raise ValueError(f"publication refuses this tracked path: {name}")
        entries.append((name, oid))
    identities = {
        address.lower().encode()
        for address in git(source, "log", "--all", "--format=%ae%n%ce")
        .decode()
        .splitlines()
        if not service_identity(address)
    }
    total = 0
    for name, oid in entries:
        size = int(git(source, "cat-file", "-s", oid))
        total += size
        if size > 64 << 20 or total > 512 << 20:
            raise ValueError("publication source exceeds its byte budget")
        content = git(source, "cat-file", "blob", oid).lower()
        if any(address in content for address in identities):
            raise ValueError(
                f"tracked content contains a historical contact identity: {name}"
            )
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="publication-", dir=output.parent
    ) as temporary:
        work = Path(temporary)
        snapshot = work / "source"
        snapshot.mkdir(mode=0o700)
        archive = git(source, "archive", "--format=tar", commit)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(snapshot, filter="data")
        for name, oid in entries:
            path = snapshot / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"archive omitted a tracked source file: {name}")
            if path.read_bytes() != git(source, "cat-file", "blob", oid):
                raise ValueError(f"archive changed a tracked source file: {name}")
        metadata = {
            "schema": 1,
            "source_commit": commit,
            "source_tree": tree,
            "history": "Fresh snapshot. Private development history, PR refs and Actions logs are excluded.",
            "model_assets": False,
        }
        metadata_path = snapshot / "PUBLICATION_SOURCE.json"
        if metadata_path.exists():
            raise ValueError("source already contains a publication snapshot receipt")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        subprocess.run(
            ["git", "init", "--quiet", "--initial-branch=main", str(snapshot)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        environment = dict(os.environ)
        for key in tuple(environment):
            if key.startswith("GIT_"):
                del environment[key]
        timestamp = git(source, "show", "-s", "--format=%cI", commit).decode().strip()
        environment.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL="/dev/null",
            GIT_AUTHOR_NAME="San Digital",
            GIT_AUTHOR_EMAIL=EMAIL,
            GIT_COMMITTER_NAME="San Digital",
            GIT_COMMITTER_EMAIL=EMAIL,
            GIT_AUTHOR_DATE=timestamp,
            GIT_COMMITTER_DATE=timestamp,
        )
        git(snapshot, "add", "--all", env=environment)
        git(
            snapshot,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "prepare source for publication",
            env=environment,
        )
        published = git(snapshot, "rev-parse", "HEAD", env=environment).decode().strip()
        for name, oid in entries:
            if git(snapshot, "show", f"HEAD:{name}", env=environment) != git(
                source, "cat-file", "blob", oid
            ):
                raise ValueError(f"Git changed a source or notice byte: {name}")
        if (
            git(snapshot, "rev-list", "--count", "HEAD", env=environment).strip()
            != b"1"
        ):
            raise ValueError("publication snapshot retained earlier history")
        bundle = work / "source.bundle"
        git(
            snapshot,
            "bundle",
            "create",
            str(bundle),
            "refs/heads/main",
            "HEAD",
            env=environment,
        )
        git(snapshot, "bundle", "verify", str(bundle), env=environment)
        git(snapshot, "fsck", "--strict", "--no-reflogs", env=environment)
        receipt = {
            **metadata,
            "snapshot_commit": published,
            "tracked_source_files": len(entries),
            "excluded_contact_identities": len(identities),
            "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        }
        (work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        work.rename(output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    os.umask(0o077)
    try:
        receipt = prepare(
            arguments.source, arguments.output, revision=arguments.revision
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"publication snapshot failed: {type(error).__name__}\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
