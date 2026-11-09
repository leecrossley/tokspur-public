#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check source and licence records without downloading models or executing upstream code."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "native/third_party/sources.json"
MAX_SOURCE = 5 << 20
PROJECT = re.compile(
    r"llama\.cpp|oMLX|mlx-vlm|Splash|m5-max-kernels|ggml authors", re.I
)
COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".metal", ".mm", ".inc"}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative(name: str) -> PurePosixPath:
    need(isinstance(name, str), "source path must be text")
    path = PurePosixPath(name)
    need(
        isinstance(name, str)
        and bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == name
        and "\\" not in name,
        "source path must be relative and canonical",
    )
    return path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_bytes(root: Path, name: str) -> bytes:
    path = root / relative(name)
    need(path.resolve().is_relative_to(root.resolve()), "source path leaves checkout")
    need(path.is_file() and not path.is_symlink(), "source file is missing or unsafe")
    need(path.stat().st_size <= MAX_SOURCE, "source file exceeds audit bound")
    return path.read_bytes()


def validate(root: Path, *, fetch: bool = False, opener=None) -> dict:
    manifest = json.loads(local_bytes(root, MANIFEST))
    need(manifest.get("schema") == 1, "unsupported source manifest")
    projects = {}
    fetched = 0
    for project in manifest["projects"]:
        name = project["id"]
        need(name not in projects, "duplicate source project")
        projects[name] = project
        need(
            re.fullmatch(r"[0-9a-f]{40}", project["revision"]),
            "full source commit required",
        )
        need(
            re.fullmatch(
                r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
                project["repository"],
            ),
            "source repository must be an explicit GitHub project",
        )
        need(
            project["pin_kind"] in {"recorded-source", "comparison-only"},
            "source pin must distinguish an import claim from a comparison",
        )
        need(
            digest(local_bytes(root, project["licence_file"]))
            == project["licence_sha256"],
            "bundled licence differs from source record",
        )
        seen = set()
        for source in project["files"]:
            relative(source["path"])
            need(source["path"] not in seen, "duplicate upstream file")
            seen.add(source["path"])
            need(
                re.fullmatch(r"[0-9a-f]{64}", source["sha256"]),
                "source SHA-256 required",
            )
            need(0 < source["bytes"] <= MAX_SOURCE, "upstream file exceeds audit bound")
            if fetch:
                repository = project["repository"].removeprefix("https://github.com/")
                url = f"https://raw.githubusercontent.com/{repository}/{project['revision']}/{source['path']}"
                with (opener or urllib.request.urlopen)(url, timeout=30) as response:
                    data = response.read(MAX_SOURCE + 1)
                need(
                    len(data) == source["bytes"] and digest(data) == source["sha256"],
                    "upstream bytes differ from source record",
                )
                fetched += 1

    listed = {}
    for entry in manifest["native_files"]:
        name = entry["path"]
        need(name.startswith("native/"), "native source path required")
        need(name not in listed, "duplicate native source")
        listed[name] = entry
        need(
            digest(local_bytes(root, name)) == entry["sha256"],
            "native source review has drifted",
        )
        need(set(entry["sources"]) <= projects.keys(), "unknown source project")
        need(
            entry["status"]
            in {"recorded-adaptation", "reviewed-baseline", "unresolved"},
            "invalid source status",
        )

    for directory in ("native/kernels", "native/src", "native/include"):
        for path in (root / directory).rglob("*"):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            text = local_bytes(root, path.relative_to(root).as_posix()).decode("utf-8")
            if any(PROJECT.search(comment) for comment in COMMENTS.findall(text)):
                need(
                    path.relative_to(root).as_posix() in listed,
                    "credited native source is unrecorded",
                )

    gaps = manifest["unresolved"]
    need(len({gap["id"] for gap in gaps}) == len(gaps), "duplicate source gap")
    affected = set()
    for gap in gaps:
        need(
            gap["reason"] and gap["resolution"],
            "source gap needs evidence and follow-up",
        )
        need(set(gap["files"]) <= listed.keys(), "source gap names an unknown file")
        affected.update(gap["files"])
    need(
        affected
        == {name for name, entry in listed.items() if entry["status"] == "unresolved"},
        "unresolved source files and gaps disagree",
    )
    for retired in manifest.get("retired_sources", []):
        need(retired["reason"] and retired["resolution"], "retirement needs evidence")
        need(
            re.fullmatch(r"[0-9a-f]{40}", retired["control_revision"]),
            "retirement needs a full control revision",
        )
        for name in retired["removed_files"]:
            need(
                not (root / relative(name)).exists(),
                "removed source has returned",
            )
        for control in retired["control_files"]:
            relative(control["path"])
            need(
                re.fullmatch(r"[0-9a-f]{64}", control["sha256"]),
                "retirement needs a control hash",
            )
    retired_symbols = manifest.get("retired_shader_symbols", [])
    for symbol in retired_symbols:
        need(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol), "invalid retired symbol")
    if retired_symbols:
        pattern = re.compile(r"\b(?:" + "|".join(retired_symbols) + r")\b")
        for path in (root / "native/kernels").glob("*.metal"):
            need(
                not pattern.search(path.read_text()),
                "removed shader implementation has returned",
            )
    model = manifest["model_licence"]
    need(
        digest(local_bytes(root, model["file"])) == model["sha256"],
        "retained model licence differs from source record",
    )
    need(
        digest(local_bytes(root, "profiles/flash-next-assets.json"))
        == model["asset_manifest_sha256"],
        "model assets changed without a licence review",
    )
    need(
        model["review_status"]
        in {"owner-review-required", "confirmed", "runtime-only"},
        "invalid model licence review status",
    )
    if model["review_status"] == "runtime-only":
        scope = model["distribution_scope"]
        need(
            all(
                scope[key] is False
                for key in (
                    "weights",
                    "derived_packs",
                    "derived_text",
                    "commercial_model_authorisation",
                )
            )
            and bool(scope["review"]),
            "runtime publication must grant no model rights or distribute model assets",
        )
    need(
        re.fullmatch(
            r"https://huggingface.co/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/raw/[0-9a-f]{40}/LICENSE",
            model["source"],
        ),
        "model licence needs a pinned source document",
    )
    if fetch:
        with (opener or urllib.request.urlopen)(
            model["source"], timeout=30
        ) as response:
            data = response.read(MAX_SOURCE + 1)
        need(
            digest(data) == model["sha256"],
            "upstream model licence differs from retained text",
        )
    standard = manifest["rust_standard_library"]
    toolchain = tomllib.loads(local_bytes(root, "rust/rust-toolchain.toml").decode())
    need(
        standard["toolchain"] == toolchain["toolchain"]["channel"],
        "Rust toolchain changed without reviewing library notices",
    )
    need(
        digest(local_bytes(root, standard["file"])) == standard["sha256"],
        "Rust library notice differs from source record",
    )
    need(
        re.fullmatch(r"[0-9a-f]{40}", standard["rustc_commit"]),
        "full Rust source commit required",
    )
    return {
        "projects": len(projects),
        "native_files": len(listed),
        "upstream_files_fetched": fetched,
        "unresolved": [gap["id"] for gap in gaps],
        "retired_source_claims": [
            row["id"] for row in manifest.get("retired_sources", [])
        ],
        "model_licence_review": model["review_status"],
        "rust_toolchain": standard["toolchain"],
        "publication_ready": not gaps
        and model["review_status"] in {"confirmed", "runtime-only"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true", help="verify pinned public source bytes"
    )
    parser.add_argument(
        "--publication", action="store_true", help="refuse unresolved source claims"
    )
    args = parser.parse_args()
    try:
        result = validate(ROOT, fetch=args.fetch)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"source review failed: {type(error).__name__}\n")
    print(json.dumps(result, sort_keys=True))
    return 2 if args.publication and not result["publication_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
