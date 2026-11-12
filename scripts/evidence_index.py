#!/usr/bin/env python3
"""Private evidence index: one row per receipt, by source revision and scope.

Standard library only. It walks `.tokspur/receipts` and any paths given and writes
one JSON row per receipt: path, SHA-256, kind, grade, revision, clean flag,
artifact hashes, result, reason and whether the revision is stale.

A grade is taken only from an explicit scope: a `scope` member of a JSON receipt, or
a `<receipt>.scope.json` sidecar written with the `scope` command. Nothing is
inferred from a file name or from what a receipt says it measured. Without an
explicit scope the grade is `unknown`.

Grades, weakest first: source_inspection, host_test, generated_gpu_check,
served_correctness, served_speed.

A receipt is stale when its revision is an ancestor of the repository's HEAD but
not HEAD itself; `null` when the revision is missing or not in the history.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCHEMA = 1
GRADES = (
    "source_inspection",
    "host_test",
    "generated_gpu_check",
    "served_correctness",
    "served_speed",
)
SIDECAR = ".scope.json"
MAX_RECEIPT_BYTES = 256 * 1024 * 1024
RESULT_KEYS = ("result", "verdict", "outcome")


class IndexError_(Exception):
    pass


def need(condition, message):
    if not condition:
        raise IndexError_(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def kind(path):
    name = path.name
    if name.endswith(".jsonl"):
        return "jsonl"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".log"):
        return "log"
    return "other"


def _object(path):
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _artifacts(metadata):
    found = {}
    explicit = metadata.get("artifacts")
    if isinstance(explicit, dict):
        found.update({str(k): v for k, v in explicit.items() if isinstance(v, str)})
    for key, value in metadata.items():
        if (
            key.endswith("_sha256")
            and key != "receipt_sha256"
            and isinstance(value, str)
        ):
            found[key] = value
    return dict(sorted(found.items()))


def _grade(metadata):
    scope = metadata.get("scope")
    if scope is None:
        return "unknown", None
    if scope in GRADES:
        return scope, None
    return "unknown", "invalid scope"


class Git:
    def __init__(self, repo):
        self.repo = repo
        self.head = self._run("rev-parse", "HEAD") if repo else None

    def _run(self, *args):
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def stale(self, revision):
        if not self.head or not revision:
            return None
        full = self._run("rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}")
        if not full:
            return None
        if full == self.head:
            return False
        ancestor = subprocess.run(
            ["git", "-C", str(self.repo), "merge-base", "--is-ancestor", full, "HEAD"],
            capture_output=True,
        )
        return True if ancestor.returncode == 0 else None


def row(path, git):
    path = Path(path)
    need(path.stat().st_size <= MAX_RECEIPT_BYTES, f"receipt too large: {path}")
    metadata = {}
    receipt_kind = kind(path)
    if receipt_kind == "json":
        metadata = _object(path) or {}
    digest = sha256(path)
    sidecar = Path(str(path) + SIDECAR)
    mismatch = False
    if sidecar.is_file():
        side = _object(sidecar)
        need(side is not None, f"invalid scope sidecar: {sidecar}")
        mismatch = side.get("receipt_sha256") not in (None, digest)
        if not mismatch:
            metadata = {**metadata, **side}
    grade, reason = _grade(metadata)
    if mismatch:
        grade, reason = "unknown", "scope sidecar is for other receipt bytes"
    result = next((metadata[k] for k in RESULT_KEYS if k in metadata), None)
    revision = metadata.get("source_revision")
    return {
        "schema": SCHEMA,
        "path": str(path),
        "sha256": digest,
        "kind": receipt_kind,
        "grade": grade,
        "revision": revision,
        "clean": metadata.get("source_clean"),
        "artifacts": _artifacts(metadata),
        "result": result if isinstance(result, str) else None,
        "reason": reason or metadata.get("reason"),
        "stale": git.stale(revision),
    }


def receipts(paths):
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if (
                    child.is_file()
                    and not child.name.endswith(SIDECAR)
                    and ".git" not in child.relative_to(path).parts
                ):
                    yield child
        elif path.is_file() and not path.name.endswith(SIDECAR):
            yield path


def index(paths, repo):
    git = Git(repo)
    return [row(path, git) for path in receipts(paths)]


def write_scope(path, scope, revision, clean, result, reason, artifacts):
    need(scope in GRADES, f"scope must be one of {', '.join(GRADES)}")
    need(Path(path).is_file(), f"no receipt at {path}")
    sidecar = Path(str(path) + SIDECAR)
    value = {"scope": scope, "receipt_sha256": sha256(path)}
    if revision:
        value["source_revision"] = revision
    if clean is not None:
        value["source_clean"] = clean
    if result:
        value["result"] = result
    if reason:
        value["reason"] = reason
    if artifacts:
        value["artifacts"] = artifacts
    with open(sidecar, "x", encoding="utf-8") as out:
        out.write(json.dumps(value, sort_keys=True) + "\n")
    return sidecar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("index")
    build.add_argument("paths", nargs="*")
    build.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    build.add_argument("--out")
    scope = sub.add_parser("scope")
    scope.add_argument("receipt")
    scope.add_argument("--scope", required=True, choices=GRADES)
    scope.add_argument("--revision")
    clean = scope.add_mutually_exclusive_group()
    clean.add_argument("--clean", dest="clean", action="store_true", default=None)
    clean.add_argument("--dirty", dest="clean", action="store_false")
    scope.add_argument("--result")
    scope.add_argument("--reason")
    scope.add_argument("--artifact", action="append", default=[], help="NAME=SHA256")
    args = parser.parse_args(argv)
    if args.command == "scope":
        artifacts = {}
        for item in args.artifact:
            name, _, value = item.partition("=")
            need(name and len(value) == 64, f"invalid artifact {item}")
            artifacts[name] = value
        print(
            write_scope(
                args.receipt,
                args.scope,
                args.revision,
                args.clean,
                args.result,
                args.reason,
                artifacts,
            )
        )
        return 0
    paths = list(args.paths)
    default = Path(args.repo) / ".tokspur" / "receipts"
    if default.is_dir():
        paths.insert(0, str(default))
    rows = index(paths, args.repo)
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    if args.out:
        with open(args.out, "x", encoding="utf-8") as out:
            out.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
