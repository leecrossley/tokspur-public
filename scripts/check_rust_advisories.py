#!/usr/bin/env python3
"""Check every Cargo lockfile against OSV, which merges RustSec with the GitHub advisory database.

Adapted from the over-show app's scripts/check-rust-advisories.py at commit
ecabf371a (SHA-256 12f0c56ad83552dc7228e1fe33bc446edfa4a7be35a5431b269f5eae8d595ef9),
same owner. Changes: the lockfile and audit-config paths point at the Rust port in
rust/, the user agent names tokspur, and the code is formatted for this repository.

`cargo audit` reads RustSec alone and only the lockfile it is pointed at. Advisories filed only
with GitHub (the Tauri origin confusion fix and the 2026 rust-openssl fixes, for example) never
reach it, so this check queries OSV for every crates.io package in every lockfile the repository
ships. It shares `.cargo/audit.toml`'s ignore list, so one reviewed decision covers both checks,
and like `cargo audit` it reports unmaintained and unsound notices as warnings only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

LOCKFILES = ("rust/Cargo.lock",)
AUDIT_CONFIG = "rust/.cargo/audit.toml"
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
OSV_BATCH = "https://api.osv.dev/v1/querybatch"
OSV_VULN = "https://api.osv.dev/v1/vulns/{}"
# osv accepts up to 1000 queries per batch request.
BATCH_SIZE = 1000
TIMEOUT_SECONDS = 60

Package = tuple[str, str]
Fetch = Callable[[str, "dict | None"], dict]


class CheckError(Exception):
    pass


def locked_packages(text: str) -> set[Package]:
    """crates.io packages in a Cargo.lock; path and git packages have no registry advisories."""
    packages: set[Package] = set()
    for block in text.split("[[package]]")[1:]:
        name = re.search(r'^name = "([^"]+)"$', block, re.M)
        version = re.search(r'^version = "([^"]+)"$', block, re.M)
        source = re.search(r'^source = "([^"]+)"$', block, re.M)
        if name and version and source and source.group(1) == CRATES_IO:
            packages.add((name.group(1), version.group(1)))
    return packages


def ignored_advisories(text: str) -> set[str]:
    """the `ignore` list of `[advisories]` in `.cargo/audit.toml`, comments removed."""
    section = re.search(r"^\[advisories\]$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if not section:
        return set()
    body = "\n".join(line.split("#", 1)[0] for line in section.group(1).splitlines())
    listing = re.search(r"^ignore\s*=\s*\[(.*?)\]", body, re.M | re.S)
    if not listing:
        return set()
    return set(re.findall(r'"([^"]+)"', listing.group(1)))


def http_json(url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "tokspur-rust-advisories",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise CheckError(f"osv request failed for {url}: {error}") from error


def query(packages: Iterable[Package], fetch: Fetch) -> dict[str, set[Package]]:
    ordered = sorted(packages)
    hits: dict[str, set[Package]] = {}
    for start in range(0, len(ordered), BATCH_SIZE):
        chunk = ordered[start : start + BATCH_SIZE]
        response = fetch(
            OSV_BATCH,
            {
                "queries": [
                    {
                        "package": {"name": name, "ecosystem": "crates.io"},
                        "version": version,
                    }
                    for name, version in chunk
                ]
            },
        )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(chunk):
            raise CheckError(
                "osv returned a batch response that does not match the query"
            )
        for package, result in zip(chunk, results):
            if result.get("next_page_token"):
                raise CheckError(
                    f"osv paginated the advisories for {package[0]} {package[1]}; check it by hand"
                )
            for vulnerability in result.get("vulns") or []:
                hits.setdefault(vulnerability["id"], set()).add(package)
    return hits


def informational(record: dict) -> bool:
    # rustsec marks unmaintained, unsound and notice entries this way, per affected package in osv's export; cargo audit only warns on them.
    if (record.get("database_specific") or {}).get("informational"):
        return True
    return any(
        (affected.get("database_specific") or {}).get("informational")
        for affected in record.get("affected") or []
    )


def fixed_versions(record: dict) -> list[str]:
    fixed = set()
    for affected in record.get("affected") or []:
        for version_range in affected.get("ranges") or []:
            for event in version_range.get("events") or []:
                if "fixed" in event:
                    fixed.add(event["fixed"])
    return sorted(fixed)


def group_aliases(records: dict[str, dict]) -> list[set[str]]:
    """merge each advisory with the aliases osv returned, so a GHSA and its RUSTSEC twin report once."""
    parent = {identifier: identifier for identifier in records}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    for identifier, record in records.items():
        for alias in record.get("aliases") or []:
            if alias in parent:
                parent[find(alias)] = find(identifier)
    groups: dict[str, set[str]] = {}
    for identifier in records:
        groups.setdefault(find(identifier), set()).add(identifier)
    return sorted(groups.values(), key=lambda group: sorted(group))


def check(root: Path, fetch: Fetch = http_json, out=sys.stdout) -> int:
    ignored = ignored_advisories((root / AUDIT_CONFIG).read_text())
    findings: dict[str, set[tuple[str, str, str]]] = {}
    scanned = 0
    for lockfile in LOCKFILES:
        path = root / lockfile
        if not path.is_file():
            raise CheckError(f"{lockfile} is missing")
        packages = locked_packages(path.read_text())
        scanned += len(packages)
        for identifier, affected in query(packages, fetch).items():
            for name, version in affected:
                findings.setdefault(identifier, set()).add((lockfile, name, version))
    records = {
        identifier: fetch(OSV_VULN.format(identifier), None)
        for identifier in sorted(findings)
    }

    errors = 0
    warnings = 0
    for group in group_aliases(records):
        group_records = [records[identifier] for identifier in sorted(group)]
        active_records = [
            record for record in group_records if not record.get("withdrawn")
        ]
        if not active_records:
            continue
        aliases = set(group)
        for record in group_records:
            aliases.update(record.get("aliases") or [])
        label = ", ".join(sorted(group))
        affected = sorted(set().union(*(findings[identifier] for identifier in group)))
        places = "; ".join(
            f"{name} {version} in {lockfile}" for lockfile, name, version in affected
        )
        summary = next(
            (
                record.get("summary")
                for record in group_records
                if record.get("summary")
            ),
            "no summary",
        )
        fixed = sorted(
            set().union(*(fixed_versions(record) for record in group_records))
        )
        fix_note = f"fixed in {', '.join(fixed)}" if fixed else "no fixed release"
        if aliases & ignored:
            print(
                f"ignored  {label}: {places} ({summary}; reviewed in {AUDIT_CONFIG})",
                file=out,
            )
        elif any(informational(record) for record in active_records):
            warnings += 1
            print(f"warning  {label}: {places} ({summary}; {fix_note})", file=out)
        else:
            errors += 1
            print(f"error    {label}: {places} ({summary}; {fix_note})", file=out)
    print(
        f"checked {scanned} locked crates in {len(LOCKFILES)} lockfiles: "
        f"{errors} vulnerable, {warnings} warning(s)",
        file=out,
    )
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    arguments = parser.parse_args(argv)
    try:
        return check(arguments.root)
    except CheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
