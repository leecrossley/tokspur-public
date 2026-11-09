# SPDX-License-Identifier: Apache-2.0
"""The native/ build-input rule behind native_sources in build-manifest.json.

scripts/bootstrap.py, scripts/stage_owned_native_package.py and rust/crates/tokspur-cli/src/verify.rs each implement the rule, because the bootstrap records the inventory, the Rust CLI re-checks it before every serve and the staging script re-checks it before packaging.

These tests keep the three together, hold them to one shared corpus, and require every file under native/ to be classified, so a new file type cannot silently leave the pinned identity.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests/fixtures/native-build-inputs.txt"
VERIFY_RS = ROOT / "rust/crates/tokspur-cli/src/verify.rs"

# Each scope names one rule. Bumping the scope is part of changing the rule: add the new scope with the digest of the new lists below and keep the earlier entries as history.
RULE_DIGESTS = {
    "build-inputs-v2": (
        "5829ccd1c0e85f16d141c95f1b00bb7b28253fd9d22a32321ea00455087b8d58"
    ),
    "build-inputs-v1": (
        "44642549887d6fc92af77780daa3d7cc15e0c2f101cfe0f2e8fe90cf02fde1f9"
    ),
}

# Names that carry licence text must stay pinned wherever they sit under native/.
LICENCE_LIKE = re.compile(r"licen[cs]e|copying|notice", re.IGNORECASE)


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bootstrap():
    return load("bootstrap_native_rule", "scripts/bootstrap.py")


@pytest.fixture(scope="module")
def stage():
    return load("stage_native_rule", "scripts/stage_owned_native_package.py")


def corpus() -> list[tuple[bool, str]]:
    cases = []
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        verdict, path = line.split(" ", 1)
        assert verdict in ("pinned", "unpinned"), line
        cases.append((verdict == "pinned", path))
    return cases


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o600)


def build_tree(base: Path) -> dict[str, str]:
    """Write every corpus path under base and return the pinned inventory."""
    expected = {}
    for pinned, relative in corpus():
        data = (relative + "\n").encode()
        write(base / relative, data)
        if pinned:
            expected[relative] = digest(data)
    return expected


def rust_strings(source: str, name: str) -> list[str]:
    match = re.search(rf"const {name}: &\[&str\] = &\[(.*?)\];", source, re.DOTALL)
    assert match, f"{name} is missing from verify.rs"
    return re.findall(r'"([^"\\]*)"', match.group(1))


def rust_scope(source: str) -> str:
    match = re.search(r'pub const NATIVE_SOURCE_SCOPE: &str = "([^"\\]*)";', source)
    assert match, "NATIVE_SOURCE_SCOPE is missing from verify.rs"
    return match.group(1)


def live_files() -> list[str]:
    found = []
    for directory, _names, files in os.walk(ROOT / "native"):
        for name in files:
            if name != ".DS_Store":
                found.append((Path(directory) / name).relative_to(ROOT).as_posix())
    return sorted(found)


def test_corpus_is_well_formed():
    cases = corpus()
    paths = [path for _pinned, path in cases]
    assert len(paths) == len(set(paths))
    assert all(path.startswith("native/") for path in paths)
    assert {pinned for pinned, _path in cases} == {True, False}
    # Paths that differ only in case must not share a directory, or the tree cannot be built on a case-insensitive filesystem.
    folded = [(Path(path).parent.as_posix(), Path(path).name.lower()) for path in paths]
    assert len(folded) == len(set(folded))


def test_python_rules_follow_the_corpus(bootstrap, stage):
    for pinned, path in corpus():
        name = path.rsplit("/", 1)[-1]
        assert bootstrap.is_native_build_input(name) is pinned, path
        assert stage.is_native_build_input(name) is pinned, path


def test_the_three_implementations_carry_the_same_lists(bootstrap, stage):
    source = VERIFY_RS.read_text(encoding="utf-8")
    suffixes = sorted(bootstrap.NATIVE_BUILD_INPUT_SUFFIXES)
    names = sorted(bootstrap.NATIVE_BUILD_INPUT_NAMES)
    assert sorted(stage.NATIVE_BUILD_INPUT_SUFFIXES) == suffixes
    assert sorted(stage.NATIVE_BUILD_INPUT_NAMES) == names
    assert sorted(rust_strings(source, "NATIVE_BUILD_INPUT_SUFFIXES")) == suffixes
    assert sorted(rust_strings(source, "NATIVE_BUILD_INPUT_NAMES")) == names
    assert (
        bootstrap.NATIVE_SOURCE_SCOPE == stage.NATIVE_SOURCE_SCOPE == rust_scope(source)
    )
    # A repeated entry would hide a typo in one copy.
    assert len(set(suffixes)) == len(suffixes)
    assert len(set(names)) == len(names)


def test_the_scope_names_the_rule_it_was_recorded_for(bootstrap):
    canonical = json.dumps(
        {
            "scope": bootstrap.NATIVE_SOURCE_SCOPE,
            "suffixes": sorted(bootstrap.NATIVE_BUILD_INPUT_SUFFIXES),
            "names": sorted(bootstrap.NATIVE_BUILD_INPUT_NAMES),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert bootstrap.NATIVE_SOURCE_SCOPE in RULE_DIGESTS, (
        "New scope: record the digest of its rule in RULE_DIGESTS"
    )
    assert digest(canonical.encode()) == RULE_DIGESTS[bootstrap.NATIVE_SOURCE_SCOPE], (
        "The build-input lists changed under an unchanged scope. A changed rule "
        "changes what the pinned native source identity means, so bump "
        "NATIVE_SOURCE_SCOPE in scripts/bootstrap.py, "
        "scripts/stage_owned_native_package.py and "
        "rust/crates/tokspur-cli/src/verify.rs, update "
        "tests/fixtures/native-build-inputs.txt and add the new digest here"
    )


def test_collectors_build_one_inventory_from_the_corpus(
    bootstrap, stage, tmp_path, monkeypatch
):
    tree = tmp_path.resolve()
    expected = build_tree(tree)
    monkeypatch.setattr(bootstrap, "ROOT", tree)
    assert bootstrap.collect_native_sources() == expected
    assert stage.native_sources(tree, build_inputs_only=True) == expected
    # The older regression receipts keep hashing everything under native/.
    everything = stage.native_sources(tree)
    assert set(expected) < set(everything)
    assert "native/README.md" in everything


def test_documentation_changes_leave_the_inventory_alone(
    bootstrap, tmp_path, monkeypatch
):
    tree = tmp_path.resolve()
    write(tree / "native/Makefile", b"# build\n")
    write(tree / "native/src/owned.cpp", b"// source\n")
    write(tree / "native/README.md", b"notes\n")
    write(tree / "native/PROVENANCE.md", b"provenance\n")
    monkeypatch.setattr(bootstrap, "ROOT", tree)
    before = bootstrap.collect_native_sources()
    assert sorted(before) == ["native/Makefile", "native/src/owned.cpp"]

    write(tree / "native/README.md", b"edited notes\n")
    write(tree / "native/PROVENANCE.md", b"edited provenance\n")
    write(tree / "native/NEW-NOTES.md", b"added notes\n")
    (tree / "native/README.md").unlink()
    assert bootstrap.collect_native_sources() == before

    write(tree / "native/src/owned.cpp", b"// edited source\n")
    changed = bootstrap.collect_native_sources()
    assert changed != before
    assert sorted(changed) == sorted(before)
    write(tree / "native/src/added.metal", b"// added kernel\n")
    assert "native/src/added.metal" in bootstrap.collect_native_sources()
    (tree / "native/Makefile").unlink()
    assert "native/Makefile" not in bootstrap.collect_native_sources()


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "fifo"])
@pytest.mark.parametrize("module", ["bootstrap", "stage"])
def test_unsafe_entries_are_refused_even_when_named_like_notes(
    bootstrap, stage, tmp_path, monkeypatch, module, kind
):
    tree = tmp_path.resolve()
    write(tree / "native/Makefile", b"# build\n")
    write(tree / "native/src/owned.cpp", b"// source\n")
    if kind == "file_symlink":
        (tree / "native/NOTES.md").symlink_to(tree / "native/Makefile")
    elif kind == "directory_symlink":
        (tree / "native/notes.md").symlink_to(tree / "native/src")
    else:
        os.mkfifo(tree / "native/NOTES.md", 0o600)
    if module == "bootstrap":
        monkeypatch.setattr(bootstrap, "ROOT", tree)
        with pytest.raises(ValueError, match="Unsafe"):
            bootstrap.collect_native_sources()
    else:
        with pytest.raises(RuntimeError, match="Unsafe native source entry"):
            stage.native_sources(tree, build_inputs_only=True)


def test_a_tree_of_notes_only_is_an_empty_inventory(
    bootstrap, stage, tmp_path, monkeypatch
):
    tree = tmp_path.resolve()
    write(tree / "native/README.md", b"notes\n")
    write(tree / "native/PROVENANCE.md", b"provenance\n")
    monkeypatch.setattr(bootstrap, "ROOT", tree)
    with pytest.raises(ValueError, match="Empty native source inventory"):
        bootstrap.collect_native_sources()
    with pytest.raises(RuntimeError, match="Empty native source inventory"):
        stage.native_sources(tree, build_inputs_only=True)


def test_source_only_manifest_records_the_scope(bootstrap, tmp_path, monkeypatch):
    tree = tmp_path.resolve()
    write(tree / "native/Makefile", b"# build\n")
    write(tree / "native/README.md", b"notes\n")
    write(tree / "rust/Cargo.toml", b"[workspace]\n")
    write(tree / "rust/crates/demo/src/lib.rs", b"// demo\n")
    monkeypatch.setattr(bootstrap, "ROOT", tree)
    # tests/test_cli_lifecycle.py purges and re-imports tokspur while pytest collects, so patch the module that bootstrap bound rather than one imported at the top of this file.
    monkeypatch.setattr(sys.modules["tokspur.local"], "ROOT", tree)
    monkeypatch.setattr(sys, "argv", ["bootstrap.py", "--source-only"])
    previous = os.umask(0o022)
    try:
        bootstrap.main()
    finally:
        os.umask(previous)
    manifest = json.loads(
        (tree / ".tokspur/receipts/source-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == 2
    assert manifest["native_source_scope"] == bootstrap.NATIVE_SOURCE_SCOPE
    assert list(manifest["native_sources"]) == ["native/Makefile"]


def test_every_file_under_native_is_a_build_input_or_markdown(bootstrap):
    unclassified = [
        path
        for path in live_files()
        if not bootstrap.is_native_build_input(path.rsplit("/", 1)[-1])
        and not path.endswith(".md")
    ]
    assert not unclassified, (
        "These files are neither pinned build inputs nor Markdown, so the "
        f"manifest would silently leave them out: {unclassified}. Delete them, "
        "or list their type in the build-input rule in scripts/bootstrap.py, "
        "scripts/stage_owned_native_package.py and "
        "rust/crates/tokspur-cli/src/verify.rs, bump NATIVE_SOURCE_SCOPE and "
        "extend tests/fixtures/native-build-inputs.txt"
    )


def test_licence_texts_under_native_stay_pinned(bootstrap):
    unpinned = [
        path
        for path in live_files()
        if LICENCE_LIKE.search(path.rsplit("/", 1)[-1])
        and not bootstrap.is_native_build_input(path.rsplit("/", 1)[-1])
    ]
    assert not unpinned, f"Licence texts must stay in the pinned identity: {unpinned}"


def test_live_inventory_keeps_licences_and_drops_notes(bootstrap):
    inventory = bootstrap.collect_native_sources()
    assert set(inventory) == {
        path
        for path in live_files()
        if bootstrap.is_native_build_input(path.rsplit("/", 1)[-1])
    }
    for path in (
        "native/Makefile",
        "native/data/UNICODE-LICENSE.txt",
        "native/third_party/llama-cpp/LICENSE",
    ):
        assert path in inventory, path
    notes = [path for path in live_files() if path.endswith(".md")]
    assert "native/PROVENANCE.md" in notes
    assert not set(notes) & set(inventory)


def test_the_files_staging_packages_as_notices_exist():
    for path in ("LICENSE", "native/PROVENANCE.md", "native/data/UNICODE-LICENSE.txt"):
        assert (ROOT / path).is_file(), path
