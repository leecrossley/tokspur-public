import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evidence_index", ROOT / "scripts/evidence_index.py"
)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def history(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "test")
    (repo / "a").write_text("a")
    git(repo, "add", "a")
    git(repo, "commit", "-q", "-m", "a")
    old = git(repo, "rev-parse", "HEAD")
    (repo / "b").write_text("b")
    git(repo, "add", "b")
    git(repo, "commit", "-q", "-m", "b")
    return repo, old, git(repo, "rev-parse", "HEAD")


def tree(tmp_path, old, new):
    receipts = tmp_path / "receipts"
    (receipts / "nested").mkdir(parents=True)
    (receipts / "served.json").write_text(
        json.dumps(
            {
                "scope": "served_correctness",
                "source_revision": new,
                "source_clean": True,
                "verdict": "pass",
                "worker_sha256": "1" * 64,
                "artifacts": {"metallib": "2" * 64},
            }
        )
    )
    (receipts / "old-host.json").write_text(
        json.dumps({"scope": "host_test", "source_revision": old, "result": "pass"})
    )
    # Named like a speed receipt and claiming speed in its text, but with no scope.
    (receipts / "nested" / "served-speed.log").write_text("served_speed: 99 tok/s\n")
    (receipts / "nested" / "bad.json").write_text(json.dumps({"scope": "fastest"}))
    (receipts / "stream.jsonl").write_text('{"outcome": "complete"}\n')
    return receipts


def by_name(rows):
    return {Path(row["path"]).name: row for row in rows}


def test_index_rows_take_grades_only_from_explicit_scope(tmp_path):
    repo, old, new = history(tmp_path)
    rows = by_name(evidence.index([tree(tmp_path, old, new)], repo))
    assert set(rows) == {
        "served.json",
        "old-host.json",
        "served-speed.log",
        "bad.json",
        "stream.jsonl",
    }
    served = rows["served.json"]
    assert served["grade"] == "served_correctness"
    assert served["result"] == "pass" and served["clean"] is True
    assert served["artifacts"] == {"metallib": "2" * 64, "worker_sha256": "1" * 64}
    assert served["stale"] is False
    assert len(served["sha256"]) == 64 and served["kind"] == "json"
    assert rows["old-host.json"]["grade"] == "host_test"
    assert rows["old-host.json"]["stale"] is True
    assert rows["served-speed.log"]["grade"] == "unknown"
    assert rows["served-speed.log"]["stale"] is None
    assert rows["bad.json"]["grade"] == "unknown"
    assert rows["bad.json"]["reason"] == "invalid scope"
    assert rows["stream.jsonl"]["grade"] == "unknown"


def test_sidecar_scope_grades_a_log_and_is_not_indexed_itself(tmp_path):
    repo, old, new = history(tmp_path)
    (tmp_path / "logs").mkdir()
    log = tmp_path / "logs" / "base.log"
    log.write_text("All tests passed\nexit 0\n")
    sidecar = evidence.write_scope(
        log, "host_test", new, True, "pass", None, {"worker": "3" * 64}
    )
    assert json.loads(sidecar.read_text())["receipt_sha256"] == evidence.sha256(log)
    rows = evidence.index([tmp_path / "logs"], repo)
    assert [Path(row["path"]).name for row in rows] == ["base.log"]
    # A walk over a tree with a Git directory skips it.
    assert all(
        ".git" not in Path(r["path"]).parts for r in evidence.index([tmp_path], repo)
    )
    assert rows[0]["grade"] == "host_test" and rows[0]["stale"] is False
    assert rows[0]["artifacts"] == {"worker": "3" * 64}


def test_a_sidecar_for_other_bytes_does_not_grade(tmp_path):
    repo, _, new = history(tmp_path)
    log = tmp_path / "run.log"
    log.write_text("exit 0\n")
    evidence.write_scope(log, "served_speed", new, True, "pass", None, {})
    log.write_text("exit 0\nappended later\n")
    row = evidence.index([log], repo)[0]
    assert row["grade"] == "unknown"
    assert row["reason"] == "scope sidecar is for other receipt bytes"


def test_unknown_revision_is_not_called_fresh(tmp_path):
    repo, _, _ = history(tmp_path)
    receipt = tmp_path / "r.json"
    receipt.write_text(json.dumps({"scope": "host_test", "source_revision": "f" * 40}))
    assert evidence.index([receipt], repo)[0]["stale"] is None


def test_scope_refuses_an_unknown_grade(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("x")
    try:
        evidence.write_scope(log, "served", None, None, None, None, {})
    except evidence.IndexError_ as error:
        assert "scope must be one of" in str(error)
    else:
        raise AssertionError("an unknown grade was accepted")


def test_main_indexes_explicit_paths(tmp_path, capsys):
    repo, old, new = history(tmp_path)
    receipts = tree(tmp_path, old, new)
    assert evidence.main(["index", str(receipts), "--repo", str(repo)]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 5
