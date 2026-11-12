# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pins(text):
    return {
        line.split("==")[0].lower(): line.split("==")[1].split()[0]
        for line in text.splitlines()
        if "==" in line and not line.startswith("#")
    }


def test_lock_hashes_every_constraint_at_the_same_version():
    constraints = pins((ROOT / "constraints.txt").read_text())
    lock_text = (ROOT / "requirements.lock").read_text()
    assert pins(lock_text) == constraints
    for block in lock_text.split("\n\n")[1:]:
        assert "--hash=sha256:" in block
