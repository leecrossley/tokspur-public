#!/bin/sh
# Runs both assertion suites and records their section counts with a dedicated CPU reference build. The suites have different test inventories. Use server_diff.py for byte-level parity. Outputs stay under build/rust-diff/serve-sections.
set -eu

root=$(cd "$(dirname "$0")/../../.." && pwd)
out="$root/build/rust-diff/serve-sections"
objects="$out/native/obj/base"
mkdir -p "$out"
make -C "$root/native" -j2 NATIVE_CONFIG=base ENABLE_METAL=0 BUILD="$out/native" "$out/native/test-serve_protocol"

python3 -I - "$root/native/tests/test_serve_protocol.cpp" "$out/test_serve_protocol_sections.cpp" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
for call, label in (
    ("worker_ready_admission();", "s1"),
    ("startup_execution_policy();", "s2"),
    ("broker_sessions();", "s3"),
):
    line = "        " + call + "\n"
    if source.count(line) != 1:
        raise SystemExit(f"cannot place section {label}")
    source = source.replace(line, f'        {call} std::cout << "{label} " << checks << "\\n";\n')
Path(sys.argv[2]).write_text(source)
PY

set -- "$objects"/*.o
c++ -I"$root/native/include" -DTOKSPUR_HAS_METAL=1 -std=c++20 -O2 -g -Wall -Wextra -Wpedantic \
    -Werror -pthread "$out/test_serve_protocol_sections.cpp" "$@" -o "$out/test-serve_protocol-sections"
"$out/test-serve_protocol-sections" > "$out/cpp.txt"

cd "$root/rust"
TOKSPUR_SERVE_SECTIONS=1 cargo test --quiet --frozen -p tokspur-server --test serve_protocol > "$out/rust-run.txt"
grep -E '^(s[0-9]|PASS)' "$out/rust-run.txt" > "$out/rust.txt"
python3 -I - "$out" <<'PY'
import json
import re
import sys
from pathlib import Path

directory = Path(sys.argv[1])
report = {}
for name in ("cpp", "rust"):
    text = (directory / f"{name}.txt").read_text()
    sections = re.findall(r"^s([123]) ([0-9]+)$", text, re.MULTILINE)
    final = re.search(r"^PASS owned serving protocol checks=([0-9]+)", text, re.MULTILINE)
    if [label for label, _ in sections] != ["1", "2", "3"] or final is None:
        raise SystemExit(f"{name}: incomplete section run")
    values = [int(count) for _, count in sections] + [int(final[1])]
    if any(a <= 0 or a >= b for a, b in zip(values, values[1:])):
        raise SystemExit(f"{name}: invalid section counts")
    report[name] = dict(zip(("readiness", "execution", "broker", "complete"), values))
(directory / "sections.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, sort_keys=True))
print("Both assertion suites passed. Section counts describe their separate test inventories.")
PY
