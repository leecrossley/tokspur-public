#!/usr/bin/env python3
"""Byte-level differential of the C++ and Rust tokspur-server binaries.

Both servers run, one after the other in the same directory, against the
same deterministic fake branch8 worker: the C++ private test binary in its
`target` mode, behind a recording shim. Each run gets the identical HTTP
request script (explicit seeds, so the wire carries no random seed).

Compared byte for byte:
  * the worker argv the server builds;
  * every byte the server wrote to the worker, and every byte it read back;
  * the server's stdout protocol lines and its exit status;
  * every raw HTTP response, headers included.
Only two fields are random or wall-clock by design in both servers and are
masked after a strict shape check: the `chatcmpl-<32 hex>` response ID and the
`"created":<seconds>` value. The ephemeral port in the ready line is masked
too. Anything else that differs is reported, never normalized.

The pinned tokenizer and template are read from TOKSPUR_DIFF_ASSETS or the derived-text directory from adoption. Shards can sit beside those files or under split/ or UD-Q4_K_XL/. Without an override, the adoption receipt supplies the shard paths.

Model-free: the servers parse the four GGUF headers and the pinned tokenizer
and template, as at real startup, but the fake worker never loads weights
and nothing touches the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ASSET_OVERRIDE = os.environ.get("TOKSPUR_DIFF_ASSETS")
ASSETS = Path(ASSET_OVERRIDE) if ASSET_OVERRIDE else ROOT / ".tokspur/state/gguf/derived-text"
SHIM = r"""#!/usr/bin/python3 -I
# Recording shim between a tokspur-server and the fake branch8 target. It
# records the exact bytes each side sees and swaps token IDs 112<->78 and
# 108<->74 in ID-valued fields, so the fake's planned fixture text "ok"
# arrives as the real vocabulary's "o" and "k" (a consistent bijection in
# both directions, applied textually so no JSON is reformatted). A real
# target (the cohort) is named in `target` and runs with no swap.
import os, re, subprocess, sys, threading
d = os.path.dirname(os.path.abspath(sys.argv[0]))
SWAPPING = os.path.exists(os.path.join(d, "swap"))
with open(os.path.join(d, "argv.bin"), "wb") as out:
    out.write(b"\0".join(os.fsencode(a) for a in sys.argv) + b"\0")
SWAP = {b"112": b"78", b"78": b"112", b"108": b"74", b"74": b"108"}
number = lambda m: SWAP.get(m.group(0), m.group(0))
scalar = re.compile(rb'"(id|stop_id)":([0-9]+)')
array = re.compile(rb'"(prompt_ids|tokens|generated_ids|stop_ids)":\[([0-9,]*)\]')
def swap(line):
    if not SWAPPING:
        return line
    line = scalar.sub(lambda m: b'"' + m.group(1) + b'":' + SWAP.get(m.group(2), m.group(2)), line)
    return array.sub(lambda m: b'"' + m.group(1) + b'":[' + re.sub(rb"[0-9]+", number, m.group(2)) + b"]", line)
named = os.path.join(d, "target")
executable = open(named).read().strip() if os.path.exists(named) else os.path.join(d, "fake")
fake = subprocess.Popen([executable] + sys.argv[1:], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def forward():
    with open(os.path.join(d, "wire-in"), "ab") as log:
        for line in sys.stdin.buffer:
            log.write(line); log.flush()
            fake.stdin.write(swap(line)); fake.stdin.flush()
    fake.stdin.close()
threading.Thread(target=forward, daemon=True).start()
with open(os.path.join(d, "wire-out"), "ab") as log:
    for line in fake.stdout:
        line = swap(line)
        log.write(line); log.flush()
        sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
# Output and logs have been flushed and the exact child is reaped. Exit without finalising a daemon thread blocked on buffered stdin.
os._exit(fake.wait())
"""
ID = re.compile(rb"chatcmpl-[0-9a-f]{32}")
CREATED = re.compile(rb'"created":[0-9]{1,20}')
PORT = re.compile(rb"http://127\.0\.0\.1:[0-9]{1,5}/v1")
# The real target's host memory telemetry (target.cpp describe(Memory)) is read from the OS on every report and differs between runs; the cohort masks exactly these fields, textually, and counts every masked value.
HOST_FIELDS = (
    "physical_footprint_bytes", "lifetime_peak_footprint_bytes", "free_plus_inactive_hint_bytes",
    "free_plus_file_backed_hint_bytes", "swap_used_bytes", "pressure_level",
)
HOST = re.compile(rb'"(' + "|".join(HOST_FIELDS).encode() + rb')":[0-9]{1,20}')
LOGIT = re.compile(rb'"hc_exact_final_logit_sha256":"[0-9a-f]{64}"')
GENERATED = re.compile(rb'"generated_ids":\[[0-9]')

# Schema 3 excludes declared volatile telemetry and validates/renames the per-worker broker identity only in the comparison view. The raw records are retained. Numeric model values and work/state counts remain byte comparisons; a clock-looking name inside nested data is untouched.
COHORT_MASK_SCHEMA = 3
COHORT_CLOCKS = {
    "ready": {"weight_setup_ms"},
    "control_done": {"control_wall_ms"},
    "generation_round": {
        "round_wall_ms", "lookup_and_publication_ms", "gather_wall_ms",
        "verification_including_recovery_wall_ms", "recovery_batch_wall_ms",
        "fallback_wall_ms", "batch_wall_ms", "batch_gpu_ms",
        "useful_tokens_per_second", "accepted_tokens_per_second",
    },
    "token": {"batch_wall_ms", "batch_gpu_ms"},
    "done": {
        "lookup_round_wall_ms", "prefill_wall_ms", "request_wall_ms",
        "generation_batch_wall_ms", "generation_batch_gpu_ms",
        "checkpoint_capture_wall_ms", "checkpoint_restore_wall_ms",
        "lookup_useful_tokens_per_second", "lookup_accepted_tokens_per_second",
    },
}


def cohort_wire(data: bytes, *, expected_epoch: int | None = None, incoming: bool = False) -> tuple[bytes, dict]:
    """Mask admitted root telemetry without reserialising any other bytes."""
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result

    def constant(_value):
        raise ValueError("non-finite JSON constant")

    decoder = json.JSONDecoder(object_pairs_hook=unique, parse_float=finite, parse_constant=constant)
    rows, counts, thermals = [], {}, []
    epoch, epoch_count, ready_count = expected_epoch, 0, 0
    for raw in data.splitlines(keepends=True):
        text = raw.decode("utf-8")
        start = len(text) - len(text.lstrip())
        packet, end = decoder.raw_decode(text, start)
        if not isinstance(packet, dict) or text[end:].strip():
            raise ValueError("worker record is not one JSON object")
        event = packet.get("event")
        if not incoming and not isinstance(event, str):
            raise ValueError("worker record requires a string event")
        clocks = set() if incoming else COHORT_CLOCKS.get(event, set())
        thermal = not incoming and "thermal_state" in packet and event in {"ready", "done", "generation_round", "token", "control_done"}
        if incoming:
            if epoch is None:
                raise ValueError("input epoch has no authoritative ready binding")
            if packet.get("op") != "close" and "broker_epoch" not in packet:
                raise ValueError("input command is missing broker epoch")
        elif event == "ready":
            ready_count += 1
            if ready_count != 1 or "broker_epoch" not in packet:
                raise ValueError("one authoritative broker ready is required")
        broker = "broker_epoch" in packet and (incoming or event in {"ready", "done", "generation_round", "token", "control_done", "error", "closed"})
        if broker:
            value = packet["broker_epoch"]
            if type(value) is not int or not 0 < value < (1 << 64):
                raise ValueError("broker epoch must be a nonzero uint64")
            if epoch is None:
                if incoming or event != "ready":
                    raise ValueError("broker epoch precedes its ready binding")
                epoch = value
            if value != epoch:
                raise ValueError("broker epoch changed or was not echoed")
            epoch_count += 1
        elif not incoming and event in {"done", "generation_round", "token", "control_done"}:
            raise ValueError("worker reply is missing broker epoch")
        present = clocks.intersection(packet)
        for name in present:
            value = packet[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid clock " + name)
        for small, large in (
            ("batch_gpu_ms", "batch_wall_ms"),
            ("batch_wall_ms", "round_wall_ms"),
            ("recovery_batch_wall_ms", "verification_including_recovery_wall_ms"),
            ("generation_batch_gpu_ms", "generation_batch_wall_ms"),
            ("prefill_wall_ms", "request_wall_ms"),
            ("lookup_round_wall_ms", "request_wall_ms"),
            ("lookup_accepted_tokens_per_second", "lookup_useful_tokens_per_second"),
        ):
            if small in present and large in present:
                if packet[small] > packet[large] + 1e-6:
                    raise ValueError("inconsistent clocks " + small + "/" + large)
        if thermal:
            if type(packet["thermal_state"]) is not int or packet["thermal_state"] not in (0, 1):
                raise ValueError("cohort requires nominal or fair thermal state")
            thermals.append(packet["thermal_state"])
        cursor, spans = start + 1, []
        while True:
            while text[cursor].isspace():
                cursor += 1
            if text[cursor] == "}":
                break
            key, cursor = decoder.raw_decode(text, cursor)
            while text[cursor].isspace():
                cursor += 1
            if text[cursor] != ":":
                raise ValueError("missing JSON colon")
            cursor += 1
            while text[cursor].isspace():
                cursor += 1
            value_start = cursor
            _, cursor = decoder.raw_decode(text, cursor)
            if broker and key == "broker_epoch":
                spans.append((value_start, cursor, "1"))
            elif key in clocks or (thermal and key == "thermal_state"):
                spans.append((value_start, cursor, "0"))
                counts[key] = counts.get(key, 0) + 1
            while text[cursor].isspace():
                cursor += 1
            if text[cursor] == "}":
                break
            if text[cursor] != ",":
                raise ValueError("missing JSON comma")
            cursor += 1
        for first, last, replacement in reversed(spans):
            text = text[:first] + replacement + text[last:]
        rows.append(text.encode("utf-8"))
    return b"".join(rows), {"schema": COHORT_MASK_SCHEMA, "masked_fields": counts, "thermal_states": thermals,
        "broker_epoch": epoch, "broker_epoch_occurrences": epoch_count, "ready_records": ready_count}


def cohort_exchange(result: dict) -> tuple[bytes, bytes, dict]:
    output, out = cohort_wire(result.get("wire_out", b""))
    if out["ready_records"] != 1 or out["broker_epoch"] is None:
        raise ValueError("cohort needs one authoritative broker ready")
    source, incoming = cohort_wire(result.get("wire_in", b""), expected_epoch=out["broker_epoch"], incoming=True)
    return source, output, {"epoch": out["broker_epoch"], "outgoing_occurrences": out["broker_epoch_occurrences"],
        "incoming_occurrences": incoming["broker_epoch_occurrences"], "echo_validated": True}


def cohort_epoch_report(result: dict) -> dict:
    try:
        return cohort_exchange(result)[2]
    except (ValueError, UnicodeError, IndexError, OverflowError) as error:
        return {"error": str(error)}


def cohort_telemetry_report(data: bytes) -> dict:
    try:
        return cohort_wire(data)[1]
    except (ValueError, UnicodeError, IndexError, OverflowError) as error:
        return {"schema": COHORT_MASK_SCHEMA, "error": str(error)}


def envelope_counts(result: dict) -> dict:
    data = result.get("stdout", b"") + b"".join(result.get("responses", {}).values())
    return {"response_ids": len(ID.findall(data)), "created_timestamps": len(CREATED.findall(data)), "ready_ports": len(PORT.findall(data))}


def retain_result(path: Path, result: dict) -> None:
    value = {
        key: (item.decode("utf-8", "replace") if isinstance(item, bytes) else item)
        for key, item in result.items() if key != "responses"
    }
    value["responses"] = {key: item.decode("utf-8", "replace") for key, item in result.get("responses", {}).items()}
    path.write_text(json.dumps(value, indent=1) + "\n")


def mask_host(data: bytes) -> bytes:
    return HOST.sub(lambda m: b'"' + m.group(1) + b'":<host>', data)


def host_counts(data: bytes) -> dict:
    counts: dict = {}
    for m in HOST.finditer(data):
        counts[m.group(1).decode()] = counts.get(m.group(1).decode(), 0) + 1
    return counts


class Cohort:
    """A real-model run: the unchanged C++ target, its policy and metallib."""

    def __init__(self, target: Path, policy: Path, library: Path):
        self.target, self.policy, self.library = target, policy, library


def mask(data: bytes) -> bytes:
    data = ID.sub(b"chatcmpl-<id>", data)
    data = CREATED.sub(b'"created":<created>', data)
    return PORT.sub(b"http://127.0.0.1:<port>/v1", data)


def first_line(command: list[str]) -> str:
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {type(error).__name__}"
    return output.splitlines()[0] if output else ""


def provenance() -> dict:
    """What a receipt must pin besides the binaries: source, toolchains, assets."""
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, timeout=30,
    ).stdout
    return {
        "source": first_line(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "source_modified": bool(status.strip()),
        "rustc": first_line(["rustc", "--version"]),
        "cxx": first_line(["c++", "--version"]),
        "python": sys.version.split()[0],
        "tokenizer_sha256": hashlib.sha256((ASSETS / "tokenizer.json").read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256((ASSETS / "chat_template.jinja").read_bytes()).hexdigest(),
        "shards": [path.name for path in shards()],
    }


def shards() -> list[Path]:
    for directory in (ASSETS / "split", ASSETS / "UD-Q4_K_XL", ASSETS):
        if not directory.is_dir():
            continue
        found = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix == ".gguf")
        if found:
            if len(found) != 4:
                raise SystemExit(f"expected four GGUF shards under {directory}")
            return found
    adoption = ROOT / ".tokspur/adoption.json"
    if not ASSET_OVERRIDE and adoption.is_file():
        raw = adoption.read_bytes()
        if len(raw) > 4 * 1024 * 1024:
            raise SystemExit("adoption receipt exceeds the fixture byte bound")
        paths = sorted(Path(item["path"]) for item in json.loads(raw)["target"]["assets"])
        if len(paths) == len(set(paths)) == 4 and all(path.is_absolute() and path.is_file() for path in paths):
            return paths
    raise SystemExit("prepare four pinned GGUF metadata fixtures and both text assets, then set TOKSPUR_DIFF_ASSETS, or run tokspur adopt first")


def requests(key: str) -> list[tuple[str, bytes]]:
    auth = f"Authorization: Bearer {key}\r\n"

    def post(body: object, extra: str = "", content_type: str = "application/json") -> bytes:
        payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
        return (
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"{auth}{extra}Content-Type: {content_type}\r\nContent-Length: {len(payload)}\r\n\r\n"
        ).encode() + payload

    def chat(messages: list[dict], **fields: object) -> dict:
        body = {"model": "tokspur", "messages": messages, "max_tokens": 16, "seed": 7, "enable_thinking": False}
        body.update(fields)
        return body

    first = [{"role": "user", "content": "Write a Rust add function."}]
    second = first + [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "And a test, café 🦉."},
    ]
    branch = first + [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Different branch."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    return [
        ("health", f"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n{auth}\r\n".encode()),
        ("models", f"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n{auth}\r\n".encode()),
        ("unauthorized", b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer wrong\r\n\r\n"),
        ("origin", f"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n{auth}Origin: http://x\r\n\r\n".encode()),
        ("unknown", f"GET /nope HTTP/1.1\r\nHost: 127.0.0.1\r\n{auth}\r\n".encode()),
        ("media", post(chat(first), content_type="text/plain")),
        ("json-charset", post(chat(first), content_type='Application/JSON ; Charset = "UTF-8"')),
        ("client-fields", post(chat(first, n=1, parallel_tool_calls=False, keep_alive="5m"))),
        ("translated-thinking", post(chat([
            {"role":"assistant", "content":"visible", "reasoning_content":"fixture &amp;", "thinking_blocks":[{"type":"thinking", "thinking":"fixture &amp;", "signature":""}]},
            {"role":"user", "content":"next"},
        ]))),
        ("translated-thinking-mismatch", post(chat([
            {"role":"assistant", "content":"visible", "reasoning_content":"fixture", "thinking_blocks":[{"type":"thinking", "thinking":"different"}]},
            {"role":"user", "content":"next"},
        ]))),
        ("translated-tool-index", post(chat([
            {"role":"assistant", "tool_calls":[{"id":"fixture-call", "type":"function", "index":0, "function":{"name":"read_file", "arguments":"{\"path\":\"fixture.rs\"}"}}]},
            {"role":"tool", "tool_call_id":"fixture-call", "content":"fixture"},
            {"role":"user", "content":"next"},
        ], tools=tools))),
        ("translated-tool-index-bound", post(chat([
            {"role":"assistant", "tool_calls":[{"id":"fixture-call", "type":"function", "index":128, "function":{"name":"read_file", "arguments":"{\"path\":\"fixture.rs\"}"}}]},
            {"role":"tool", "tool_call_id":"fixture-call", "content":"fixture"},
        ], tools=tools))),
        ("control-token", post(chat([{"role":"user", "content":"quoted <|im_start|>"}]))),
        ("literal-tool-response", post(chat([
            {"role":"user", "content":"first"},
            {"role":"assistant", "content":"visible", "reasoning_content":"old reasoning"},
            {"role":"user", "content":"<tool_response>quoted</tool_response>"},
        ], preserve_thinking=False))),
        ("literal-tool-metadata", post(chat(first, tools=[{
            "type":"function", "function":{
                "name":"read_file", "description":"quoted <|im_start|> <|endoftext|> <tool_call>",
                "parameters":{"type":"object", "properties":{"path":{"type":"string"}}, "required":["path"], "additionalProperties":False},
            },
        }]))),
        ("bad-json", post("{")),
        ("bad-model", post(chat(first, model="other"))),
        ("malformed-http", b"GET / HTTP/1.0\r\nHost: a\r\n\r\n"),
        ("cold", post(chat(first))),
        ("exact", post(chat(first))),
        ("append", post(chat(second))),
        ("branch", post(chat(branch))),
        ("stream", post(chat(second, stream=True))),
        ("thinking", post(chat(first, enable_thinking=True))),
        ("tools", post(chat(first, tools=tools))),
        ("stochastic-refused", post(chat(first, temperature=0.5))),
        # The client half-closes after sending: the queued job sees the disconnect before any worker command and answers nothing.
        ("cancel-before-start", post(chat(second))),
        ("after-refusal", post(chat(branch))),
        ("length", post(chat(second, max_tokens=1))),
    ]


def exchange(port: int, wire: bytes, half_close: bool = False) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=30) as client:
        client.sendall(wire)
        if half_close:
            client.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            data = client.recv(65536)
            if not data:
                return b"".join(chunks)
            chunks.append(data)


def prepare(fake: Path, directory: Path, cohort: Cohort | None = None) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    shim = directory / "branch8-target"
    shim.write_text(SHIM)
    (directory / "wire-in").write_bytes(b"")
    (directory / "wire-out").write_bytes(b"")
    shim.chmod(0o755)
    if cohort is None:
        shutil.copy2(fake, directory / "fake")
        (directory / "swap").write_text("fixture token-ID bijection\n")
        (directory / "normal").write_text("owned CPU fixture\n")
        (directory / "owned.metallib").write_text("owned CPU fixture\n")
    else:
        (directory / "target").write_text(str(cohort.target) + "\n")


def server_args(directory: Path, channel: str, cohort: Cohort | None = None) -> list[str]:
    """The fast-math startup argv after argv[0]."""
    return [
        "--worker", str(directory / "branch8-target"),
        "--policy", str(cohort.policy if cohort else directory / "normal"),
        "--library", str(cohort.library if cohort else directory / "owned.metallib"),
        "--tokenizer", str(ASSETS / "tokenizer.json"),
        "--template", str(ASSETS / "chat_template.jinja"),
        "--state-dir", str(directory / "state"),
        "--port", "0",
        "--allow-full-model", "true",
        "--native-verify8", "on",
        "--native-branch8", "on",
        "--native-qsa-stage-v8", "on",
        "--native-fast-math", channel,
        "--checkpoint-slots", "4",
        "--checkpoint-budget-bytes", "536870912",
        *map(str, shards()),
    ]


def run(server: Path, fake: Path, directory: Path, channel: str, cohort: Cohort | None = None) -> dict:
    prepare(fake, directory, cohort)
    args = [str(server), *server_args(directory, channel, cohort)]
    process = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory, env={"PATH": "/usr/bin:/bin"}
    )
    assert process.stdout is not None
    ready = process.stdout.readline()
    if not ready:
        stderr = process.stderr.read() if process.stderr else b""
        process.wait(timeout=30)
        return {"startup_failed": True, "exit": process.returncode, "stderr": stderr}
    port = int(json.loads(ready)["endpoint"].rsplit(":", 1)[1].split("/")[0])
    key = (directory / "state" / "api-key").read_text().strip()
    responses = {}
    for name, wire in requests(key):
        responses[name] = exchange(port, wire, half_close=name.startswith("cancel"))
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    # A real model unloads with its target; the fixture exits at once.
    rest, stderr = process.communicate(timeout=600 if cohort else 60)
    return {
        "argv": (directory / "argv.bin").read_bytes(),
        "wire_in": (directory / "wire-in").read_bytes(),
        "wire_out": (directory / "wire-out").read_bytes(),
        "stdout": ready + rest,
        "stderr": stderr,
        "exit": process.returncode,
        "responses": responses,
    }


def startup_cases(directory: Path) -> list[tuple[str, list[str]]]:
    """Startup argv variants: each option dropped, out-of-bound or malformed
    values, extra options and shard-list changes. Most must be refused during
    option parsing or asset checks; any that start are stopped after the ready
    line, so both servers are compared on the same terms either way."""
    base = server_args(directory, "channel")
    options = [index for index, item in enumerate(base) if item.startswith("--")]

    def replaced(option: str, value: str) -> list[str]:
        at = base.index(option)
        return base[: at + 1] + [value] + base[at + 2 :]

    # Dropping --port starts both servers on the default 127.0.0.1:18081 for the moment before SIGTERM; if that port is taken, both refuse alike.
    cases = [(f"drop {base[at]}", base[:at] + base[at + 2 :]) for at in options]
    for option, values in {
        "--port": ["-1", "65536", "", "1x", " 1", "+1", "08"],
        "--checkpoint-slots": ["5", "0", "-1"],
        "--checkpoint-budget-bytes": [str((512 << 20) + 1), "0"],
        "--native-verify8": ["yes", "off", ""],
        "--native-branch8": ["yes", "off"],
        "--native-qsa-stage-v8": ["yes", "off"],
        "--native-fast-math": ["on", ""],
        "--allow-full-model": ["false"],
        "--state-dir": ["relative-state", ""],
        "--tokenizer": ["tokenizer.json", str(ASSETS / "chat_template.jinja")],
        "--template": ["chat_template.jinja", str(ASSETS / "tokenizer.json")],
        "--policy": [str(directory / "absent")],
        "--worker": [str(directory / "absent")],
    }.items():
        cases += [(f"{option}={value!r}", replaced(option, value)) for value in values]
    shard_at = len(base) - 4
    for option, values in {
        "--model-alias": ["", "a" * 65, "bad alias", "alias.ok-1"],
        "--request-timeout-seconds": ["0", "3601"],
        "--startup-timeout-seconds": ["0", "3601"],
        "--step-timeout-seconds": ["0", "301"],
        "--stack-execution": ["recurrent-replay", "capture"],
        "--port": ["0"],
        "--bogus": ["1"],
    }.items():
        cases += [(f"add {option}={value!r}", base[:shard_at] + [option, value] + base[shard_at:]) for value in values]
    cases += [
        ("trailing option without value", base[:shard_at] + base[shard_at:] + ["--port"]),
        ("three shards", base[:-1]),
        ("duplicate shard", base[:-1] + [base[-2]]),
        ("five shards", base + [base[-1]]),
        ("absent shard", base[:-1] + [str(directory / "absent.gguf")]),
    ]
    return cases


def startup(server: Path, fake: Path, directory: Path, args: list[str]) -> dict:
    prepare(fake, directory)
    process = subprocess.Popen(
        [str(server), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory, env={"PATH": "/usr/bin:/bin"},
    )
    assert process.stdout is not None
    ready = process.stdout.readline()
    if ready:
        process.send_signal(signal.SIGTERM)
    rest, stderr = process.communicate(timeout=120)
    return {"stdout": mask(ready + rest), "stderr": stderr, "exit": process.returncode}


def compare(left: dict, right: dict, cohort: bool = False) -> list[str]:
    differences = []
    host = mask_host if cohort else (lambda data: data)
    for field in ("argv", "exit"):
        if left.get(field) != right.get(field):
            differences.append(field)
    wires, inputs = [], []
    for side, result in (("cpp", left), ("rust", right)):
        wire = result.get("wire_out", b"")
        source = result.get("wire_in", b"")
        if cohort:
            try:
                source, wire, _ = cohort_exchange(result)
            except (ValueError, UnicodeError, IndexError, OverflowError) as error:
                differences.append(f"telemetry-shape:{side}:{error}")
        wires.append(host(wire))
        inputs.append(source)
    if inputs[0] != inputs[1]:
        differences.append("wire_in")
    if wires[0] != wires[1]:
        differences.append("wire_out")
    if host(mask(left.get("stdout", b""))) != host(mask(right.get("stdout", b""))):
        differences.append("stdout")
    for name in left.get("responses", {}):
        a = left["responses"][name]
        b = right["responses"].get(name)
        if b is None or host(mask(a)) != host(mask(b)):
            differences.append(f"response:{name}")
        for data in (a, b or b""):
            ids = ID.findall(data)
            if b"chatcmpl-" in data and len(ids) != data.count(b"chatcmpl-"):
                differences.append(f"id-shape:{name}")
    return differences


def startup_main(args: argparse.Namespace, work: Path) -> int:
    cases = startup_cases(work)
    results = []
    for name, argv in cases:
        cpp = startup(args.cpp.resolve(), args.fake.resolve(), work, argv)
        rust = startup(args.rust.resolve(), args.fake.resolve(), work, argv)
        fields = [field for field in ("exit", "stdout", "stderr") if cpp[field] != rust[field]]
        results.append(
            {
                "case": name,
                "exit": cpp["exit"],
                "started": bool(cpp["stdout"]),
                "cpp_stderr": cpp["stderr"].decode("utf-8", "replace"),
                "differences": fields,
                **({"rust_stderr": rust["stderr"].decode("utf-8", "replace")} if "stderr" in fields else {}),
                **({"rust_exit": rust["exit"]} if "exit" in fields else {}),
            }
        )
    differences = [result["case"] for result in results if result["differences"]]
    report = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provenance": provenance(),
        "mode": "startup",
        "cpp_server_sha256": hashlib.sha256(args.cpp.read_bytes()).hexdigest(),
        "rust_server_sha256": hashlib.sha256(args.rust.read_bytes()).hexdigest(),
        "cases": len(results),
        "refused": sum(result["exit"] != 0 for result in results),
        "distinct_refusals": len({result["cpp_stderr"] for result in results if result["exit"] != 0}),
        "differences": differences,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    kept = args.report.parent / "receipts"
    kept.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(4).hex()
    (kept / f"{args.report.stem}-{stamp}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=1))
    return 1 if differences else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cpp", type=Path, default=ROOT / "build/http-fast-math/tokspur-server")
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--fake", type=Path, default=ROOT / "build/http-fast-math/test-http_fast_math")
    parser.add_argument("--work", type=Path, default=ROOT / "build/rust-diff/server")
    parser.add_argument("--channel", choices=("channel", "staged"), default="channel")
    parser.add_argument(
        "--rust-channel",
        choices=("channel", "staged"),
        help="negative control: run the second server with another arm; differences must appear",
    )
    parser.add_argument("--report", type=Path, default=ROOT / "build/rust-diff/server-report.json")
    parser.add_argument(
        "--target",
        type=Path,
        help="real-model cohort: the unchanged C++ fast-math branch8 target (needs --policy and --library)",
    )
    parser.add_argument("--policy", type=Path, help="cohort: the target's policy file")
    parser.add_argument("--library", type=Path, help="cohort: the target's metallib")
    parser.add_argument(
        "--startup",
        action="store_true",
        help="compare startup argv variants (refusals and their messages) instead of the request script",
    )
    args = parser.parse_args()
    for name in ("tokenizer.json", "chat_template.jinja"):
        if not (ASSETS / name).is_file():
            parser.error("pinned text fixtures are missing. Run tokspur adopt or set TOKSPUR_DIFF_ASSETS to prepared metadata/text fixtures")
    shards()
    # The servers refuse a relative --state-dir, which lives under --work.
    work = args.work.absolute()
    if args.startup:
        return startup_main(args, work)
    if (args.target is None) != (args.policy is None) or (args.target is None) != (args.library is None):
        parser.error("--target, --policy and --library go together")
    cohort = Cohort(args.target.resolve(), args.policy.resolve(), args.library.resolve()) if args.target else None
    kept = args.report.parent / "receipts"
    kept.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(4).hex()
    attempt = kept / f"{args.report.stem}-{stamp}"
    attempt.mkdir(mode=0o700)
    # One target (and so at most one model) is resident at a time: the servers run one after the other and each stops its target first.
    cpp = run(args.cpp.resolve(), args.fake.resolve(), work, args.channel, cohort)
    # Persist before prepare() resets the shared argv directory for Rust.
    retain_result(attempt / "cpp.json", cpp)
    rust = run(args.rust.resolve(), args.fake.resolve(), work, args.rust_channel or args.channel, cohort)
    retain_result(attempt / "rust.json", rust)
    differences = compare(cpp, rust, cohort is not None)
    digest = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731
    report = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provenance": provenance(),
        "harness_sha256": digest(Path(__file__).read_bytes()),
        "raw_receipts": str(attempt),
        "channel": args.channel,
        "rust_channel": args.rust_channel or args.channel,
        "cpp_server_sha256": digest(args.cpp.read_bytes()),
        "rust_server_sha256": digest(args.rust.read_bytes()),
        "mode": "cohort" if cohort else "fixture",
        "fake_sha256": None if cohort else digest(args.fake.read_bytes()),
        "cohort": None
        if cohort is None
        else {
            "target_sha256": digest(cohort.target.read_bytes()),
            "policy_sha256": digest(cohort.policy.read_bytes()),
            "library_sha256": digest(cohort.library.read_bytes()),
            "logit_digests_compared": len(LOGIT.findall(cpp.get("wire_out", b""))),
            "generations_compared": len(GENERATED.findall(cpp.get("wire_out", b""))),
            "masked_host_fields": {"cpp": host_counts(cpp.get("wire_out", b"")), "rust": host_counts(rust.get("wire_out", b""))},
            "normalization_schema": COHORT_MASK_SCHEMA,
            "volatile_wire_telemetry": {"cpp": cohort_telemetry_report(cpp.get("wire_out", b"")), "rust": cohort_telemetry_report(rust.get("wire_out", b""))},
            "envelope_masks": {"cpp": envelope_counts(cpp), "rust": envelope_counts(rust)},
            "broker_epoch_bindings": {"cpp": cohort_epoch_report(cpp), "rust": cohort_epoch_report(rust)},
            "scope": "comparison view excludes declared clocks/rates and admissible host thermals and alpha-renames one ready-bound broker epoch after validating both directions; raw bytes are retained; model values, IDs, hashes and state/work counts remain exact",
        },
        "requests": len(cpp.get("responses", {})),
        "wire_in_bytes": len(cpp.get("wire_in", b"")),
        "wire_out_bytes": len(cpp.get("wire_out", b"")),
        "wire_in_sha256": digest(cpp.get("wire_in", b"")),
        "cpp_exit": cpp.get("exit"),
        "rust_exit": rust.get("exit"),
        "differences": differences,
    }
    # A comparison only counts if both servers started and served every scripted request; two identical startup failures prove nothing.
    failures = [
        f"{side}: startup failed (exit {result.get('exit')}): "
        + result.get("stderr", b"").decode("utf-8", "replace").strip()
        for side, result in (("cpp", cpp), ("rust", rust))
        if result.get("startup_failed")
    ]
    expected = len(list(requests("")))
    if not failures and (
        len(cpp["responses"]) != expected
        or len(rust["responses"]) != expected
        or not cpp["wire_in"]
        or not cpp["wire_out"]
    ):
        failures.append("a server did not exchange every scripted request with the worker")
    if cohort and not failures and (not LOGIT.search(cpp["wire_out"]) or not GENERATED.search(cpp["wire_out"])):
        failures.append("the cohort compared no generated IDs or no final-logit digests")
    negative = args.rust_channel is not None and args.rust_channel != args.channel
    report["negative_control"] = negative
    report["harness_failures"] = failures
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    # Every run, passing or failing, also keeps its own receipt.
    (kept / f"{args.report.stem}-{stamp}.json").write_text(json.dumps(report, indent=1) + "\n")
    for side, result in (("cpp", cpp), ("rust", rust)):
        (args.report.parent / f"{args.report.stem}-{side}.json").write_text(
            json.dumps(
                {
                    key: (value.decode("utf-8", "replace") if isinstance(value, bytes) else value)
                    for key, value in result.items()
                    if key != "responses"
                }
                | {"responses": {k: v.decode("utf-8", "replace") for k, v in result.get("responses", {}).items()}},
                indent=1,
            )
        )
    print(json.dumps(report, indent=1))
    if failures:
        return 2
    if negative:
        # The control passes only if the comparison notices the other arm.
        return 0 if differences else 1
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
