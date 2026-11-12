#!/usr/bin/env python3
"""Private, bounded runtime-check evidence. No model downloads or campaign retries."""

import argparse
import base64
import hashlib
import json
import math
import os
import random
import selectors
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

BASE = "tokspur-execution-v3"
SAMPLING = ("greedy", "full", "topk")
CONTROLLER_SAMPLING = SAMPLING + ("coding",)
MAX_TRACE_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 8 * 1024 * 1024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def profiling_abort(done, generation_id):
    """Drain only a still-active measured generation; completed IDs are stale."""
    need(done.get("event") == "done", "profiling cleanup requires a completed packet")
    state = done.get("native_mtp_generation")
    need(
        isinstance(state, dict) and type(state.get("active")) is bool,
        "profiling cleanup requires generation state",
    )
    if not state["active"]:
        return None
    need(state.get("id") == generation_id, "profiling cleanup generation differs")
    return {
        "tokens": [],
        "generate": 0,
        "reset": False,
        "mtp_generation": {"id": generation_id, "abort": True},
    }


def profiling_controller(ready):
    need(
        ready.get("native_measured_controller") is True,
        "profiling requires measured startup negotiation",
    )
    policy = ready.get("native_controller_policy_digest")
    need(
        isinstance(policy, list)
        and len(policy) == 32
        and all(type(v) is int and 0 <= v <= 255 for v in policy),
        "profiling requires a valid startup controller digest",
    )
    # Fresh begins carry no learned observations. Prefill metadata deliberately
    # omits this object; native admission seeds it from the current profile.
    return {"controller_policy_digest": list(policy)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def private_file(path):
    root = Path(__file__).resolve().parents[1]
    resolved = Path(path).resolve()
    if resolved.is_relative_to(root):
        relative = resolved.relative_to(root)
        need(
            relative.parts
            and relative.parts[0]
            in {"build", ".tokspur", "runs", "outputs", "artifacts", "logs"},
            "private evidence must use an ignored output directory",
        )
    # An existing failed attempt is evidence, never an output to overwrite.
    return os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb"
    )


def archive(source, output):
    root = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    need(not status, "source archive requires a clean committed checkout")
    with private_file(output) as stream:
        subprocess.run(
            ["git", "-C", str(root), "archive", "--format=tar", revision],
            stdout=stream,
            check=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "schema": 1,
        "source_revision": revision,
        "source_tree": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", revision + "^{tree}"], text=True
        ).strip(),
        "git_version": subprocess.check_output(["git", "--version"], text=True).strip(),
        "sha256": digest(output),
        "bytes": Path(output).stat().st_size,
    }


def generation_command(value):
    command = value.get("mtp_generation")
    if command is not None and not command.get("abort"):
        need(
            {
                "id",
                "begin",
                "remaining",
                "committed",
                "emitted",
                "pending",
                "remaining_compute",
                "total_budget",
            }
            <= command.keys(),
            "controller replay refuses legacy or incomplete generation packets",
        )
        if not command["begin"]:
            need(
                "stage_a_controller" in command,
                "controller continuation omits live feedback",
            )
    return command


def read_exchange(path, require_controller=True):
    need(Path(path).stat().st_size <= MAX_TRACE_BYTES, "exchange byte budget exhausted")
    records = []
    for line in Path(path).read_bytes().splitlines():
        record = json.loads(line)
        need(
            record.get("schema") == 1 and record.get("direction") in {"host", "worker"},
            "invalid exchange record",
        )
        raw = base64.b64decode(record["bytes"], validate=True)
        need(
            len(raw) <= MAX_LINE_BYTES and raw.endswith(b"\n"), "invalid exchange line"
        )
        value = json.loads(raw)
        if record["direction"] == "host":
            generation_command(value)
        records.append((record["direction"], raw, value))
    need(
        records and records[0][0] == "worker" and records[0][2].get("event") == "ready",
        "exchange has no worker readiness",
    )
    need(
        not require_controller
        or any(
            d == "host"
            and (command := generation_command(v))
            and not command.get("abort")
            and v.get("generate", 0) > 0
            for d, _, v in records
        ),
        "exchange never exercises v2 generation",
    )
    return records


def capture(trace, argv, arm="unlabelled"):
    """Transparent worker wrapper for an owned Rust host. Trace both directions as exact bytes."""
    need(argv, "missing owned worker argv")
    identity = {
        "schema": 1,
        "argv": argv,
        "worker_sha256": digest(argv[0]),
        "runtime_base": BASE,
        "capture_sha256": digest(__file__),
        "capture_id": os.urandom(16).hex(),
        "arm": arm,
        "model_worker": Path(argv[0]).name
        in {"tokspur-native", "scalar-channel-target"},
    }
    source_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    identity["source_revision"] = (
        revision.stdout.strip() if revision.returncode == 0 else ""
    )
    identity["source_clean"] = status.returncode == 0 and not status.stdout
    flags_path = Path(argv[0]).parent / ".cflags"
    if flags_path.is_file():
        identity["native_build_flags"] = flags_path.read_text().split()
    if "--library" in argv:
        identity["library_path"] = argv[argv.index("--library") + 1]
        identity["library_sha256"] = digest(identity["library_path"])
    with private_file(str(trace) + ".identity.json") as manifest:
        manifest.write(json.dumps(identity, sort_keys=True).encode() + b"\n")
    # Preserve the supervised PID while the recorder establishes a separate session before exec.
    import signal

    worker_pid = os.getpid()
    worker_in, recorder_in = os.pipe()
    recorder_out, worker_out = os.pipe()
    detached_read, detached_write = os.pipe()
    sink = private_file(trace)
    recorder = os.fork()
    if recorder:
        os.close(detached_write)
        need(
            os.read(detached_read, 1) == b"R", "recorder did not establish its lifetime"
        )
        os.close(detached_read)
        sink.close()
        os.close(recorder_in)
        os.close(recorder_out)
        os.dup2(worker_in, 0)
        os.dup2(worker_out, 1)
        os.close(worker_in)
        os.close(worker_out)
        os.execvp(argv[0], argv)
    os.close(detached_read)
    os.setsid()
    os.write(detached_write, b"R")
    os.close(detached_write)
    os.close(worker_in)
    os.close(worker_out)

    # Pipe closure ends recording after the worker stops. A separate session preserves finalisation during guardian group cleanup, with a bounded grace period.
    def drain_recorder(_signum, _frame):
        signal.alarm(3)

    signal.signal(signal.SIGTERM, drain_recorder)
    lock = threading.Lock()
    failures = []
    closed = {"host": False, "worker": False}
    total = 0

    def pump(source, destination, direction):
        nonlocal total
        try:
            while True:
                raw = source.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    closed[direction] = True
                    break
                need(
                    len(raw) <= MAX_LINE_BYTES and raw.endswith(b"\n"),
                    "exchange line budget exhausted",
                )
                record = (
                    json.dumps(
                        {
                            "schema": 1,
                            "direction": direction,
                            "monotonic_ns": time.monotonic_ns(),
                            "bytes": base64.b64encode(raw).decode("ascii"),
                        },
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
                with lock:
                    total += len(record)
                    need(total <= MAX_TRACE_BYTES, "exchange byte budget exhausted")
                    sink.write(record)
                    sink.flush()
                destination.write(raw)
                destination.flush()
        except (OSError, ValueError) as error:
            failures.append(type(error).__name__)
            signal.alarm(1)
        finally:
            if direction == "host":
                destination.close()

    host = threading.Thread(
        target=pump,
        args=(sys.stdin.buffer, os.fdopen(recorder_in, "wb"), "host"),
        daemon=True,
    )
    host.start()
    pump(os.fdopen(recorder_out, "rb"), sys.stdout.buffer, "worker")
    signal.alarm(3)
    host.join(timeout=2)
    sink.close()
    closure = (
        json.dumps(
            {
                "schema": 1,
                "worker_pid": worker_pid,
                "recorder_pid": os.getpid(),
                "identity_sha256": digest(str(trace) + ".identity.json"),
                "exit_evidence": "pipe EOF; supervisor exit status required separately",
                "exchange_sha256": digest(trace),
                "host_eof": closed["host"],
                "worker_eof": closed["worker"],
                "failures": failures,
            },
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    with private_file(str(trace) + ".lifecycle.json") as lifecycle:
        lifecycle.write(closure)
    os._exit(1 if failures else 0)


def semantic(value, parent=None):
    # Only clocks/counters are excluded. Tokens, full generation/controller feedback,
    # RNG frontiers, state digests and errors remain comparison requirements.
    timing = {
        "wall_ms",
        "gpu_ms",
        "round_wall_ms",
        "draft_wall_ms",
        "draft_gpu_ms",
        "verify_gpu_ms",
        "recovery_gpu_ms",
        "gpu_profile",
        "weight_setup_ms",
    }
    host_memory = {
        "physical_footprint_bytes",
        "resident_size_bytes",
        "lifetime_peak_footprint_bytes",
        "pageins",
        "compressed_bytes",
        "swap_used_bytes",
        "free_plus_file_backed_hint_bytes",
        "free_plus_inactive_hint_bytes",
    }
    if isinstance(value, dict):
        return {
            key: semantic(item, key)
            for key, item in value.items()
            if key not in timing
            and not key.endswith("_wall_ms")
            and not key.endswith("_gpu_ms")
            and not (key == "pid" and value.get("event") == "ready")
            and not (parent == "memory" and key in host_memory)
        }
    if isinstance(value, list):
        return [semantic(item, parent) for item in value]
    return value


def comparison_packet(value, prepared_candidate=False):
    # The exact optimisation acknowledges selection in readiness and completion.
    # Keep
    # every generation, controller, state and entropy field in the comparison.
    if prepared_candidate and value.get("event") in ("ready", "done"):
        need(
            "native_prepared_validation" not in value,
            "prepared comparison requires the unchanged default reference",
        )
        return {**value, "native_prepared_validation": True}
    return value


def replay(
    trace,
    argv,
    stderr,
    require_controller=True,
    prepared_candidate=False,
    command_timeout=3600,
):
    need(
        isinstance(command_timeout, (int, float)) and 1 <= command_timeout <= 86400,
        "replay command timeout must be 1..86400 seconds",
    )
    records = read_exchange(trace, require_controller=require_controller)
    lifecycle = json.loads(Path(str(trace) + ".lifecycle.json").read_bytes())
    need(
        lifecycle.get("schema") == 1
        and lifecycle.get("exchange_sha256") == digest(trace)
        and lifecycle.get("host_eof") is True
        and lifecycle.get("worker_eof") is True
        and not lifecycle.get("failures"),
        "reference capture lacks a complete owned drain",
    )
    identity = json.loads(Path(str(trace) + ".identity.json").read_bytes())
    original = identity["argv"]
    need(
        digest(original[0]) == identity["worker_sha256"],
        "captured production worker changed",
    )
    need(
        identity.get("native_build_flags", [None])[0] == "carrier",
        "parity needs a captured production carrier build",
    )
    need(
        digest(identity["library_path"]) == identity["library_sha256"],
        "captured production library changed",
    )
    diagnostic_flags = (Path(argv[0]).parent / ".cflags").read_text().split()
    need(
        diagnostic_flags[0] == "carrier-diagnostic",
        "substitution is not the separate diagnostic build",
    )
    diagnostic_only = {
        "-DTOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC=1",
        "-DTOKSPUR_PRIVATE_BATCH_PHASE_CLOCK=1",
        "-DTOKSPUR_PRIVATE_DIAGNOSTIC_BATCH_CLOCK=1",
    }
    need(
        [f for f in diagnostic_flags[1:] if f not in diagnostic_only]
        == identity["native_build_flags"][1:],
        "diagnostic feature set differs from the production reference",
    )
    need(len(argv) >= len(original), "diagnostic argv is incomplete")
    for index, argument in enumerate(original):
        if index == 0 or (index and original[index - 1] == "--library"):
            need(
                Path(argv[index]).is_absolute() and Path(argv[index]).is_file(),
                "invalid worker/library substitution",
            )
        elif index and original[index - 1] == "--checkpoint-ssd-dir":
            if argv[index] != argument:
                need(
                    records[0][2].get("checkpoint_ssd_enabled") is True
                    and records[0][2].get("checkpoint_ssd_occupied_bytes") == 0
                    and Path(argv[index]).is_absolute(),
                    "checkpoint relocation requires an initially empty reference cache",
                )
                # Never alter the captured cache. Recreate its declared empty
                # initial state in a fresh private directory, under the same cap.
                Path(argv[index]).mkdir(mode=0o700)
        else:
            need(
                argv[index] == argument, "diagnostic argv changes a production setting"
            )
    extra = argv[len(original) :]
    need(len(extra) % 2 == 0, "incomplete diagnostic control")
    supplied_tape = None
    prepared_controls = 0
    for flag, value in zip(extra[::2], extra[1::2]):
        if flag == "--native-prepared-validation":
            need(
                prepared_candidate and value == "on",
                "reference parity cannot change prepared validation",
            )
            prepared_controls += 1
            continue
        need(
            flag
            in {
                "--carrier-diagnostic-mode",
                "--carrier-diagnostic-cost-tape",
                "--carrier-diagnostic-identity-out",
            },
            "parity replay permits disabled diagnostics and timing inputs only",
        )
        if flag == "--carrier-diagnostic-mode":
            need(value == "off", "reference parity must precede intrusive profiling")
        elif flag == "--carrier-diagnostic-cost-tape":
            supplied_tape = json.loads(Path(value).read_bytes())
            need(
                supplied_tape["reference_exchange_sha256"] == digest(trace),
                "controller timing tape belongs to another exchange",
            )
    need(
        prepared_controls == int(prepared_candidate),
        "prepared comparison requires exactly one explicit opt-in",
    )
    if (
        records[0][2].get("native_measured_controller")
        or records[0][2].get("native_mtp_depth") == 7
    ):
        expected_tape = timing_tape(trace)
        need(
            supplied_tape == expected_tape
            if expected_tape["observations_ms"]
            else supplied_tape is None,
            "timing-dependent replay requires the exact recorded observations",
        )
    with private_file(stderr) as diagnostics:
        child = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostics
        )
        try:
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)
            writer = selectors.DefaultSelector()
            os.set_blocking(child.stdin.fileno(), False)
            writer.register(child.stdin, selectors.EVENT_WRITE)
            buffered = bytearray()

            def next_line():
                deadline = time.monotonic() + command_timeout
                while b"\n" not in buffered:
                    need(
                        time.monotonic() < deadline,
                        "diagnostic replay response timeout",
                    )
                    need(
                        selector.select(max(0, deadline - time.monotonic())),
                        "diagnostic replay response timeout",
                    )
                    block = os.read(child.stdout.fileno(), 65536)
                    need(block, "diagnostic replay ended early")
                    buffered.extend(block)
                    need(
                        len(buffered) <= MAX_LINE_BYTES,
                        "diagnostic replay line budget exhausted",
                    )
                end = buffered.index(10) + 1
                line = bytes(buffered[:end])
                del buffered[:end]
                return line

            for direction, raw, expected in records:
                if direction == "host":
                    pending = memoryview(raw)
                    deadline = time.monotonic() + command_timeout
                    while pending:
                        need(
                            writer.select(max(0, deadline - time.monotonic())),
                            "diagnostic replay command timeout",
                        )
                        try:
                            pending = pending[os.write(child.stdin.fileno(), pending) :]
                        except BlockingIOError:
                            continue
                else:
                    actual = next_line()
                    need(
                        actual and len(actual) <= MAX_LINE_BYTES,
                        "diagnostic replay ended early or exceeded bounds",
                    )
                    decoded = json.loads(actual)
                    compared = comparison_packet(expected, prepared_candidate)
                    if semantic(decoded) != semantic(compared):
                        # Retain the first differing packet privately before stopping.
                        # Tokens and histories must never enter an exception or Git.
                        with private_file(str(stderr) + ".mismatch.json") as mismatch:
                            mismatch.write(
                                json.dumps(
                                    {"expected": expected, "actual": decoded}
                                ).encode()
                                + b"\n"
                            )
                        need(
                            False, "diagnostics-disabled replay differs from production"
                        )
            child.stdin.close()
            need(child.wait(timeout=30) == 0, "diagnostic replay failed")
            need(
                not buffered and not child.stdout.read(1),
                "diagnostic replay emitted unrecorded trailing output",
            )
        finally:
            selector.close()
            writer.close()
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
    return {
        "reference_sha256": digest(trace),
        "parity": True,
        "comparison": "prepared-validation"
        if prepared_candidate
        else "diagnostics-disabled",
    }


def replay_manifest(trace, cases):
    """Bound a separate carrier replay format without changing historical COST inputs."""
    records = read_exchange(trace, require_controller=False)
    need(1 <= len(cases) <= 4, "replay manifest requires one to four decoded actions")
    for case in cases:
        need(
            0 < case["context"] <= 65536 and 1 <= case["width"] <= 8,
            "replay context or width outside bounds",
        )
        need(
            1 <= case["accepted_prefix"] <= case["width"],
            "invalid replay accepted prefix",
        )
        need(case["sampling"] in CONTROLLER_SAMPLING, "unknown replay sampling mode")
    rounds = [
        value
        for direction, _, value in records
        if direction == "worker" and value.get("event") == "mtp_round"
    ]
    components = [
        value
        for direction, _, value in records
        if direction == "worker" and value.get("carrier_component") is True
    ]
    need(
        not (rounds and components),
        "replay cannot mix live rounds and target components",
    )
    need(len(rounds) <= len(cases), "recorded actions exceed the declared replay bound")
    if components:
        commands = [
            value
            for direction, _, value in records
            if direction == "host" and "carrier_component" in value
        ]
        need(
            len(components) == len(commands) == len(cases),
            "target component cases must match recorded commands and completions",
        )
        for actual, command, case in zip(components, commands, cases):
            need(
                actual.get("event") == "done"
                and actual.get("component_kind") == "target-rows"
                and actual["base"] == case["context"]
                and actual["rows"] == case["width"] == len(command["tokens"])
                and actual["accepted_prefix"]
                == case["accepted_prefix"]
                == command["carrier_component"]["accepted_prefix"]
                and actual["committed_prefix"]
                == case["context"] + case["accepted_prefix"]
                and not command.get("generate"),
                "declared target component differs from recorded execution",
            )
    elif rounds:
        need(
            len(rounds) == len(cases),
            "replay cases must match recorded speculative actions",
        )
        for actual, case in zip(rounds, cases):
            need(
                actual["base"] == case["context"]
                and actual["proposed"] + 1 == case["width"]
                and actual["accepted"] + 1 == case["accepted_prefix"],
                "declared replay case differs from recorded execution",
            )
    else:
        known = [
            v
            for d, _, v in records
            if d == "host"
            and v.get("tokens")
            and not v.get("known_prefill")
            and not v.get("reset")
        ]
        need(
            len(known) == len(cases)
            and all(
                len(v["tokens"]) == 1
                and not v.get("generate")
                and c["width"] == c["accepted_prefix"] == 1
                for v, c in zip(known, cases)
            ),
            "component replay requires a recorded forced ordinary action per case",
        )
    result = {
        "schema": 1,
        "format": "carrier-bounded-replay-v1",
        "runtime_base": BASE,
        "exchange_sha256": digest(trace),
        "cases": cases,
        "maximum_actions": 4,
        "maximum_batches_per_action": 8,
        "maximum_dispatches_per_batch": 8192,
    }
    batch_only = any(
        d == "worker"
        and v.get("event") == "ready"
        and v.get("carrier_diagnostic_batch_only") is True
        for d, _, v in records
    )
    if batch_only:
        result["format"] = "carrier-bounded-batch-totals-v1"
        result.pop("maximum_dispatches_per_batch")
        result["maximum_dispatch_records_per_batch"] = 8192
        result["dispatch_records"] = "none"
        result["per_dispatch_attribution"] = False
    return result


def timing_tape(trace):
    records = read_exchange(trace, require_controller=False)
    ready = records[0][2]
    need(
        ready.get("native_measured_controller") or ready.get("native_mtp_depth") == 7,
        "legacy MTP has no timing-dependent decision input to replay",
    )
    observations = []
    latest = {}
    covered = set()
    stops = {}
    active_generation = None
    compute_left = None
    for direction, _, value in records:
        if direction == "host":
            command = value.get("mtp_generation", {})
            if command.get("begin"):
                stops[command["id"]] = value.get("stop_ids", [])
            if command and not command.get("abort"):
                active_generation = command["id"]
                compute_left = command["remaining_compute"]
            continue
        controller = (
            value.get("native_mtp_generation", {})
            .get("stage_a_controller", {})
            .get("measured")
        )
        if ready.get("native_measured_controller") and value.get("event") == "done":
            need(
                ready.get("native_measured_timing_schema")
                == value.get("native_measured_timing_schema")
                == 1,
                "measured capture lacks complete packet timing negotiation",
            )
            packet = value.get("native_measured_decode_observations", {})
            times = packet.get("milliseconds")
            need(
                isinstance(times, list) and len(times) <= 64,
                "missing or unbounded measured packet observations",
            )
            if times:
                need(
                    controller is not None,
                    "measured packet omits its controller frontier",
                )
                generation = controller["generation"]
                first = packet.get("first_action")
                need(
                    first == latest.get(generation, 0) + 1
                    and controller["actions"] == first + len(times) - 1
                    and times[-1] == controller["last_decode_ms"],
                    "capture skipped or duplicated controller observations",
                )
                latest[generation] = controller["actions"]
                observations.extend(times)
            else:
                need(
                    packet.get("first_action") == 0,
                    "empty measured timing packet has an action",
                )
        elif not ready.get("native_measured_controller"):
            if value.get("event") == "mtp_round" and "round_wall_ms" in value:
                observations.append(value["round_wall_ms"])
                generation = active_generation
                need(
                    generation is not None
                    and compute_left is not None
                    and 0 < value["count"] <= compute_left,
                    "DFlash2 round differs from its v2 compute frontier",
                )
                covered.update(
                    (generation, value["base"] + t + 1) for t in range(value["count"])
                )
                compute_left -= value["count"]
            elif value.get("event") == "token":
                if "token_prefix" in value:
                    need(
                        (active_generation, value["token_prefix"]) in covered,
                        "unrecorded speculative or overflow token",
                    )
                else:
                    # These declared ordinary branches never call the legacy
                    # budget's timing observer. A truncated draft does, and its
                    # clock is unavailable, so it must still fail admission.
                    need(
                        compute_left is not None
                        and compute_left > 0
                        and (
                            compute_left < 2
                            or value.get("id") in stops.get(active_generation, [])
                            or value.get("committed_prefix") == 65536
                        ),
                        "legacy DFlash2 truncated action has no recorded learning clock",
                    )
                    compute_left -= 1
    need(
        not ready.get("native_measured_controller") or observations,
        "capture omits measured controller observations",
    )
    need(
        len(observations) <= 4
        and all(math.isfinite(v) and v > 0 for v in observations),
        "incomplete or unbounded controller timing capture",
    )
    return {
        "schema": 1,
        "reference_exchange_sha256": digest(trace),
        "observations_ms": observations,
    }


def ledger(serial, concurrent):
    def records(path):
        output = []
        for line in Path(path).read_bytes().splitlines():
            if line.startswith(b'{"event":"carrier_diagnostic_'):
                output.append(json.loads(line))
        actions = [r for r in output if r["event"] == "carrier_diagnostic_action"]
        need(
            1 <= len(actions) <= 4 and all(a["complete"] for a in actions),
            "missing or invalid diagnostic action closure",
        )
        need(
            not any(a.get("correctness_capture") for a in actions),
            "correctness readback cannot be used as a cost ledger",
        )
        batches = [r for r in output if r["event"] == "carrier_diagnostic_batch"]
        need(
            all(r.get("dispatch_records_enabled", True) for r in batches),
            "batch-only totals cannot establish per-dispatch family attribution",
        )
        need(
            batches
            and all(
                1 <= r["dispatches"] <= 8192 and 1 <= r["batch"] <= 8 for r in batches
            ),
            "diagnostic batch bound",
        )
        need(
            all(0 <= r["error_flag_barriers"] <= r["barriers"] for r in batches),
            "invalid diagnostic barrier accounting",
        )
        need(
            all(r["action"] in {a["action"] for a in actions} for r in batches),
            "unclosed diagnostic action",
        )
        need(
            len({a["action"] for a in actions}) == len(actions),
            "duplicate diagnostic action",
        )
        need(
            len({(r["action"], r["batch"]) for r in batches}) == len(batches),
            "duplicate diagnostic batch",
        )
        need(
            {r["action"] for r in batches} == {a["action"] for a in actions},
            "diagnostic action has no batches",
        )
        for batch in batches:
            need(
                len(batch["kernels"]) == batch["dispatches"],
                "diagnostic dispatch count differs",
            )
            need(
                type(batch["gpu_ms"]) in (int, float)
                and math.isfinite(batch["gpu_ms"])
                and batch["gpu_ms"] >= 0,
                "invalid diagnostic batch timing",
            )
        return batches

    a, b = records(serial), records(concurrent)
    need(
        all(r["serial_counter_schedule"] for r in a)
        and not any(r["serial_counter_schedule"] for r in b),
        "serial and concurrent schedules are not interchangeable",
    )
    need(
        [
            (
                r["action"],
                r["batch"],
                [(k["kernel"], k["shape"], k["layer"]) for k in r["kernels"]],
            )
            for r in a
        ]
        == [
            (
                r["action"],
                r["batch"],
                [(k["kernel"], k["shape"], k["layer"]) for k in r["kernels"]],
            )
            for r in b
        ],
        "diagnostic dispatch identities differ",
    )
    families = {}
    for batch in a:
        for kernel in batch["kernels"]:
            need(
                math.isfinite(kernel["gpu_ms"]) and kernel["gpu_ms"] >= 0,
                "invalid serial kernel timing",
            )
            families[kernel["kernel"]] = (
                families.get(kernel["kernel"], 0) + kernel["gpu_ms"]
            )
    serial_total = sum(families.values())
    batch_total = sum(r["gpu_ms"] for r in b)
    barriers = sum(r["barriers"] for r in b)
    error_flag_barriers = sum(r["error_flag_barriers"] for r in b)
    return {
        "schema": 1,
        "serial_family_ms": families,
        "serial_family_total_ms": serial_total,
        "concurrent_batch_ms": batch_total,
        "unexplained_difference_ms": batch_total - serial_total,
        "concurrent_barriers": barriers,
        "error_flag_barriers": error_flag_barriers,
        "data_only_barriers": barriers - error_flag_barriers,
        "barrier_accounting": "explicit concurrent fences; an error-flag fence can also satisfy data dependencies",
        "performance_qualified": False,
    }


def confirmation_size(screening):
    need(
        len(screening["histories"]) == 24
        and all(
            h["sampling"] in SAMPLING and h.get("kind", "4k") == "4k"
            for h in screening["histories"]
        ),
        "coding histories or extra modes cannot size throughput confirmation",
    )
    need(
        screening["metric"] == "useful_tokens_per_server_decode_second",
        "primary measure must exclude prefill",
    )
    latency = screening["registered_latency_measures"]
    need(
        latency and len(latency) == len(set(latency)),
        "register latency measures before screening",
    )
    normal = statistics.NormalDist()
    power_z = normal.inv_cdf(0.8)
    z = normal.inv_cdf(0.975)
    latency_family_size = len(SAMPLING) * len(latency)
    simultaneous = normal.inv_cdf(1 - 0.05 / latency_family_size)
    result = {}
    for sampling in SAMPLING:
        histories = [h for h in screening["histories"] if h["sampling"] == sampling]
        need(
            all(h.get("kind", "4k") == "4k" for h in histories),
            "coding histories cannot size throughput confirmation",
        )
        need(
            len(histories) == 8 and len({h["source_cluster"] for h in histories}) == 8,
            "screening requires eight independent source clusters per mode",
        )
        need(
            len({h["history_id"] for h in histories}) == 8,
            "seeds and repeated sessions are not histories",
        )
        required = {}
        scale = 2 * statistics.stdev(h["throughput_log_ratio"] for h in histories)
        required["point_estimate"] = (
            power_z * scale / (math.log(1.08) - math.log(1.05))
        ) ** 2
        required["positive_interval"] = ((power_z + z) * scale / math.log(1.08)) ** 2
        for measure in latency:
            sigma = 2 * statistics.stdev(
                h["latency_log_ratios"][measure] for h in histories
            )
            required[measure] = ((power_z + simultaneous) * sigma / math.log(1.03)) ** 2
        need(
            all(math.isfinite(n) and n >= 0 for n in required.values()),
            "invalid screening variation",
        )
        n = max(24, math.ceil(max(required.values())))
        result[sampling] = {
            "histories": n,
            "requirements": required,
            "bounded_campaign": n <= 64,
        }
    return {
        "schema": 1,
        "primary": screening["metric"],
        "confirmation": result,
        "locked_before_confirmation": True,
        "screening_history_ids": [h["history_id"] for h in screening["histories"]],
        "screening_source_clusters": sorted(
            {h["source_cluster"] for h in screening["histories"]}
        ),
        "registered_latency_measures": latency,
        "latency_family_size": latency_family_size,
    }


def context_anchors(histogram):
    need(1 <= len(histogram) <= 65536, "context histogram bound")
    counts = {}
    for row in histogram:
        context, count = row["context"], row["count"]
        need(
            type(context) is int
            and 0 < context <= 65536
            and type(count) is int
            and count > 0,
            "invalid context histogram",
        )
        counts[context] = counts.get(context, 0) + count
    points = sorted(counts)
    total = sum(counts.values())
    result = {points[0], points[-1]}
    for fraction in (0.25, 0.5, 0.75):
        accumulated = 0
        for point in points:
            accumulated += counts[point]
            if accumulated >= fraction * total:
                result.add(point)
                break
    if points[0] <= 4352 and points[-1] >= 4096:
        result.update((4096, 4352))
    return sorted(result)


def coding_calibration(data, receipt, receipt_sha256):
    """Keep the original coding-policy entry point and refusal contract."""
    return cost_calibration(data, receipt, receipt_sha256, "coding")


def cost_calibration(data, receipt, receipt_sha256, sampling):
    """Assemble one declared sampling surface from complete guarded actions."""
    policies = {
        "greedy": {"temperature": 0, "top_k": 0, "top_p": 1, "min_p": 0},
        "full": {"temperature": 1, "top_k": 0, "top_p": 1, "min_p": 0},
        "topk": {"temperature": 1, "top_k": 20, "top_p": 1, "min_p": 0},
        "coding": {"temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0},
    }
    need(sampling in policies, "unknown calibration sampling mode")
    need(
        data.get("schema") == 1
        and data.get("policy") == policies[sampling]
        and data.get("clock") == "complete-native-decode"
        and data.get("initial_prompt_priming_excluded") is True,
        f"{sampling} calibration requires its exact policy and complete decode clock",
    )
    drafter, maximum = data["drafter"], data["maximum_rows"]
    need(
        drafter in ("mtp", "dflash2")
        and type(maximum) is int
        and 2 <= maximum <= (4 if drafter == "mtp" else 5),
        f"{sampling} calibration drafter or row bound",
    )
    contexts = context_anchors(data["context_histogram"])
    need(len(contexts) <= 8, f"{sampling} calibration context bound")
    need(
        receipt.get("schema") == 3
        and receipt.get("runtime_base") == BASE
        and receipt.get("execution_digest") == data["execution_digest"]
        and len(data["execution_digest"]) == 64
        and all(c in "0123456789abcdef" for c in data["execution_digest"])
        and len(receipt_sha256) == 64
        and all(c in "0123456789abcdef" for c in receipt_sha256),
        f"{sampling} calibration exactness identity differs",
    )
    cases = receipt["comparisons"]
    need(1 <= len(cases) <= 2048, f"{sampling} calibration exactness case bound")
    for context in contexts:
        for rows in range(2, maximum + 1):
            for prefix in range(1, rows + 1):
                matched = [
                    c
                    for c in cases
                    if (c["context"], c["rows"], c["accepted_prefix"], c["sampling"])
                    == (context, rows, prefix, sampling)
                ]
                need(
                    len(matched) == 1,
                    f"{sampling} calibration lacks a distinct {sampling} exactness case",
                )
                for name in (
                    "initial_state",
                    "complete_logits",
                    "accepted_state",
                    "sampling_distribution",
                    "routes",
                ):
                    value = matched[0]["ordinary"][name]
                    need(
                        len(value) == 64
                        and all(c in "0123456789abcdef" for c in value)
                        and value == matched[0]["verification"][name],
                        f"{sampling} calibration exactness mismatch",
                    )
    observations = data["observations"]
    need(1 <= len(observations) <= 4096, f"{sampling} calibration observation bound")
    labels = data.get("acceptance_observations", observations)
    need(
        1 <= len(labels) <= 4096, f"{sampling} calibration acceptance observation bound"
    )
    label_groups = {context: [] for context in contexts}
    for item in labels:
        context, rows, accepted = item["context"], item["rows"], item["accepted"]
        need(
            type(context) is int
            and context in label_groups
            and type(rows) is int
            and 1 <= rows <= maximum
            and type(accepted) is int
            and 0 <= accepted < rows
            and item.get("sampling") == sampling
            and item.get("guards_passed") is True
            and item.get("complete_action") is True
            and type(item.get("stopped")) is bool,
            f"{sampling} calibration requires guarded reached-position labels",
        )
        label_groups[context].append(item)
    groups = {c: {} for c in contexts}
    for item in observations:
        context, rows, accepted = item["context"], item["rows"], item["accepted"]
        need(
            type(context) is int
            and context in groups
            and type(rows) is int
            and 1 <= rows <= maximum
            and type(accepted) is int
            and 0 <= accepted < rows
            and item.get("sampling") == sampling
            and item.get("drafted_rows")
            == (1 if rows == 1 else maximum if drafter == "dflash2" else rows)
            and item.get("guards_passed") is True
            and item.get("complete_action") is True
            and item.get("stopped") is False,
            f"{sampling} calibration requires guarded complete nonterminal actions and matching drafted rows at its anchors",
        )
        draft, total = item["draft_ms"], item["decode_ms"]
        need(
            all(type(v) in (float, int) and math.isfinite(v) for v in (draft, total))
            and 0 <= draft < total
            and total > 0
            and (draft == 0 if rows == 1 else draft > 0),
            f"invalid {sampling} decode or drafting cost",
        )
        groups[context].setdefault((rows, accepted), []).append(item)
    anchors, coverage = [], []
    for context, groups_at_context in groups.items():
        need(
            all(
                (rows, accepted) in groups_at_context
                for rows in range(1, maximum + 1)
                for accepted in range(rows)
            ),
            f"{sampling} calibration lacks a width or accepted-prefix cost; no table is issued",
        )
        ordinary = statistics.fmean(o["decode_ms"] for o in groups_at_context[1, 0])
        draft_costs, remaining = [0], [[ordinary]]
        reached, succeeded = [0] * (maximum - 1), [0] * (maximum - 1)
        full_block_draft = statistics.fmean(
            o["draft_ms"]
            for (rows, _), values in groups_at_context.items()
            if rows > 1
            for o in values
        )
        counts = []
        for rows in range(2, maximum + 1):
            values = [
                o for accepted in range(rows) for o in groups_at_context[rows, accepted]
            ]
            draft_costs.append(
                full_block_draft
                if drafter == "dflash2"
                else statistics.fmean(o["draft_ms"] for o in values)
            )
            remaining.append(
                [
                    statistics.fmean(
                        o["decode_ms"] - o["draft_ms"]
                        for o in groups_at_context[rows, accepted]
                    )
                    for accepted in range(rows)
                ]
            )
            counts.append(
                [len(groups_at_context[rows, accepted]) for accepted in range(rows)]
            )
        for observation in label_groups[context]:
            for position in range(observation["rows"] - 1):
                if observation["accepted"] < position or (
                    observation["stopped"] and observation["accepted"] == position
                ):
                    break
                reached[position] += 1
                succeeded[position] += observation["accepted"] > position
        need(
            all(reached), f"{sampling} calibration lacks reached conditional positions"
        )
        anchors.append(
            {
                "context": context,
                "ordinary_ms": ordinary,
                "draft_ms": draft_costs,
                "remaining_ms": remaining,
                "conditional_acceptance": [s / n for s, n in zip(succeeded, reached)],
            }
        )
        coverage.append(
            {
                "context": context,
                "ordinary_actions": len(groups_at_context[1, 0]),
                "prefix_counts": counts,
                "conditional_reached": reached,
                "conditional_accepted": succeeded,
            }
        )
    profile = {
        "schema": 1,
        "drafter": drafter,
        "maximum_rows": maximum,
        "execution_digest": data["execution_digest"],
        "exactness_receipt": receipt_sha256,
        "modes": [{"sampling": sampling, "anchors": anchors}],
    }
    need(
        type(data.get("prepared_validation", False)) is bool,
        f"invalid prepared {sampling} calibration selection",
    )
    if data.get("prepared_validation", False):
        profile["prepared_validation"] = True
    need(
        type(data.get("shared_pack_mapping", False)) is bool,
        f"invalid {sampling} calibration pack mapping",
    )
    if data.get("shared_pack_mapping", False):
        profile["shared_pack_mapping"] = True
    return profile, {
        "schema": 1,
        "purpose": "separate " + sampling + "-controller calibration",
        "coverage": coverage,
        "performance_qualified": False,
        "coding_regression_qualified": False,
        "source_observations": len(observations),
        "acceptance_observations": len(labels),
        "separate_acceptance_panel": "acceptance_observations" in data,
        "pack_mapping": "shared"
        if data.get("shared_pack_mapping", False)
        else "private",
        "throughput_modes": list(SAMPLING),
    }


def exactness_receipt(execution_digest, cases):
    need(
        len(execution_digest) == 64
        and all(c in "0123456789abcdef" for c in execution_digest),
        "invalid execution digest",
    )
    need(1 <= len(cases) <= 2048, "exactness case bound")
    comparisons = []
    seen = set()
    for case in cases:
        key = tuple(case[k] for k in ("context", "rows", "accepted_prefix", "sampling"))
        need(
            key not in seen
            and 0 < key[0] <= 65536
            and 2 <= key[1] <= 8
            and 1 <= key[2] <= key[1]
            and key[3] in CONTROLLER_SAMPLING,
            "invalid or duplicate exactness case",
        )
        seen.add(key)
        item = {k: case[k] for k in ("context", "rows", "accepted_prefix", "sampling")}
        item["capture_ids"], item["artifact_sizes"] = {}, {}
        item["source_identity"] = None
        inodes = set()
        for arm in ("ordinary", "verification"):
            witness = json.loads(Path(case[arm]["capture_identity"]).read_bytes())
            common = {
                key: witness.get(key)
                for key in (
                    "worker_sha256",
                    "library_sha256",
                    "capture_sha256",
                    "source_revision",
                )
            }
            need(
                witness.get("runtime_base") == BASE
                and witness.get("arm") == arm
                and witness.get("source_clean") is True
                and witness.get("model_worker") is True
                and all(
                    isinstance(v, str)
                    and len(v) == (40 if k == "source_revision" else 64)
                    and all(c in "0123456789abcdef" for c in v)
                    for k, v in common.items()
                ),
                "exactness requires pinned independent model captures",
            )
            capture_id = witness.get("capture_id")
            need(
                isinstance(capture_id, str)
                and len(capture_id) == 32
                and all(c in "0123456789abcdef" for c in capture_id)
                and capture_id not in item["capture_ids"].values(),
                "exactness captures are not independent",
            )
            item["capture_ids"][arm] = capture_id
            if item["source_identity"] is None:
                item["source_identity"] = common
            need(item["source_identity"] == common, "exactness capture builds differ")
            item[arm] = {}
            item["artifact_sizes"][arm] = {}
            for name in (
                "initial_state",
                "complete_logits",
                "accepted_state",
                "sampling_distribution",
                "routes",
            ):
                path = Path(case[arm][name])
                need(
                    path.is_file() and path.stat().st_size > 0,
                    "missing or empty exactness evidence",
                )
                info = path.stat()
                need(
                    (info.st_dev, info.st_ino) not in inodes,
                    "exactness evidence reuses one file",
                )
                inodes.add((info.st_dev, info.st_ino))
                need(info.st_size <= MAX_TRACE_BYTES, "exactness artefact byte bound")
                raw = path.read_bytes()
                if name in ("initial_state", "accepted_state"):
                    need(
                        len(raw) >= 256 and len(raw) % 4 == 0,
                        "invalid exactness state capture",
                    )
                elif name == "complete_logits":
                    need(
                        len(raw) == 248320 * 4
                        and all(
                            math.isfinite(v) for (v,) in struct.iter_unpack("<f", raw)
                        ),
                        "incomplete or nonfinite exactness logits",
                    )
                elif name == "sampling_distribution":
                    need(len(raw) >= 20, "invalid exactness distribution")
                    mode, count = struct.unpack_from("<2I", raw)
                    need(
                        mode < len(CONTROLLER_SAMPLING)
                        and 1 <= count <= 248320
                        and len(raw) == 8 + count * 12,
                        "invalid exactness distribution",
                    )
                    entries = list(struct.iter_unpack("<Id", raw[8:]))
                    need(
                        len({token for token, _ in entries}) == count
                        and all(
                            token < 248320
                            and math.isfinite(probability)
                            and 0 <= probability <= 1
                            for token, probability in entries
                        )
                        and abs(sum(p for _, p in entries) - 1) <= 1e-8,
                        "invalid exactness distribution support",
                    )
                else:
                    need(
                        len(raw) >= key[1] * 4 and len(raw) % 4 == 0,
                        "incomplete exactness routes",
                    )
                item["artifact_sizes"][arm][name] = len(raw)
                item[arm][name] = hashlib.sha256(raw).hexdigest()
        need(
            item["ordinary"] == item["verification"],
            "exact controller admission failed: byte or route mismatch",
        )
        comparisons.append(item)
    # Only hashes and portable case identifiers leave the private evidence directory.
    return {
        "schema": 3,
        "runtime_base": BASE,
        "execution_digest": execution_digest,
        "comparisons": comparisons,
    }


def capture_rows(source, directory):
    need(
        Path(source).stat().st_size <= MAX_TRACE_BYTES,
        "correctness capture byte budget",
    )
    data = memoryview(Path(source).read_bytes())
    root = Path(directory)
    root.mkdir(mode=0o700)
    output, seen = [], set()
    formats, widths = {}, {}
    offset = 0

    def take(count):
        nonlocal offset
        need(0 <= count <= len(data) - offset, "truncated correctness capture")
        value = data[offset : offset + count]
        offset += count
        return value

    while offset < len(data):
        magic, action, rows, row, vocabulary, topk = struct.unpack("<6I", take(24))
        need(
            magic in (0x32474F4C, 0x33474F4C)
            and 1 <= action <= 4
            and 1 <= rows <= 8
            and row < rows
            and 0 < vocabulary <= 1 << 20
            and topk <= 1,
            "invalid correctness row header",
        )
        capture_sampling = SAMPLING
        if magic == 0x33474F4C:
            need(struct.unpack("<I", take(4))[0] == 4, "invalid captured policy count")
            capture_sampling = CONTROLLER_SAMPLING
        need(
            (action, row) not in seen and len(seen) < 32,
            "duplicate or excessive correctness rows",
        )
        seen.add((action, row))
        shape = (magic, rows, vocabulary, topk)
        need(
            action not in formats or formats[action] == shape,
            "inconsistent correctness action rows",
        )
        formats[action] = shape
        widths[action] = rows
        stem = f"action-{action}-row-{row}"
        logits_path = root / f"{stem}-logits.bin"
        with private_file(logits_path) as sink:
            sink.write(take(vocabulary * 4))
        if topk:
            take(528)
        record = {
            "action": action,
            "rows": rows,
            "row": row,
            "complete_logits": str(logits_path.resolve()),
            "distributions": {},
        }
        for expected_mode, name in enumerate(capture_sampling):
            begin = offset
            mode, count = struct.unpack("<2I", take(8))
            need(
                mode == expected_mode and 1 <= count <= vocabulary,
                "invalid captured distribution",
            )
            take(count * 12)
            path = root / f"{stem}-{name}.bin"
            with private_file(path) as sink:
                sink.write(data[begin:offset])
            record["distributions"][name] = str(path.resolve())
        output.append(record)
    need(output, "empty correctness capture")
    need(
        all(
            {row for found, row in seen if found == action} == set(range(rows))
            for action, rows in widths.items()
        ),
        "incomplete correctness action rows",
    )
    with private_file(root / "rows.json") as sink:
        sink.write(json.dumps(output, sort_keys=True, indent=2).encode() + b"\n")
    return output


def confirmation(plan, data):
    need(
        plan.get("locked_before_confirmation") and data.get("plan_sha256"),
        "confirmation needs an immutable sizing plan",
    )
    need(
        data["metric"] == plan["primary"] == "useful_tokens_per_server_decode_second",
        "confirmation clock differs from screening",
    )
    latency = plan["registered_latency_measures"]
    need(
        plan.get("latency_family_size") == len(SAMPLING) * len(latency),
        "confirmation plan omits simultaneous latency tests across sampling modes",
    )
    screening_ids = set(plan["screening_history_ids"])
    screening_clusters = set(plan["screening_source_clusters"])
    output = {}
    random_source = random.Random(44)

    def quantile(values, q):
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]

    for mode in SAMPLING:
        registered = plan["confirmation"][mode]
        n = registered["histories"]
        need(
            registered["bounded_campaign"] and 24 <= n <= 64,
            "qualification is beyond the bounded campaign",
        )
        histories = [h for h in data["histories"] if h["sampling"] == mode]
        need(
            len(histories) == n
            and len({h["history_id"] for h in histories}) == n
            and len({h["source_cluster"] for h in histories}) == n,
            "confirmation independence or fixed size differs",
        )
        need(
            not screening_ids.intersection(h["history_id"] for h in histories)
            and not screening_clusters.intersection(
                h["source_cluster"] for h in histories
            ),
            "confirmation overlaps screening",
        )
        ratios, latency_ratios, sessions = [], {key: [] for key in latency}, {}
        for history in histories:
            need(
                history.get("kind", "4k") == "4k"
                and history["prompt_tokens"] == 4096
                and history["output_budget"] == 256,
                "confirmation requires the sealed 4K/256 workload",
            )
            seeds = (44,) if mode == "greedy" else (44, 45)
            pairs = history["pairs"]
            identities = {(p["session"], p["seed"]) for p in pairs}
            session_ids = {p["session"] for p in pairs}
            need(
                len(session_ids) >= 1
                and len(identities) == len(pairs)
                and identities == {(s, seed) for s in session_ids for seed in seeds},
                "seeds or independent session pairs are incomplete",
            )
            for session in session_ids:
                sessions[session] = sessions.get(session, 0) + 1
            throughput, times = [], {key: [] for key in latency}
            for pair in pairs:
                need(
                    pair["guards_passed"] is True,
                    "failed receipt cannot qualify a campaign",
                )
                arms = [pair[name] for name in ("reference", "candidate")]
                need(
                    all(
                        a["useful_committed_tokens"] > 0
                        and math.isfinite(a["server_decode_ms"])
                        and a["server_decode_ms"] > 0
                        for a in arms
                    ),
                    "invalid serving decode clock or useful token count",
                )
                rates = [
                    a["useful_committed_tokens"] / a["server_decode_ms"] for a in arms
                ]
                throughput.append(math.log(rates[1] / rates[0]))
                for measure in latency:
                    a, b = [arm["latency_ms"][measure] for arm in arms]
                    need(
                        math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0,
                        "invalid registered latency",
                    )
                    times[measure].append(math.log(b / a))
            ratios.append(statistics.mean(throughput))
            for measure in latency:
                latency_ratios[measure].append(statistics.mean(times[measure]))
        need(
            len(sessions) >= 2 and max(sessions.values()) - min(sessions.values()) <= 1,
            "histories are unbalanced across nominal sessions",
        )
        boot = []
        boot_latency = {key: [] for key in latency}
        for _ in range(10000):
            indices = [random_source.randrange(n) for _ in range(n)]
            boot.append(statistics.mean(ratios[i] for i in indices))
            for key in latency:
                boot_latency[key].append(
                    statistics.mean(latency_ratios[key][i] for i in indices)
                )
        point = math.expm1(statistics.mean(ratios))
        interval = [math.expm1(quantile(boot, q)) for q in (0.025, 0.975)]
        upper = {
            key: math.expm1(
                quantile(boot_latency[key], 1 - 0.05 / (len(SAMPLING) * len(latency)))
            )
            for key in latency
        }
        output[mode] = {
            "histories": n,
            "throughput_change": point,
            "paired_95_interval": interval,
            "latency_simultaneous_upper": upper,
            "passes_serving_thresholds": point >= 0.05
            and interval[0] > 0
            and all(v <= 0.03 for v in upper.values()),
        }
    return {
        "schema": 1,
        "sizing_plan_sha256": data["plan_sha256"],
        "modes": output,
        "coding_or_numerical_qualification": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("archive")
    a.add_argument("source")
    a.add_argument("output")
    c = sub.add_parser("capture")
    c.add_argument("--arm", choices=("ordinary", "verification"), default="unlabelled")
    c.add_argument("trace")
    c.add_argument("argv", nargs=argparse.REMAINDER)
    r = sub.add_parser("replay")
    r.add_argument("--command-timeout", type=float, default=3600)
    r.add_argument("--component", action="store_true")
    r.add_argument("trace")
    r.add_argument("stderr")
    r.add_argument("argv", nargs=argparse.REMAINDER)
    candidate = sub.add_parser("check-prepared")
    candidate.add_argument("--command-timeout", type=float, default=3600)
    candidate.add_argument("trace")
    candidate.add_argument("stderr")
    candidate.add_argument("argv", nargs=argparse.REMAINDER)
    p = sub.add_parser("power")
    p.add_argument("screening")
    p.add_argument("output")
    t = sub.add_parser("timing-tape")
    t.add_argument("trace")
    t.add_argument("output")
    m = sub.add_parser("manifest")
    m.add_argument("trace")
    m.add_argument("cases")
    m.add_argument("output")
    ledger_parser = sub.add_parser("ledger")
    ledger_parser.add_argument("serial")
    ledger_parser.add_argument("concurrent")
    ledger_parser.add_argument("output")
    anchors_parser = sub.add_parser("anchors")
    anchors_parser.add_argument("histogram")
    anchors_parser.add_argument("output")
    proof_parser = sub.add_parser("exactness")
    proof_parser.add_argument("execution_digest")
    proof_parser.add_argument("cases")
    proof_parser.add_argument("output")
    rows_parser = sub.add_parser("capture-rows")
    rows_parser.add_argument("capture")
    rows_parser.add_argument("directory")
    confirm_parser = sub.add_parser("confirmation")
    confirm_parser.add_argument("plan")
    confirm_parser.add_argument("observations")
    confirm_parser.add_argument("output")
    coding_parser = sub.add_parser("calibrate-coding")
    coding_parser.add_argument("observations")
    coding_parser.add_argument("exactness")
    coding_parser.add_argument("output")
    args = parser.parse_args()
    if args.command == "archive":
        archive(args.source, args.output)
    elif args.command == "capture":
        capture(
            args.trace,
            args.argv[1:] if args.argv[:1] == ["--"] else args.argv,
            arm=args.arm,
        )
    elif args.command in ("replay", "check-prepared"):
        print(
            json.dumps(
                replay(
                    args.trace,
                    args.argv[1:] if args.argv[:1] == ["--"] else args.argv,
                    args.stderr,
                    require_controller=not getattr(args, "component", False),
                    prepared_candidate=args.command == "check-prepared",
                    command_timeout=args.command_timeout,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "power":
        result = confirmation_size(json.loads(Path(args.screening).read_bytes()))
        with private_file(args.output) as output:
            output.write(json.dumps(result, sort_keys=True, indent=2).encode() + b"\n")
    elif args.command == "capture-rows":
        capture_rows(args.capture, args.directory)
    elif args.command == "calibrate-coding":
        need(
            all(
                Path(path).stat().st_size <= MAX_TRACE_BYTES
                for path in (args.observations, args.exactness)
            ),
            "coding calibration byte budget",
        )
        receipt_bytes = Path(args.exactness).read_bytes()
        profile, audit = coding_calibration(
            json.loads(Path(args.observations).read_bytes()),
            json.loads(receipt_bytes),
            hashlib.sha256(receipt_bytes).hexdigest(),
        )
        audit["observations_sha256"] = digest(args.observations)
        audit["exactness_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
        for path, value in (
            (args.output, profile),
            (args.output + ".calibration.json", audit),
        ):
            with private_file(path) as output:
                output.write(
                    json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
                )
        with private_file(args.output + ".exactness.json") as output:
            output.write(receipt_bytes)
    else:
        if args.command == "timing-tape":
            result = timing_tape(args.trace)
        elif args.command == "manifest":
            result = replay_manifest(
                args.trace, json.loads(Path(args.cases).read_bytes())
            )
        elif args.command == "anchors":
            result = {
                "schema": 1,
                "contexts": context_anchors(
                    json.loads(Path(args.histogram).read_bytes())
                ),
            }
        elif args.command == "exactness":
            result = exactness_receipt(
                args.execution_digest, json.loads(Path(args.cases).read_bytes())
            )
        elif args.command == "confirmation":
            data = json.loads(Path(args.observations).read_bytes())
            need(
                data.get("plan_sha256") == digest(args.plan),
                "confirmation sizing plan hash differs",
            )
            result = confirmation(json.loads(Path(args.plan).read_bytes()), data)
        else:
            result = ledger(args.serial, args.concurrent)
        with private_file(args.output) as output:
            output.write(json.dumps(result, sort_keys=True, indent=2).encode() + b"\n")


if __name__ == "__main__":
    main()
