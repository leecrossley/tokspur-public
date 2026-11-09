"""Model-free contracts for private capture, replay bounds and power planning."""

import base64
import importlib.util
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "runtime_checks", ROOT / "scripts/runtime_checks.py"
)
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def test_profiling_cleanup_never_aborts_an_ended_generation():
    done = {"event": "done", "native_mtp_generation": {"id": 12, "active": False}}
    assert checks.profiling_abort(done, 12) is None
    done["native_mtp_generation"]["active"] = True
    assert checks.profiling_abort(done, 12)["mtp_generation"] == {
        "id": 12,
        "abort": True,
    }
    with pytest.raises(ValueError, match="generation differs"):
        checks.profiling_abort(done, 13)
    for bad in (
        {"event": "error"},
        {"event": "done"},
        {"event": "done", "native_mtp_generation": {"active": 0}},
    ):
        with pytest.raises(ValueError, match="profiling cleanup"):
            checks.profiling_abort(bad, 12)


def test_profiling_fresh_controller_comes_from_startup_not_prefill():
    ready = {
        "native_measured_controller": True,
        "native_controller_policy_digest": list(range(32)),
    }
    value = checks.profiling_controller(ready)
    assert value == {"controller_policy_digest": list(range(32))}
    value["controller_policy_digest"][0] = 255
    assert ready["native_controller_policy_digest"][0] == 0
    for bad in (
        {},
        {"native_measured_controller": True},
        {
            "native_measured_controller": True,
            "native_controller_policy_digest": [256] * 32,
        },
    ):
        with pytest.raises(ValueError, match="profiling"):
            checks.profiling_controller(bad)


def test_ledger_reports_overlapping_flag_causes_without_assigning_timing_gap(tmp_path):
    def trace(name, serial):
        path = tmp_path / name
        records = [
            {"event": "carrier_diagnostic_action", "action": 1, "complete": True},
            {
                "event": "carrier_diagnostic_batch",
                "action": 1,
                "batch": 1,
                "dispatches": 1,
                "serial_counter_schedule": serial,
                "gpu_ms": 8,
                "barriers": 3,
                "error_flag_barriers": 1,
                "kernels": [
                    {"kernel": "rms", "shape": [2560], "layer": 0, "gpu_ms": 10}
                ],
            },
        ]
        path.write_text(
            "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
        )
        return path

    serial, concurrent = trace("serial", True), trace("concurrent", False)
    result = checks.ledger(serial, concurrent)
    assert result["unexplained_difference_ms"] == -2
    assert result["concurrent_barriers"] == 3
    assert result["error_flag_barriers"] == 1
    assert result["data_only_barriers"] == 2
    assert not result["performance_qualified"]
    concurrent.write_text(
        concurrent.read_text().replace(
            '"error_flag_barriers":1', '"error_flag_barriers":4'
        )
    )
    with pytest.raises(ValueError, match="barrier accounting"):
        checks.ledger(serial, concurrent)
    concurrent = trace("batch-only", False)
    concurrent.write_text(
        concurrent.read_text().replace(
            '"dispatches":1', '"dispatches":2,"dispatch_records_enabled":false'
        )
    )
    with pytest.raises(ValueError, match="family attribution"):
        checks.ledger(serial, concurrent)


def test_exact_candidate_comparison_keeps_frontiers_rng_and_controller_observations():
    ready = {"event": "ready", "native_mtp_depth": 3, "execution_digest": "a" * 64}
    assert checks.comparison_packet(ready) is ready
    candidate = checks.comparison_packet(ready, prepared_candidate=True)
    assert candidate == {**ready, "native_prepared_validation": True}
    assert "native_prepared_validation" not in ready
    with pytest.raises(ValueError, match="unchanged default reference"):
        checks.comparison_packet(candidate, prepared_candidate=True)
    for key in ("entropy_words", "committed_prefix", "stage_a_controller"):
        reference = {"event": "done", key: 7}
        changed = {"event": "done", key: 8}
        assert checks.semantic(
            checks.comparison_packet(reference, True)
        ) != checks.semantic(checks.comparison_packet(changed, True))


def test_capture_preserves_supervised_pid_and_exact_bidirectional_bytes(tmp_path):
    trace = tmp_path / "exchange.jsonl"
    worker = 'import json,os,sys; print(json.dumps({"event":"ready","pid":os.getpid()}),flush=True)\nfor line in sys.stdin.buffer: sys.stdout.buffer.write(line); sys.stdout.buffer.flush()'
    raw = b'{"mtp_generation":{"id":1,"begin":true,"remaining":4,"committed":0,"emitted":0,"pending":0,"remaining_compute":4,"total_budget":4},"cost":0.33333333333333331}\n'
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/runtime_checks.py"),
            "capture",
            str(trace),
            "--",
            sys.executable,
            "-u",
            "-c",
            worker,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ready = process.stdout.readline()
    assert json.loads(ready)["pid"] == process.pid
    output, error = process.communicate(raw, timeout=10)
    assert process.returncode == 0, error
    assert output == raw
    records = checks.read_exchange(trace, require_controller=False)
    assert [(direction, value) for direction, _, value in records][0][1][
        "event"
    ] == "ready"
    assert [data for _, data, _ in records if data == raw] == [raw, raw]
    assert os.stat(trace).st_mode & 0o777 == 0o600
    lifecycle = Path(str(trace) + ".lifecycle.json")
    # The exec-transparent recorder closes immediately after the supervised worker.
    deadline = time.monotonic() + 5
    while not lifecycle.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    closure = json.loads(lifecycle.read_bytes())
    assert closure["worker_pid"] == process.pid
    assert closure["host_eof"] and closure["worker_eof"] and not closure["failures"]
    assert closure["exchange_sha256"] == checks.digest(trace)
    with pytest.raises(FileExistsError):
        checks.private_file(trace)


def test_capture_finishes_its_eof_receipt_during_owned_group_drain(tmp_path):
    trace = tmp_path / "drain.jsonl"
    worker = 'import json,os,sys; print(json.dumps({"event":"ready","pid":os.getpid()}),flush=True); sys.stdin.buffer.read()'
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/runtime_checks.py"),
            "capture",
            str(trace),
            "--",
            sys.executable,
            "-u",
            "-c",
            worker,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert json.loads(process.stdout.readline())["pid"] == process.pid
    os.killpg(process.pid, signal.SIGTERM)
    process.communicate(timeout=10)
    lifecycle = Path(str(trace) + ".lifecycle.json")
    deadline = time.monotonic() + 5
    while (
        not lifecycle.exists() or not lifecycle.stat().st_size
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    closure = json.loads(lifecycle.read_bytes())
    assert closure["host_eof"] and closure["worker_eof"] and not closure["failures"]
    assert closure["exchange_sha256"] == checks.digest(trace)


def test_controller_capture_refuses_legacy_packet(tmp_path):
    path = tmp_path / "old.jsonl"
    records = [
        ("worker", b'{"event":"ready"}\n'),
        ("host", b'{"mtp_generation":{"id":1,"begin":true,"remaining":4}}\n'),
    ]
    path.write_text(
        "".join(
            json.dumps(
                {"schema": 1, "direction": d, "bytes": base64.b64encode(raw).decode()}
            )
            + "\n"
            for d, raw in records
        )
    )
    with pytest.raises(ValueError, match="legacy"):
        checks.read_exchange(path)


def test_packet_parity_excludes_host_telemetry_but_retains_model_limits():
    a = {
        "memory": {"physical_footprint_bytes": 12, "gpu_budget_bytes": 128},
        "committed_prefix": 4,
    }
    b = {
        "memory": {"physical_footprint_bytes": 20, "gpu_budget_bytes": 128},
        "committed_prefix": 4,
    }
    assert checks.semantic(a) == checks.semantic(b)
    b["memory"]["gpu_budget_bytes"] = 256
    assert checks.semantic(a) != checks.semantic(b)
    b["memory"]["gpu_budget_bytes"] = 128
    b["committed_prefix"] = 5
    assert checks.semantic(a) != checks.semantic(b)


def test_replay_cli_reports_the_verified_result(monkeypatch, capsys):
    verified = {"reference_sha256": "a" * 64, "parity": True}
    monkeypatch.setattr(checks, "replay", lambda *args, **kwargs: verified)
    monkeypatch.setattr(
        sys, "argv", ["runtime_checks.py", "replay", "trace", "errors", "--", "/worker"]
    )
    checks.main()
    assert json.loads(capsys.readouterr().out) == verified


def test_target_component_manifest_requires_matching_physical_frontier(tmp_path):
    trace = tmp_path / "component.jsonl"
    records = [
        ("worker", {"event": "ready"}),
        ("host", {"tokens": [3, 4, 5], "carrier_component": {"accepted_prefix": 2}}),
        (
            "worker",
            {
                "event": "done",
                "carrier_component": True,
                "component_kind": "target-rows",
                "base": 4096,
                "rows": 3,
                "accepted_prefix": 2,
                "committed_prefix": 4098,
            },
        ),
        ("host", {"op": "drain"}),
        ("worker", {"event": "drained"}),
    ]
    trace.write_text(
        "".join(
            json.dumps(
                {
                    "schema": 1,
                    "direction": direction,
                    "bytes": base64.b64encode(
                        (json.dumps(value) + "\n").encode()
                    ).decode(),
                }
            )
            + "\n"
            for direction, value in records
        )
    )
    case = {"context": 4096, "width": 3, "accepted_prefix": 2, "sampling": "full"}
    assert checks.replay_manifest(trace, [case])["cases"] == [case]
    with pytest.raises(ValueError, match="differs from recorded execution"):
        checks.replay_manifest(trace, [{**case, "accepted_prefix": 1}])

    live = tmp_path / "live-round.jsonl"
    live.write_text(
        "".join(
            json.dumps(
                {
                    "schema": 1,
                    "direction": "worker",
                    "bytes": base64.b64encode(
                        (json.dumps(value) + "\n").encode()
                    ).decode(),
                }
            )
            + "\n"
            for value in [
                {"event": "ready"},
                {"event": "mtp_round", "base": 4096, "proposed": 2, "accepted": 1},
            ]
        )
    )
    assert checks.replay_manifest(live, [case])["cases"] == [case]
    with pytest.raises(ValueError, match="differs from recorded execution"):
        checks.replay_manifest(live, [{**case, "context": 4097}])


def test_dflash_timing_replay_proves_ordinary_tails_and_refuses_truncated_drafts(
    tmp_path,
):
    def trace(name, remaining):
        records = [
            ("worker", {"event": "ready", "native_mtp_depth": 7}),
            (
                "host",
                {
                    "mtp_generation": {
                        "id": 1,
                        "begin": True,
                        "remaining": remaining + 2,
                        "committed": 0,
                        "emitted": 0,
                        "pending": 0,
                        "remaining_compute": remaining + 2,
                        "total_budget": remaining + 2,
                    }
                },
            ),
            (
                "worker",
                {
                    "event": "mtp_round",
                    "base": 4096,
                    "count": 2,
                    "round_wall_ms": 0.33333333333333331,
                },
            ),
            (
                "worker",
                {
                    "event": "token",
                    "id": 12,
                    "token_prefix": 4097,
                },
            ),
            (
                "worker",
                {
                    "event": "token",
                    "id": 13,
                    "token_prefix": 4098,
                },
            ),
            (
                "worker",
                {
                    "event": "token",
                    "id": 14,
                    "committed_prefix": 4099,
                },
            ),
        ]
        path = tmp_path / name
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "schema": 1,
                        "direction": d,
                        "bytes": base64.b64encode(
                            (json.dumps(v) + "\n").encode()
                        ).decode(),
                    }
                )
                + "\n"
                for d, v in records
            )
        )
        return path

    tape = checks.timing_tape(trace("ordinary.jsonl", 1))
    assert struct.pack("<d", tape["observations_ms"][0]) == struct.pack("<d", 1 / 3)
    with pytest.raises(ValueError, match="truncated action"):
        checks.timing_tape(trace("truncated.jsonl", 3))


def test_measured_tape_keeps_middle_actions_from_normal_output_packets(tmp_path):
    def trace(name, first=1, times=None):
        times = times if times is not None else [1 / 3, 71.12345678901234]
        records = [
            (
                "worker",
                {
                    "event": "ready",
                    "native_measured_controller": True,
                    "native_measured_timing_schema": 1,
                },
            ),
            (
                "host",
                {
                    "mtp_generation": {
                        "id": 44,
                        "begin": True,
                        "remaining": 4,
                        "committed": 0,
                        "emitted": 0,
                        "pending": 0,
                        "remaining_compute": 4,
                        "total_budget": 4,
                    }
                },
            ),
            (
                "worker",
                {
                    "event": "done",
                    "native_measured_timing_schema": 1,
                    "native_measured_decode_observations": {
                        "first_action": first,
                        "milliseconds": times,
                    },
                    "native_mtp_generation": {
                        "stage_a_controller": {
                            "measured": {
                                "generation": 44,
                                "actions": 2,
                                "last_decode_ms": 71.12345678901234,
                            }
                        }
                    },
                },
            ),
        ]
        path = tmp_path / name
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "schema": 1,
                        "direction": d,
                        "bytes": base64.b64encode(
                            (json.dumps(v) + "\n").encode()
                        ).decode(),
                    }
                )
                + "\n"
                for d, v in records
            )
        )
        return path

    tape = checks.timing_tape(trace("complete"))
    assert [struct.pack("<d", value) for value in tape["observations_ms"]] == [
        struct.pack("<d", 1 / 3),
        struct.pack("<d", 71.12345678901234),
    ]
    for name, first, times in [
        ("skipped", 1, [71.12345678901234]),
        ("offset", 2, [1 / 3, 71.12345678901234]),
    ]:
        with pytest.raises(ValueError, match="skipped or duplicated"):
            checks.timing_tape(trace(name, first, times))


@pytest.mark.parametrize(
    "control",
    [
        "--native-prepared-validation=off",
        "--native-verify-cost-profile=/absent/profile.json",
        "--native-dflash2-draft-policy=always",
        "--native-verify-m8-splits=/absent/splits.json",
        "--worker-burst-pacing=on",
    ],
)
def test_python_mode_refuses_runtime_controls_before_launch(control):
    env = {**os.environ, "TOKSPUR_CLI_IMPL": "python", "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tokspur.cli import main; main()",
            "serve",
            control,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=ROOT,
    )
    assert result.returncode != 0 and "require the Rust CLI" in result.stderr


def screen(spread):
    return {
        "metric": "useful_tokens_per_server_decode_second",
        "registered_latency_measures": ["first_text", "request", "cached_turn"],
        "histories": [
            {
                "history_id": f"{mode}-{i}",
                "source_cluster": f"{mode}-{i}",
                "sampling": mode,
                "throughput_log_ratio": (i - 3.5) * spread,
                "latency_log_ratios": {
                    key: (i - 3.5) * spread
                    for key in ("first_text", "request", "cached_turn")
                },
            }
            for mode in checks.SAMPLING
            for i in range(8)
        ],
    }


def test_power_is_bounded_and_does_not_silently_weaken_thresholds():
    small = checks.confirmation_size(screen(0.001))
    assert all(mode["histories"] == 24 for mode in small["confirmation"].values())
    large = checks.confirmation_size(screen(0.1))
    assert all(
        not mode["bounded_campaign"] and mode["histories"] > 64
        for mode in large["confirmation"].values()
    )
    data = screen(0.01)
    data["histories"][1]["source_cluster"] = data["histories"][0]["source_cluster"]
    with pytest.raises(ValueError, match="independent source clusters"):
        checks.confirmation_size(data)
    data = screen(0.01)
    data["metric"] = "complete_request_tokens_per_second"
    with pytest.raises(ValueError, match="exclude prefill"):
        checks.confirmation_size(data)


def test_probe_and_diagnostic_outputs_are_separate_from_serving():
    makefile = (ROOT / "native/Makefile").read_text()
    assert "../build/carrier-diagnostic,../build/owned-native" in makefile
    assert "probes/runtime_checks.metal" in makefile
    owned_rule = makefile.split("$(BUILD)/tokspur-owned.metallib:", 1)[1].split(
        "\n\n", 1
    )[0]
    assert "runtime_checks" not in owned_rule and "probe" not in owned_rule
    layout = (ROOT / "rust/crates/tokspur-cli/src/layout.rs").read_text()
    assert "carrier-diagnostic" not in layout and "runtime-check-probe" not in layout


def test_context_anchors_use_weighted_decode_histogram():
    assert checks.context_anchors(
        [{"context": c, "count": w} for c, w in [(4000, 1), (4200, 20), (8000, 1)]]
    ) == [4000, 4096, 4200, 4352, 8000]
    with pytest.raises(ValueError, match="histogram"):
        checks.context_anchors([{"context": 4096, "count": 0}])


def test_exactness_receipt_compares_artifacts_and_keeps_paths_private(tmp_path):
    case = {"context": 4096, "rows": 2, "accepted_prefix": 1, "sampling": "full"}
    for arm, capture_id in (("ordinary", "1" * 32), ("verification", "2" * 32)):
        directory = tmp_path / arm
        directory.mkdir()
        payloads = {
            "initial_state": b"s" * 256,
            "complete_logits": b"\0" * (248320 * 4),
            "accepted_state": b"a" * 256,
            "sampling_distribution": struct.pack("<2IId", 1, 1, 0, 1.0),
            "routes": struct.pack("<2I", 0, 1),
        }
        case[arm] = {}
        for name, payload in payloads.items():
            path = directory / (name + ".bin")
            path.write_bytes(payload)
            case[arm][name] = str(path)
        witness = directory / "identity.json"
        witness.write_text(
            json.dumps(
                {
                    "runtime_base": checks.BASE,
                    "arm": arm,
                    "source_clean": True,
                    "model_worker": True,
                    "worker_sha256": "a" * 64,
                    "library_sha256": "b" * 64,
                    "capture_sha256": "c" * 64,
                    "source_revision": "d" * 40,
                    "capture_id": capture_id,
                }
            )
        )
        case[arm]["capture_identity"] = str(witness)
    receipt = checks.exactness_receipt("a" * 64, [case])
    assert receipt["schema"] == 3
    assert str(tmp_path) not in json.dumps(receipt)
    path = Path(case["verification"]["complete_logits"])
    raw = path.read_bytes()
    path.write_bytes(struct.pack("<f", 1.0) + raw[4:])
    with pytest.raises(ValueError, match="byte or route mismatch"):
        checks.exactness_receipt("a" * 64, [case])
    path.write_bytes(raw)
    case["verification"]["routes"] = case["verification"]["initial_state"]
    with pytest.raises(ValueError, match="reuses one file"):
        checks.exactness_receipt("a" * 64, [case])


def test_coding_capture_count_is_bounded(tmp_path):
    source = tmp_path / "capture.bin"
    source.write_bytes(struct.pack("<7I", 0x33474F4C, 1, 1, 0, 2, 0, 3))
    with pytest.raises(ValueError, match="policy count"):
        checks.capture_rows(source, tmp_path / "rows")
    assert checks.SAMPLING == ("greedy", "full", "topk")
    assert checks.CONTROLLER_SAMPLING == ("greedy", "full", "topk", "coding")


def coding_calibration_fixture(drafter="mtp"):
    maximum = 4 if drafter == "mtp" else 5
    arm = {
        name: "a" * 64
        for name in (
            "initial_state",
            "complete_logits",
            "accepted_state",
            "sampling_distribution",
            "routes",
        )
    }
    receipt = {
        "schema": 3,
        "runtime_base": checks.BASE,
        "execution_digest": "b" * 64,
        "comparisons": [
            {
                "context": context,
                "rows": rows,
                "accepted_prefix": prefix,
                "sampling": "coding",
                "ordinary": arm.copy(),
                "verification": arm.copy(),
            }
            for context in (4096, 4352)
            for rows in range(2, maximum + 1)
            for prefix in range(1, rows + 1)
        ],
    }
    observations = []
    for context in (4096, 4352):
        for rows in range(1, maximum + 1):
            for accepted in range(rows):
                draft = 0 if rows == 1 else rows - 1
                observations.append(
                    {
                        "context": context,
                        "rows": rows,
                        "accepted": accepted,
                        "drafted_rows": 1
                        if rows == 1
                        else maximum
                        if drafter == "dflash2"
                        else rows,
                        "draft_ms": draft,
                        "decode_ms": 10 + rows + accepted + draft,
                        "sampling": "coding",
                        "guards_passed": True,
                        "complete_action": True,
                        "stopped": False,
                    }
                )
    data = {
        "schema": 1,
        "drafter": drafter,
        "maximum_rows": maximum,
        "execution_digest": "b" * 64,
        "policy": {"temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0},
        "clock": "complete-native-decode",
        "initial_prompt_priming_excluded": True,
        "context_histogram": [
            {"context": context, "count": 1} for context in (4096, 4352)
        ],
        "observations": observations,
    }
    return data, receipt


@pytest.mark.parametrize("drafter", ["mtp", "dflash2"])
def test_coding_calibration_has_separate_complete_surface_and_live_conditional_labels(
    drafter,
):
    data, receipt = coding_calibration_fixture(drafter)
    profile, audit = checks.coding_calibration(data, receipt, "c" * 64)
    assert [mode["sampling"] for mode in profile["modes"]] == ["coding"]
    assert (
        not audit["performance_qualified"] and not audit["coding_regression_qualified"]
    )
    assert audit["throughput_modes"] == list(checks.SAMPLING)
    for anchor in profile["modes"][0]["anchors"]:
        assert anchor["ordinary_ms"] == 11
        assert [len(row) for row in anchor["remaining_ms"]] == list(
            range(1, data["maximum_rows"] + 1)
        )
        assert anchor["remaining_ms"][3] == [14, 15, 16, 17]
        if drafter == "mtp":
            assert anchor["conditional_acceptance"] == [6 / 9, 3 / 5, 1 / 2]
        else:
            assert len(set(anchor["draft_ms"][1:])) == 1


def test_coding_calibration_binds_mapping_and_preserves_private_default():
    data, receipt = coding_calibration_fixture()
    default, audit = checks.coding_calibration(data, receipt, "c" * 64)
    assert "shared_pack_mapping" not in default and audit["pack_mapping"] == "private"
    data["shared_pack_mapping"] = False
    assert checks.coding_calibration(data, receipt, "c" * 64)[0] == default
    data["shared_pack_mapping"] = True
    shared, audit = checks.coding_calibration(data, receipt, "c" * 64)
    assert (
        shared == {**default, "shared_pack_mapping": True}
        and audit["pack_mapping"] == "shared"
    )
    data["shared_pack_mapping"] = 1
    with pytest.raises(ValueError, match="pack mapping"):
        checks.coding_calibration(data, receipt, "c" * 64)


@pytest.mark.parametrize(
    "mode,policy",
    [
        ("greedy", {"temperature": 0, "top_k": 0, "top_p": 1, "min_p": 0}),
        ("full", {"temperature": 1, "top_k": 0, "top_p": 1, "min_p": 0}),
        ("topk", {"temperature": 1, "top_k": 20, "top_p": 1, "min_p": 0}),
    ],
)
def test_cost_calibration_keeps_policy_costs_and_admission_separate(mode, policy):
    data, receipt = coding_calibration_fixture()
    data["policy"] = policy
    for observation in data["observations"]:
        observation["sampling"] = mode
    for case in receipt["comparisons"]:
        case["sampling"] = mode
    profile, audit = checks.cost_calibration(data, receipt, "c" * 64, mode)
    assert profile["modes"][0]["sampling"] == mode
    assert not audit["performance_qualified"]
    with pytest.raises(ValueError):
        checks.cost_calibration(data, receipt, "c" * 64, "coding")
    with pytest.raises(ValueError):
        checks.cost_calibration(data, receipt, "c" * 64, "unknown")


@pytest.mark.parametrize(
    "failure",
    [
        "policy",
        "missing_cost",
        "wrong_proof",
        "state_mismatch",
        "guard",
        "partial",
        "stop",
        "prefill",
        "wrong_mode",
    ],
)
def test_coding_calibration_refuses_aliases_and_incomplete_evidence(failure):
    data, receipt = coding_calibration_fixture()
    if failure == "policy":
        data["policy"]["top_p"] = 1
    elif failure == "missing_cost":
        data["observations"].pop()
    elif failure == "wrong_proof":
        receipt["comparisons"][0]["sampling"] = "topk"
    elif failure == "state_mismatch":
        receipt["comparisons"][0]["verification"]["initial_state"] = "0" * 64
    elif failure == "prefill":
        data["initial_prompt_priming_excluded"] = False
    else:
        key, value = {
            "guard": ("guards_passed", False),
            "partial": ("complete_action", False),
            "stop": ("stopped", True),
            "wrong_mode": ("sampling", "topk"),
        }[failure]
        data["observations"][0][key] = value
    with pytest.raises(ValueError, match="coding calibration"):
        checks.coding_calibration(data, receipt, "c" * 64)


def test_coding_mode_cannot_be_added_to_throughput_screening():
    data = screen(0.001)
    data["histories"].append({"sampling": "coding", "kind": "coding"})
    with pytest.raises(ValueError, match="coding histories"):
        checks.confirmation_size(data)


def test_dflash_coding_calibration_refuses_budget_limited_draft_costs():
    data, receipt = coding_calibration_fixture("dflash2")
    data["observations"][1]["drafted_rows"] = 2
    with pytest.raises(ValueError, match="matching drafted rows"):
        checks.coding_calibration(data, receipt, "c" * 64)


def test_coding_calibration_fixed_acceptance_panel_is_separate_and_censors_stops():
    data, receipt = coding_calibration_fixture()
    labels = []
    for context in (4096, 4352):
        for accepted, stopped in ((1, False), (3, False), (1, True)):
            labels.append(
                {
                    "context": context,
                    "rows": 4,
                    "accepted": accepted,
                    "sampling": "coding",
                    "guards_passed": True,
                    "complete_action": True,
                    "stopped": stopped,
                }
            )
    data["acceptance_observations"] = labels
    profile, audit = checks.coding_calibration(data, receipt, "c" * 64)
    assert audit["separate_acceptance_panel"] and audit["acceptance_observations"] == 6
    assert profile["modes"][0]["anchors"][0]["conditional_acceptance"] == [1, 0.5, 1]
    assert audit["coverage"][0]["conditional_reached"] == [3, 2, 1]
    labels[0]["guards_passed"] = False
    with pytest.raises(ValueError, match="reached-position labels"):
        checks.coding_calibration(data, receipt, "c" * 64)


def test_coding_calibration_cli_writes_exact_companion_and_refuses_overwrite(tmp_path):
    data, receipt = coding_calibration_fixture()
    observations, proof, output = [
        tmp_path / name for name in ("costs.json", "proof.json", "coding.json")
    ]
    observations.write_text(json.dumps(data))
    proof.write_text(json.dumps(receipt))
    command = [
        sys.executable,
        str(ROOT / "scripts/runtime_checks.py"),
        "calibrate-coding",
        str(observations),
        str(proof),
        str(output),
    ]
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert Path(str(output) + ".exactness.json").read_bytes() == proof.read_bytes()
    profile = json.loads(output.read_bytes())
    assert profile["exactness_receipt"] == checks.digest(proof)
    audit = json.loads(Path(str(output) + ".calibration.json").read_bytes())
    assert audit["observations_sha256"] == checks.digest(observations)
    assert (
        not audit["performance_qualified"] and not audit["coding_regression_qualified"]
    )
    previous = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == previous


def test_confirmation_refuses_resizing_coding_and_overlapping_sources():
    plan = checks.confirmation_size(screen(0.001))
    data = {"metric": plan["primary"], "plan_sha256": "a" * 64, "histories": []}
    with pytest.raises(ValueError, match="fixed size"):
        checks.confirmation(plan, data)
    coding = screen(0.001)
    coding["histories"][0]["kind"] = "coding"
    with pytest.raises(ValueError, match="coding histories"):
        checks.confirmation_size(coding)


def test_confirmation_uses_history_resampling_and_disjoint_balanced_sessions():
    plan = checks.confirmation_size(screen(0.001))
    data = {"metric": plan["primary"], "plan_sha256": "a" * 64, "histories": []}
    for mode in checks.SAMPLING:
        for index in range(24):
            pairs = []
            for session in ("nominal-a", "nominal-b"):
                for seed in (44,) if mode == "greedy" else (44, 45):
                    arms = {
                        arm: {
                            "useful_committed_tokens": 256,
                            "server_decode_ms": duration,
                            "latency_ms": {
                                key: 1000 for key in plan["registered_latency_measures"]
                            },
                        }
                        for arm, duration in (
                            ("reference", 1000),
                            ("candidate", 1000 / 1.08),
                        )
                    }
                    pairs.append(
                        {
                            "session": session,
                            "seed": seed,
                            "guards_passed": True,
                            **arms,
                        }
                    )
            data["histories"].append(
                {
                    "history_id": f"confirmation-{mode}-{index}",
                    "source_cluster": f"confirmation-{mode}-{index}",
                    "sampling": mode,
                    "prompt_tokens": 4096,
                    "output_budget": 256,
                    "pairs": pairs,
                }
            )
    output = checks.confirmation(plan, data)
    assert all(
        result["passes_serving_thresholds"] and result["histories"] == 24
        for result in output["modes"].values()
    )
    # Histories, rather than mandatory repeated sessions, supply independence.
    # One complete seeded pair per history is valid when nominal sessions are
    # balanced across histories. More seeds do not manufacture more histories.
    for index, history in enumerate(data["histories"]):
        selected = "nominal-a" if index % 2 == 0 else "nominal-b"
        history["pairs"] = [p for p in history["pairs"] if p["session"] == selected]
    assert all(
        result["histories"] == 24
        for result in checks.confirmation(plan, data)["modes"].values()
    )
    original_pairs = data["histories"][0]["pairs"]
    data["histories"][0]["pairs"] = original_pairs + original_pairs
    with pytest.raises(ValueError, match="session pairs are incomplete"):
        checks.confirmation(plan, data)
    data["histories"][0]["pairs"] = original_pairs
    data["histories"][0]["source_cluster"] = plan["screening_source_clusters"][0]
    with pytest.raises(ValueError, match="overlaps screening"):
        checks.confirmation(plan, data)
