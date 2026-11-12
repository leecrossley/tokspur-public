"""Deterministic model-free serve lifecycle checks; no actual process is launched."""

from __future__ import annotations

import contextlib
import signal
import subprocess
import sys
import types
from pathlib import Path

import psutil

_SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, _SRC)
for _m in list(sys.modules):
    if _m == "tokspur" or _m.startswith("tokspur."):
        del sys.modules[_m]

import pytest  # noqa: E402

from tokspur import cli  # noqa: E402


class Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.root = tmp_path
        self.signals = {
            signal.SIGINT: object(),
            signal.SIGTERM: object(),
        }
        self.original = dict(self.signals)
        self.events = []
        self.launched = 0
        self.returncode = None
        self.ignore_term = False
        self.fail_popen = False
        self.fail_write = False
        self.fail_kill_wait = False
        self.stop_on_install = False
        self.stop_during_popen = False
        self.stop_during_wait = True
        self.repeat_during_cleanup = False
        self.clock = 0.0
        self.child = types.SimpleNamespace(pid=12345, poll=self.poll, wait=self.wait)
        dummy_server = tmp_path / "build/rust-fast-math/release/tokspur-server"
        dummy_server.parent.mkdir(parents=True, exist_ok=True)
        dummy_server.touch()
        dummy_server.chmod(0o755)
        dummy_pack = tmp_path / ".tokspur/derived/tokspur-pack-v2.asa.gguf"
        dummy_pack.parent.mkdir(parents=True, exist_ok=True)
        dummy_pack.touch()
        dummy_mtp = (
            tmp_path
            / ".tokspur/state/mtp-draft/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
        )
        dummy_mtp.parent.mkdir(parents=True, exist_ok=True)
        dummy_mtp.touch()
        monkeypatch.setattr(cli, "ROOT", tmp_path)
        monkeypatch.setattr(cli, "private_dir", lambda path: path)
        monkeypatch.setattr(cli, "state_lock", contextlib.nullcontext)
        monkeypatch.setattr(cli, "check_adoption", lambda data: None)
        monkeypatch.setattr(cli, "port_available", lambda port: True)
        monkeypatch.setattr(cli, "inference_processes", lambda: [])
        monkeypatch.setattr(cli, "isolated_environment", lambda: {})
        monkeypatch.setattr(cli, "read_json", self.read_json)
        monkeypatch.setattr(cli, "write_json", self.write_json)
        monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: None)
        monkeypatch.setattr(cli.subprocess, "Popen", self.popen)
        monkeypatch.setattr(cli.signal, "signal", self.install)
        monkeypatch.setattr(cli.os, "killpg", self.killpg)
        monkeypatch.setattr(cli.time, "monotonic", lambda: self.clock)
        monkeypatch.setattr(cli.time, "sleep", self.sleep)
        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: types.SimpleNamespace(total=128 * 1024**3)
        )
        monkeypatch.setattr(
            psutil,
            "Process",
            lambda pid: types.SimpleNamespace(children=lambda recursive: []),
        )

    def read_json(self, path):
        if path.name == "configured.json":
            return {"port": 18081}
        if path.name == "api-key.json":
            return {"api_key": "fixture-not-a-secret"}
        if path.name == "adoption.json":
            return {
                "schema": 1,
                "verification": "full-sha256",
                "target": {"directory": str(self.root), "assets": []},
                "draft": {
                    "directory": str(self.root / ".tokspur/state/mtp-draft"),
                    "assets": [
                        {
                            "path": str(
                                self.root
                                / ".tokspur/state/mtp-draft/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
                            ),
                            "sha256": "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6",
                            "size": 2786568256,
                        }
                    ],
                },
            }
        return {}

    def write_json(self, path, value):
        self.events.append(("receipt", path.name))
        if self.fail_write:
            raise OSError("fixture receipt write failed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")

    def install(self, number, handler):
        previous = self.signals[number]
        self.signals[number] = handler
        if handler is self.original[number]:
            self.events.append(("restore", number))
        elif self.stop_on_install and number == signal.SIGTERM:
            handler(number, None)
        return previous

    def stop(self):
        self.signals[signal.SIGINT](signal.SIGINT, None)

    def popen(self, *args, **kwargs):
        self.last_command = args[0] if args else kwargs.get("args")
        self.events.append(("popen",))
        self.launched += 1
        if self.fail_popen:
            raise OSError("fixture launch failed")
        if self.stop_during_popen:
            self.stop()
        return self.child

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        if timeout == 0.2 and self.stop_during_wait:
            self.stop()
            raise subprocess.TimeoutExpired("fixture-child", timeout)
        if timeout in (60, 15) and self.repeat_during_cleanup:
            # Our non-raising handler must still be installed during both waits.
            self.stop()
        if timeout == 60 and self.ignore_term:
            raise subprocess.TimeoutExpired("fixture-child", timeout)
        if timeout == 15 and self.fail_kill_wait:
            raise subprocess.TimeoutExpired("fixture-child", timeout)
        if self.returncode is None:
            self.returncode = 7
        return self.returncode

    def killpg(self, pid, number):
        assert pid == self.child.pid
        self.events.append(("signal", number))
        if number == signal.SIGKILL or not self.ignore_term:
            self.returncode = -number

    def sleep(self, seconds):
        self.clock += seconds

    def assert_restored(self):
        assert self.signals == self.original

    def assert_bounded_kill(self):
        assert ("wait", 60) in self.events
        assert ("signal", signal.SIGKILL) in self.events
        assert ("wait", 15) in self.events
        first_restore = next(
            index for index, event in enumerate(self.events) if event[0] == "restore"
        )
        assert first_restore > self.events.index(("wait", 15))


@pytest.fixture
def harness(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


def test_early_stop_before_launch_does_not_start_child(harness):
    harness.stop_on_install = True
    with pytest.raises(SystemExit) as exit:
        cli.serve()
    assert exit.value.code == 0
    assert harness.launched == 0
    harness.assert_restored()


def test_stop_during_popen_reaches_bounded_escalation(harness):
    harness.stop_during_popen = True
    harness.ignore_term = True
    with pytest.raises(SystemExit) as exit:
        cli.serve()
    assert exit.value.code == 0
    harness.assert_bounded_kill()
    harness.assert_restored()
    assert not (harness.root / ".tokspur/state/process.json").exists()


def test_stop_breaks_poll_loop_even_when_child_ignores_term(harness):
    harness.ignore_term = True
    with pytest.raises(SystemExit) as exit:
        cli.serve()
    assert exit.value.code == 0
    assert harness.events.count(("wait", 0.2)) == 1
    harness.assert_bounded_kill()
    harness.assert_restored()


def test_repeated_signals_cannot_skip_cleanup_waits(harness):
    harness.ignore_term = True
    harness.repeat_during_cleanup = True
    with pytest.raises(SystemExit) as exit:
        cli.serve()
    assert exit.value.code == 0
    harness.assert_bounded_kill()
    harness.assert_restored()


def test_receipt_write_failure_still_drains_child(harness):
    harness.ignore_term = True
    harness.fail_write = True
    with pytest.raises(OSError, match="receipt write"):
        cli.serve()
    harness.assert_bounded_kill()
    harness.assert_restored()


def test_popen_failure_restores_handlers_without_child_operations(harness):
    harness.fail_popen = True
    with pytest.raises(OSError, match="launch failed"):
        cli.serve()
    assert not any(event[0] in ("signal", "wait") for event in harness.events)
    harness.assert_restored()


def test_unverified_kill_exit_retains_receipt_and_restores_handlers(harness):
    harness.ignore_term = True
    harness.fail_kill_wait = True
    with pytest.raises(subprocess.TimeoutExpired):
        cli.serve()
    assert (harness.root / ".tokspur/state/process.json").exists()
    harness.assert_restored()


def test_natural_child_exit_code_is_preserved(harness):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit) as exit:
        cli.serve()
    assert exit.value.code == 7
    assert not any(event[0] == "signal" for event in harness.events)
    harness.assert_restored()


def test_serve_refuses_when_derived_pack_missing(harness):
    (harness.root / ".tokspur/derived/tokspur-pack-v2.asa.gguf").unlink()
    with pytest.raises(
        FileNotFoundError,
        match=r"Lossless derived weight pack not found.*tokspur-pack-v3\.asa\.gguf",
    ):
        cli.serve()
    harness.stop_during_wait = False
    with pytest.raises(SystemExit) as exit:
        cli.serve(allow_raw_gguf=True)
    assert exit.value.code == 7


def test_serve_prefers_asa_pack_and_falls_back_to_legacy_gguf(harness):
    harness.stop_during_wait = False
    asa_v2 = harness.root / ".tokspur/derived/tokspur-pack-v2.asa.gguf"
    legacy_v2 = harness.root / ".tokspur/derived/tokspur-pack-v2.gguf"
    asa_v3 = harness.root / ".tokspur/derived/tokspur-pack-v3.asa.gguf"
    legacy_v3 = harness.root / ".tokspur/derived/tokspur-pack-v3.gguf"

    # Prefer .asa.gguf when both .asa.gguf and legacy .gguf exist.
    legacy_v2.touch()
    with pytest.raises(SystemExit):
        cli.serve()
    cmd = harness.last_command
    assert cmd[cmd.index("--derived-pack") + 1] == str(asa_v2.resolve())

    # Fall back to legacy .gguf when .asa.gguf is absent.
    asa_v2.unlink()
    with pytest.raises(SystemExit):
        cli.serve()
    cmd = harness.last_command
    assert cmd[cmd.index("--derived-pack") + 1] == str(legacy_v2.resolve())

    # --pack-v3 names .asa.gguf when missing, falls back to legacy v3 .gguf, and prefers v3 .asa.gguf.
    with pytest.raises(FileNotFoundError, match=r"tokspur-pack-v3\.asa\.gguf"):
        cli.serve(pack_v3=True)

    legacy_v3.touch()
    with pytest.raises(SystemExit):
        cli.serve(pack_v3=True)
    cmd = harness.last_command
    assert cmd[cmd.index("--derived-pack") + 1] == str(legacy_v3.resolve())

    asa_v3.touch()
    with pytest.raises(SystemExit):
        cli.serve(pack_v3=True)
    cmd = harness.last_command
    assert cmd[cmd.index("--derived-pack") + 1] == str(asa_v3.resolve())


def test_serve_refuses_when_mtp_sidecar_missing_from_adoption(harness, monkeypatch):
    monkeypatch.setattr(
        cli,
        "read_json",
        lambda p: (
            {"port": 18081}
            if p.name == "configured.json"
            else {
                "schema": 1,
                "verification": "full-sha256",
                "target": {"directory": str(harness.root), "assets": []},
                "draft": {"directory": str(harness.root), "assets": []},
            }
            if p.name == "adoption.json"
            else {}
        ),
    )
    with pytest.raises(ValueError, match="MTP sidecar"):
        cli.serve()


def test_serve_passes_checkpoint_ssd_and_overlap_flags(harness):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit):
        cli.serve(
            checkpoint_ssd=True,
            checkpoint_ssd_budget_gib=12.0,
            checkpoint_ssd_floor_gib=25.0,
            overlap_prefill=True,
        )
    command = harness.last_command
    assert "--checkpoint-ssd" in command
    assert "12.0" in command
    assert "25.0" in command
    assert "--overlap-prefill" in command


def test_serve_passes_m6_checkpoint_and_overlap_flags(harness):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit):
        cli.serve(
            checkpoint_cap_gib=16.0,
            checkpoint_floor_gib=30.0,
            checkpoint_dir="/tmp/test_checkpoints",
            checkpoint_policy="lru",
            overlap_prefill=True,
        )
    command = harness.last_command
    assert "--checkpoint-ssd" in command
    assert "--checkpoint-cap-gib" in command
    assert "16.0" in command
    assert "--checkpoint-floor-gib" in command
    assert "30.0" in command
    assert "--checkpoint-dir" in command
    assert "/tmp/test_checkpoints" in command
    assert "--checkpoint-policy" in command
    assert "lru" in command
    assert "--overlap-prefill" in command


def test_serve_default_uses_f32_and_keeps_speed_lead_paths_opt_in(harness):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit):
        cli.serve()
    command = harness.last_command
    expected_pairs = (
        ("--native-prefill-dense", "f32-staged"),
        ("--native-prefill-grouped", "f32"),
        ("--native-gdn-ported", "off"),
        ("--native-prefill-router-window", "off"),
        ("--native-prefill-hc-down-window", "off"),
        ("--native-qsa-sparse", "off"),
        ("--native-mtp-output-packet", "8"),
        ("--native-known-token-capacity", "2048"),
        ("--native-concurrent-dispatch", "on"),
        ("--native-prefill-swizzle", "on"),
        ("--native-qsa-prefill-fused", "on"),
        ("--native-hc-consumer-fused", "on"),
        ("--native-ple-overlap", "on"),
        ("--shared-prefix-cache", "on"),
        ("--tool-enum-validation", "strict"),
    )
    for flag, expected_val in expected_pairs:
        idx = command.index(flag)
        assert command[idx + 1] == expected_val
    for opt_in_flag in (
        "--native-q8-gemv-rg4",
        "--native-q8-three-msg",
        "--native-prefill-multiseg",
        "--native-host-logit-views",
        "--native-dflash2-defer-context",
        "--derived-pack-trust-mode",
        "--native-mtp-draft-slice",
        "--native-verify-nax",
        "--native-drafter",
        "--native-mtp-lookup",
        "--native-mtp-lookup-match",
        "--private-prefix-trace",
        "--telemetry-stdout",
        "--overlap-prefill",
        "--checkpoint-ssd",
        "--checkpoint-cap-gib",
        "--checkpoint-floor-gib",
        "--checkpoint-policy",
    ):
        assert opt_in_flag not in command


def test_serve_opt_out_flags_disable_default_exact_optimisations(harness):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit):
        cli.serve(
            known_token_capacity=512,
            concurrent_dispatch=False,
            q8_gemv_rg4="off",
            q8_three_msg="off",
            prefill_multiseg=False,
            prefill_swizzle=False,
            qsa_prefill_fused=False,
            hc_consumer_fused=False,
            ple_overlap=False,
            shared_prefix_cache=False,
            checkpoint_cap_gib=0.0,
        )
    command = harness.last_command
    assert command[command.index("--native-known-token-capacity") + 1] == "512"
    for disabled_flag in (
        "--native-concurrent-dispatch",
        "--native-q8-gemv-rg4",
        "--native-q8-three-msg",
        "--native-prefill-multiseg",
        "--native-prefill-swizzle",
        "--native-qsa-prefill-fused",
        "--native-hc-consumer-fused",
        "--native-ple-overlap",
        "--shared-prefix-cache",
        "--checkpoint-ssd",
    ):
        assert disabled_flag not in command


def test_serve_opt_in_speed_lead_flags_are_forwarded(harness, tmp_path):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit):
        cli.serve(
            derived_pack_trust_mode="local-seal",
            output_packet=64,
            prefill_grouped="auto",
            known_token_capacity=2048,
            mtp_draft_slice=True,
            concurrent_dispatch=True,
            q8_gemv_rg4="wide",
            q8_three_msg="wide",
            mtp_lookup="hybrid",
            mtp_lookup_match=4,
            tool_enum_validation="executor",
            private_prefix_trace=True,
            telemetry_stdout=True,
        )
    command = harness.last_command
    assert command[command.index("--derived-pack-trust-mode") + 1] == "local-seal"
    assert command[command.index("--native-mtp-output-packet") + 1] == "64"
    assert command[command.index("--native-prefill-grouped") + 1] == "auto"
    assert command[command.index("--native-known-token-capacity") + 1] == "2048"
    assert command[command.index("--native-mtp-draft-slice") + 1] == "on"
    assert command[command.index("--native-concurrent-dispatch") + 1] == "on"
    assert command[command.index("--native-q8-gemv-rg4") + 1] == "wide"
    assert command[command.index("--native-q8-three-msg") + 1] == "wide"
    assert command[command.index("--native-mtp-lookup") + 1] == "hybrid"
    assert command[command.index("--native-mtp-lookup-match") + 1] == "4"
    assert command[command.index("--tool-enum-validation") + 1] == "executor"
    assert command[command.index("--private-prefix-trace") + 1] == "on"
    assert command[command.index("--telemetry-stdout") + 1] == "on"

    dflash2_dir = tmp_path / "dflash2-ckpt"
    dflash2_dir.mkdir(parents=True)
    with pytest.raises(SystemExit):
        cli.serve(
            prefill_dense="nax",
            prefill_grouped="nax",
            verify_nax=True,
            drafter="dflash2",
            dflash2_dir=dflash2_dir,
        )
    dflash2_cmd = harness.last_command
    assert dflash2_cmd[dflash2_cmd.index("--native-drafter") + 1] == "dflash2"
    assert dflash2_cmd[dflash2_cmd.index("--native-mtp-depth") + 1] == "7"
    assert (
        dflash2_cmd[dflash2_cmd.index("--native-mtp-draft-attention") + 1] == "split16"
    )
    assert dflash2_cmd[dflash2_cmd.index("--native-prefill-dense") + 1] == "nax"
    assert dflash2_cmd[dflash2_cmd.index("--native-prefill-grouped") + 1] == "nax"
    assert dflash2_cmd[dflash2_cmd.index("--native-verify-nax") + 1] == "on"
    assert "--checkpoint-ssd" not in dflash2_cmd
    assert "--shared-prefix-cache" not in dflash2_cmd

    with pytest.raises(SystemExit):
        cli.serve(
            drafter="dflash2",
            dflash2_dir=dflash2_dir,
            shared_prefix_cache=True,
            checkpoint_cap_gib=4.0,
        )
    cached = harness.last_command
    assert cached[cached.index("--shared-prefix-cache") + 1] == "on"
    assert cached[cached.index("--checkpoint-ssd") + 1] == "on"

    for bad_kwargs in (
        {"drafter": "dflash2"},
        {"dflash2_dir": dflash2_dir},
        {"drafter": "dflash2", "dflash2_dir": dflash2_dir, "mtp_draft_slice": True},
        {"drafter": "dflash2", "dflash2_dir": dflash2_dir, "mtp_lookup": "bypass"},
        {"prefill_grouped": "serial"},
        {"prefill_dense": "f32", "prefill_grouped": "nax"},
    ):
        with pytest.raises(ValueError):
            cli.serve(**bad_kwargs)


def test_main_defaults_to_rust_cli_and_preserves_python_selector(
    monkeypatch, tmp_path, capsys
):
    assert cli.DEFAULT_CLI_IMPL == "rust"
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("TOKSPUR_CLI_IMPL", raising=False)
    monkeypatch.setattr(sys, "argv", ["tokspur", "doctor"])

    with pytest.raises(SystemExit) as missing_exit:
        cli.main()
    assert missing_exit.value.code == 1
    assert "Rust CLI binary not found" in capsys.readouterr().err

    rust_bin = tmp_path / "build/rust-fast-math/release/tokspur"
    rust_bin.parent.mkdir(parents=True, exist_ok=True)
    rust_bin.write_text("#!/bin/sh\n")
    rust_bin.chmod(0o755)
    exec_calls: list[tuple[str, list[str]]] = []

    def fake_execv(path: str, argv: list[str]) -> None:
        exec_calls.append((path, argv))
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execv", fake_execv)
    with pytest.raises(SystemExit) as ok_exit:
        cli.main()
    assert ok_exit.value.code == 0
    assert exec_calls == [(str(rust_bin), [str(rust_bin), "doctor"])]

    monkeypatch.setenv("TOKSPUR_CLI_IMPL", "python")
    monkeypatch.setattr(sys, "argv", ["tokspur", "--version"])
    with pytest.raises(SystemExit) as ver_exit:
        cli.main()
    assert ver_exit.value.code == 0
    assert cli.__version__ in capsys.readouterr().out

    monkeypatch.setenv("TOKSPUR_CLI_IMPL", "invalid")
    with pytest.raises(SystemExit) as bad_exit:
        cli.main()
    assert bad_exit.value.code == 2
    assert "invalid TOKSPUR_CLI_IMPL" in capsys.readouterr().err


def test_launcher_script_defaults_to_rust_and_honours_selector(tmp_path):
    repo_launcher = Path(__file__).resolve().parents[1] / "scripts/tokspur-launcher.sh"
    (tmp_path / "profiles").mkdir(parents=True)
    (tmp_path / "profiles/flash-next-assets.json").write_text("{}\n")
    (tmp_path / "native").mkdir(parents=True)
    (tmp_path / "native/Makefile").write_text("# fixture\n")
    venv_bin = tmp_path / ".venv/bin"
    venv_bin.mkdir(parents=True)
    launcher = venv_bin / "tokspur"
    launcher.write_bytes(repo_launcher.read_bytes())
    launcher.chmod(0o755)
    fake_python = venv_bin / "python"
    fake_python.write_text('#!/bin/sh\nprintf "py:%s:%s\\n" "$TOKSPUR_CLI_IMPL" "$*"\n')
    fake_python.chmod(0o755)

    clean_env = {"PATH": "/usr/bin:/bin"}
    missing = subprocess.run(
        [str(launcher), "doctor"],
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 1
    assert "Rust CLI binary not found" in missing.stderr

    rust_bin = tmp_path / "build/rust-fast-math/release/tokspur"
    rust_bin.parent.mkdir(parents=True)
    rust_bin.write_text('#!/bin/sh\nprintf "rust:%s\\n" "$*"\n')
    rust_bin.chmod(0o755)

    default_run = subprocess.run(
        [str(launcher), "serve", "--help"],
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert default_run.returncode == 0
    assert default_run.stdout.strip() == "rust:serve --help"

    py_run = subprocess.run(
        [str(launcher), "doctor"],
        env={**clean_env, "TOKSPUR_CLI_IMPL": "python"},
        capture_output=True,
        text=True,
    )
    assert py_run.returncode == 0
    assert py_run.stdout.strip() == "py:python:-m tokspur.cli doctor"

    bad_run = subprocess.run(
        [str(launcher), "doctor"],
        env={**clean_env, "TOKSPUR_CLI_IMPL": "other"},
        capture_output=True,
        text=True,
    )
    assert bad_run.returncode == 2
    assert "invalid TOKSPUR_CLI_IMPL" in bad_run.stderr


@pytest.mark.parametrize("value", ["off", "on"])
def test_native_sync_selectors_are_forwarded_without_loading_model(harness, value):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit):
        cli.serve(native_host_logit_views=value, native_dflash2_defer_context=value)
    command = harness.last_command
    for flag in ("--native-host-logit-views", "--native-dflash2-defer-context"):
        assert command[command.index(flag) + 1] == value


@pytest.mark.parametrize(
    "option", ["native_host_logit_views", "native_dflash2_defer_context"]
)
@pytest.mark.parametrize("value", ["true", "1", "auto"])
def test_native_sync_selectors_refuse_invalid_mode_before_launch(
    harness, option, value
):
    with pytest.raises(ValueError, match="Unsupported"):
        cli.serve(**{option: value})
    assert harness.launched == 0


@pytest.mark.parametrize("capacity", [512, 1024, 2048])
def test_large_prefill_capacity_is_forwarded(harness, capacity):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit):
        cli.serve(known_token_capacity=capacity)
    command = harness.last_command
    assert command[command.index("--native-known-token-capacity") + 1] == str(capacity)


@pytest.mark.parametrize(
    "options",
    [
        {"native_verify_hc_gdn_fused": "on"},
        {"speed_preset": True, "verify_nax": False},
        {"native_verify_nax": "off", "native_verify_hc_gdn_fused": "on"},
    ],
)
def test_verify_nax_m8_enables_fused_verification_without_separate_nax_flag(
    harness, options
):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit):
        cli.serve(verify_nax_m8=True, **options)
    command = harness.last_command
    for flag in (
        "--native-verify-nax",
        "--native-verify-nax-m8",
        "--native-verify-hc-gdn-fused",
    ):
        assert command[command.index(flag) + 1] == "on"


@pytest.mark.parametrize("mode", ["all", "auto"])
def test_dflash2_verify_rows_are_forwarded_and_refused_for_mtp(harness, tmp_path, mode):
    with pytest.raises(ValueError, match="requires --drafter dflash2"):
        cli.serve(native_dflash2_verify_rows=mode)
    assert harness.launched == 0

    harness.stop_during_wait = False
    dflash2_dir = tmp_path / "dflash2"
    dflash2_dir.mkdir()
    with pytest.raises(SystemExit):
        cli.serve(
            drafter="dflash2",
            dflash2_dir=dflash2_dir,
            native_dflash2_verify_rows=mode,
        )
    command = harness.last_command
    assert command[command.index("--native-dflash2-verify-rows") + 1] == mode


def test_dflash2_verify_rows_refuses_invalid_mode_before_launch(harness):
    with pytest.raises(ValueError, match="expected all or auto"):
        cli.serve(native_dflash2_verify_rows="off")
    assert harness.launched == 0


@pytest.mark.parametrize("speed_preset", [False, True])
def test_m8_and_adaptive_verify_rows_remain_opt_in_for_mtp_and_non_preset(
    harness, tmp_path, speed_preset
):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit):
        cli.serve(speed_preset=speed_preset)
    assert "--native-verify-nax-m8" not in harness.last_command
    assert "--native-dflash2-verify-rows" not in harness.last_command

    if not speed_preset:
        dflash2_dir = tmp_path / "dflash2-optin"
        dflash2_dir.mkdir()
        with pytest.raises(SystemExit):
            cli.serve(drafter="dflash2", dflash2_dir=dflash2_dir, speed_preset=False)
        assert "--native-verify-nax-m8" not in harness.last_command
        assert "--native-dflash2-verify-rows" not in harness.last_command
        assert "--native-prepared-validation" not in harness.last_command


def test_mtp_depth_speed_preset_and_dflash2_precision_forwarding(harness, tmp_path):
    harness.stop_during_wait = False
    with pytest.raises(SystemExit):
        cli.serve(speed_preset=True)
    command = harness.last_command
    assert command[command.index("--derived-pack") + 1].endswith(
        "tokspur-pack-v2.asa.gguf"
    )
    assert command[command.index("--native-mtp-depth") + 1] == "3"
    assert command[command.index("--native-mtp-output-packet") + 1] == "64"
    assert command[command.index("--native-mtp-first-output-packet") + 1] == "8"
    assert command[command.index("--native-prefill-dense") + 1] == "nax"
    assert command[command.index("--native-prefill-grouped") + 1] == "nax"
    assert "--native-prefill-router-fast" not in command
    assert command[command.index("--native-gdn-ported") + 1] == "off"
    assert "--native-gdn-prefill-ported" not in command
    assert command[command.index("--native-mtp-draft-slice") + 1] == "on"
    assert command[command.index("--native-verify-nax") + 1] == "on"
    assert command[command.index("--native-verify-hc-gdn-fused") + 1] == "on"
    assert command[command.index("--native-host-logit-views") + 1] == "on"
    assert command[command.index("--native-logit-topk") + 1] == "on"
    assert command[command.index("--native-ple-async-gather") + 1] == "on"
    assert command[command.index("--native-prepared-validation") + 1] == "on"
    assert command[command.index("--tool-enum-validation") + 1] == "executor"
    assert command[command.index("--tool-arg-validation") + 1] == "executor"
    assert "--native-mtp-fast-round" not in command

    v3_pack = tmp_path / ".tokspur/derived/tokspur-pack-v3.asa.gguf"
    v3_pack.touch()
    with pytest.raises(SystemExit):
        cli.serve(speed_preset=True)
    assert harness.last_command[
        harness.last_command.index("--derived-pack") + 1
    ].endswith("tokspur-pack-v3.asa.gguf")
    with pytest.raises(SystemExit):
        cli.serve()
    assert harness.last_command[
        harness.last_command.index("--derived-pack") + 1
    ].endswith("tokspur-pack-v3.asa.gguf")

    with pytest.raises(SystemExit):
        cli.serve(speed_preset=True, native_mtp_fast_round="on")
    assert (
        harness.last_command[harness.last_command.index("--native-mtp-fast-round") + 1]
        == "on"
    )

    with pytest.raises(SystemExit):
        cli.serve(
            speed_preset=True,
            mtp_depth=2,
            verify_nax=False,
            tool_enum_validation="strict",
            tool_arg_validation="strict",
        )
    override_cmd = harness.last_command
    assert override_cmd[override_cmd.index("--native-mtp-depth") + 1] == "2"
    assert "--native-verify-nax" not in override_cmd
    assert "--native-verify-hc-gdn-fused" not in override_cmd
    assert override_cmd[override_cmd.index("--tool-enum-validation") + 1] == "strict"
    assert "--tool-arg-validation" not in override_cmd

    dflash2_dir = tmp_path / "dflash2"
    dflash2_dir.mkdir()
    with pytest.raises(SystemExit):
        cli.serve(
            drafter="dflash2",
            dflash2_dir=dflash2_dir,
            speed_preset=True,
        )
    df_default_cmd = harness.last_command
    assert (
        df_default_cmd[df_default_cmd.index("--native-dflash2-precision") + 1] == "q8_0"
    )
    assert (
        df_default_cmd[df_default_cmd.index("--native-dflash2-max-drafts") + 1] == "7"
    )
    assert (
        df_default_cmd[df_default_cmd.index("--native-dflash2-vocab-slice") + 1]
        == "32768"
    )
    assert (
        df_default_cmd[df_default_cmd.index("--native-dflash2-defer-context") + 1]
        == "on"
    )
    assert (
        df_default_cmd[df_default_cmd.index("--native-dflash2-verify-rows") + 1]
        == "auto"
    )
    assert df_default_cmd[df_default_cmd.index("--native-verify-nax-m8") + 1] == "on"
    assert (
        df_default_cmd[df_default_cmd.index("--native-prepared-validation") + 1] == "on"
    )

    with pytest.raises(SystemExit):
        cli.serve(
            drafter="dflash2",
            dflash2_dir=dflash2_dir,
            dflash2_precision="bf16",
            dflash2_max_drafts=4,
            native_dflash2_verify_rows="all",
            native_prepared_validation="off",
            verify_nax=False,
            speed_preset=True,
        )
    df_command = harness.last_command
    assert df_command[df_command.index("--native-dflash2-precision") + 1] == "bf16"
    assert df_command[df_command.index("--native-dflash2-max-drafts") + 1] == "4"
    assert df_command[df_command.index("--native-dflash2-vocab-slice") + 1] == "32768"
    assert df_command[df_command.index("--native-dflash2-defer-context") + 1] == "on"
    assert df_command[df_command.index("--native-dflash2-verify-rows") + 1] == "all"
    assert "--native-verify-nax" not in df_command
    assert "--native-verify-nax-m8" not in df_command
    assert "--native-verify-hc-gdn-fused" not in df_command
    assert "--native-prepared-validation" not in df_command

    with pytest.raises(SystemExit):
        cli.serve(verify_nax=True, native_verify_nax="off")
    assert "--native-verify-nax" not in harness.last_command

    with pytest.raises(SystemExit):
        cli.serve(native_verify_nax="on")
    vnax_cmd = harness.last_command
    assert vnax_cmd[vnax_cmd.index("--native-verify-nax") + 1] == "on"

    with pytest.raises(SystemExit):
        cli.serve(speed_preset=True, output_packet=64)
    p64_cmd = harness.last_command
    assert p64_cmd[p64_cmd.index("--native-mtp-output-packet") + 1] == "64"
    assert p64_cmd[p64_cmd.index("--native-mtp-first-output-packet") + 1] == "8"

    with pytest.raises(ValueError, match="cannot exceed"):
        cli.serve(output_packet=8, first_output_packet=16)

    with pytest.raises(SystemExit):
        cli.serve(tool_arg_validation="executor")
    arg_cmd = harness.last_command
    assert arg_cmd[arg_cmd.index("--tool-arg-validation") + 1] == "executor"

    with pytest.raises(ValueError, match="Unsupported tool_arg_validation"):
        cli.serve(tool_arg_validation="permissive")

    with pytest.raises(ValueError, match="removed router-window"):
        cli.serve(native_prefill_router_fast="on")

    with pytest.raises(ValueError, match="removed router-window"):
        cli.serve(speed_preset=True, native_prefill_router_fast="on")


def test_retired_gdn_prefill_refuses_before_launch(harness):
    with pytest.raises(ValueError, match="removed because its source"):
        cli.serve(gdn_prefill_only=True)
    assert harness.launched == 0
