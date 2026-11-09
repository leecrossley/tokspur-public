#!/usr/bin/env python3
"""Record model-free native CPU/Metal regression evidence; never a speed campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted((ROOT / "native").rglob("*"))
        if path.is_file()
    }


def host_state(*, deadline: float | None = None) -> dict:
    def output(command):
        options = {}
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Host probe absolute deadline")
            options["timeout"] = remaining
        return subprocess.check_output(command, text=True, **options)

    pressure = output(["memory_pressure", "-Q"])
    match = re.search(r"free percentage:\s*(\d+)%", pressure)
    if match is None:
        raise RuntimeError("Cannot read host headroom for the regression run")
    return {
        "memory_free_percent": int(match[1]),
        "swap": output(["sysctl", "vm.swapusage"]).strip(),
    }


CPU_SUITES = tuple(
    (
        "pipeline_key dflash2_checkpoint shared_prefix_restore speculative_controls gguf numeric tokenizer unicode json chat operators model routing gdn "
        "reference attention_math memory sampling response response_protocol "
        "q8_layout q8_many gdn_tile recurrent_timing_protocol "
        "recurrent_replay_timing_protocol serve_protocol http owned_worker "
        "prompt_lookup transcript_replay derived_pack host_telemetry"
    ).split()
)
METAL_SUITES = tuple(
    (
        "metal graph moe recurrent attention ple stack profile small_gemv "
        "block_gemv layout_gemv many_gemv gdn_tile_gpu gdn_core_tile "
        "recurrent_tile graph_replay hc_q8_gemv q8_split_gemv"
    ).split()
)
ARTIFACT_NAMES = ("tokspur-native", "tokspur-server", "tokspur-owned.metallib") + tuple(
    "test-" + name for name in CPU_SUITES + METAL_SUITES
)
LINK_COMMAND_INDICES = (3, 4)
PROTOCOL_COMMAND_INDEX = 5
BASE_PROTOCOL_CHECKS, STACK_PROTOCOL_CHECKS = 245, 59
PROTOCOL_CHECKS = BASE_PROTOCOL_CHECKS + STACK_PROTOCOL_CHECKS


def generated_commands(build, python):
    """One owned producer/consumer source of truth; no command is run here."""
    binary, server, library = (
        str(build / "tokspur-native"),
        str(build / "tokspur-server"),
        str(build / "tokspur-owned.metallib"),
    )
    return [
        ["c++", "--version"],
        ["xcrun", "metal", "--version"],
        [binary, "host-info"],
        ["otool", "-L", binary],
        ["otool", "-L", server],
        [python, str(build.parent.parent / "scripts/check_owned_target_protocol.py")],
        *[[str(build / ("test-" + name))] for name in CPU_SUITES],
        *[[str(build / ("test-" + name)), library] for name in METAL_SUITES],
    ]


def system_only_links(output):
    dependencies = [
        line.strip().split()[0] for line in output.splitlines()[1:] if line.strip()
    ]
    return bool(dependencies) and all(
        path.startswith(("/System/Library/", "/usr/lib/")) for path in dependencies
    )


def complete_protocol_output(output):
    return re.findall(
        r"^Owned CPU target protocol checks passed: (.*)$", output, re.M
    ) == [str(BASE_PROTOCOL_CHECKS)] and re.findall(
        r"^Owned CPU target stack-experiment protocol checks passed: (.*)$",
        output,
        re.M,
    ) == [str(STACK_PROTOCOL_CHECKS)]


def main(argv: list[str] | tuple[str, ...] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-plane", choices=("cpp", "rust"), default="cpp")
    args = parser.parse_args(list(argv))
    os.umask(0o077)
    build = ROOT / "build/owned-native"
    rust_release = ROOT / "build/rust-fast-math/release"
    files = [build / name for name in ARTIFACT_NAMES]
    protocol_checker = ROOT / "scripts/check_owned_target_protocol.py"
    directory = (
        ROOT
        / ".tokspur/receipts"
        / (
            "owned-gpu-regression-"
            + time.strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    directory.mkdir(mode=0o700, parents=True)
    receipt: dict[str, object] = {
        "schema": 2 if args.control_plane == "rust" else 1,
        "scope": "owned model-free CPU/Metal regression, not full-model quality/performance/package qualification",
        "pass": False,
        "native_sources": sources(),
        "harness_sha256": digest(Path(__file__)),
        "protocol_checker_sha256": digest(protocol_checker),
        "artifacts": {p.name: digest(p) for p in files},
        "commands": [],
    }
    if args.control_plane == "rust":
        receipt["control_plane"] = "rust"
        receipt["rust_artifacts"] = {
            name: digest(rust_release / name) for name in ("tokspur", "tokspur-server")
        }

    def run(argv, timeout=180):
        result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        receipt["commands"].append(
            {
                "argv": list(map(str, argv)),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.returncode:
            raise RuntimeError("Native component command failed: " + str(argv[0]))
        return result.stdout

    try:
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] in {
                "tokspur-native",
                "tokspur-server",
                "splash",
                "llama-server",
                "llama-cli",
                "omlx",
                "omlx-server",
                "ollama",
            }:
                raise RuntimeError(
                    "Stop the exact known inference process before native component tests"
                )
        receipt["host_before"] = host_state()
        for index, argv in enumerate(generated_commands(build, os.sys.executable)):
            output = run(argv)
            if index in LINK_COMMAND_INDICES and not system_only_links(output):
                raise RuntimeError("Missing or non-system owned executable dependency")
            if index == PROTOCOL_COMMAND_INDEX and not complete_protocol_output(output):
                raise RuntimeError("Incomplete owned target protocol proof")
            if index > PROTOCOL_COMMAND_INDEX:
                print("Native component passed: " + Path(argv[0]).name, flush=True)
        if args.control_plane == "rust":
            for rname in ("tokspur", "tokspur-server"):
                rout = run(["/usr/bin/otool", "-L", str(rust_release / rname)])
                if not system_only_links(rout):
                    raise RuntimeError(
                        "Missing or non-system Rust executable dependency"
                    )
            metallib = build / "tokspur-owned.metallib"
            if metallib.read_bytes()[:4] != b"MTLB":
                raise RuntimeError("tokspur-owned.metallib lacks MTLB magic")
            objdump = run(
                ["/usr/bin/xcrun", "metal-objdump", "--function-list", str(metallib)]
            )
            if not [ln for ln in objdump.splitlines() if ln.strip()]:
                raise RuntimeError("Empty metal-objdump kernel inventory")
        receipt["system_only_component_link_check"] = True
        if (
            sources() != receipt["native_sources"]
            or {p.name: digest(p) for p in files} != receipt["artifacts"]
            or digest(Path(__file__)) != receipt["harness_sha256"]
            or digest(protocol_checker) != receipt["protocol_checker_sha256"]
        ):
            raise RuntimeError(
                "Native source, harness or artifact changed during checks"
            )
        receipt["host_after"] = host_state()
        receipt["pass"] = True
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("Private native regression receipt: " + str(directory), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
