# SPDX-License-Identifier: Apache-2.0
"""Explicit preparation and foreground serving. No implicit service switch."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import __version__

DEFAULT_CLI_IMPL = "rust"
from .local import (  # noqa: E402
    ROOT,
    check_adoption,
    inference_processes,
    isolated_environment,
    port_available,
    private_dir,
    profile,
    read_json,
    state_lock,
    verify_asset,
    write_json,
)


def adopt(target: Path, draft: Path) -> None:
    specs = read_json(ROOT / "profiles/flash-next-assets.json")
    result = {"schema": 1, "verification": "full-sha256"}
    for group, directory in (("target", target), ("draft", draft)):
        directory = directory.resolve(strict=True)
        assets = []
        for spec in specs[group]["files"]:
            print(f"Verifying {group}: {spec['name']}", flush=True)
            assets.append(verify_asset(directory / spec["name"], spec))
        result[group] = {"directory": str(directory), "assets": assets}
    write_json(ROOT / ".tokspur/adoption.json", result)
    print("Existing weights verified in place. No model files copied or downloaded.")


def valid_api_key(path: Path) -> bool:
    """Whether `path` holds a key the server admits: 64 hex digits and a newline."""
    try:
        status = path.lstat()
        text = path.read_bytes()
    except FileNotFoundError:
        return False
    return (
        (status.st_mode & 0o170777) == 0o100600
        and status.st_nlink == 1
        and len(text) == 65
        and text.endswith(b"\n")
        and all(c in b"0123456789abcdef" for c in text[:64])
    )


def configure(port: int) -> None:
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Use an unprivileged local port")
    if not port_available(port):
        raise ValueError(f"Port {port} is occupied; configuration left unchanged")
    data = read_json(ROOT / ".tokspur/adoption.json")
    check_adoption(data)
    p = profile()
    state = private_dir(ROOT / ".tokspur/state")
    if (state / "owner.json").exists() and read_json(state / "owner.json") != {
        "project": "tokspur",
        "schema": 1,
    }:
        raise ValueError("Unknown state owner")
    write_json(state / "owner.json", {"project": "tokspur", "schema": 1})

    # Keep a key only if the server would admit it (64 hex characters + newline,
    # 0600, one link); replace anything else atomically so an interrupted write
    # cannot leave an empty key behind.
    key_file = state / "api-key"
    if key_file.is_symlink():
        raise ValueError("state/api-key must not be a symlink")
    if not valid_api_key(key_file):
        temporary = state / f".api-key.{os.getpid()}.tmp"
        fd = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_WRONLY | os.O_CLOEXEC,
            0o600,
        )
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(secrets.token_hex(32) + "\n")
                stream.flush()
                os.fchmod(stream.fileno(), 0o600)
                os.fsync(stream.fileno())
            os.replace(temporary, key_file)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    target_dir = Path(data["target"]["directory"])
    if target_dir.is_dir():
        model_dir = private_dir(state / "models")
        link = model_dir / p["public_model"]
        if link.is_symlink():
            if (state / "gguf").exists() and link.resolve().is_relative_to(
                state / "gguf"
            ):
                link.unlink()
                link.symlink_to(target_dir, target_is_directory=True)
            elif link.resolve() != target_dir.resolve():
                raise ValueError("Existing model link points elsewhere")
        elif link.exists():
            raise ValueError("Refusing to replace an existing model directory")
        else:
            link.symlink_to(target_dir, target_is_directory=True)

    write_json(
        ROOT / ".tokspur/configured.json",
        {
            "port": port,
            "profile": p,
            "model": p["public_model"],
        },
    )
    print(f"Configured tokspur at http://127.0.0.1:{port}; inference not started.")


def default_derived_pack_path(
    root: Path | None = None, *, pack_v3: bool = False, prefer_v3: bool = False
) -> Path:
    base = ROOT if root is None else root
    derived_dir = base / ".tokspur" / "derived"
    if not pack_v3 and prefer_v3:
        for v3_name in ("tokspur-pack-v3.asa.gguf", "tokspur-pack-v3.gguf"):
            v3_path = derived_dir / v3_name
            if v3_path.is_file():
                return v3_path.resolve()
        # With neither format installed, name the default format (v3) in the refusal.
        if not any(
            (derived_dir / name).is_file()
            for name in ("tokspur-pack-v2.asa.gguf", "tokspur-pack-v2.gguf")
        ):
            return (derived_dir / "tokspur-pack-v3.asa.gguf").resolve()
    asa_name = "tokspur-pack-v3.asa.gguf" if pack_v3 else "tokspur-pack-v2.asa.gguf"
    legacy_name = "tokspur-pack-v3.gguf" if pack_v3 else "tokspur-pack-v2.gguf"
    asa_path = derived_dir / asa_name
    if asa_path.is_file():
        return asa_path.resolve()
    legacy_path = derived_dir / legacy_name
    if legacy_path.is_file():
        return legacy_path.resolve()
    return asa_path.resolve()


def serve(
    allow_raw_gguf: bool = False,
    pack_v3: bool = False,
    derived_pack: Path | str | None = None,
    gdn_prefill_only: bool = False,
    derived_pack_trust_mode: str = "full",
    output_packet: int | None = None,
    first_output_packet: int | None = None,
    prefill_dense: str | None = None,
    prefill_grouped: str | None = None,
    known_token_capacity: int | None = 2048,
    mtp_draft_slice: bool = False,
    verify_nax: bool | None = None,
    verify_nax_m8: bool = False,
    speed_preset: bool = False,
    mtp_depth: int | None = None,
    dflash2_precision: str | None = None,
    dflash2_max_drafts: int | None = None,
    dflash2_vocab_slice: int | None = None,
    native_host_logit_views: str | None = None,
    native_logit_topk: str | None = None,
    native_ple_async_gather: str | None = None,
    native_mtp_fast_round: str | None = None,
    native_prefill_router_fast: str | None = None,
    native_verify_hc_gdn_fused: str | None = None,
    native_dflash2_defer_context: str | None = None,
    native_dflash2_verify_rows: str | None = None,
    native_prepared_validation: str | None = None,
    native_verify_nax: str | None = None,
    drafter: str = "mtp",
    dflash2_dir: Path | str | None = None,
    concurrent_dispatch: bool = True,
    q8_gemv_rg4: str = "off",
    q8_three_msg: str = "off",
    prefill_multiseg: bool = False,
    prefill_swizzle: bool = True,
    qsa_prefill_fused: bool = True,
    hc_consumer_fused: bool = True,
    ple_overlap: bool = True,
    shared_prefix_cache: bool | None = None,
    mtp_lookup: str = "off",
    mtp_lookup_match: int | None = None,
    tool_enum_validation: str | None = None,
    tool_arg_validation: str | None = None,
    private_prefix_trace: bool = False,
    telemetry_stdout: bool = False,
    checkpoint_dir: Path | str | None = None,
    checkpoint_cap_gib: float | None = None,
    checkpoint_floor_gib: float | None = None,
    checkpoint_policy: str = "recompute-cost",
    checkpoint_ssd: bool | None = None,
    checkpoint_ssd_budget_gib: float | None = None,
    checkpoint_ssd_floor_gib: float = 20.0,
    checkpoint_ssd_dir: str | Path | None = None,
    overlap_prefill: bool = False,
) -> None:
    for flag, value in (
        ("native_host_logit_views", native_host_logit_views),
        ("native_logit_topk", native_logit_topk),
        ("native_ple_async_gather", native_ple_async_gather),
        ("native_mtp_fast_round", native_mtp_fast_round),
        ("native_prefill_router_fast", native_prefill_router_fast),
        ("native_verify_hc_gdn_fused", native_verify_hc_gdn_fused),
        ("native_dflash2_defer_context", native_dflash2_defer_context),
        ("native_prepared_validation", native_prepared_validation),
        ("native_verify_nax", native_verify_nax),
    ):
        if value not in (None, "off", "on"):
            raise ValueError(f"Unsupported {flag} mode: {value}")
    if gdn_prefill_only:
        raise ValueError(
            "--gdn-prefill-only was removed because its source could not be verified"
        )
    if native_prefill_router_fast == "on":
        raise ValueError(
            "--native-prefill-router-fast requires the removed router-window implementation"
        )
    if output_packet is None:
        output_packet = 64 if speed_preset else 8
    if prefill_dense is None:
        prefill_dense = "nax" if speed_preset else "f32-staged"
    if prefill_grouped is None:
        prefill_grouped = "nax" if speed_preset else "f32"
    if native_verify_nax is not None:
        verify_nax_enabled = native_verify_nax == "on"
    elif verify_nax is not None:
        verify_nax_enabled = bool(verify_nax)
    else:
        verify_nax_enabled = bool(speed_preset)
    verify_nax_enabled = verify_nax_enabled or verify_nax_m8
    if speed_preset:
        if native_verify_hc_gdn_fused is None and verify_nax_enabled:
            native_verify_hc_gdn_fused = "on"
        if native_host_logit_views is None:
            native_host_logit_views = "on"
        if native_logit_topk is None:
            native_logit_topk = "on"
        if native_ple_async_gather is None:
            native_ple_async_gather = "on"
        if native_prepared_validation is None:
            native_prepared_validation = "on"
        if drafter == "mtp":
            if mtp_depth is None:
                mtp_depth = 3
            mtp_draft_slice = True
        elif drafter == "dflash2":
            if verify_nax_enabled:
                verify_nax_m8 = True
            if dflash2_precision is None:
                dflash2_precision = "q8_0"
            if native_dflash2_defer_context is None:
                native_dflash2_defer_context = "on"
            if dflash2_max_drafts is None:
                dflash2_max_drafts = 7
            if dflash2_vocab_slice is None:
                dflash2_vocab_slice = 32768
            if native_dflash2_verify_rows is None:
                native_dflash2_verify_rows = "auto"
        if output_packet == 64 and first_output_packet is None:
            first_output_packet = 8
    if tool_enum_validation is None:
        tool_enum_validation = "executor" if speed_preset else "strict"
    if tool_arg_validation is None:
        tool_arg_validation = "executor" if speed_preset else "strict"
    if native_mtp_fast_round == "on" and drafter != "mtp":
        raise ValueError(
            "--native-mtp-fast-round on is incompatible with --drafter dflash2"
        )
    if (
        native_prefill_router_fast == "on"
        and prefill_dense != "nax"
        and prefill_grouped != "nax"
    ):
        raise ValueError(
            "--native-prefill-router-fast on requires --prefill-dense nax or --prefill-grouped nax"
        )
    if (
        native_verify_hc_gdn_fused == "on"
        and not verify_nax_enabled
        and drafter != "dflash2"
    ):
        raise ValueError(
            "--native-verify-hc-gdn-fused on requires --verify-nax or --drafter dflash2"
        )
    if native_dflash2_verify_rows not in (None, "all", "auto"):
        raise ValueError(
            f"Unsupported native_dflash2_verify_rows: {native_dflash2_verify_rows} (expected all or auto)"
        )
    if native_dflash2_verify_rows is not None and drafter != "dflash2":
        raise ValueError("--native-dflash2-verify-rows requires --drafter dflash2")
    if dflash2_max_drafts is not None:
        if drafter != "dflash2":
            raise ValueError("--dflash2-max-drafts requires --drafter dflash2")
        if dflash2_max_drafts not in range(1, 8):
            raise ValueError(
                f"Unsupported dflash2_max_drafts: {dflash2_max_drafts} (expected 1..7)"
            )
    if dflash2_vocab_slice is not None:
        if drafter != "dflash2":
            raise ValueError("--dflash2-vocab-slice requires --drafter dflash2")
        if not (
            dflash2_vocab_slice == 0
            or (256 <= dflash2_vocab_slice < 248320 and dflash2_vocab_slice % 32 == 0)
        ):
            raise ValueError(
                f"Unsupported dflash2_vocab_slice: {dflash2_vocab_slice} (expected 0 or a multiple of 32 in 256..248319)"
            )
    if mtp_depth not in (None, 2, 3):
        raise ValueError(f"Unsupported mtp_depth: {mtp_depth} (expected 2 or 3)")
    if dflash2_precision not in (None, "q8_0", "bf16"):
        raise ValueError(
            f"Unsupported dflash2_precision: {dflash2_precision} (expected q8_0 or bf16)"
        )
    if allow_raw_gguf and (pack_v3 or derived_pack is not None):
        raise ValueError(
            "--raw-gguf cannot be combined with --pack-v3 or --derived-pack"
        )
    if derived_pack_trust_mode not in ("full", "local-seal"):
        raise ValueError(
            f"Unsupported derived_pack_trust_mode: {derived_pack_trust_mode}"
        )
    if output_packet not in (8, 64):
        raise ValueError(f"Unsupported output_packet: {output_packet}")
    if first_output_packet is not None:
        if not (1 <= first_output_packet <= 64):
            raise ValueError(
                f"Unsupported first_output_packet: {first_output_packet} (expected 1..64)"
            )
        if first_output_packet > output_packet:
            raise ValueError(
                f"--first-output-packet ({first_output_packet}) cannot exceed --output-packet ({output_packet})"
            )
    if prefill_dense not in ("f32-staged", "nax", "f32", "f16"):
        raise ValueError(f"Unsupported prefill_dense mode: {prefill_dense}")
    if prefill_grouped not in ("f32", "auto", "serial", "nax"):
        raise ValueError(f"Unsupported prefill_grouped mode: {prefill_grouped}")
    dense_staged_or_nax = prefill_dense in ("f32-staged", "nax")
    if dense_staged_or_nax and prefill_grouped == "serial":
        raise ValueError("--prefill-grouped serial requires --prefill-dense f32 or f16")
    if prefill_grouped == "nax" and not dense_staged_or_nax:
        raise ValueError(
            "--prefill-grouped nax requires --prefill-dense f32-staged or nax"
        )
    resolved_known_capacity = (
        2048 if known_token_capacity is None else known_token_capacity
    )
    if resolved_known_capacity not in (512, 1024, 2048):
        raise ValueError(
            f"Unsupported known_token_capacity: {resolved_known_capacity} (expected 512, 1024 or 2048)"
        )
    if drafter not in ("mtp", "dflash2"):
        raise ValueError(f"Unsupported drafter mode: {drafter}")
    if q8_gemv_rg4 not in ("off", "on", "wide"):
        raise ValueError(f"Unsupported q8_gemv_rg4 mode: {q8_gemv_rg4}")
    if q8_three_msg not in ("off", "on", "wide"):
        raise ValueError(f"Unsupported q8_three_msg mode: {q8_three_msg}")
    if mtp_lookup not in ("off", "bypass", "hybrid"):
        raise ValueError(f"Unsupported mtp_lookup mode: {mtp_lookup}")
    if mtp_lookup_match is not None and not (2 <= mtp_lookup_match <= 16):
        raise ValueError(
            f"Unsupported mtp_lookup_match: {mtp_lookup_match} (expected 2..16)"
        )
    if tool_enum_validation not in ("strict", "executor"):
        raise ValueError(
            f"Unsupported tool_enum_validation mode: {tool_enum_validation}"
        )
    if tool_arg_validation not in ("strict", "executor"):
        raise ValueError(f"Unsupported tool_arg_validation mode: {tool_arg_validation}")
    if checkpoint_policy not in ("recompute-cost", "lru"):
        raise ValueError(f"Unsupported checkpoint_policy: {checkpoint_policy}")
    if drafter == "dflash2":
        if dflash2_dir is None:
            raise ValueError("--drafter dflash2 requires --dflash2-dir")
        if mtp_depth is not None:
            raise ValueError("--mtp-depth is incompatible with --drafter dflash2")
        if mtp_draft_slice:
            raise ValueError("--mtp-draft-slice is incompatible with --drafter dflash2")
        if mtp_lookup != "off" or mtp_lookup_match is not None:
            raise ValueError("--mtp-lookup is incompatible with --drafter dflash2")
    elif dflash2_dir is not None:
        raise ValueError("--dflash2-dir requires --drafter dflash2")
    elif dflash2_precision is not None:
        raise ValueError("--dflash2-precision requires --drafter dflash2")

    if checkpoint_cap_gib is not None:
        effective_checkpoint_cap_gib = float(checkpoint_cap_gib)
    elif checkpoint_ssd is False:
        effective_checkpoint_cap_gib = 0.0
    elif checkpoint_ssd_budget_gib is not None:
        effective_checkpoint_cap_gib = float(checkpoint_ssd_budget_gib)
    elif checkpoint_ssd is True:
        effective_checkpoint_cap_gib = 8.0
    else:
        effective_checkpoint_cap_gib = 0.0

    effective_checkpoint_floor_gib = (
        float(checkpoint_floor_gib)
        if checkpoint_floor_gib is not None
        else float(checkpoint_ssd_floor_gib)
    )
    if not (0.0 <= effective_checkpoint_cap_gib <= 32.0):
        raise ValueError("checkpoint_cap_gib must be between 0.0 and 32.0 GiB")
    if not (0.0 <= effective_checkpoint_floor_gib <= 100.0):
        raise ValueError("checkpoint_floor_gib must be between 0.0 and 100.0 GiB")
    effective_shared_prefix = (
        (drafter == "mtp") if shared_prefix_cache is None else bool(shared_prefix_cache)
    )

    state = private_dir(ROOT / ".tokspur/state")
    with state_lock():
        config = read_json(ROOT / ".tokspur/configured.json")
        adoption = read_json(ROOT / ".tokspur/adoption.json")
        check_adoption(adoption)
        if not port_available(config["port"]):
            raise ValueError("Requested port is occupied")
        workers = inference_processes()
        if workers:
            raise ValueError(
                f"Other inference workers found: {workers}. Stop them explicitly first."
            )
        import psutil

        if psutil.virtual_memory().total < 128 * 1024**3:
            raise ValueError(
                "The selected 96 GiB profile requires 128 GiB unified memory"
            )
        # Validate source provenance before loading hundreds of GB of weights.
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/bootstrap.py"), "--verify-only"],
            check=True,
        )
        # The server reads and checks state/api-key itself.
        env = isolated_environment()

        server_bin = None
        for candidate in (
            ROOT / "build/rust-fast-math/release/tokspur-server",
            ROOT / "build/rust-target/release/tokspur-server",
            ROOT / "rust/target/release/tokspur-server",
        ):
            if candidate.is_file():
                server_bin = candidate
                break
        if server_bin is None:
            raise FileNotFoundError(
                "Rust release tokspur-server binary not found; build it with make -C native -f rust.mk rust-server"
            )

        build_dir = ROOT / "build/owned-native"
        worker_bin = build_dir / "scalar-channel-target"
        if not worker_bin.is_file():
            worker_bin = build_dir / "tokspur-native"

        metallib = build_dir / "tokspur-owned.metallib"
        policy = ROOT / "profiles/owned-target-experimental.json"
        if not policy.is_file():
            policy = ROOT / "profiles/m5-max-128.json"

        target_info = adoption.get("target", {})
        target_dir = Path(target_info.get("directory", ""))
        tokenizer = target_dir / "tokenizer.json"
        template = target_dir / "chat_template.jinja"
        if not tokenizer.is_file() or not template.is_file():
            gguf_dir = state / "gguf"
            candidates: list[Path] = []
            internal_model = config.get("internal_model")
            if internal_model and gguf_dir.is_dir():
                candidates.extend(sorted(gguf_dir.glob(f"{internal_model}*")))
            if gguf_dir.is_dir():
                candidates.extend(
                    sorted(
                        p
                        for p in gguf_dir.iterdir()
                        if p.is_dir() and p not in candidates
                    )
                )
            if target_dir.parent.is_dir() and target_dir.parent not in candidates:
                candidates.append(target_dir.parent)
            for candidate in candidates:
                cand_tok = candidate / "tokenizer.json"
                cand_tmpl = candidate / "chat_template.jinja"
                if cand_tok.is_file() and not tokenizer.is_file():
                    tokenizer = cand_tok
                if cand_tmpl.is_file() and not template.is_file():
                    template = cand_tmpl
                if tokenizer.is_file() and template.is_file():
                    break

        if derived_pack is not None:
            dp = Path(derived_pack)
            pack_file = dp if dp.is_absolute() else (Path.cwd() / dp).resolve()
        else:
            pack_file = default_derived_pack_path(ROOT, pack_v3=pack_v3, prefer_v3=True)
        shards = [asset["path"] for asset in target_info.get("assets", [])]

        if not allow_raw_gguf:
            if not pack_file.is_file():
                raise FileNotFoundError(
                    f"Lossless derived weight pack not found at {pack_file}; "
                    "run pack construction or pass --raw-gguf to allow raw GGUF fallback"
                )
            derived_pack_flags = [
                "--derived-pack",
                str(pack_file),
                "--derived-pack-auth-workers",
                "16",
                *(
                    ["--derived-pack-trust-mode", derived_pack_trust_mode]
                    if derived_pack_trust_mode != "full"
                    else []
                ),
            ]
        else:
            derived_pack_flags = []

        if drafter == "dflash2":
            assert dflash2_dir is not None
            dflash2_path = Path(dflash2_dir).resolve()
            if not dflash2_path.is_dir():
                raise FileNotFoundError(
                    f"DFlash2 checkpoint directory not found: {dflash2_path}"
                )
            speculative_flags = [
                "--native-drafter",
                "dflash2",
                "--native-dflash2-dir",
                str(dflash2_path),
                "--native-mtp-depth",
                "7",
                "--native-mtp-request-carry",
                "on",
                "--native-mtp-draft-attention",
                "split16",
                *(
                    ["--native-dflash2-precision", dflash2_precision]
                    if dflash2_precision is not None
                    else []
                ),
                *(
                    ["--native-dflash2-max-drafts", str(dflash2_max_drafts)]
                    if dflash2_max_drafts is not None
                    else []
                ),
                *(
                    ["--native-dflash2-vocab-slice", str(dflash2_vocab_slice)]
                    if dflash2_vocab_slice is not None
                    else []
                ),
            ]
        else:
            draft_assets = adoption.get("draft", {}).get("assets", [])
            mtp_asset = None
            for asset in draft_assets:
                if (
                    asset.get("sha256")
                    == "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6"
                ):
                    mtp_asset = asset
                    break
            if mtp_asset is None and len(draft_assets) == 1:
                mtp_asset = draft_assets[0]

            if mtp_asset is None:
                raise ValueError(
                    "Adopted draft assets do not contain verified MTP sidecar (blk.48); re-run tokspur adopt"
                )
            mtp_sidecar = Path(mtp_asset.get("path", ""))
            if not mtp_sidecar.is_file():
                raise FileNotFoundError(
                    f"Adopted MTP sidecar file not found: {mtp_sidecar}"
                )
            speculative_flags = [
                "--native-mtp-sidecar",
                str(mtp_sidecar),
                "--native-mtp-depth",
                str(2 if mtp_depth is None else mtp_depth),
                "--native-mtp-request-carry",
                "on",
                "--native-mtp-draft-attention",
                "tiled128",
            ]

        staged_carrier = (
            "on"
            if dense_staged_or_nax and prefill_grouped in ("f32", "auto", "nax")
            else "off"
        )
        raw_ckpt_dir = checkpoint_dir or checkpoint_ssd_dir
        if raw_ckpt_dir is None:
            resolved_ckpt_dir = state / "checkpoints"
        else:
            ckpt_path = Path(raw_ckpt_dir)
            resolved_ckpt_dir = (
                ckpt_path if ckpt_path.is_absolute() else (Path.cwd() / ckpt_path)
            )

        carrier_flags = [
            "--native-scalar-channel",
            "on",
            "--native-fast-math",
            "channel",
            "--native-qsa-stage-v8",
            "on",
            *speculative_flags,
            "--native-mtp-output-packet",
            str(output_packet),
            *(
                ["--native-mtp-first-output-packet", str(first_output_packet)]
                if first_output_packet is not None
                else []
            ),
            "--native-qsa-sparse",
            "off",
            "--native-gdn-ported",
            "off",
            "--native-prefill-input512",
            "on",
            "--native-known-token-capacity",
            str(resolved_known_capacity),
            *(["--native-raw-decode", "on"] if allow_raw_gguf else []),
            *(["--native-mtp-draft-slice", "on"] if mtp_draft_slice else []),
            *(["--native-verify-nax", "on"] if verify_nax_enabled else []),
            *(["--native-verify-nax-m8", "on"] if verify_nax_m8 else []),
            *(
                ["--native-dflash2-verify-rows", native_dflash2_verify_rows]
                if native_dflash2_verify_rows is not None
                else []
            ),
            *(
                ["--native-prepared-validation", "on"]
                if native_prepared_validation == "on"
                else []
            ),
            *(
                ["--native-host-logit-views", native_host_logit_views]
                if native_host_logit_views is not None
                else []
            ),
            *(
                ["--native-logit-topk", native_logit_topk]
                if native_logit_topk is not None
                else []
            ),
            *(
                ["--native-ple-async-gather", native_ple_async_gather]
                if native_ple_async_gather is not None
                else []
            ),
            *(
                ["--native-mtp-fast-round", native_mtp_fast_round]
                if native_mtp_fast_round is not None
                else []
            ),
            *(
                ["--native-prefill-router-fast", native_prefill_router_fast]
                if native_prefill_router_fast is not None
                else []
            ),
            *(
                ["--native-verify-hc-gdn-fused", native_verify_hc_gdn_fused]
                if native_verify_hc_gdn_fused is not None
                else []
            ),
            *(
                ["--native-dflash2-defer-context", native_dflash2_defer_context]
                if native_dflash2_defer_context is not None
                else []
            ),
            "--native-prefill-dense",
            prefill_dense,
            "--native-prefill-grouped",
            prefill_grouped,
            "--native-prefill-hc-rows",
            staged_carrier,
            "--native-prefill-router-window",
            "off",
            "--native-prefill-hc-down-window",
            "off",
            "--native-prefill-hc-down-f32",
            "off",
            "--native-prefill-final-logits",
            "on",
            "--native-q8-three-rows",
            "on",
            *(["--native-concurrent-dispatch", "on"] if concurrent_dispatch else []),
            *(["--native-q8-gemv-rg4", q8_gemv_rg4] if q8_gemv_rg4 != "off" else []),
            *(["--native-q8-three-msg", q8_three_msg] if q8_three_msg != "off" else []),
            *(["--native-prefill-multiseg", "on"] if prefill_multiseg else []),
            *(["--native-prefill-swizzle", "on"] if prefill_swizzle else []),
            *(["--native-qsa-prefill-fused", "on"] if qsa_prefill_fused else []),
            *(["--native-hc-consumer-fused", "on"] if hc_consumer_fused else []),
            *(["--native-ple-overlap", "on"] if ple_overlap else []),
            *(["--shared-prefix-cache", "on"] if effective_shared_prefix else []),
            *(["--native-mtp-lookup", mtp_lookup] if mtp_lookup != "off" else []),
            *(
                ["--native-mtp-lookup-match", str(mtp_lookup_match)]
                if mtp_lookup_match is not None
                else []
            ),
            "--tool-enum-validation",
            tool_enum_validation,
            *(
                ["--tool-arg-validation", tool_arg_validation]
                if tool_arg_validation != "strict"
                else []
            ),
            *(["--private-prefix-trace", "on"] if private_prefix_trace else []),
            *(["--telemetry-stdout", "on"] if telemetry_stdout else []),
            *(
                [
                    "--checkpoint-ssd",
                    "on",
                    "--checkpoint-cap-gib",
                    str(effective_checkpoint_cap_gib),
                    "--checkpoint-floor-gib",
                    str(effective_checkpoint_floor_gib),
                    "--checkpoint-dir",
                    str(resolved_ckpt_dir),
                    "--checkpoint-policy",
                    str(checkpoint_policy),
                ]
                if effective_checkpoint_cap_gib > 0.0
                else []
            ),
            *(["--overlap-prefill", "on"] if overlap_prefill else []),
        ]

        command = [
            str(server_bin),
            "--worker",
            str(worker_bin),
            "--policy",
            str(policy),
            "--library",
            str(metallib),
            "--tokenizer",
            str(tokenizer),
            "--template",
            str(template),
            "--state-dir",
            str(state),
            "--model-alias",
            config.get("model", "tokspur"),
            "--port",
            str(config["port"]),
            "--allow-full-model",
            "true",
            *derived_pack_flags,
            *carrier_flags,
            *shards,
        ]
        child = None
        previous = {}
        owned = {}
        stop_requested = False

        def group_signal(number):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, number)

        def stop(number, frame):
            nonlocal stop_requested
            # Signal callbacks only request shutdown. Process operations and
            # their bounded TERM/KILL waits happen in one cleanup path below.
            # Repeated signals cannot interrupt that drain or reenter Popen.
            stop_requested = True

        def live_owned():
            remaining = []
            for (pid, created), owned_proc in owned.items():
                try:
                    if (
                        owned_proc.is_running()
                        and owned_proc.create_time() == created
                        and owned_proc.status() != psutil.STATUS_ZOMBIE
                    ):
                        remaining.append((pid, owned_proc))
                except psutil.NoSuchProcess:
                    continue
            return remaining

        def drain_owned(seconds):
            deadline = time.monotonic() + seconds
            while True:
                remaining = live_owned()
                if not remaining or time.monotonic() >= deadline:
                    return remaining
                time.sleep(0.1)

        # Install the handlers before the child exists, so an early Ctrl-C is
        # recorded and acted on instead of raised between Popen and the loop.
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, stop)
        try:
            # Reuse the pinned supervisor primitive: an abruptly killed tokspur
            # parent must not leave its API (and therefore native worker) orphaned.
            # A signal delivered after handler installation but before launch
            # must not start a model just to terminate it. If it arrives during
            # Popen, the assigned child still reaches the same bounded finally.
            if not stop_requested:
                child = subprocess.Popen(command, env=env, start_new_session=True)
                write_json(
                    state / "process.json",
                    {"pid": child.pid, "supervisor": os.getpid()},
                )
            while child is not None and not stop_requested:
                try:
                    for process in psutil.Process(child.pid).children(recursive=True):
                        # macOS XPC shader compiler services are system-owned,
                        # not model workers; do not manage their lifetime.
                        if not process.exe().startswith(("/System/", "/usr/libexec/")):
                            owned[(process.pid, process.create_time())] = process
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                try:
                    result = child.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            try:
                if child is not None:
                    if child.poll() is None:
                        group_signal(signal.SIGTERM)
                        try:
                            child.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            # The API group ignored SIGTERM; never leave it running.
                            group_signal(signal.SIGKILL)
                            child.wait(timeout=15)
                    remaining = drain_owned(45)
                    if remaining:
                        # Model workers hold read-only mappings, so a forced exit
                        # loses nothing; an orphan would block the one-model rule.
                        for _pid, process in remaining:
                            with contextlib.suppress(
                                psutil.NoSuchProcess, psutil.AccessDenied
                            ):
                                process.kill()
                        remaining = drain_owned(15)
                    if remaining:
                        pids = [pid for pid, _process in remaining]
                        raise ValueError(
                            f"Owned processes have not exited: {pids}; retained process receipt"
                        )
                    (state / "process.json").unlink(missing_ok=True)
            finally:
                # Keep our non-raising handlers installed through every wait,
                # escalation and descendant drain, even when cleanup fails.
                for number, handler in previous.items():
                    signal.signal(number, handler)
        raise SystemExit(0 if stop_requested else result)


def main() -> None:
    if os.environ.get("TOKSPUR_CLI_IMPL") == "python" and any(
        arg.split("=", 1)[0]
        in {
            "--native-prepared-validation",
            "--native-verify-cost-profile",
            "--native-dflash2-draft-policy",
            "--native-verify-m8-splits",
            "--worker-burst-pacing",
        }
        for arg in sys.argv[1:]
    ):
        raise SystemExit("these runtime-check controls require the Rust CLI")
    mode = os.environ.get("TOKSPUR_CLI_IMPL") or DEFAULT_CLI_IMPL
    if mode not in ("rust", "python"):
        sys.stderr.write(
            "tokspur: invalid TOKSPUR_CLI_IMPL (expected rust or python)\n"
        )
        raise SystemExit(2)
    if mode == "rust" or sys.argv[1:2] == ["check-model-config"]:
        rust_bin = ROOT / "build/rust-fast-math/release/tokspur"
        if not rust_bin.is_file():
            sys.stderr.write(
                f"tokspur: Rust CLI binary not found at {rust_bin}; run scripts/bootstrap.py\n"
            )
            raise SystemExit(1)
        os.execv(str(rust_bin), [str(rust_bin), *sys.argv[1:]])

    parser = argparse.ArgumentParser(description="Isolated tokspur runtime experiments")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Read-only readiness summary; never loads a model")
    sub.add_parser(
        "check-model-config",
        help="Compare a local config.json with the Flash-Next text profile",
    )
    a = sub.add_parser("adopt", help="Fully hash existing weights without copying them")
    a.add_argument("--target", type=Path, required=False, default=None)
    a.add_argument("--draft", type=Path, required=False, default=None)
    a.add_argument(
        "--check-only",
        action="store_true",
        default=False,
        help="Check existing adopted weights without rehashing",
    )
    c = sub.add_parser(
        "configure", help="Write isolated settings; never starts inference"
    )
    c.add_argument("--port", type=int, default=profile()["port"])
    sub.add_parser("plan", help="Show the selected native policy")
    sub.add_parser(
        "verify-build", help="Verify build-manifest.json and launch artefacts"
    )
    s = sub.add_parser("serve", help="Foreground API; refuses other inference workers")
    s.add_argument(
        "--raw-gguf",
        action="store_true",
        default=False,
        help="Explicit opt-in to raw GGUF instead of derived pack",
    )
    s.add_argument(
        "--pack-v3",
        action="store_true",
        default=False,
        help="Require planar derived pack v3 (the default when installed; v2 is the fallback)",
    )
    s.add_argument(
        "--derived-pack",
        type=Path,
        default=None,
        help="Explicit derived pack GGUF file path",
    )
    s.add_argument(
        "--gdn-prefill-only",
        action="store_true",
        default=False,
        help="Retired GDN prefill experiment",
    )
    s.add_argument(
        "--derived-pack-trust-mode",
        choices=["full", "local-seal"],
        default="full",
        help="Derived pack verification trust mode",
    )
    s.add_argument(
        "--output-packet",
        type=int,
        default=None,
        choices=[8, 64],
        help="MTP output packet size (8 or 64)",
    )
    s.add_argument(
        "--first-output-packet",
        "--native-mtp-first-output-packet",
        dest="first_output_packet",
        type=int,
        default=None,
        help="Opt-in initial speculative output packet cap (1..64) before steady-state packets",
    )
    s.add_argument(
        "--prefill-dense",
        choices=["f32-staged", "nax", "f32", "f16"],
        default=None,
        help="Dense Q8 prefill kernel mode (default: f32-staged)",
    )
    s.add_argument(
        "--prefill-grouped",
        choices=["f32", "auto", "serial", "nax"],
        default=None,
        help="Grouped MoE prefill kernel mode (default: f32)",
    )
    s.add_argument(
        "--known-token-capacity",
        type=int,
        choices=[512, 1024, 2048],
        default=2048,
        help="Known-token IPC packet capacity (default: 2048)",
    )
    s.add_argument(
        "--mtp-draft-slice",
        action="store_true",
        default=False,
        help="Opt-in top-32K MTP draft vocabulary slice",
    )
    s.add_argument(
        "--verify-nax",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt-in Metal 4 NAX batched speculative verification (default: off)",
    )
    s.add_argument(
        "--native-verify-nax",
        choices=["off", "on"],
        default=None,
        help="Explicit native batched verification mode (default: off)",
    )
    s.add_argument(
        "--verify-nax-m8",
        action="store_true",
        default=False,
        help="Opt-in Metal 4 NAX m=8 dense tensor-ops speculative verification",
    )
    s.add_argument(
        "--speed-preset",
        action="store_true",
        default=False,
        help="Opt-in Pack v3, NAX, MTP3/DFlash2 and executor speed preset",
    )
    s.add_argument(
        "--mtp-depth",
        type=int,
        choices=[2, 3],
        default=None,
        help="Opt-in MTP speculative lookahead depth (default: 2)",
    )
    s.add_argument(
        "--dflash2-precision",
        choices=["q8_0", "bf16"],
        default=None,
        help="Opt-in DFlash2 dense projection precision (default: q8_0)",
    )
    s.add_argument(
        "--dflash2-max-drafts",
        "--native-dflash2-max-drafts",
        dest="dflash2_max_drafts",
        type=int,
        choices=range(1, 8),
        default=None,
        help="Opt-in DFlash2 maximum drafted proposals per round (1..7)",
    )
    s.add_argument(
        "--dflash2-vocab-slice",
        "--native-dflash2-vocab-slice",
        dest="dflash2_vocab_slice",
        type=int,
        default=None,
        help="Opt-in DFlash2 draft vocabulary slice (0 or multiple of 32)",
    )
    s.add_argument(
        "--native-host-logit-views",
        choices=["off", "on"],
        default=None,
        help="Opt-in native host logit views (default: off)",
    )
    s.add_argument(
        "--native-logit-topk",
        choices=["off", "on"],
        default=None,
        help="Opt-in GPU two-stage logit top-k verification (default: off)",
    )
    s.add_argument(
        "--native-ple-async-gather",
        choices=["off", "on"],
        default=None,
        help="Opt-in asynchronous 16-bucket PLE gather overlap (default: off)",
    )
    s.add_argument(
        "--native-mtp-fast-round",
        "--mtp-fast-round",
        dest="native_mtp_fast_round",
        choices=["off", "on"],
        default=None,
        help="Opt-in GPU-resident MTP fast round and fused priming (default: off)",
    )
    s.add_argument(
        "--native-prefill-router-fast",
        "--prefill-router-fast",
        dest="native_prefill_router_fast",
        choices=["off", "on"],
        default=None,
        help="Opt-in 1-pass BF16 NAX MoE router projection in prefill (default: off)",
    )
    s.add_argument(
        "--native-verify-hc-gdn-fused",
        choices=["off", "on"],
        default=None,
        help="Opt-in fused HC+GDN multi-row verification (default: off)",
    )
    s.add_argument(
        "--native-dflash2-defer-context",
        choices=["off", "on"],
        default=None,
        help="Opt-in native deferred DFlash2 context (default: off)",
    )
    s.add_argument(
        "--native-dflash2-verify-rows",
        choices=["all", "auto"],
        default=None,
        help="Opt-in DFlash2 adaptive verify row budget (default: all)",
    )
    s.add_argument(
        "--drafter",
        choices=["mtp", "dflash2"],
        default="mtp",
        help="Speculative drafter mode (default: mtp)",
    )
    s.add_argument(
        "--dflash2-dir",
        type=Path,
        default=None,
        help="Opt-in DFlash2 safetensors checkpoint directory",
    )
    s.add_argument(
        "--concurrent-dispatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Metal concurrent compute dispatch (default: on)",
    )
    s.add_argument(
        "--q8-gemv-rg4",
        choices=["off", "on", "wide"],
        default="off",
        help="Retired Q8 GEMV experiment (default: off)",
    )
    s.add_argument(
        "--q8-three-msg",
        choices=["off", "on", "wide"],
        default="off",
        help="Retired multi-SG Q8 experiment (default: off)",
    )
    s.add_argument(
        "--prefill-multiseg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retired multi-segment prefill (default: off)",
    )
    s.add_argument(
        "--prefill-swizzle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Swizzled QSA prefill projection (default: on)",
    )
    s.add_argument(
        "--qsa-prefill-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fused QSA prefill operator (default: on)",
    )
    s.add_argument(
        "--hc-consumer-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fused HC consumer operator (default: on)",
    )
    s.add_argument(
        "--ple-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Chunk-ahead PLE prefill overlap (default: on)",
    )
    s.add_argument(
        "--shared-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Host-RAM shared system-prefix cache (default: on for mtp)",
    )
    s.add_argument(
        "--mtp-lookup",
        choices=["off", "bypass", "hybrid"],
        default="off",
        help="Opt-in MTP prompt-lookup mode (default: off)",
    )
    s.add_argument(
        "--mtp-lookup-match",
        type=int,
        choices=range(2, 17),
        metavar="2..16",
        default=None,
        help="Opt-in MTP prompt-lookup match length (2..16)",
    )
    s.add_argument(
        "--tool-enum-validation",
        choices=["strict", "executor"],
        default=None,
        help="OpenAI tool enum schema policy (default: strict)",
    )
    s.add_argument(
        "--tool-arg-validation",
        choices=["strict", "executor"],
        default=None,
        help="OpenAI tool argument schema policy (default: strict)",
    )
    s.add_argument(
        "--private-prefix-trace",
        action="store_true",
        default=False,
        help="Opt-in private prefix reuse trace telemetry",
    )
    s.add_argument(
        "--telemetry-stdout",
        action="store_true",
        default=False,
        help="Opt-in structured JSONL telemetry on stdout",
    )
    s.add_argument(
        "--checkpoint-dir",
        "--checkpoint-ssd-dir",
        dest="checkpoint_dir",
        type=Path,
        default=None,
        help="Custom SSD checkpoint storage directory (default: .tokspur/state/checkpoints)",
    )
    s.add_argument(
        "--checkpoint-cap-gib",
        "--checkpoint-ssd-budget-gib",
        dest="checkpoint_cap_gib",
        type=float,
        default=None,
        help="SSD checkpoint capacity budget in GiB (default: 8.0 for mtp, 0 disables, max: 32.0)",
    )
    s.add_argument(
        "--checkpoint-floor-gib",
        "--checkpoint-ssd-floor-gib",
        dest="checkpoint_floor_gib",
        type=float,
        default=None,
        help="APFS free space floor in GiB (default: 20.0)",
    )
    s.add_argument(
        "--checkpoint-policy",
        choices=["recompute-cost", "lru"],
        default="recompute-cost",
        help="Eviction priority policy (default: recompute-cost)",
    )
    s.add_argument(
        "--overlap-prefill",
        action="store_true",
        default=False,
        help="Enable 16-token overlap background prefill with boundary protection",
    )
    args = parser.parse_args()
    try:
        if args.command == "adopt":
            with state_lock():
                if args.check_only:
                    data = read_json(ROOT / ".tokspur/adoption.json")
                    check_adoption(data)
                    print(
                        "Existing weights verified in place. No model files copied or downloaded."
                    )
                else:
                    if not args.target or not args.draft:
                        raise ValueError(
                            "--target and --draft are required unless --check-only is specified"
                        )
                    adopt(args.target, args.draft)
        elif args.command == "configure":
            with state_lock():
                configure(args.port)
        elif args.command == "serve":
            serve(
                allow_raw_gguf=args.raw_gguf,
                pack_v3=args.pack_v3,
                derived_pack=args.derived_pack,
                gdn_prefill_only=args.gdn_prefill_only,
                derived_pack_trust_mode=args.derived_pack_trust_mode,
                output_packet=args.output_packet,
                first_output_packet=args.first_output_packet,
                prefill_dense=args.prefill_dense,
                prefill_grouped=args.prefill_grouped,
                known_token_capacity=args.known_token_capacity,
                mtp_draft_slice=args.mtp_draft_slice,
                verify_nax=args.verify_nax,
                verify_nax_m8=args.verify_nax_m8,
                speed_preset=args.speed_preset,
                mtp_depth=args.mtp_depth,
                dflash2_precision=args.dflash2_precision,
                dflash2_max_drafts=args.dflash2_max_drafts,
                dflash2_vocab_slice=args.dflash2_vocab_slice,
                native_host_logit_views=args.native_host_logit_views,
                native_logit_topk=args.native_logit_topk,
                native_ple_async_gather=args.native_ple_async_gather,
                native_mtp_fast_round=args.native_mtp_fast_round,
                native_prefill_router_fast=args.native_prefill_router_fast,
                native_verify_hc_gdn_fused=args.native_verify_hc_gdn_fused,
                native_dflash2_defer_context=args.native_dflash2_defer_context,
                native_dflash2_verify_rows=args.native_dflash2_verify_rows,
                native_verify_nax=args.native_verify_nax,
                drafter=args.drafter,
                dflash2_dir=args.dflash2_dir,
                concurrent_dispatch=args.concurrent_dispatch,
                q8_gemv_rg4=args.q8_gemv_rg4,
                q8_three_msg=args.q8_three_msg,
                prefill_multiseg=args.prefill_multiseg,
                prefill_swizzle=args.prefill_swizzle,
                qsa_prefill_fused=args.qsa_prefill_fused,
                hc_consumer_fused=args.hc_consumer_fused,
                ple_overlap=args.ple_overlap,
                shared_prefix_cache=args.shared_prefix_cache,
                mtp_lookup=args.mtp_lookup,
                mtp_lookup_match=args.mtp_lookup_match,
                tool_enum_validation=args.tool_enum_validation,
                tool_arg_validation=args.tool_arg_validation,
                private_prefix_trace=args.private_prefix_trace,
                telemetry_stdout=args.telemetry_stdout,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_cap_gib=args.checkpoint_cap_gib,
                checkpoint_floor_gib=args.checkpoint_floor_gib,
                checkpoint_policy=args.checkpoint_policy,
                overlap_prefill=args.overlap_prefill,
            )
        elif args.command == "verify-build":
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/bootstrap.py"), "--verify-only"],
                check=True,
            )
        elif args.command == "plan":
            print(json.dumps(profile(), indent=2))
        else:
            print(
                json.dumps(
                    {
                        "version": __version__,
                        "cli_implementation": "python",
                        "layout": "checkout",
                        "native_built": (
                            ROOT / "build/owned-native/tokspur-native"
                        ).is_file(),
                        "server_built": (
                            (
                                ROOT / "build/rust-fast-math/release/tokspur-server"
                            ).is_file()
                            or (
                                ROOT / "build/rust-target/release/tokspur-server"
                            ).is_file()
                            or (ROOT / "rust/target/release/tokspur-server").is_file()
                        ),
                        "weights_adopted": (ROOT / ".tokspur/adoption.json").is_file(),
                        "configured": (ROOT / ".tokspur/configured.json").is_file(),
                        "default_port_available": port_available(profile()["port"]),
                        "other_inference_workers": inference_processes(),
                    },
                    indent=2,
                )
            )
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        parser.exit(1, f"tokspur: {error}\n")
    except KeyError as error:
        parser.exit(
            1, f"tokspur: missing configuration field {error}; re-run adopt/configure\n"
        )
    except ModuleNotFoundError as error:
        parser.exit(1, f"tokspur: {error}; run scripts/bootstrap.py first\n")


if __name__ == "__main__":
    main()
