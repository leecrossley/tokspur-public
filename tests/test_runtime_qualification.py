"""Generated controls for phase boundaries and complete warm-state admission."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "runtime_qualification",
    Path(__file__).resolve().parents[1] / "scripts/runtime_qualification.py",
)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def reading(n=1, **changes):
    return {
        "monotonic_raw_ns": n * 100_000_000,
        "swapins": 10,
        "swapouts": 20,
        "swap_used": 100,
        "page_size": 16384,
        "pressure": 1,
        "thermal": 0,
        "pid": 12,
        "rusage_available": True,
        "footprint": 30,
        "peak_footprint": 40,
        **changes,
    }


def test_phase_boundaries_charge_previous_phase_and_preserve_untimed_io():
    g = q.PhaseSwapGuard(reading(), native_limit_bytes=50)
    g.sample(reading(2), "cooling")
    g.sample(reading(3, swapins=14), "decode")
    g.sample(reading(4, swapins=14), "restore")
    g.sample(reading(5, swapins=16, swap_used=90), "decode")
    g.sample(reading(6, swapins=16, swap_used=90), "drain")
    g.sample(reading(7, swapins=16, swap_used=90))
    assert g.finish()["untimed_swapins_pages"] == 6
    assert g.finish()["timed_swapins_pages"] == 0


def test_finish_requires_an_observed_drain_interval():
    g = q.PhaseSwapGuard(reading(), native_limit_bytes=50)
    g.sample(reading(2), "decode")
    g.sample(reading(3), "drain")
    with pytest.raises(ValueError, match="did not drain"):
        g.finish()
    final = reading(4, pid=0, rusage_available=False)
    del final["footprint"]
    del final["peak_footprint"]
    g.sample(final)
    receipt = g.finish()
    assert receipt["schema"] == 2
    assert receipt["policy"] == "phase-swap-v2"
    assert receipt["native_pid"] == 12
    assert receipt["observed_drain_ns"] == 100_000_000


@pytest.mark.parametrize("phase", sorted(q.PhaseSwapGuard.phases))
def test_process_identity_is_stable_in_every_phase(phase):
    g = q.PhaseSwapGuard(reading(), native_limit_bytes=50)
    g.sample(reading(2), phase)
    with pytest.raises(ValueError, match="sensor identity changed"):
        g.sample(reading(3, pid=13))
    assert g.failure is not None


def test_exchange_phase_begins_before_decode_and_never_ends_at_abort():
    generation = {
        "id": 1,
        "begin": True,
        "remaining": 256,
        "committed": 0,
        "emitted": 0,
        "pending": 0,
        "remaining_compute": 256,
        "total_budget": 256,
    }
    command = {
        "tokens": [],
        "generate": 64,
        "reset": False,
        "mtp_generation": generation,
    }
    assert q.exchange_phase(command) == "decode"
    assert q.exchange_phase({"tokens": [3], "generate": 0, "reset": True}) == "prefill"
    assert (
        q.exchange_phase(
            {
                "tokens": [],
                "generate": 0,
                "reset": False,
                "mtp_generation": {"id": 1, "abort": True},
            }
        )
        is None
    )
    assert q.exchange_phase({"carrier_warm_state": "inspect"}) is None
    for missing in generation:
        bad = copy.deepcopy(command)
        del bad["mtp_generation"][missing]
        with pytest.raises(ValueError, match="complete v2"):
            q.exchange_phase(bad)
    with pytest.raises(ValueError):
        q.exchange_phase({**command, "tokens": [3]})


def test_correctness_capture_does_not_claim_a_timed_decode_interval():
    assert q.component_phase(correctness_capture=True, timed_cost=False) == "validation"
    assert q.component_phase(correctness_capture=False, timed_cost=True) == "decode"
    for capture, timed in [(True, True), (False, False), (1, False), (False, None)]:
        with pytest.raises(ValueError):
            q.component_phase(correctness_capture=capture, timed_cost=timed)


def test_swapin_cannot_escape_through_decode_end_and_failure_latches():
    g = q.PhaseSwapGuard(reading(), native_limit_bytes=50)
    g.sample(reading(2), "decode")
    with pytest.raises(ValueError, match="timed decode"):
        g.sample(reading(3, swapins=11), "cooling")
    assert g.failure["phase"] == "decode"
    with pytest.raises(ValueError, match="already invalidated"):
        g.sample(reading(4), "startup")
    with pytest.raises(ValueError, match="did not drain"):
        g.finish()


@pytest.mark.parametrize("phase", sorted(q.PhaseSwapGuard.phases))
@pytest.mark.parametrize(
    "changes",
    [
        {"swapouts": 21, "swap_used": 90},
        {"swap_used": 101},
        {"pressure": 2},
        {"thermal": 1},
        {"footprint": 51},
        {"peak_footprint": 51},
    ],
)
def test_original_safety_limits_and_gross_swapouts_apply_in_every_phase(phase, changes):
    g = q.PhaseSwapGuard(reading(), native_limit_bytes=50)
    g.sample(reading(2), phase)
    with pytest.raises(ValueError):
        g.sample(reading(3, **changes))
    assert g.failure is not None


@pytest.mark.parametrize(
    "changes",
    [
        {"swapins": 9},
        {"swapouts": 19},
        {"page_size": 4096},
        {"pressure": True},
        {"swapins": None},
        {"monotonic_raw_ns": 0},
        {"monotonic_raw_ns": 3_000_000_001},
    ],
)
def test_counter_regression_unavailable_sensor_and_sampling_gap_refuse(changes):
    g = q.PhaseSwapGuard(reading(), native_limit_bytes=50)
    with pytest.raises(ValueError):
        g.sample(reading(2, **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"pid": True},
        {"pid": -1},
        {"pid": 12, "rusage_available": None},
        {"pid": 12, "rusage_available": False},
    ],
)
def test_a_declared_native_process_requires_its_memory_sensor(changes):
    g = q.PhaseSwapGuard(reading(), native_limit_bytes=50)
    with pytest.raises(ValueError):
        g.sample(reading(2, **changes))
    assert g.failure is not None


def test_process_memory_sensor_must_supply_both_caps():
    g = q.PhaseSwapGuard(reading(), native_limit_bytes=50)
    g.sample(reading(2, pid=12, rusage_available=True))
    missing = reading(3, pid=12, rusage_available=True)
    del missing["peak_footprint"]
    with pytest.raises(ValueError, match="missing native process"):
        g.sample(missing)


def fixture():
    policy = {"temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0}
    identity = {
        "context": 875,
        **{
            k: "a" * 64
            for k in (
                "execution_digest",
                "history_sha256",
                "worker_sha256",
                "library_sha256",
                "assets_sha256",
            )
        },
        "policy_sha256": q.sampling_digest(policy),
    }
    state = {
        **{
            k: "a" * 64
            for k in (
                "execution_digest",
                "stack_sha256",
                "drafter_sha256",
                "complete_logits_sha256",
            )
        },
        "stack_bytes": 100,
        "drafter_bytes": 200,
        "prefix": 875,
        "drafter_first": 0,
        "drafter_rows": 875,
        "generation_active": False,
        "pending": 0,
        "carried": False,
        "fresh_rng": True,
        "sampling": {
            **policy,
            "valid": True,
            "seed": 44,
            "entropy_words": 0,
            "entropy_prefix": 875,
        },
    }
    packet = {"event": "done", "carrier_warm_state": state}
    restored = {
        "event": "done",
        "checkpoint_ssd_restored": True,
        "checkpoint_restored_tokens": 875,
        "replayed_input_tokens": 0,
        "committed_prefix": 875,
    }
    return identity, packet, restored


def test_warm_reuse_requires_native_restore_and_complete_readback_with_fresh_seed():
    identity, cold, restored = fixture()
    c = q.WarmStateSeal(identity, cold, seed=44)
    warm = copy.deepcopy(cold)
    warm["carrier_warm_state"]["sampling"]["seed"] = 45
    assert c.validate_restore(identity, restored, warm, seed=45)[
        "complete_state_readback"
    ]
    assert (
        q.sampling_digest(
            {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}
        )
        == identity["policy_sha256"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("stack_sha256", "b" * 64),
        ("drafter_sha256", "b" * 64),
        ("complete_logits_sha256", "b" * 64),
        ("execution_digest", "b" * 64),
        ("prefix", 876),
        ("drafter_rows", 874),
        ("drafter_bytes", 201),
        ("generation_active", True),
        ("pending", 1),
        ("carried", True),
        ("fresh_rng", False),
    ],
)
def test_stale_or_partial_warm_state_invalidates_certificate(field, value):
    identity, cold, restored = fixture()
    c = q.WarmStateSeal(identity, cold, seed=44)
    warm = copy.deepcopy(cold)
    warm["carrier_warm_state"][field] = value
    with pytest.raises(ValueError):
        c.validate_restore(identity, restored, warm, seed=44)
    with pytest.raises(ValueError, match="already invalidated"):
        c.validate_restore(identity, restored, cold, seed=44)


@pytest.mark.parametrize(
    "failure",
    [
        "seed",
        "entropy",
        "entropy_prefix",
        "sampling",
        "history",
        "worker",
        "fallback",
        "partial",
        "abort",
    ],
)
def test_warm_identity_rng_ownership_and_fallback_controls(failure):
    identity, cold, restored = fixture()
    c = q.WarmStateSeal(identity, cold, seed=44)
    warm = copy.deepcopy(cold)
    if failure == "seed":
        warm["carrier_warm_state"]["sampling"]["seed"] = 45
    elif failure == "entropy":
        warm["carrier_warm_state"]["sampling"]["entropy_words"] = 1
    elif failure == "entropy_prefix":
        warm["carrier_warm_state"]["sampling"]["entropy_prefix"] = 876
    elif failure == "sampling":
        warm["carrier_warm_state"]["sampling"]["top_p"] = 1
    elif failure == "history":
        identity["history_sha256"] = "b" * 64
    elif failure == "worker":
        identity["worker_sha256"] = "b" * 64
    elif failure == "fallback":
        restored["checkpoint_ssd_restored"] = False
    elif failure == "partial":
        restored["replayed_input_tokens"] = 1
    else:
        c.invalidate()
    with pytest.raises(ValueError):
        c.validate_restore(identity, restored, warm, seed=44)
