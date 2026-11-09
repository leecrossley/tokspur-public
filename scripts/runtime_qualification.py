"""Model-free policy for private qualification sensors and validated warm reuse.

Callers own sampling, process lifetime, cooling, traces and resource budgets.
These checks never change runtime defaults or retry a failed observation.
"""

import hashlib
import json
import math


def need(condition, message):
    if not condition:
        raise ValueError(message)


def seal(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sampling_digest(policy):
    need(
        type(policy.get("top_k")) is int and 0 <= policy["top_k"] <= 248320,
        "invalid warm sampling policy",
    )
    normal = {"top_k": policy["top_k"]}
    for key in ("temperature", "top_p", "min_p"):
        value = policy.get(key)
        need(
            type(value) in (int, float) and math.isfinite(value),
            "invalid warm sampling policy",
        )
        normal[key] = float(value)
    need(
        normal["temperature"] >= 0
        and 0 < normal["top_p"] <= 1
        and 0 <= normal["min_p"] <= 1,
        "invalid warm sampling policy",
    )
    return seal(normal)


def exchange_phase(command):
    """Classify work before an owned recorder forwards the exact host bytes.

    Decode remains active through Rust publication until the HTTP driver closes
    the phase. An abort or an empty control packet cannot end it early.
    """
    if "generate" not in command:
        return None
    count, tokens = command.get("generate"), command.get("tokens")
    need(
        type(count) is int and 0 <= count <= 4096 and isinstance(tokens, list),
        "invalid phase command",
    )
    if count:
        generation = command.get("mtp_generation")
        need(
            isinstance(generation, dict)
            and not tokens
            and command.get("reset") is False
            and {
                "id",
                "begin",
                "remaining",
                "committed",
                "emitted",
                "pending",
                "remaining_compute",
                "total_budget",
            }
            <= generation.keys()
            and not generation.get("abort"),
            "decode phase requires a complete v2 command",
        )
        return "decode"
    return "prefill" if tokens else None


def component_phase(*, correctness_capture, timed_cost):
    """Intrusive correctness output has no qualified timing interval."""
    need(
        type(correctness_capture) is bool
        and type(timed_cost) is bool
        and correctness_capture != timed_cost,
        "component must declare correctness capture or timed cost",
    )
    return "validation" if correctness_capture else "decode"


class PhaseSwapGuard:
    """Samples and transitions must be serialised by the owning monitor.

    A boundary sample closes the OLD phase before entering the new one. Thus
    an untimed swap-in cannot leak into decode, nor a decode swap-in escape
    into cooling. Failures latch for the whole invocation.
    """

    phases = frozenset(
        ("startup", "prefill", "restore", "validation", "cooling", "decode", "drain")
    )
    counters = (
        "monotonic_raw_ns",
        "swapins",
        "swapouts",
        "swap_used",
        "page_size",
        "pressure",
        "thermal",
    )

    def __init__(self, baseline, *, native_limit_bytes, maximum_gap_ns=2_000_000_000):
        need(
            type(native_limit_bytes) is int and native_limit_bytes > 0,
            "invalid native memory limit",
        )
        need(
            type(maximum_gap_ns) is int and maximum_gap_ns > 0,
            "invalid sensor interval bound",
        )
        self.native_limit = native_limit_bytes
        self.maximum_gap = maximum_gap_ns
        self.phase = "startup"
        self.failure = None
        self.last = None
        self.baseline = None
        self.intervals = []
        self.native_pid = None
        self.sample(baseline)

    def sample(self, reading, next_phase=None):
        need(self.failure is None, "qualification guard already invalidated")
        try:
            need(isinstance(reading, dict), "invalid or missing VM sensor")
            need(
                next_phase is None or next_phase in self.phases,
                "unknown qualification phase",
            )
            need(
                all(
                    type(reading.get(k)) is int and reading[k] >= 0
                    for k in self.counters
                ),
                "invalid or missing VM sensor",
            )
            need(reading["page_size"] in (4096, 16384), "invalid VM page size")
            need(
                reading["pressure"] == 1 and reading["thermal"] == 0,
                "pressure or thermal guard failed",
            )
            if self.phase == "decode" or next_phase == "decode":
                need(
                    type(reading.get("pid")) is int
                    and reading["pid"] > 0
                    and reading.get("rusage_available") is True
                    and all(
                        name in reading for name in ("footprint", "peak_footprint")
                    ),
                    "missing native process memory sensor",
                )
                if self.native_pid is None:
                    self.native_pid = reading["pid"]
                need(
                    reading["pid"] == self.native_pid,
                    "native process sensor identity changed",
                )
            if "pid" in reading:
                need(
                    type(reading["pid"]) is int and reading["pid"] >= 0,
                    "invalid process sensor identity",
                )
                if reading["pid"]:
                    need(
                        reading.get("rusage_available") is True
                        and all(
                            name in reading for name in ("footprint", "peak_footprint")
                        ),
                        "missing native process memory sensor",
                    )
                    if self.native_pid is None:
                        self.native_pid = reading["pid"]
                    need(
                        reading["pid"] == self.native_pid,
                        "native process sensor identity changed",
                    )
            for name in ("footprint", "peak_footprint"):
                if name in reading:
                    need(
                        type(reading[name]) is int
                        and 0 <= reading[name] <= self.native_limit,
                        "native memory limit exceeded",
                    )
            if self.baseline is None:
                self.baseline = dict(reading)
            need(
                reading["swap_used"] <= self.baseline["swap_used"],
                "swap occupancy grew",
            )
            need(
                reading["swapouts"] == self.baseline["swapouts"],
                "new swap-outs during qualification",
            )
            interval = {
                "phase": self.phase,
                "swapins_pages": 0,
                "swapouts_pages": 0,
                "elapsed_ns": 0,
            }
            if self.last is not None:
                delta = reading["monotonic_raw_ns"] - self.last["monotonic_raw_ns"]
                need(0 < delta <= self.maximum_gap, "VM sensor gap or clock regression")
                need(
                    reading["page_size"] == self.last["page_size"],
                    "VM page size changed",
                )
                for name in ("swapins", "swapouts"):
                    need(reading[name] >= self.last[name], "VM counter regression")
                    interval[name + "_pages"] = reading[name] - self.last[name]
                interval["elapsed_ns"] = delta
                need(
                    self.phase != "decode" or interval["swapins_pages"] == 0,
                    "swap-in during timed decode",
                )
                self.intervals.append(interval)
            self.last = dict(reading)
            if next_phase is not None:
                self.phase = next_phase
            return {**interval, "next_phase": self.phase, "valid": True}
        except (ValueError, TypeError, KeyError) as error:
            self.failure = {"phase": self.phase, "reason": str(error)}
            raise

    def finish(self):
        drain_ns = sum(v["elapsed_ns"] for v in self.intervals if v["phase"] == "drain")
        need(
            self.failure is None
            and self.phase == "drain"
            and self.native_pid is not None
            and drain_ns > 0
            and any(
                v["phase"] == "decode" and v["elapsed_ns"] > 0 for v in self.intervals
            ),
            "qualification monitor did not drain validly",
        )
        return {
            "schema": 2,
            "valid": True,
            "policy": "phase-swap-v2",
            "native_pid": self.native_pid,
            "observed_drain_ns": drain_ns,
            "swap_occupancy_growth_allowed": False,
            "swapouts_allowed": False,
            "decode_swapins_allowed": False,
            "untimed_swapins_pages": sum(
                v["swapins_pages"] for v in self.intervals if v["phase"] != "decode"
            ),
            "timed_swapins_pages": 0,
            "maximum_gap_ns": self.maximum_gap,
        }


class WarmStateSeal:
    """One private prompt/state identity; native restore still verifies every chunk.

    The native readback fingerprint covers target and drafter state and complete
    logits. RNG and generation checks stay live on every reuse. Certificates
    are harness evidence, never model/checkpoint identity or runtime state.
    """

    state_fields = (
        "execution_digest",
        "stack_sha256",
        "drafter_sha256",
        "complete_logits_sha256",
        "stack_bytes",
        "drafter_bytes",
        "prefix",
        "drafter_first",
        "drafter_rows",
    )

    def __init__(self, identity, cold_packet, *, seed):
        self.invalid = False
        self.identity = dict(identity)
        need(
            type(identity.get("context")) is int and 0 < identity["context"] <= 65536,
            "invalid warm context",
        )
        for name in (
            "execution_digest",
            "history_sha256",
            "worker_sha256",
            "library_sha256",
            "assets_sha256",
            "policy_sha256",
        ):
            self._hash(identity.get(name))
        state = self._state(cold_packet, seed)
        need(
            state["execution_digest"] == identity["execution_digest"]
            and state["prefix"] == identity["context"],
            "warm model identity differs",
        )
        need(
            sampling_digest(state["sampling"]) == identity["policy_sha256"],
            "warm sampling policy changed",
        )
        self.state = {k: state[k] for k in self.state_fields}
        self.identity_digest = seal(self.identity)
        self.state_digest = seal(self.state)

    @staticmethod
    def _hash(value):
        need(
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            "invalid warm digest",
        )

    @classmethod
    def _state(cls, packet, seed):
        need(
            packet.get("event") == "done",
            "warm state requires completed native inspection",
        )
        state = packet.get("carrier_warm_state")
        need(isinstance(state, dict), "missing native warm-state fingerprint")
        for name in cls.state_fields[:4]:
            cls._hash(state.get(name))
        for name in cls.state_fields[4:]:
            need(
                type(state.get(name)) is int and state[name] >= 0,
                "invalid warm shape or frontier",
            )
        need(
            state["stack_bytes"] > 0
            and state["drafter_bytes"] > 0
            and state["drafter_rows"] > 0,
            "empty warm payload",
        )
        need(
            state.get("generation_active") is False
            and type(state.get("pending")) is int
            and state["pending"] == 0
            and state.get("carried") is False,
            "warm state retains generation output",
        )
        sampling = state.get("sampling", {})
        need(
            type(seed) is int
            and 0 <= seed < 1 << 64
            and sampling.get("valid") is True
            and type(sampling.get("seed")) is int
            and sampling["seed"] == seed
            and type(sampling.get("entropy_words")) is int
            and sampling["entropy_words"] == 0
            and sampling.get("entropy_prefix") == state["prefix"]
            and state.get("fresh_rng") is True,
            "warm RNG or sampling frontier differs",
        )
        return state

    def validate_restore(self, identity, restore_packet, inspection_packet, *, seed):
        need(not self.invalid, "warm reuse certificate already invalidated")
        try:
            need(
                seal(identity) == self.identity_digest,
                "warm inputs or execution policy changed",
            )
            need(
                restore_packet.get("event") == "done"
                and restore_packet.get("checkpoint_ssd_restored") is True
                and restore_packet.get("checkpoint_restored_tokens")
                == self.state["prefix"]
                and restore_packet.get("replayed_input_tokens") == 0
                and restore_packet.get("committed_prefix") == self.state["prefix"],
                "warm restore missed or replayed input",
            )
            state = self._state(inspection_packet, seed)
            need(
                {k: state[k] for k in self.state_fields} == self.state,
                "warm readback state or complete logits differ",
            )
            sampling = state["sampling"]
            need(
                sampling_digest(sampling) == self.identity["policy_sha256"],
                "warm sampling policy changed",
            )
            return {
                "schema": 1,
                "identity_sha256": self.identity_digest,
                "state_sha256": self.state_digest,
                "seed": seed,
                "native_chunk_checks": True,
                "complete_state_readback": True,
                "valid": True,
            }
        except (ValueError, TypeError, KeyError):
            self.invalid = True
            raise

    def invalidate(self):
        self.invalid = True
