"""CPU checks of the actual READY/DONE formatter and compile-time cache policy.

The graph is a controlled snapshot provider; this does not run Metal or establish
operator parity. Allocation accounting itself is exercised by test_memory.cpp.
"""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = r"""
#include "tokspur/json.hpp"
#include "tokspur/memory.hpp"
#include <iostream>
using namespace tokspur;
constexpr uint64_t cache_budget = 536870912;
struct MetalGraph {
    Q8Tile64CacheStats stats;
    mutable unsigned calls = 0;
    Q8Tile64CacheStats q8_tile64_cache_stats() const { ++calls; return stats; }
};
"""

CHECKS = r"""
int main() {
    MetalGraph graph;
    graph.stats.enabled = q8_tile64_cache_enabled;
    graph.stats.registered_candidates = 3;
    graph.stats.registered_source_bytes = 9792;
    graph.stats.candidate_repack_logical_bytes = 10880;
    graph.stats.covered_candidates = q8_tile64_cache_enabled ? 2 : 0;
    graph.stats.omitted_candidates = q8_tile64_cache_enabled ? 1 : 3;
    graph.stats.omitted_repack_logical_bytes = q8_tile64_cache_enabled ? 4352 : 10880;
    graph.stats.owning_allocations = q8_tile64_cache_enabled ? 1 : 0;
    graph.stats.owned_logical_bytes = q8_tile64_cache_enabled ? 4352 : 0;
    graph.stats.owned_charged_bytes = q8_tile64_cache_enabled ? 16384 : 0;
    graph.stats.aliases = q8_tile64_cache_enabled ? 2 : 0;
    for (const auto* event : {"ready", "done", "token"}) {
        std::cout << cache_memory_metadata(Json(Json::Object{
            {"event", Json(event)}, {"existing", Json::integer(17)}}), graph).dump() << '\n';
    }
    std::cout << cache_memory_metadata(Json(Json::Object{}), graph).dump() << '\n';
    std::cout << graph.calls << '\n';
}
"""


@pytest.mark.parametrize("policy", [None, "1", "0"])
def test_ready_done_actual_cache_fields_and_snapshot_scope(tmp_path, policy):
    implementation = (ROOT / "native/src/target.cpp").read_text()
    start = implementation.index("Json cache_memory_metadata(")
    end = implementation.index("\n#ifdef TOKSPUR_PRIVATE_PLE_GATE_COMPAT", start)
    source = tmp_path / "cache_metadata.cpp"
    executable = tmp_path / "cache_metadata"
    source.write_text(PRELUDE + implementation[start:end] + CHECKS)
    argv = ["c++", "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror"]
    if policy is not None:
        argv.append("-DTOKSPUR_PRIVATE_Q8_TILE64_CACHE=" + policy)
    compiled = subprocess.run(
        argv
        + [
            "-I",
            str(ROOT / "native/include"),
            str(source),
            str(ROOT / "native/src/json.cpp"),
            str(ROOT / "native/src/unicode.cpp"),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=True
    )
    ready, done, token, missing, calls = [
        json.loads(x) for x in result.stdout.splitlines()
    ]
    enabled = policy != "0"
    expected = {
        "existing": 17,
        "host_row_cache_budget_bytes": 536870912,
        "q8_tile64_cache_enabled": enabled,
        "q8_tile64_registered_candidates": 3,
        "q8_tile64_registered_source_bytes": 9792,
        "q8_tile64_candidate_repack_logical_bytes": 10880,
        "q8_tile64_covered_candidates": 2 if enabled else 0,
        "q8_tile64_omitted_candidates": 1 if enabled else 3,
        "q8_tile64_omitted_repack_logical_bytes": 4352 if enabled else 10880,
        "q8_tile64_owning_allocations": 1 if enabled else 0,
        "q8_tile64_owned_logical_bytes": 4352 if enabled else 0,
        "q8_tile64_owned_charged_bytes": 16384 if enabled else 0,
        "q8_tile64_aliases": 2 if enabled else 0,
        "q8_tile64_candidate_scope": (
            "raw registered matrices; sums may overlap; omissions checked per source key; "
            "allocation bytes count distinct live owners only"
        ),
    }
    assert ready == {"event": "ready", **expected}
    assert done == {"event": "done", **expected}
    assert token == {"event": "token", "existing": 17}
    assert missing == {}
    assert calls == 2
