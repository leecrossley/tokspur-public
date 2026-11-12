"""Model-free admission and actual terminal metadata checks for verify batching.

Metal parity/dispatch coverage lives in test_moe_mm_many_gpu.cpp; these checks
neither execute GPU work nor establish model or performance qualification.
"""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLAG = "TOKSPUR_PRIVATE_VERIFY_NAX_BATCH_TOP10_MIXTURE"
KERNEL_FLAGS = [
    "-DTOKSPUR_PRIVATE_PREFILL_BATCH_TOP10=1",
    "-DTOKSPUR_PRIVATE_PREFILL_BATCH_MIXTURE=1",
]


def compile_source(tmp_path, code, defines, *, link=False):
    source = tmp_path / "verify_batch.cpp"
    source.write_text(code)
    executable = tmp_path / "verify_batch"
    argv = [
        "c++",
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(ROOT / "native/include"),
        *defines,
        str(source),
    ]
    if link:
        argv += [
            str(ROOT / "native/src/json.cpp"),
            str(ROOT / "native/src/unicode.cpp"),
            "-o",
            str(executable),
        ]
    else:
        argv += ["-fsyntax-only"]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    return result, executable


@pytest.mark.parametrize("policy", [None, "0", "1"])
def test_immutable_selection_and_terminal_metadata(tmp_path, policy):
    selected = policy == "1"
    defines = ["-DTOKSPUR_PRIVATE_GROUPED_MM=1", *KERNEL_FLAGS]
    if policy is not None:
        defines.append(f"-D{FLAG}={policy}")
    implementation = (ROOT / "native/src/target.cpp").read_text()
    start = implementation.index("Json verify_nax_batch_metadata(")
    end = implementation.index("\n#endif\nJson cache_memory_metadata(", start)
    formatter = implementation[start:end].replace(
        "const MetalGraph& gpu", "const SnapshotGraph& gpu"
    )
    code = r"""
#include "tokspur/verify_nax.hpp"
#include "tokspur/json.hpp"
#include <iostream>
using namespace tokspur;
constexpr MoeShape shape{2560, 640, 512, 10};
static_assert(verify_nax_batch_top10_mixture_enabled == EXPECTED);
static_assert(verify_nax_batch_top10_supported(shape, 2) == EXPECTED);
static_assert(verify_nax_batch_top10_supported(shape, 8) == EXPECTED);
static_assert(verify_nax_batch_mixture_supported(shape, 2) == EXPECTED);
static_assert(verify_nax_batch_mixture_supported(shape, 8) == EXPECTED);
static_assert(!verify_nax_batch_top10_supported(shape, 1));
static_assert(!verify_nax_batch_top10_supported(shape, 9));
static_assert(!verify_nax_batch_mixture_supported(shape, 1));
static_assert(!verify_nax_batch_mixture_supported(shape, 9));
static_assert(!verify_nax_batch_top10_supported({2560, 640, 513, 10}, 8));
static_assert(!verify_nax_batch_top10_supported({2560, 640, 0, 10}, 8));
static_assert(!verify_nax_batch_top10_supported({2560, 640, 512, 65}, 8));
static_assert(!verify_nax_batch_top10_supported({2560, 640, 4, 10}, 8));
static_assert(!verify_nax_batch_mixture_supported({2592, 640, 512, 10}, 8));
static_assert(!verify_nax_batch_mixture_supported({63, 640, 512, 10}, 8));
static_assert(!verify_nax_batch_mixture_supported({2560, 640, 512, 0}, 8));
static_assert(!verify_nax_batch_mixture_supported({2560, 640, 512, 65}, 8));
struct SnapshotGraph {
    mutable unsigned calls = 0;
    VerifyNaxTop10MixtureTelemetry verify_nax_top10_mixture_telemetry() const {
        ++calls; return {3, 24, 5, 4, 32, 6};
    }
};
""".replace("EXPECTED", "true" if selected else "false")
    code += (
        formatter
        + r"""
int main() {
    SnapshotGraph graph;
    for (const auto* event : {"ready", "done", "token", "error"})
        std::cout << verify_nax_batch_metadata(Json(Json::Object{
            {"event", Json(event)}, {"existing", Json::integer(7)}}), graph).dump() << '\n';
    std::cout << verify_nax_batch_metadata(Json(Json::Object{}), graph).dump() << '\n';
    std::cout << graph.calls << '\n';
}
"""
    )
    compiled, executable = compile_source(tmp_path, code, defines, link=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=True
    )
    ready, done, token, error, missing, calls = map(
        json.loads, result.stdout.splitlines()
    )
    expected = {
        "existing": 7,
        "native_verify_nax_batch_top10_mixture_enabled": selected,
        "native_verify_nax_ops_counter_scope": (
            "worker-lifetime completed verification operator encodings; includes subsequently "
            "aborted submissions; excludes prefill and drafter"
        ),
        "native_verify_top10_batched_calls": 3,
        "native_verify_top10_batched_rows": 24,
        "native_verify_top10_scalar_calls": 5,
        "native_verify_mixture_batched_calls": 4,
        "native_verify_mixture_batched_rows": 32,
        "native_verify_mixture_scalar_calls": 6,
    }
    assert ready == {"event": "ready", **expected}
    assert done == {"event": "done", **expected}
    assert token == {"event": "token", "existing": 7}
    assert error == {"event": "error", "existing": 7}
    assert missing == {}
    assert calls == 2


@pytest.mark.parametrize("policy", ["2", "-1", "invalid"])
def test_invalid_direct_cpp_policy_refused(tmp_path, policy):
    result, _ = compile_source(
        tmp_path,
        '#include "tokspur/graph.hpp"\n',
        KERNEL_FLAGS + [f"-D{FLAG}={policy}"],
    )
    assert result.returncode != 0


@pytest.mark.parametrize("kernel_flags", [[], KERNEL_FLAGS[:1], KERNEL_FLAGS[1:]])
def test_direct_cpp_requires_both_existing_kernels(tmp_path, kernel_flags):
    result, _ = compile_source(
        tmp_path,
        '#include "tokspur/graph.hpp"\n',
        [f"-D{FLAG}=1", *kernel_flags],
    )
    assert result.returncode != 0
    assert "requires both existing batch kernels" in result.stderr
