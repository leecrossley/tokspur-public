"""Actual target's extracted HC policy checks; CPU-only, no model or Metal."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_model_free_qualification_includes_new_hc_and_split_suites(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "hc_qualification_inventory", ROOT / "scripts/check_owned_gpu_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = ROOT / "build/owned-native"
    commands = module.generated_commands(build, "python")
    for suite in ("hc_q8_gemv", "q8_split_gemv"):
        assert suite in module.METAL_SUITES
        assert "test-" + suite in module.ARTIFACT_NAMES
        assert [
            str(build / ("test-" + suite)),
            str(build / "tokspur-owned.metallib"),
        ] in commands


def test_hc_policy_executes_owned_source(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "hc_protocol_extract", ROOT / "scripts/check_owned_target_protocol.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    extra = r"""
void run_hc() {
    hc_q8_preparation(0, 2, true, 0, true, true, true, true, true);
    for (uint32_t splits : {1, 2, 4, 8, 16, 32}) {
        hc_q8_preparation(splits, 0, false, 1, false, false, false, false, false);
        hc_q8_preparation(splits, 0, false, 3, false, false, false, false, false);
        for (uint32_t flag = 0; flag < 8; ++flag)
            refuses([&] { hc_q8_preparation(splits, flag == 0, flag == 1, flag == 2 ? 2 : 3,
                flag == 3, flag == 4, flag == 5, flag == 6, flag == 7); }, "incompatible HC startup admitted");
        refuses([&] { hc_q8_preparation(splits, 2, false, 3, false, false, false, false, false); },
            "HC recurrent replay admitted");
        check(std::string_view(hc_q8_numerical_tier(splits)) == (splits == 1 ? "split1-exact-block-q8" : "reassociated-fp32"),
            "HC numerical tier mislabelled");
    }
    for (uint32_t splits : {3u, 7u, 9u, 33u, std::numeric_limits<uint32_t>::max()})
        refuses([&] { hc_q8_preparation(splits, 0, false, 3, false, false, false, false, false); }, "invalid HC count admitted");
    check(std::string_view(hc_q8_numerical_tier(0)) == "ordinary", "default HC numerical tier changed");
    // HC is startup-fixed, not a request selector, even if all experiment
    // consents are granted. Both zero and nonzero wire values must refuse.
    for (const auto value : {"0", "1", "8", "32", "null", "true", "\"4\"", "[]", "{}"}) {
        const auto body = std::string(R"({"tokens":[1],"generate":0,"reset":true,"hc_q8_splits":)") + value + "}";
        refuses([&] { request(parse_json(body), 0, 16, 3, true, false, false); }, "HC request override admitted");
    }
    const auto ordinary = request(parse_json(R"({"tokens":[1],"generate":1,"reset":true})"),
        0, 16, 3, false, false, false, true, 0, false, false, 8, false, false);
    check(ordinary.known_prefill && !ordinary.residual_reuse && ordinary.execution == 0 && !ordinary.lookup,
        "fixed HC-compatible known-input request lost its ordinary contract");
    const auto sampled = request(parse_json(R"({"tokens":[1],"generate":1,"reset":true,"sampling":{"temperature":0.7,"top_k":4,"top_p":0.9,"min_p":0.1},"seed":42})"),
        0, 16, 3, false, false, false, true, 0, false, false, 8, false, false);
    check(sampled.sampling.policy.temperature == 0.7 && sampled.sampling.seed == 42,
        "HC-compatible fixed execution must not silently alter stochastic sampling");
}
"""
    unit = (
        '#include "tokspur/json.hpp"\n#include "tokspur/gguf.hpp"\n'
        '#include "tokspur/sampling.hpp"\n#include "tokspur/prompt_lookup.hpp"\n'
        "#include <algorithm>\n#include <array>\n#include <charconv>\n#include <cmath>\n#include <limits>\n"
        "#include <iostream>\n#include <optional>\n#include <set>\n#include <type_traits>\n#include <utility>\n"
        "namespace tokspur { namespace {\n"
        + protocol.extract((ROOT / "native/src/target.cpp").read_text())
        + protocol.CHECKS
        + extra
        + "\n} }\nint main() { try { tokspur::run(); tokspur::run_stack_experiments(); tokspur::run_hc(); return 0; } "
        + "catch (const std::exception& error) { std::cerr << error.what() << '\\n'; return 1; } }\n"
    )
    path, executable = tmp_path / "hc protocol.cpp", tmp_path / "hc protocol"
    path.write_text(unit)
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I" + str(ROOT / "native/include"),
            str(path),
            *[
                str(ROOT / ("native/src/" + name + ".cpp"))
                for name in ("json", "unicode", "sampling", "prompt_lookup")
            ],
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
