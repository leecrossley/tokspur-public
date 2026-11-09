"""Private depth-eight caller bookkeeping, extracted from owned target source.

CPU fakes only: these do not validate native verification, GPU state or speed.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = r"""
struct EightVerifiedFake {
    uint32_t accepted, position, evaluated, head_dispatches;
    std::span<const float> logits;
    std::array<float, 8> source_logits{};
};
struct EightFixture {
    TargetLookup<FakeResult> lookup;
    CommittedResult<FakeResult> retained;
    TargetSampler sampler;
    uint32_t prefix = 0, wanted = 7, verifies = 0, ordinary_steps = 0, tracked = 0;
    bool fail_verify = false, fail_track = false, wrong_prefix = false;
    std::vector<float> verified_logits = std::vector<float>(16, 0);
    explicit EightFixture(uint32_t proposals = 7)
        : lookup(16, 2, 2, 8ull << 20, proposals) {
        std::vector<uint32_t> prompt(16, 15);
        for (uint32_t i = 0; i < 7; ++i) prompt.push_back(i);
        prompt.insert(prompt.end(), 16, 15);
        prefix = uint32_t(prompt.size()); lookup.committed(prefix, prompt);
        std::vector<float> base(16, 0); base[0] = 100;
        retained.commit(FakeResult{std::move(base)}, prefix);
        verified_logits[8] = 90;
        for (uint32_t i = 1; i <= prefix; ++i)
            require(sampler.committed_input(i), "fake committed prompt entropy");
    }
    LookupRound run(uint32_t remaining = 8, std::span<const uint32_t> stops = {}) {
        return lookup_round(lookup, retained, sampler, remaining, stops,
            [&] { return prefix; },
            [&](std::span<const uint32_t> ids, uint32_t base, std::span<const float> logits,
                uint32_t reservation, std::span<const uint32_t> terminal) {
                ++verifies;
                check(ids.size() == lookup.proposal_tokens() && logits[0] == 100,
                    "actual caller changed proposal rows or retained authority");
                if (fail_verify) throw FormatError("before fake native publication");
                auto active = std::min(uint32_t(ids.size()), reservation);
                for (uint32_t i = 0; i < active; ++i)
                    if (std::find(terminal.begin(), terminal.end(), ids[i]) != terminal.end()) {
                        active = i + 1; break;
                    }
                const auto accepted = std::min(wanted, active);
                prefix += accepted + uint32_t(wrong_prefix);
                uint32_t head = 0;
                for (auto n = active; n; n >>= 1) head += n & 1;
                EightVerifiedFake result{accepted, base, active, head, verified_logits};
                for (uint32_t i = 0; i < 8; ++i) result.source_logits[i] = 100.0f + i;
                return result;
            },
            [&] {
                ++ordinary_steps;
                const auto& logits = retained.get(prefix).logits;
                const auto id = TargetLookup<FakeResult>::greedy(logits);
                const auto score = logits[id];
                ++prefix;
                std::vector<float> next(16, 0); next[9] = 91;
                retained.commit(FakeResult{std::move(next)}, prefix);
                require(sampler.committed_input(prefix), "fake ordinary entropy");
                lookup.committed(prefix, std::span(&id, 1)); ++tracked;
                return LookupRoundToken{id, score, false};
            },
            [&](uint32_t, std::span<const uint32_t> ids) {
                tracked += uint32_t(ids.size());
                if (fail_track) throw FormatError("after fake native publication");
            });
    }
};
void run_eight() {
    for (const auto limit : {2u, 7u}) {
        EightFixture fixture(limit);
        check(fixture.lookup.proposal_tokens() == limit &&
            fixture.lookup.propose(fixture.prefix).count == limit &&
            fixture.lookup.rows.size() == 2 * limit &&
            fixture.lookup.embeddings.size() == 2 * limit,
            "private proposal and prepared caller-row limits");
    }
    for (const auto bad : {0u, 1u, 3u, 6u, 8u})
        refuses([&] { EightFixture fixture(bad); }, "unregistered private proposal limit");
    for (uint32_t k = 0; k <= 7; ++k) {
        EightFixture fixture; fixture.wanted = k;
        const auto base = fixture.prefix;
        const auto words = fixture.sampler.random().words_consumed();
        const auto round = fixture.run();
        const auto ordinary = uint32_t(k < 7);
        check(round.accepted == k && round.evaluated == 7 && round.head_dispatches == 3 &&
            round.proposed == 7 && round.count == k + ordinary &&
            fixture.ordinary_steps == ordinary && fixture.tracked == round.count &&
            fixture.prefix == base + round.count && fixture.lookup.matches(fixture.prefix) &&
            fixture.retained.matches(fixture.prefix) && fixture.sampler.matches(fixture.prefix) &&
            fixture.sampler.random().words_consumed() == words,
            "accepted0..7 bookkeeping or entropy changed");
        for (uint32_t i = 0; i < k; ++i)
            check(round.tokens[i].id == i && round.tokens[i].verified &&
                round.tokens[i].source_logit == 100.0f + i,
                "accepted source score or chronological token changed");
        if (ordinary) check(round.tokens[k].id == (k ? 8u : 0u) &&
            !round.tokens[k].verified, "correction did not evaluate authoritative retained logits");
    }
    for (uint32_t remaining = 1; remaining <= 7; ++remaining) {
        EightFixture fixture;
        const auto round = fixture.run(remaining);
        check(round.count == remaining && fixture.verifies == uint32_t(remaining >= 2) &&
            round.accepted == (remaining >= 2 ? remaining : 0) &&
            fixture.ordinary_steps == uint32_t(remaining == 1),
            "remaining reservation changed verification or ordinary fallback");
    }
    for (uint32_t terminal = 0; terminal < 7; ++terminal) {
        EightFixture fixture;
        const std::array<uint32_t, 1> stops{terminal};
        const auto round = fixture.run(8, stops);
        check(round.count == terminal + 1 &&
            round.accepted == (terminal ? terminal + 1 : 0) &&
            fixture.ordinary_steps == uint32_t(terminal == 0) &&
            fixture.retained.stop_id() == terminal && !fixture.retained.continuable(fixture.prefix),
            "matching terminal gained a correction or committed beyond its prefix");
    }
    EightFixture zero;
    refuses([&] { zero.run(0); }, "zero reservation reached model work");
    check(!zero.verifies && !zero.ordinary_steps, "zero reservation executed a step");
    EightFixture stale;
    auto proposal = stale.lookup.propose(stale.prefix);
    stale.lookup.restore(stale.prefix);
    refuses([&] { static_cast<void>(stale.lookup.eligible(proposal, stale.prefix,
        stale.retained.get(stale.prefix).logits, 8, {})); }, "stale eight-row proposal survived epoch renewal");
    EightFixture tied;
    tied.retained.commit(FakeResult{std::vector<float>(16, 100)}, tied.prefix);
    check(TargetLookup<FakeResult>::greedy(tied.retained.get(tied.prefix).logits) == 0,
        "greedy ties changed lowest-ID policy");
    for (uint32_t fault = 0; fault < 3; ++fault) {
        EightFixture fixture;
        fixture.fail_verify = fault == 0; fixture.fail_track = fault == 1;
        fixture.wrong_prefix = fault == 2;
        refuses([&] { fixture.run(); }, "verification/publication fault was hidden");
        check(!fixture.lookup.matches(fixture.prefix) && !fixture.retained.matches(fixture.prefix) &&
            !fixture.sampler.valid() && !fixture.ordinary_steps,
            "fault retained reusable identity or implicitly retried generation");
    }
    EightFixture nonfinite;
    nonfinite.verified_logits[0] = std::numeric_limits<float>::infinity();
    refuses([&] { nonfinite.run(); }, "nonfinite verified logits were published");
    EightFixture stochastic;
    stochastic.sampler = TargetSampler({{0.5, 0, 1, 0}, 7});
    for (uint32_t i = 1; i <= stochastic.prefix; ++i)
        require(stochastic.sampler.committed_input(i), "fake stochastic prompt entropy");
    refuses([&] { stochastic.run(); }, "sampled policy reached greedy eight-row verifier");
}
"""


def test_private_eight_round_uses_extracted_owned_bookkeeping(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "eight_protocol", ROOT / "scripts/check_owned_target_protocol.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    unit = (
        '#include "tokspur/json.hpp"\n#include "tokspur/gguf.hpp"\n'
        '#include "tokspur/sampling.hpp"\n#include "tokspur/prompt_lookup.hpp"\n'
        "#include <algorithm>\n#include <array>\n#include <charconv>\n#include <cmath>\n"
        "#include <limits>\n#include <iostream>\n#include <optional>\n#include <set>\n"
        "#include <type_traits>\n#include <utility>\nnamespace tokspur { namespace {\n"
        + protocol.extract((ROOT / "native/src/target.cpp").read_text())
        + protocol.CHECKS
        + CHECKS
        + "\n} }\nint main() { try { tokspur::run(); tokspur::run_stack_experiments(); tokspur::run_eight(); }"
        + "catch (const std::exception& error) { std::cerr << error.what(); return 1; } }\n"
    )
    source, executable = tmp_path / "eight.cpp", tmp_path / "eight"
    source.write_text(unit)
    built = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-DTOKSPUR_PRIVATE_VERIFY_EIGHT",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-pthread",
            "-I" + str(ROOT / "native/include"),
            str(source),
            *[
                str(ROOT / f"native/src/{name}.cpp")
                for name in ("json", "unicode", "sampling", "prompt_lookup")
            ],
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert built.returncode == 0, built.stderr
    subprocess.run([str(executable)], check=True, timeout=60)
