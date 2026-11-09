"""Authored CPU/extracted PLD caller coverage; no model or Metal fixtures."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKS = r"""
struct VerifiedFake {
    uint32_t accepted, position;
    std::span<const float> logits;
    std::array<float, 2> source_logits{10, 20};
};
struct LookupFixture {
    TargetLookup<FakeResult> lookup{4, 2, 2, 8ull << 20};
    CommittedResult<FakeResult> retained;
    TargetSampler sampler;
    uint32_t prefix = 0, accept = 2, verifies = 0, evaluated = 0, tracked = 0;
    bool fail_verify = false, fail_track = false, wrong_prefix = false;
    std::vector<float> verified_logits{0, 9, 0, 0};
    LookupFixture(bool second_stop = false) {
        std::vector<uint32_t> prompt(18, 0);
        if (second_stop) { prompt.assign(16, 2); prompt.push_back(0); prompt.push_back(1); prompt.insert(prompt.end(), 16, 2); }
        prefix = uint32_t(prompt.size()); lookup.committed(prefix, prompt);
        retained.commit(FakeResult{{10, 0, 0, 0}}, prefix);
        for (uint32_t i = 1; i <= prefix; ++i) require(sampler.committed_input(i), "fake prompt entropy");
    }
    LookupRound run(uint32_t remaining = 2, std::span<const uint32_t> stops = {}) {
        return lookup_round(lookup, retained, sampler, remaining, stops, [&] { return prefix; },
            [&](std::span<const uint32_t> tokens, uint32_t base, std::span<const float> logits, uint32_t budget, std::span<const uint32_t>) {
                ++verifies; check(tokens.size() == 2 && logits[0] == 10 && budget >= 2, "fake verifier received wrong base");
                if (fail_verify) throw FormatError("before native publication");
                prefix += accept + uint32_t(wrong_prefix);
                return VerifiedFake{accept, base, verified_logits};
            },
            [&] {
                ++evaluated; const auto& logits = retained.get(prefix).logits;
                const auto id = TargetLookup<FakeResult>::greedy(logits); const auto score = logits[id];
                ++prefix; retained.commit(FakeResult{{0, 0, 0, 11}}, prefix);
                require(sampler.committed_input(prefix), "fake correction entropy");
                lookup.committed(prefix, std::span(&id, 1)); ++tracked;
                return LookupRoundToken{id, score, false};
            },
            [&](uint32_t, std::span<const uint32_t> tokens) {
                tracked += uint32_t(tokens.size()); if (fail_track) throw FormatError("after native publication");
            });
    }
};
void run_lookup() {
    lookup_preparation(false, 0, 2, true, 0, 0, true, true, true, true);
    lookup_preparation(true, 128ull << 20, 0, false, 16, 3, false, false, false, false);
    for (uint32_t flag = 0; flag < 8; ++flag)
        refuses([&] { lookup_preparation(true, 128ull << 20, flag == 0, flag == 1,
            flag == 2 ? 0 : 16, flag == 3 ? 1 : 3, flag == 4, flag == 5, flag == 6, flag == 7); }, "unsafe lookup startup");
    refuses([] { lookup_preparation(true, 0, 0, false, 16, 3, false, false, false, false); }, "unpaired lookup startup");
    refuses([] { lookup_preparation(false, 1, 0, false, 16, 3, false, false, false, false); }, "unexpected lookup budget");
    refuses([] { lookup_preparation(true, (128ull << 20) + 1, 0, false, 16, 3, false, false, false, false); }, "oversized lookup budget");
    refuses([] { lookup_preparation(false, 0, 0, false, 16, 3, false, false, false, false, true); }, "unprepared selector consent");
    const auto select = [](std::string_view body, bool prepared = true, bool consent = true, bool current = true) {
        return request(parse_json(body), 7, 16, 3, false, true, true, true,
            0, false, false, 0, false, false, 0, prepared, consent, current);
    };
    const auto off = R"({"tokens":[1],"generate":2,"reset":true,"prompt_lookup":false})";
    const auto on = R"({"tokens":[1],"generate":2,"reset":true,"prompt_lookup":true})";
    check(!select(off).lookup && select(on, true, true, false).lookup, "reset-only resident lookup selector");
    check(select(R"({"tokens":[1],"generate":0,"reset":false})").lookup, "append did not inherit active mode");
    check(!select(R"({"tokens":[1],"generate":0,"reset":true})", false, false, false).lookup, "ordinary default silently enabled lookup");
    refuses([&] { select(off, true, false); }, "selector bypassed startup consent");
    refuses([&] { select(off, false, false, false); }, "selector bypassed startup capability");
    for (const auto value : {"true", "false", "0", "null", "\"false\""}) {
        const auto body = std::string(R"({"tokens":[1],"generate":0,"reset":false,"prompt_lookup":)") + value + "}";
        refuses([&] { select(body); }, "append changed lookup mode");
    }
    for (const auto value : {"0", "1", "null", "\"true\""}) {
        const auto body = std::string(R"({"tokens":[1],"generate":0,"reset":true,"prompt_lookup":)") + value + "}";
        refuses([&] { select(body); }, "lookup selector was not exact boolean");
    }
    LookupFixture disabled; const auto before_disable = disabled.lookup.ledger(); disabled.lookup.invalidate();
    check(disabled.lookup.ledger() == before_disable, "off reset paid hidden history maintenance");
    disabled.lookup.reset(); const std::array<uint32_t, 1> fresh{2}; disabled.lookup.committed(1, fresh);
    check(disabled.lookup.matches(1) && disabled.lookup.propose(1).count == 0,
        "enabling did not rebuild only exact fresh committed history");
    LookupFixture two; const auto base = two.prefix; const auto words = two.sampler.random().words_consumed();
    const auto full = two.run();
    check(full.accepted == 2 && full.count == 2 && two.evaluated == 0 && two.tracked == 2 && two.prefix == base + 2 &&
        full.tokens[0].source_logit == 10 && full.tokens[1].source_logit == 20 && two.lookup.matches(two.prefix) &&
        two.sampler.random().words_consumed() == words, "k2 publication/source logits/entropy");
    LookupFixture one; one.accept = 1; const auto partial = one.run();
    check(partial.accepted == 1 && partial.count == 2 && partial.tokens[0].id == 0 && partial.tokens[1].id == 1 &&
        one.evaluated == 1 && one.tracked == 2 && one.retained.get(one.prefix).logits[3] == 11,
        "k1 correction was not an ordinary evaluated step");
    LookupFixture zero; zero.accept = 0; const auto miss = zero.run();
    check(miss.accepted == 0 && miss.count == 1 && zero.evaluated == 1 && zero.tracked == 1, "k0 ordinary fallback");
    LookupFixture budget; const auto last = budget.run(1);
    check(last.count == 1 && !budget.verifies && budget.evaluated == 1, "budget guard reached gather/verifier");
    LookupFixture empty_budget; refuses([&] { empty_budget.run(0); }, "zero output budget executed a token");
    check(!empty_budget.verifies && !empty_budget.evaluated, "zero budget reached model work");
    refuses([] { TargetLookup<FakeResult> too_small(4, 2, 2, 1024); }, "insufficient prepared host budget");
    LookupFixture first_stop; const std::array<uint32_t, 1> stop0{0}; const auto terminal = first_stop.run(2, stop0);
    check(terminal.count == 1 && !first_stop.verifies && first_stop.retained.stop_id() == 0 &&
        !first_stop.retained.continuable(first_stop.prefix), "zero first stop was not ordinary one-step");
    LookupFixture second_stop(true); const std::array<uint32_t, 1> stop1{1}; const auto terminal2 = second_stop.run(2, stop1);
    check(terminal2.count == 2 && terminal2.accepted == 2 && !second_stop.evaluated &&
        second_stop.retained.stop_id() == 1 && !second_stop.retained.continuable(second_stop.prefix), "second stop did not commit exactly two");
    LookupFixture stale; const auto proposal = stale.lookup.propose(stale.prefix); const auto epoch = stale.lookup.epoch();
    stale.lookup.restore(stale.prefix);
    check(stale.lookup.epoch() != epoch, "same-prefix restore did not renew lookup epoch");
    refuses([&] { static_cast<void>(stale.lookup.eligible(proposal, stale.prefix, stale.retained.get(stale.prefix).logits, 2, {})); }, "stale proposal survived restore");
    auto wrong_base = stale.lookup.propose(stale.prefix); ++wrong_base.base_prefix;
    refuses([&] { static_cast<void>(stale.lookup.eligible(wrong_base, stale.prefix, stale.retained.get(stale.prefix).logits, 2, {})); }, "wrong proposal base admitted");
    LookupFixture nonfinite; nonfinite.verified_logits[0] = std::numeric_limits<float>::infinity();
    refuses([&] { nonfinite.run(); }, "nonfinite returned row published");
    check(!nonfinite.lookup.matches(nonfinite.prefix) && !nonfinite.sampler.valid(), "bad row left reusable history");
    LookupFixture policy; policy.sampler = TargetSampler::fresh_branch({{0, 0, 1, 0}, 7}, policy.prefix);
    policy.sampler.invalidate(); refuses([&] { policy.run(); }, "invalid sampler authorized verification");
    LookupFixture stochastic; stochastic.sampler = TargetSampler({{0.5, 0, 1, 0}, 7});
    for (uint32_t i = 1; i <= stochastic.prefix; ++i) require(stochastic.sampler.committed_input(i), "stochastic setup");
    refuses([&] { stochastic.run(); }, "stochastic policy reached greedy verifier");
    for (uint32_t failure = 0; failure < 3; ++failure) {
        LookupFixture failed; const auto before = failed.prefix;
        failed.fail_verify = failure == 0; failed.fail_track = failure == 1; failed.wrong_prefix = failure == 2;
        refuses([&] { failed.run(); }, "lookup failure hidden");
        check(failed.prefix == before + (failure == 0 ? 0 : failure == 1 ? 2 : 3) &&
            !failed.lookup.matches(failed.prefix) && !failed.retained.matches(failed.prefix) && !failed.sampler.valid(),
            "failure reused mismatched committed bookkeeping");
        refuses([&] { static_cast<void>(failed.lookup.propose(failed.prefix)); }, "failure did not require explicit reset");
        failed.lookup.reset(); check(failed.lookup.matches(0), "explicit lookup reset did not clear history");
    }
}
"""


def test_lookup_round_executes_extracted_owned_bookkeeping(tmp_path):
    protocol = load("lookup_protocol", "scripts/check_owned_target_protocol.py")
    unit = (
        '#include "tokspur/json.hpp"\n#include "tokspur/gguf.hpp"\n'
        '#include "tokspur/sampling.hpp"\n#include "tokspur/prompt_lookup.hpp"\n'
        "#include <algorithm>\n#include <array>\n#include <charconv>\n#include <cmath>\n"
        "#include <limits>\n#include <iostream>\n#include <optional>\n#include <set>\n"
        "#include <type_traits>\n#include <utility>\nnamespace tokspur { namespace {\n"
        + protocol.extract((ROOT / "native/src/target.cpp").read_text())
        + protocol.CHECKS
        + CHECKS
        + "\n} }\nint main() { try { tokspur::run(); tokspur::run_stack_experiments(); tokspur::run_lookup(); }"
        + "catch (const std::exception& error) { std::cerr << error.what(); return 1; } }\n"
    )
    source, executable = tmp_path / "lookup.cpp", tmp_path / "lookup"
    source.write_text(unit)
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
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run([str(executable)], check=True, timeout=60)
