#!/usr/bin/env python3
"""Check the actual owned target's CPU-only request/result/entropy protocol.

This extracts a deliberately marked region from target.cpp, rather than keeping a
second validator implementation. It never loads Metal, a model or another engine.
The resulting checks do not qualify real-stack continuation or cancellation.
"""

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(source):
    def between(start, end):
        at = source.index(start)
        return source[at : source.index(end, at)]

    return "\n".join(
        (
            between("void require(bool ok,", "\nJson number("),
            between("const Json& field(", "\nuint32_t choice("),
            between("void keys(", "\nJson policy_file("),
            between("// BEGIN owned target protocol:", "// END owned target protocol."),
        )
    )


CHECKS = r"""
struct FakeResult { std::vector<float> logits; };
int checks = 0;
void check(bool condition, const char* label) {
    ++checks; if (!condition) throw std::runtime_error(label);
}
template<class F> void refuses(F action, const char* label) {
    bool refused = false;
    try { action(); } catch (const FormatError&) { refused = true; }
    check(refused, label);
}
Request parse(std::string_view body, uint32_t prefix = 0, bool has_logits = false,
              bool continuable = false, bool allow = false, uint32_t small = 0,
              uint32_t block = 0, bool sampling_matches = true) {
    return request(parse_json(body), prefix, small, block, allow, has_logits, continuable, sampling_matches);
}
void run() {
    hc_down_f32_preparation(false, false, false, false, false, false, true);
    hc_down_f32_preparation(true, true, true, true, true, true, false);
    for (uint32_t missing = 0; missing < 5; ++missing) refuses([&] { hc_down_f32_preparation(true, missing != 0, missing != 1, missing != 2, missing != 3, missing != 4, false); }, "HC down startup refuses incompatible input/precision/scheduling/carry");
    refuses([&] { hc_down_f32_preparation(true, true, true, true, true, true, true); }, "HC down startup refuses mutable kernel/stack/prefill experiments");
    gdn_old_order_packet(false, 65536); gdn_old_order_packet(true, 512);
    refuses([&] { gdn_old_order_packet(true, 513); }, "old-order GDN refuses oversized native input packets before state mutation");
    final_input_trace_preparation(false, false); final_input_trace_preparation(false, true); final_input_trace_preparation(true, false);
    refuses([&] { final_input_trace_preparation(true, true); }, "deferred final-input logits refuse full-logit tracing before model setup");
    CommittedResult<FakeResult> state;
    check(!state.matches(0) && !state.matches(7) && !state.continuable(7), "empty retained state");
    refuses([&] { static_cast<void>(state.get(7)); }, "empty state get refusal");
    state.commit(FakeResult{}, 6, false);
    check(!state.matches(6) && !state.continuable(6), "committed trunk without logits cannot generate");
    refuses([&] { static_cast<void>(state.get(6)); }, "unready committed result cannot expose logits");
    FakeResult first{{1.0f, 3.0f, 2.0f}}; const auto* original = first.logits.data();
    state.commit(std::move(first), 7);
    check(state.matches(7) && state.continuable(7), "committed result available");
    check(state.get(7).logits.data() == original, "retention moves rather than copies logits");
    check(!state.matches(6) && !state.matches(8), "retention is exact-prefix only");
    refuses([&] { static_cast<void>(state.get(8)); }, "stale result refused");
    state.stopped(0);
    check(state.matches(7) && !state.continuable(7), "zero-valued stop is terminal");
    state.commit(FakeResult{{2.0f}}, 8);
    check(state.continuable(8) && !state.matches(7), "new committed input clears terminal status");
    // A failure before publication keeps the matching result. A future executor
    // that advances then throws must be treated as a mismatch and invalidated.
    check(state.matches(8), "failed step before commit retains exact state");
    if (!state.matches(9)) state.clear();
    check(!state.matches(8) && !state.continuable(9), "advanced-prefix mismatch invalidates logits");
    state.commit(FakeResult{{4.0f}}, 8); state.clear();
    check(!state.matches(8) && !state.matches(0), "reset clears retained results");

    // Retained verified publication checks finiteness, not subnormality. A
    // later nonfinite value must still win, before any entropy is published.
    for (bool invalid : {false, true}) {
        TargetLookup<FakeResult> lookup(3, 3, 3, 8ull << 20);
        lookup.reset(); const std::array<uint32_t, 1> input{0}; lookup.committed(1, input);
        auto random = TargetSampler::fresh_branch({}, 1);
        const auto before = random.random();
        CommittedResult<FakeResult> published; published.commit(FakeResult{{1, 2, 3}}, 1);
        lookup.prepare(published, random, 1);
        LookupProposal proposal{}; proposal.epoch = lookup.epoch(); proposal.base_prefix = 1;
        proposal.count = 1; proposal.tokens[0] = 0;
        const std::array<float, 3> row{std::numeric_limits<float>::denorm_min(), 1,
            invalid ? std::numeric_limits<float>::quiet_NaN() : 0};
        bool refused = false;
        try { lookup.publish(proposal, 1, 2, row, published, random, [](uint32_t, std::span<const uint32_t>) {}); }
        catch (const FormatError& error) {
            check(invalid && std::string_view(error.what()) == "nonfinite verified retained logits", "retained publication error precedence");
            refused = true;
        }
        check(refused == invalid && random.random() == before, "retained publication preserves RNG ownership");
        if (invalid) check(!lookup.matches(1) && !published.matches(1) && !random.matches(1), "failed publication revokes stale state");
        else check(published.matches(2) && std::fpclassify(published.get(2).logits[0]) == FP_SUBNORMAL, "retained publication preserves finite subnormals");
    }

#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    const auto final_parse = [&](std::string_view body, bool ready = false) {
        return request(parse_json(body), 512, 16, 3, false, ready, ready, true, 0, false, false, 512, false, true, 0, false, false, false, true);
    };
    auto deferred = final_parse(R"({"tokens":[1],"generate":0,"reset":false,"prefill_final":false})");
    check(deferred.final_logits && !deferred.prefill_final && deferred.known_prefill, "explicit intermediate known input remains admissible without logits");
    check(final_parse(R"({"tokens":[1],"generate":0,"reset":false,"prefill_final":true})").prefill_final, "explicit final input accepted");
    refuses([&] { final_parse(R"({"tokens":[1],"generate":0,"reset":false})"); }, "selected input requires final flag");
    refuses([&] { final_parse(R"({"tokens":[1],"generate":0,"reset":false,"prefill_final":1})"); }, "final flag must be boolean");
    refuses([&] { final_parse(R"({"tokens":[],"generate":1,"reset":false})"); }, "generation before final logits refused");
    refuses([&] { final_parse(R"({"tokens":[],"generate":1,"reset":false,"prefill_final":true})", true); }, "generation cannot carry final-input flag");
    refuses([&] { parse(R"({"tokens":[1],"generate":0,"reset":true,"prefill_final":false})"); }, "unselected final-input flag refused");
    check(final_parse(R"({"tokens":[],"generate":1,"reset":false})", true).generate == 1, "ordinary continuation after final logits accepted");
#endif
    const auto fresh = parse(R"({"tokens":[1],"generate":4096,"reset":true})");
    check(fresh.generate == 4096 && fresh.reset && fresh.tokens == std::vector<uint32_t>{1}, "raised bounded output budget");
    refuses([&] { parse(R"({"tokens":[1],"generate":4097,"reset":true})"); }, "output budget still bounded");
    refuses([&] { parse(R"({"tokens":[1],"generate":-1,"reset":true})"); }, "negative output budget");
    refuses([&] { parse(R"({"tokens":[248320],"generate":0,"reset":true})"); }, "vocabulary bound");
    refuses([&] { parse(R"({"tokens":[-1],"generate":0,"reset":true})"); }, "negative token");
    refuses([&] { parse(R"({"tokens":[],"generate":1,"reset":false})"); }, "initial empty continuation");
    refuses([&] { parse(R"({"tokens":[],"generate":1,"reset":true})", 7, true, true); }, "empty reset is not continuation");
    refuses([&] { parse(R"({"tokens":[],"generate":0,"reset":true})", 7, true, true); }, "empty reset without generation refused");
    refuses([&] { parse(R"({"tokens":[],"generate":1,"reset":false})", 7, false, true); }, "missing exact-prefix logits");
    const auto continuation = parse(R"({"tokens":[],"generate":300,"reset":false})", 7, true, true);
    check(continuation.tokens.empty() && continuation.generate == 300 && !continuation.reset, "empty valid continuation accepted");
    const auto peek = parse(R"({"tokens":[],"generate":0,"reset":false})", 7, true, true);
    check(peek.generate == 0 && peek.tokens.empty(), "zero-work retained-result query");
    refuses([&] { parse(R"({"tokens":[],"generate":1,"reset":false})", 7, true, false); }, "no generation after stopped completion");
    check(parse(R"({"tokens":[],"generate":0,"reset":false})", 7, true, false).generate == 0, "terminal logits may be queried without reevaluation");
    check(parse(R"({"tokens":[1],"generate":1,"reset":false})", 7, true, false).tokens.size() == 1, "new input can begin another turn after stop");
    check(parse(R"({"tokens":[],"generate":1,"reset":false})", 65535, true, true).generate == 1, "last context slot can be committed once");
    refuses([&] { parse(R"({"tokens":[],"generate":2,"reset":false})", 65535, true, true); }, "continuation includes output reservation");
    check(parse(R"({"tokens":[],"generate":0,"reset":false})", 65536, true, true).generate == 0, "full prefix no-work query");
    refuses([&] { parse(R"({"tokens":[],"generate":1,"reset":false})", 65536, true, true); }, "full prefix cannot generate");
    check(parse(R"({"tokens":[1],"generate":4096,"reset":false})", 61439, true, true).generate == 4096, "input and full output reservation boundary");
    refuses([&] { parse(R"({"tokens":[1],"generate":4096,"reset":false})", 61440, true, true); }, "input counted in reservation");
    check(parse(R"({"tokens":[1],"generate":4096,"reset":true})", 65536, true, false).generate == 4096, "reset uses empty-prefix capacity");

    for (uint32_t formats = 0; formats <= 3; ++formats) {
        const auto body = std::string(R"({"tokens":[1],"generate":0,"reset":true,"block_gemv_formats":)") + std::to_string(formats) + "}";
        check(parse(body, 7, true, true, true).block_formats == formats, "fresh-prefix block format mask accepted");
        refuses([&] { parse(body, 7, true, true, false); }, "block override requires startup consent");
    }
    refuses([&] { parse(R"({"tokens":[1],"generate":0,"reset":false,"block_gemv_formats":0})", 7, true, true, true); }, "even identical block append override refused");
    refuses([&] { parse(R"({"tokens":[1],"generate":0,"reset":true,"block_gemv_formats":4})", 7, true, true, true); }, "invalid block format mask refused");
    refuses([&] { parse(R"({"tokens":[1],"generate":0,"reset":true,"block_gemv_formats":-1})", 7, true, true, true); }, "negative block format mask refused");
    for (uint32_t splits : {0u, 1u, 4u, 8u, 16u, 32u}) {
        const auto body = std::string(R"({"tokens":[1],"generate":0,"reset":true,"small_gemv_splits":)") + std::to_string(splits) + "}";
        check(parse(body, 7, true, true, true).small_splits == splits, "fresh-prefix small split accepted");
        refuses([&] { parse(body, 7, true, true, false); }, "small override requires startup consent");
    }
    refuses([&] { parse(R"({"tokens":[1],"generate":0,"reset":false,"small_gemv_splits":0})", 7, true, true, true); }, "even identical small append override refused");
    refuses([&] { parse(R"({"tokens":[1],"generate":0,"reset":true,"small_gemv_splits":2})", 7, true, true, true); }, "invalid small split refused");
    const auto inherited = parse(R"({"tokens":[],"generate":1,"reset":false})", 7, true, true, false, 16, 3);
    check(inherited.small_splits == 16 && inherited.block_formats == 3, "continuation inherits kernel policy unchanged");
    const auto both = parse(R"({"tokens":[1],"generate":0,"reset":true,"small_gemv_splits":16,"block_gemv_formats":3,"stop_ids":[0,248319]})", 7, true, true, true);
    check(both.small_splits == 16 && both.block_formats == 3 && both.stops == std::vector<uint32_t>({0, 248319}), "combined validated policy and stop range");
    refuses([&] { parse(R"({"tokens":[1],"generate":0,"reset":true,"stop_ids":[248320]})"); }, "stop vocabulary limit");
    refuses([&] { parse(R"({"tokens":[1],"generate":0,"reset":true,"unknown":true})"); }, "unknown request field refused");
    Json::Array many(65537, Json::integer(0));
    const auto large = Json(Json::Object{{"tokens", Json(std::move(many))}, {"generate", Json::integer(0)}, {"reset", Json(true)}});
    refuses([&] { request(large, 0, 0, 0, false, false, false); }, "input token count bounded");
    Json::Array stops(17, Json::integer(0));
    const auto too_many_stops = Json(Json::Object{{"tokens", Json(Json::Array{Json::integer(0)})}, {"generate", Json::integer(0)}, {"reset", Json(true)}, {"stop_ids", Json(std::move(stops))}});
    refuses([&] { request(too_many_stops, 0, 0, 0, false, false, false); }, "stop list bounded");

    const auto stochastic = parse(R"({"tokens":[1],"generate":4,"reset":true,"sampling":{"temperature":0.75,"top_k":20,"top_p":0.9,"min_p":0.03},"seed":18446744073709551615})");
    check(stochastic.sampling.policy.temperature == 0.75 && stochastic.sampling.policy.top_k == 20 &&
          stochastic.sampling.policy.top_p == 0.9 && stochastic.sampling.policy.min_p == 0.03 &&
          stochastic.sampling.seed == std::numeric_limits<uint64_t>::max(), "explicit sampling policy and full uint64 seed");
    check(fresh.sampling.policy.temperature == 0 && fresh.sampling.policy.top_k == 0 &&
          fresh.sampling.policy.top_p == 1 && fresh.sampling.policy.min_p == 0 && fresh.sampling.seed == 0,
          "omitted reset sampling returns to original greedy defaults");
    check(parse(R"({"tokens":[1],"generate":0,"reset":true,"seed":17})").sampling.seed == 17,
          "explicit seed may accompany greedy reset without consuming entropy");
    for (const auto body : {
        R"({"tokens":[1],"generate":1,"reset":true,"sampling":{"temperature":1,"top_k":0,"top_p":1,"min_p":0}})",
        R"({"tokens":[1],"generate":1,"reset":false,"seed":0})",
        R"({"tokens":[],"generate":1,"reset":false,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0}})",
        R"({"tokens":[1],"generate":0,"reset":true,"sampling":{}})",
        R"({"tokens":[1],"generate":0,"reset":true,"sampling":null})",
        R"({"tokens":[1],"generate":0,"reset":true,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0,"unknown":1}})",
        R"({"tokens":[1],"generate":0,"reset":true,"seed":-1})",
        R"({"tokens":[1],"generate":0,"reset":true,"seed":18446744073709551616})",
        R"({"tokens":[1],"generate":0,"reset":true,"seed":1.0})",
        R"({"tokens":[1],"generate":0,"reset":true,"seed":"1"})"})
        refuses([&] { parse(body, 7, true, true); }, "malformed or append sampling override refused");
    const auto with_sampling = [](const std::string& temperature, const std::string& k,
                                  const std::string& p, const std::string& m) {
        return std::string(R"({"tokens":[1],"generate":0,"reset":true,"seed":0,"sampling":{"temperature":)") +
            temperature + ",\"top_k\":" + k + ",\"top_p\":" + p + ",\"min_p\":" + m + "}}";
    };
    for (const auto value : {"-1", "100.01", "1e999", "1e-999", "true", "null", "\"1\""})
        refuses([&] { parse(with_sampling(value, "0", "1", "0")); }, "sampling temperature is finite typed and bounded");
    for (const auto value : {"-1", "248321", "1.0", "true"})
        refuses([&] { parse(with_sampling("0", value, "1", "0")); }, "top-k integer vocabulary bound");
    for (const auto value : {"-1", "0", "1.01", "1e999", "false"})
        refuses([&] { parse(with_sampling("0", "0", value, "0")); }, "top-p strict positive bound");
    for (const auto value : {"-0.01", "1.01", "1e999", "false"})
        refuses([&] { parse(with_sampling("0", "0", "1", value)); }, "min-p closed unit interval");
    refuses([&] { parse(with_sampling("0." + std::string(70, '0'), "0", "1", "0")); }, "sampling numeric lexeme bound");
    check(parse(with_sampling("100", "248320", "1", "1")).sampling.policy.temperature == 100, "sampling upper endpoints");
    refuses([&] { parse(R"({"tokens":[1],"generate":0,"reset":false})", 7, true, true, false, 0, 0, false); },
            "new input cannot resurrect invalidated entropy");
    refuses([&] { parse(R"({"tokens":[],"generate":0,"reset":false})", 7, true, true, false, 0, 0, false); },
            "no-work query cannot hide invalidated entropy");
    check(parse(R"({"tokens":[1],"generate":0,"reset":true})", 7, true, true, false, 0, 0, false).reset,
          "validated reset may replace invalidated entropy");

    const std::vector<float> logits{1, 3, 3, 0};
    TargetSampler greedy;
    const auto greedy_random = greedy.random();
    check(greedy.committed_input(1) && greedy.random() == greedy_random, "prefill consumes no entropy");
    const auto greedy_id = sampled_step(greedy, logits, 1, [&](uint32_t id) {
        check(id == 1, "default greedy preserves lowest-ID tie");
        check(greedy.random().words_consumed() == 0, "no entropy publication before greedy commit");
        check(greedy.committed_input(2), "greedy model commit");
    });
    check(greedy_id == 1 && greedy.matches(2) && greedy.random() == greedy_random, "greedy default byte selection and entropy");
    const TargetSampling options{{0.8, 0, 1, 0}, 938};
    TargetSampler sampled(options);
    check(sampled.committed_input(1), "stochastic prefill prefix");
    auto reference = sampled.random();
    const auto expected = sample(sampling_distribution(logits, options.policy), reference);
    const auto actual = sampled_step(sampled, logits, 1, [&](uint32_t id) {
        check(id == expected && sampled.random().words_consumed() == 0, "tentative stochastic draw not yet published");
        check(sampled.committed_input(2), "stochastic model commit");
        check(sampled.random().words_consumed() == 0, "prefix publication precedes entropy publication");
    });
    check(actual == expected && sampled.random() == reference && sampled.random().words_consumed() == 1,
          "successful token publishes exactly its tentative entropy");
    const auto after_token = sampled.random();
    check(sampled.committed_input(3) && sampled.random() == after_token, "new turn input inherits exact stream without reseeding");
    CommittedResult<FakeResult> held;
    held.commit(FakeResult{{1}}, 3);
    const auto before_refusal = sampled.random();
    refuses([&] { parse(R"({"tokens":[1],"generate":4097,"reset":true,"seed":999})"); }, "refused reset request");
    check(sampled.random() == before_refusal && sampled.matches(3), "refused request does not mutate entropy");
    refuses([&] { reset_session(held, sampled, TargetSampling{}, [] { throw FormatError("fake reset refusal"); }); },
            "executor reset refusal");
    check(sampled.random() == before_refusal && sampled.matches(3) && held.matches(3), "refused stack reset preserves session");
    reset_session(held, sampled, TargetSampling{}, [] {});
    check(sampled.matches(0) && sampled.random().words_consumed() == 0 && sampled.options().policy.temperature == 0 &&
          sampled.random().seed() == 0 && !held.matches(3), "successful reset installs fresh greedy session");

    TargetSampler failed(options);
    check(failed.committed_input(1), "failure test prefill");
    const auto untouched = failed.random();
    refuses([&] { sampled_step(failed, logits, 1, [](uint32_t) { throw FormatError("before commit"); }); }, "precommit refusal");
    check(failed.matches(1) && failed.random() == untouched, "precommit failure consumes no entropy");
    refuses([&] { sampled_step(failed, logits, 1, [&](uint32_t) {
        check(failed.committed_input(2), "fake committed prefix"); throw FormatError("after commit");
    }); }, "postcommit callback failure");
    check(!failed.valid() && failed.random() == untouched, "commit-then-throw invalidates unpublished draw");
    refuses([&] { sampled_step(failed, logits, 2, [](uint32_t) {}); }, "invalidated draw cannot be silently reused");
    TargetSampler hidden_commit(options);
    check(hidden_commit.committed_input(1), "hidden commit prefill");
    held.commit(FakeResult{{1}}, 1);
    uint32_t fake_model_prefix = 1;
    refuses([&] { sampled_step(hidden_commit, logits, 1, [&](uint32_t) { ++fake_model_prefix; throw FormatError("GPU committed before host publication"); }); },
            "GPU commit before prefix publication");
    reconcile_failure(held, hidden_commit, fake_model_prefix, false);
    check(!hidden_commit.valid() && !held.matches(1) && hidden_commit.random().words_consumed() == 0,
          "outer actual mismatch recovery invalidates hidden commit entropy and logits");
    TargetSampler callback_missing(options); check(callback_missing.committed_input(1), "missing commit prefill");
    refuses([&] { sampled_step(callback_missing, logits, 1, [](uint32_t) {}); }, "callback cannot omit prefix publication");
    check(!callback_missing.valid(), "missing commit invalidates session");
    TargetSampler harmless(options); check(harmless.committed_input(1), "recovery prefix");
    held.commit(FakeResult{{1}}, 1);
    reconcile_failure(held, harmless, 1, false);
    check(harmless.matches(1) && held.matches(1), "precommit format refusal preserves exact session");
    reconcile_failure(held, harmless, 1, true);
    check(!harmless.valid() && !held.matches(1), "fatal errors invalidate even matching state");

    TargetSampler whole(options), chunked(options);
    check(whole.committed_input(1) && chunked.committed_input(1), "paired replay prefill");
    const auto generate = [&](TargetSampler& owner, uint32_t count) {
        std::vector<uint32_t> ids;
        for (uint32_t i = 0; i < count; ++i) {
            const auto prefix = owner.prefix();
            ids.push_back(sampled_step(owner, logits, prefix, [&](uint32_t) { check(owner.committed_input(prefix + 1), "paired token commit"); }));
        }
        return ids;
    };
    const auto all_ids = generate(whole, 30);
    auto split_ids = generate(chunked, 11);
    const auto rest = generate(chunked, 19); split_ids.insert(split_ids.end(), rest.begin(), rest.end());
    check(all_ids == split_ids && whole.random() == chunked.random() && whole.matches(31) && chunked.matches(31),
          "one-shot and chunked stochastic continuation share exact entropy and IDs");
    reset_session(held, chunked, options, [] {}); check(chunked.committed_input(1), "reset replay prefill");
    check(generate(chunked, 30) == all_ids && chunked.random() == whole.random(), "seeded reset replay exact");
    TargetSampler point({{1, 1, 1, 0}, 1}); check(point.committed_input(1), "point mass prefill");
    static_cast<void>(sampled_step(point, logits, 1, [&](uint32_t) { check(point.committed_input(2), "point mass commit"); }));
    check(point.random().words_consumed() == 0, "stochastic point mass consumes no entropy");
}
// Separate extension counter keeps the frozen historical base cohort legible.
// Every extension check still runs before either success message is emitted.
int stack_checks = 0;
void stack_check(bool condition, const char* label) {
    ++stack_checks; if (!condition) throw std::runtime_error(label);
}
template<class F> void stack_refuses(F action, const char* label) {
    bool refused = false;
    try { action(); } catch (const FormatError&) { refused = true; }
    stack_check(refused, label);
}
Request stack_parse(std::string_view body, uint32_t current = 2, bool allow = true,
                    bool prepared = true, bool allow_kernel = false) {
    return request(parse_json(body), 7, 16, 3, allow_kernel, true, true, true, current, allow, prepared);
}
void run_stack_experiments() {
    const std::array<std::string_view, 3> names{"direct", "indirect-compatible", "recurrent-replay"};
    for (uint32_t current = 0; current < names.size(); ++current) {
        const auto inherited = stack_parse(R"({"tokens":[1],"generate":0,"reset":false})", current);
        stack_check(inherited.execution == current && !inherited.reset && inherited.small_splits == 16 && inherited.block_formats == 3,
            "same-resident append inherits current execution and startup kernels");
        for (uint32_t next = 0; next < names.size(); ++next) {
            const auto body = std::string(R"({"tokens":[1],"generate":0,"reset":true,"stack_execution":")") + std::string(names[next]) + R"("})";
            const auto chosen = stack_parse(body, current);
            stack_check(chosen.execution == next && chosen.reset && chosen.small_splits == 16 && chosen.block_formats == 3,
                "every same-resident arm switch requires fresh prefix and preserves kernel identity");
            stack_refuses([&] { stack_parse(body, current, false); }, "stack override requires separate startup consent");
            stack_refuses([&] { stack_parse(body, current, true, false); }, "stack override requires startup replay preparation");
            const auto append = std::string(R"({"tokens":[1],"generate":0,"reset":false,"stack_execution":")") + std::string(names[next]) + R"("})";
            stack_refuses([&] { stack_parse(append, current); }, "even identical stack append override refused");
        }
        for (const auto key : {"small_gemv_splits", "block_gemv_formats"}) {
            const auto body = std::string(R"({"tokens":[1],"generate":0,"reset":true,")") + key + R"(":0})";
            stack_refuses([&] { stack_parse(body, current, true, true, true); },
                "prepared captures freeze request kernels even in direct/compatible arms");
        }
    }
    stack_check(stack_parse(R"({"tokens":[1],"generate":0,"reset":false})", 0, false, false).execution == 0,
        "ordinary direct default still works without stack experiment consent");
    stack_check(stack_parse(R"({"tokens":[1],"generate":0,"reset":true})", 1, false, false).execution == 1,
        "startup compatible still works without preparing captures");
    stack_refuses([&] { stack_parse(R"({"tokens":[1],"generate":0,"reset":true})", 3); }, "invalid inherited execution refused");
    stack_refuses([&] { stack_parse(R"({"tokens":[1],"generate":0,"reset":true})", 2, false, false); }, "replay inheritance needs preparation");
    stack_refuses([&] { stack_parse(R"({"tokens":[1],"generate":0,"reset":true,"stack_execution":"other"})"); }, "invalid stack string refused");
    stack_refuses([&] { stack_parse(R"({"tokens":[1],"generate":0,"reset":true,"stack_execution":2})"); }, "numeric stack override refused");
    stack_refuses([&] { stack_parse(R"({"tokens":[1],"generate":0,"reset":true,"stack_execution":null})"); }, "null stack override refused");
    const auto sampled = stack_parse(R"({"tokens":[1],"generate":0,"reset":true,"stack_execution":"direct","sampling":{"temperature":0.7,"top_k":4,"top_p":0.9,"min_p":0.1},"seed":99})");
    stack_check(sampled.execution == 0 && sampled.sampling.policy.temperature == 0.7 && sampled.sampling.seed == 99,
        "same-resident mode switch preserves allowed fresh-prefix sampling changes");
    uint32_t resets = 0;
    TargetSampler sampler; CommittedResult<FakeResult> retained;
    bool advanced = true;
    for (uint32_t prefix = 1; prefix <= 7; ++prefix) advanced = sampler.committed_input(prefix) && advanced;
    stack_check(advanced, "refused reset fixture starts with exact entropy prefix");
    retained.commit(FakeResult{{1}}, 7); const auto entropy = sampler.random();
    const auto attempt = [&](std::string_view body) {
        const auto parsed = stack_parse(body);
        if (parsed.reset) reset_session(retained, sampler, parsed.sampling, [&] { ++resets; });
    };
    for (const auto body : {
        R"({"tokens":[1],"generate":0,"reset":true,"stack_execution":"direct","stop_ids":[248320]})",
        R"({"tokens":[1],"generate":0,"reset":true,"stack_execution":"direct","seed":-1})",
        R"({"tokens":[1],"generate":4097,"reset":true,"stack_execution":"direct"})",
        R"({"tokens":[1],"generate":0,"reset":true,"stack_execution":"direct","small_gemv_splits":16})"})
        stack_refuses([&] { attempt(body); }, "all malformed options must refuse before reset publication");
    stack_check(!resets && retained.matches(7) && sampler.matches(7) && sampler.random() == entropy,
        "invalid same-resident options do not publish reset, retained-logit or entropy changes");
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Verify extraction only; do not compile or claim executed checks.",
    )
    parser.add_argument(
        "--input512",
        action="store_true",
        help="Also compile the explicit final-input512 request controls.",
    )
    args = parser.parse_args()
    source = (ROOT / "native/src/target.cpp").read_text()
    protocol = extract(source)
    fingerprint = hashlib.sha256(protocol.encode()).hexdigest()
    if args.source_only:
        print(f"CPU protocol extracted, not compiled or executed: {fingerprint}")
        return
    unit = (
        '#include "tokspur/json.hpp"\n#include "tokspur/gguf.hpp"\n#include "tokspur/sampling.hpp"\n#include "tokspur/prompt_lookup.hpp"\n'
        "#include <algorithm>\n#include <array>\n#include <charconv>\n#include <cmath>\n#include <limits>\n"
        "#include <iostream>\n#include <optional>\n#include <set>\n"
        "#include <type_traits>\n#include <utility>\n"
        "namespace tokspur { namespace {\n"
        + (
            "#define TOKSPUR_PRIVATE_HTTP_PREFILL512 1\n#define TOKSPUR_PRIVATE_PREFILL_CHUNK 1\n"
            if args.input512
            else ""
        )
        + protocol
        + CHECKS
        + "\n} }\nint main() { try { tokspur::run(); tokspur::run_stack_experiments(); "
        + 'std::cout << "Owned CPU target protocol checks passed: " << tokspur::checks << "\\n" '
        + '<< "Owned CPU target stack-experiment protocol checks passed: " << tokspur::stack_checks << "\\n"; return 0; } '
        + 'catch (const std::exception& error) { std::cerr << error.what() << "\\n"; return 1; } }\n'
    )
    with tempfile.TemporaryDirectory(prefix="tokspur target protocol ") as directory:
        unit_path = Path(directory) / "target protocol.cpp"
        executable = Path(directory) / "target protocol"
        unit_path.write_text(unit)
        subprocess.run(
            [
                args.cxx,
                "-std=c++20",
                "-O1",
                "-g",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
                "-I" + str(ROOT / "native/include"),
                str(unit_path),
                str(ROOT / "native/src/json.cpp"),
                str(ROOT / "native/src/unicode.cpp"),
                str(ROOT / "native/src/sampling.cpp"),
                str(ROOT / "native/src/prompt_lookup.cpp"),
                "-o",
                str(executable),
            ],
            check=True,
            timeout=120,
        )
        subprocess.run([str(executable)], check=True, timeout=60)
    print(f"Extracted owned CPU protocol SHA256: {fingerprint}")
    print(
        "Scope: request/result/entropy publication with fake commits; not real-model continuation or stochastic quality."
    )


if __name__ == "__main__":
    main()
