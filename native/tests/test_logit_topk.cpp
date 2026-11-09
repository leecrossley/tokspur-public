#include "tokspur/graph.hpp"
#include "tokspur/sampled_path.hpp"
#include "tokspur/sampling.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string_view>
#include <vector>

using namespace tokspur;

namespace {

void check(bool ok, const char* why) {
    if (!ok) throw FormatError(why);
}

template <class F>
void rejects_with(F&& action, std::string_view expected_substr) {
    try {
        action();
    } catch (const FormatError& err) {
        if (std::string_view(err.what()).find(expected_substr) != std::string_view::npos) return;
        std::cerr << "unexpected error: " << err.what() << " (expected substring: " << expected_substr << ")\n";
        throw FormatError("unexpected refusal message in test_logit_topk");
    }
    throw FormatError("expected FormatError refusal in test_logit_topk");
}

uint32_t float_bits(float v) {
    uint32_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

float bits_to_float(uint32_t b) {
    float v = 0.0f;
    std::memcpy(&v, &b, sizeof(v));
    return v;
}

SampledSparseLaw make_test_selector_law(uint32_t vocab_size, std::span<const uint32_t, 16> cands,
                                        std::span<const float, 16> scores, double temperature) {
    if (temperature == 0.0) {
        size_t best = 0;
        for (size_t i = 1; i < 16; ++i) {
            if (scores[i] > scores[best] || (scores[i] == scores[best] && cands[i] < cands[best])) best = i;
        }
        const std::array<uint32_t, 1> id{{cands[best]}};
        const std::array<uint64_t, 1> u{{sampled_path_grid_size}};
        return SampledSparseLaw::from_grid_units(vocab_size, id, u);
    }
    float max_s = scores[0];
    for (size_t i = 1; i < 16; ++i) max_s = std::max(max_s, scores[i]);
    std::array<double, 16> probs{};
    double sum = 0.0;
    for (size_t i = 0; i < 16; ++i) {
        probs[i] = std::exp((double(scores[i]) - double(max_s)) / temperature);
        sum += probs[i];
    }
    std::array<uint32_t, 16> ids{};
    std::array<uint64_t, 16> units{};
    size_t out_n = 0;
    double cum = 0.0;
    uint64_t prev = 0;
    for (size_t i = 0; i < 16; ++i) {
        cum += probs[i] / sum;
        const uint64_t end = (i == 15)
            ? sampled_path_grid_size
            : std::min<uint64_t>(sampled_path_grid_size, uint64_t(std::llround(std::min(1.0, cum) * double(sampled_path_grid_size))));
        if (end > prev) {
            ids[out_n] = cands[i];
            units[out_n] = end - prev;
            prev = end;
            ++out_n;
        }
    }
    if (out_n == 0) {
        ids[0] = cands[0];
        units[0] = sampled_path_grid_size;
        out_n = 1;
    } else if (prev < sampled_path_grid_size) {
        units[out_n - 1] += sampled_path_grid_size - prev;
    }
    return SampledSparseLaw::from_grid_units(vocab_size, std::span(ids).first(out_n), std::span(units).first(out_n));
}

void run_gpu_logit_topk(MetalGraph& graph, GpuRowsF32 logits, GpuView scratch, GpuView out_view,
                        std::span<GpuLogitTopKRow> out_host) {
    graph.begin();
    graph.logit_topk(logits, scratch, out_view);
    static_cast<void>(graph.submit());
    graph.read(out_view, std::as_writable_bytes(out_host));
}

void test_gpu_and_cpu_parity(MetalGraph& graph) {
    for (uint32_t vocab : {32u, 64u, 5000u, 248320u}) {
        const uint32_t rows = (vocab == 248320u) ? 4u : 3u;
        std::vector<float> logits(size_t(rows) * vocab);
        std::mt19937 rng(0x51a70000u ^ vocab);
        std::normal_distribution<float> dist(0.0f, 3.25f);
        for (size_t i = 0; i < logits.size(); ++i) {
            float v = dist(rng);
            if (!std::isfinite(v) || std::fpclassify(v) == FP_SUBNORMAL) v = 0.0f;
            logits[i] = v;
        }
        // Inject exact ties and signed zeroes on row 0 to check deterministic lowest-ID tie breaking and bit preservation.
        logits[3] = 17.5f;
        logits[19] = 17.5f;
        logits[31] = 17.5f;
        if (vocab >= 64u) {
            logits[vocab - 2u] = -0.0f;
            logits[vocab - 1u] = +0.0f;
            for (uint32_t c = 0; c < vocab - 2u; ++c) {
                if (c != 3u && c != 19u && c != 31u) logits[c] = -100.0f - float(c % 11u);
            }
        }

        const auto logits_gpu = graph.upload(std::as_bytes(std::span(logits)), "test_logit_topk.logits");
        const auto scratch_gpu = graph.zeros(MetalGraph::logit_topk_scratch_bytes(rows, vocab), "test_logit_topk.scratch");
        const auto out_gpu = graph.zeros(MetalGraph::logit_topk_output_bytes(rows), "test_logit_topk.out");

        std::vector<GpuLogitTopKRow> gpu_rows(rows);
        run_gpu_logit_topk(graph, GpuRowsF32{logits_gpu, rows, vocab, uint64_t(vocab) * sizeof(float)},
                           scratch_gpu, out_gpu, gpu_rows);

        for (uint32_t r = 0; r < rows; ++r) {
            const auto row_span = std::span<const float>(logits.data() + size_t(r) * vocab, vocab);
            const auto cpu_row = cpu_logit_topk_row(row_span);
            const auto& g = gpu_rows[r];
            check(g.vocabulary == vocab && g.count == std::min<uint32_t>(vocab, gpu_logit_topk_capacity),
                  "GPU topk header mismatch");
            check(g.first_nonfinite_col == 0xffffffffu && g.first_subnormal_col == 0xffffffffu,
                  "unexpected fault column on clean row");
            for (uint32_t i = 0; i < g.count; ++i) {
                check(g.ids[i] == cpu_row.ids[i], "GPU vs CPU topk token ID mismatch");
                check(float_bits(g.logits[i]) == float_bits(cpu_row.logits[i]),
                      "GPU vs CPU topk raw float bits mismatch");
            }
        }
        if (vocab >= 64u) {
            check(gpu_rows[0].ids[3] == vocab - 2u && float_bits(gpu_rows[0].logits[3]) == 0x80000000u,
                  "signed -0.0f bit preservation or tie-break order failed");
            check(gpu_rows[0].ids[4] == vocab - 1u && float_bits(gpu_rows[0].logits[4]) == 0x00000000u,
                  "signed +0.0f tie-break order failed");
        }

        const std::array<SamplingPolicy, 6> policies{{
            SamplingPolicy{0.0, 0u, 1.0, 0.0},
            SamplingPolicy{1.0, 1u, 1.0, 0.0},
            SamplingPolicy{0.6, 20u, 0.9, 0.0},
            SamplingPolicy{0.8, 64u, 0.95, 0.02},
            SamplingPolicy{1.0, 64u, 1.0, 0.0},
            SamplingPolicy{0.7, 16u, 0.85, 0.05},
        }};
        for (const auto& pol : policies) {
            check(sampling_topk_eligible(pol), "expected eligible policy");
            for (uint32_t r = 0; r < rows; ++r) {
                const auto row_span = std::span<const float>(logits.data() + size_t(r) * vocab, vocab);
                auto dense_dist = sampling_distribution(row_span, pol);
                auto sparse_dist = sampling_distribution_from_topk(gpu_rows[r], pol);
                check(dense_dist.vocabulary() == sparse_dist.vocabulary(), "distribution vocabulary mismatch");
                check(dense_dist.deterministic() == sparse_dist.deterministic(), "distribution deterministic flag mismatch");
                check(dense_dist.support().size() == sparse_dist.support().size(), "support size mismatch");
                for (size_t s = 0; s < dense_dist.support().size(); ++s) {
                    const uint32_t id = dense_dist.support()[s];
                    check(sparse_dist.support()[s] == id, "support ID order mismatch");
                    check(dense_dist.probability(id) == sparse_dist.probability(id), "exact support probability mismatch");
                }
                check(dense_dist.probability(vocab - 1u) == sparse_dist.probability(vocab - 1u), "tail probability mismatch");
                SamplingRng rng_a(1234567ull + r), rng_b(1234567ull + r);
                for (uint32_t draw = 0; draw < 16u; ++draw) {
                    check(sample(dense_dist, rng_a) == sample(sparse_dist, rng_b), "sampled token mismatch");
                    check(rng_a.words_consumed() == rng_b.words_consumed(), "RNG word consumption mismatch");
                }
                auto moved = std::move(sparse_dist);
                check(sparse_dist.vocabulary() == 0u, "moved-from SamplingDistribution must zero vocabulary");
                SamplingRng rng_bad(99ull);
                rejects_with([&] { static_cast<void>(sample(sparse_dist, rng_bad)); }, "empty sampling distribution");
                check(moved.vocabulary() == vocab, "moved-to SamplingDistribution lost vocabulary");
            }
        }
        check(!sampling_topk_eligible(SamplingPolicy{0.8, 0u, 0.9, 0.0}), "unbounded top_k must not be topk-eligible");
        check(!sampling_topk_eligible(SamplingPolicy{0.8, 65u, 0.9, 0.0}), "top_k > 64 must not be topk-eligible");

        graph.release(out_gpu);
        graph.release(scratch_gpu);
        graph.release(logits_gpu);
    }
}

void test_refusal_parity(MetalGraph& graph) {
    constexpr uint32_t vocab = 8192u;
    constexpr uint32_t rows = 2u;
    std::vector<float> logits(size_t(rows) * vocab, -1.0f);
    // Row 0: subnormal at column 5, NaN at column 5000.
    logits[5] = bits_to_float(0x00000001u);
    logits[5000] = std::numeric_limits<float>::quiet_NaN();
    // Row 1: -Inf at column 17, negative subnormal at column 99.
    logits[vocab + 17] = -std::numeric_limits<float>::infinity();
    logits[vocab + 99] = bits_to_float(0x80000001u);

    const auto logits_gpu = graph.upload(std::as_bytes(std::span(logits)), "test_logit_topk.bad_logits", false);
    const auto scratch_gpu = graph.zeros(MetalGraph::logit_topk_scratch_bytes(rows, vocab), "test_logit_topk.bad_scratch");
    const auto out_gpu = graph.zeros(MetalGraph::logit_topk_output_bytes(rows), "test_logit_topk.bad_out");

    std::vector<GpuLogitTopKRow> gpu_rows(rows);
    run_gpu_logit_topk(graph, GpuRowsF32{logits_gpu, rows, vocab, uint64_t(vocab) * sizeof(float)},
                       scratch_gpu, out_gpu, gpu_rows);

    check(gpu_rows[0].first_subnormal_col == 5u && gpu_rows[0].first_nonfinite_col == 5000u,
          "GPU fault column tracking mismatch on row 0");
    check(gpu_rows[1].first_nonfinite_col == 17u && gpu_rows[1].first_subnormal_col == 99u,
          "GPU fault column tracking mismatch on row 1");

    const SamplingPolicy pol{0.7, 32u, 0.9, 0.0};
    rejects_with([&] { static_cast<void>(sampling_distribution_from_topk(gpu_rows[0], pol)); }, "subnormal sampling logit");
    rejects_with([&] { static_cast<void>(sampling_distribution_from_topk(gpu_rows[1], pol)); }, "nonfinite sampling logit");

    const SampledPathIdentity ident{1ull, 1ull, 1ull, 100u, vocab, 7u};
    const std::array<uint32_t, 1> cand_id{{10u}};
    const std::array<uint64_t, 1> cand_mass{{sampled_path_grid_size}};
    SampledProposalRow prop{100u, 7u, 10u, SampledSparseLaw::from_grid_units(vocab, cand_id, cand_mass)};
    std::array<SampledTargetTopKRow, 2> target_topk{{
        SampledTargetTopKRow{100u, gpu_rows[0]},
        SampledTargetTopKRow{101u, gpu_rows[1]},
    }};
    SamplingRng rng_clean(1ull);
    rejects_with([&] {
        static_cast<void>(decide_sampled_path_topk(ident, std::span(&prop, 1), ident, target_topk, pol, 16u, {}, rng_clean, 0ull));
    }, "sampled path nonfinite target logit");

    // Pure subnormal without non-finite must refuse with "subnormal sampling logit" in decide_sampled_path_topk as well.
    logits[5000] = 0.25f;
    logits[vocab + 17] = 0.5f;
    logits[vocab + 99] = 0.5f;
    graph.write(logits_gpu, std::as_bytes(std::span(logits)));
    run_gpu_logit_topk(graph, GpuRowsF32{logits_gpu, rows, vocab, uint64_t(vocab) * sizeof(float)},
                       scratch_gpu, out_gpu, gpu_rows);
    target_topk[0].topk = gpu_rows[0];
    target_topk[1].topk = gpu_rows[1];
    rejects_with([&] {
        static_cast<void>(decide_sampled_path_topk(ident, std::span(&prop, 1), ident, target_topk, pol, 16u, {}, rng_clean, 0ull));
    }, "subnormal sampling logit");

    graph.release(out_gpu);
    graph.release(scratch_gpu);
    graph.release(logits_gpu);
}

void test_decide_sampled_path_parity_and_bench(MetalGraph& graph) {
    constexpr uint32_t vocab = 248320u;
    constexpr uint32_t k = 3u;
    constexpr uint32_t verify_rows = k + 1u;

    std::vector<float> logits(size_t(verify_rows) * vocab);
    std::mt19937 gen(0x857051eeu);
    std::normal_distribution<float> dist(0.0f, 2.5f);
    for (float& v : logits) v = dist(gen);

    const auto logits_gpu = graph.upload(std::as_bytes(std::span(logits)), "test_logit_topk.bench_logits", false);
    const auto scratch_gpu = graph.zeros(MetalGraph::logit_topk_scratch_bytes(verify_rows, vocab), "test_logit_topk.bench_scratch");
    const auto out_gpu = graph.zeros(MetalGraph::logit_topk_output_bytes(verify_rows), "test_logit_topk.bench_out");

    const SamplingPolicy pol{0.7, 40u, 0.95, 0.02};
    std::vector<GpuLogitTopKRow> gpu_rows(verify_rows);
    run_gpu_logit_topk(graph, GpuRowsF32{logits_gpu, verify_rows, vocab, uint64_t(vocab) * sizeof(float)},
                       scratch_gpu, out_gpu, gpu_rows);

    for (uint32_t round = 0; round < 25u; ++round) {
        for (uint32_t r = 0; r < verify_rows; ++r) {
            const uint32_t hot = (round * 131u + r * 17u) % vocab;
            logits[size_t(r) * vocab + hot] = 9.5f + float(r) * 0.25f;
        }
        graph.write(logits_gpu, std::as_bytes(std::span(logits)));
        run_gpu_logit_topk(graph, GpuRowsF32{logits_gpu, verify_rows, vocab, uint64_t(vocab) * sizeof(float)},
                           scratch_gpu, out_gpu, gpu_rows);

        const SampledPathIdentity ident{1ull, 1ull, 1ull, 1000u, vocab, 42u};
        SamplingRng prop_rng(9000ull + round);
        const uint64_t begin_words = prop_rng.words_consumed();
        std::array<SampledProposalRow, k> props_dense{};
        std::array<SampledProposalRow, k> props_sparse{};
        uint32_t prev = 42u;
        for (uint32_t pos = 0; pos < k; ++pos) {
            std::array<uint32_t, 16> cands{};
            std::array<float, 16> scores{};
            for (uint32_t j = 0; j < 16u; ++j) {
                cands[j] = gpu_rows[pos].ids[j];
                scores[j] = gpu_rows[pos].logits[j] + (j == (round + pos) % 3u ? 0.5f : -0.2f);
            }
            auto law_a = make_test_selector_law(vocab, cands, scores, pol.temperature);
            auto law_b = make_test_selector_law(vocab, cands, scores, pol.temperature);
            const uint64_t ticket = law_a.deterministic() ? 0ull : uint64_t(std::ldexp(prop_rng.uniform(), 53));
            const uint32_t sel = law_a.sample_at(ticket);
            props_dense[pos] = SampledProposalRow{1000u + pos, prev, sel, std::move(law_a)};
            props_sparse[pos] = SampledProposalRow{1000u + pos, prev, sel, std::move(law_b)};
            prev = sel;
        }

        std::array<SampledTargetRow, verify_rows> dense_targets{};
        std::array<SampledTargetTopKRow, verify_rows> topk_targets{};
        for (uint32_t r = 0; r < verify_rows; ++r) {
            dense_targets[r] = SampledTargetRow{1000u + r, std::span<const float>(logits.data() + size_t(r) * vocab, vocab)};
            topk_targets[r] = SampledTargetTopKRow{1000u + r, gpu_rows[r]};
        }

        const std::array<uint32_t, 1> stops{{props_dense[1].selected}};
        const auto stop_span = (round % 5u == 0u) ? std::span<const uint32_t>(stops) : std::span<const uint32_t>{};

        const auto dec_dense = decide_sampled_path(ident, props_dense, ident, dense_targets, pol, 32u, stop_span, prop_rng, begin_words);
        const auto dec_topk = decide_sampled_path_topk(ident, props_sparse, ident, topk_targets, pol, 32u, stop_span, prop_rng, begin_words);

        check(dec_dense.accepted == dec_topk.accepted && dec_dense.count == dec_topk.count &&
              dec_dense.stopped == dec_topk.stopped && dec_dense.terminal == dec_topk.terminal &&
              dec_dense.random_after.words_consumed() == dec_topk.random_after.words_consumed(),
              "decide_sampled_path vs decide_sampled_path_topk header mismatch");
        for (uint32_t t = 0; t < dec_dense.count; ++t) {
            check(dec_dense.tokens[t].id == dec_topk.tokens[t].id &&
                  dec_dense.tokens[t].accepted_proposal == dec_topk.tokens[t].accepted_proposal &&
                  float_bits(dec_dense.tokens[t].source_logit) == float_bits(dec_topk.tokens[t].source_logit),
                  "decide_sampled_path vs decide_sampled_path_topk token or source_logit mismatch");
        }
    }

    // Synthetic latency comparison on 4 x 248320 logits (warmup + 20 iterations).
    constexpr uint32_t iters = 20u;
    for (uint32_t w = 0; w < 12u; ++w) {
        run_gpu_logit_topk(graph, GpuRowsF32{logits_gpu, verify_rows, vocab, uint64_t(vocab) * sizeof(float)},
                           scratch_gpu, out_gpu, gpu_rows);
    }
    double gpu_ms_sum = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t sink = 0;
    for (uint32_t it = 0; it < iters; ++it) {
        graph.begin();
        graph.logit_topk(GpuRowsF32{logits_gpu, verify_rows, vocab, uint64_t(vocab) * sizeof(float)}, scratch_gpu, out_gpu);
        const auto b = graph.submit();
        gpu_ms_sum += b.gpu_ms;
        graph.read(out_gpu, std::as_writable_bytes(std::span(gpu_rows)));
        for (uint32_t r = 0; r < verify_rows; ++r) {
            auto d = sampling_distribution_from_topk(gpu_rows[r], pol);
            SamplingRng rng(100ull + it * 8u + r);
            sink ^= sample(d, rng);
        }
    }
    const double topk_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;

    const auto t1 = std::chrono::steady_clock::now();
    uint32_t sink_ref = 0;
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t r = 0; r < verify_rows; ++r) {
            auto d = sampling_distribution(std::span<const float>(logits.data() + size_t(r) * vocab, vocab), pol);
            SamplingRng rng(100ull + it * 8u + r);
            sink_ref ^= sample(d, rng);
        }
    }
    const double dense_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / iters;
    check(sink == sink_ref, "timed benchmark sampling mismatch");

    std::cout << "logit_topk: exact CPU/GPU parity, signed-zero tie-break, refusal ordering and 25 sampled-path rounds passed"
              << " | 4x248320 dense_host_ms=" << dense_wall_ms
              << " gpu_topk_total_ms=" << topk_wall_ms
              << " (gpu_ms=" << (gpu_ms_sum / iters) << ")\n";

    graph.release(out_gpu);
    graph.release(scratch_gpu);
    graph.release(logits_gpu);
}

} // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 2, "usage: test-logit_topk METALLIB");
        MetalGraph graph(argv[1], 64ull << 20);
        const auto initial_bytes = graph.memory().live_bytes;
        graph.prepare_logit_topk();
        test_gpu_and_cpu_parity(graph);
        test_refusal_parity(graph);
        test_decide_sampled_path_parity_and_bench(graph);
        check(graph.memory().live_bytes == initial_bytes, "test_logit_topk leaked GPU buffers");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
