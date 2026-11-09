#include "tokspur/mtp_head.hpp"
#include "tokspur/attention_math.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include "tokspur/mtp_sampling.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
using namespace tokspur;
size_t checks = 0;
void check(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
void refused(const std::function<void()>& action) {
    ++checks; try { action(); } catch (const FormatError&) { return; }
    throw std::runtime_error("invalid MTP operation accepted");
}
template<class T> std::vector<T> read(MetalGraph& g, GpuView view) {
    const auto bytes = g.read(view); std::vector<T> result(bytes.size() / sizeof(T));
    std::memcpy(result.data(), bytes.data(), bytes.size()); return result;
}
struct Fixture {
    MetalGraph& graph;
    std::mt19937 random{8293};
    std::vector<GpuView> owned;
    explicit Fixture(MetalGraph& g) : graph(g) { owned.reserve(64); }
    ~Fixture() { for (const auto view : owned) graph.release(view); }
    std::vector<float> values(size_t count, float bias = 0) {
        std::vector<float> out(count);
        for (auto& x : out) x = bias + float(int(random() % 2001) - 1000) * 0.00015f;
        return out;
    }
    GpuView upload(std::span<const float> values) {
        const auto view = graph.upload(std::as_bytes(values), "MTP generated parameter"); owned.push_back(view); return view;
    }
    GpuView vector(size_t count, float bias = 0) { return upload(values(count, bias)); }
    GpuMatrix matrix(size_t count) { return {vector(count), 0}; }
    ResidualWeights residual(ResidualShape s) {
        const auto n = uint64_t(s.branches) * s.channels;
        return {vector(n, 1), matrix(n * s.rank), matrix(s.rank * n), matrix(n * s.branches)};
    }
    MoeWeights moe(MoeShape s) {
        return {matrix(s.channels * s.experts), matrix(s.channels * s.hidden * s.experts),
            matrix(s.channels * s.hidden * s.experts), matrix(s.hidden * s.channels * s.experts),
            matrix(s.channels * s.hidden), matrix(s.channels * s.hidden), matrix(s.hidden * s.channels), matrix(s.channels)};
    }
    MtpHeadWeights weights(MtpHeadShape s) {
        const auto c = s.residual.channels, h = c * s.residual.branches;
        const auto q = s.query_heads * s.head_width, k = s.kv_heads * s.head_width;
        const auto frequencies = rotary_frequencies(s.rotary_width, 10000000);
        return {vector(h, 1), vector(c, 1), matrix(uint64_t(2) * c * c), residual(s.residual), residual(s.residual),
            {matrix(uint64_t(c) * q * 2), matrix(uint64_t(c) * k), matrix(uint64_t(c) * k), matrix(uint64_t(q) * c),
                vector(s.head_width, 1), vector(s.head_width, 1), upload(frequencies)}, moe(s.experts),
            {vector(h, 1), matrix(uint64_t(h) * s.residual.rank), matrix(uint64_t(s.residual.rank) * h)},
            matrix(uint64_t(c) * s.vocabulary)};
    }
};
void primitives(MetalGraph& graph) {
    Fixture f(graph); graph.prepare_mtp_head(); graph.prepare_mtp_head_tiled();
    const std::array<float, 3> e{1, 2, 3};
    const std::array<float, 6> h{4, 5, 6, 7, 8, 9};
    const auto ev = f.upload(e), hv = f.upload(h), output = graph.zeros(12 * 4, "MTP concat output");
    graph.begin(); graph.mtp_concat(ev, hv, output, 2, 3); static_cast<void>(graph.submit());
    check(read<float>(graph, output) == std::vector<float>({1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9}), "MTP EH concatenation order");
    graph.release(output);
    for (const auto geometry : std::array<std::array<uint32_t, 4>, 2>{{{4, 2, 4, 7}, {24, 2, 256, 9}}}) {
    const auto heads = geometry[0], kv_heads = geometry[1], width = geometry[2], rows = geometry[3];
    const auto q = f.values(heads * width), raw_k = f.values(rows * kv_heads * width), raw_v = f.values(raw_k.size());
    std::vector<uint16_t> k(raw_k.size()), v(raw_v.size());
    for (size_t i = 0; i < k.size(); ++i) { k[i] = finite_bf16(raw_k[i]); v[i] = finite_bf16(raw_v[i]); }
    const auto qv = f.upload(q), kv = graph.upload(std::as_bytes(std::span(k)), "MTP generated BF16 keys"),
        vv = graph.upload(std::as_bytes(std::span(v)), "MTP generated BF16 values"), out = graph.zeros(q.size() * 4, "MTP dense result");
    graph.begin(); graph.mtp_dense_attention(qv, kv, vv, out, rows, heads, kv_heads, width); static_cast<void>(graph.submit());
    std::array<std::vector<float>, 3> actual{read<float>(graph, out), {}, {}};
    const auto partials = graph.zeros(uint64_t(heads) * ((rows + 127) / 128) * (width + 2) * 4, "MTP tiny tiled partials");
    for (uint32_t arm = 1; arm < 3; ++arm) {
        graph.begin(); graph.mtp_dense_attention_tiled(qv, kv, vv, out, partials, rows, heads, kv_heads, width, arm == 1 ? 128 : 256);
        static_cast<void>(graph.submit()); actual[arm] = read<float>(graph, out);
    }
    for (uint32_t head = 0; head < heads; ++head) {
        std::vector<double> scores(rows); double maximum = -std::numeric_limits<double>::infinity();
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t c = 0; c < width; ++c)
                scores[row] += double(q[head * width + c]) * read_bf16(k[(row * kv_heads + head / (heads / kv_heads)) * width + c]);
            scores[row] /= std::sqrt(double(width)); maximum = std::max(maximum, scores[row]);
        }
        double denominator = 0; for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
        for (uint32_t c = 0; c < width; ++c) {
            double expected = 0;
            for (uint32_t row = 0; row < rows; ++row) expected += scores[row] * read_bf16(v[(row * kv_heads + head / (heads / kv_heads)) * width + c]);
            for (const auto& arm : actual)
                check(std::abs(arm[head * width + c] - expected / denominator) < 1e-6, "MTP dense GQA reference mismatch");
        }
    }
    graph.release(partials);
    graph.release(out); graph.release(vv); graph.release(kv);
    }
}
void draft_attention_tiles(MetalGraph& graph) {
    Fixture fixture(graph); graph.prepare_mtp_head(); graph.prepare_mtp_head_tiled();
    constexpr uint32_t heads = 24, kv_heads = 2, width = 256;
    double worst_fp64 = 0, worst_serial = 0;
    const auto screen = [&](uint32_t prefix, uint32_t pattern) {
        auto q = fixture.values(heads * width), raw_k = fixture.values(prefix * kv_heads * width), raw_v = fixture.values(raw_k.size());
        if (pattern) std::fill(q.begin(), q.end(), 0);
        if (pattern == 1) {
            std::fill(raw_v.begin(), raw_v.end(), 0);
            for (uint32_t h = 0; h < kv_heads; ++h) for (uint32_t d = 0; d < width; ++d)
                raw_v[((prefix - 1) * kv_heads + h) * width + d] = 1;
        } else if (pattern == 2) {
            for (uint32_t row = 0; row < prefix; ++row) for (uint32_t h = 0; h < kv_heads; ++h)
                for (uint32_t d = 0; d < width; ++d) raw_v[(row * kv_heads + h) * width + d] = h ? -0.5f : 0.25f;
        } else if (pattern == 3) {
            std::fill(raw_k.begin(), raw_k.end(), 0);
            for (uint32_t h = 0; h < heads; ++h) q[h * width] = 16;
            for (uint32_t row = 0; row < prefix; ++row) for (uint32_t h = 0; h < kv_heads; ++h)
                raw_k[(row * kv_heads + h) * width] = row + 1 == prefix ? 32 : -32;
        }
        std::vector<uint16_t> keys(raw_k.size()), values(raw_v.size());
        for (size_t i = 0; i < keys.size(); ++i) { keys[i] = finite_bf16(raw_k[i]); values[i] = finite_bf16(raw_v[i]); }
        const auto query = fixture.upload(q), key = graph.upload(std::as_bytes(std::span(keys)), "draft tile BF16 keys"),
            value = graph.upload(std::as_bytes(std::span(values)), "draft tile BF16 values"),
            output = graph.zeros(heads * width * 4, "draft tile result");
        // This same workspace first holds the longer actual3K request, then a
        // shorter query. Only active tiles may be read, never stale capacity.
        const auto partials = graph.zeros(uint64_t(heads) * ((3072 + 127) / 128) * (width + 2) * 4, "draft tile partials");
        const auto allocations = graph.memory().allocations;
        graph.begin(); graph.mtp_dense_attention(query, key, value, output, prefix, heads, kv_heads, width);
        static_cast<void>(graph.submit()); const auto serial = read<float>(graph, output);
        std::vector<double> expected(heads * width), scores(prefix);
        for (uint32_t h = 0; h < heads; ++h) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (uint32_t row = 0; row < prefix; ++row) {
                scores[row] = 0;
                for (uint32_t d = 0; d < width; ++d)
                    scores[row] += double(q[h * width + d]) * read_bf16(keys[(row * kv_heads + h / (heads / kv_heads)) * width + d]);
                scores[row] /= std::sqrt(double(width)); maximum = std::max(maximum, scores[row]);
            }
            double denominator = 0;
            for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (uint32_t d = 0; d < width; ++d) {
                for (uint32_t row = 0; row < prefix; ++row)
                    expected[h * width + d] += scores[row] * read_bf16(values[(row * kv_heads + h / (heads / kv_heads)) * width + d]);
                expected[h * width + d] /= denominator;
            }
        }
        for (const auto tile_tokens : {128u, 256u}) {
            graph.begin(); graph.mtp_dense_attention_tiled(query, key, value, output, partials, prefix, heads, kv_heads, width, tile_tokens);
            const auto batch = graph.submit(); check(batch.dispatches == 2, "draft tiled attention dispatch count");
            const auto actual = read<float>(graph, output);
            double error = 0, energy = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                const auto delta = actual[i] - expected[i]; error += delta * delta; energy += expected[i] * expected[i];
                worst_fp64 = std::max(worst_fp64, std::abs(delta));
                worst_serial = std::max(worst_serial, std::abs(double(actual[i]) - serial[i]));
                check(std::abs(delta) < 1e-5, "draft tiled attention FP64 absolute error");
                check(std::abs(double(actual[i]) - serial[i]) < 1e-5, "draft tiled attention serial error");
                if (pattern == 1) check(std::abs(double(actual[i]) - 1.0 / prefix) < 1e-7, "draft tail mean-of-means canary");
                if (pattern == 2) check(actual[i] == (i / width < heads / 2 ? 0.25f : -0.5f), "draft GQA head pairing");
            }
            check(energy ? error / energy < 1e-7 : error == 0, "draft tiled attention FP64 normalized error");
            check(graph.memory().allocations == allocations, "draft tiled attention allocated request GPU storage");
        }
        // A shorter prefix reuses the dirty partial buffer and changes its
        // compact head stride. With zero queries its expected output is a
        // direct BF16 value mean, independent of all now-invisible future rows.
        if (prefix == 3072 && pattern == 0) {
            const std::vector<float> zeros(heads * width);
            const auto zero_query = graph.upload(std::as_bytes(std::span(zeros)), "draft zero-query stale-tail test");
            for (const auto tile_tokens : {128u, 256u}) {
                graph.begin(); graph.mtp_dense_attention_tiled(zero_query, key, value, output, partials, 129, heads, kv_heads, width, tile_tokens);
                static_cast<void>(graph.submit()); const auto actual = read<float>(graph, output);
                for (uint32_t h = 0; h < heads; ++h) for (uint32_t d = 0; d < width; ++d) {
                    double expected_value = 0;
                    for (uint32_t row = 0; row < 129; ++row) expected_value += read_bf16(values[(row * kv_heads + h / 12) * width + d]);
                    check(std::abs(double(actual[h * width + d]) - expected_value / 129) < 1e-6, "draft stale partial rows leaked");
                }
            }
            graph.release(zero_query);
        }
        graph.release(partials); graph.release(output); graph.release(value); graph.release(key);
    };
    for (const auto prefix : {1u, 127u, 128u, 129u, 255u, 256u, 257u, 3072u}) screen(prefix, 0);
    screen(257, 1); screen(129, 2); screen(257, 3);
    std::cout << "draft tiled attention FP64 max_abs=" << worst_fp64 << " serial max_abs=" << worst_serial
              << "; numerical draft-Q component only\n";
}

void draft_attention_screen(MetalGraph& graph) {
    Fixture fixture(graph); graph.prepare_mtp_head_tiled();
    constexpr uint32_t prefix = 3072, heads = 24, kv_heads = 2, width = 256;
    const auto q = fixture.values(heads * width), raw = fixture.values(prefix * kv_heads * width);
    std::vector<uint16_t> cache(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) cache[i] = finite_bf16(raw[i]);
    const auto query = fixture.upload(q), keys = graph.upload(std::as_bytes(std::span(cache)), "draft screen keys"),
        values = graph.upload(std::as_bytes(std::span(cache)), "draft screen values"),
        output = graph.zeros(heads * width * 4, "draft screen result"),
        partials = graph.zeros(uint64_t(heads) * ((prefix + 127) / 128) * (width + 2) * 4, "draft screen partials");
    const auto ledger = graph.memory();
    const auto run = [&](uint32_t arm) {
        graph.begin();
        if (!arm) graph.mtp_dense_attention(query, keys, values, output, prefix, heads, kv_heads, width);
        else graph.mtp_dense_attention_tiled(query, keys, values, output, partials, prefix, heads, kv_heads, width, arm == 1 ? 128 : 256);
        const auto batch = graph.submit();
        check(std::isfinite(batch.gpu_ms) && batch.gpu_ms > 0, "draft screen GPU duration unavailable");
        return batch.gpu_ms;
    };
    for (uint32_t arm = 0; arm < 3; ++arm) static_cast<void>(run(arm));
    std::array<std::vector<double>, 3> samples;
    // Symmetric within-owner order; this is a descriptive generated-kernel
    // screen, never task/runtime or quiet-host timing qualification.
    for (uint32_t sweep = 0; sweep < 3; ++sweep)
        for (const auto arm : {0u, 1u, 2u, 2u, 1u, 0u}) samples[arm].push_back(run(arm));
    for (uint32_t arm = 0; arm < 3; ++arm) {
        auto sorted = samples[arm]; std::sort(sorted.begin(), sorted.end());
        std::cout << "draft attention screen prefix3072 arm=" << arm << " gpu_median_ms=" << (sorted[2] + sorted[3]) / 2
                  << " samples=" << samples[arm].size() << "; generated/unqualified\n";
    }
    check(graph.memory().live_bytes == ledger.live_bytes && graph.memory().allocations == ledger.allocations,
        "draft kernel screen changed fixed GPU ledger");
    graph.release(partials); graph.release(output); graph.release(values); graph.release(keys);
}

void heads(MetalGraph& graph, MtpDraftAttention mode = MtpDraftAttention::serial) {
    Fixture fixture(graph);
    const MtpHeadShape shape{{4, 8, 3}, {8, 7, 5, 3}, 4, 2, 4, 4, 11, 16};
    const auto weights = fixture.weights(shape);
    const auto before = graph.memory().live_bytes;
    MtpHeadPolicy invalid; invalid.draft_attention = static_cast<MtpDraftAttention>(3);
    refused([&] { MtpHead bad(graph, weights, shape, invalid); });
    {
        MtpHeadPolicy policy; policy.draft_attention = mode;
        MtpHead head(graph, weights, shape, policy);
        check(head.draft_attention() == mode, "immutable MTP draft attention policy differs");
        check(head.workspace_bytes() == graph.memory().live_bytes - before, "MTP workspace accounting");
        const auto allocations = graph.memory().allocations;
        std::array<std::vector<float>, 4> hc, embedding;
        for (uint32_t i = 0; i < 4; ++i) { hc[i] = fixture.values(32); embedding[i] = fixture.values(8); }
        for (uint32_t i = 0; i < 3; ++i) {
            const auto result = head.advance(hc[i], embedding[i], i + 1);
            check(result.cache_rows == i + 1 && result.absolute_position == i + 1 && result.batch.dispatches > 30, "MTP full advance publication");
        }
        const std::vector<float> full(head.logits().begin(), head.logits().end()), full_hc(head.hidden().begin(), head.hidden().end());
        head.reset();
        for (uint32_t i = 0; i < 2; ++i) {
            const auto batch = head.prime_pair(hc[i], embedding[i], i + 1);
            check(batch.dispatches == 19, "MTP cache-only dispatch count");
            refused([&] { static_cast<void>(head.logits()); }); refused([&] { static_cast<void>(head.hidden()); });
        }
        static_cast<void>(head.advance(hc[2], embedding[2], 3));
        check(std::equal(full.begin(), full.end(), head.logits().begin()), "MTP cache-only/full priming next-logit mismatch");
        check(std::equal(full_hc.begin(), full_hc.end(), head.hidden().begin()), "MTP cache-only/full priming next-HC mismatch");
        head.truncate(2); refused([&] { static_cast<void>(head.logits()); });
        static_cast<void>(head.advance(hc[3], embedding[3], 3));
        const std::vector<float> changed(head.logits().begin(), head.logits().end());
        head.reset(); static_cast<void>(head.prime_pair(hc[0], embedding[0], 1));
        static_cast<void>(head.prime_pair(hc[1], embedding[1], 2)); static_cast<void>(head.advance(hc[3], embedding[3], 3));
        check(std::equal(changed.begin(), changed.end(), head.logits().begin()), "MTP truncated suffix leaked into corrected continuation");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        head.reset(); static_cast<void>(head.advance(hc[0],embedding[0],1));
        const std::vector<float> first_hidden(head.hidden().begin(),head.hidden().end());
        static_cast<void>(head.advance(head.hidden(),embedding[1],2));
        const std::vector<float> recursive(head.logits().begin(),head.logits().end());
        head.reset(); static_cast<void>(head.advance(hc[0],embedding[0],1));
        static_cast<void>(head.advance(first_hidden,embedding[1],2));
        check(std::equal(recursive.begin(),recursive.end(),head.logits().begin()),"borrowed full-stream hidden changed recursive proposal");
        head.truncate(1); static_cast<void>(head.prime_pair(hc[1],embedding[1],2));
        static_cast<void>(head.advance(hc[2],embedding[2],3));
        const std::vector<float> corrected(head.logits().begin(),head.logits().end()),corrected_hc(head.hidden().begin(),head.hidden().end());
        head.reset(); static_cast<void>(head.prime_pair(hc[0],embedding[0],1));
        static_cast<void>(head.prime_pair(hc[1],embedding[1],2));static_cast<void>(head.advance(hc[2],embedding[2],3));
        check(std::equal(corrected.begin(),corrected.end(),head.logits().begin()) && std::equal(corrected_hc.begin(),corrected_hc.end(),head.hidden().begin()),
            "speculative second-cache row survived authoritative target-HC catch-up");
#endif
        const auto retained_rows = head.cache_rows();
        refused([&] { static_cast<void>(head.advance(hc[0], embedding[0], 5)); });
        auto bad = hc[0]; bad[0] = std::numeric_limits<float>::quiet_NaN();
        refused([&] { static_cast<void>(head.advance(bad, embedding[0], 4)); });
        check(head.cache_rows() == retained_rows && head.next_absolute_position() == 4, "MTP refused input mutated prefix");
        check(graph.memory().allocations == allocations, "MTP step allocated GPU payload");
        head.reset(65535); static_cast<void>(head.prime_pair(hc[0], embedding[0], 65535));
        refused([&] { static_cast<void>(head.advance(hc[1], embedding[1], 65536)); });
    }
    check(graph.memory().live_bytes == before, "MTP workspace leaked");
}
void q8_embedding_hidden_opt_in(MetalGraph& graph) {
    for (const uint32_t branches : {1u, 2u, 4u, 8u}) {
        Fixture fixture(graph);
        const MtpHeadShape shape{{branches, 32, 8}, {32, 16, 4, 2}, 4, 2, 8, 4, 32, 8};
        auto weights = fixture.weights(shape);
        std::vector<std::byte> encoded(64u * 34u);
        for (size_t block = 0; block < 64; ++block) {
            encoded[block * 34] = std::byte{0x00};
            encoded[block * 34 + 1] = std::byte{0x24};
            for (size_t column = 0; column < 32; ++column) {
                const int8_t value = int8_t(int((block * 7 + column * 3) % 17) - 8);
                encoded[block * 34 + 2 + column] = std::byte(uint8_t(value));
            }
        }
        const auto matrix = graph.upload(encoded, "MTP Q8 EH opt-in fixture");
        fixture.owned.push_back(matrix);
        weights.embedding_hidden = {matrix, 8};
        const auto hc = fixture.values(branches * 32), embedding = fixture.values(32);
        MtpHead ordinary(graph, weights, shape, {}), candidate(graph, weights, shape, {});
        candidate.configure_fast_round(true);
        for (uint32_t position = 1; position <= 2; ++position) {
            const auto expected = ordinary.advance(hc, embedding, position);
            const auto actual = candidate.advance(hc, embedding, position);
            check(expected.batch.dispatches == actual.batch.dispatches + branches - 1,
                "ordinary MTP EH projection no longer uses per-branch Q8 GEMV");
            for (size_t token = 0; token < expected.logits.size(); ++token) {
                check(std::abs(actual.logits[token] - expected.logits[token]) <= 1e-5f * (1 + std::abs(expected.logits[token])),
                    "opt-in MTP EH Q8 many logits differ from ordinary GEMV");
            }
        }
        ordinary.reset(); candidate.reset(); candidate.configure_fast_round(false);
        const auto expected = ordinary.advance(hc, embedding, 1), actual = candidate.advance(hc, embedding, 1);
        check(expected.batch.dispatches == actual.batch.dispatches &&
            std::equal(expected.logits.begin(), expected.logits.end(), actual.logits.begin()),
            "disabling MTP fast-round did not restore ordinary EH projection");
    }
}
void fast_round_parity(MetalGraph& graph) {
    Fixture fixture(graph);
    const MtpHeadShape shape{{4, 8, 3}, {8, 7, 5, 3}, 4, 2, 4, 4, 32, 32};
    const auto weights = fixture.weights(shape);
    std::array<std::vector<float>, 8> hc_rows, emb_rows;
    std::array<GpuView, 8> hc_gpu{};
    for (uint32_t i = 0; i < 8; ++i) {
        hc_rows[i] = fixture.values(32);
        emb_rows[i] = fixture.values(8);
        hc_gpu[i] = fixture.upload(hc_rows[i]);
    }
    std::vector<float> prime2_hc(64), prime2_emb(16);
    std::copy(hc_rows[1].begin(), hc_rows[1].end(), prime2_hc.begin());
    std::copy(hc_rows[2].begin(), hc_rows[2].end(), prime2_hc.begin() + 32);
    std::copy(emb_rows[1].begin(), emb_rows[1].end(), prime2_emb.begin());
    std::copy(emb_rows[2].begin(), emb_rows[2].end(), prime2_emb.begin() + 8);
    const auto prime2_hc_gpu = fixture.upload(prime2_hc);
    const auto before = graph.memory().live_bytes;
    {
        MtpHeadPolicy policy;
        policy.draft_attention = MtpDraftAttention::tiled128;
        MtpHead ref_head(graph, weights, shape, policy);
        MtpHead fast_head(graph, weights, shape, policy);
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        ref_head.prepare_priming_batch(4, 2ull << 20);
        fast_head.prepare_priming_batch(4, 2ull << 20);
#endif
        fast_head.configure_fast_round(true);
        check(fast_head.fast_round() && !ref_head.fast_round(), "MTP fast-round toggle state");
        const auto allocations = graph.memory().allocations;

        const SamplingPolicy sp{0.9, 0, 0.92, 0.02};
        SamplingRng rng_ref(77), rng_fast(77);

        // Round 1 (depth 3 chain) -> accept 2 drafts -> defer_prime_batch(2 rows) -> Round 2 (depth 2 chain) -> flush.
        const auto cb_ref_start = graph.lifetime_command_buffers();
        const auto r1_1 = ref_head.advance(hc_rows[0], emb_rows[0], 1);
        const auto d_ref_1 = draw_mtp_depth1(r1_1.logits, sp, rng_ref, shape.vocabulary);
        const auto r1_2 = ref_head.advance(ref_head.hidden(), emb_rows[1], 2);
        const auto d_ref_2 = draw_mtp_depth1(r1_2.logits, sp, d_ref_1.random_after, shape.vocabulary);
        const auto r1_3 = ref_head.advance(ref_head.hidden(), emb_rows[2], 3);
        const auto d_ref_3 = draw_mtp_depth1(r1_3.logits, sp, d_ref_2.random_after, shape.vocabulary);
        ref_head.truncate(1);
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        static_cast<void>(ref_head.prime_batch(prime2_hc, prime2_emb, 2));
#else
        static_cast<void>(ref_head.prime_pair(hc_rows[1], emb_rows[1], 2));
        static_cast<void>(ref_head.prime_pair(hc_rows[2], emb_rows[2], 3));
#endif
        const auto r2_1 = ref_head.advance(hc_rows[3], emb_rows[3], 4);
        const auto d_ref_4 = draw_mtp_depth1(r2_1.logits, sp, d_ref_3.random_after, shape.vocabulary);
        const std::vector<float> ref_hidden_r2(ref_head.hidden().begin(), ref_head.hidden().end());
        const auto r2_2 = ref_head.advance(ref_head.hidden(), emb_rows[4], 5);
        const auto d_ref_5 = draw_mtp_depth1(r2_2.logits, sp, d_ref_4.random_after, shape.vocabulary);
        ref_head.truncate(4);
        static_cast<void>(ref_head.prime_pair(hc_rows[4], emb_rows[4], 5));
        const auto cb_ref = graph.lifetime_command_buffers() - cb_ref_start;

        const auto cb_fast_start = graph.lifetime_command_buffers();
        const auto f1_1 = fast_head.advance_fast(hc_gpu[0], emb_rows[0], 1);
        const auto d_fast_1 = draw_mtp_depth1_topk(f1_1.topk, sp, rng_fast, shape.vocabulary);
        const auto f1_2 = fast_head.advance_fast_chain(emb_rows[1], 2);
        const auto d_fast_2 = draw_mtp_depth1_topk(f1_2.topk, sp, d_fast_1.random_after, shape.vocabulary);
        const auto f1_3 = fast_head.advance_fast_chain(emb_rows[2], 3);
        const auto d_fast_3 = draw_mtp_depth1_topk(f1_3.topk, sp, d_fast_2.random_after, shape.vocabulary);
        fast_head.truncate(1);
        fast_head.defer_prime_batch(prime2_hc_gpu, prime2_emb, 2);
        check(fast_head.deferred_prime_rows() == 2 && fast_head.cache_rows() == 3 && fast_head.next_absolute_position() == 4,
            "MTP deferred priming logical frontier");
        // Next round fuses the 2 deferred prime rows + advance_fast into a single command buffer.
        const auto f2_1 = fast_head.advance_fast(hc_gpu[3], emb_rows[3], 4);
        check(fast_head.deferred_prime_rows() == 0 && fast_head.cache_rows() == 4, "MTP deferred priming fused flush");
        const auto d_fast_4 = draw_mtp_depth1_topk(f2_1.topk, sp, d_fast_3.random_after, shape.vocabulary);
        const std::vector<float> fast_hidden_r2(fast_head.hidden().begin(), fast_head.hidden().end());
        const auto f2_2 = fast_head.advance_fast_chain(emb_rows[4], 5);
        const auto d_fast_5 = draw_mtp_depth1_topk(f2_2.topk, sp, d_fast_4.random_after, shape.vocabulary);
        fast_head.truncate(4);
        fast_head.defer_prime_batch(hc_gpu[4], emb_rows[4], 5);
        static_cast<void>(fast_head.flush_deferred_prime());
        const auto cb_fast = graph.lifetime_command_buffers() - cb_fast_start;

        check(d_fast_1.token == d_ref_1.token && d_fast_2.token == d_ref_2.token && d_fast_3.token == d_ref_3.token &&
            d_fast_4.token == d_ref_4.token && d_fast_5.token == d_ref_5.token,
            "MTP fast-round draft token parity");
        check(d_fast_5.random_after == d_ref_5.random_after, "MTP fast-round RNG parity");
        check(std::equal(ref_hidden_r2.begin(), ref_hidden_r2.end(), fast_hidden_r2.begin()),
            "MTP fast-round lazy hidden readback parity");
        check(cb_fast < cb_ref, "MTP fast-round fused priming must reduce command buffer submissions");
        check(graph.memory().allocations == allocations, "MTP fast-round steady-state allocation invariance");
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        std::vector<std::byte> ref_kv(ref_head.priming_cache_prefix_bytes()), fast_kv(fast_head.priming_cache_prefix_bytes());
        ref_head.read_priming_cache_prefix(ref_kv);
        fast_head.read_priming_cache_prefix(fast_kv);
        check(ref_kv == fast_kv, "MTP fast-round KV cache prefix byte parity");
#endif
    }
    check(graph.memory().live_bytes == before, "MTP fast-round workspace leaked");
}
void greedy_chain_parity(MetalGraph& graph) {
    Fixture fixture(graph);
    const MtpHeadShape shape{{4, 32, 8}, {32, 16, 5, 3}, 4, 2, 8, 4, 32, 32};
    const auto weights = fixture.weights(shape);
    const uint32_t c = shape.residual.channels, h = shape.residual.branches * c, vocab = shape.vocabulary;
    // Build deterministic Q8_0 token_embd table (vocab=32, c=32 -> 34 bytes/row).
    std::vector<std::byte> q8_embd(vocab * 34u);
    for (uint32_t tok = 0; tok < vocab; ++tok) {
        const uint16_t scale_f16 = 0x2c00u + uint16_t((tok % 7u) * 0x0040u); // small positive normal FP16 scales
        q8_embd[tok * 34u + 0u] = static_cast<std::byte>(scale_f16 & 0xffu);
        q8_embd[tok * 34u + 1u] = static_cast<std::byte>(scale_f16 >> 8u);
        for (uint32_t col = 0; col < 32u; ++col) {
            const int8_t q = int8_t(int((tok * 13u + col * 7u + 3u) % 61u) - 30);
            q8_embd[tok * 34u + 2u + col] = static_cast<std::byte>(static_cast<uint8_t>(q));
        }
    }
    const auto embd_gpu = graph.upload(q8_embd, "MTP generated Q8_0 token embeddings");
    fixture.owned.push_back(embd_gpu);
    auto cpu_embed = [&](uint32_t tok) {
        std::vector<float> row(c);
        dequantize(8u, std::span<const std::byte>(q8_embd).subspan(tok * 34u, 34u), row);
        return row;
    };

    std::array<std::vector<float>, 6> hc_rows;
    std::array<GpuView, 6> hc_gpu{};
    for (uint32_t i = 0; i < 6; ++i) {
        hc_rows[i] = fixture.values(h);
        hc_gpu[i] = fixture.upload(hc_rows[i]);
    }
    std::vector<float> prime2_hc(2u * h);
    std::copy(hc_rows[1].begin(), hc_rows[1].end(), prime2_hc.begin());
    std::copy(hc_rows[2].begin(), hc_rows[2].end(), prime2_hc.begin() + h);
    const auto prime2_hc_gpu = fixture.upload(prime2_hc);
    const auto before = graph.memory().live_bytes;
    {
        MtpHeadPolicy policy;
        policy.draft_attention = MtpDraftAttention::tiled128;
        MtpHead ref_head(graph, weights, shape, policy);
        MtpHead chain_head(graph, weights, shape, policy);
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        ref_head.prepare_priming_batch(4, 2ull << 20);
        chain_head.prepare_priming_batch(4, 2ull << 20);
#endif
        chain_head.configure_fast_round(true, {embd_gpu, 8u});
        check(chain_head.fast_round() && chain_head.can_greedy_chain() && !ref_head.can_greedy_chain(),
            "MTP greedy chain readiness state");
        const auto allocations = graph.memory().allocations;

        const SamplingPolicy greedy_sp{0.0, 0, 1.0, 0.0};
        SamplingRng rng_ref(123), rng_chain(123);
        const uint32_t anchor0 = 5u, anchor1 = 11u, anchor2 = 19u;

        // Round 1: depth-3 greedy chain in 1 command buffer vs 3 separate advance() calls!
        const auto r1_1 = ref_head.advance(hc_rows[0], cpu_embed(anchor0), 1);
        const auto d_ref_1 = draw_mtp_depth1(r1_1.logits, greedy_sp, rng_ref, vocab);
        const auto r1_2 = ref_head.advance(ref_head.hidden(), cpu_embed(d_ref_1.token), 2);
        const auto d_ref_2 = draw_mtp_depth1(r1_2.logits, greedy_sp, d_ref_1.random_after, vocab);
        const auto r1_3 = ref_head.advance(ref_head.hidden(), cpu_embed(d_ref_2.token), 3);
        const auto d_ref_3 = draw_mtp_depth1(r1_3.logits, greedy_sp, d_ref_2.random_after, vocab);
        const std::vector<float> ref_hidden_d3(ref_head.hidden().begin(), ref_head.hidden().end());
        ref_head.truncate(1);

        const auto cb_r1_before = graph.lifetime_command_buffers();
        const auto c1 = chain_head.advance_fast_greedy_chain(hc_gpu[0], cpu_embed(anchor0), 1, 3);
        const auto cb_r1 = graph.lifetime_command_buffers() - cb_r1_before;
        check(cb_r1 == 1 && c1.depth == 3 && c1.cache_rows == 3, "MTP depth-3 greedy chain must execute in 1 command buffer");
        const auto d_ch_1 = draw_mtp_depth1_topk(c1.topks[0], greedy_sp, rng_chain, vocab);
        const auto d_ch_2 = draw_mtp_depth1_topk(c1.topks[1], greedy_sp, d_ch_1.random_after, vocab);
        const auto d_ch_3 = draw_mtp_depth1_topk(c1.topks[2], greedy_sp, d_ch_2.random_after, vocab);
        const std::vector<float> ch_hidden_d3(chain_head.hidden().begin(), chain_head.hidden().end());
        check(d_ch_1.token == d_ref_1.token && d_ch_2.token == d_ref_2.token && d_ch_3.token == d_ref_3.token,
            "MTP depth-3 greedy chain token parity");
        check(ref_hidden_d3 == ch_hidden_d3, "MTP depth-3 greedy chain final_hc parity");
        chain_head.truncate(1);

        // Accept 2 drafts -> defer_prime_batch(2 rows) -> Round 2: depth-2 greedy chain in 1 command buffer (fusing prime + d1 + d2)!
        std::vector<float> prime2_emb(2u * c);
        const auto e1 = cpu_embed(d_ref_1.token), e2 = cpu_embed(d_ref_2.token);
        std::copy(e1.begin(), e1.end(), prime2_emb.begin());
        std::copy(e2.begin(), e2.end(), prime2_emb.begin() + c);

#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        static_cast<void>(ref_head.prime_batch(prime2_hc, prime2_emb, 2));
#else
        static_cast<void>(ref_head.prime_pair(hc_rows[1], e1, 2));
        static_cast<void>(ref_head.prime_pair(hc_rows[2], e2, 3));
#endif
        const auto r2_1 = ref_head.advance(hc_rows[3], cpu_embed(anchor1), 4);
        const auto d_ref_4 = draw_mtp_depth1(r2_1.logits, greedy_sp, d_ref_3.random_after, vocab);
        const auto r2_2 = ref_head.advance(ref_head.hidden(), cpu_embed(d_ref_4.token), 5);
        const auto d_ref_5 = draw_mtp_depth1(r2_2.logits, greedy_sp, d_ref_4.random_after, vocab);
        ref_head.truncate(4);

        chain_head.defer_prime_batch(prime2_hc_gpu, prime2_emb, 2);
        const auto cb_r2_before = graph.lifetime_command_buffers();
        const auto c2 = chain_head.advance_fast_greedy_chain_host(hc_rows[3], cpu_embed(anchor1), 4, 2);
        const auto cb_r2 = graph.lifetime_command_buffers() - cb_r2_before;
        check(cb_r2 == 1 && c2.depth == 2 && c2.cache_rows == 5,
            "MTP fused deferred-prime + depth-2 greedy chain must execute in 1 command buffer");
        const auto d_ch_4 = draw_mtp_depth1_topk(c2.topks[0], greedy_sp, d_ch_3.random_after, vocab);
        const auto d_ch_5 = draw_mtp_depth1_topk(c2.topks[1], greedy_sp, d_ch_4.random_after, vocab);
        check(d_ch_4.token == d_ref_4.token && d_ch_5.token == d_ref_5.token, "MTP depth-2 greedy chain token parity");
        chain_head.truncate(4);

        // Simulate first_stop early truncation on Round 3 (depth-3 chain -> truncate(cache_before + 1 = 5))!
        static_cast<void>(ref_head.advance(hc_rows[4], cpu_embed(anchor2), 5));
        static_cast<void>(chain_head.advance_fast_greedy_chain(hc_gpu[4], cpu_embed(anchor2), 5, 3));
        chain_head.truncate(5);
        const auto r4_ref = ref_head.advance(hc_rows[5], cpu_embed(3u), 6);
        const auto r4_ch = chain_head.advance_fast(hc_gpu[5], cpu_embed(3u), 6);
        check(draw_mtp_depth1(r4_ref.logits, greedy_sp, rng_ref, vocab).token ==
              draw_mtp_depth1_topk(r4_ch.topk, greedy_sp, rng_chain, vocab).token,
              "MTP greedy chain first_stop truncation continuation parity");
        check(graph.memory().allocations == allocations, "MTP greedy chain steady-state allocation invariance");
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
        std::vector<std::byte> ref_kv(ref_head.priming_cache_prefix_bytes()), ch_kv(chain_head.priming_cache_prefix_bytes());
        ref_head.read_priming_cache_prefix(ref_kv);
        chain_head.read_priming_cache_prefix(ch_kv);
        check(ref_kv == ch_kv, "MTP greedy chain KV cache prefix byte parity");
#endif
    }
    check(graph.memory().live_bytes == before, "MTP greedy chain workspace leaked");

    // Sticky GPU error checks on mtp_embed_from_topk_q8 (out-of-bounds token_id, nonfinite topk flag, nonfinite Q8_0 scale).
    {
        const auto out = graph.zeros(c * 4u, "MTP embed error test out");
        GpuLogitTopKRow bad_id{};
        bad_id.vocabulary = vocab; bad_id.count = vocab; bad_id.ids[0] = vocab;
        const auto row_bad_id = graph.upload(std::as_bytes(std::span(&bad_id, 1)), "MTP bad id topk");
        refused([&] {
            graph.begin();
            graph.mtp_embed_from_topk_q8(embd_gpu, row_bad_id, out, c, vocab);
            static_cast<void>(graph.submit());
        });

        GpuLogitTopKRow bad_nf{};
        bad_nf.vocabulary = vocab; bad_nf.count = vocab; bad_nf.first_nonfinite_col = 3u; bad_nf.ids[0] = 1u;
        const auto row_bad_nf = graph.upload(std::as_bytes(std::span(&bad_nf, 1)), "MTP nonfinite topk");
        refused([&] {
            graph.begin();
            graph.mtp_embed_from_topk_q8(embd_gpu, row_bad_nf, out, c, vocab);
            static_cast<void>(graph.submit());
        });

        auto bad_q8 = q8_embd;
        bad_q8[2u * 34u + 0u] = static_cast<std::byte>(0x00u);
        bad_q8[2u * 34u + 1u] = static_cast<std::byte>(0x7cu); // +inf FP16 scale
        const auto embd_nan = graph.upload(bad_q8, "MTP bad Q8_0 scale");
        GpuLogitTopKRow ok_row{};
        ok_row.vocabulary = vocab; ok_row.count = vocab; ok_row.ids[0] = 2u;
        const auto row_ok = graph.upload(std::as_bytes(std::span(&ok_row, 1)), "MTP ok topk");
        refused([&] {
            graph.begin();
            graph.mtp_embed_from_topk_q8(embd_nan, row_ok, out, c, vocab);
            static_cast<void>(graph.submit());
        });

        graph.release(row_ok);
        graph.release(embd_nan);
        graph.release(row_bad_nf);
        graph.release(row_bad_id);
        graph.release(out);
    }
}
void sampled_chain_parity(MetalGraph& graph) {
    Fixture fixture(graph);
    const MtpHeadShape shape{{4, 32, 8}, {32, 16, 5, 3}, 4, 2, 8, 4, 32, 32};
    const auto weights = fixture.weights(shape);
    const uint32_t c = shape.residual.channels, h = shape.residual.branches * c, vocab = shape.vocabulary;
    std::vector<std::byte> q8_embd(vocab * 34u);
    for (uint32_t tok = 0; tok < vocab; ++tok) {
        const uint16_t scale_f16 = 0x2c00u + uint16_t((tok % 7u) * 0x0040u);
        q8_embd[tok * 34u + 0u] = static_cast<std::byte>(scale_f16 & 0xffu);
        q8_embd[tok * 34u + 1u] = static_cast<std::byte>(scale_f16 >> 8u);
        for (uint32_t col = 0; col < 32u; ++col) {
            const int8_t q = int8_t(int((tok * 13u + col * 7u + 3u) % 61u) - 30);
            q8_embd[tok * 34u + 2u + col] = static_cast<std::byte>(static_cast<uint8_t>(q));
        }
    }
    const auto embd_gpu = graph.upload(q8_embd, "MTP sampled chain Q8_0 token embeddings");
    fixture.owned.push_back(embd_gpu);
    auto cpu_embed = [&](uint32_t tok) {
        std::vector<float> row(c);
        dequantize(8u, std::span<const std::byte>(q8_embd).subspan(tok * 34u, 34u), row);
        return row;
    };
    auto draw_law = [&](const GpuMtpSampledLawRow& row, const SamplingRng& rng_before) -> MtpDraftDraw {
        check(row.count >= 1 && row.count <= 20, "MTP sampled chain support count");
        std::array<uint64_t, 20> grid_units{};
        for (uint32_t j = 0; j < row.count; ++j) grid_units[j] = uint64_t(row.units24[j]) << 29u;
        auto law = SampledSparseLaw::from_grid_units(
            vocab, std::span(row.ids.data(), row.count), std::span(grid_units.data(), row.count));
        auto rng = rng_before;
        const auto begin = rng.words_consumed();
        const auto ticket = law.deterministic() ? uint64_t(0) : uint64_t(std::ldexp(rng.uniform(), 53));
        check(uint32_t(!law.deterministic()) == row.used_ticket, "MTP sampled chain used_ticket parity");
        const auto token = law.sample_at(ticket);
        check(token == row.chosen_token, "MTP sampled chain chosen_token parity");
        return {std::move(law), token, begin, std::move(rng)};
    };

    std::array<std::vector<float>, 4> hc_rows;
    std::array<GpuView, 4> hc_gpu{};
    for (uint32_t i = 0; i < 4; ++i) {
        hc_rows[i] = fixture.values(h);
        hc_gpu[i] = fixture.upload(hc_rows[i]);
    }
    const auto before = graph.memory().live_bytes;
    {
        MtpHeadPolicy policy;
        policy.draft_attention = MtpDraftAttention::tiled128;
        MtpHead chain_head(graph, weights, shape, policy);
        MtpHead ref_head(graph, weights, shape, policy);
        chain_head.configure_fast_round(true, {embd_gpu, 8u});
        const auto allocations = graph.memory().allocations;

        // Round 1: stochastic T=1.0, top_p=0.95, min_p=0.05 depth-3 chain in 1 command buffer!
        const SamplingPolicy sp{1.0, 0, 0.95, 0.05};
        SamplingRng rng(20261104ull);
        auto ticket_rng = rng;
        const std::array<uint64_t, 3> tickets{
            uint64_t(std::ldexp(ticket_rng.uniform(), 53)),
            uint64_t(std::ldexp(ticket_rng.uniform(), 53)),
            uint64_t(std::ldexp(ticket_rng.uniform(), 53)),
        };
        const auto cb_before = graph.lifetime_command_buffers();
        const auto sc = chain_head.advance_fast_sampled_chain(hc_gpu[0], cpu_embed(5u), 1, 3, sp, tickets);
        const auto cb_used = graph.lifetime_command_buffers() - cb_before;
        check(cb_used == 1 && sc.depth == 3 && sc.cache_rows == 3, "MTP depth-3 sampled chain must execute in 1 command buffer");
        const auto d1 = draw_law(sc.laws[0], rng);
        const auto d2 = draw_law(sc.laws[1], d1.random_after);
        const auto d3 = draw_law(sc.laws[2], d2.random_after);
        const std::vector<float> ch_hidden(chain_head.hidden().begin(), chain_head.hidden().end());

        // Verify exact teacher-forcing KV/hidden parity when ref_head advances on the chosen tokens!
        static_cast<void>(ref_head.advance(hc_rows[0], cpu_embed(5u), 1));
        static_cast<void>(ref_head.advance(ref_head.hidden(), cpu_embed(d1.token), 2));
        static_cast<void>(ref_head.advance(ref_head.hidden(), cpu_embed(d2.token), 3));
        const std::vector<float> ref_hidden(ref_head.hidden().begin(), ref_head.hidden().end());
        check(ch_hidden == ref_hidden, "MTP depth-3 sampled chain final_hc parity");
        chain_head.truncate(1);
        ref_head.truncate(1);

        // Round 2: sharp T=0.001, min_p=0.999999 (forces deterministic single-token support -> 0 tickets consumed!)
        const SamplingPolicy sharp_sp{0.001, 0, 1.0, 0.999999};
        auto sharp_trng = d3.random_after;
        const std::array<uint64_t, 3> sharp_tickets{
            uint64_t(std::ldexp(sharp_trng.uniform(), 53)),
            uint64_t(std::ldexp(sharp_trng.uniform(), 53)),
            uint64_t(std::ldexp(sharp_trng.uniform(), 53)),
        };
        const auto sc2 = chain_head.advance_fast_sampled_chain_host(hc_rows[1], cpu_embed(7u), 2, 2, sharp_sp, sharp_tickets);
        const auto s1 = draw_law(sc2.laws[0], d3.random_after);
        const auto s2 = draw_law(sc2.laws[1], s1.random_after);
        check(s1.law.deterministic() && s2.law.deterministic() &&
              s2.random_after.words_consumed() == d3.random_after.words_consumed(),
              "MTP deterministic sampled chain consumes zero entropy words");
        check(graph.memory().allocations == allocations, "MTP sampled chain steady-state allocation invariance");
    }
    check(graph.memory().live_bytes == before, "MTP sampled chain workspace leaked");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        tokspur::MetalGraph graph(argv[1], 256'000'000);
        primitives(graph); draft_attention_tiles(graph);
        heads(graph); heads(graph, MtpDraftAttention::tiled128); heads(graph, MtpDraftAttention::tiled256);
        q8_embedding_hidden_opt_in(graph); fast_round_parity(graph);
        greedy_chain_parity(graph);
        sampled_chain_parity(graph);
        draft_attention_screen(graph);
        std::cout << "MTP head generated checks passed: " << checks << '\n'; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
