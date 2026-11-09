#include "tokspur/attention_check.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
using namespace tokspur;
size_t checks = 0, batches = 0, compared = 0, boundary_differences = 0;
double worst = 0;
void require(bool b, const char* m) { ++checks; if (!b) throw std::runtime_error(m); }
void reject(const std::function<void()>& f) {
    ++checks; try { f(); } catch (const FormatError&) { return; }
    throw std::runtime_error("invalid attention operation accepted");
}
template<class T> std::vector<T> read(MetalGraph& g, GpuView v) {
    const auto data = g.read(v); require(data.size() % sizeof(T) == 0, "readback alignment");
    std::vector<T> out(data.size() / sizeof(T)); std::memcpy(out.data(), data.data(), data.size()); return out;
}
struct Fixture {
    MetalGraph& g; std::mt19937 rng{0x71736131};
    std::deque<std::vector<float>> data; std::deque<std::vector<double>> real_data; std::vector<GpuView> owned;
    ~Fixture() { try { g.abort(); } catch (const FormatError&) {} for (const auto v : owned) g.release(v); }
    std::vector<float> values(size_t n, float scale = 0.15f) { std::vector<float> out(n); for (auto& x : out) x = float(int(rng() % 2001) - 1000) / 1000 * scale; return out; }
    GpuView bytes(std::span<const std::byte> b, bool immutable = true) { const auto v = g.upload(b, "attention fixture", immutable); owned.push_back(v); return v; }
    GpuView floats(std::span<const float> x, bool immutable = true) { return bytes(std::as_bytes(x), immutable); }
    GpuView zeros(uint64_t n) { const auto v = g.zeros(n * 4, "attention output"); owned.push_back(v); return v; }
    std::pair<GpuView, std::span<const double>> vector(size_t n, float offset = 0) {
        data.push_back(values(n)); for (auto& x : data.back()) x += offset;
        real_data.emplace_back(data.back().begin(), data.back().end()); return {floats(data.back()), real_data.back()};
    }
    std::pair<GpuMatrix, ReferenceMatrix> matrix(size_t n) { const auto v = vector(n); return {{v.first, 0}, {std::as_bytes(std::span(data.back())), 0}}; }
    std::pair<ResidualWeights, ReferenceResidualWeights> residual(ResidualShape s) {
        const auto n = vector(s.branches * s.channels, 1); const auto d = matrix(s.branches * s.channels * s.rank), u = matrix(s.rank * s.channels * s.branches), i = matrix(s.branches * s.channels * s.branches);
        return {{n.first, d.first, u.first, i.first}, {n.second, d.second, u.second, i.second}};
    }
    std::pair<MoeWeights, ReferenceMoeWeights> moe(MoeShape s) {
        const auto r = matrix(s.channels * s.experts), g1 = matrix(s.channels * s.hidden * s.experts), u = matrix(s.channels * s.hidden * s.experts), d = matrix(s.hidden * s.channels * s.experts),
            sg = matrix(s.channels * s.hidden), su = matrix(s.channels * s.hidden), sd = matrix(s.hidden * s.channels), sr = matrix(s.channels);
        return {{r.first, g1.first, u.first, d.first, sg.first, su.first, sd.first, sr.first}, {r.second, g1.second, u.second, d.second, sg.second, su.second, sd.second, sr.second}};
    }
    std::pair<QsaWeights, ReferenceQsaWeights> attention(QsaShape s) {
        const auto q = matrix(s.channels * s.query_heads * s.head_width * 2), k = matrix(s.channels * s.kv_heads * s.head_width), v = matrix(s.channels * s.kv_heads * s.head_width),
            o = matrix(s.query_heads * s.head_width * s.channels), iq = matrix(s.channels * s.index_heads * s.index_width), ik = matrix(s.channels * s.index_width);
        const auto qn = vector(s.head_width, 1), kn = vector(s.head_width, 1), iqn = vector(s.index_width, 1), ikn = vector(s.index_width, 1);
        data.push_back(rotary_frequencies(s.rotary_width, 10000)); const auto frequency = floats(data.back());
        return {{q.first, k.first, v.first, o.first, iq.first, ik.first, qn.first, kn.first, iqn.first, ikn.first, frequency},
            {q.second, k.second, v.second, o.second, iq.second, ik.second, qn.second, kn.second, iqn.second, ikn.second, data.back()}};
    }
};
void selection(MetalGraph& g) {
    for (uint32_t n : {1, 17, 511, 512, 513, 1023, 1024, 1025, 16384, 65536}) for (uint32_t k : {1, 7, 64, 512}) {
        Fixture f{g, std::mt19937(0x71736131), {}, {}, {}};
        std::vector<float> scores(n); for (auto& x : scores) x = float(f.rng() % 29); // exact ties across chunks
        const auto input = f.floats(scores), ids = f.zeros(k); const auto slots = ((n + 511) / 512) * k;
        const SelectionScratch s{f.zeros(slots), f.zeros(slots), f.zeros(slots), f.zeros(slots)};
        const auto expected = select_blocks_reference(std::vector<double>(scores.begin(), scores.end()), k);
        const auto allocations = g.memory().allocations;
        g.begin(); g.select_blocks(input, ids, s, n, k); static_cast<void>(g.submit()); ++batches;
        auto actual = read<uint32_t>(g, ids); actual.resize(expected.size());
        require(actual == expected, "hierarchical top blocks differ"); require(g.memory().allocations == allocations, "selection allocated inside batch");
    }
}
void complete_layers(MetalGraph& g) {
    for (const AttentionLayerShape& shape : std::array<AttentionLayerShape, 2>{{{{4, 7, 3}, {7, 4, 2, 8, 2, 8, 4, 2, 4, 32}, {7, 9, 5, 3}},
        {{1, 1, 1}, {1, 1, 1, 2, 1, 2, 2, 4, 8, 24}, {1, 1, 1, 1}}}}) {
        Fixture f{g, std::mt19937(0x71736131), {}, {}, {}};
        const auto r1 = f.residual(shape.residual), r2 = f.residual(shape.residual); const auto a = f.attention(shape.attention); const auto m = f.moe(shape.moe);
        const AttentionLayerWeights w{r1.first, r2.first, a.first, m.first}; const ReferenceAttentionLayerWeights rw{r1.second, r2.second, a.second, m.second};
        const auto scratch = allocate_attention_layer_scratch(g, shape);
        const auto cache = allocate_qsa_cache(g, shape.attention);
        auto sum = f.zeros(shape.attention.index_width), next = f.zeros(shape.attention.index_width);
        const auto input = f.zeros(shape.residual.branches * shape.residual.channels), output = f.zeros(shape.residual.branches * shape.residual.channels);
        const auto allocations = g.memory().allocations;
        for (const auto packing : {QueryPacking::head_query_gate, QueryPacking::all_queries_then_gates}) for (const auto rotary : {RotaryPairs::split_half, RotaryPairs::adjacent})
        for (const auto mode : {RouterMode::softmax, RouterMode::sigmoid}) for (bool normalize : {false, true}) {
            const QsaPolicy policy{packing, rotary, 1e-6f};
            std::vector<double> cpu_sum(shape.attention.index_width, 0); const std::vector<float> zero(cpu_sum.size()); g.write(sum, std::as_bytes(std::span(zero)));
            ReferenceQsaCache reference{std::vector<uint16_t>(cache.keys.bytes / 2), std::vector<uint16_t>(cache.values.bytes / 2), std::vector<uint16_t>(cache.index_keys.bytes / 2)};
            for (uint32_t position = 0; position < 20; ++position) {
                const auto x = f.values(shape.residual.branches * shape.residual.channels, 1);
                g.write(input, std::as_bytes(std::span(x)));
                const auto accepted_k = position ? g.read(g.slice(cache.keys, 0, uint64_t(position) * shape.attention.kv_heads * shape.attention.head_width * 2)) : std::vector<std::byte>{};
                const auto accepted_sum = g.read(sum);
                const auto expected = attention_layer_reference(rw, reference, cpu_sum, std::vector<double>(x.begin(), x.end()), position, shape, policy, mode, normalize);
                g.begin(); encode_attention_layer(g, w, scratch, cache, sum, next, input, output, position, shape, policy, mode, normalize);
                const auto result = g.submit(); ++batches;
                require(result.dispatches == 28 + qsa_dispatches(shape.attention, position) && g.memory().allocations == allocations, "attention layer dispatch/allocation invariant");
                require(g.read(sum) == accepted_sum && (!position || g.read(g.slice(cache.keys, 0, accepted_k.size())) == accepted_k), "accepted attention cache/sum mutated");
                const auto comparison = compare_attention_layer(g, scratch, cache, next, output, position, shape, expected);
                compared += comparison.compared_values; boundary_differences += comparison.bf16_reference_boundary_differences; worst = std::max(worst, comparison.maximum);
                if (!comparison.passed) {
                    std::cerr << "position=" << position << " packing=" << uint32_t(packing) << " rotary=" << uint32_t(rotary) << " storage=" << comparison.storage_matches << " routes=" << comparison.routes_match << " selected=" << comparison.selected_blocks_match << '\n';
                    for (const auto& [name, error] : comparison.scaled_errors) if (error > 5e-5) std::cerr << name << ": " << error << '\n';
                }
                require(comparison.passed, "connected attention numerical comparison failed");
                cpu_sum = expected.attention.pooling.next_sum; std::swap(sum, next);
            }
        }
        const auto old = g.read(cache.keys), old_sum = g.read(sum), old_out = g.read(output);
        const QsaPolicy policy{QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f};
        g.begin(); encode_attention_layer(g, w, scratch, cache, sum, next, input, output, 20, shape, policy, RouterMode::softmax, true); g.abort();
        require(g.read(cache.keys) == old && g.read(sum) == old_sum && g.read(output) == old_out, "aborted attention executed writes");
        auto bad = w; bad.ffn.down.encoding = 0xffffffffu;
        g.begin(); reject([&] { encode_attention_layer(g, bad, scratch, cache, sum, next, input, output, 20, shape, policy, RouterMode::softmax, true); });
        require(g.read(cache.keys) == old, "late attention encode failure executed partial work");
        g.begin(); reject([&] { encode_attention_layer(g, w, scratch, cache, sum, sum, input, output, 20, shape, policy, RouterMode::softmax, true); });
        g.begin(); reject([&] { encode_attention_layer(g, w, scratch, cache, sum, next, input, output, shape.attention.capacity, shape, policy, RouterMode::softmax, true); });
        // Fail at a complete-block boundary after KV appends have executed.
        // Candidate rows may be dirty, but every accepted cache row stays exact.
        const uint32_t failure_position = 23;
        const auto key_bytes = uint64_t(failure_position) * shape.attention.kv_heads * shape.attention.head_width * 2;
        const auto index_bytes = uint64_t(failure_position / shape.attention.compression) * shape.attention.index_width * 2;
        const auto accepted_k = g.read(g.slice(cache.keys, 0, key_bytes)), accepted_v = g.read(g.slice(cache.values, 0, key_bytes)), accepted_i = g.read(g.slice(cache.index_keys, 0, index_bytes));
        bad = w; bad.attention.index_key_gains = f.floats(std::vector<float>(shape.attention.index_width, std::numeric_limits<float>::infinity()));
        g.begin(); encode_attention_layer(g, bad, scratch, cache, sum, next, input, output, failure_position, shape, policy, RouterMode::softmax, true);
        reject([&] { static_cast<void>(g.submit()); });
        require(g.read(g.slice(cache.keys, 0, key_bytes)) == accepted_k && g.read(g.slice(cache.values, 0, key_bytes)) == accepted_v &&
            g.read(g.slice(cache.index_keys, 0, index_bytes)) == accepted_i && g.read(sum) == old_sum, "failed device attention damaged accepted prefix");
        ReferenceQsaCache reuse_cache{read<uint16_t>(g, cache.keys), read<uint16_t>(g, cache.values), read<uint16_t>(g, cache.index_keys)};
        const auto reuse_x = read<float>(g, input), reuse_sum = read<float>(g, sum);
        const auto expected = attention_layer_reference(rw, reuse_cache, std::vector<double>(reuse_sum.begin(), reuse_sum.end()), std::vector<double>(reuse_x.begin(), reuse_x.end()), failure_position, shape, policy, RouterMode::softmax, true);
        g.begin(); encode_attention_layer(g, w, scratch, cache, sum, next, input, output, failure_position, shape, policy, RouterMode::softmax, true);
        static_cast<void>(g.submit()); ++batches;
        const auto reuse = compare_attention_layer(g, scratch, cache, next, output, failure_position, shape, expected);
        compared += reuse.compared_values; boundary_differences += reuse.bf16_reference_boundary_differences; worst = std::max(worst, reuse.maximum);
        require(reuse.passed, "valid attention reuse after device error failed");
        release_attention_layer_scratch(g, scratch); release_qsa_cache(g, cache);
    }
}
void rotary_storage(MetalGraph& g) {
    Fixture f{g, std::mt19937(1), {}, {}, {}};
    const auto x = f.values(256), freq_values = rotary_frequencies(64, 10000000);
    const auto input = f.floats(x), freq = f.floats(freq_values), output = f.zeros(x.size());
    for (auto pairs : {RotaryPairs::split_half, RotaryPairs::adjacent}) for (uint32_t pos : {0, 1, 2048, 59000, 65535}) {
        const auto expected = rotary_reference(std::vector<double>(x.begin(), x.end()), 1, 256, pos, freq_values, pairs);
        g.begin(); g.rotary(input, freq, output, 1, 256, pos, pairs); static_cast<void>(g.submit()); ++batches;
        const auto actual = read<float>(g, output);
        for (size_t i = 0; i < actual.size(); ++i) require(std::abs(actual[i] - expected[i]) < 1e-6, "long-position rotary phase mismatch");
    }
    const auto row = f.zeros(128);
    g.begin(); g.store_bf16(input, row); static_cast<void>(g.submit()); ++batches;
    const auto encoded = read<uint16_t>(g, row);
    for (size_t i = 0; i < encoded.size(); ++i) require(encoded[i] == finite_bf16(x[i]), "BF16 GPU rounding differs");
    const auto bad = f.floats(std::vector<float>(256, std::numeric_limits<float>::max()));
    g.begin(); g.store_bf16(bad, row); reject([&] { static_cast<void>(g.submit()); });
    g.begin(); g.store_bf16(input, row); static_cast<void>(g.submit()); ++batches;
}
void invalid_indices(MetalGraph& g) {
    Fixture f{g, std::mt19937(1), {}, {}, {}};
    const QsaShape shape{2, 2, 1, 2, 1, 2, 2, 2, 2, 8};
    std::vector<uint16_t> keys(16), values(16);
    for (uint32_t t = 0; t < 8; ++t) { values[t * 2] = finite_bf16(float(t)); values[t * 2 + 1] = finite_bf16(float(t * 2)); }
    const auto q = f.zeros(4), k = f.bytes(std::as_bytes(std::span(keys))), v = f.bytes(std::as_bytes(std::span(values))), ids = f.zeros(1), output = f.zeros(4);
    const std::array<uint32_t, 1> bad{0xffffffffu}, good{1};
    g.write(ids, std::as_bytes(std::span(bad)));
    g.begin(); g.sparse_attention(q, k, v, ids, output, 5, shape); reject([&] { static_cast<void>(g.submit()); });
    g.write(ids, std::as_bytes(std::span(good)));
    g.begin(); g.sparse_attention(q, k, v, ids, output, 5, shape); static_cast<void>(g.submit()); ++batches;
    require(read<float>(g, output) == std::vector<float>({3, 6, 3, 6}), "uniform sparse/tail device countercheck or reuse failed");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph g(argv[1], 256 * 1024 * 1024); const auto initial = g.memory().live_bytes;
        selection(g); complete_layers(g); rotary_storage(g); invalid_indices(g);
        require(g.memory().live_bytes == initial, "attention allocation leak");
        MetalGraph limited(argv[1], 131072); const auto base = limited.memory().live_bytes;
        reject([&] { static_cast<void>(allocate_attention_layer_scratch(limited, {{4, 2560, 320}, {2560, 24, 2, 256, 4, 128, 64, 4, 2048, 65536}, {2560, 640, 512, 10}})); });
        require(limited.memory().live_bytes == base, "partial attention allocation leaked");
        std::cout << "Owned attention checks passed: " << checks << ", compared values: " << compared << ", batches: " << batches << ", max scaled error: " << worst << ", BF16 reference boundary differences: " << boundary_differences << '\n';
    } catch (const std::exception& e) { std::cerr << "Owned attention check failed: " << e.what() << '\n'; return 1; }
}
