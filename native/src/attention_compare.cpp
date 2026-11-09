#include "tokspur/attention_check.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tokspur {
namespace {
template<class T> std::vector<T> read(MetalGraph& g, GpuView v) {
    const auto data = g.read(v);
    if (data.size() % sizeof(T)) throw FormatError("unaligned attention diagnostic readback");
    std::vector<T> out(data.size() / sizeof(T)); std::memcpy(out.data(), data.data(), data.size()); return out;
}
}
AttentionComparison compare_attention_layer(MetalGraph& g, const AttentionLayerScratch& s, const QsaCache& cache,
                                              GpuView next_sum, GpuView output, uint32_t position, AttentionLayerShape shape,
                                              const ReferenceAttentionLayer& e) {
    AttentionComparison out{{}, false, true, true, 0, 0, 0, false};
    const auto check = [&](const char* name, GpuView view, std::span<const double> expected) {
        const auto actual = read<float>(g, view);
        if (actual.size() != expected.size()) throw FormatError("attention diagnostic shape mismatch");
        double maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) throw FormatError("nonfinite attention diagnostic");
            maximum = std::max(maximum, std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i])));
        }
        out.compared_values += actual.size(); out.maximum = std::max(out.maximum, maximum); out.scaled_errors.emplace_back(name, maximum);
    };
    check("output_streams", output, e.output_streams); check("intermediate_streams", s.intermediate_streams, e.intermediate_streams);
    check("attention_output", s.attention_output, e.attention.output); check("ffn_output", s.ffn_output, e.ffn.output);
    check("ffn_read_normalized", s.residual.normalized, e.ffn_read.normalized); check("ffn_read_bottleneck", s.residual.bottleneck, e.ffn_read.bottleneck);
    check("ffn_read_scores", s.residual.scores, e.ffn_read.scores); check("ffn_read_inject", s.residual.inject_logits, e.ffn_read.inject_logits);
    check("ffn_read_input", s.residual.input, e.ffn_read.read.input); check("ffn_read_scales", s.residual.scales, e.ffn_read.read.write_scales);
    const auto& a = s.attention; const auto& r = e.attention;
    check("packed_query_gate", a.packed, r.packed); check("query", a.query, r.split.query); check("gate", a.gate, r.split.gate);
    check("raw_key", a.raw_key, r.raw_key); check("raw_value", a.raw_value, r.raw_value);
    check("normalized_query", a.normalized_query, r.normalized_query); check("normalized_key", a.normalized_key, r.normalized_key);
    check("rotary_query", a.rotary_query, r.rotary_query); check("rotary_key", a.rotary_key, r.rotary_key);
    check("raw_index_query", a.raw_index_query, r.raw_index_query); check("raw_index_key", a.raw_index_key, r.raw_index_key);
    check("normalized_index_query", a.normalized_index_query, r.normalized_index_query); check("rotary_index_query", a.rotary_index_query, r.rotary_index_query);
    check("pooled", a.pooled, r.pooling.pooled); check("next_index_sum", next_sum, r.pooling.next_sum);
    if (r.pooling.complete) { check("normalized_index_key", a.normalized_index_key, r.normalized_index_key); check("rotary_index_key", a.rotary_index_key, r.rotary_index_key); }
    const auto& sh = shape.attention;
    const auto blocks = (position + 1) / sh.compression;
    if (blocks > qsa_block_budget(sh)) {
        check("index_scores", g.slice(a.scores, 0, uint64_t(blocks) * 4), r.scores);
        out.selected_blocks_match = read<uint32_t>(g, a.selected_ids) == r.selected;
    }
    check("sparse_attention", a.attention, r.attention); check("gated_attention", a.gated, r.gated);
    check("moe_logits", s.moe.logits, e.ffn.logits); check("moe_route_weights", s.moe.weights, e.ffn.routes.weights);
    check("moe_activated", s.moe.gate, e.ffn.activated); check("moe_routed", s.moe.down, e.ffn.routed);
    check("moe_shared_activated", s.moe.shared_gate, e.ffn.shared_activated); check("moe_shared_output", s.moe.shared_down, e.ffn.shared_output);
    out.routes_match = read<uint32_t>(g, s.moe.indices) == e.ffn.routes.indices;
    const auto storage = [&](GpuView source, GpuView row, std::span<const double> ideal) {
        const auto actual = read<float>(g, source); const auto encoded = read<uint16_t>(g, row);
        if (actual.size() != encoded.size() || ideal.size() != actual.size()) throw FormatError("attention storage diagnostic shape mismatch");
        for (size_t i = 0; i < actual.size(); ++i) {
            out.storage_matches = out.storage_matches && finite_bf16(actual[i]) == encoded[i];
            out.bf16_reference_boundary_differences += finite_bf16(float(ideal[i])) != encoded[i];
        }
    };
    const uint64_t row_bytes = uint64_t(sh.kv_heads) * sh.head_width * 2;
    storage(a.rotary_key, g.slice(cache.keys, position * row_bytes, row_bytes), r.rotary_key);
    storage(a.raw_value, g.slice(cache.values, position * row_bytes, row_bytes), r.raw_value);
    if (r.pooling.complete) storage(a.rotary_index_key, g.slice(cache.index_keys, uint64_t(blocks - 1) * sh.index_width * 2, sh.index_width * 2), r.rotary_index_key);
    out.passed = out.routes_match && out.selected_blocks_match && out.storage_matches && out.maximum <= 5e-5;
    return out;
}
} // namespace tokspur
