#include "tokspur/attention.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>

namespace tokspur {
namespace {
template<size_t N> std::array<GpuView, N> allocate(MetalGraph& g, const std::array<uint64_t, N>& counts, uint64_t width = 4) {
    std::array<GpuView, N> out{}; size_t made = 0;
    try { for (; made < N; ++made) out[made] = g.zeros(checked_mul(counts[made], width), "sparse attention workspace"); }
    catch (...) { for (size_t i = 0; i < made; ++i) g.release(out[i]); throw; }
    return out;
}
void abort(MetalGraph& g) { try { g.abort(); } catch (const FormatError&) {} }
std::array<GpuView, 11> weights(const QsaWeights& w) {
    return {w.query_gate.view, w.key.view, w.value.view, w.output.view, w.index_query.view, w.index_key.view,
        w.query_gains, w.key_gains, w.index_query_gains, w.index_key_gains, w.frequencies};
}
void layer_shape(AttentionLayerShape s) {
    validate_residual_shape(s.residual); validate_qsa_shape(s.attention);
    if (s.residual.channels != s.attention.channels || s.moe.channels != s.attention.channels || !s.moe.hidden || s.moe.hidden > 65536 ||
        !s.moe.experts || s.moe.experts > 4096 || !s.moe.selected || s.moe.selected > 64 || s.moe.selected > s.moe.experts) throw FormatError("inconsistent attention layer shape");
}
}
std::array<GpuView, 3> QsaCache::views() const { return {keys, values, index_keys}; }
std::array<GpuView, 24> QsaScratch::views() const {
    return {packed, raw_key, raw_value, query, gate, normalized_query, normalized_key, rotary_query, rotary_key, raw_index_query,
        raw_index_key, normalized_index_query, rotary_index_query, pooled, normalized_index_key, rotary_index_key, scores,
        selection.scores_a, selection.ids_a, selection.scores_b, selection.ids_b, selected_ids, attention, gated};
}
std::array<GpuView, 43> AttentionLayerScratch::views() const {
    std::array<GpuView, 43> out;
    const auto r = residual.views(); auto at = std::copy(r.begin(), r.end(), out.begin());
    const auto a = attention.views(); at = std::copy(a.begin(), a.end(), at);
    const auto m = moe.views(); at = std::copy(m.begin(), m.end(), at);
    *at++ = intermediate_streams; *at++ = attention_output; *at = ffn_output; return out;
}
QsaCache allocate_qsa_cache(MetalGraph& g, QsaShape s) {
    validate_qsa_shape(s);
    const uint64_t kv = uint64_t(s.capacity) * s.kv_heads * s.head_width, index = uint64_t(qsa_block_capacity(s)) * s.index_width;
    const auto v = allocate(g, std::array{kv, kv, index}, 2); return {v[0], v[1], v[2]};
}
void release_qsa_cache(MetalGraph& g, const QsaCache& c) { for (const auto v : c.views()) g.release(v); }
QsaScratch allocate_qsa_scratch(MetalGraph& g, QsaShape s) {
    validate_qsa_shape(s);
    const uint64_t q = s.query_heads * s.head_width, k = s.kv_heads * s.head_width, iq = s.index_heads * s.index_width, ik = s.index_width;
    const uint64_t blocks = qsa_block_capacity(s), budget = qsa_block_budget(s), slots = ((blocks + 511) / 512) * budget;
    const auto v = allocate(g, std::array<uint64_t, 24>{q * 2, k, k, q, q, q, k, q, k, iq, ik, iq, iq, ik, ik, ik, blocks, slots, slots, slots, slots, budget, q, q});
    return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16],
        {v[17], v[18], v[19], v[20]}, v[21], v[22], v[23]};
}
void release_qsa_scratch(MetalGraph& g, const QsaScratch& s) { for (const auto v : s.views()) g.release(v); }
void encode_qsa(MetalGraph& g, const QsaWeights& w, const QsaScratch& s, const QsaCache& c,
                 GpuView sum, GpuView next_sum, GpuView input, GpuView output, uint32_t position, QsaShape shape, QsaPolicy p) {
    try {
        validate_qsa_shape(shape); validate_qsa_policy(p);
        if (position >= shape.capacity || c.keys.bytes != uint64_t(shape.capacity) * shape.kv_heads * shape.head_width * 2 ||
            c.values.bytes != c.keys.bytes || c.index_keys.bytes != uint64_t(qsa_block_capacity(shape)) * shape.index_width * 2 ||
            sum.bytes != uint64_t(shape.index_width) * 4 || next_sum.bytes != sum.bytes || w.frequencies.bytes != uint64_t(shape.rotary_width) * 2) throw FormatError("sparse cache/prefix geometry mismatch");
        std::array<GpuView, 13> reads{input, sum}; const auto wv = weights(w); std::copy(wv.begin(), wv.end(), reads.begin() + 2);
        const auto cv = c.views(); g.validate_disjoint(reads, cv);
        std::array<GpuView, 16> protected_reads{input, sum};
        auto at = std::copy(wv.begin(), wv.end(), protected_reads.begin() + 2); std::copy(cv.begin(), cv.end(), at);
        std::array<GpuView, 26> outputs{output, next_sum}; const auto sv = s.views(); std::copy(sv.begin(), sv.end(), outputs.begin() + 2);
        g.validate_disjoint(protected_reads, outputs);
        const auto q = shape.query_heads * shape.head_width, k = shape.kv_heads * shape.head_width, iq = shape.index_heads * shape.index_width;
        g.gemv(w.query_gate.encoding, w.query_gate.view, input, s.packed, shape.channels, q * 2);
        g.gemv(w.key.encoding, w.key.view, input, s.raw_key, shape.channels, k);
        g.gemv(w.value.encoding, w.value.view, input, s.raw_value, shape.channels, k);
        g.gemv(w.index_query.encoding, w.index_query.view, input, s.raw_index_query, shape.channels, iq);
        g.gemv(w.index_key.encoding, w.index_key.view, input, s.raw_index_key, shape.channels, shape.index_width);
        g.split_query_gate(s.packed, s.query, s.gate, shape.query_heads, shape.head_width, p.packing);
        g.rms(s.query, w.query_gains, s.normalized_query, shape.head_width, p.rms_epsilon);
        g.rms(s.raw_key, w.key_gains, s.normalized_key, shape.head_width, p.rms_epsilon);
        g.rms(s.raw_index_query, w.index_query_gains, s.normalized_index_query, shape.index_width, p.rms_epsilon);
        g.rotary(s.normalized_query, w.frequencies, s.rotary_query, shape.query_heads, shape.head_width, position, p.rotary_pairs);
        g.rotary(s.normalized_key, w.frequencies, s.rotary_key, shape.kv_heads, shape.head_width, position, p.rotary_pairs);
        g.rotary(s.normalized_index_query, w.frequencies, s.rotary_index_query, shape.index_heads, shape.index_width, position, p.rotary_pairs);
        g.store_bf16(s.rotary_key, g.slice(c.keys, uint64_t(position) * k * 2, k * 2));
        g.store_bf16(s.raw_value, g.slice(c.values, uint64_t(position) * k * 2, k * 2));
        g.index_accumulate(sum, s.raw_index_key, next_sum, s.pooled, position, shape.compression);
        const auto blocks = (position + 1) / shape.compression;
        if ((position + 1) % shape.compression == 0) {
            g.rms(s.pooled, w.index_key_gains, s.normalized_index_key, shape.index_width, p.rms_epsilon);
            g.rotary(s.normalized_index_key, w.frequencies, s.rotary_index_key, 1, shape.index_width, position + 1 - shape.compression, p.rotary_pairs);
            g.store_bf16(s.rotary_index_key, g.slice(c.index_keys, uint64_t(blocks - 1) * shape.index_width * 2, shape.index_width * 2));
        }
        if (blocks > qsa_block_budget(shape)) {
            g.index_scores(s.rotary_index_query, c.index_keys, s.scores, shape.index_heads, shape.index_width, blocks);
            g.select_blocks(s.scores, s.selected_ids, s.selection, blocks, qsa_block_budget(shape));
        }
        g.sparse_attention(s.rotary_query, c.keys, c.values, s.selected_ids, s.attention, position + 1, shape);
        g.multiply(s.attention, s.gate, s.gated, true);
        g.gemv(w.output.encoding, w.output.view, s.gated, output, q, shape.channels);
    } catch (...) { abort(g); throw; }
}
AttentionLayerScratch allocate_attention_layer_scratch(MetalGraph& g, AttentionLayerShape s) {
    layer_shape(s); AttentionLayerScratch out{};
    try {
        out.residual = allocate_residual_scratch(g, s.residual); out.attention = allocate_qsa_scratch(g, s.attention); out.moe = allocate_moe_scratch(g, s.moe);
        const auto v = allocate(g, std::array<uint64_t, 3>{uint64_t(s.residual.branches) * s.residual.channels, s.residual.channels, s.residual.channels});
        out.intermediate_streams = v[0]; out.attention_output = v[1]; out.ffn_output = v[2];
    } catch (...) { for (const auto v : out.views()) if (v.owner) g.release(v); throw; }
    return out;
}
void release_attention_layer_scratch(MetalGraph& g, const AttentionLayerScratch& s) { for (const auto v : s.views()) g.release(v); }
void encode_attention_layer(MetalGraph& g, const AttentionLayerWeights& w, const AttentionLayerScratch& s, const QsaCache& c,
                             GpuView sum, GpuView next_sum, GpuView input, GpuView output, uint32_t position,
                             AttentionLayerShape shape, QsaPolicy policy, RouterMode mode, bool normalize
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                             , uint32_t route_row_offset
#endif
                             ) {
    try {
        layer_shape(shape); validate_qsa_policy(policy);
        if (uint32_t(mode) > 1) throw FormatError("invalid attention layer router policy");
        std::array<GpuView, 29> reads{input, sum, w.attention_read.gains, w.attention_read.down.view, w.attention_read.up.view, w.attention_read.inject.view,
            w.ffn_read.gains, w.ffn_read.down.view, w.ffn_read.up.view, w.ffn_read.inject.view, w.ffn.router.view, w.ffn.gate.view, w.ffn.up.view,
            w.ffn.down.view, w.ffn.shared_gate.view, w.ffn.shared_up.view, w.ffn.shared_down.view, w.ffn.shared_router.view};
        const auto wv = weights(w.attention); std::copy(wv.begin(), wv.end(), reads.begin() + 18);
        const auto cv = c.views(); g.validate_disjoint(reads, cv);
        std::array<GpuView, 32> protected_reads; auto at = std::copy(reads.begin(), reads.end(), protected_reads.begin()); std::copy(cv.begin(), cv.end(), at);
        std::array<GpuView, 45> outputs{output, next_sum}; const auto sv = s.views(); std::copy(sv.begin(), sv.end(), outputs.begin() + 2);
        g.validate_disjoint(protected_reads, outputs);
        encode_residual_read(g, w.attention_read, s.residual, input, shape.residual, policy.rms_epsilon);
        encode_qsa(g, w.attention, s.attention, c, sum, next_sum, s.residual.input, s.attention_output, position, shape.attention, policy);
        g.residual_write(input, s.attention_output, s.residual.scales, s.intermediate_streams);
        encode_residual_read(g, w.ffn_read, s.residual, s.intermediate_streams, shape.residual, policy.rms_epsilon);
        encode_moe(g, w.ffn, s.moe, s.residual.input, s.ffn_output, shape.moe, mode, normalize
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
            , route_row_offset
#endif
            );
        g.residual_write(s.intermediate_streams, s.ffn_output, s.residual.scales, output);
    } catch (...) { abort(g); throw; }
}
} // namespace tokspur
