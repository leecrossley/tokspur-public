#include "tokspur/attention_reference.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <numeric>

namespace tokspur {
namespace {
std::vector<double> project(ReferenceMatrix m, std::span<const double> x, uint32_t rows) {
    if (x.empty() || x.size() > 65536) throw FormatError("attention reference projection width limit");
    return reference_gemv_doubleinput(m.encoding, m.bytes, static_cast<uint32_t>(x.size()), rows, x);
}
std::vector<double> norm(std::span<const double> input, std::span<const double> gain, uint32_t width, double epsilon) {
    if (gain.size() != width) throw FormatError("attention gain shape mismatch");
    std::vector<double> repeated(input.size());
    for (size_t i = 0; i < repeated.size(); ++i) repeated[i] = gain[i % width];
    return group_rms_reference(input, repeated, width, epsilon);
}
void store(std::vector<uint16_t>& out, size_t offset, std::span<const double> input) {
    if (offset > out.size() || input.size() > out.size() - offset) throw FormatError("reference BF16 append span mismatch");
    for (size_t i = 0; i < input.size(); ++i) out[offset + i] = finite_bf16(float(input[i]));
}
}
ReferenceQsa qsa_reference(const ReferenceQsaWeights& w, ReferenceQsaCache& cache, std::span<const double> sum,
                            std::span<const double> x, uint32_t position, QsaShape s, QsaPolicy p) {
    validate_qsa_shape(s); validate_qsa_policy(p);
    const auto kv = s.kv_heads * s.head_width;
    if (position >= s.capacity || x.size() != s.channels || cache.keys.size() != uint64_t(s.capacity) * kv || cache.values.size() != cache.keys.size() ||
        cache.index_keys.size() != uint64_t(qsa_block_capacity(s)) * s.index_width || w.frequencies.size() != s.rotary_width / 2) throw FormatError("attention reference cache/input mismatch");
    ReferenceQsa out;
    out.packed = project(w.query_gate, x, s.query_heads * s.head_width * 2);
    out.raw_key = project(w.key, x, kv); out.raw_value = project(w.value, x, kv);
    out.raw_index_query = project(w.index_query, x, s.index_heads * s.index_width); out.raw_index_key = project(w.index_key, x, s.index_width);
    out.split = split_query_reference(out.packed, s.query_heads, s.head_width, p.packing);
    out.normalized_query = norm(out.split.query, w.query_gains, s.head_width, p.rms_epsilon);
    out.normalized_key = norm(out.raw_key, w.key_gains, s.head_width, p.rms_epsilon);
    out.normalized_index_query = norm(out.raw_index_query, w.index_query_gains, s.index_width, p.rms_epsilon);
    out.rotary_query = rotary_reference(out.normalized_query, s.query_heads, s.head_width, position, w.frequencies, p.rotary_pairs);
    out.rotary_key = rotary_reference(out.normalized_key, s.kv_heads, s.head_width, position, w.frequencies, p.rotary_pairs);
    out.rotary_index_query = rotary_reference(out.normalized_index_query, s.index_heads, s.index_width, position, w.frequencies, p.rotary_pairs);
    store(cache.keys, size_t(position) * kv, out.rotary_key); store(cache.values, size_t(position) * kv, out.raw_value);
    out.pooling = index_accumulate_reference(sum, out.raw_index_key, position, s.compression);
    const auto blocks = (position + 1) / s.compression;
    if (out.pooling.complete) {
        out.normalized_index_key = norm(out.pooling.pooled, w.index_key_gains, s.index_width, p.rms_epsilon);
        out.rotary_index_key = rotary_reference(out.normalized_index_key, 1, s.index_width, position + 1 - s.compression, w.frequencies, p.rotary_pairs);
        store(cache.index_keys, size_t(blocks - 1) * s.index_width, out.rotary_index_key);
    }
    if (blocks > qsa_block_budget(s)) {
        out.scores = index_scores_reference(out.rotary_index_query, cache.index_keys, s.index_heads, s.index_width, blocks);
        out.selected = select_blocks_reference(out.scores, qsa_block_budget(s));
    } else { out.selected.resize(blocks); std::iota(out.selected.begin(), out.selected.end(), 0); }
    out.attention = sparse_attention_reference(out.rotary_query, cache.keys, cache.values, out.selected, position + 1, s);
    out.gated = out.attention;
    for (size_t i = 0; i < out.gated.size(); ++i) out.gated[i] *= stable_sigmoid(out.split.gate[i]);
    out.output = project(w.output, out.gated, s.channels);
    return out;
}
ReferenceAttentionLayer attention_layer_reference(const ReferenceAttentionLayerWeights& w, ReferenceQsaCache& cache,
                                                   std::span<const double> sum, std::span<const double> streams, uint32_t position,
                                                   AttentionLayerShape s, QsaPolicy p, RouterMode mode, bool normalize) {
    validate_qsa_shape(s.attention); validate_qsa_policy(p);
    if (!s.residual.branches || s.residual.branches > 64 || !s.residual.channels || s.residual.channels > 65536 || !s.residual.rank || s.residual.rank > 4096 ||
        s.residual.channels != s.attention.channels || s.moe.channels != s.attention.channels || !s.moe.hidden || s.moe.hidden > 65536 ||
        !s.moe.experts || s.moe.experts > 4096 || !s.moe.selected || s.moe.selected > 64 || s.moe.selected > s.moe.experts) throw FormatError("reference attention-layer shape mismatch");
    ReferenceAttentionLayer out;
    out.attention_read = residual_projection_reference(w.attention_read, streams, s.residual, p.rms_epsilon);
    out.attention = qsa_reference(w.attention, cache, sum, out.attention_read.read.input, position, s.attention, p);
    out.intermediate_streams = residual_write_reference(streams, out.attention.output, out.attention_read.read.write_scales);
    out.ffn_read = residual_projection_reference(w.ffn_read, out.intermediate_streams, s.residual, p.rms_epsilon);
    out.ffn = moe_projection_reference(w.ffn, out.ffn_read.read.input, s.moe, mode, normalize);
    out.output_streams = residual_write_reference(out.intermediate_streams, out.ffn.output, out.ffn_read.read.write_scales);
    return out;
}
} // namespace tokspur
