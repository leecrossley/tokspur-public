#pragma once

#include "tokspur/recurrent.hpp"

namespace tokspur {

struct QsaWeights {
    GpuMatrix query_gate, key, value, output, index_query, index_key;
    GpuView query_gains, key_gains, index_query_gains, index_key_gains, frequencies;
};
// Append-only BF16 arrays. Accepted rows [0,position) are never written by the
// token assembler; candidates may dirty only the new row/block. Prefix length
// and all-layer publication remain the caller's responsibility, not this slice.
struct QsaCache {
    GpuView keys, values, index_keys;
    [[nodiscard]] std::array<GpuView, 3> views() const;
};
struct QsaScratch {
    GpuView packed, raw_key, raw_value, query, gate, normalized_query, normalized_key, rotary_query, rotary_key;
    GpuView raw_index_query, raw_index_key, normalized_index_query, rotary_index_query, pooled, normalized_index_key, rotary_index_key;
    GpuView scores;
    SelectionScratch selection;
    GpuView selected_ids, attention, gated;
    [[nodiscard]] std::array<GpuView, 24> views() const;
};
QsaCache allocate_qsa_cache(MetalGraph& graph, QsaShape shape);
void release_qsa_cache(MetalGraph& graph, const QsaCache& cache);
QsaScratch allocate_qsa_scratch(MetalGraph& graph, QsaShape shape);
void release_qsa_scratch(MetalGraph& graph, const QsaScratch& scratch);
uint64_t qsa_dispatches(QsaShape shape, uint32_t position);
void encode_qsa(MetalGraph& graph, const QsaWeights& weights, const QsaScratch& scratch, const QsaCache& cache,
                 GpuView current_index_sum, GpuView next_index_sum, GpuView input, GpuView output,
                 uint32_t position, QsaShape shape, QsaPolicy policy);

struct AttentionLayerShape { ResidualShape residual; QsaShape attention; MoeShape moe; };
struct AttentionLayerWeights { ResidualWeights attention_read, ffn_read; QsaWeights attention; MoeWeights ffn; };
struct AttentionLayerScratch {
    ResidualScratch residual; QsaScratch attention; MoeScratch moe;
    GpuView intermediate_streams, attention_output, ffn_output;
    [[nodiscard]] std::array<GpuView, 43> views() const;
};
AttentionLayerScratch allocate_attention_layer_scratch(MetalGraph& graph, AttentionLayerShape shape);
void release_attention_layer_scratch(MetalGraph& graph, const AttentionLayerScratch& scratch);
void encode_attention_layer(MetalGraph& graph, const AttentionLayerWeights& weights, const AttentionLayerScratch& scratch,
                             const QsaCache& cache, GpuView current_index_sum, GpuView next_index_sum,
                             GpuView input_streams, GpuView output_streams, uint32_t position, AttentionLayerShape shape,
                             QsaPolicy policy, RouterMode mode, bool normalize_selected
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                             , uint32_t route_row_offset = 0
#endif
                             );

} // namespace tokspur
