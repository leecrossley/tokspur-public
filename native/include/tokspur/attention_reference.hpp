#pragma once

#include "tokspur/attention.hpp"
#include "tokspur/recurrent_reference.hpp"

namespace tokspur {
struct ReferenceQsaWeights {
    ReferenceMatrix query_gate, key, value, output, index_query, index_key;
    std::span<const double> query_gains, key_gains, index_query_gains, index_key_gains;
    std::span<const float> frequencies;
};
struct ReferenceQsaCache { std::vector<uint16_t> keys, values, index_keys; };
struct ReferenceQsa {
    std::vector<double> packed, raw_key, raw_value, normalized_query, normalized_key, rotary_query, rotary_key;
    QueryGate split;
    std::vector<double> raw_index_query, raw_index_key, normalized_index_query, rotary_index_query;
    IndexStep pooling;
    std::vector<double> normalized_index_key, rotary_index_key, scores, attention, gated, output;
    std::vector<uint32_t> selected;
};
// Diagnostic cache only: writes new rows, never accepted rows. Not a runtime
// fallback. BF16 boundaries are explicit, double accumulation elsewhere.
ReferenceQsa qsa_reference(const ReferenceQsaWeights& weights, ReferenceQsaCache& cache, std::span<const double> sum,
                            std::span<const double> input, uint32_t position, QsaShape shape, QsaPolicy policy);
struct ReferenceAttentionLayerWeights { ReferenceResidualWeights attention_read, ffn_read; ReferenceQsaWeights attention; ReferenceMoeWeights ffn; };
struct ReferenceAttentionLayer {
    ReferenceResidual attention_read, ffn_read;
    ReferenceQsa attention;
    ReferenceMoe ffn;
    std::vector<double> intermediate_streams, output_streams;
};
ReferenceAttentionLayer attention_layer_reference(const ReferenceAttentionLayerWeights& weights, ReferenceQsaCache& cache,
                                                   std::span<const double> sum, std::span<const double> streams, uint32_t position,
                                                   AttentionLayerShape shape, QsaPolicy policy, RouterMode mode, bool normalize);
} // namespace tokspur
