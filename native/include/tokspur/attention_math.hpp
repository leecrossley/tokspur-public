#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace tokspur {

enum class QueryPacking : uint32_t { head_query_gate, all_queries_then_gates };
enum class RotaryPairs : uint32_t { split_half, adjacent };
struct QsaShape {
    uint32_t channels, query_heads, kv_heads, head_width, index_heads, index_width;
    uint32_t rotary_width, compression, token_budget, capacity;
};
struct QsaPolicy { QueryPacking packing; RotaryPairs rotary_pairs; float rms_epsilon; };
void validate_qsa_shape(QsaShape shape);
void validate_qsa_policy(QsaPolicy policy);
uint32_t qsa_block_capacity(QsaShape shape);
uint32_t qsa_block_budget(QsaShape shape);
// CPU-visible physical schedule, shared by the encoder and protocol admission.
uint64_t qsa_dispatches(QsaShape shape, uint32_t position);

// Declared storage/phase boundaries: BF16 round-to-nearest-even after FP32;
// rotary angles are FP32 products of FP32 inverse frequencies and positions.
uint16_t finite_bf16(float value);
float read_bf16(uint16_t bits);
std::vector<float> rotary_frequencies(uint32_t rotary_width, double base);
std::vector<double> rotary_reference(std::span<const double> input, uint32_t heads, uint32_t width,
                                      uint32_t position, std::span<const float> frequencies, RotaryPairs pairs);
struct QueryGate { std::vector<double> query, gate; };
QueryGate split_query_reference(std::span<const double> input, uint32_t heads, uint32_t width, QueryPacking packing);

struct IndexStep { std::vector<double> next_sum, pooled; bool complete; };
IndexStep index_accumulate_reference(std::span<const double> sum, std::span<const double> key,
                                      uint32_t position, uint32_t compression);
std::vector<double> index_scores_reference(std::span<const double> query, std::span<const uint16_t> keys,
                                            uint32_t heads, uint32_t width, uint32_t blocks);
std::vector<uint32_t> select_blocks_reference(std::span<const double> scores, uint32_t budget);
std::vector<double> sparse_attention_reference(std::span<const double> query, std::span<const uint16_t> keys,
                                                std::span<const uint16_t> values, std::span<const uint32_t> selected,
                                                uint32_t prefix, QsaShape shape);

} // namespace tokspur
