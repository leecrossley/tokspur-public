#pragma once

#include "tokspur/graph.hpp"
#include "tokspur/attention_math.hpp"
#include "tokspur/qsa_prefill_ops.hpp"

namespace tokspur {
inline constexpr uint32_t qsa_prefill_capacity = 512, qsa_prefill_query_stripe = 32, qsa_prefill_token_tile = 128;
struct QsaPrefillPlan {
    uint32_t base, rows, capacity, first_pool, pool_count, final_remainder;
    uint32_t block_capacity, block_budget, selection_stride, maximum_attended, attention_tiles, stripe_rows;
    uint64_t key_append_offset, key_append_bytes, index_append_offset, index_append_bytes;
    uint64_t index_scores_bytes, selection_buffer_bytes, selected_ids_bytes, attention_partials_bytes;
};
QsaPrefillPlan qsa_prefill_plan(uint32_t base, uint32_t rows, uint32_t capacity, QsaShape shape, bool allow_nax_2k = false);
uint32_t qsa_nax_indexer_stripe(uint32_t base, uint32_t rows, uint32_t block_budget);
uint64_t qsa_nax_attention_dispatches(uint32_t base, uint32_t rows, QsaShape shape);
// Projected core includes all normalisation, rotation, cache append, pooling, selection, attention and sigmoid-gate dispatches. It excludes projections, HC/residual/FFN, submissions and host copies.
uint64_t qsa_prefill_projected_dispatches(uint32_t base, uint32_t rows, QsaShape shape, bool allow_nax_2k = false, bool nax_qsa = false, bool fused_prep = false);
// Complete QSA adds two owned BF16 indexer dispatches and four caller-owned dense Q8 dispatches (query/gate, key, value and output), each admitted as one fixed operator call.
uint64_t qsa_prefill_dispatches(uint32_t base, uint32_t rows, QsaShape shape, bool allow_nax_2k = false, bool nax_qsa = false, bool fused_prep = false);
void validate_qsa_prefill_rows(GpuRowsF32 rows, uint32_t expected_rows, uint32_t expected_columns, bool allow_nax_2k = false);
} // namespace tokspur
