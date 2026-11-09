#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace tokspur {
struct GpuView;
// Fixed private component allowlist. Buffer6 is params, buffer7 is the graph error flag.
enum class QsaPrefillOp : uint32_t {
    bf16_project, split_gate, rope_rows, store_bf16, index_pool, index_scores,
    select_sort, select_merge, select_ids, attention_partial128, attention_reduce,
    prep_query, prep_kv, prep_index_query, prep_index_pool, verify_index_pool
};
inline constexpr std::array qsa_prefill_operations{QsaPrefillOp::bf16_project, QsaPrefillOp::split_gate,
    QsaPrefillOp::rope_rows, QsaPrefillOp::store_bf16, QsaPrefillOp::index_pool, QsaPrefillOp::index_scores,
    QsaPrefillOp::select_sort, QsaPrefillOp::select_merge, QsaPrefillOp::select_ids,
    QsaPrefillOp::attention_partial128, QsaPrefillOp::attention_reduce,
    QsaPrefillOp::prep_query, QsaPrefillOp::prep_kv, QsaPrefillOp::prep_index_query, QsaPrefillOp::prep_index_pool,
    QsaPrefillOp::verify_index_pool};
// All strides below are elements, converted from checked byte-strided row views by the builder. Unused words must be zero.
// bf16_project: N,C,R,input_stride,output_stride. Buffers: BF16 W,FP32 X,FP32 Y.
// split_gate: N,H,D,packing,input_stride,query_stride,gate_stride. Buffers: packed,Q,gate.
// rope_rows: N,H,D,pairs,first_position,position_step,input_stride,output_stride,pair_mode. Buffers: X,freq,Y.
// store_bf16: N,C,input_stride,output_stride_u16. Buffers: FP32 X,BF16 Y.
// index_pool: N,D,base,raw_stride,pool_stride,pool_count,compression. Buffers: raw,accepted_sum,pools,candidate_sum.
// index_scores: stripe_N,stripe_base,H,D,score_stride,query_stride,budget. Buffers: Q,BF16 pools,scores.
// select_sort: stripe_N,stripe_base,score_stride,slot_stride,budget. Buffers: scores,sorted_scores,sorted_ids.
// select_merge: stripe_N,stripe_base,slot_stride,budget,span. Buffers: in_scores,in_ids,out_scores,out_ids.
// select_ids: stripe_N,stripe_base,input_slot_stride,output_id_stride,budget. Buffers: ranked_ids,canonical_ids.
// attention_partial128: stripe_N,stripe_base,QH,KVH,D,compression,budget,query_stride,id_stride,max_tiles. Buffers: Q,K,V,ids,partials.
// attention_reduce: stripe_N,stripe_base,QH,D,compression,budget,max_tiles,output_stride. Buffers: partials,Y.
using QsaPrefillParams = std::array<uint32_t, 16>;
struct QsaPrefillDispatchSpec {
    const char* kernel;
    uint32_t buffers, writable_mask;
    std::array<uint32_t, 3> groups, threads;
};
const char* qsa_prefill_kernel_name(QsaPrefillOp op);
QsaPrefillDispatchSpec qsa_prefill_spec(QsaPrefillOp op, const QsaPrefillParams& params, bool allow_nax_2k = false);
void validate_qsa_prefill_dispatch(QsaPrefillOp op, std::span<const GpuView> buffers, const QsaPrefillParams& params, bool allow_nax_2k = false);
} // namespace tokspur
