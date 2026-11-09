#pragma once
#include "tokspur/attention.hpp"
#include "tokspur/qsa_prefill_plan.hpp"

namespace tokspur {
// Borrowed FP32 rows. The four Q8 projection families are filled by the caller's layout-aware dense builder; this component fills only the BF16 indexers.
struct QsaPrefillProjections {
    GpuRowsF32 packed, key, value, index_query, index_key;
};
struct QsaPrefillScratch {
    uint32_t capacity=0;
    QsaPrefillProjections projected;
    GpuView query,gate,normalized_query,normalized_key,rotary_query,rotary_key,normalized_index_query,rotary_index_query;
    GpuView pools,normalized_pools,rotary_pools,scores,score_a,id_a,score_b,id_b,selected_ids,partials,attention,gated;
    bool suballocated = false;
    [[nodiscard]] std::array<GpuView,25> views() const;
};
QsaPrefillScratch allocate_qsa_prefill_scratch(MetalGraph& graph,QsaShape shape,uint32_t capacity=512,GpuView arena={});
void release_qsa_prefill_scratch(MetalGraph& graph,const QsaPrefillScratch& scratch);
QsaPrefillProjections qsa_prefill_projection_rows(const QsaPrefillScratch& scratch,uint32_t rows);
// The caller begins/submits. No allocation, host readback, command buffer or publication occurs here.
void encode_qsa_prefill_indexers(MetalGraph& graph,const QsaWeights& weights,const QsaPrefillScratch& scratch,GpuRowsF32 input,QsaShape shape);
// Accepted sum and published cache prefix remain read-only. Only [base,base+rows) KV and newly completed pools may be dirtied. Full rollback of every observed capacity byte remains the caller's responsibility after drain.
GpuRowsF32 encode_qsa_prefill_projected(MetalGraph& graph,const QsaWeights& weights,const QsaPrefillScratch& scratch,
    const QsaCache& cache,GpuView accepted_sum,GpuView candidate_sum,uint32_t base,uint32_t rows,QsaShape shape,QsaPolicy policy);
} // namespace tokspur
