#pragma once

#include "tokspur/attention.hpp"
#include "tokspur/qsa_verify_rows.hpp"
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
#include "tokspur/verify_nax.hpp"
#endif

namespace tokspur {
// Private, independently selectable component schedules. The new caller is the private COST verifier only.
// Ordinary layer entry points and serving policy stay unchanged.
enum class AttentionTileMode : uint32_t { core_only, core_and_q8 };
// Preserve the old owned component signature; this adaptation is prepared only
// with explicit combined-HC and chronological-channel ON.
enum class AttentionTileHcMode : uint32_t { off, combined };
// core_and_q8 requires separately admitted byte checks of all four projection
// families on matching inputs/weights. This encoder does not self-test weights.
struct AttentionTileScratch {
    AttentionLayerScratch rows; // All43 views are capacity8/token-major.
    GpuView prefix_sums; // Seven chronological intermediate index sums.
    GpuView core_scores, core_factors; // Fixed score/factor allocations, shared across QSA layers.
    [[nodiscard]] std::array<GpuView, 46> views() const;
};
// Idle startup preparation is identical in both modes. No request-time payload
// allocation. The existing Q8-many kernels are used, never a new projection.
void prepare_attention_tile(MetalGraph& graph, AttentionTileHcMode hc = AttentionTileHcMode::combined);
AttentionTileScratch allocate_attention_tile_scratch(MetalGraph& graph, AttentionLayerShape shape);
// Exact logical payload, checked before an opt-in caller allocates the46 owners.
[[nodiscard]] uint64_t attention_tile_scratch_bytes(AttentionLayerShape shape);
// Only the24 capacity-eight QSA views are newly owned. A whole-stack prefill
// caller may combine these with its existing19 recurrent common views; it must
// never release the resulting borrowed composition as46 independent owners.
QsaScratch allocate_attention_tile_qsa_scratch(MetalGraph& graph, AttentionLayerShape shape);
void release_attention_tile_scratch(MetalGraph& graph, const AttentionTileScratch& scratch);
AttentionLayerScratch attention_tile_token(MetalGraph& graph, const AttentionTileScratch& scratch,
    AttentionLayerShape shape, uint32_t row);
// Source-derived physical count, computed independently of observed batches.
uint64_t attention_tile_dispatches(MetalGraph& graph, const AttentionLayerWeights& weights,
    uint32_t base_prefix, uint32_t steps, AttentionLayerShape shape, AttentionTileMode mode,
    AttentionTileHcMode hc = AttentionTileHcMode::combined);
// One layer only. All N input rows already exist. Each query retains its own
// causal prefix/selection, the BF16 boundaries and chronological index sum.
// Prefix_sum[r] retains state after r+1 rows (r<N-1); final state lands in next.
// Accepted sum and accepted cache prefixes are read-only. Future append rows,
// output, scratch and next may be dirty after a failed submit: caller discards
// them. There is no all-layer publication, cancellation polling, RNG, partial
// acceptance or V8 recovery here. An encode exception aborts the WHOLE batch.
void encode_attention_layer_tile(MetalGraph& graph, const AttentionLayerWeights& weights,
    const AttentionTileScratch& scratch, const QsaCache& cache, GpuView accepted_sum, GpuView next_sum,
    GpuView input_rows, GpuView output_rows, uint32_t base_prefix, uint32_t steps,
    AttentionLayerShape shape, QsaPolicy policy, RouterMode router, bool normalize_selected,
    AttentionTileMode mode, AttentionTileHcMode hc = AttentionTileHcMode::combined
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    , const VerifyNaxMoeScratch* verify_nax_moe = nullptr,
    bool input_prenormalized = false, GpuView next_rms_gains = {}, float next_rms_epsilon = 0.0f
#endif
    );
} // namespace tokspur
