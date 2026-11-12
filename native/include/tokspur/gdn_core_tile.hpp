#pragma once

#include "tokspur/recurrent.hpp"

namespace tokspur {
// Private scheduling experiment; existing shaders retain their arithmetic.
inline constexpr bool gdn_core_batch_elementwise =
#if defined(TOKSPUR_PRIVATE_GDN_ELEMENTWISE_BATCH)
    true;
#else
    false;
#endif
constexpr uint64_t gdn_core_elementwise_saved_dispatches(uint32_t steps) {
    return gdn_core_batch_elementwise && steps ? uint64_t(steps - 1) * 3 : 0;
}
// Row batching retains the scalar-row preparation/gate arithmetic. Only the
// NAX verifier selects this experiment; decode, prefill and replay stay scalar.
inline constexpr bool verify_gdn_row_batch =
#if defined(TOKSPUR_PRIVATE_VERIFY_GDN_ROW_BATCH)
    true;
#else
    false;
#endif
// Ordered attention-core tile; residual/FFN composition lives in recurrent_tile.
// Thirteen capacity-sized, token-major scratch views; key128/kernel4, Q8
// QKV/gate/output and F32 alpha/beta. Projection input widths are multiples32.
// Prepare all pipelines/resources while idle, never inside either encode.
GdnScratch allocate_gdn_core_tile_scratch(MetalGraph& graph, GdnShape shape, uint32_t capacity);
void encode_gdn_core_tile(MetalGraph& graph, const GdnWeights& weights, const GdnScratch& scratch,
                          const GdnState& accepted, const GdnState& candidate,
                          GpuView inputs, GpuView outputs, GdnShape shape,
                          GdnPolicy policy, StateTile tile);

struct GdnCoreReplayScratch { GpuView convolution_output, delta_output; };
// Compact replay tape, independent of the seven forward-only scratch arrays.
// Bind these six capacity-sized views into otherwise shared forward scratch;
// retain one tape per layer until the whole-stack acceptance decision is made.
struct GdnCoreTape {
    GpuView projected, query, key, value, decay, write_gate;
    [[nodiscard]] std::array<GpuView, 6> views() const {
        return {projected, query, key, value, decay, write_gate};
    }
};
GdnCoreTape allocate_gdn_core_tape(MetalGraph& graph, GdnShape shape, uint32_t capacity);
GdnScratch bind_gdn_core_tape(GdnScratch shared, const GdnCoreTape& tape);
void encode_gdn_core_tape_prefix(MetalGraph& graph, GpuView convolution_weights,
    const GdnCoreTape& tape, const GdnState& accepted, const GdnState& candidate,
    const GdnCoreReplayScratch& scratch, GdnShape shape, StateTile tile);
GdnCoreReplayScratch allocate_gdn_core_replay_scratch(MetalGraph& graph, GdnShape shape, uint32_t capacity);
void release_gdn_core_replay_scratch(MetalGraph& graph, const GdnCoreReplayScratch& scratch);
// Replay from the SAME accepted base, not the future candidate. No tape or
// accepted/output buffer writes. k=0 retains the base without GPU work; caller
// publication is allowed only after whole-batch success and ownership checks.
void encode_gdn_core_prefix(MetalGraph& graph, GpuView convolution_weights,
                            const GdnScratch& tape, GpuView retained_outputs, const GdnState& accepted,
                            const GdnState& candidate, const GdnCoreReplayScratch& scratch,
                            GdnShape shape, StateTile tile);
int gdn_core_tile_check_cli(int argc, char** argv);
} // namespace tokspur
