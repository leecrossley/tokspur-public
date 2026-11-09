#pragma once
#include "tokspur/moe.hpp"

namespace tokspur {
// Three independent routed expert rows retain ordinary GEMVs. Only shared Q8 projections reuse their weights across all three rows. Scratch is borrowed token-major capacity storage.
void encode_moe_q8_three_rows(MetalGraph&, const MoeWeights&, const MoeScratch&, GpuView input, GpuView output,
    MoeShape, RouterMode, bool normalize_selected, uint32_t capacity);
bool moe_three_union_layer_supported(const MetalGraph&, const MoeWeights&, MoeShape);
uint64_t moe_q8_three_rows_saved_dispatches(const MetalGraph&, const MoeWeights&, MoeShape);
}
