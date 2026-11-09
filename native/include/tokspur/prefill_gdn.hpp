#pragma once
#include "tokspur/gdn_chunk.hpp"
#include "tokspur/moe_mm_many.hpp"
#include "tokspur/prefill_residual.hpp"

namespace tokspur {
struct GdnPrefillLayouts {
    PrefillWeightLayout qkv = PrefillWeightLayout::gguf, gate = PrefillWeightLayout::gguf, output = PrefillWeightLayout::gguf;
    MoeMmLayouts ffn;
};
GdnPrefillLayouts gdn_prefill_layouts(const MetalGraph&, const RecurrentWeights&);
void prepare_gdn_prefill_layer(MetalGraph&, const RecurrentWeights&, RecurrentShape, GdnPrefillLayouts, bool legacy_order = false);
uint64_t gdn_prefill_layer_dispatches(const MetalGraph&, const RecurrentWeights&, RecurrentShape, uint32_t rows, GdnChunkMode);
// A complete committed-input layer. Accepted GDN state is read-only and publication belongs to the stack.
void encode_gdn_prefill_layer(MetalGraph&, const RecurrentWeights&, const PrefillLayerScratch&,
    const GdnChunkScratch&, const MoeMmManyScratch&, const GdnState& accepted, const GdnState& candidate,
    GpuRowsF32 input_streams, GpuRowsF32 output_streams, RecurrentShape, GdnPolicy,
    RouterMode, bool normalize_selected, GdnChunkMode, GdnPrefillLayouts, bool legacy_order = false,
    const GdnState* work_state = nullptr);
}
