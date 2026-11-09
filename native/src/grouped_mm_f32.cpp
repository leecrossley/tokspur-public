#include "tokspur/grouped_mm_f32.hpp"
#include "tokspur/gguf.hpp"

namespace tokspur {
GroupedMmPlan grouped_mm_f32_plan(uint32_t encoding, PrefillWeightLayout layout,
    uint32_t columns, uint32_t rows, uint32_t experts, uint32_t selected, uint32_t tokens,
    uint64_t input_stride, uint64_t output_stride, bool broadcast, bool allow_nax_2k) {
    const bool ok_layout = layout == PrefillWeightLayout::gguf ||
        (layout == PrefillWeightLayout::q51_lanes_v1 && encoding == 7) ||
        (allow_nax_2k && layout == PrefillWeightLayout::q8_lanes_v1 && encoding == 8);
    if (!ok_layout)
        throw FormatError("FP32 grouped prefill requires original GGUF layout or Q5_1 lanes layout");
    auto plan = grouped_mm_plan(encoding, layout, columns, rows, experts, selected, tokens,
        input_stride, output_stride, broadcast, allow_nax_2k);
    plan.kernel = grouped_mm_f32_kernel;
    plan.threadgroup_bytes = grouped_mm_f32_threadgroup_bytes;
    return plan;
}
}
