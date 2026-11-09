#pragma once
#include "tokspur/grouped_mm.hpp"

namespace tokspur {
// Same raw grouping ABI and active extents, with full FP32 operand staging.
GroupedMmPlan grouped_mm_f32_plan(uint32_t encoding, PrefillWeightLayout layout,
    uint32_t columns, uint32_t rows, uint32_t experts, uint32_t selected, uint32_t tokens,
    uint64_t input_stride_bytes, uint64_t output_stride_bytes, bool broadcast, bool allow_nax_2k = false);
inline constexpr uint32_t grouped_mm_f32_threadgroup_bytes = 12288;
inline constexpr const char* grouped_mm_f32_kernel = "graph_grouped_mm_f32";
}
