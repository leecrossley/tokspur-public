#pragma once
#include <cstdint>

namespace tokspur {
// Layout is separate from encoding. A packed view must be registered by Graph.
enum class PrefillWeightLayout : uint32_t { gguf = 0, q8_lanes_v1 = 1, q51_lanes_v1 = 2 };
// Shared CPU/Metal constants. Row strides are bytes, including the 680-byte Q8 row.
struct GroupedMmArgs {
    uint32_t columns, rows, row_bytes, experts;
    uint32_t selected, tokens, broadcast, maximum_tasks;
    uint64_t expert_stride_bytes, input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(GroupedMmArgs) == 56);
struct GroupedMmPlan {
    GroupedMmArgs args{};
    uint64_t matrix_bytes = 0, indices_bytes = 0, input_bytes = 0, output_bytes = 0;
    uint64_t token_map_bytes = 0, counts_bytes = 0, tasks_bytes = 0;
    uint32_t grid_x = 0, grid_y = 0, threads = 128, threadgroup_bytes = 8192;
    const char* kernel = nullptr;
};
// Pure geometry and byte bounds. No allocation, model access or Metal calls.
GroupedMmPlan grouped_mm_plan(uint32_t encoding, PrefillWeightLayout layout,
    uint32_t columns, uint32_t rows, uint32_t experts, uint32_t selected, uint32_t tokens,
    uint64_t input_stride_bytes, uint64_t output_stride_bytes, bool broadcast, bool allow_nax_2k = false);
GroupedMmPlan dense_mm_q8_plan(PrefillWeightLayout layout, uint32_t columns, uint32_t rows,
    uint32_t tokens, uint64_t input_stride_bytes, uint64_t output_stride_bytes, bool allow_nax_2k = false);
const char* grouped_mm_kernel(uint32_t encoding, PrefillWeightLayout layout);
const char* dense_mm_q8_kernel(PrefillWeightLayout layout);
}
