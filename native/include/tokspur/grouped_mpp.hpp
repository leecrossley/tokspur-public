#pragma once
#include <cstdint>

namespace tokspur {
struct GroupedMppShape {
    uint32_t columns, rows, experts, selected, batches, tile;
    uint32_t staging_k = 0; // 0: cooperative; 32/128: FP32 staged; private 1: ordered GEMV.
};
struct GroupedMppBytes {
    uint64_t matrix, indices, input, output, token_map, counts, tasks;
    uint32_t maximum_tasks, row_bytes;
};
// Private O-tier known-input expert projection. Routing IDs are supplied,
// grouped stably on the GPU, and outputs retain token/selection-slot order.
GroupedMppBytes grouped_mpp_bytes(uint32_t encoding, GroupedMppShape shape, bool broadcast);
const char* grouped_mpp_kernel(uint32_t encoding, uint32_t tile, uint32_t staging_k = 0);
}
