#include "tokspur/grouped_mm.hpp"
#include "tokspur/gguf.hpp"
#include <limits>

namespace tokspur {
namespace {
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
void layout_check(uint32_t type, PrefillWeightLayout layout) {
    require(type == 7 || type == 8 || type == 12 || type == 13, "grouped mm encoding");
    require(layout == PrefillWeightLayout::gguf || (layout == PrefillWeightLayout::q8_lanes_v1 && type == 8) ||
        (layout == PrefillWeightLayout::q51_lanes_v1 && type == 7), "grouped mm layout/encoding mismatch");
}
uint64_t row_extent(uint32_t count, uint32_t width, uint64_t stride) {
    require(count && stride >= checked_mul(width, 4) && stride % 16 == 0, "tiled prefill FP32 row stride");
    return checked_add(checked_mul(count - 1, stride), checked_mul(width, 4));
}
}
const char* grouped_mm_kernel(uint32_t type, PrefillWeightLayout layout) {
    layout_check(type, layout); return "graph_grouped_mm";
}
const char* dense_mm_q8_kernel(PrefillWeightLayout layout) {
    layout_check(8, layout); return "graph_dense_mm_q8";
}
GroupedMmPlan grouped_mm_plan(uint32_t type, PrefillWeightLayout layout,
    uint32_t columns, uint32_t rows, uint32_t experts, uint32_t selected, uint32_t tokens,
    uint64_t input_stride, uint64_t output_stride, bool broadcast, bool allow_nax_2k) {
    layout_check(type, layout); const auto& format = encoding(type);
    const uint32_t max_tokens = allow_nax_2k ? 2048u : 512u;
    require(columns && columns <= 65536 && columns % format.elements_per_block == 0 && rows && rows <= 32768 &&
        experts && experts <= 512 && selected && selected <= 16 && selected <= experts && tokens && tokens <= max_tokens,
        "grouped mm geometry");
    if (layout == PrefillWeightLayout::q51_lanes_v1) require(columns == 640 && experts > 1, "registered Q5_1 lane geometry");
    if (layout == PrefillWeightLayout::q8_lanes_v1) {
        if (allow_nax_2k) require(columns % 32 == 0, "registered Q8 lane geometry");
        else require(experts == 1 && (columns % 256 == 0 || columns < 512), "registered Q8 lane geometry");
    }
    const auto routes = checked_mul(tokens, selected), stride = checked_mul(columns / format.elements_per_block, format.bytes_per_block);
    require(stride <= std::numeric_limits<uint32_t>::max(), "grouped mm row stride overflow");
    const auto expert_stride = checked_mul(stride, rows), maximum_tasks = routes / 32 + experts;
    GroupedMmPlan p;
    p.args = {columns, rows, uint32_t(stride), experts, selected, tokens, uint32_t(broadcast), uint32_t(maximum_tasks), expert_stride, input_stride, output_stride};
    p.matrix_bytes = checked_mul(expert_stride, experts); p.indices_bytes = checked_mul(routes, 4);
    p.input_bytes = row_extent(uint32_t(broadcast ? tokens : routes), columns, input_stride); p.output_bytes = row_extent(uint32_t(routes), rows, output_stride);
    p.token_map_bytes = checked_mul(checked_mul(experts, tokens), 4); p.counts_bytes = checked_mul(experts, 4); p.tasks_bytes = checked_mul(checked_add(2, checked_mul(maximum_tasks, 2)), 4);
    require(p.matrix_bytes <= (2ull << 30), "grouped mm matrix budget");
    for (auto bytes : {p.input_bytes, p.output_bytes, p.token_map_bytes, p.counts_bytes, p.tasks_bytes}) require(bytes <= (256ull << 20), "grouped mm workspace budget");
    p.grid_x = (rows + 63) / 64; p.grid_y = uint32_t(maximum_tasks); p.kernel = grouped_mm_kernel(type, layout); return p;
}
GroupedMmPlan dense_mm_q8_plan(PrefillWeightLayout layout, uint32_t columns, uint32_t rows,
    uint32_t tokens, uint64_t input_stride, uint64_t output_stride, bool allow_nax_2k) {
    auto p = grouped_mm_plan(8, layout, columns, rows, 1, 1, tokens, input_stride, output_stride, true, allow_nax_2k);
    p.indices_bytes = p.token_map_bytes = p.counts_bytes = p.tasks_bytes = 0; p.args.maximum_tasks = (tokens + 31) / 32;
    p.grid_y = p.args.maximum_tasks; p.kernel = dense_mm_q8_kernel(layout); return p;
}
}
