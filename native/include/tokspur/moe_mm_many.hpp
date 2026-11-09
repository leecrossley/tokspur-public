#pragma once
#include "tokspur/moe.hpp"
#include "tokspur/grouped_mm.hpp"

namespace tokspur {
// Explicit registered identities. Raw GGUF is the default, never a packed-byte fallback.
struct MoeMmLayouts {
    PrefillWeightLayout gate = PrefillWeightLayout::gguf, up = PrefillWeightLayout::gguf, down = PrefillWeightLayout::gguf;
    PrefillWeightLayout shared_gate = PrefillWeightLayout::gguf, shared_up = PrefillWeightLayout::gguf, shared_down = PrefillWeightLayout::gguf;
};
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
// Raw FP32 router projection only. Strides are byte counts in both host and Metal.
struct MoeRouterF32Args {
    uint32_t columns, experts, tokens, reserved;
    uint64_t input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(MoeRouterF32Args)==32, "router FP32 argument ABI");
struct MoeRouterF32Plan {
    MoeRouterF32Args args;
    uint64_t matrix_bytes, input_bytes, output_bytes;
    uint32_t grid_x, grid_y;
};
inline constexpr uint32_t moe_router_f32_threadgroup_bytes=8192;
inline constexpr const char* moe_router_f32_kernel="graph_moe_router_f32_batch";
MoeRouterF32Plan moe_router_f32_plan(uint32_t columns, uint32_t experts, uint32_t tokens, uint64_t input_stride_bytes, uint64_t output_stride_bytes);
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
// Exact windowed FP32 router projection and row-batched top-k, shared router and mixture.
struct MoeRouterWindowArgs {
    uint32_t columns, experts, tokens, reserved;
    uint64_t input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(MoeRouterWindowArgs) == 32, "router window argument ABI");
struct MoeRouterWindowPlan {
    MoeRouterWindowArgs args;
    uint64_t matrix_bytes, input_bytes, output_bytes;
    uint32_t grid_x, grid_y;
};
inline constexpr uint32_t moe_router_window_rows = 16;
inline constexpr const char* moe_router_window_kernel = "graph_moe_router_f32_window";
inline constexpr const char* moe_route_rows_kernel = "graph_moe_route_rows";
inline constexpr const char* moe_shared_router_rows_kernel = "graph_moe_shared_router_rows";
inline constexpr const char* moe_mixture_rows_kernel = "graph_moe_mixture_rows";
MoeRouterWindowPlan moe_router_window_plan(uint32_t columns, uint32_t experts, uint32_t tokens, uint64_t input_stride_bytes, uint64_t output_stride_bytes, bool allow_nax_2k = false);
#endif
struct MoeMmManyBytes {
    std::array<uint64_t, 13> views{};
    uint64_t total = 0;
};
struct MoeMmManyScratch {
    MoeScratch rows;
    GpuView token_map, counts, tasks;
    uint32_t capacity = 0;
    [[nodiscard]] std::array<GpuView, 13> views() const;
};
MoeMmManyBytes moe_mm_many_bytes(MoeShape, uint32_t capacity, bool allow_nax_2k = false);
MoeMmManyScratch allocate_moe_mm_many_scratch(MetalGraph&, MoeShape, uint32_t capacity, bool allow_nax_2k = false);
void release_moe_mm_many_scratch(MetalGraph&, const MoeMmManyScratch&);
void prepare_moe_mm_many(MetalGraph&, const MoeWeights&, MoeShape, MoeMmLayouts = {});
uint64_t moe_mm_many_dispatches(const MetalGraph&, const MoeWeights&, MoeShape, uint32_t rows);
// Encode only. The caller owns allocation, command buffers and committed state.
void encode_moe_mm_many(MetalGraph&, const MoeWeights&, const MoeMmManyScratch&,
    GpuRowsF32 input, GpuRowsF32 output, MoeShape, RouterMode, bool normalize_selected, MoeMmLayouts = {});
}
