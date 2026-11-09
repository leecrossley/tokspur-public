#pragma once

#include "tokspur/recurrent.hpp"
#include <array>
#include <span>
#include <vector>

namespace tokspur {

// Committed prefill only. Accepted state is native [head,key,value], value contiguous.
enum class GdnChunkMode : uint32_t { fused = 0, cs64 = 1 };
enum class GdnChunkOp : uint32_t { projection_f32, convolution, prepare, gates, fused, rms_gate, rms_rows, prefix64, coefficients64, solve64, emit64, projection_gates_f32, conv_silu_prepare };

// Shared verbatim with gdn_chunk.metal. Strides are bytes, all state is FP32.
struct GdnChunkParams {
    uint32_t rows = 0, columns = 0, outputs = 0, heads = 0;
    uint32_t key_heads = 0, key_width = 0, value_width = 0, mode = 0;
    uint32_t offset = 0, count = 0, l2_mode = 0, reserved = 0;
    uint64_t input_stride = 0, output_stride = 0, auxiliary_stride = 0, gate_stride = 0;
    float query_scale = 0, l2_epsilon = 0, rms_epsilon = 0, reserved_float = 0;
};
static_assert(sizeof(GdnChunkParams) == 96 && offsetof(GdnChunkParams, input_stride) == 48 && offsetof(GdnChunkParams, query_scale) == 80);
struct GdnChunkDispatch {
    GdnChunkOp op = GdnChunkOp::projection_f32;
    std::array<GpuView, 10> buffers{}; // Physical bindings. Slot7 is reserved for the graph error flag.
    GdnChunkParams params;
};
inline constexpr std::array<const char*, 13> gdn_chunk_kernel_names{"gdn_chunk_f32", "gdn_chunk_conv", "gdn_chunk_prepare", "gdn_chunk_gates", "gdn_chunk_fused", "gdn_chunk_rms_gate", "gdn_chunk_rms_rows", "gdn_chunk_prefix64", "gdn_chunk_coeff64", "gdn_chunk_solve64", "gdn_chunk_emit64", "gdn_chunk_proj_gates_f32", "gdn_chunk_conv_silu_prepare"};
struct GdnChunkSpec {
    const char* kernel;
    uint16_t reads, writes;
    std::array<uint32_t, 3> groups;
    uint32_t threads = 128;
};
GdnChunkSpec gdn_chunk_spec(const GdnChunkDispatch&);
void validate_gdn_chunk_dispatch(const GdnChunkDispatch&, bool allow_nax_2k = false);
uint64_t gdn_chunk_row_extent(GpuRowsF32, bool allow_nax_2k = false);

struct GdnChunkScratch {
    uint32_t capacity = 0;
    GpuView projected, convolved, query, key, value, z, alpha, beta_logits, log_decay, beta, raw, gated;
    GpuView prefix, lower, attention, residual, state_a, state_b;
    bool suballocated = false;
    [[nodiscard]] std::array<GpuView, 18> views() const;
};
struct GdnChunkBytes {
    uint64_t rows = 0, recurrence = 0, total = 0;
};
GdnChunkBytes gdn_chunk_bytes(GdnShape, uint32_t capacity, GdnChunkMode, bool allow_nax_2k = false);
GdnChunkScratch allocate_gdn_chunk_scratch(MetalGraph&, GdnShape, uint32_t capacity, GdnChunkMode, GpuView arena = {});
void release_gdn_chunk_scratch(MetalGraph&, const GdnChunkScratch&);

// Dense Q8 projection scheduling is supplied by the integration owner. It must encode only, retain FP32 rows, and honour all strides without changing state or command buffers.
using GdnChunkProjection = void (*)(MetalGraph&, const GpuMatrix&, GpuRowsF32 input, GpuRowsF32 output);
void encode_gdn_prefill_chunk(MetalGraph&, const GdnWeights&, const GdnChunkScratch&,
    const GdnState& accepted, const GdnState& candidate, GpuRowsF32 input, GpuRowsF32 output,
    GdnShape, GdnPolicy, GdnChunkMode, GdnChunkProjection, bool legacy_order = false, bool allow_multiseg = false);

// Independent small-domain double equations, never selected by the runtime or an inference fallback.
struct GdnChunkReference { std::vector<double> output, state; };
GdnChunkReference gdn_chunk_reference(std::span<const double> state, std::span<const double> query_scaled,
    std::span<const double> key, std::span<const double> value, std::span<const double> log_decay,
    std::span<const double> beta, uint32_t heads, uint32_t key_width, uint32_t value_width,
    uint32_t rows, uint32_t chunk_size = 64);

} // namespace tokspur
