#pragma once
#include "tokspur/recurrent.hpp"

namespace tokspur {
struct PrefillResidualScratch {
    ResidualScratch rows;
    uint32_t capacity = 0;
};
struct PrefillLayerScratch {
    PrefillResidualScratch residual;
    GpuView intermediate_streams, attention_output, ffn_output;
    [[nodiscard]] std::array<GpuView, 9> views() const;
};
PrefillLayerScratch allocate_prefill_layer_scratch(MetalGraph&, ResidualShape, uint32_t capacity);
void release_prefill_layer_scratch(MetalGraph&, const PrefillLayerScratch&);
PrefillResidualScratch allocate_prefill_residual_scratch(MetalGraph&, ResidualShape, uint32_t capacity);
void release_prefill_residual_scratch(MetalGraph&, const PrefillResidualScratch&);
void prepare_prefill_residual(MetalGraph&, const ResidualWeights&, ResidualShape);
uint64_t prefill_residual_dispatches(const MetalGraph&, const ResidualWeights&, ResidualShape, uint32_t rows);
uint64_t prefill_residual_write_dispatches(const MetalGraph&, ResidualShape, uint32_t rows);
// The caller owns submission and state publication. Injection retains its existing per-row arithmetic. HC down does too, except for the explicit FP32 raw-Q8 full-512 experiment at the model HC geometry.
void encode_prefill_residual(MetalGraph&, const ResidualWeights&, const PrefillResidualScratch&,
    GpuRowsF32 streams, ResidualShape, float rms_epsilon);
GpuRowsF32 prefill_residual_input(const PrefillResidualScratch&, ResidualShape, uint32_t rows);
void encode_prefill_residual_write(MetalGraph&, const PrefillResidualScratch&, GpuRowsF32 streams,
    GpuRowsF32 values, GpuRowsF32 output, ResidualShape);
bool can_fuse_prefill_hc_consumer(const MetalGraph&, ResidualShape, uint32_t rows);
void encode_prefill_residual_write_and_read(MetalGraph&, const ResidualWeights&, const PrefillResidualScratch&,
    GpuRowsF32 streams, GpuRowsF32 values, GpuRowsF32 intermediate, ResidualShape, float rms_epsilon);
}
