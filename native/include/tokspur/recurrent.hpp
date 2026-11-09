#pragma once

#include "tokspur/moe.hpp"

#include <array>

namespace tokspur {

struct ResidualShape { uint32_t branches, channels, rank; };
struct ResidualWeights { GpuView gains; GpuMatrix down, up, inject; };
struct ResidualScratch {
    GpuView normalized, bottleneck, scores, inject_logits, input, scales;
    [[nodiscard]] std::array<GpuView, 6> views() const;
};
void validate_residual_shape(ResidualShape shape);
ResidualScratch allocate_residual_scratch(MetalGraph& graph, ResidualShape shape);
void release_residual_scratch(MetalGraph& graph, const ResidualScratch& scratch);
void encode_residual_read(MetalGraph& graph, const ResidualWeights& weights, const ResidualScratch& scratch,
                          GpuView streams, ResidualShape shape, float rms_epsilon);

struct GdnWeights {
    GpuMatrix qkv, gate, alpha, beta, output;
    GpuView convolution, negative_rates, dt_bias, gains;
};
// Current and next state must be distinct. Completion alone doesn't publish a
// prefix: the caller may swap them only after successful submit and ownership
// checks. A failed submit leaves the current state untouched; next is invalid.
struct GdnState {
    GpuView matrix, convolution;
    [[nodiscard]] std::array<GpuView, 2> views() const;
};
struct GdnScratch {
    GpuView projected, convolved, query, key, value, gate, alpha, beta, decay, write_gate, raw, normalized, gated;
    [[nodiscard]] std::array<GpuView, 13> views() const;
};
GdnState allocate_gdn_state(MetalGraph& graph, GdnShape shape);
void release_gdn_state(MetalGraph& graph, const GdnState& state);
GdnScratch allocate_gdn_scratch(MetalGraph& graph, GdnShape shape);
void release_gdn_scratch(MetalGraph& graph, const GdnScratch& scratch);
void encode_gdn(MetalGraph& graph, const GdnWeights& weights, const GdnScratch& scratch,
                 const GdnState& current, const GdnState& next, GpuView input, GpuView output,
                 GdnShape shape, GdnPolicy policy);

struct RecurrentShape { ResidualShape residual; GdnShape gdn; MoeShape moe; };
struct RecurrentWeights { ResidualWeights attention_read, ffn_read; GdnWeights attention; MoeWeights ffn; };
struct RecurrentScratch {
    ResidualScratch residual;
    GdnScratch gdn;
    MoeScratch moe;
    GpuView intermediate_streams, attention_output, ffn_output;
    [[nodiscard]] std::array<GpuView, 32> views() const;
};
void validate_recurrent_shape(RecurrentShape shape);
RecurrentScratch allocate_recurrent_scratch(MetalGraph& graph, RecurrentShape shape);
void release_recurrent_scratch(MetalGraph& graph, const RecurrentScratch& scratch);

// Complete ordinary recurrent layer: GR read -> GDN -> residual write -> GR
// read -> routed/shared FFN -> residual write. Exactly 40 dispatches, no host
// readback/allocation/commit. This is NOT the sparse-attention or PLE layer.
// Any composite encode exception aborts the entire unsubmitted caller batch.
void encode_recurrent_layer(MetalGraph& graph, const RecurrentWeights& weights, const RecurrentScratch& scratch,
                             const GdnState& current, const GdnState& next, GpuView input_streams, GpuView output_streams,
                             RecurrentShape shape, GdnPolicy policy, RouterMode router_mode, bool normalize_selected);

} // namespace tokspur
