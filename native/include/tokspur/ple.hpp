#pragma once

#include "tokspur/recurrent.hpp"
#include "tokspur/recurrent_reference.hpp"

namespace tokspur {
struct PleShape {
    uint32_t branches, channels, embedding, kernel, dilation;
#ifdef TOKSPUR_PRIVATE_PLE_GATE_COMPAT
    PleGateTransform gate_transform = PleGateTransform::signed_root;
#else
    PleGateTransform gate_transform = PleGateTransform::plain;
#endif
};
// The explicit compatibility build uses Flash-Next signed-root scores. Ordinary builds retain the historical plain default.
double ple_gate_reference(double normalized_score, PleGateTransform transform);
struct PleWeights { GpuMatrix key, value; GpuView key_gains, query_gains, conv_gains, convolution; };
struct PleScratch {
    GpuView key, value, normalized_key, normalized_query, gated, gates, normalized_gated, convolved;
    [[nodiscard]] std::array<GpuView, 8> views() const;
};
void validate_ple_shape(PleShape shape);
PleScratch allocate_ple_scratch(MetalGraph& graph, PleShape shape);
void release_ple_scratch(MetalGraph& graph, const PleScratch& scratch);
void encode_ple(MetalGraph& graph, const PleWeights& weights, const PleScratch& scratch, GpuView embedding,
                  GpuView streams, GpuView current_history, GpuView next_history, GpuView output, PleShape shape, float epsilon);

struct ReferencePleWeights {
    ReferenceMatrix key, value;
    std::span<const double> key_gains, query_gains, conv_gains, convolution;
};
struct ReferencePle {
    std::vector<double> key, value, normalized_key, normalized_query, gated, gates, normalized_gated, output;
    ConvStep convolution;
};
ReferencePle ple_reference(const ReferencePleWeights& weights, std::span<const double> embedding,
                             std::span<const double> streams, std::span<const double> history, PleShape shape, double epsilon);

struct MixerWeights { GpuView gains; GpuMatrix down, up; };
struct MixerScratch {
    GpuView normalized, bottleneck, scores, mixed;
    [[nodiscard]] std::array<GpuView, 4> views() const;
};
MixerScratch allocate_mixer_scratch(MetalGraph& graph, ResidualShape shape);
void release_mixer_scratch(MetalGraph& graph, const MixerScratch& scratch);
void encode_mixer(MetalGraph& graph, const MixerWeights& weights, const MixerScratch& scratch, GpuView streams,
                    ResidualShape shape, float epsilon);
struct ReferenceMixer { std::vector<double> normalized, bottleneck, scores, mixed; };
ReferenceMixer mixer_reference(std::span<const double> gains, ReferenceMatrix down, ReferenceMatrix up,
                                 std::span<const double> streams, ResidualShape shape, double epsilon);
} // namespace tokspur
