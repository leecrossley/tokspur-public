#pragma once

#include "tokspur/operators.hpp"
#include "tokspur/recurrent.hpp"

namespace tokspur {

// Development diagnostics only, never a production fallback. Matrix bytes stay
// in their declared encoding; scalar reference projections accumulate in double.
struct ReferenceMatrix { std::span<const std::byte> bytes; uint32_t encoding; };
struct ReferenceResidualWeights { std::span<const double> gains; ReferenceMatrix down, up, inject; };
struct ReferenceGdnWeights {
    ReferenceMatrix qkv, gate, alpha, beta, output;
    std::span<const double> convolution, negative_rates, dt_bias, gains;
};
struct ReferenceMoeWeights { ReferenceMatrix router, gate, up, down, shared_gate, shared_up, shared_down, shared_router; };
struct ReferenceLayerWeights { ReferenceResidualWeights attention_read, ffn_read; ReferenceGdnWeights attention; ReferenceMoeWeights ffn; };

struct ReferenceResidual {
    std::vector<double> normalized, bottleneck, scores, inject_logits;
    ResidualRead read;
};
struct ReferenceGdn {
    std::vector<double> projected, gate, alpha, beta, normalized, gated, output;
    ConvStep convolution;
    GdnPrepared prepared;
    GdnGates gates;
    DeltaStep delta;
};
struct ReferenceMoe {
    std::vector<double> logits, activated, routed, shared_activated, shared_output, output;
    Routes routes;
};
ReferenceResidual residual_projection_reference(const ReferenceResidualWeights& weights, std::span<const double> streams,
                                                  ResidualShape shape, double epsilon);
ReferenceGdn gdn_projection_reference(const ReferenceGdnWeights& weights, std::span<const double> input,
                                      std::span<const double> state, std::span<const double> history,
                                      GdnShape shape, GdnPolicy policy);
ReferenceMoe moe_projection_reference(const ReferenceMoeWeights& weights, std::span<const double> input,
                                       MoeShape shape, RouterMode mode, bool normalize);
struct ReferenceLayer {
    ReferenceResidual attention_read, ffn_read;
    ReferenceGdn attention;
    ReferenceMoe ffn;
    std::vector<double> intermediate_streams, output_streams;
};
ReferenceLayer recurrent_reference(const ReferenceLayerWeights& weights, std::span<const double> streams,
                                    std::span<const double> state, std::span<const double> convolution_history,
                                    RecurrentShape shape, GdnPolicy policy, RouterMode mode, bool normalize);

} // namespace tokspur
