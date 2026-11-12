#pragma once

#include "tokspur/attention.hpp"
#include "tokspur/ple.hpp"
#include "tokspur/recurrent.hpp"

#include <array>
#include <span>

namespace tokspur {

struct VerifyNaxMoeScratch {
    [[nodiscard]] std::span<const GpuView> views() const { return {}; }
};

struct VerifyNaxScopeGuard {
    MetalGraph& g;
    bool active = false, target = false;
    VerifyNaxScopeGuard(MetalGraph& graph, bool enable, bool target_verification = false) : g(graph), active(enable), target(enable && target_verification) {
        if (active) g.set_verify_nax_scope(true);
        if (target) g.set_target_verification_role(true);
    }
    ~VerifyNaxScopeGuard() {
        if (active) {
            try { if (target) g.set_target_verification_role(false); g.set_verify_nax_scope(false); } catch (...) {}
        }
    }
    VerifyNaxScopeGuard(const VerifyNaxScopeGuard&) = delete;
    VerifyNaxScopeGuard& operator=(const VerifyNaxScopeGuard&) = delete;
};

[[nodiscard]] bool verify_nax_dense_q8_supported(const MetalGraph& g, GpuMatrix matrix, uint32_t columns, uint32_t rows);
[[nodiscard]] bool verify_nax_bf16_supported(const MetalGraph& g, GpuMatrix matrix, uint32_t columns, uint32_t rows);
[[nodiscard]] bool verify_nax_hc_rows_supported(const MetalGraph& g, ResidualShape shape);
[[nodiscard]] bool verify_nax_router_window_supported(const MetalGraph& g, const MoeWeights& weights, MoeShape shape);
[[nodiscard]] bool verify_nax_grouped_moe_supported(const MoeWeights& weights, MoeShape shape, uint32_t steps);
// The dispatch estimator runs before the active scope guard. These predicates
// depend only on immutable compilation and shape, never transient Graph scope.
[[nodiscard]] constexpr bool verify_nax_batch_top10_supported(MoeShape shape, uint32_t steps) {
    return verify_nax_batch_top10_mixture_enabled && steps >= 2 && steps <= 8 &&
        shape.experts >= 1 && shape.experts <= 512 && shape.selected >= 1 &&
        shape.selected <= 64 && shape.selected <= shape.experts;
}
[[nodiscard]] constexpr bool verify_nax_batch_mixture_supported(MoeShape shape, uint32_t steps) {
    return verify_nax_batch_top10_mixture_enabled && steps >= 2 && steps <= 8 &&
        shape.channels >= 32 && shape.channels <= 2560 && shape.channels % 32 == 0 &&
        shape.selected >= 1 && shape.selected <= 64;
}
[[nodiscard]] bool verify_nax_qsa_batch_supported(QsaShape shape);

[[nodiscard]] uint64_t verify_nax_projection_dispatches(const MetalGraph& g, GpuMatrix matrix, uint32_t columns, uint32_t rows, uint32_t steps);
void encode_verify_nax_projection(MetalGraph& g, GpuMatrix matrix, GpuView input, GpuView output, uint32_t columns, uint32_t rows, uint32_t steps);

[[nodiscard]] uint64_t verify_nax_residual_read_dispatches(const MetalGraph& g, const ResidualWeights& weights, ResidualShape shape, uint32_t steps);
void encode_verify_nax_residual_read(MetalGraph& g, const ResidualWeights& weights, const ResidualScratch& scratch, GpuView input_streams, ResidualShape shape, uint32_t steps, float epsilon);
void encode_verify_nax_residual_read_prenorm(MetalGraph& g, const ResidualWeights& weights, const ResidualScratch& scratch, ResidualShape shape, uint32_t steps);

[[nodiscard]] uint64_t verify_nax_residual_write_dispatches(const MetalGraph& g, ResidualShape shape, uint32_t steps);
void encode_verify_nax_residual_write(MetalGraph& g, GpuView streams, GpuView values, GpuView scales, GpuView destination, ResidualShape shape, uint32_t steps);

[[nodiscard]] bool verify_nax_hc_write_rms_gains_supported(const MetalGraph& g, GpuView gains, ResidualShape shape);
[[nodiscard]] bool verify_nax_hc_write_rms_supported(const MetalGraph& g, const ResidualWeights& next_read, ResidualShape shape);
void encode_verify_nax_residual_write_rms(MetalGraph& g, GpuView streams, GpuView values, GpuView scales, GpuView next_gains,
    GpuView destination, GpuView next_normalized, ResidualShape shape, uint32_t steps, float next_epsilon);
[[nodiscard]] uint64_t verify_nax_residual_write_read_dispatches(const MetalGraph& g, const ResidualWeights& next_read, ResidualShape shape, uint32_t steps);
void encode_verify_nax_residual_write_read(MetalGraph& g, const ResidualWeights& next_read, const ResidualScratch& scratch,
    GpuView streams, GpuView values, GpuView scales, GpuView destination, ResidualShape shape, uint32_t steps, float epsilon);

[[nodiscard]] uint64_t verify_nax_moe_dispatches(const MetalGraph& g, const MoeWeights& weights, MoeShape shape, uint32_t steps);
void encode_verify_nax_moe(MetalGraph& g, const MoeWeights& weights, const MoeScratch& scratch, const VerifyNaxMoeScratch& nax_moe, GpuView input, GpuView output, MoeShape shape, RouterMode mode, bool normalise, uint32_t steps);

[[nodiscard]] uint64_t verify_nax_gdn_core_dispatches(const MetalGraph& g, const GdnWeights& weights, GdnShape shape, uint32_t steps, bool ported);
void encode_verify_nax_gdn_f32_gates(MetalGraph& g, const GdnWeights& weights, const GdnScratch& scratch, GpuView input, GdnShape shape, GdnPolicy policy, uint32_t steps);

[[nodiscard]] uint64_t verify_nax_recurrent_layer_dispatches(const MetalGraph& g, const RecurrentWeights& weights, RecurrentShape shape, uint32_t steps, bool ported);
[[nodiscard]] uint64_t verify_nax_attention_layer_dispatches(const MetalGraph& g, const AttentionLayerWeights& weights, AttentionLayerShape shape, uint32_t base_prefix, uint32_t steps);

[[nodiscard]] bool verify_nax_ple_supported(const MetalGraph& g, const PleWeights& weights, const RecurrentScratch& batch_scratch, PleShape shape);
[[nodiscard]] uint64_t verify_nax_ple_dispatches(const MetalGraph& g, const PleWeights& weights, const RecurrentScratch& batch_scratch, PleShape shape, uint32_t steps);
void encode_verify_nax_ple(MetalGraph& g, const PleWeights& weights, const PleScratch& single_scratch, const RecurrentScratch& batch_scratch, GpuView embeddings, GpuView streams, GpuView current_memory, GpuView prefix_memory, GpuView next_memory, GpuView output, PleShape shape, uint32_t steps, float epsilon);

[[nodiscard]] bool verify_nax_final_supported(const MetalGraph& g, const MixerWeights& mixer, GpuMatrix output, const ResidualScratch& scratch, ResidualShape shape, uint32_t vocabulary);
[[nodiscard]] uint64_t verify_nax_final_dispatches(const MetalGraph& g, const MixerWeights& mixer, GpuMatrix output, const ResidualScratch& scratch, ResidualShape shape, uint32_t vocabulary, uint32_t steps, bool input_prenormalized = false);
void encode_verify_nax_final_logits(MetalGraph& g, const MixerWeights& mixer, GpuMatrix output, const ResidualScratch& scratch, GpuView streams, GpuView mixed, GpuView logits, ResidualShape shape, uint32_t vocabulary, uint32_t steps, float epsilon, bool input_prenormalized = false);

} // namespace tokspur
