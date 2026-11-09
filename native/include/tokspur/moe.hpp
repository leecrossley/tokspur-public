#pragma once

#include "tokspur/graph.hpp"

#include <array>

namespace tokspur {

struct GpuMatrix { GpuView view; uint32_t encoding; };
struct MoeShape { uint32_t channels, hidden, experts, selected; };
struct MoeWeights {
    GpuMatrix router, gate, up, down, shared_gate, shared_up, shared_down, shared_router;
};
struct MoeScratch {
    GpuView logits, indices, weights, gate, up, down, shared_gate, shared_up, shared_down, shared_logit;
    [[nodiscard]] std::array<GpuView, 10> views() const;
};

MoeScratch allocate_moe_scratch(MetalGraph& graph, MoeShape shape);
void release_moe_scratch(MetalGraph& graph, const MoeScratch& scratch);

[[nodiscard]] inline bool is_planar_moe(const MetalGraph& g, const MoeWeights& w, MoeShape s) {
    const auto lo = g.planar_matrix_shape(w.gate.view);
    const auto hi = g.planar_matrix_shape(w.up.view);
    const auto dn = g.planar_matrix_shape(w.down.view);
    if (!lo || !hi || !dn) return false;
    return (lo->kind == 2u || lo->kind == 4u || lo->kind == 6u) &&
           (hi->kind == 3u || hi->kind == 5u || hi->kind == 7u) &&
           (dn->kind == 8u || dn->kind == 9u) &&
           lo->source_type == w.gate.encoding && hi->source_type == w.up.encoding &&
           dn->source_type == w.down.encoding && lo->source_type == hi->source_type &&
           lo->columns == s.channels && lo->rows == s.hidden && lo->experts == s.experts &&
           hi->columns == s.channels && hi->rows == s.hidden && hi->experts == s.experts &&
           dn->columns == s.hidden && dn->rows == s.channels && dn->experts == s.experts;
}

// Composes 14 owned dispatches in the caller's current batch. It neither
// allocates/reads/commits nor selects any expert on the CPU. No normalization or
// checkpoint-specific input transform is silently added around this sublayer.
void encode_moe(MetalGraph& graph, const MoeWeights& weights, const MoeScratch& scratch,
                 GpuView input, GpuView output, MoeShape shape, RouterMode mode, bool normalize_selected
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                 , uint32_t route_row_offset = 0
#endif
                 );

} // namespace tokspur
