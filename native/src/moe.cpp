#include "tokspur/moe.hpp"
#include "tokspur/operator_checkpoints.hpp"
#include "tokspur/gguf.hpp"

namespace tokspur {
namespace {
void shape_check(MoeShape shape) {
    if (!shape.channels || shape.channels > 65536 || !shape.hidden || shape.hidden > 65536 || !shape.experts ||
        shape.experts > 4096 || !shape.selected || shape.selected > 64 || shape.selected > shape.experts) throw FormatError("invalid mixture shape");
}
}

std::array<GpuView, 10> MoeScratch::views() const {
    return {logits, indices, weights, gate, up, down, shared_gate, shared_up, shared_down, shared_logit};
}

MoeScratch allocate_moe_scratch(MetalGraph& graph, MoeShape shape) {
    shape_check(shape);
    std::array<GpuView, 10> views{};
    const std::array<uint64_t, 10> counts{shape.experts, shape.selected, shape.selected,
        checked_mul(shape.hidden, shape.selected), checked_mul(shape.hidden, shape.selected), checked_mul(shape.channels, shape.selected),
        shape.hidden, shape.hidden, shape.channels, 1};
    size_t made = 0;
    try {
        for (; made < counts.size(); ++made) views[made] = graph.zeros(checked_mul(counts[made], 4), "mixture scratch");
    } catch (...) {
        for (size_t i = 0; i < made; ++i) graph.release(views[i]);
        throw;
    }
    return {views[0], views[1], views[2], views[3], views[4], views[5], views[6], views[7], views[8], views[9]};
}

void release_moe_scratch(MetalGraph& graph, const MoeScratch& scratch) {
    for (const auto view : scratch.views()) graph.release(view);
}

void encode_moe(MetalGraph& graph, const MoeWeights& w, const MoeScratch& s, GpuView input, GpuView output,
                 MoeShape shape, RouterMode mode, bool normalize_selected
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                 , uint32_t route_row_offset
#endif
                 ) {
    shape_check(shape);

        TOKSPUR_OPERATOR_CHECKPOINT(graph,"unclassified","before_router",{shape.channels,shape.experts,1,0});
    graph.gemv(w.router.encoding, w.router.view, input, s.logits, shape.channels, shape.experts);
        TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","router_f32",{shape.channels,shape.experts,1,0});
    graph.route(s.logits, s.indices, s.weights, shape.selected, mode, normalize_selected);
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    graph.route_lock_rows(s.logits,s.indices,s.weights,1,shape.channels,shape.experts,shape.selected,route_row_offset);
#endif
        TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","topk",{shape.experts,shape.selected,1,0});
    if (is_planar_moe(graph, w, shape)) {
        graph.planar_moe_routed_m8(
            w.gate.view, w.up.view, w.down.view, s.indices,
            {input, 1, shape.channels, uint64_t(shape.channels) * 4},
            {s.gate, shape.selected, shape.hidden, uint64_t(shape.hidden) * 4},
            {s.down, shape.selected, shape.channels, uint64_t(shape.channels) * 4},
            shape.experts, shape.selected);
        TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","down_routed_gemv",{shape.hidden,shape.channels,shape.selected,shape.experts});
    } else {
        graph.routed_gemv(w.gate.encoding, w.gate.view, s.indices, input, s.gate, shape.channels, shape.hidden, shape.experts, true);
            TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","gate_routed_gemv",{shape.channels,shape.hidden,shape.selected,shape.experts});
        graph.routed_gemv(w.up.encoding, w.up.view, s.indices, input, s.up, shape.channels, shape.hidden, shape.experts, true);
            TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","up_routed_gemv",{shape.channels,shape.hidden,shape.selected,shape.experts});
        graph.swiglu(s.gate, s.up, s.gate);
            TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","expert_activation_multiply",{shape.hidden,shape.selected,1,0});
        graph.routed_gemv(w.down.encoding, w.down.view, s.indices, s.gate, s.down, shape.hidden, shape.channels, shape.experts, false);
            TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","down_routed_gemv",{shape.hidden,shape.channels,shape.selected,shape.experts});
    }
    graph.gemv(w.shared_gate.encoding, w.shared_gate.view, input, s.shared_gate, shape.channels, shape.hidden);
        TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","shared_gate_gemv",{shape.channels,shape.hidden,1,0});
    graph.gemv(w.shared_up.encoding, w.shared_up.view, input, s.shared_up, shape.channels, shape.hidden);
        TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","shared_up_gemv",{shape.channels,shape.hidden,1,0});
    graph.swiglu(s.shared_gate, s.shared_up, s.shared_gate);
        TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","shared_activation_multiply",{shape.hidden,1,0,0});
    graph.gemv(w.shared_down.encoding, w.shared_down.view, s.shared_gate, s.shared_down, shape.hidden, shape.channels);
        TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","shared_down_gemv",{shape.hidden,shape.channels,1,0});
    graph.gemv(w.shared_router.encoding, w.shared_router.view, input, s.shared_logit, shape.channels, 1);
        TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","shared_router_f32",{shape.channels,1,0,0});
    graph.mixture(s.down, s.weights, s.shared_down, s.shared_logit, output);
        TOKSPUR_OPERATOR_CHECKPOINT(graph,"moe","mixture",{shape.channels,shape.selected,1,0});
}

} // namespace tokspur
