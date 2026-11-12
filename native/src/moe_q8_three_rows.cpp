#include "tokspur/moe_q8_three_rows.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/operator_checkpoints.hpp"
#include <algorithm>

namespace tokspur {
namespace {
void need(bool ok, const char* why) { if (!ok) throw FormatError(why); }
std::array<uint64_t, 10> counts(MoeShape s) {
    need(s.channels && s.channels <= 65536 && s.channels % 32 == 0 && s.hidden && s.hidden <= 65536 && s.hidden % 32 == 0 &&
        s.experts && s.experts <= 4096 && s.selected && s.selected <= 64 && s.selected <= s.experts, "three-row MoE geometry");
    return {s.experts,s.selected,s.selected,uint64_t(s.hidden)*s.selected,uint64_t(s.hidden)*s.selected,
        uint64_t(s.channels)*s.selected,s.hidden,s.hidden,s.channels,1};
}
void weights(const MetalGraph& g, const MoeWeights& w, MoeShape s) {
    static_cast<void>(counts(s));
    need(w.shared_gate.encoding == 8 && w.shared_up.encoding == 8 && w.shared_down.encoding == 8,
        "three-row MoE requires shared Q8 projections");
    const auto matrix = [](GpuMatrix m, uint32_t c, uint32_t r, uint32_t e) {
        const auto& f = encoding(m.encoding);
        need(c % f.elements_per_block == 0 && m.view.bytes == checked_mul(checked_mul(uint64_t(c / f.elements_per_block) * f.bytes_per_block,r),e), "three-row MoE matrix extent");
    };
    matrix(w.router,s.channels,s.experts,1);matrix(w.shared_router,s.channels,1,1);
    if (!is_planar_moe(g, w, s)) {
        matrix(w.gate,s.channels,s.hidden,s.experts);matrix(w.up,s.channels,s.hidden,s.experts);matrix(w.down,s.hidden,s.channels,s.experts);
    }
    matrix(w.shared_gate,s.channels,s.hidden,1);matrix(w.shared_up,s.channels,s.hidden,1);matrix(w.shared_down,s.hidden,s.channels,1);
}
GpuView row(MetalGraph& g, GpuView v, uint64_t width, uint32_t t) { return g.slice(v,uint64_t(t)*width*4,width*4); }
}
bool moe_three_union_layer_supported(const MetalGraph& g, const MoeWeights& w, MoeShape s) {
    weights(g,w,s);
    if (is_planar_moe(g, w, s)) return false;
    return g.moe_three_union_enabled() && w.router.encoding == 0 && w.shared_router.encoding == 0 &&
        w.gate.encoding == w.up.encoding &&
        g.moe_three_union_matrix_supported(w.gate.encoding, w.gate.view, s.channels, s.hidden, s.experts, false) &&
        g.moe_three_union_matrix_supported(w.up.encoding, w.up.view, s.channels, s.hidden, s.experts, false) &&
        g.moe_three_union_matrix_supported(w.down.encoding, w.down.view, s.hidden, s.channels, s.experts, true);
}
uint64_t moe_q8_three_rows_saved_dispatches(const MetalGraph& g, const MoeWeights& w, MoeShape s) {
    weights(g,w,s);
    const uint64_t old = g.gemv_dispatches(8,s.channels,s.hidden)*2ull + g.gemv_dispatches(8,s.hidden,s.channels);
    if (moe_three_union_layer_supported(g,w,s)) {
        const uint64_t single_moe = (g.swiglu_fused_enabled() ? 12ull : 14ull) +
            (g.gemv_dispatches(w.router.encoding,s.channels,s.experts) - 1ull) +
            (g.gemv_dispatches(w.shared_router.encoding,s.channels,1) - 1ull) +
            (old - 3ull);
        return 3ull * single_moe - 11ull;
    }
    const uint64_t planar_saved = is_planar_moe(g, w, s) ? 3ull : 0ull;
    if (g.swiglu_fused_enabled()) return 3ull * old - 1ull + planar_saved;
    // Three shared GEMVs per ordinary token become three physical multi-input dispatches. Elementwise operations retain their per-row scheduling.
    return 3ull * old - 3ull + planar_saved;
}
void encode_moe_q8_three_rows(MetalGraph& g, const MoeWeights& w, const MoeScratch& s, GpuView input, GpuView output,
    MoeShape shape, RouterMode mode, bool normalize, uint32_t capacity) {
    try {
        need(g.q8_three_rows_enabled() && capacity >= 3 && capacity <= 512 && uint32_t(mode) <= 1, "three-row MoE selector/capacity/router");
        need((g.kernel_policy()[1] & 1) != 0, "three-row MoE requires the original block-Q8 scalar policy");
        weights(g,w,shape);const auto n=counts(shape);const auto v=s.views();
        need(input.bytes==uint64_t(capacity)*shape.channels*4 && output.bytes==input.bytes,"three-row MoE input/output capacity");
        for(size_t i=0;i<v.size();++i)need(v[i].bytes==n[i]*capacity*4,"three-row MoE scratch capacity");
        const std::array read{input,w.router.view,w.gate.view,w.up.view,w.down.view,w.shared_gate.view,w.shared_up.view,w.shared_down.view,w.shared_router.view};
        std::array<GpuView,11> write{output};std::copy(v.begin(),v.end(),write.begin()+1);g.validate_disjoint(read,write);
        const auto active=[&](GpuView x,uint64_t width){return g.slice(x,0,uint64_t(3)*width*4);};
        if (moe_three_union_layer_supported(g,w,shape)) {
            const auto x3 = active(input, shape.channels);
            const auto logits3 = active(s.logits, shape.experts);
            const auto indices3 = active(s.indices, shape.selected);
            const auto weights3 = active(s.weights, shape.selected);
            const auto gate3 = active(s.gate, uint64_t(shape.hidden) * shape.selected);
            const auto down3 = active(s.down, uint64_t(shape.channels) * shape.selected);
            g.moe_three_router(w.router.view, x3, logits3, shape.channels, shape.experts);
            g.moe_three_route(logits3, indices3, weights3, shape.experts, shape.selected, mode, normalize);
            TOKSPUR_OPERATOR_CHECKPOINT(g, "moe", "three_union_router_topk", {3, shape.channels, shape.experts, shape.selected});
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
            g.route_lock_rows(logits3, indices3, weights3, 3, shape.channels, shape.experts, shape.selected, 0);
#endif
            g.moe_three_build_union(indices3, shape.experts, shape.selected, shape.hidden, shape.channels);
            g.moe_three_gate_up_swiglu(w.gate.encoding, w.gate.view, w.up.view, x3, gate3,
                shape.channels, shape.hidden, shape.experts, shape.selected);
            g.moe_three_down(w.down.encoding, w.down.view, gate3, down3,
                shape.hidden, shape.channels, shape.experts, shape.selected);
            TOKSPUR_OPERATOR_CHECKPOINT(g, "moe", "three_union_routed_experts", {3, shape.channels, shape.hidden, shape.selected});
            g.q8_matvec_many(w.shared_gate.view, x3, active(s.shared_gate, shape.hidden), shape.channels, shape.hidden, 3);
            g.q8_matvec_many(w.shared_up.view, x3, active(s.shared_up, shape.hidden), shape.channels, shape.hidden, 3);
            g.swiglu_direct(active(s.shared_gate, shape.hidden), active(s.shared_up, shape.hidden), active(s.shared_gate, shape.hidden));
            g.q8_matvec_many(w.shared_down.view, active(s.shared_gate, shape.hidden), active(s.shared_down, shape.channels), shape.hidden, shape.channels, 3);
            TOKSPUR_OPERATOR_CHECKPOINT(g, "dense_q8", "three_union_shared_experts", {3, shape.channels, shape.hidden, 8});
            g.moe_three_shared_router(w.shared_router.view, x3, active(s.shared_logit, 1), shape.channels);
            g.moe_three_mixture(down3, weights3, active(s.shared_down, shape.channels), active(s.shared_logit, 1),
                active(output, shape.channels), shape.channels, shape.selected);
            TOKSPUR_OPERATOR_CHECKPOINT(g, "moe", "three_union_shared_router_mixture", {3, shape.channels, shape.selected, 0});
            return;
        }
        const auto token=[&](uint32_t t){std::array<GpuView,10> x{};for(size_t i=0;i<x.size();++i)x[i]=row(g,v[i],n[i],t);return MoeScratch{x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]};};
        const bool planar = is_planar_moe(g, w, shape);
        // Rows are independent: every projection, then every top-k, so a concurrent batch needs one barrier here instead of one per row.
        const auto rows3=[&](GpuView x,uint32_t width){return GpuRowsF32{active(x,width),3,width,uint64_t(width)*4};};
        g.gemv_rows(w.router.encoding,w.router.view,rows3(input,shape.channels),rows3(s.logits,shape.experts));
        for(uint32_t t=0;t<3;++t){const auto r=token(t);g.route(r.logits,r.indices,r.weights,shape.selected,mode,normalize);}
        TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","three_row_router_topk",{3,shape.channels,shape.experts,shape.selected});
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        for(uint32_t t=0;t<3;++t){const auto r=token(t);g.route_lock_rows(r.logits,r.indices,r.weights,1,shape.channels,shape.experts,shape.selected,t);}
#endif
        if (!planar) {
            // The same per-row dispatches, stage by stage: each stage reads only its own row's earlier results.
            for(uint32_t t=0;t<3;++t){const auto r=token(t);const auto x=row(g,input,shape.channels,t);
                g.routed_gemv(w.gate.encoding,w.gate.view,r.indices,x,r.gate,shape.channels,shape.hidden,shape.experts,true);
                g.routed_gemv(w.up.encoding,w.up.view,r.indices,x,r.up,shape.channels,shape.hidden,shape.experts,true);}
            for(uint32_t t=0;t<3;++t){const auto r=token(t);g.swiglu(r.gate,r.up,r.gate);}
            for(uint32_t t=0;t<3;++t){const auto r=token(t);g.routed_gemv(w.down.encoding,w.down.view,r.indices,r.gate,r.down,shape.hidden,shape.channels,shape.experts,false);}
            TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","three_row_routed_experts",{3,shape.channels,shape.hidden,shape.selected});
        }
        if (planar) {
            g.planar_moe_routed_m8(
                w.gate.view, w.up.view, w.down.view, active(s.indices, shape.selected),
                {active(input, shape.channels), 3, shape.channels, uint64_t(shape.channels) * 4},
                {active(s.gate, uint64_t(shape.hidden) * shape.selected), 3 * shape.selected, shape.hidden, uint64_t(shape.hidden) * 4},
                {active(s.down, uint64_t(shape.channels) * shape.selected), 3 * shape.selected, shape.channels, uint64_t(shape.channels) * 4},
                shape.experts, shape.selected);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","three_row_routed_experts",{3,shape.channels,shape.hidden,shape.selected});
        }
        g.q8_matvec_many(w.shared_gate.view,active(input,shape.channels),active(s.shared_gate,shape.hidden),shape.channels,shape.hidden,3);
        g.q8_matvec_many(w.shared_up.view,active(input,shape.channels),active(s.shared_up,shape.hidden),shape.channels,shape.hidden,3);
        if (g.swiglu_fused_enabled()) {
            g.swiglu(active(s.shared_gate,shape.hidden),active(s.shared_up,shape.hidden),active(s.shared_gate,shape.hidden));
        } else {
            for(uint32_t t=0;t<3;++t){const auto r=token(t);g.activate(r.shared_gate,r.shared_gate,Activation::silu);g.multiply(r.shared_gate,r.shared_up,r.shared_gate);}
        }
        g.q8_matvec_many(w.shared_down.view,active(s.shared_gate,shape.hidden),active(s.shared_down,shape.channels),shape.hidden,shape.channels,3);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","three_row_shared_experts",{3,shape.channels,shape.hidden,8});
        g.gemv_rows(w.shared_router.encoding,w.shared_router.view,rows3(input,shape.channels),rows3(s.shared_logit,1));
        for(uint32_t t=0;t<3;++t){const auto r=token(t);g.mixture(r.down,r.weights,r.shared_down,r.shared_logit,row(g,output,shape.channels,t));}
        TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","three_row_shared_router_mixture",{3,shape.channels,shape.selected,0});
    } catch(...) { try {g.abort();}catch(const FormatError&){}throw; }
}
}
