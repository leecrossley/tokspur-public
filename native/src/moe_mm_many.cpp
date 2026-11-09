#include "tokspur/moe_mm_many.hpp"
#include "tokspur/operator_checkpoints.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>

namespace tokspur {
namespace {
void need(bool ok,const char* why){if(!ok)throw FormatError(why);}
void matrix(GpuMatrix w,uint32_t c,uint32_t r,uint32_t e=1){
    const auto& f=encoding(w.encoding);need(c%f.elements_per_block==0,"tiled MoE matrix block width");
    need(w.view.bytes==checked_mul(checked_mul(uint64_t(c/f.elements_per_block)*f.bytes_per_block,r),e),"tiled MoE matrix extent");
}
void weights(MetalGraph& g,const MoeWeights& w,MoeShape s,MoeMmLayouts l){
    need(w.router.encoding==0 && w.shared_router.encoding==0 && w.shared_gate.encoding==8 && w.shared_up.encoding==8 && w.shared_down.encoding==8,"tiled MoE F32 routers/Q8 shared projections");
    matrix(w.router,s.channels,s.experts);matrix(w.shared_router,s.channels,1);
    need(g.prefill_weight_layout(w.router.view)==PrefillWeightLayout::gguf && g.prefill_weight_layout(w.shared_router.view)==PrefillWeightLayout::gguf,"tiled MoE router storage identity");
    const std::array matrices{w.gate,w.up,w.down,w.shared_gate,w.shared_up,w.shared_down};
    const std::array layouts{l.gate,l.up,l.down,l.shared_gate,l.shared_up,l.shared_down};
    const bool nax_2k = g.prefill_grouped_nax() || g.prefill_dense_nax();
    const bool planar = is_planar_moe(g, w, s);
    for(size_t i=0;i<matrices.size();++i){
        const uint32_t c=i==2 || i==5?s.hidden:s.channels,r=i==2 || i==5?s.channels:s.hidden,e=i<3?s.experts:1;
        if (i < 3 && planar) {
            need(g.prefill_weight_layout(matrices[i].view)==layouts[i],"tiled MoE explicit registered layout mismatch");
            continue;
        }
        matrix(matrices[i],c,r,e);need(g.prefill_weight_layout(matrices[i].view)==layouts[i],"tiled MoE explicit registered layout mismatch");
        if(i<3)static_cast<void>(grouped_mm_plan(matrices[i].encoding,layouts[i],c,r,e,s.selected,1,uint64_t(c)*4,uint64_t(r)*4,i<2,nax_2k));
        else static_cast<void>(dense_mm_q8_plan(layouts[i],c,r,1,uint64_t(c)*4,uint64_t(r)*4,nax_2k));
    }
}
GpuRowsF32 rows(GpuView v,uint32_t n,uint32_t c){return {v,n,c,uint64_t(c)*4};}
GpuView one(MetalGraph& g,GpuView v,uint64_t stride,uint64_t bytes,uint32_t t){return g.slice(v,uint64_t(t)*stride,bytes);}
MoeScratch token(MetalGraph& g,const MoeScratch& s,MoeShape shape,uint32_t at){
    const auto b=moe_mm_many_bytes(shape,1);const auto all=s.views();std::array<GpuView,10> v{};
    for(size_t i=0;i<v.size();++i)v[i]=one(g,all[i],b.views[i],b.views[i],at);
    return {v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9]};
}
GpuView active(MetalGraph& g,GpuView v,uint64_t width,uint32_t n){return g.slice(v,0,checked_mul(checked_mul(width,n),4));}
}
MoeMmManyScratch allocate_moe_mm_many_scratch(MetalGraph& g,MoeShape shape,uint32_t capacity,bool allow_nax_2k){
    const auto b=moe_mm_many_bytes(shape,capacity,allow_nax_2k);std::array<GpuView,13> v{};
    try{for(size_t i=0;i<v.size();++i)v[i]=g.zeros(b.views[i],"bounded tiled MoE scratch");}catch(...){for(auto x:v)if(x.owner)g.release(x);throw;}
    return {{v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9]},v[10],v[11],v[12],capacity};
}
void release_moe_mm_many_scratch(MetalGraph& g,const MoeMmManyScratch& s){for(auto v:s.views())if(v.owner)g.release(v);}
void prepare_moe_mm_many(MetalGraph& g,const MoeWeights& w,MoeShape shape,MoeMmLayouts l){
    static_cast<void>(moe_mm_many_bytes(shape,1,g.prefill_grouped_nax()));weights(g,w,shape,l);
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    g.prepare_moe_router_f32_batch();
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    g.prepare_moe_router_window();
#endif
    if (is_planar_moe(g, w, shape)) {
        g.prepare_planar_moe_v3(w.gate.encoding, w.down.encoding, true,
            g.prefill_grouped_nax() ? 2048u : 512u, shape.experts, shape.selected, shape.channels, shape.hidden);
    } else {
        g.prepare_grouped_mm(w.gate.encoding,l.gate);g.prepare_grouped_mm(w.up.encoding,l.up);g.prepare_grouped_mm(w.down.encoding,l.down);
    }
    g.prepare_dense_mm_q8(l.shared_gate);g.prepare_dense_mm_q8(l.shared_up);g.prepare_dense_mm_q8(l.shared_down);
}
uint64_t moe_mm_many_dispatches(const MetalGraph& g,const MoeWeights& w,MoeShape s,uint32_t n){
    const bool grouped_nax =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        g.prefill_grouped_nax();
#else
        false;
#endif
    const uint32_t max_chunk = grouped_nax ? 2048u : std::min(512u, s.selected ? uint32_t(10240 / s.selected) : 512u);
    if (n > max_chunk) {
        uint64_t total = 0;
        for (uint32_t offset = 0; offset < n; offset += max_chunk) {
            total = checked_add(total, moe_mm_many_dispatches(g, w, s, std::min(max_chunk, n - offset)));
        }
        return total;
    }
    static_cast<void>(moe_mm_many_bytes(s,n,grouped_nax));
    const bool shared_nax_swiglu = g.can_nax_dense_q8_swiglu(
        g.prefill_weight_layout(w.shared_gate.view), g.prefill_weight_layout(w.shared_up.view), s.channels, s.hidden, n);
    const bool shared_multiseg = !shared_nax_swiglu && g.can_dense_mm_q8_multiseg({
        g.prefill_weight_layout(w.shared_gate.view), g.prefill_weight_layout(w.shared_up.view)}, n);
    const bool planar = is_planar_moe(g, w, s);
    const bool fuse_expert_gu =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        !planar && grouped_nax && w.gate.encoding == w.up.encoding &&
        g.prefill_weight_layout(w.gate.view) == g.prefill_weight_layout(w.up.view);
#else
        false;
#endif
    const bool handoff_expert_gu =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        fuse_expert_gu && !g.prefill_grouped_auto() &&
        g.can_grouped_mm_handoff_bf16(s.experts, s.selected, n, s.channels, s.hidden);
#else
        false;
#endif
    const uint32_t expert_dispatches = planar ? (n == 1u ? 2u : (n <= 8u ? 3u : 5u)) :
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        handoff_expert_gu ? 5u : fuse_expert_gu ? 6u : grouped_nax ? 9u : g.prefill_grouped_f32() ? 7u :
#endif
        11u;
    const uint32_t expert_swiglu_saved = (!planar && !fuse_expert_gu && g.swiglu_fused_enabled()) ? 1u : 0u;
    const uint32_t shared_extra = shared_nax_swiglu ? 0u : (3u - (g.swiglu_fused_enabled() ? 1u : 0u) - (shared_multiseg ? 1u : 0u));
    const uint32_t shared_dispatches = shared_extra + g.dense_mm_q8_dispatches(s.channels) + g.dense_mm_q8_dispatches(s.hidden);
    const uint32_t fixed = expert_dispatches - expert_swiglu_saved + shared_dispatches;
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    const uint32_t max_router_window = grouped_nax ? 2048u : 512u;
    if (g.prefill_router_window() && w.router.encoding == 0 && w.shared_router.encoding == 0 && s.channels == 2560 && s.experts == 512 && n >= 2 && n <= max_router_window) {
        const uint32_t router_proj = (g.prefill_dense_nax() || grouped_nax) ? 2u : 1u;
        const uint32_t reuse_router_bf16 = (g.prefill_dense_nax() && !shared_multiseg && (planar || (!g.prefill_grouped_auto() && !g.router_margin_audit()))) ? 1u : 0u;
        return fixed - reuse_router_bf16 + router_proj + 3u;
    }
#endif
    uint64_t router=uint64_t(n)*g.gemv_dispatches(w.router.encoding,s.channels,s.experts);
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    if(g.moe_router_f32_batch_enabled() && w.router.encoding==0 && s.channels==2560 && s.experts==512 && n>1)router=1;
#endif
    return fixed+router+uint64_t(n)*(2+g.gemv_dispatches(w.shared_router.encoding,s.channels,1));
}
void encode_moe_mm_many(MetalGraph& g,const MoeWeights& w,const MoeMmManyScratch& s,GpuRowsF32 input,GpuRowsF32 output,MoeShape shape,RouterMode mode,bool normalise,MoeMmLayouts l){
    const bool grouped_nax =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        g.prefill_grouped_nax();
#else
        false;
#endif
    const uint32_t max_moe_chunk = grouped_nax ? std::min(s.capacity, 2048u) : std::min({s.capacity, 512u, shape.selected ? uint32_t(10240 / shape.selected) : 512u});
    if (input.rows > max_moe_chunk) {
        need(max_moe_chunk > 0 && input.rows == output.rows && input.columns == shape.channels && output.columns == shape.channels, "tiled MoE chunked rows extent");
        for (uint32_t r_offset = 0; r_offset < input.rows; r_offset += max_moe_chunk) {
            const uint32_t chunk_r = std::min(max_moe_chunk, input.rows - r_offset);
            const auto in_chunk_view = g.slice(input.view, uint64_t(r_offset) * input.stride_bytes, checked_add(checked_mul(chunk_r - 1, input.stride_bytes), uint64_t(input.columns) * 4));
            const auto out_chunk_view = g.slice(output.view, uint64_t(r_offset) * output.stride_bytes, checked_add(checked_mul(chunk_r - 1, output.stride_bytes), uint64_t(output.columns) * 4));
            encode_moe_mm_many(g, w, s, {in_chunk_view, chunk_r, input.columns, input.stride_bytes}, {out_chunk_view, chunk_r, output.columns, output.stride_bytes}, shape, mode, normalise, l);
        }
        return;
    }
    try{
        const auto b=moe_mm_many_bytes(shape,s.capacity,grouped_nax);const auto all=s.views();
        need(input.rows && input.rows<=s.capacity && input.rows==output.rows && input.columns==shape.channels && output.columns==shape.channels && uint32_t(mode)<=1,"tiled MoE active rows/router mode");
        for(size_t i=0;i<all.size();++i)need(all[i].bytes==b.views[i],"tiled MoE scratch capacity extent");
        const auto n=input.rows;for(const auto r:{input,output})need(r.stride_bytes>=uint64_t(shape.channels)*4 && r.stride_bytes%16==0 && r.view.bytes>=checked_add(checked_mul(n-1,r.stride_bytes),uint64_t(shape.channels)*4),"tiled MoE strided row extent");
        weights(g,w,shape,l);
        const std::array read{input.view,w.router.view,w.gate.view,w.up.view,w.down.view,w.shared_gate.view,w.shared_up.view,w.shared_down.view,w.shared_router.view};
        std::array<GpuView,14> write{output.view};std::copy(all.begin(),all.end(),write.begin()+1);g.validate_disjoint(read,write);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"unclassified","before_router",{n,shape.channels,shape.experts,shape.selected});
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
        const uint32_t max_router_window = grouped_nax ? 2048u : 512u;
        const bool router_window=g.prefill_router_window() && shape.channels==2560 && shape.experts==512 && n>=2 && n<=max_router_window;
        if(router_window){
            const auto logits=active(g,s.rows.logits,shape.experts,n);
            g.moe_router_f32_window(w.router.view,input,rows(logits,n,shape.experts));
            TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","router_f32_window",{n,shape.channels,shape.experts,shape.selected});
            g.moe_route_rows(logits,active(g,s.rows.indices,shape.selected,n),active(g,s.rows.weights,shape.selected,n),shape.experts,shape.selected,n,mode,normalise);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","topk_rows_after_router_window",{n,shape.experts,shape.selected,0});
        }else
#endif
        {
            bool router_batched=false;
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
            router_batched=g.moe_router_f32_batch_enabled() && shape.channels==2560 && shape.experts==512 && n>1 && n<=512;
            if(router_batched){g.moe_router_f32_batch(w.router.view,input,rows(active(g,s.rows.logits,shape.experts,n),n,shape.experts));
                TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","router_f32_batched",{n,shape.channels,shape.experts,shape.selected});}
#endif
            for(uint32_t t=0;t<n;++t){const auto r=token(g,s.rows,shape,t);if(!router_batched)g.gemv(0,w.router.view,one(g,input.view,input.stride_bytes,uint64_t(shape.channels)*4,t),r.logits,shape.channels,shape.experts);g.route(r.logits,r.indices,r.weights,shape.selected,mode,normalise);}
            if(router_batched){TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","topk_after_batched_router",{n,shape.experts,shape.selected,0});}
            else{TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","router_f32_and_topk_interleaved",{n,shape.channels,shape.experts,shape.selected});}
        }
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
        g.route_attribution_rows(w.router.view,input,active(g,s.rows.logits,shape.experts,n),active(g,s.rows.indices,shape.selected,n),active(g,s.rows.weights,shape.selected,n),shape.selected,mode,normalise);
#endif
        const auto ids=active(g,s.rows.indices,shape.selected,n);
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        g.route_lock_rows(active(g,s.rows.logits,shape.experts,n),ids,active(g,s.rows.weights,shape.selected,n),n,shape.channels,shape.experts,shape.selected);
#endif
        if (is_planar_moe(g, w, shape)) {
            const auto gate = rows(s.rows.gate, n * shape.selected, shape.hidden);
            const auto down_rows = rows(s.rows.down, n * shape.selected, shape.channels);
            if (n <= 8u) {
                g.planar_moe_routed_m8(w.gate.view, w.up.view, w.down.view, ids, input, gate, down_rows, shape.experts, shape.selected);
            } else {
                g.planar_nax_grouped_moe(w.gate.view, w.up.view, w.down.view, ids, input, down_rows, s.token_map, s.counts, s.tasks, shape.experts, shape.selected);
            }
        } else {
            bool use_grouped_project = false;
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            use_grouped_project = g.prefill_grouped_f32() || g.prefill_grouped_nax();
            if (g.prefill_grouped_auto()) {
                use_grouped_project = g.evaluate_router_margin(n);
            } else if (g.router_margin_audit()) {
                static_cast<void>(g.evaluate_router_margin(n));
            }
            const bool grouped_routes_enabled = use_grouped_project;
            GroupedMmRoutes routes;
            if(grouped_routes_enabled)routes=g.grouped_mm_routes(ids,s.token_map,s.counts,s.tasks,shape.experts,shape.selected,n);
#endif
            const auto project=[&](GpuMatrix m,PrefillWeightLayout layout,GpuRowsF32 x,GpuRowsF32 y,bool broadcast){
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
                if(use_grouped_project){g.grouped_mm_project(m.encoding,layout,m.view,routes,x,y,broadcast);return;}
#endif
                g.grouped_mm(m.encoding,layout,m.view,ids,x,y,s.token_map,s.counts,s.tasks,shape.experts,shape.selected,broadcast);
            };
            const auto gate=rows(s.rows.gate,n*shape.selected,shape.hidden),up=rows(s.rows.up,n*shape.selected,shape.hidden);
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            if(use_grouped_project && grouped_nax && w.gate.encoding == w.up.encoding && l.gate == l.up){
                const bool handoff = !g.prefill_grouped_auto() && g.can_grouped_mm_handoff_bf16(shape.experts, shape.selected, n, shape.channels, shape.hidden);
                g.grouped_mm_gate_up_swiglu(w.gate.encoding,l.gate,w.gate.view,w.up.view,routes,input,gate,true,handoff);
                TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","expert_gate_up_swiglu",{n,shape.hidden,shape.selected,0});
            }else
#endif
            {
                project(w.gate,l.gate,input,gate,true);project(w.up,l.up,input,up,true);
                const auto gv=active(g,s.rows.gate,uint64_t(shape.hidden)*shape.selected,n),uv=active(g,s.rows.up,uint64_t(shape.hidden)*shape.selected,n);
                g.swiglu(gv,uv,gv);
                TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","expert_activation_multiply",{n,shape.hidden,shape.selected,0});
            }
            project(w.down,l.down,gate,rows(s.rows.down,n*shape.selected,shape.channels),false);
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            if(grouped_routes_enabled)g.end_grouped_mm_routes(routes);
#endif
        }
        const auto sg=rows(s.rows.shared_gate,n,shape.hidden),su=rows(s.rows.shared_up,n,shape.hidden);
        if (g.can_nax_dense_q8_swiglu(l.shared_gate,l.shared_up,shape.channels,shape.hidden,n)) {
            g.nax_dense_q8_swiglu(l.shared_gate,w.shared_gate.view,l.shared_up,w.shared_up.view,input,sg,g.can_reuse_nax_dense_bf16(input));
            TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","shared_gate_up_swiglu_nax",{n,shape.channels,shape.hidden*2,8});
        } else {
            if (g.can_dense_mm_q8_multiseg({l.shared_gate,l.shared_up},n)) {
                g.dense_mm_q8_multiseg2(l.shared_gate,w.shared_gate.view,l.shared_up,w.shared_up.view,input,sg,su);
                TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","shared_gate_up_multiseg2",{n,shape.channels,shape.hidden*2,8});
            } else {
                g.dense_mm_q8(l.shared_gate,w.shared_gate.view,input,sg,g.can_reuse_nax_dense_bf16(input));
                TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","shared_gate",{n,shape.channels,shape.hidden,8});
                g.dense_mm_q8(l.shared_up,w.shared_up.view,input,su,g.can_reuse_nax_dense_bf16(input));
                TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","shared_up",{n,shape.channels,shape.hidden,8});
            }
            const auto sgv=active(g,sg.view,shape.hidden,n),suv=active(g,su.view,shape.hidden,n);
            g.swiglu(sgv,suv,sgv);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","shared_activation_multiply",{n,shape.hidden,0,0});
        }
        g.dense_mm_q8(l.shared_down,w.shared_down.view,sg,rows(s.rows.shared_down,n,shape.channels));
        TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","shared_down",{n,shape.hidden,shape.channels,8});
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
        if(router_window){
            const auto shared_logits=active(g,s.rows.shared_logit,1,n);
            g.moe_shared_router_rows(w.shared_router.view,input,shared_logits);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","shared_router_f32_rows",{n,shape.channels,1,0});
            g.moe_mixture_rows(active(g,s.rows.down,uint64_t(shape.selected)*shape.channels,n),active(g,s.rows.weights,shape.selected,n),active(g,s.rows.shared_down,shape.channels,n),shared_logits,output,shape.selected);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","mixture_rows",{n,shape.channels,shape.selected,0});
        }else
#endif
        {
            for(uint32_t t=0;t<n;++t){const auto r=token(g,s.rows,shape,t);g.gemv(0,w.shared_router.view,one(g,input.view,input.stride_bytes,uint64_t(shape.channels)*4,t),r.shared_logit,shape.channels,1);g.mixture(r.down,r.weights,r.shared_down,r.shared_logit,one(g,output.view,output.stride_bytes,uint64_t(shape.channels)*4,t));}

            TOKSPUR_OPERATOR_CHECKPOINT(g,"moe","shared_router_f32_and_mixture_interleaved",{n,shape.channels,shape.selected,0});
        }
    }catch(...){try{g.abort();}catch(const FormatError&){}throw;}
}
}
