#include "tokspur/moe_many.hpp"
#include "tokspur/gguf.hpp"
#include <algorithm>

namespace tokspur {
namespace {
#ifdef TOKSPUR_PRIVATE_GROUPED_PACKED
constexpr uint32_t projection_tile = 1, projection_mode = 1;
#elif defined(TOKSPUR_PRIVATE_GROUPED_ORDERED)
constexpr uint32_t projection_tile = 8, projection_mode = 1;
#else
constexpr uint32_t projection_tile = 32, projection_mode = 32;
#endif
void require(bool ok,const char* why){if(!ok)throw FormatError(why);}
std::array<uint64_t,10> counts(MoeShape s,uint32_t capacity) {
    require(s.channels&&s.channels<=2560&&s.channels%32==0&&s.hidden&&s.hidden<=640&&s.hidden%32==0&&
        s.experts&&s.experts<=512&&s.selected&&s.selected<=16&&s.selected<=s.experts&&capacity&&capacity<=2048,"unsupported known-input FFN shape");
    const std::array<uint64_t,10> widths{s.experts,s.selected,s.selected,uint64_t(s.hidden)*s.selected,uint64_t(s.hidden)*s.selected,
        uint64_t(s.channels)*s.selected,s.hidden,s.hidden,s.channels,1};
    auto n=widths;for(auto& v:n){v=checked_mul(v,capacity);require(v*4<=(256ull<<20),"known-input FFN workspace bound");}
    require(n[3]<=16777216&&n[6]<=16777216,"known-input FFN activation vector bound");return n;
}
void bytes(GpuView v,uint64_t n){require(v.bytes==n,"known-input FFN view shape mismatch");}
void matrix(GpuMatrix w,uint32_t c,uint32_t r,uint32_t e=1) {
    const auto& f=encoding(w.encoding);require(c%f.elements_per_block==0,"known-input FFN block geometry");
    bytes(w.view,checked_mul(checked_mul(uint64_t(c/f.elements_per_block)*f.bytes_per_block,r),e));
}
void weights(const MoeWeights& w,MoeShape s) {
    require(w.router.encoding==0&&w.shared_router.encoding==0&&w.shared_gate.encoding==8&&w.shared_up.encoding==8&&w.shared_down.encoding==8,
        "known-input FFN requires declared F32 routers and Q8 shared projections");
    matrix(w.router,s.channels,s.experts);matrix(w.gate,s.channels,s.hidden,s.experts);matrix(w.up,s.channels,s.hidden,s.experts);
    matrix(w.down,s.hidden,s.channels,s.experts);matrix(w.shared_gate,s.channels,s.hidden);matrix(w.shared_up,s.channels,s.hidden);
    matrix(w.shared_down,s.hidden,s.channels);matrix(w.shared_router,s.channels,1);
}
GpuView active(MetalGraph& g,GpuView v,uint64_t width,uint32_t steps){return g.slice(v,0,checked_mul(checked_mul(width,steps),4));}
GpuView row(MetalGraph& g,GpuView v,uint64_t width,uint32_t at){return g.slice(v,uint64_t(at)*width*4,width*4);}
GpuRowsF32 rows(GpuView v,uint32_t steps,uint32_t width){return {v,steps,width,uint64_t(width)*4};}
MoeScratch token(MetalGraph& g,const MoeScratch& s,MoeShape shape,uint32_t at) {
    const auto n=counts(shape,1);const auto v=s.views();std::array<GpuView,10> result{};
    for(size_t i=0;i<v.size();++i)result[i]=row(g,v[i],n[i],at);
    return {result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9]};
}
void shared(MetalGraph& g,GpuMatrix w,GpuView input,GpuView output,uint32_t c,uint32_t r,uint32_t steps) {
    uint32_t at=0;while(at<steps){const auto n=g.q8_many_width(steps-at);
        g.q8_matvec_many(w.view,g.slice(input,uint64_t(at)*c*4,uint64_t(n)*c*4),g.slice(output,uint64_t(at)*r*4,uint64_t(n)*r*4),c,r,n);at+=n;}
}
}
std::array<GpuView,13> MoeManyScratch::views() const {
    std::array<GpuView,13> result{};const auto v=rows.views();std::copy(v.begin(),v.end(),result.begin());
    result[10]=token_map;result[11]=counts;result[12]=tasks;return result;
}
MoeManyScratch allocate_moe_many_scratch(MetalGraph& g,MoeShape s,uint32_t capacity) {
    const auto n=counts(s,capacity);const auto meta=grouped_mpp_bytes(8,{s.channels,s.hidden,s.experts,s.selected,capacity,projection_tile,projection_mode},true);
    std::array<GpuView,13> v{};try{
        for(size_t i=0;i<n.size();++i)v[i]=g.zeros(n[i]*4,"known-input FFN rows");
        v[10]=g.zeros(meta.token_map,"known-input expert map");v[11]=g.zeros(meta.counts,"known-input expert counts");v[12]=g.zeros(meta.tasks,"known-input expert tasks");
    }catch(...){for(auto x:v)if(x.owner)g.release(x);throw;}
    return {{v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9]},v[10],v[11],v[12],capacity};
}
void release_moe_many_scratch(MetalGraph& g,const MoeManyScratch& s){for(auto v:s.views())g.release(v);}
void prepare_moe_many(MetalGraph& g,const MoeWeights& w) {
    for(auto type:{w.gate.encoding,w.up.encoding,w.down.encoding})g.prepare_routed_matmul_many(type,projection_tile,projection_mode);
    g.prepare_q8_matvec_many_all();
}
uint64_t moe_many_dispatches(const MetalGraph& g,const MoeWeights& w,MoeShape s,uint32_t steps) {
    static_cast<void>(counts(s,steps));const auto chunks=g.q8_many_chunks(steps);
    return (g.swiglu_fused_enabled()?11:13)+uint64_t(chunks)*3+uint64_t(steps)*(2+g.gemv_dispatches(w.router.encoding,s.channels,s.experts)+g.gemv_dispatches(w.shared_router.encoding,s.channels,1));
}
void encode_moe_many(MetalGraph& g,const MoeWeights& w,const MoeManyScratch& s,GpuView input,GpuView output,
    MoeShape shape,RouterMode mode,bool normalize,uint32_t steps) {
    try {
        const auto n=counts(shape,s.capacity);const auto v=s.rows.views();require(steps&&steps<=s.capacity&&uint32_t(mode)<=1,"known-input FFN row count/mode");
        weights(w,shape);for(size_t i=0;i<n.size();++i)bytes(v[i],n[i]*4);
        bytes(input,uint64_t(s.capacity)*shape.channels*4);bytes(output,input.bytes);
        const auto meta=grouped_mpp_bytes(w.gate.encoding,{shape.channels,shape.hidden,shape.experts,shape.selected,s.capacity,projection_tile,projection_mode},true);
        bytes(s.token_map,meta.token_map);bytes(s.counts,meta.counts);bytes(s.tasks,meta.tasks);
        const std::array reads{input,w.router.view,w.gate.view,w.up.view,w.down.view,w.shared_gate.view,w.shared_up.view,w.shared_down.view,w.shared_router.view};
        std::array<GpuView,14> writes{output};const auto all=s.views();std::copy(all.begin(),all.end(),writes.begin()+1);g.validate_disjoint(reads,writes);
        // Rows are independent: every projection, then every top-k, so a concurrent batch needs one barrier here instead of one per row.
        g.gemv_rows(w.router.encoding,w.router.view,rows(active(g,input,shape.channels,steps),steps,shape.channels),rows(active(g,s.rows.logits,shape.experts,steps),steps,shape.experts));
        for(uint32_t t=0;t<steps;++t){const auto r=token(g,s.rows,shape,t);g.route(r.logits,r.indices,r.weights,shape.selected,mode,normalize);}
        const auto ids=active(g,s.rows.indices,shape.selected,steps);
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
        g.route_lock_rows(active(g,s.rows.logits,shape.experts,steps),ids,active(g,s.rows.weights,shape.selected,steps),steps,shape.channels,shape.experts,shape.selected);
#endif
        const auto project=[&](GpuMatrix m,GpuView x,GpuView y,uint32_t c,uint32_t r,bool broadcast){
            const GroupedMppShape ms{c,r,shape.experts,shape.selected,steps,projection_tile,projection_mode};const auto b=grouped_mpp_bytes(m.encoding,ms,broadcast);
            g.routed_matmul_many(m.encoding,m.view,ids,active(g,x,uint64_t(c)*(broadcast?1:shape.selected),steps),active(g,y,uint64_t(r)*shape.selected,steps),
                g.slice(s.token_map,0,b.token_map),s.counts,g.slice(s.tasks,0,b.tasks),ms,broadcast);};
        project(w.gate,input,s.rows.gate,shape.channels,shape.hidden,true);project(w.up,input,s.rows.up,shape.channels,shape.hidden,true);
        const auto gate=active(g,s.rows.gate,uint64_t(shape.selected)*shape.hidden,steps),up=active(g,s.rows.up,uint64_t(shape.selected)*shape.hidden,steps);
        g.swiglu(gate,up,gate);project(w.down,s.rows.gate,s.rows.down,shape.hidden,shape.channels,false);
        shared(g,w.shared_gate,input,s.rows.shared_gate,shape.channels,shape.hidden,steps);shared(g,w.shared_up,input,s.rows.shared_up,shape.channels,shape.hidden,steps);
        const auto sg=active(g,s.rows.shared_gate,shape.hidden,steps),su=active(g,s.rows.shared_up,shape.hidden,steps);g.swiglu(sg,su,sg);
        shared(g,w.shared_down,s.rows.shared_gate,s.rows.shared_down,shape.hidden,shape.channels,steps);
        g.gemv_rows(w.shared_router.encoding,w.shared_router.view,rows(active(g,input,shape.channels,steps),steps,shape.channels),rows(active(g,s.rows.shared_logit,1,steps),steps,1));
        for(uint32_t t=0;t<steps;++t){const auto r=token(g,s.rows,shape,t);g.mixture(r.down,r.weights,r.shared_down,r.shared_logit,row(g,output,shape.channels,t));}
    }catch(...){try{g.abort();}catch(const FormatError&){}throw;}
}
}
