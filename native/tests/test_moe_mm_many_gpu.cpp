#include "grouped_mm_fixture.hpp"
#include "tokspur/moe_mm_many.hpp"
#include "tokspur/metal.hpp"
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
#include "tokspur/verify_nax.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>

using namespace tokspur;
namespace {
// Prospective generated-composition screens, frozen before GPU execution. These are empirical domain checks, not universal half-error guarantees or D-Cal admission.
constexpr double scaled_limit=0.001,relative_l2_limit=0.005;
unsigned checks=0,cases=0;
void need(bool ok,const char* why){++checks;if(!ok)throw FormatError(why);}
void refuses(const std::function<void()>& call){bool caught=false;try{call();}catch(const FormatError&){caught=true;}need(caught,"invalid generated MoE composition admitted");}
struct Arena{
    MetalGraph& g;std::vector<GpuView> views;
    GpuView bytes(std::span<const std::byte> data,bool immutable=true){auto v=g.upload(data,"generated MoE weights/rows",immutable);views.push_back(v);return v;}
    GpuView floats(std::span<const float> data,bool immutable=true){return bytes(std::as_bytes(data),immutable);}
    ~Arena(){try{g.abort();}catch(const FormatError&){}for(auto v:views)g.release(v);}
};
struct Matrix { uint32_t type,k,r,e;std::vector<std::byte> raw;std::vector<float> decoded;GpuMatrix gpu; };
Matrix generated_matrix(uint32_t type,uint32_t k,uint32_t r,uint32_t e,uint32_t seed){
    Matrix m{type,k,r,e,{},{},{}};const auto& q=encoding(type);m.raw.resize(uint64_t(k/q.elements_per_block)*q.bytes_per_block*r*e);
    uint32_t rng=seed;
    for(size_t at=0;at<m.raw.size();at+=q.bytes_per_block){
        for(uint32_t i=0;i<q.bytes_per_block;++i){rng=rng*1664525u+1013904223u;m.raw[at+i]=std::byte(rng>>24);}
        m.raw[at]=std::byte{1};m.raw[at+1]=std::byte{0x0c}; // Non-power-of-two normal half scale.
        if(type==8){for(unsigned i=0;i<32;++i){rng=rng*1664525u+1013904223u;m.raw[at+2+i]=std::byte(uint8_t(int(rng%31)-15));}}
        else{
            m.raw[at+2]=std::byte{1};m.raw[at+3]=std::byte{4};
            if(type==7){for(unsigned i=0;i<4;++i)m.raw[at+4+i]=std::byte{};}
            else{for(unsigned i=0;i<8;++i)m.raw[at+4+i]=std::byte{1};for(unsigned i=8;i<12;++i)m.raw[at+4+i]=std::byte{0x11};if(type==13)for(unsigned i=0;i<32;++i)m.raw[at+16+i]=std::byte{};}
            if(type==12)for(unsigned i=0;i<128;++i)m.raw[at+16+i]&=std::byte{0x77};
        }
    }
    m.decoded.resize(uint64_t(k)*r*e);dequantize(type,m.raw,m.decoded);
    for(auto x:m.decoded)need(std::isfinite(x) && std::abs(x)<=1.0f/256,"declared bounded nonzero quant domain");
    return m;
}
Matrix make_matrix(Arena& a,uint32_t type,uint32_t k,uint32_t r,uint32_t e,uint32_t seed){auto m=generated_matrix(type,k,r,e,seed);m.gpu={a.bytes(m.raw),type};return m;}
std::vector<float> read(MetalGraph& g,GpuView view){auto bytes=g.read(view);std::vector<float> out(bytes.size()/4);std::memcpy(out.data(),bytes.data(),bytes.size());return out;}
MoeScratch token_scratch(MetalGraph& g,const MoeScratch& s,MoeShape shape,uint32_t t){
    const auto b=moe_mm_many_bytes(shape,1);auto views=s.views();for(size_t i=0;i<views.size();++i)views[i]=g.slice(views[i],uint64_t(t)*b.views[i],b.views[i]);
    return {views[0],views[1],views[2],views[3],views[4],views[5],views[6],views[7],views[8],views[9]};
}
std::vector<float> dot(const Matrix& m,uint32_t expert,std::span<const float> input,bool half){
    std::vector<float> output(m.r);for(uint32_t r=0;r<m.r;++r){double sum=0;for(uint32_t c=0;c<m.k;++c){const auto w=m.decoded[(uint64_t(expert)*m.r+r)*m.k+c];sum+=double(half?grouped_fixture::staged(w):w)*(half?grouped_fixture::staged(input[c]):input[c]);}output[r]=float(sum);}return output;
}
std::vector<float> gate(std::vector<float> g,std::span<const float> up){for(size_t i=0;i<g.size();++i)g[i]=float(g[i]/(1+std::exp(-g[i])))*up[i];return g;}
std::vector<double> reference(MoeShape s,std::span<const float> input,const std::array<Matrix*,6>& m,const Routes& routes,bool half){
    std::vector<double> experts(uint64_t(s.selected)*s.channels),weights(routes.weights.begin(),routes.weights.end());
    for(uint32_t slot=0;slot<s.selected;++slot){auto hidden=gate(dot(*m[0],routes.indices[slot],input,half),dot(*m[1],routes.indices[slot],input,half));auto down=dot(*m[2],routes.indices[slot],hidden,half);std::copy(down.begin(),down.end(),experts.begin()+uint64_t(slot)*s.channels);}
    auto hidden=gate(dot(*m[3],0,input,half),dot(*m[4],0,input,half));auto shared=dot(*m[5],0,hidden,half);std::vector<double> shared_double(shared.begin(),shared.end());
    return mixture_reference(experts,weights,shared_double,0.25);
}
struct Error{double maximum=0,l2=0;};
Error error(std::span<const float> actual,std::span<const double> expected){
    need(actual.size()==expected.size(),"generated MoE reference extent");double square=0,norm=0;Error out;
    for(size_t i=0;i<actual.size();++i){need(std::isfinite(actual[i]) && std::isfinite(expected[i]),"finite full MoE composition");const double d=double(actual[i])-expected[i];square+=d*d;norm+=expected[i]*expected[i];out.maximum=std::max(out.maximum,std::abs(d)/std::max(1.0,std::abs(expected[i])));}
    need(norm>0,"nonzero generated MoE signal");out.l2=std::sqrt(square/norm);return out;
}
void exercise(const char* library,MoeShape shape,uint32_t n,uint32_t gate_type,uint32_t down_type,RouterMode mode,bool router_batch=false,bool concurrent=false){
    MetalGraph g(library,512ull<<20);Arena a{g,{}};g.configure_small_gemv(16);g.configure_block_gemv(3);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    g.configure_concurrent_dispatch(concurrent);
#else
    need(!concurrent,"router concurrent controls require the concurrent build");
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    need(!g.moe_router_f32_batch_enabled(),"MoE router batch default remains off");
    g.configure_moe_router_f32_batch(router_batch);
#else
    need(!router_batch,"router batch requires the private capability build");
#endif
    std::cout<<"MoE case C="<<shape.channels<<" H="<<shape.hidden<<" E="<<shape.experts<<" S="<<shape.selected<<" N="<<n<<" gate="<<gate_type<<" up="<<(gate_type==12?13:12)<<" down="<<down_type<<" thermal_before="<<host_thermal_state()<<std::endl;
    const auto candidate=allocate_moe_mm_many_scratch(g,shape,512),control=allocate_moe_mm_many_scratch(g,shape,512);
    try{
        auto gate_w=make_matrix(a,gate_type,shape.channels,shape.hidden,shape.experts,44),up_w=make_matrix(a,gate_type==12?13:12,shape.channels,shape.hidden,shape.experts,45),down_w=make_matrix(a,down_type,shape.hidden,shape.channels,shape.experts,46);
        auto shared_gate=make_matrix(a,8,shape.channels,shape.hidden,1,47),shared_up=make_matrix(a,8,shape.channels,shape.hidden,1,48),shared_down=make_matrix(a,8,shape.hidden,shape.channels,1,49);
        std::vector<float> router(uint64_t(shape.channels)*shape.experts),shared_router(shape.channels);
        std::vector<double> logits(shape.experts);for(uint32_t e=0;e<shape.experts;++e){router[uint64_t(e)*shape.channels]=float(int(e)-8)/32;router[uint64_t(e)*shape.channels+1]=float(int((e*7)%17)-8)/16;}shared_router[0]=0.25f;
        const auto rw=a.floats(router),srw=a.floats(shared_router);const MoeWeights weights{{rw,0},gate_w.gpu,up_w.gpu,down_w.gpu,shared_gate.gpu,shared_up.gpu,shared_down.gpu,{srw,0}};
        prepare_moe_mm_many(g,weights,shape);
        const uint64_t stride=uint64_t(shape.channels)*4+32;std::vector<float> input(512*stride/4,-9876.25f),poison(input.size()+16,-7654.25f);uint32_t rng=44;
        constexpr std::array<float,4> route_input{0.5f,-0.5f,0.25f,-0.25f};
        std::vector<Routes> expected_routes;
        for(uint32_t t=0;t<n;++t){for(uint32_t c=0;c<shape.channels;++c){rng=rng*1664525u+1013904223u;input[t*stride/4+c]=c==0?1:c==1?route_input[t%4]:float(int(rng%1993)-996)/997;}
            for(uint32_t e=0;e<shape.experts;++e)logits[e]=double(router[uint64_t(e)*shape.channels])+double(input[t*stride/4+1])*router[uint64_t(e)*shape.channels+1];expected_routes.push_back(route_reference(logits,shape.selected,mode,true));}
        const auto x=a.floats(input,false),y=a.floats(poison,false),baseline=a.floats(poison,false);const GpuRowsF32 in{x,n,shape.channels,stride},out{y,n,shape.channels,stride};
        const auto ledger=g.memory();g.begin();uint64_t ordinary_dispatches=0;
        for(uint32_t t=0;t<n;++t){encode_moe(g,weights,token_scratch(g,control.rows,shape,t),g.slice(x,uint64_t(t)*stride,uint64_t(shape.channels)*4),g.slice(baseline,uint64_t(t)*stride,uint64_t(shape.channels)*4),shape,mode,true);
            ordinary_dispatches+=9+g.gemv_dispatches(0,shape.channels,shape.experts)+g.gemv_dispatches(8,shape.channels,shape.hidden)*2+g.gemv_dispatches(8,shape.hidden,shape.channels)+g.gemv_dispatches(0,shape.channels,1);}
        auto b=g.submit();need(b.dispatches==ordinary_dispatches,"ordinary row-by-row dispatch count");
        g.begin();encode_moe_mm_many(g,weights,candidate,in,out,shape,mode,true);b=g.submit();
        need(b.dispatches==moe_mm_many_dispatches(g,weights,shape,n) && g.memory().allocations==ledger.allocations && g.memory().live_bytes==ledger.live_bytes,"MoE deterministic dispatch/no-encode-allocation invariant");
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        // Rows are independent, so barriers follow the operator stages. One per row would serialise every router and mixture dispatch.
        if(concurrent){std::cout<<"MoE concurrent barriers="<<g.concurrent_barriers()<<" rows="<<n<<std::endl;need(g.concurrent_barriers()<std::max(n,64u),"MoE rows serialised by per-row barriers");}
#endif
        const auto got=read(g,y),owned=read(g,baseline);std::vector<float> active;std::vector<double> expected,half_reference;const std::array<Matrix*,6> matrices{&gate_w,&up_w,&down_w,&shared_gate,&shared_up,&shared_down};
        const auto route_bytes=uint64_t(n)*shape.selected*4;
        need(g.read(g.slice(candidate.rows.indices,0,route_bytes))==g.read(g.slice(control.rows.indices,0,route_bytes)) && g.read(g.slice(candidate.rows.weights,0,route_bytes))==g.read(g.slice(control.rows.weights,0,route_bytes)),"unchanged router IDs and weights");
        const auto indices=g.read(g.slice(candidate.rows.indices,0,route_bytes));
        for(uint32_t t=0;t<n;++t)for(uint32_t slot=0;slot<shape.selected;++slot){uint32_t id=0;std::memcpy(&id,indices.data()+(uint64_t(t)*shape.selected+slot)*4,4);need(id==expected_routes[t].indices[slot],"independent router selection");}
        for(uint32_t t=0;t<n;++t){
            const auto source=std::span(input).subspan(t*stride/4,shape.channels);auto ref=reference(shape,source,matrices,expected_routes[t],true);half_reference.insert(half_reference.end(),ref.begin(),ref.end());
            for(uint32_t c=0;c<shape.channels;++c){active.push_back(got[t*stride/4+c]);expected.push_back(owned[t*stride/4+c]);}
            for(uint32_t c=shape.channels;c<stride/4;++c)need(got[t*stride/4+c]==-7654.25f && owned[t*stride/4+c]==-7654.25f,"MoE row padding changed");
        }
        for(size_t i=n*stride/4;i<got.size();++i)need(got[i]==-7654.25f && owned[i]==-7654.25f,"inactive capacity/end canary changed");
        const auto raw_error=error(active,expected),stage_error=error(active,half_reference);
        std::cout<<"MoE errors raw_scaled="<<raw_error.maximum<<" raw_relative_l2="<<raw_error.l2<<" half_reference_scaled="<<stage_error.maximum<<" half_reference_relative_l2="<<stage_error.l2<<std::endl;
        need(raw_error.maximum<=scaled_limit && raw_error.l2<=relative_l2_limit,"prospective owned-row MoE composition screen failed");
        need(stage_error.maximum<=scaled_limit && stage_error.l2<=relative_l2_limit,"prospective CPU half-staged MoE composition screen failed");
        for(auto m:matrices)need(g.read(m->gpu.view)==m->raw,"MoE changed quantised weights");
        need(g.read(rw)==std::vector<std::byte>(std::as_bytes(std::span(router)).begin(),std::as_bytes(std::span(router)).end()) && g.read(srw)==std::vector<std::byte>(std::as_bytes(std::span(shared_router)).begin(),std::as_bytes(std::span(shared_router)).end()) && g.read(x)==std::vector<std::byte>(std::as_bytes(std::span(input)).begin(),std::as_bytes(std::span(input)).end()),"MoE changed router/input bytes");
        const auto saved=g.read(y);const auto fail=[&](const std::function<void()>& call,bool compatible=false){refuses([&]{if(compatible)g.begin_indirect_compatible();else g.begin();call();});try{g.abort();}catch(const FormatError&){}need(g.read(y)==saved,"MoE refusal wrote output");};
        auto wrong_layout=MoeMmLayouts{};wrong_layout.down=PrefillWeightLayout::q8_lanes_v1;
        refuses([&]{prepare_moe_mm_many(g,weights,shape,wrong_layout);});fail([&]{encode_moe_mm_many(g,weights,candidate,in,out,shape,mode,true,wrong_layout);});
        auto aliased=out;aliased.view=x;fail([&]{encode_moe_mm_many(g,weights,candidate,in,aliased,shape,mode,true);});
        auto wrong_rows=out;wrong_rows.rows=n+1;fail([&]{encode_moe_mm_many(g,weights,candidate,in,wrong_rows,shape,mode,true);});
        auto alias_metadata=candidate;alias_metadata.counts=g.slice(candidate.token_map,0,candidate.counts.bytes);fail([&]{encode_moe_mm_many(g,weights,alias_metadata,in,out,shape,mode,true);});
        auto immutable_metadata=candidate;immutable_metadata.counts=a.bytes(std::vector<std::byte>(candidate.counts.bytes));fail([&]{encode_moe_mm_many(g,weights,immutable_metadata,in,out,shape,mode,true);});
        fail([&]{encode_moe_mm_many(g,weights,candidate,in,out,shape,mode,true);},true);
        const auto immutable=a.bytes(saved);auto immutable_out=out;immutable_out.view=immutable;fail([&]{encode_moe_mm_many(g,weights,candidate,in,immutable_out,shape,mode,true);});
        g.begin();encode_moe_mm_many(g,weights,candidate,in,out,shape,mode,true);static_cast<void>(g.submit());need(g.read(y)==saved,"valid deterministic reuse after MoE refusal");
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
        if(router_batch && shape.channels==2560 && shape.experts==512) {
            const auto single_ledger=g.memory();g.begin();encode_moe_mm_many(g,weights,candidate,{x,1,shape.channels,stride},{y,1,shape.channels,stride},shape,mode,true);const auto single=g.submit();
            need(single.dispatches==moe_mm_many_dispatches(g,weights,shape,1) && g.read(y)==saved && g.memory().allocations==single_ledger.allocations && g.memory().live_bytes==single_ledger.live_bytes,"router selected single-row scalar fallback/reuse dispatch invariant");
        }
#endif
        std::cout<<"MoE case_pass thermal_after="<<host_thermal_state()<<" component_only=true\n";++cases;
    }catch(...){try{g.abort();}catch(const FormatError&){}release_moe_mm_many_scratch(g,candidate);release_moe_mm_many_scratch(g,control);throw;}
    release_moe_mm_many_scratch(g,candidate);release_moe_mm_many_scratch(g,control);
}
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
// Frozen old scheduling reference: route/tasks are regenerated separately by each existing grouped_mm call. No shared-route API is used here.
void three_routes_reference(MetalGraph& g,const MoeWeights& w,const MoeMmManyScratch& s,GpuRowsF32 input,GpuRowsF32 output,MoeShape shape){
    const auto n=input.rows;
    const auto one=[&](GpuView v,uint64_t stride,uint64_t bytes,uint32_t at){return g.slice(v,uint64_t(at)*stride,bytes);};
    const auto active=[&](GpuView v,uint64_t width){return g.slice(v,0,width*n*4);};
    const auto rows=[](GpuView v,uint32_t n,uint32_t c){return GpuRowsF32{v,n,c,uint64_t(c)*4};};
    for(uint32_t t=0;t<n;++t){const auto r=token_scratch(g,s.rows,shape,t);g.gemv(0,w.router.view,one(input.view,input.stride_bytes,uint64_t(shape.channels)*4,t),r.logits,shape.channels,shape.experts);g.route(r.logits,r.indices,r.weights,shape.selected,RouterMode::softmax,true);}
    const auto ids=active(s.rows.indices,shape.selected);
    const auto project=[&](GpuMatrix m,GpuRowsF32 x,GpuRowsF32 y,bool broadcast){g.grouped_mm(m.encoding,PrefillWeightLayout::gguf,m.view,ids,x,y,s.token_map,s.counts,s.tasks,shape.experts,shape.selected,broadcast);};
    const auto gate=rows(s.rows.gate,n*shape.selected,shape.hidden),up=rows(s.rows.up,n*shape.selected,shape.hidden);
    project(w.gate,input,gate,true);project(w.up,input,up,true);
    const auto gv=active(s.rows.gate,uint64_t(shape.hidden)*shape.selected),uv=active(s.rows.up,uint64_t(shape.hidden)*shape.selected);
    g.activate(gv,gv,Activation::silu);g.multiply(gv,uv,gv);
    project(w.down,gate,rows(s.rows.down,n*shape.selected,shape.channels),false);
    const auto sg=rows(s.rows.shared_gate,n,shape.hidden),su=rows(s.rows.shared_up,n,shape.hidden);
    g.dense_mm_q8(PrefillWeightLayout::gguf,w.shared_gate.view,input,sg);g.dense_mm_q8(PrefillWeightLayout::gguf,w.shared_up.view,input,su);
    const auto sgv=active(sg.view,shape.hidden),suv=active(su.view,shape.hidden);g.activate(sgv,sgv,Activation::silu);g.multiply(sgv,suv,sgv);
    g.dense_mm_q8(PrefillWeightLayout::gguf,w.shared_down.view,sg,rows(s.rows.shared_down,n,shape.channels));
    for(uint32_t t=0;t<n;++t){const auto r=token_scratch(g,s.rows,shape,t);g.gemv(0,w.shared_router.view,one(input.view,input.stride_bytes,uint64_t(shape.channels)*4,t),r.shared_logit,shape.channels,1);g.mixture(r.down,r.weights,r.shared_down,r.shared_logit,one(output.view,output.stride_bytes,uint64_t(shape.channels)*4,t));}
}
void f32_route_scope_control(const char* library,uint32_t n){
    const MoeShape shape{2560,640,16,10};MetalGraph g(library,512ull<<20);Arena a{g,{}};
    g.configure_small_gemv(16);g.configure_block_gemv(3);g.configure_prefill_dense_f32(true);g.configure_prefill_grouped_f32(true);
    const auto old=allocate_moe_mm_many_scratch(g,shape,512),current=allocate_moe_mm_many_scratch(g,shape,512);
    try{
        auto gate=make_matrix(a,12,shape.channels,shape.hidden,shape.experts,44),up=make_matrix(a,13,shape.channels,shape.hidden,shape.experts,45),down=make_matrix(a,7,shape.hidden,shape.channels,shape.experts,46);
        auto sg=make_matrix(a,8,shape.channels,shape.hidden,1,47),su=make_matrix(a,8,shape.channels,shape.hidden,1,48),sd=make_matrix(a,8,shape.hidden,shape.channels,1,49);
        std::vector<float> router(uint64_t(shape.channels)*shape.experts),shared(shape.channels),input(uint64_t(512)*shape.channels),poison(input.size()+16,-7654.25f);
        uint32_t rng=44;for(auto& v:input){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997;}
        for(uint32_t e=0;e<shape.experts;++e)router[uint64_t(e)*shape.channels]=float(int(e)-8)/32;shared[0]=.25f;
        const MoeWeights w{{a.floats(router),0},gate.gpu,up.gpu,down.gpu,sg.gpu,su.gpu,sd.gpu,{a.floats(shared),0}};prepare_moe_mm_many(g,w,shape);
        const auto x=a.floats(input),baseline=a.floats(poison,false),candidate=a.floats(poison,false);
        const GpuRowsF32 in{x,n,shape.channels,uint64_t(shape.channels)*4},old_out{baseline,n,shape.channels,uint64_t(shape.channels)*4},new_out{candidate,n,shape.channels,uint64_t(shape.channels)*4};
        const auto ledger=g.memory();g.begin();three_routes_reference(g,w,old,in,old_out,shape);const auto before=g.submit();
        g.begin();encode_moe_mm_many(g,w,current,in,new_out,shape,RouterMode::softmax,true);const auto after=g.submit();
        need(before.dispatches==after.dispatches+4 && after.dispatches==moe_mm_many_dispatches(g,w,shape,n),"F32 MoE scope did not remove exactly four route/task dispatches");
        need(g.read(baseline)==g.read(candidate),"F32 MoE scope changed output/capacity/canary bytes");
        const auto routebytes=uint64_t(n)*shape.selected*4;
        need(g.read(g.slice(old.rows.indices,0,routebytes))==g.read(g.slice(current.rows.indices,0,routebytes)) && g.read(g.slice(old.rows.weights,0,routebytes))==g.read(g.slice(current.rows.weights,0,routebytes)),"F32 MoE scope changed route IDs/weights");
        need(g.read(old.token_map)==g.read(current.token_map) && g.read(old.counts)==g.read(current.counts) && g.read(old.tasks)==g.read(current.tasks),"F32 MoE scope changed stable routing metadata");
        need(g.memory().allocations==ledger.allocations && g.memory().live_bytes==ledger.live_bytes,"F32 MoE scope allocated scratch");
        for(const auto* matrix:{&gate,&up,&down,&sg,&su,&sd})need(g.read(matrix->gpu.view)==matrix->raw,"F32 MoE scope changed weight bytes");
        std::cout<<"F32 route scope N="<<n<<" old_dispatches="<<before.dispatches<<" shared_dispatches="<<after.dispatches<<" old_gpu_ms="<<before.gpu_ms<<" shared_gpu_ms="<<after.gpu_ms<<" old_batch_wall_ms="<<before.wall_ms<<" shared_batch_wall_ms="<<after.wall_ms<<" sample_only=true byte_exact=true thermal_after="<<host_thermal_state()<<" component_only=true\n";
    }catch(...){try{g.abort();}catch(const FormatError&){}release_moe_mm_many_scratch(g,old);release_moe_mm_many_scratch(g,current);throw;}
    release_moe_mm_many_scratch(g,old);release_moe_mm_many_scratch(g,current);
}
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
void router_projection_controls(const char* library,bool concurrent=false) {
    MetalGraph g(library,128ull<<20);Arena a{g,{}};g.configure_small_gemv(16);g.configure_block_gemv(3);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    g.configure_concurrent_dispatch(concurrent);
#else
    need(!concurrent,"router concurrent controls require the concurrent build");
#endif
    need(!g.moe_router_f32_batch_enabled(),"router batching is default off");
    need(std::string(g.moe_router_f32_projection_kernel())=="graph_quantized_gemv","router scalar kernel identity");
    g.configure_moe_router_f32_batch(true);
    need(std::string(g.moe_router_f32_projection_kernel())=="graph_moe_router_f32_batch","router selected kernel identity");
    constexpr uint32_t c=2560,e=512,capacity=512;
    constexpr uint64_t xs=c*4+32,ys=e*4+32;
    std::vector<float> weights(uint64_t(c)*e),input(capacity*xs/4,-9876.25f),poison(capacity*ys/4+16,-7654.25f);
    // Dense non-binary values expose operand rounding and reduction changes.
    uint32_t rng=44;
    for(uint32_t expert=0;expert<e;++expert)for(uint32_t k=0;k<c;++k) {
        rng=rng*1664525u+1013904223u;
        weights[uint64_t(expert)*c+k]=float(int(rng%1993)-996)/997/256;
    }
    for(uint32_t t=0;t<capacity;++t)for(uint32_t k=0;k<c;++k) {
        rng=rng*1664525u+1013904223u;input[t*xs/4+k]=float(int(rng%1993)-996)/997;
    }
    const auto w=a.floats(weights),x=a.floats(input),y=a.floats(poison,false),scalar=a.floats(poison,false);
    const auto saved_w=g.read(w),saved_x=g.read(x);
    refuses([&]{g.begin();try{g.moe_router_f32_batch(w,{x,3,c,xs},{y,3,e,ys});}catch(...){g.abort();throw;}});
    need(g.read(y)==std::vector<std::byte>(std::as_bytes(std::span(poison)).begin(),std::as_bytes(std::span(poison)).end()),"router requires idle preparation before any write");
    g.prepare_moe_router_f32_batch();refuses([&]{g.configure_moe_router_f32_batch(false);});
    // Zero logits check deterministic low-index route ties separately.
    const auto zero=a.floats(std::vector<float>(uint64_t(c)*e));
    const auto ids=g.zeros(10*4,"router generated IDs"),scores=g.zeros(10*4,"router generated weights");a.views.push_back(ids);a.views.push_back(scores);
    const auto all_ids=a.floats(std::vector<float>(capacity*10),false),old_ids=a.floats(std::vector<float>(capacity*10),false),all_weights=a.floats(std::vector<float>(capacity*10),false),old_weights=a.floats(std::vector<float>(capacity*10),false);
    for(auto n:{1u,2u,3u,8u,31u,32u,33u,511u,512u}) {
        const GpuRowsF32 in{x,n,c,xs},out{y,n,e,ys};
        const auto ledger=g.memory();g.begin();for(uint32_t t=0;t<n;++t)g.gemv(0,w,g.slice(x,uint64_t(t)*xs,c*4),g.slice(scalar,uint64_t(t)*ys,e*4),c,e);const auto ordinary=g.submit();
        need(ordinary.dispatches==uint64_t(n)*g.gemv_dispatches(0,c,e),"router scalar control actual dispatch count");
        g.begin();g.moe_router_f32_batch(w,in,out);const auto batch=g.submit();
        need(batch.dispatches==1 && g.memory().allocations==ledger.allocations && g.memory().live_bytes==ledger.live_bytes,"router physical dispatch/no-encode-allocation invariant");
        const auto got=read(g,y),legacy=read(g,scalar);std::vector<double> references(uint64_t(n)*e);
        for(uint32_t t=0;t<n;++t) {
            auto logits=std::span(references).subspan(uint64_t(t)*e,e);
            for(uint32_t expert=0;expert<e;++expert) {
                double sum=0;for(uint32_t k=0;k<c;++k)sum+=double(weights[uint64_t(expert)*c+k])*input[t*xs/4+k];
                logits[expert]=sum;need(std::isfinite(got[t*ys/4+expert]) && std::abs(double(got[t*ys/4+expert])-sum)<=0.00005*std::max(1.0,std::abs(sum)),"router full FP32 logit bound");
                need(std::isfinite(legacy[t*ys/4+expert]) && std::abs(double(legacy[t*ys/4+expert])-sum)<=0.00005*std::max(1.0,std::abs(sum)) && std::abs(double(got[t*ys/4+expert])-legacy[t*ys/4+expert])<=0.00005*std::max(1.0,std::abs(double(legacy[t*ys/4+expert]))),"router candidate/scalar full-row logit guard");
            }
            for(uint32_t k=e;k<ys/4;++k)need(got[t*ys/4+k]==-7654.25f,"router output row padding");
        }
        for(auto mode:{RouterMode::softmax,RouterMode::sigmoid})for(bool normalise:{false,true}) {
            g.begin();for(uint32_t t=0;t<n;++t){g.route(g.slice(y,uint64_t(t)*ys,e*4),g.slice(all_ids,uint64_t(t)*40,40),g.slice(all_weights,uint64_t(t)*40,40),10,mode,normalise);g.route(g.slice(scalar,uint64_t(t)*ys,e*4),g.slice(old_ids,uint64_t(t)*40,40),g.slice(old_weights,uint64_t(t)*40,40),10,mode,normalise);}static_cast<void>(g.submit());
            const auto selected=g.read(g.slice(all_ids,0,uint64_t(n)*40)),old_selected=g.read(g.slice(old_ids,0,uint64_t(n)*40));need(selected==old_selected,"router all-row selected IDs differ from actual scalar route");
            const auto selected_weights=read(g,all_weights),legacy_weights=read(g,old_weights);
            for(uint32_t t=0;t<n;++t){const auto expected=route_reference(std::span(references).subspan(uint64_t(t)*e,e),10,mode,normalise);for(uint32_t j=0;j<10;++j){uint32_t id;std::memcpy(&id,selected.data()+(uint64_t(t)*10+j)*4,4);need(id==expected.indices[j],"router all-row IDs differ from independent double route");const auto actual=selected_weights[uint64_t(t)*10+j],control=legacy_weights[uint64_t(t)*10+j];need(std::isfinite(actual)&&std::isfinite(control)&&std::abs(actual-control)<=0.00005&&std::abs(actual-expected.weights[j])<=0.00005,"router all-row selected weight bound");}}
        }
        // Calls use increasing n, so all still-inactive rows and end guards remain poisoned.
        for(size_t k=n*ys/4;k<got.size();++k)need(got[k]==-7654.25f,"router inactive capacity/end guard");
        need(g.read(w)==saved_w && g.read(x)==saved_x,"router inputs/weights remain immutable");
    }
    g.begin();g.moe_router_f32_batch(zero,{x,3,c,xs},{y,3,e,ys});g.route(g.slice(y,0,e*4),ids,scores,10,RouterMode::softmax,true);static_cast<void>(g.submit());
    const auto tied=g.read(ids);for(uint32_t j=0;j<10;++j){uint32_t id;std::memcpy(&id,tied.data()+j*4,4);need(id==j,"router exact tie selects low IDs");}
    const auto saved_y=g.read(y);
    const auto fail=[&](const std::function<void()>& f,bool indirect=false){refuses([&]{if(indirect)g.begin_indirect_compatible();else g.begin();try{f();}catch(...){g.abort();throw;}});try{g.abort();}catch(const FormatError&){}need(g.read(y)==saved_y,"router refusal does not write output");};
    const GpuRowsF32 in{x,3,c,xs},out{y,3,e,ys};
    auto bad=out;bad.rows=2;fail([&]{g.moe_router_f32_batch(w,in,bad);});
    bad=out;bad.stride_bytes=e*4-4;fail([&]{g.moe_router_f32_batch(w,in,bad);});
    bad=out;bad.view=x;fail([&]{g.moe_router_f32_batch(w,in,bad);});
    bad=out;bad.view=a.floats(poison);fail([&]{g.moe_router_f32_batch(w,in,bad);});
    const auto mutable_x=a.floats(input,false);auto overlap=out;overlap.view=mutable_x;
    fail([&]{g.moe_router_f32_batch(w,{mutable_x,3,c,xs},overlap);});
    bad=out;bad.view=g.slice(y,0,moe_router_f32_plan(c,e,3,xs,ys).output_bytes-4);fail([&]{g.moe_router_f32_batch(w,in,bad);});
    bad=out;++bad.view.offset;--bad.view.bytes;fail([&]{g.moe_router_f32_batch(w,in,bad);});
    fail([&]{g.moe_router_f32_batch(g.slice(w,0,w.bytes-4),in,out);});
    fail([&]{g.moe_router_f32_batch(w,in,out);},true);
    auto nan=input;nan[0]=std::numeric_limits<float>::quiet_NaN();const auto nx=a.floats(nan);
    refuses([&]{g.begin();g.moe_router_f32_batch(w,{nx,3,c,xs},out);static_cast<void>(g.submit());});
    try{g.abort();}catch(const FormatError&){}
    auto inf=weights;inf[0]=std::numeric_limits<float>::infinity();const auto iw=a.floats(inf);
    refuses([&]{g.begin();g.moe_router_f32_batch(iw,in,out);static_cast<void>(g.submit());});
    try{g.abort();}catch(const FormatError&){}
    g.begin();g.moe_router_f32_batch(w,in,out);static_cast<void>(g.submit());
    std::cout<<"PASS generated raw FP32 router projection controls, numerical/routing scope only, no model or speed qualification\n";
}
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
void router_window_controls(const char* library,bool concurrent=false) {
    for(uint32_t splits:{16u,0u}) {
        MetalGraph g(library,256ull<<20);Arena a{g,{}};g.configure_small_gemv(splits);g.configure_block_gemv(3);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        g.configure_concurrent_dispatch(concurrent);
#else
        need(!concurrent,"router window concurrent controls require the concurrent build");
#endif
        need(!g.prefill_router_window(),"router window is default off");
        need(std::string(g.moe_router_window_projection_kernel())=="graph_quantized_gemv","router window scalar kernel identity");
        g.configure_prefill_router_window(true);
        need(g.prefill_router_window() && std::string(g.moe_router_window_projection_kernel())=="graph_moe_router_f32_window","router window selected kernel identity");
        constexpr uint32_t c=2560,e=512,selected_k=10,capacity=512;
        constexpr uint64_t xs=c*4+32,ys=e*4+32;
        std::vector<float> weights(uint64_t(c)*e),shared_w(c),input(capacity*xs/4,-9876.25f),poison_y(capacity*ys/4+16,-7654.25f),poison_out(capacity*xs/4+16,-7654.25f);
        std::vector<float> down_rows(uint64_t(capacity)*selected_k*c),shared_down_rows(uint64_t(capacity)*c);
        uint32_t rng=44;
        for(uint32_t expert=0;expert<e;++expert)for(uint32_t k=0;k<c;++k) {
            rng=rng*1664525u+1013904223u;
            weights[uint64_t(expert)*c+k]=float(int(rng%1993)-996)/997/256;
        }
        for(uint32_t k=0;k<c;++k){rng=rng*1664525u+1013904223u;shared_w[k]=float(int(rng%1993)-996)/997/64;}
        for(uint32_t t=0;t<capacity;++t)for(uint32_t k=0;k<c;++k) {
            rng=rng*1664525u+1013904223u;input[t*xs/4+k]=float(int(rng%1993)-996)/997;
        }
        for(auto& v:down_rows){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997/16;}
        for(auto& v:shared_down_rows){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997/16;}
        const auto w=a.floats(weights),sw=a.floats(shared_w),x=a.floats(input),y=a.floats(poison_y,false),scalar=a.floats(poison_y,false);
        const auto down_gpu=a.floats(down_rows),shared_down_gpu=a.floats(shared_down_rows);
        const auto saved_w=g.read(w),saved_sw=g.read(sw),saved_x=g.read(x);
        refuses([&]{g.begin();try{g.moe_router_f32_window(w,{x,3,c,xs},{y,3,e,ys});}catch(...){g.abort();throw;}});
        need(g.read(y)==std::vector<std::byte>(std::as_bytes(std::span(poison_y)).begin(),std::as_bytes(std::span(poison_y)).end()),"router window requires idle preparation before any write");
        g.prepare_moe_router_window();refuses([&]{g.configure_prefill_router_window(false);});
        const auto all_ids=a.floats(std::vector<float>(capacity*selected_k,-1234.5f),false),old_ids=a.floats(std::vector<float>(capacity*selected_k,-1234.5f),false);
        const auto all_weights=a.floats(std::vector<float>(capacity*selected_k,-1234.5f),false),old_weights=a.floats(std::vector<float>(capacity*selected_k,-1234.5f),false);
        const auto packed_logits=a.floats(std::vector<float>(capacity*e,-1234.5f),false);
        const auto all_slogit=a.floats(std::vector<float>(capacity+4,-1234.5f),false),old_slogit=a.floats(std::vector<float>(capacity+4,-1234.5f),false);
        const auto all_mix=a.floats(poison_out,false),old_mix=a.floats(poison_out,false);
        for(auto n:{1u,2u,3u,8u,16u,17u,31u,32u,33u,511u,512u}) {
            const GpuRowsF32 in{x,n,c,xs},out{y,n,e,ys};
            const auto ledger=g.memory();g.begin();for(uint32_t t=0;t<n;++t)g.gemv(0,w,g.slice(x,uint64_t(t)*xs,c*4),g.slice(scalar,uint64_t(t)*ys,e*4),c,e);const auto ordinary=g.submit();
            need(ordinary.dispatches==uint64_t(n)*g.gemv_dispatches(0,c,e),"router window scalar control actual dispatch count");
            g.begin();g.moe_router_f32_window(w,in,out);const auto win=g.submit();
            need(win.dispatches==1 && g.memory().allocations==ledger.allocations && g.memory().live_bytes==ledger.live_bytes,"router window physical dispatch/no-encode-allocation invariant");
            need(g.read(y)==g.read(scalar),"router window projection must be byte-identical to 32-lane simd_sum scalar GEMV");
            if(n>=2) {
                g.begin();g.moe_router_f32_window(w,in,{packed_logits,n,e,e*4});static_cast<void>(g.submit());
                for(auto mode:{RouterMode::softmax,RouterMode::sigmoid})for(bool normalise:{false,true}) {
                    g.begin();for(uint32_t t=0;t<n;++t)g.route(g.slice(packed_logits,uint64_t(t)*e*4,e*4),g.slice(old_ids,uint64_t(t)*selected_k*4,selected_k*4),g.slice(old_weights,uint64_t(t)*selected_k*4,selected_k*4),selected_k,mode,normalise);static_cast<void>(g.submit());
                    g.begin();g.moe_route_rows(g.slice(packed_logits,0,uint64_t(n)*e*4),g.slice(all_ids,0,uint64_t(n)*selected_k*4),g.slice(all_weights,0,uint64_t(n)*selected_k*4),e,selected_k,n,mode,normalise);const auto rbatch=g.submit();
                    need(rbatch.dispatches==1 && g.read(all_ids)==g.read(old_ids) && g.read(all_weights)==g.read(old_weights),"router window multi-row route must be byte-identical to scalar route");
                }
                g.begin();for(uint32_t t=0;t<n;++t)g.gemv(0,sw,g.slice(x,uint64_t(t)*xs,c*4),g.slice(old_slogit,uint64_t(t)*4,4),c,1);static_cast<void>(g.submit());
                g.begin();g.moe_shared_router_rows(sw,in,g.slice(all_slogit,0,uint64_t(n)*4));const auto sbatch=g.submit();
                need(sbatch.dispatches==1 && g.read(all_slogit)==g.read(old_slogit),"router window shared router must be byte-identical to scalar split/direct GEMV");
                g.begin();for(uint32_t t=0;t<n;++t)g.mixture(g.slice(down_gpu,uint64_t(t)*selected_k*c*4,uint64_t(selected_k)*c*4),g.slice(old_weights,uint64_t(t)*selected_k*4,selected_k*4),g.slice(shared_down_gpu,uint64_t(t)*c*4,c*4),g.slice(old_slogit,uint64_t(t)*4,4),g.slice(old_mix,uint64_t(t)*xs,c*4));static_cast<void>(g.submit());
                g.begin();g.moe_mixture_rows(g.slice(down_gpu,0,uint64_t(n)*selected_k*c*4),g.slice(all_weights,0,uint64_t(n)*selected_k*4),g.slice(shared_down_gpu,0,uint64_t(n)*c*4),g.slice(all_slogit,0,uint64_t(n)*4),{all_mix,n,c,xs},selected_k);const auto mbatch=g.submit();
                need(mbatch.dispatches==1 && g.read(all_mix)==g.read(old_mix),"router window mixture must be byte-identical to scalar mixture including strided padding");
            }
            need(g.read(w)==saved_w && g.read(sw)==saved_sw && g.read(x)==saved_x,"router window inputs/weights remain immutable");
        }
    }
    // Full encode_moe_mm_many byte-equality against scalar across n = {1, 2, 3, 16, 17, 512}
    const MoeShape shape{2560,32,512,10};
    MetalGraph g_ref(library,512ull<<20),g_win(library,512ull<<20);
    Arena a_ref{g_ref,{}},a_win{g_win,{}};
    for(auto* g:{&g_ref,&g_win}){g->configure_small_gemv(16);g->configure_block_gemv(3);g->configure_prefill_dense_f32(true);g->configure_prefill_grouped_f32(true);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        g->configure_concurrent_dispatch(concurrent);
#endif
    }
    g_win.configure_prefill_router_window(true);
    const auto s_ref=allocate_moe_mm_many_scratch(g_ref,shape,512),s_win=allocate_moe_mm_many_scratch(g_win,shape,512);
    try{
        auto gate_r=make_matrix(a_ref,12,shape.channels,shape.hidden,shape.experts,44),up_r=make_matrix(a_ref,13,shape.channels,shape.hidden,shape.experts,45),down_r=make_matrix(a_ref,7,shape.hidden,shape.channels,shape.experts,46);
        auto sg_r=make_matrix(a_ref,8,shape.channels,shape.hidden,1,47),su_r=make_matrix(a_ref,8,shape.channels,shape.hidden,1,48),sd_r=make_matrix(a_ref,8,shape.hidden,shape.channels,1,49);
        auto gate_w=make_matrix(a_win,12,shape.channels,shape.hidden,shape.experts,44),up_w=make_matrix(a_win,13,shape.channels,shape.hidden,shape.experts,45),down_w=make_matrix(a_win,7,shape.hidden,shape.channels,shape.experts,46);
        auto sg_w=make_matrix(a_win,8,shape.channels,shape.hidden,1,47),su_w=make_matrix(a_win,8,shape.channels,shape.hidden,1,48),sd_w=make_matrix(a_win,8,shape.hidden,shape.channels,1,49);
        std::vector<float> router(uint64_t(shape.channels)*shape.experts),shared(shape.channels);
        uint32_t rng=44;
        for(auto& v:router){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997/256;}
        for(auto& v:shared){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997/64;}
        const MoeWeights w_ref{{a_ref.floats(router),0},gate_r.gpu,up_r.gpu,down_r.gpu,sg_r.gpu,su_r.gpu,sd_r.gpu,{a_ref.floats(shared),0}};
        const MoeWeights w_win{{a_win.floats(router),0},gate_w.gpu,up_w.gpu,down_w.gpu,sg_w.gpu,su_w.gpu,sd_w.gpu,{a_win.floats(shared),0}};
        prepare_moe_mm_many(g_ref,w_ref,shape);prepare_moe_mm_many(g_win,w_win,shape);
        const uint64_t stride=uint64_t(shape.channels)*4;
        std::vector<float> input(512*stride/4,-9876.25f),poison(input.size()+16,-7654.25f);
        for(uint32_t t=0;t<512;++t)for(uint32_t c=0;c<shape.channels;++c){rng=rng*1664525u+1013904223u;input[t*stride/4+c]=float(int(rng%1993)-996)/997;}
        const auto x_ref=a_ref.floats(input),y_ref=a_ref.floats(poison,false),x_win=a_win.floats(input),y_win=a_win.floats(poison,false);
        for(uint32_t n:{1u,2u,3u,16u,17u,512u}) {
            g_ref.begin();encode_moe_mm_many(g_ref,w_ref,s_ref,{x_ref,n,shape.channels,stride},{y_ref,n,shape.channels,stride},shape,RouterMode::softmax,true);const auto b_ref=g_ref.submit();
            g_win.begin();encode_moe_mm_many(g_win,w_win,s_win,{x_win,n,shape.channels,stride},{y_win,n,shape.channels,stride},shape,RouterMode::softmax,true);const auto b_win=g_win.submit();
            need(b_ref.dispatches==moe_mm_many_dispatches(g_ref,w_ref,shape,n) && b_win.dispatches==moe_mm_many_dispatches(g_win,w_win,shape,n),"router window full MoE dispatch count contract");
            if(n==1)need(b_win.dispatches==b_ref.dispatches && g_win.router_window_encoded()==std::array<uint64_t,4>{0,0,0,0},"router window N=1 fallback must not encode window kernels");
            else need(b_win.dispatches==16 && b_ref.dispatches==12+5*uint64_t(n),"router window N>=2 replaces 5*N dispatches with 4 dispatches");
            need(g_ref.read(y_ref)==g_win.read(y_win),"router window full encode_moe_mm_many output must be byte-identical to scalar");
            const auto lbytes=uint64_t(n)*shape.experts*4,rbytes=uint64_t(n)*shape.selected*4,sbytes=uint64_t(n)*4;
            need(g_ref.read(g_ref.slice(s_ref.rows.logits,0,lbytes))==g_win.read(g_win.slice(s_win.rows.logits,0,lbytes)),"router window full MoE logits byte identity");
            need(g_ref.read(g_ref.slice(s_ref.rows.indices,0,rbytes))==g_win.read(g_win.slice(s_win.rows.indices,0,rbytes)),"router window full MoE indices byte identity");
            need(g_ref.read(g_ref.slice(s_ref.rows.weights,0,rbytes))==g_win.read(g_win.slice(s_win.rows.weights,0,rbytes)),"router window full MoE weights byte identity");
            need(g_ref.read(g_ref.slice(s_ref.rows.shared_logit,0,sbytes))==g_win.read(g_win.slice(s_win.rows.shared_logit,0,sbytes)),"router window full MoE shared_logit byte identity");
            std::cout<<"MoE router_window full_compare N="<<n<<" scalar_dispatches="<<b_ref.dispatches<<" window_dispatches="<<b_win.dispatches<<" scalar_gpu_ms="<<b_ref.gpu_ms<<" window_gpu_ms="<<b_win.gpu_ms<<" scalar_wall_ms="<<b_ref.wall_ms<<" window_wall_ms="<<b_win.wall_ms<<" byte_exact=true\n";
        }
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
        // Same-binary projection speed check at N=512 (48 layers in one command buffer): scalar vs held batch vs exact window
        {
            MetalGraph g_batch(library,256ull<<20);Arena a_batch{g_batch,{}};
            g_batch.configure_small_gemv(16);g_batch.configure_block_gemv(3);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
            g_batch.configure_concurrent_dispatch(concurrent);
#endif
            g_batch.configure_moe_router_f32_batch(true);g_batch.prepare_moe_router_f32_batch();
            const auto w_b=a_batch.floats(router),x_b=a_batch.floats(input),y_b=a_batch.floats(std::vector<float>(512*512),false);
            for(int warmup=0;warmup<2;++warmup){
                g_batch.begin();for(int l=0;l<48;++l)g_batch.moe_router_f32_batch(w_b,{x_b,512,2560,stride},{y_b,512,512,512*4});static_cast<void>(g_batch.submit());
                g_win.begin();for(int l=0;l<48;++l)g_win.moe_router_f32_window(w_win.router.view,{x_win,512,2560,stride},{s_win.rows.logits,512,512,512*4});static_cast<void>(g_win.submit());
            }
            double batch_gpu=0,window_gpu=0;
            for(int rep=0;rep<5;++rep){
                g_batch.begin();for(int l=0;l<48;++l)g_batch.moe_router_f32_batch(w_b,{x_b,512,2560,stride},{y_b,512,512,512*4});batch_gpu+=g_batch.submit().gpu_ms;
                g_win.begin();for(int l=0;l<48;++l)g_win.moe_router_f32_window(w_win.router.view,{x_win,512,2560,stride},{s_win.rows.logits,512,512,512*4});window_gpu+=g_win.submit().gpu_ms;
            }
            std::cout<<"MoE router_projection_48layers_N512 held_batch_gpu_ms="<<(batch_gpu/5.0)<<" exact_window_gpu_ms="<<(window_gpu/5.0)<<" component_only=true\n";
        }
#endif
    }catch(...){try{g_ref.abort();}catch(const FormatError&){}try{g_win.abort();}catch(const FormatError&){}release_moe_mm_many_scratch(g_ref,s_ref);release_moe_mm_many_scratch(g_win,s_win);throw;}
    release_moe_mm_many_scratch(g_ref,s_ref);release_moe_mm_many_scratch(g_win,s_win);
    std::cout<<"PASS generated exact MoE router window controls, byte-equality verified, no model qualification\n";
}
#endif
void cpu_controls(){
    bool weight_rounding=false;
    for(auto type:{7u,8u,12u,13u}){const auto m=generated_matrix(type,type==12 || type==13?2560:640,17,16,44);for(auto value:m.decoded)weight_rounding|=value!=grouped_fixture::staged(value);}
    need(weight_rounding && grouped_fixture::staged(719.0f/997)!=719.0f/997,"declared half operand rounding is exercised");
    const std::array<float,2> zero{0,0};const std::array<double,2> signal{0.001,-0.002};need(error(zero,signal).l2==1,"relative L2 rejects omitted output despite scaled floor");
    const MoeShape s{256,64,16,10};auto gw=generated_matrix(12,256,64,16,44),uw=generated_matrix(13,256,64,16,45),dw=generated_matrix(7,64,256,16,46),sg=generated_matrix(8,256,64,1,47),su=generated_matrix(8,256,64,1,48),sd=generated_matrix(8,64,256,1,49);
    const std::array<Matrix*,6> matrices{&gw,&uw,&dw,&sg,&su,&sd};std::vector<double> logits(16);for(unsigned e=0;e<16;++e)logits[e]=double(int(e)-8)/32;const auto routes=route_reference(logits,10,RouterMode::softmax,true);
    std::vector<float> input(256);uint32_t rng=44;for(auto& value:input){rng=rng*1664525u+1013904223u;value=float(int(rng%1993)-996)/997;}input[0]=1;
    const auto raw=reference(s,input,matrices,routes,false),half=reference(s,input,matrices,routes,true);std::vector<float> rounded(half.begin(),half.end());const auto difference=error(rounded,raw);
    need(difference.maximum<=scaled_limit && difference.l2<=relative_l2_limit,"prospective CPU raw/half generated domain screen");
    std::cout<<"PASS CPU generated MoE checks="<<checks<<" raw_half_scaled="<<difference.maximum<<" raw_half_relative_l2="<<difference.l2<<" no Metal/model execution\n";
}
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
void verify_nax_top10_mixture_controls(const char* library) {
    MetalGraph g(library, 256ull << 20); Arena a{g, {}};
    g.configure_small_gemv(16); g.configure_block_gemv(3);
    g.configure_verify_nax(true); g.configure_verify_nax_m8(true);
    // No prefill configuration or prepare_moe_mm_many: verification owns preparation.
    constexpr MoeShape shape{64, 32, 16, 4};
    constexpr uint32_t capacity = 8;
    constexpr float poison = -7654.25f;
    auto gate_w = make_matrix(a, 8, shape.channels, shape.hidden, shape.experts, 44);
    auto up_w = make_matrix(a, 8, shape.channels, shape.hidden, shape.experts, 45);
    auto down_w = make_matrix(a, 8, shape.hidden, shape.channels, shape.experts, 46);
    auto shared_gate = make_matrix(a, 8, shape.channels, shape.hidden, 1, 47);
    auto shared_up = make_matrix(a, 8, shape.channels, shape.hidden, 1, 48);
    auto shared_down = make_matrix(a, 8, shape.hidden, shape.channels, 1, 49);
    std::vector<float> router(shape.channels * shape.experts), shared_router(shape.channels);
    for (uint32_t e = 0; e < shape.experts; ++e) router[e * shape.channels] = float(e) / 16;
    shared_router[0] = 0.25f;
    const MoeWeights weights{{a.floats(router), 0}, gate_w.gpu, up_w.gpu, down_w.gpu,
        shared_gate.gpu, shared_up.gpu, shared_down.gpu, {a.floats(shared_router), 0}};
    const auto scratch = allocate_moe_mm_many_scratch(g, shape, capacity);
    const auto control = allocate_moe_mm_many_scratch(g, shape, capacity);
    std::vector<float> input(capacity * shape.channels, 0.125f), output(capacity * shape.channels + 16, poison);
    for (uint32_t t = 0; t < capacity; ++t) input[t * shape.channels] = t % 2 ? -0.5f : 0.5f;
    const auto x = a.floats(input, false), y = a.floats(output, false), baseline = a.floats(output, false);
    const std::array layouts{std::pair{8u, PrefillWeightLayout::gguf}};
    g.prepare_verify_nax(layouts);
    static_cast<void>(g.configure_profiling(128)); // Kernel selection proof only, no timing assertion.
    const auto ledger = g.memory();
    const auto check_counts = [&](VerifyNaxTop10MixtureTelemetry before, VerifyNaxTop10MixtureTelemetry after, uint32_t n) {
        const bool enabled = verify_nax_batch_top10_mixture_enabled;
        need(after.top10_batched_calls - before.top10_batched_calls == (enabled ? 1u : 0u) &&
            after.top10_batched_rows - before.top10_batched_rows == (enabled ? n : 0u) &&
            after.top10_scalar_calls - before.top10_scalar_calls == (enabled ? 0u : n) &&
            after.mixture_batched_calls - before.mixture_batched_calls == (enabled ? 1u : 0u) &&
            after.mixture_batched_rows - before.mixture_batched_rows == (enabled ? n : 0u) &&
            after.mixture_scalar_calls - before.mixture_scalar_calls == (enabled ? 0u : n),
            "verification-only accounting disagrees with selected dispatches");
    };
    const auto slice = [&](GpuView v, uint32_t row, uint64_t width) { return g.slice(v, uint64_t(row) * width * 4, width * 4); };
    try {
        for (uint32_t n = 2; n <= 8; ++n) for (auto mode : {RouterMode::softmax, RouterMode::sigmoid}) for (bool normalise : {false, true}) {
            g.write(y, std::as_bytes(std::span(output))); g.write(baseline, std::as_bytes(std::span(output)));
            // Production computes this before entering VerifyNaxScopeGuard.
            const auto expected = verify_nax_moe_dispatches(g, weights, shape, n);
            const auto before = g.verify_nax_top10_mixture_telemetry();
            {
                VerifyNaxScopeGuard scope(g, true, true);
                need(expected == verify_nax_moe_dispatches(g, weights, shape, n), "verify batch estimator depends on transient scope");
                g.begin();
                encode_verify_nax_moe(g, weights, scratch.rows, {}, x, y, shape, mode, normalise, n);
                const auto result = g.submit();
                need(result.dispatches == expected, "verify batch encoder/estimator disagreement");
                g.record_prefill_moe_chunk(n, true, false, true, false);
            }
            const auto profile = g.last_profile();
            const auto count = [&](const char* kernel) { return uint64_t(std::count_if(profile.kernels.begin(), profile.kernels.end(), [&](const auto& k) { return k.kernel == kernel; })); };
            const bool enabled = verify_nax_batch_top10_mixture_enabled;
            need(count("graph_route_batch") == (enabled ? 1 : 0) && count("graph_route") == (enabled ? 0 : n), "verify top10 selected kernel count");
            need(count("graph_mixture_batch") == (enabled ? 1 : 0) && count("graph_mixture") == (enabled ? 0 : n), "verify mixture selected kernel count");
            const auto after = g.verify_nax_top10_mixture_telemetry();
            check_counts(before, after, n);
            need(g.prefill_top10_mixture_telemetry().moe_batch_calls == 0, "verification polluted prefill counters");
            // Compare the exact scalar operators on the encoded projection results.
            g.begin();
            for (uint32_t t = 0; t < n; ++t) {
                g.route(slice(scratch.rows.logits, t, shape.experts), slice(control.rows.indices, t, shape.selected),
                    slice(control.rows.weights, t, shape.selected), shape.selected, mode, normalise);
                g.mixture(slice(scratch.rows.down, t, shape.selected * shape.channels), slice(control.rows.weights, t, shape.selected),
                    slice(scratch.rows.shared_down, t, shape.channels), slice(scratch.rows.shared_logit, t, 1), slice(baseline, t, shape.channels));
            }
            static_cast<void>(g.submit());
            need(g.read(g.slice(scratch.rows.indices, 0, uint64_t(n) * shape.selected * 4)) == g.read(g.slice(control.rows.indices, 0, uint64_t(n) * shape.selected * 4)), "verify top10 IDs differ from scalar");
            need(g.read(g.slice(scratch.rows.weights, 0, uint64_t(n) * shape.selected * 4)) == g.read(g.slice(control.rows.weights, 0, uint64_t(n) * shape.selected * 4)), "verify top10 weights differ from scalar");
            need(g.read(y) == g.read(baseline), "verify mixture output or inactive canary differs from scalar");
            for (float value : read(g, y)) need(std::isfinite(value), "nonfinite valid verification output");
            need(g.memory().allocations == ledger.allocations && g.memory().live_bytes == ledger.live_bytes, "verify batch encode allocated scratch");
            need(g.verify_nax_top10_mixture_telemetry().top10_scalar_calls == after.top10_scalar_calls &&
                g.verify_nax_top10_mixture_telemetry().mixture_scalar_calls == after.mixture_scalar_calls,
                "ordinary scalar reference polluted verification counters");
        }
        refuses([&] { g.record_verify_nax_moe_ops(8, true, false); });
        {
            VerifyNaxScopeGuard scope(g, true, true);
            g.set_mtp_priming_scope(true);
            refuses([&] { g.record_verify_nax_moe_ops(8, true, false); });
            g.set_mtp_priming_scope(false);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            g.set_dflash2_scope(true);
            refuses([&] { g.record_verify_nax_moe_ops(8, false, false); });
            g.set_dflash2_scope(false);
#endif
        }
        // A failed submission must not clear lifetime counts or poison the valid retry.
        const auto retry_before = g.read(y);
        const auto before_failure = g.verify_nax_top10_mixture_telemetry();
        auto bad = input; bad[shape.channels + 3] = std::numeric_limits<float>::quiet_NaN();
        g.write(x, std::as_bytes(std::span(bad)));
        refuses([&] {
            VerifyNaxScopeGuard scope(g, true, true); g.begin();
            encode_verify_nax_moe(g, weights, scratch.rows, {}, x, y, shape, RouterMode::sigmoid, true, 8);
            static_cast<void>(g.submit());
        });
        try { g.abort(); } catch (const FormatError&) {}
        const auto failed = g.verify_nax_top10_mixture_telemetry();
        check_counts(before_failure, failed, 8);
        g.write(x, std::as_bytes(std::span(input))); g.write(y, std::as_bytes(std::span(output)));
        {
            VerifyNaxScopeGuard scope(g, true, true); g.begin();
            encode_verify_nax_moe(g, weights, scratch.rows, {}, x, y, shape, RouterMode::sigmoid, true, 8);
            static_cast<void>(g.submit());
        }
        need(g.read(y) == retry_before, "verify batching valid retry changed output/canaries");
        const auto retried = g.verify_nax_top10_mixture_telemetry();
        check_counts(failed, retried, 8);
    } catch (...) {
        try { g.abort(); } catch (const FormatError&) {}
        release_moe_mm_many_scratch(g, scratch); release_moe_mm_many_scratch(g, control); throw;
    }
    release_moe_mm_many_scratch(g, scratch); release_moe_mm_many_scratch(g, control);
    std::cout << "PASS verify_nax_top10_mixture_controls enabled=" << verify_nax_batch_top10_mixture_enabled << " checks=" << checks << '\n';
}
#endif
void prefill_top10_mixture_controls(const char* library){
    MetalGraph g(library,1536ull<<20);Arena a{g,{}};
    g.configure_small_gemv(16);g.configure_block_gemv(3);g.configure_prefill_dense_f32(true);g.configure_prefill_dense_nax(true);g.configure_prefill_grouped_f32(true);g.configure_prefill_grouped_nax(true);
    g.prepare_prefill_top10_mixture();
    const bool b_top10 = g.prefill_batch_top10();
    const bool b_mix = g.prefill_batch_mixture();
    std::cout<<"prefill_top10_mixture_controls batch_top10="<<b_top10<<" batch_mixture="<<b_mix<<std::endl;
    constexpr uint32_t c=2560,e=512,selected_k=10,capacity=2048;
    constexpr uint64_t xs=c*4+32;
    std::vector<float> logits_host(uint64_t(capacity)*e),down_rows(uint64_t(capacity)*selected_k*c),shared_down_rows(uint64_t(capacity)*c),slogit_host(capacity);
    uint32_t rng=44;
    for(uint32_t t=0;t<capacity;++t){
        slogit_host[t]=float(int((rng=rng*1664525u+1013904223u)%1993)-996)/997;
        for(uint32_t ex=0;ex<e;++ex){
            rng=rng*1664525u+1013904223u;
            logits_host[uint64_t(t)*e+ex]=float(int(rng%1993)-996)/997;
        }
    }
    // Inject deterministic tie patterns in rows 0, 1, 2:
    // Row 0: all 512 experts tied at 0.0f -> must select 0..9
    for(uint32_t ex=0;ex<e;++ex)logits_host[ex]=0.0f;
    // Row 1: 4-way ties across SIMD groups (ex, ex+128, ex+256, ex+384)
    for(uint32_t ex=0;ex<128;++ex){
        const float val = float(128 - int(ex)) / 64.0f;
        logits_host[e + ex] = val;
        logits_host[e + ex + 128] = val;
        logits_host[e + ex + 256] = val;
        logits_host[e + ex + 384] = val;
    }
    // Row 2: negative ties
    for(uint32_t ex=0;ex<e;++ex)logits_host[2*e + ex] = (ex % 17 == 0) ? -1.5f : -12.5f;
    for(auto& v:down_rows){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997/16;}
    for(auto& v:shared_down_rows){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997/16;}
    const auto packed_logits=a.floats(logits_host);
    const auto down_gpu=a.floats(down_rows),shared_down_gpu=a.floats(shared_down_rows),all_slogit=a.floats(slogit_host);
    std::vector<float> poison_routes(uint64_t(capacity)*selected_k+16,-1234.5f),poison_out(uint64_t(capacity)*(xs/4)+16,-7654.25f);
    const auto all_ids=a.floats(poison_routes,false),old_ids=a.floats(poison_routes,false);
    const auto all_weights=a.floats(poison_routes,false),old_weights=a.floats(poison_routes,false);
    const auto all_mix=a.floats(poison_out,false),old_mix=a.floats(poison_out,false);

    for(uint32_t n:{2u,3u,4u,5u,6u,7u,8u,16u,17u,31u,32u,33u,511u,512u,2048u}){
        for(auto mode:{RouterMode::softmax,RouterMode::sigmoid})for(bool normalise:{false,true}){
            g.write(all_ids,std::as_bytes(std::span(poison_routes)));
            g.write(old_ids,std::as_bytes(std::span(poison_routes)));
            g.write(all_weights,std::as_bytes(std::span(poison_routes)));
            g.write(old_weights,std::as_bytes(std::span(poison_routes)));
            g.begin();for(uint32_t t=0;t<n;++t)g.route(g.slice(packed_logits,uint64_t(t)*e*4,e*4),g.slice(old_ids,uint64_t(t)*selected_k*4,selected_k*4),g.slice(old_weights,uint64_t(t)*selected_k*4,selected_k*4),selected_k,mode,normalise);const auto rscalar=g.submit();
            need(rscalar.dispatches==n,"scalar route dispatch count");
            if(b_top10){
                const auto ledger=g.memory();
                g.begin();g.prefill_route_batch(g.slice(packed_logits,0,uint64_t(n)*e*4),g.slice(all_ids,0,uint64_t(n)*selected_k*4),g.slice(all_weights,0,uint64_t(n)*selected_k*4),e,selected_k,n,mode,normalise);const auto rbatch=g.submit();
                need(rbatch.dispatches==1 && g.memory().allocations==ledger.allocations && g.memory().live_bytes==ledger.live_bytes,"prefill_route_batch single dispatch/no-allocation");
                need(g.read(all_ids)==g.read(old_ids) && g.read(all_weights)==g.read(old_weights),"prefill_route_batch byte-identical IDs/weights/tail-padding to scalar route");
                const auto tied=g.read(g.slice(all_ids,0,selected_k*4));
                for(uint32_t j=0;j<selected_k;++j){uint32_t id=0;std::memcpy(&id,tied.data()+j*4,4);need(id==j,"prefill_route_batch exact tie selects lowest ID");}
            }
        }
        g.write(all_mix,std::as_bytes(std::span(poison_out)));
        g.write(old_mix,std::as_bytes(std::span(poison_out)));
        g.begin();for(uint32_t t=0;t<n;++t)g.mixture(g.slice(down_gpu,uint64_t(t)*selected_k*c*4,uint64_t(selected_k)*c*4),g.slice(old_weights,uint64_t(t)*selected_k*4,selected_k*4),g.slice(shared_down_gpu,uint64_t(t)*c*4,c*4),g.slice(all_slogit,uint64_t(t)*4,4),g.slice(old_mix,uint64_t(t)*xs,c*4));const auto mscalar=g.submit();
        need(mscalar.dispatches==n,"scalar mixture dispatch count");
        if(b_mix){
            const auto ledger=g.memory();
            g.begin();g.prefill_mixture_batch(g.slice(down_gpu,0,uint64_t(n)*selected_k*c*4),g.slice(old_weights,0,uint64_t(n)*selected_k*4),g.slice(shared_down_gpu,0,uint64_t(n)*c*4),g.slice(all_slogit,0,uint64_t(n)*4),{all_mix,n,c,xs},selected_k);const auto mbatch=g.submit();
            need(mbatch.dispatches==1 && g.memory().allocations==ledger.allocations && g.memory().live_bytes==ledger.live_bytes,"prefill_mixture_batch single dispatch/no-allocation");
            need(g.read(all_mix)==g.read(old_mix),"prefill_mixture_batch byte-identical output/strided-padding/tail-padding to scalar mixture");
        }
    }
    // Sticky error flag checks for non-finite inputs
    if(b_top10){
        auto bad_logits=logits_host;bad_logits[e+7]=std::numeric_limits<float>::quiet_NaN();
        const auto bad_l_gpu=a.floats(bad_logits);
        refuses([&]{g.begin();g.prefill_route_batch(g.slice(bad_l_gpu,0,4ull*e*4),g.slice(all_ids,0,4ull*selected_k*4),g.slice(all_weights,0,4ull*selected_k*4),e,selected_k,4,RouterMode::softmax,true);static_cast<void>(g.submit());});
        try{g.abort();}catch(const FormatError&){}
    }
    if(b_mix){
        auto bad_slogit=slogit_host;bad_slogit[2]=std::numeric_limits<float>::infinity();
        const auto bad_sl_gpu=a.floats(bad_slogit);
        refuses([&]{g.begin();g.prefill_mixture_batch(g.slice(down_gpu,0,4ull*selected_k*c*4),g.slice(old_weights,0,4ull*selected_k*4),g.slice(shared_down_gpu,0,4ull*c*4),g.slice(bad_sl_gpu,0,4ull*4),{all_mix,4,c,xs},selected_k);static_cast<void>(g.submit());});
        try{g.abort();}catch(const FormatError&){}
    }
    // Full encode_moe_mm_many check across n = {1, 2, 3, 16, 17, 512, 2048}
    const MoeShape shape{2560,64,512,10};
    const auto s_cand=allocate_moe_mm_many_scratch(g,shape,2048,true);
    try{
        auto gate_w=make_matrix(a,12,shape.channels,shape.hidden,shape.experts,44),up_w=make_matrix(a,13,shape.channels,shape.hidden,shape.experts,45),down_w=make_matrix(a,7,shape.hidden,shape.channels,shape.experts,46);
        auto sg_w=make_matrix(a,8,shape.channels,shape.hidden,1,47),su_w=make_matrix(a,8,shape.channels,shape.hidden,1,48),sd_w=make_matrix(a,8,shape.hidden,shape.channels,1,49);
        std::vector<float> router(uint64_t(shape.channels)*shape.experts),shared(shape.channels);
        for(auto& v:router){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997/256;}
        for(auto& v:shared){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997/64;}
        const MoeWeights w_cand{{a.floats(router),0},gate_w.gpu,up_w.gpu,down_w.gpu,sg_w.gpu,su_w.gpu,sd_w.gpu,{a.floats(shared),0}};
        prepare_moe_mm_many(g,w_cand,shape);
        const uint64_t stride=uint64_t(shape.channels)*4;
        std::vector<float> input(uint64_t(2048)*shape.channels,-9876.25f),poison(input.size()+16,-7654.25f);
        for(auto& v:input){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997;}
        const auto x_in=a.floats(input),y_out=a.floats(poison,false);
        const auto fnv1a=[](std::span<const std::byte> bytes){
            uint64_t h=14695981039346656037ull;
            for(auto b:bytes){h^=uint64_t(std::to_integer<uint8_t>(b));h*=1099511628211ull;}
            return h;
        };
        for(uint32_t n:{1u,2u,3u,16u,17u,512u,2048u}){
            g.write(y_out,std::as_bytes(std::span(poison)));
            g.reset_prefill_top10_mixture_telemetry();
            g.begin();encode_moe_mm_many(g,w_cand,s_cand,{x_in,n,shape.channels,stride},{y_out,n,shape.channels,stride},shape,RouterMode::softmax,true);const auto b=g.submit();
            const auto expected_d=moe_mm_many_dispatches(g,w_cand,shape,n);
            need(b.dispatches==expected_d,"encode_moe_mm_many dispatch contract across n");
            const auto tel=g.prefill_top10_mixture_telemetry();
            need(tel.moe_batch_calls==1 && tel.moe_batch_rows==n && tel.moe_batch_min_rows==n && tel.moe_batch_max_rows==n,"telemetry batch width accounting");
            if(n==1){
                need(tel.top10_batched_calls==0 && tel.top10_scalar_calls==1 && tel.top10_fallbacks==(b_top10?1u:0u),"N=1 top10 fallback accounting");
                need(tel.mixture_batched_calls==0 && tel.mixture_scalar_calls==1 && tel.mixture_fallbacks==(b_mix?1u:0u),"N=1 mixture fallback accounting");
            }else{
                need(tel.top10_batched_calls==(b_top10?1u:0u) && tel.top10_batched_rows==(b_top10?n:0u) && tel.top10_scalar_calls==(b_top10?0u:n) && tel.top10_fallbacks==0,"N>=2 top10 batch accounting");
                need(tel.mixture_batched_calls==(b_mix?1u:0u) && tel.mixture_batched_rows==(b_mix?n:0u) && tel.mixture_scalar_calls==(b_mix?0u:n) && tel.mixture_fallbacks==0,"N>=2 mixture batch accounting");
            }
            const auto out_fnv=fnv1a(g.read(y_out));
            const auto ids_fnv=fnv1a(g.read(g.slice(s_cand.rows.indices,0,uint64_t(n)*shape.selected*4)));
            const auto wts_fnv=fnv1a(g.read(g.slice(s_cand.rows.weights,0,uint64_t(n)*shape.selected*4)));
            std::cout<<"encode_moe_mm_many N="<<n<<" dispatches="<<b.dispatches<<" top10_batched="<<tel.top10_batched_calls<<" mix_batched="<<tel.mixture_batched_calls<<" fallbacks="<<(tel.top10_fallbacks+tel.mixture_fallbacks)<<" out_fnv="<<std::hex<<out_fnv<<" ids_fnv="<<ids_fnv<<" wts_fnv="<<wts_fnv<<std::dec<<std::endl;
        }
    }catch(...){try{g.abort();}catch(const FormatError&){}release_moe_mm_many_scratch(g,s_cand);throw;}
    release_moe_mm_many_scratch(g,s_cand);
    std::cout<<"PASS prefill_top10_mixture_controls checks="<<checks<<std::endl;
}
}
int main(int argc,char** argv){try{
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    if (argc == 3 && std::string(argv[1]) == "--execute-verify-nax-top10-mixture-controls" && argv[2][0] == '/') {
        need(host_thermal_state() == 0, "verify batching controls require a coordinated nominal GPU window");
        verify_nax_top10_mixture_controls(argv[2]); return 0;
    }
#endif

    if(argc==2 && std::string(argv[1])=="--check-generated-cpu"){cpu_controls();return 0;}
    if(argc==3 && std::string(argv[1])=="--execute-prefill-top10-mixture-controls" && argv[2][0]=='/'){
        while(host_thermal_state()!=0){std::cerr<<"waiting for nominal thermal state\n";std::this_thread::sleep_for(std::chrono::seconds(5));}
        prefill_top10_mixture_controls(argv[2]);
        return 0;
    }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    if(argc==3 && std::string(argv[1])=="--execute-f32-routes" && argv[2][0]=='/'){
        need(host_thermal_state()==0,"F32 route scope control requires a coordinated nominal GPU window");
        for(auto n:{3u,512u})f32_route_scope_control(argv[2],n);
        std::cout<<"PASS F32 route scope generated checks="<<checks<<" no numerical/state/model/task gate admission\n";return 0;
    }
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    const bool router_only=(argc==3||(argc==4&&std::string(argv[3])=="concurrent"))&&std::string(argv[1])=="--execute-router-controls";
#else
    const bool router_only=false;
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    const bool router_window_only=(argc==3||(argc==4&&std::string(argv[3])=="concurrent"))&&std::string(argv[1])=="--execute-router-window-controls";
#else
    const bool router_window_only=false;
#endif
    need(((argc==3&&std::string(argv[1])=="--execute-generated")||router_only||router_window_only) && argv[2][0]=='/',"usage: test-moe_mm_many_gpu --execute-generated|--execute-router-controls|--execute-router-window-controls /absolute/library.metallib");
    std::cout<<"prospective scaled_limit="<<scaled_limit<<" relative_l2_limit="<<relative_l2_limit<<" max_weight=1/256 input_domain=[-1,1] half_u=2^-11 no_model_admission=true\n";
    while(host_thermal_state()!=0){std::cerr<<"waiting for nominal thermal state\n";std::this_thread::sleep_for(std::chrono::seconds(5));}
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    if(router_window_only){
        router_window_controls(argv[2],argc==4);
        std::cout<<"PASS bounded router window controls and full MoE byte-equality cases; no model admission\n";return 0;
    }
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    const bool router_concurrent=router_only&&argc==4;
    router_projection_controls(argv[2],router_concurrent);
    // Actual router geometry with bounded hidden width exercises composition without model weights.
    exercise(argv[2],{2560,32,512,1},3,12,7,RouterMode::softmax,true,router_concurrent);
    // An enabled capability must retain the scalar projection for other router geometries.
    exercise(argv[2],{256,64,16,10},3,12,7,RouterMode::sigmoid,true,router_concurrent);
    if(router_only){std::cout<<"PASS bounded router controls and existing MoE composition/fallback cases; no model admission\n";return 0;}
#endif
    exercise(argv[2],{256,64,16,10},3,12,7,RouterMode::softmax);exercise(argv[2],{256,64,16,10},512,13,8,RouterMode::sigmoid);
    exercise(argv[2],{2560,640,16,10},3,12,7,RouterMode::sigmoid);exercise(argv[2],{2560,640,16,10},3,13,8,RouterMode::softmax);
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    // Production channel width, where the shared router is a two-stage GEMV, across more rows than it has partial-sum regions.
    exercise(argv[2],{2560,64,16,10},130,13,8,RouterMode::softmax,false,true);
    // Production router geometry, where the 512-expert projection is a single dispatch per row.
    exercise(argv[2],{2560,32,512,10},130,13,8,RouterMode::softmax,false,true);
#endif
    std::cout<<"PASS generated full MoE cases="<<cases<<" checks="<<checks<<" D-Cal/state/model/task gates remain open\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
