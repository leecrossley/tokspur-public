#include "tokspur/gdn_tile.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
using namespace tokspur;
size_t checks=0,batches=0;double worst=0;
constexpr float sentinel=-12345.25f;
void require(bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);}
void reject(const std::function<void()>& f){++checks;try{f();}catch(const FormatError&){return;}throw std::runtime_error("invalid GDN tile operation admitted");}
std::vector<double> real(std::span<const float> x){return {x.begin(),x.end()};}
struct Arena {
    MetalGraph& g;std::vector<GpuView> all;
    GpuView values(std::span<const float> x,bool immutable=true){
        std::vector<std::byte> bytes(x.size_bytes()+8,std::byte{0x6d});
        const auto raw=std::as_bytes(x);std::copy(raw.begin(),raw.end(),bytes.begin()+4);
        const auto full=g.upload(bytes,"GDN tile fixture",immutable);all.push_back(full);return g.slice(full,4,x.size_bytes());
    }
    GpuView zero(size_t count){return values(std::vector<float>(count),false);}
    GpuView output(size_t count){return values(std::vector<float>(count,sentinel),false);}
    void guards()const{for(auto v:all){const auto raw=g.read(v);for(size_t i=0;i<4;++i)require(raw[i]==std::byte{0x6d}&&raw[raw.size()-1-i]==std::byte{0x6d},"GDN tile crossed a view boundary");}}
    ~Arena(){try{g.abort();}catch(const FormatError&){}for(auto v:all)g.release(v);}
};
std::vector<float> read(MetalGraph& g,GpuView v){const auto raw=g.read(v);std::vector<float>x(raw.size()/4);std::memcpy(x.data(),raw.data(),raw.size());return x;}
void close(std::span<const float> x,std::span<const double> expected){
    require(x.size()==expected.size(),"GDN tile scalar-reference geometry");
    for(size_t i=0;i<x.size();++i){const auto e=std::abs(double(x[i])-expected[i])/std::max(1.0,std::abs(expected[i]));worst=std::max(worst,e);require(std::isfinite(x[i])&&e<=5e-5,"GDN tile independent double-reference error");}
}
void submit(MetalGraph& g,uint64_t dispatches,GraphMemory before){
    const auto batch=g.submit();++batches;const auto after=g.memory();
    require(batch.dispatches==dispatches&&before.allocations==after.allocations&&before.live_bytes==after.live_bytes&&before.imported_backing_bytes==after.imported_backing_bytes,"GDN tile dispatch/buffer-ledger invariant");
}
std::vector<float> random_values(size_t count,std::mt19937& random,float scale){
    std::vector<float>x(count);for(auto& value:x)value=float(int(random()%2049)-1024)/1024*scale;return x;
}
void convolution(MetalGraph& g,uint32_t channels,StateTile t,std::mt19937& random){
    const auto bytes=gdn_conv_tile_bytes(channels,t);Arena a{g,{}};
    const auto h=random_values(bytes.history/4,random,0.01f),x=random_values(bytes.projected/4,random,0.01f),w=random_values(bytes.weights/4,random,0.25f);
    const auto accepted=a.values(h),input=a.values(x),weights=a.values(w),candidate=a.zero(h.size()),out=a.output(x.size());
    std::array<GpuView,2> ping{a.zero(h.size()),a.zero(h.size())};const auto baseline=a.output(x.size());const auto before=g.memory();
    auto final=accepted;g.begin();
    for(uint32_t step=0;step<t.steps;++step){g.convolution(final,g.slice(input,uint64_t(step)*channels*4,uint64_t(channels)*4),weights,ping[step%2],g.slice(baseline,uint64_t(step)*channels*4,uint64_t(channels)*4),4,1);final=ping[step%2];}
    submit(g,t.steps,before);const auto expected_state=g.read(final),expected_output=g.read(baseline);
    g.begin();g.gdn_convolution_tile(accepted,input,weights,candidate,out,channels,t);submit(g,1,before);
    require(g.read(candidate)==expected_state&&g.read(out)==expected_output,"convolution tile changed existing GPU bits or unused output rows");
    require(g.read(accepted)==std::vector<std::byte>(std::as_bytes(std::span(h)).begin(),std::as_bytes(std::span(h)).end()),"convolution tile mutated accepted history");
    const uint64_t reference_budget=t.capacity==512?67108864:16777216;
    const auto reference=gdn_convolution_tile_reference(real(h),real(x),real(w),channels,t,reference_budget);
    close(read(g,candidate),reference.history);const auto got=read(g,out);close(std::span(got).first(uint64_t(t.steps)*channels),reference.output);
    for(size_t i=uint64_t(t.steps)*channels;i<got.size();++i)require(got[i]==sentinel,"convolution tile wrote padding");a.guards();
}
void delta(MetalGraph& g,uint32_t heads,uint32_t width,StateTile t,std::mt19937& random){
    const auto bytes=gdn_delta_tile_bytes(heads,width,t);Arena a{g,{}};
    auto state=random_values(bytes.state/4,random,0.01f),q=random_values(bytes.query_key/4,random,0.01f),key=random_values(q.size(),random,0.01f);
    const auto value=random_values(bytes.value_output/4,random,0.01f);std::vector<float> decay(bytes.gates/4),beta(decay.size());
    for(size_t i=0;i<decay.size();++i){decay[i]=i%3==0?0:i%3==1?1:0.95f;beta[i]=i%3==0?1:i%3==1?0:0.5f;}
    const auto accepted=a.values(state),query=a.values(q),keys=a.values(key),values=a.values(value),aa=a.values(decay),bb=a.values(beta);
    const auto candidate=a.zero(state.size()),out=a.output(value.size()),baseline=a.output(value.size());std::array<GpuView,2>ping{a.zero(state.size()),a.zero(state.size())};
    const auto before=g.memory();const uint64_t qstride=uint64_t(heads)*128,vstride=uint64_t(heads)*width;
    auto final=accepted;g.begin();
    for(uint32_t step=0;step<t.steps;++step){
        g.delta(final,g.slice(query,step*qstride*4,qstride*4),g.slice(keys,step*qstride*4,qstride*4),g.slice(values,step*vstride*4,vstride*4),
            g.slice(aa,uint64_t(step)*heads*4,uint64_t(heads)*4),g.slice(bb,uint64_t(step)*heads*4,uint64_t(heads)*4),ping[step%2],g.slice(baseline,step*vstride*4,vstride*4),heads,128,width);final=ping[step%2];
    }
    submit(g,t.steps,before);const auto expected_state=g.read(final),expected_output=g.read(baseline);
    g.begin();g.gdn_delta_tile(accepted,query,keys,values,aa,bb,candidate,out,heads,width,t);submit(g,1,before);
    require(g.read(candidate)==expected_state&&g.read(out)==expected_output,"delta tile changed existing GPU bits or unused output rows");
    require(g.read(accepted)==std::vector<std::byte>(std::as_bytes(std::span(state)).begin(),std::as_bytes(std::span(state)).end()),"delta tile mutated accepted state");
    const uint64_t reference_budget=t.capacity==512?67108864:16777216;
    const auto reference=gdn_delta_tile_reference(real(state),real(q),real(key),real(value),real(decay),real(beta),heads,width,t,reference_budget);
    close(read(g,candidate),reference.state);const auto got=read(g,out);close(std::span(got).first(t.steps*vstride),reference.output);
    for(size_t i=t.steps*vstride;i<got.size();++i)require(got[i]==sentinel,"delta tile wrote padding");a.guards();
}
void padding_and_refusal(MetalGraph& g){
    Arena a{g,{}};constexpr StateTile tile{4,3};
    std::vector<float> history(3,0),projected{1,2,3,std::numeric_limits<float>::quiet_NaN()},weights{0,0,0,1};
    const auto h=a.values(history,false),x=a.values(projected,false),w=a.values(weights,false),next=a.zero(3),out=a.output(4),immutable_history=a.values(history),immutable_output=a.values(std::vector<float>(4));
    const auto before=g.memory();g.begin();g.gdn_convolution_tile(h,x,w,next,out,1,tile);submit(g,1,before);
    require(read(g,next)==std::vector<float>({1,2,3})&&read(g,out)==std::vector<float>({1,2,3,sentinel}),"convolution consumed padded NaN or wrong final history");
    g.begin();reject([&]{g.gdn_convolution_tile(h,x,w,h,out,1,tile);});g.abort();
    g.begin();reject([&]{g.gdn_convolution_tile(h,x,w,next,x,1,tile);});g.abort();
    // All shapes/mutability valid: candidate and output partially overlap.
    g.begin();reject([&]{g.gdn_convolution_tile(h,g.slice(x,0,8),w,next,g.slice(next,4,8),1,{2,1});});g.abort();
    g.begin();reject([&]{g.gdn_convolution_tile(h,x,w,immutable_history,out,1,tile);});g.abort();
    g.begin();reject([&]{g.gdn_convolution_tile(h,x,w,next,immutable_output,1,tile);});g.abort();
    g.begin();reject([&]{g.gdn_convolution_tile(h,x,w,next,out,1,{4,0});});g.abort();
    for(uint32_t badstep:{0,1,2}){
        auto bad=projected;bad[badstep]=INFINITY;g.write(x,std::as_bytes(std::span(bad)));const auto accepted=g.read(h);
        g.begin();g.gdn_convolution_tile(h,x,w,next,out,1,tile);reject([&]{static_cast<void>(g.submit());});require(g.read(h)==accepted,"failed convolution published accepted state");
        g.write(x,std::as_bytes(std::span(projected)));g.begin();g.gdn_convolution_tile(h,x,w,next,out,1,tile);submit(g,1,before);
    }
    for(uint32_t tap=0;tap<4;++tap){
        auto bad=weights;bad[tap]=std::numeric_limits<float>::quiet_NaN();g.write(w,std::as_bytes(std::span(bad)));const auto accepted=g.read(h);
        g.begin();g.gdn_convolution_tile(h,x,w,next,out,1,tile);reject([&]{static_cast<void>(g.submit());});require(g.read(h)==accepted,"failed convolution weight validation mutated accepted history");
        g.write(w,std::as_bytes(std::span(weights)));g.begin();g.gdn_convolution_tile(h,x,w,next,out,1,tile);submit(g,1,before);
    }
    for(uint32_t position:{0,1,2})for(float bad:{INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()}){
        auto invalid=history;invalid[position]=bad;g.write(h,std::as_bytes(std::span(invalid)));const auto accepted=g.read(h);
        g.begin();g.gdn_convolution_tile(h,x,w,next,out,1,tile);reject([&]{static_cast<void>(g.submit());});require(g.read(h)==accepted,"nonfinite accepted history was modified on refusal");
        g.write(h,std::as_bytes(std::span(history)));g.begin();g.gdn_convolution_tile(h,x,w,next,out,1,tile);submit(g,1,before);
        require(read(g,next)==std::vector<float>({1,2,3}),"valid convolution recovery after bad accepted history failed");
    }
    auto huge_input=projected,huge_weights=weights;huge_input[0]=huge_weights[3]=std::numeric_limits<float>::max();
    g.write(x,std::as_bytes(std::span(huge_input)));g.write(w,std::as_bytes(std::span(huge_weights)));
    g.begin();g.gdn_convolution_tile(h,x,w,next,out,1,tile);reject([&]{static_cast<void>(g.submit());});
    g.write(x,std::as_bytes(std::span(projected)));g.write(w,std::as_bytes(std::span(weights)));
    g.begin();g.gdn_convolution_tile(h,x,w,next,out,1,tile);submit(g,1,before);
    std::vector<float> s(128,0),q(4*128,0),key(q.size(),0),value{1,2,3,std::numeric_limits<float>::quiet_NaN()},decay{0,0,0,std::numeric_limits<float>::quiet_NaN()},beta{1,1,1,std::numeric_limits<float>::quiet_NaN()};
    for(uint32_t step=0;step<3;++step)q[step*128]=key[step*128]=1;
    const auto ss=a.values(s,false),qq=a.values(q,false),kk=a.values(key,false),vv=a.values(value,false),aa=a.values(decay,false),bb=a.values(beta,false),candidate=a.zero(128),result=a.output(4),immutable=a.values(s);
    const auto count=g.memory();g.begin();g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,tile);submit(g,1,count);
    require(read(g,result)==std::vector<float>({1,2,3,sentinel})&&read(g,candidate)[0]==3,"delta consumed padded NaN or produced wrong final state");
    g.begin();reject([&]{g.prepare_gdn_tiles(32);});g.abort();
    for(uint32_t threads:{0,16,48,256})reject([&]{g.prepare_gdn_tiles(threads);});
    g.begin();reject([&]{g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,ss,result,1,1,tile);});g.abort();
    g.begin();reject([&]{g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,g.slice(candidate,4,16),1,1,tile);});g.abort();
    g.begin();reject([&]{g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,immutable,result,1,1,tile);});g.abort();
    g.begin();reject([&]{g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,{3,2});});g.abort();
    g.begin();reject([&]{g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,{4,5});});g.abort();
    for(uint32_t position:{0,64,127})for(float bad:{INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()}){
        auto invalid=s;invalid[position]=bad;g.write(ss,std::as_bytes(std::span(invalid)));const auto accepted=g.read(ss);
        g.begin();g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,tile);reject([&]{static_cast<void>(g.submit());});require(g.read(ss)==accepted,"nonfinite accepted matrix was modified on refusal");
        g.write(ss,std::as_bytes(std::span(s)));g.begin();g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,tile);submit(g,1,count);
        require(read(g,result)==std::vector<float>({1,2,3,sentinel}),"valid delta recovery after bad accepted state failed");
    }
    for(uint32_t field:{0,1})for(uint32_t step:{0,1,2})for(float bad:{-0.01f,1.01f,INFINITY,std::numeric_limits<float>::quiet_NaN()}){
        auto bad_gate=field?decay:beta;bad_gate[step]=bad;const auto destination=field?aa:bb;g.write(destination,std::as_bytes(std::span(bad_gate)));const auto accepted=g.read(ss);
        g.begin();g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,tile);reject([&]{static_cast<void>(g.submit());});require(g.read(ss)==accepted,"failed delta published accepted state");
        g.write(destination,std::as_bytes(std::span(field?decay:beta)));g.begin();g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,tile);submit(g,1,count);
    }
    for(uint32_t field:{0,1,2})for(uint32_t step:{0,1,2}){
        auto bad=field==0?q:field==1?key:value;const auto stride=field==2?1:128;bad[step*stride]=std::numeric_limits<float>::quiet_NaN();
        const auto destination=field==0?qq:field==1?kk:vv;g.write(destination,std::as_bytes(std::span(bad)));const auto accepted=g.read(ss);
        g.begin();g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,tile);reject([&]{static_cast<void>(g.submit());});require(g.read(ss)==accepted,"nonfinite prepared delta input changed accepted state");
        const auto& original=field==0?q:field==1?key:value;g.write(destination,std::as_bytes(std::span(original)));
        g.begin();g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,tile);submit(g,1,count);
    }
    auto huge_key=key,huge_value=value;huge_key[0]=huge_value[0]=std::numeric_limits<float>::max();
    g.write(kk,std::as_bytes(std::span(huge_key)));g.write(vv,std::as_bytes(std::span(huge_value)));const auto accepted=g.read(ss);
    g.begin();g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,tile);reject([&]{static_cast<void>(g.submit());});require(g.read(ss)==accepted,"overflowing delta changed accepted state");
    g.write(kk,std::as_bytes(std::span(key)));g.write(vv,std::as_bytes(std::span(value)));
    g.begin();g.gdn_delta_tile(ss,qq,kk,vv,aa,bb,candidate,result,1,1,tile);submit(g,1,count);
    reject([&]{static_cast<void>(g.zeros(g.memory().budget,"must exceed configured tile budget"));});a.guards();
}
void replay_and_extremes(MetalGraph& g){
    Arena a{g,{}};constexpr uint32_t heads=2,width=3,capacity=8;constexpr uint64_t qstride=heads*128,vstride=heads*width;
    std::vector<float> state(heads*128*width),q(capacity*qstride),key(q.size()),value(capacity*vstride),decay(capacity*heads,0.95f),beta(decay.size(),0.5f);
    for(size_t i=0;i<state.size();++i)state[i]=float(int(i%17)-8)/1024;
    for(size_t i=0;i<q.size();++i){q[i]=float(int(i%23)-11)/1024;key[i]=float(int(i%31)-15)/1024;}
    for(size_t i=0;i<value.size();++i)value[i]=float(int(i%19)-9)/512;
    const auto accepted=a.values(state),qq=a.values(q),kk=a.values(key),vv=a.values(value),aa=a.values(decay),bb=a.values(beta),candidate=a.zero(state.size()),output=a.output(value.size()),baseline=a.output(value.size());
    const std::array<GpuView,2>ping{a.zero(state.size()),a.zero(state.size())},serial_ping{a.zero(state.size()),a.zero(state.size())};
    const auto before=g.memory();const auto accepted_bytes=g.read(accepted);
    // First dirty candidates with a full future tile, then recover every strict
    // prefix from the SAME immutable base and prepared transition tape.
    g.begin();g.gdn_delta_tile(accepted,qq,kk,vv,aa,bb,candidate,output,heads,width,{capacity,capacity});submit(g,1,before);
    const std::vector<float> cleared(value.size(),sentinel);
    for(uint32_t k=1;k<=capacity;++k){
        g.write(output,std::as_bytes(std::span(cleared)));g.write(baseline,std::as_bytes(std::span(cleared)));
        auto final=accepted;g.begin();
        for(uint32_t step=0;step<k;++step){g.delta(final,g.slice(qq,step*qstride*4,qstride*4),g.slice(kk,step*qstride*4,qstride*4),g.slice(vv,step*vstride*4,vstride*4),g.slice(aa,uint64_t(step)*heads*4,uint64_t(heads)*4),g.slice(bb,uint64_t(step)*heads*4,uint64_t(heads)*4),ping[step%2],g.slice(baseline,step*vstride*4,vstride*4),heads,128,width);final=ping[step%2];}
        submit(g,k,before);g.begin();g.gdn_delta_tile(accepted,qq,kk,vv,aa,bb,candidate,output,heads,width,{capacity,k});submit(g,1,before);
        require(g.read(candidate)==g.read(final)&&g.read(output)==g.read(baseline)&&g.read(accepted)==accepted_bytes,"delta transition-tape prefix replay differs");
    }
    // Repeated whole-tile publication uses two writable candidates; the
    // immutable rollback base is never recycled as a writable destination.
    auto current=accepted,serial=accepted;
    for(uint32_t round=0;round<16;++round){
        const auto next=ping[round%2];g.begin();g.gdn_delta_tile(current,qq,kk,vv,aa,bb,next,output,heads,width,{capacity,capacity});submit(g,1,before);
        g.begin();for(uint32_t step=0;step<capacity;++step){g.delta(serial,g.slice(qq,step*qstride*4,qstride*4),g.slice(kk,step*qstride*4,qstride*4),g.slice(vv,step*vstride*4,vstride*4),g.slice(aa,uint64_t(step)*heads*4,uint64_t(heads)*4),g.slice(bb,uint64_t(step)*heads*4,uint64_t(heads)*4),serial_ping[step%2],g.slice(baseline,step*vstride*4,vstride*4),heads,128,width);serial=serial_ping[step%2];}submit(g,capacity,before);
        require(g.read(next)==g.read(serial)&&g.read(output)==g.read(baseline),"repeated committed delta tile differs from serial trajectory");current=next;
    }
    require(g.read(accepted)==accepted_bytes,"repeated tile publication changed original rollback base");
    const std::vector<float> history{-3,-2,-1},projected{1,2,3,4,5,6,7,8},weights{0.125f,-0.25f,0.5f,1};
    const auto ch=a.values(history),cx=a.values(projected),cw=a.values(weights),cn=a.zero(3),co=a.output(capacity),cbo=a.output(capacity);
    const std::array<GpuView,2> cp{a.zero(3),a.zero(3)};const auto conv_ledger=g.memory();const auto saved_history=g.read(ch);
    g.begin();g.gdn_convolution_tile(ch,cx,cw,cn,co,1,{capacity,capacity});submit(g,1,conv_ledger);
    const std::vector<float> conv_cleared(capacity,sentinel);
    for(uint32_t k=1;k<=capacity;++k){
        g.write(co,std::as_bytes(std::span(conv_cleared)));g.write(cbo,std::as_bytes(std::span(conv_cleared)));
        auto final=ch;g.begin();for(uint32_t step=0;step<k;++step){g.convolution(final,g.slice(cx,uint64_t(step)*4,4),cw,cp[step%2],g.slice(cbo,uint64_t(step)*4,4),4,1);final=cp[step%2];}submit(g,k,conv_ledger);
        g.begin();g.gdn_convolution_tile(ch,cx,cw,cn,co,1,{capacity,k});submit(g,1,conv_ledger);
        require(g.read(cn)==g.read(final)&&g.read(co)==g.read(cbo)&&g.read(ch)==saved_history,"convolution transition-tape prefix replay differs");
    }
    // Signed zeros and smallest FP32 subnormals remain an exact comparison to
    // the old GPU path, not an assumption about platform denormal behaviour.
    for(float initial:{0.0f,-0.0f,std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min()}){
        std::vector<float> edge(128,initial),query(256,0),keys(256,0),values(2,0),decays(2,1),betas(2,0);query[0]=query[128]=1;
        const auto base=a.values(edge),eq=a.values(query),ek=a.values(keys),ev=a.values(values),ea=a.values(decays),eb=a.values(betas),next=a.zero(128),scratch=a.zero(128),tileout=a.zero(2),serialout=a.zero(2);const auto ledger=g.memory();
        g.begin();g.delta(base,g.slice(eq,0,512),g.slice(ek,0,512),g.slice(ev,0,4),g.slice(ea,0,4),g.slice(eb,0,4),scratch,g.slice(serialout,0,4),1,128,1);
        g.delta(scratch,g.slice(eq,512,512),g.slice(ek,512,512),g.slice(ev,4,4),g.slice(ea,4,4),g.slice(eb,4,4),next,g.slice(serialout,4,4),1,128,1);submit(g,2,ledger);
        const auto expected=g.read(next);g.begin();g.gdn_delta_tile(base,eq,ek,ev,ea,eb,next,tileout,1,1,{2,2});submit(g,1,ledger);
        require(g.read(next)==expected&&g.read(tileout)==g.read(serialout),"zero/subnormal tile differs from old GPU arithmetic");
    }
    a.guards();
}
}
int main(int argc,char**argv){
    try{
        if(argc!=2)throw std::runtime_error("expected owned Metal library");
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        MetalGraph g(argv[1],268435456);
#else
        MetalGraph g(argv[1],32*1024*1024);
#endif
        const auto initial=g.memory().live_bytes;std::mt19937 random(0x47444e54);
        {
            Arena a{g,{}};const auto h=a.zero(3),x=a.zero(2),w=a.zero(4),next=a.zero(3),out=a.zero(2),s=a.zero(128),q=a.zero(256),k=a.zero(256),v=a.zero(2),aa=a.zero(2),bb=a.zero(2),candidate=a.zero(128);
            g.begin();reject([&]{g.gdn_convolution_tile(h,x,w,next,out,1,{2,1});});g.abort();
            g.begin();reject([&]{g.gdn_delta_tile(s,q,k,v,aa,bb,candidate,out,1,1,{2,1});});g.abort();
        }
        uint32_t supported=0;
        for(uint32_t threads:{32,64,128}){
            try{g.prepare_gdn_tiles(threads);}catch(const FormatError&error){std::cout<<"GDN tile group safely refused: "<<threads<<" ("<<error.what()<<")\n";continue;}
            ++supported;std::cout<<"GDN tile supported group: "<<threads<<'\n';
            for(uint32_t capacity:{2,4,8,16,32})for(uint32_t steps=1;steps<=capacity;++steps){convolution(g,31,{capacity,steps},random);delta(g,1,3,{capacity,steps},random);}
            convolution(g,10240,{8,8},random);delta(g,48,128,{8,8},random);
        }
        require(supported,"no supported GDN tile threadgroup");g.prepare_gdn_tiles(32);
        for(uint32_t capacity:{2,4,16})for(uint32_t steps:{uint32_t(1),capacity}){convolution(g,10240,{capacity,steps},random);delta(g,48,128,{capacity,steps},random);}
        for(uint32_t steps:{1,7,16})delta(g,3,127,{16,steps},random);
        for(uint32_t steps:{1,17,32}){convolution(g,10240,{32,steps},random);delta(g,48,128,{32,steps},random);}
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        // Same ordinary GPU arithmetic, new capacity only. Partial writes and
        // final state are compared independently from the grouped FFN tier.
        for(uint32_t steps:{1u,19u,65u,512u}){convolution(g,31,{512,steps},random);delta(g,1,3,{512,steps},random);}
        convolution(g,10240,{512,512},random);delta(g,48,128,{512,65},random);
#endif
        padding_and_refusal(g);replay_and_extremes(g);require(g.memory().live_bytes==initial,"GDN tile GPU buffers leaked");
        const auto memory=g.memory();
        std::cout<<"Owned GDN tile Metal checks passed: "<<checks<<", batches: "<<batches<<", max scaled error: "<<worst
            <<", GPU buffer peak bytes: "<<memory.peak_bytes<<", GPU buffer budget bytes: "<<memory.budget<<'\n';
    }catch(const std::exception&error){std::cerr<<"Owned GDN tile Metal failed: "<<error.what()<<'\n';return 1;}
}
