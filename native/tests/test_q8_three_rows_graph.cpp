#include "tokspur/graph.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/q8_many.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/moe_q8_three_rows.hpp"
#include "grouped_mm_fixture.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>
namespace {
using namespace tokspur;
uint64_t checks=0,cases=0;
void need(bool x,const char* why){++checks;if(!x)throw std::runtime_error(why);}
void refuses(const std::function<void()>& f){bool bad=false;try{f();}catch(const FormatError&){bad=true;}need(bad,"invalid N3 Graph call admitted");}
struct Arena{MetalGraph&g;std::vector<GpuView> owned;
    GpuView bytes(std::span<const std::byte>x,bool immutable=true){std::vector<std::byte> p(x.size()+32,std::byte{0x6d});std::copy(x.begin(),x.end(),p.begin()+16);auto v=g.upload(p,"N3 generated Graph fixture",immutable);owned.push_back(v);return g.slice(v,16,x.size());}
    GpuView floats(std::span<const float>x,bool immutable=true){return bytes(std::as_bytes(x),immutable);}
    void guards(){for(auto v:owned){auto p=g.read(v);for(size_t i=0;i<16;++i)need(p[i]==std::byte{0x6d}&&p[p.size()-1-i]==std::byte{0x6d},"N3 Graph canary changed");}}
    ~Arena(){try{g.abort();}catch(const FormatError&){}for(auto v:owned)g.release(v);}
};
void exercise(const char*lib,uint32_t c,uint32_t r,bool packed,bool wide,bool concurrent){
    const auto thermal_before=host_thermal_state();
    MetalGraph g(lib,128ull<<20);g.configure_q8_three_rows(true);g.configure_small_gemv(16);g.configure_block_gemv(3);
    g.configure_concurrent_dispatch(concurrent);g.configure_hc_exact(HcExactMode::combined);g.configure_wide_q8_many(wide);g.prepare_q8_matvec_many_all();
    refuses([&]{g.configure_q8_three_rows(false);});Arena a{g,{}};
    const auto b=q8_many_bytes({c,r,3});std::vector<std::byte> w(b.matrix);uint32_t rng=44;
    constexpr std::array<uint16_t,7> scales{0x1801,0x3555,0xb555,0x0001,0x8001,0x3c01,0x8000};
    for(size_t at=0;at<w.size();at+=34){const auto d=scales[at/34%scales.size()];std::memcpy(w.data()+at,&d,2);for(size_t j=0;j<32;++j){rng=rng*1664525u+1013904223u;w[at+2+j]=std::byte(rng>>24);}}
    std::vector<std::byte> payload=w;if(packed)lanes::pack_rows(lanes::row_layout(8,c),w,payload);
    const auto matrix=a.bytes(payload);if(packed)g.register_lane_matrix(matrix,{8,c,r,1});
    std::vector<float>x(uint64_t(c)*3);for(auto&v:x){rng=rng*1664525u+1013904223u;v=float(int(rng%1993)-996)/997;}
    const auto input=a.floats(x,false);std::vector<float> poison(uint64_t(r)*3,-7654.25f);const auto y=a.floats(poison,false),old=a.floats(poison,false);
    const auto ledger=g.memory();g.begin();for(uint32_t at=0;at<3;){const auto n=tokspur::q8_many_width(3-at,false);g.q8_matvec_many(matrix,g.slice(input,uint64_t(at)*c*4,uint64_t(n)*c*4),g.slice(old,uint64_t(at)*r*4,uint64_t(n)*r*4),c,r,n);at+=n;}need(g.submit().dispatches==2,"old N3 control was not two physical calls");
    const auto expected=g.read(old);const auto hc=g.hc_exact_routes();const auto wr=g.wide_q8_many_routes();const auto prior=g.q8_three_rows_routes();
    const auto call=[&](GpuView m,GpuView xx,GpuView yy){g.q8_matvec_many(m,xx,yy,c,r,3);};
    g.begin();call(matrix,input,y);need(g.submit().dispatches==1,"N3 Graph did not use one physical dispatch");need(g.read(y)==expected,"N3 Graph differs from old2+1 output bits");
    need(g.memory().allocations==ledger.allocations&&g.memory().live_bytes==ledger.live_bytes,"N3 dispatch allocated GPU memory");
    need(g.hc_exact_routes()==hc&&g.wide_q8_many_routes()==wr,"N3 was miscounted in legacy HC/wide buckets");
    auto want=prior;const size_t bucket=packed?1:c==10240&&r==320?2:wide&&c==2560&&r==10240?3:wide&&c==6144&&r==2560?4:0;++want[bucket];need(g.q8_three_rows_routes()==want,"N3 completed physical route bucket differs");
    const auto fail=[&](const std::function<void()>&fn){g.begin();refuses(fn);g.abort();need(g.q8_three_rows_routes()==want,"refused encode published N3 routes");};
    g.begin();call(matrix,input,y);g.abort();need(g.q8_three_rows_routes()==want,"aborted encode published N3 routes");
    auto bad=x;bad.back()=INFINITY;g.write(input,std::as_bytes(std::span(bad)));g.begin();call(matrix,input,y);refuses([&]{static_cast<void>(g.submit());});need(g.q8_three_rows_routes()==want,"GPU failure published N3 routes");
    g.write(input,std::as_bytes(std::span(x)));g.begin();call(matrix,input,y);need(g.submit().dispatches==1&&g.read(y)==expected,"N3 valid reuse after failure differs");++want[bucket];
    fail([&]{call(matrix,input,g.slice(input,0,std::min(input.bytes,y.bytes)));});fail([&]{call(matrix,g.slice(input,0,input.bytes-4),y);});const auto immutable=a.bytes(g.read(y));fail([&]{call(matrix,input,immutable);});
    // Rejected non-owner access cannot clear the owner's successful pending route.
    g.begin();call(matrix,input,y);bool foreign=false;std::thread t([&]{try{g.abort();}catch(const FormatError&){foreign=true;}});t.join();need(foreign,"non-owner abort accepted");need(g.submit().dispatches==1,"owner pending N3 lost after foreign refusal");++want[bucket];need(g.q8_three_rows_routes()==want,"non-owner abort cleared pending N3 route");
    refuses([&]{g.begin_indirect_compatible();call(matrix,input,y);});try{g.abort();}catch(const FormatError&){}
    if(!packed){refuses([&]{g.begin_capture(8);call(matrix,input,y);});try{g.abort();}catch(const FormatError&){}}
    else{fail([&]{call(g.slice(matrix,0,matrix.bytes-4),input,y);});}
    need(g.read(matrix)==payload&&g.read(input)==std::vector<std::byte>(std::as_bytes(std::span(x)).begin(),std::as_bytes(std::span(x)).end()),"N3 modified weights/input");a.guards();++cases;
    std::cout<<"N3 Graph case K="<<c<<" R="<<r<<" packed="<<packed<<" wide="<<wide<<" concurrent="<<concurrent<<" exact_bits=true thermal_before="<<thermal_before<<" thermal_after="<<host_thermal_state()<<" pass\n";
}
void moe_case(const char* lib,uint32_t c,uint32_t h,bool packed,bool concurrent){
    const auto thermal_before=host_thermal_state();
    MetalGraph g(lib,128ull<<20);g.configure_q8_three_rows(true);g.configure_concurrent_dispatch(concurrent);g.configure_small_gemv(16);g.configure_block_gemv(3);g.prepare_q8_matvec_many_all();
    Arena a{g,{}};const MoeShape shape{c,h,16,10};std::vector<std::pair<GpuView,std::vector<std::byte>>> held;
    const auto quant=[&](uint32_t type,uint32_t k,uint32_t r,uint32_t e,bool lane){auto raw=grouped_fixture::weights({type,k,r,e,10,3,true});auto bytes=raw;if(lane)lanes::pack_rows(lanes::row_layout(type,k),raw,bytes);auto v=a.bytes(bytes);if(lane)g.register_lane_matrix(v,{type,k,r,e});held.push_back({v,std::move(bytes)});return GpuMatrix{v,type};};
    const auto dense=[&](uint32_t r){std::vector<float> f(uint64_t(c)*r);for(size_t i=0;i<f.size();++i)f[i]=float(int((i*131+44)%1993)-996)/99700;return GpuMatrix{a.floats(f),0};};
    const auto router=dense(16),sr=dense(1);const auto gate=quant(12,c,h,16,false),up=quant(12,c,h,16,false),down=quant(7,h,c,16,packed);
    const auto sg=quant(8,c,h,1,packed),su=quant(8,c,h,1,packed),sd=quant(8,h,c,1,false);const MoeWeights w{router,gate,up,down,sg,su,sd,sr};
    const std::array<uint64_t,10> count{16,10,10,uint64_t(h)*10,uint64_t(h)*10,uint64_t(c)*10,h,h,c,1};
    const auto scratch=[&](){std::array<GpuView,10> v{};for(size_t i=0;i<v.size();++i)v[i]=a.floats(std::vector<float>(count[i]*8,-7654.25f),false);return MoeScratch{v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9]};};
    const auto s=scratch(),old=scratch();std::vector<float> values(uint64_t(c)*8,-9987.25f);for(uint32_t t=0;t<3;++t)for(uint32_t j=0;j<c;++j)values[uint64_t(t)*c+j]=float(int((uint64_t(t)*c+j)*97%1993)-996)/997;
    const auto input=a.floats(values,false),output=a.floats(std::vector<float>(uint64_t(c)*8,-7654.25f),false),expected=a.floats(std::vector<float>(uint64_t(c)*8,-7654.25f),false);
    const auto token=[&](const MoeScratch& ss,uint32_t t){auto v=ss.views();for(size_t i=0;i<v.size();++i)v[i]=g.slice(v[i],uint64_t(t)*count[i]*4,count[i]*4);return MoeScratch{v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9]};};
    g.configure_block_gemv(0);g.begin();refuses([&]{encode_moe_q8_three_rows(g,w,s,input,output,shape,RouterMode::softmax,true,8);});try{g.abort();}catch(const FormatError&){}g.configure_block_gemv(3);
    g.begin();for(uint32_t t=0;t<3;++t)encode_moe(g,w,token(old,t),g.slice(input,uint64_t(t)*c*4,uint64_t(c)*4),g.slice(expected,uint64_t(t)*c*4,uint64_t(c)*4),shape,RouterMode::softmax,true);const auto ordinary=g.submit();
    const auto ledger=g.memory();const auto prior=g.q8_three_rows_routes();g.begin();encode_moe_q8_three_rows(g,w,s,input,output,shape,RouterMode::softmax,true,8);const auto batch=g.submit();
    need(batch.dispatches+moe_q8_three_rows_saved_dispatches(g,w,shape)==ordinary.dispatches,"N3 MoE physical count differs");need(g.memory().allocations==ledger.allocations&&g.memory().live_bytes==ledger.live_bytes,"N3 MoE encoded an allocation");
    need(g.read(output)==g.read(expected),"N3 shared MoE differs from ordinary output bits or inactive rows");const auto actual=s.views(),control=old.views();for(size_t i=0;i<actual.size();++i)need(g.read(actual[i])==g.read(control[i]),"N3 MoE router/intermediate bits differ from ordinary");
    auto routes=prior;routes[0]+=packed?1:3;routes[1]+=packed?2:0;need(g.q8_three_rows_routes()==routes,"N3 shared MoE did not perform exactly3 shared projections");
    const auto good=g.read(output);g.begin();refuses([&]{encode_moe_q8_three_rows(g,w,s,input,input,shape,RouterMode::softmax,true,8);});try{g.abort();}catch(const FormatError&){}need(g.read(output)==good,"N3 MoE alias refusal changed output");
    auto bad=values;bad[uint64_t(3)*c-1]=INFINITY;g.write(input,std::as_bytes(std::span(bad)));g.begin();encode_moe_q8_three_rows(g,w,s,input,output,shape,RouterMode::softmax,true,8);refuses([&]{static_cast<void>(g.submit());});need(g.q8_three_rows_routes()==routes,"failed N3 MoE published routes");
    g.write(input,std::as_bytes(std::span(values)));g.begin();encode_moe_q8_three_rows(g,w,s,input,output,shape,RouterMode::softmax,true,8);static_cast<void>(g.submit());need(g.read(output)==good,"N3 MoE valid reuse after failure differs");
    for(const auto& item:held)need(g.read(item.first)==item.second,"N3 MoE changed an immutable weight");a.guards();++cases;
    std::cout<<"N3 MoE case C="<<c<<" H="<<h<<" E=16 S=10 packed="<<packed<<" concurrent="<<concurrent<<" exact_router_intermediate_output_bits=true thermal_before="<<thermal_before<<" thermal_after="<<host_thermal_state()<<" pass\n";
}
}
int main(int argc,char**argv){try{need(argc==3&&std::string(argv[1])=="--execute-generated","usage: test-q8_three_rows_graph --execute-generated OWNED_METALLIB; root GPU owner only");
    {MetalGraph g(argv[2],16ull<<20);need(!g.q8_three_rows_enabled()&&g.q8_many_width(3)==2,"N3 enabled by default");refuses([&]{g.prepare_q8_matvec_many(3);});g.prepare_q8_matvec_many(1);refuses([&]{g.configure_q8_three_rows(true);});}
    for(bool concurrent:{false,true})for(auto shape:std::array<std::array<uint32_t,2>,7>{{{32,7},{320,11},{640,2560},{2560,640},{2560,10240},{6144,2560},{10240,320}}}){
        exercise(argv[2],shape[0],shape[1],false,false,concurrent);if(shape[0]%256==0||shape[0]<512)exercise(argv[2],shape[0],shape[1],true,false,concurrent);
        if((shape[0]==2560&&shape[1]==10240)||(shape[0]==6144&&shape[1]==2560))exercise(argv[2],shape[0],shape[1],false,true,concurrent);
    }
    for(bool concurrent:{false,true}){moe_case(argv[2],256,64,false,concurrent);moe_case(argv[2],2560,640,false,concurrent);moe_case(argv[2],2560,640,true,concurrent);}
    std::cout<<"PASS N3 Graph cases="<<cases<<" checks="<<checks<<" exact_control_bits=true padding_rows=0; no model/speed admission\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
