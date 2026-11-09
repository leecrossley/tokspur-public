#include "tokspur/moe_mm_many.hpp"
#include "tokspur/gguf.hpp"
#include <iostream>
#include <limits>

using namespace tokspur;
namespace {
unsigned checks=0;
void need(bool value,const char* why){++checks;if(!value)throw FormatError(why);}
template<class F>void rejects(F f){bool refused=false;try{f();}catch(const FormatError&){refused=true;}need(refused,"invalid tiled MoE geometry admitted");}
}
int main(){try{
    const MoeShape actual{2560,640,512,10};const auto b=moe_mm_many_bytes(actual,512);
    need(b.views[0]==1048576 && b.views[1]==20480 && b.views[2]==20480,"runtime router/index shapes");
    need(b.views[3]==13107200 && b.views[4]==13107200 && b.views[5]==52428800,"runtime selected expert row shapes");
    need(b.views[6]==1310720 && b.views[7]==1310720 && b.views[8]==5242880 && b.views[9]==2048,"shared rows");
    need(b.views[10]==1048576 && b.views[11]==2048 && b.views[12]==5384,"route map/count/tasks");
    uint64_t sum=0;for(auto bytes:b.views)sum+=bytes;need(sum==b.total && b.total<96ull*1024*1024,"accounted reusable scratch budget");
    for(auto n:{1u,3u,8u,16u,17u,31u,32u,33u,511u,512u})for(auto selected:{1u,3u,10u,16u}){
        const auto v=moe_mm_many_bytes({2560,640,512,selected},n);
        need(v.views[5]==uint64_t(n)*selected*2560*4,"per-route down count");
        need(v.views[10]==uint64_t(n)*512*4 && v.views[11]==2048,"active map stride");
        need(v.views[12]==(2+2*(uint64_t(n)*selected/32+512))*4,"conservative tile task bound");
    }
    const auto b1024 = moe_mm_many_bytes(actual, 1024);
    need(b1024.total <= (300ull << 20), "1024 MoE total budget for production shape");
    rejects([&]{moe_mm_many_bytes(actual, 1025);});
    rejects([&]{moe_mm_many_bytes(actual, 2048);});
    for(auto n:{0u,1025u,2048u,2049u,std::numeric_limits<uint32_t>::max()})rejects([&]{moe_mm_many_bytes(actual,n);});
    const auto b2k = moe_mm_many_bytes({2560,640,512,5}, 2048); need(b2k.total <= (300ull << 20), "2K MoE total budget");
    rejects([&]{moe_mm_many_bytes({2560,640,512,5}, 2049);});
    rejects([&]{moe_mm_many_bytes({2560,640,512,6}, 2048);});
    for(auto s:{MoeShape{0,640,512,10},MoeShape{2561,640,512,10},MoeShape{2560,641,512,10},MoeShape{2560,640,0,10},MoeShape{2560,640,513,10},MoeShape{2560,640,512,17},MoeShape{2560,640,8,10}})rejects([&]{moe_mm_many_bytes(s,512);});
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    for(auto n:{1u,2u,3u,8u,31u,32u,33u,511u,512u}) {
        const auto p=moe_router_f32_plan(2560,512,n,10272,2080);
        need(p.matrix_bytes==5242880 && p.input_bytes==uint64_t(n-1)*10272+10240 && p.output_bytes==uint64_t(n-1)*2080+2048,"router exact last active byte with strided rows");
        need(p.grid_x==16 && p.grid_y==(n+31)/32 && p.args.tokens==n && p.args.reserved==0,"router tile tail/argument contract");
    }
    for(auto n:{0u,513u,std::numeric_limits<uint32_t>::max()})rejects([&]{moe_router_f32_plan(2560,512,n,10240,2048);});
    rejects([]{moe_router_f32_plan(2559,512,3,10240,2048);});
    rejects([]{moe_router_f32_plan(2560,511,3,10240,2048);});
    rejects([]{moe_router_f32_plan(2560,512,3,10236,2048);});
    rejects([]{moe_router_f32_plan(2560,512,3,10240,2044);});
    rejects([]{moe_router_f32_plan(2560,512,512,std::numeric_limits<uint64_t>::max()-15,2048);});
    rejects([]{moe_router_f32_plan(2560,512,512,10240,std::numeric_limits<uint64_t>::max()-15);});
    rejects([]{moe_router_f32_plan(2560,512,2,256ull<<20,2048);});
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    for(auto n:{1u,2u,3u,8u,16u,17u,31u,32u,33u,511u,512u}) {
        const auto p=moe_router_window_plan(2560,512,n,10272,2080);
        need(p.matrix_bytes==5242880 && p.input_bytes==uint64_t(n-1)*10272+10240 && p.output_bytes==uint64_t(n-1)*2080+2048,"router window exact last active byte with strided rows");
        need(p.grid_x==128 && p.grid_y==(n+15)/16 && p.args.tokens==n && p.args.reserved==0,"router window tile tail/argument contract");
    }
    for(auto n:{0u,513u,std::numeric_limits<uint32_t>::max()})rejects([&]{moe_router_window_plan(2560,512,n,10240,2048);});
    rejects([]{moe_router_window_plan(2559,512,3,10240,2048);});
    rejects([]{moe_router_window_plan(2560,511,3,10240,2048);});
    rejects([]{moe_router_window_plan(2560,512,3,10236,2048);});
    rejects([]{moe_router_window_plan(2560,512,3,10240,2044);});
    rejects([]{moe_router_window_plan(2560,512,512,std::numeric_limits<uint64_t>::max()-15,2048);});
    rejects([]{moe_router_window_plan(2560,512,512,10240,std::numeric_limits<uint64_t>::max()-15);});
    rejects([]{moe_router_window_plan(2560,512,2,256ull<<20,2048);});
#endif
    MoeMmManyScratch scratch{};uint64_t number=1;
    for(auto* view:{&scratch.rows.logits,&scratch.rows.indices,&scratch.rows.weights,&scratch.rows.gate,&scratch.rows.up,&scratch.rows.down,&scratch.rows.shared_gate,&scratch.rows.shared_up,&scratch.rows.shared_down,&scratch.rows.shared_logit,&scratch.token_map,&scratch.counts,&scratch.tasks})view->bytes=number++;
    number=1;for(auto v:scratch.views())need(v.bytes==number++,"release/validation view order");
    std::cout<<"PASS "<<checks<<" tiled MoE CPU checks, no Metal/model execution\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
