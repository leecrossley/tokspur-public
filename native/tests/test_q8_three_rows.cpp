#include "tokspur/q8_three_rows.hpp"
#include "tokspur/q8_many.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/expert_lanes.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <vector>

using namespace tokspur;
namespace {
unsigned checks=0;
void need(bool ok,const char* why){++checks;if(!ok)throw FormatError(why);}
template<class F>void refuses(F f){bool caught=false;try{f();}catch(const FormatError&){caught=true;}need(caught,"invalid three-row Q8 plan admitted");}
}
int main(){try{
    for(auto shape:std::array<std::array<uint32_t,2>,10>{{{2560,10240},{2560,6144},{2560,12288},{2560,512},{6144,2560},{2560,640},{640,2560},{320,10240},{10240,320},{2560,248320}}}){
        const auto c=shape[0],r=shape[1];const auto p=q8_three_rows_plan(Q8ThreeLayout::gguf,c,r,uint64_t(c)*4,uint64_t(r)*4);
        need(p.matrix_bytes==uint64_t(c/32)*34*r && p.input_bytes==uint64_t(c)*12 && p.output_bytes==uint64_t(r)*12,"actual dense family extents");
        need(p.groups==(r+3)/4 && p.threads==128 && p.args.reserved==0,"fixed three-row dispatch ABI");
    }
    const auto q8=q8_three_rows_plan(Q8ThreeLayout::gguf,640,2560,2564,10244);need(q8.args.row_bytes==680 && q8.input_bytes==7688 && q8.output_bytes==30728,"raw680-byte rows and4-byte FP32 stride alignment");
    need(q8_three_rows_plan(Q8ThreeLayout::gguf,32,3,128,12).matrix_bytes==102,"raw half-aligned odd row bytes");
    for(uint32_t c=32;c<=65536;c+=32)for(auto r:{1u,3u,4u,5u,31u,32u,33u}){
        const auto p=q8_three_rows_plan(Q8ThreeLayout::gguf,c,r,uint64_t(c)*4+4,uint64_t(r)*4+12);
        need(p.input_bytes==uint64_t(c)*12+8 && p.output_bytes==uint64_t(r)*12+24,"three active strided row extents");
        if(c<512 || c%256==0)need(q8_three_rows_plan(Q8ThreeLayout::q8_lanes_v1,c,r,uint64_t(c)*4,uint64_t(r)*4).matrix_bytes==p.matrix_bytes,"registered packed shape preserves byte extent");
        else refuses([&]{q8_three_rows_plan(Q8ThreeLayout::q8_lanes_v1,c,r,uint64_t(c)*4,uint64_t(r)*4);});
    }
    for(auto c:{0u,31u,65537u})refuses([&]{q8_three_rows_plan(Q8ThreeLayout::gguf,c,1,262148,4);});
    for(auto r:{0u,1000001u})refuses([&]{q8_three_rows_plan(Q8ThreeLayout::gguf,32,r,128,4000004);});
    refuses([]{q8_three_rows_plan(Q8ThreeLayout(2),32,1,128,4);});
    refuses([]{q8_three_rows_plan(Q8ThreeLayout::gguf,640,1,2556,4);});
    refuses([]{q8_three_rows_plan(Q8ThreeLayout::gguf,640,1,2562,4);});
    refuses([]{q8_three_rows_plan(Q8ThreeLayout::gguf,640,1,2560,std::numeric_limits<uint64_t>::max()-3);});
    refuses([]{q8_three_rows_plan(Q8ThreeLayout::gguf,65536,1000000,262144,4000000);});
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    const auto b3 = q8_many_bytes({640,2560,3});
    need(b3.matrix == 1740800 && b3.input == 7680 && b3.output == 30720, "three-row Q8 many width admitted");
#else
    refuses([]{q8_many_bytes({640,2560,3});});
#endif
    refuses([]{q8_many_bytes({640,2560,5});});
    for(auto c:{32u,64u,96u,320u,512u,768u,2560u,6144u,10240u}){
        const auto layout=lanes::row_layout(8,c);std::vector<std::byte> raw(layout.row_bytes*3),packed(raw.size()),restored(raw.size());
        for(size_t i=0;i<raw.size();++i)raw[i]=std::byte((i*29+17)%256);
        constexpr std::array<uint16_t,6> scales{0x0001,0x0401,0x2403,0x8000,0xa401,0x7bff};
        for(size_t b=0;b<raw.size()/34;++b){auto d=scales[b%scales.size()];raw[b*34]=std::byte(d&255);raw[b*34+1]=std::byte(d>>8);}
        lanes::pack_rows(layout,raw,packed);lanes::unpack_rows(layout,packed,restored);need(raw==restored,"existing lane byte inverse");
        for(uint32_t row=0;row<3;++row)for(uint32_t column=0;column<c;++column)need(lanes::decode_original(layout,raw.data()+row*layout.row_bytes,column)==lanes::decode_packed(layout,packed.data()+row*layout.row_bytes,column),"packed/raw represented FP32 value");
    }
    std::cout<<"PASS "<<checks<<" three-row Q8 CPU checks, no GPU/model execution\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
