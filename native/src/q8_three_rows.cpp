#include "tokspur/q8_three_rows.hpp"
#include "tokspur/gguf.hpp"

namespace tokspur {
namespace {
void need(bool x,const char* why){if(!x)throw FormatError(why);}
uint64_t extent(uint32_t width,uint64_t stride){
    const auto bytes=checked_mul(width,4);need(stride>=bytes && stride%4==0,"three-row FP32 stride");return checked_add(checked_mul(2,stride),bytes);
}
}
const char* q8_three_rows_kernel(Q8ThreeLayout layout){
    need(layout==Q8ThreeLayout::gguf || layout==Q8ThreeLayout::q8_lanes_v1,"three-row Q8 layout");
    return layout==Q8ThreeLayout::gguf?"graph_q8_three_rows":"graph_lanes_q8_three_rows";
}
const char* q8_three_rows_msg_kernel(Q8ThreeLayout layout){
    need(layout==Q8ThreeLayout::gguf || layout==Q8ThreeLayout::q8_lanes_v1,"three-row Q8 layout");
    return layout==Q8ThreeLayout::gguf?"graph_q8_three_rows_msg":"graph_lanes_q8_three_rows_msg";
}
Q8ThreePlan q8_three_rows_plan(Q8ThreeLayout layout,uint32_t columns,uint32_t rows,uint64_t xs,uint64_t ys){
    const auto name=q8_three_rows_kernel(layout);
    need(columns && columns<=65536 && columns%32==0 && rows && rows<=1000000,"three-row Q8 shape");
    if(layout==Q8ThreeLayout::q8_lanes_v1)need(columns<512 || columns%256==0,"registered Q8 lane eligibility");
    const uint32_t row_bytes=columns/32*34;Q8ThreePlan p{{columns,rows,row_bytes,0,xs,ys},0,0,0,0,128,name};
    p.matrix_bytes=checked_mul(row_bytes,rows);p.input_bytes=extent(columns,xs);p.output_bytes=extent(rows,ys);p.groups=(rows+3)/4;
    need(p.matrix_bytes<=(2ull<<30) && p.input_bytes<=(256ull<<20) && p.output_bytes<=(256ull<<20),"three-row Q8 byte budget");
    return p;
}
}
