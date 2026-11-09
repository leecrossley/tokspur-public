#include <metal_stdlib>
using namespace metal;
struct Q8ThreeArgs {
    uint columns,rows,row_bytes,reserved;
    ulong input_stride_bytes,output_stride_bytes;
};
static_assert(sizeof(Q8ThreeArgs)==32,"three-row Q8 ABI");
// Same FP32 expression, ascending block order and simd_sum as q8_many/expert_lanes. Each lane decodes its weight once and reuses it across exactly three inputs.
inline void three_block(uint block,float scale,uint byte,device const float* x0,device const float* x1,device const float* x2,
    ushort lane,thread float (&sums)[3]){
    const float weight=scale*float(as_type<char>(uchar(byte)));const uint column=block*32+lane;
    sums[0]+=weight*x0[column];sums[1]+=weight*x1[column];sums[2]+=weight*x2[column];
}
template<bool Packed>
void three_rows(device const uchar* weights,device const float* input,device float* output,constant Q8ThreeArgs& a,
    device atomic_uint* error,uint group,ushort subgroup,ushort lane){
    const uint row=group*4+subgroup;
    if(!a.columns || a.columns>65536 || a.columns%32 || !a.rows || a.rows>1000000 || a.row_bytes!=a.columns/32*34 || a.reserved ||
        a.input_stride_bytes<ulong(a.columns)*4 || a.output_stride_bytes<ulong(a.rows)*4 || a.input_stride_bytes%4 || a.output_stride_bytes%4 ||
        (Packed && a.columns>=512 && a.columns%256)){
        if(!lane)atomic_fetch_or_explicit(error,4u,memory_order_relaxed);return;
    }
    if(row>=a.rows)return;
    device const uchar* data=weights+ulong(row)*a.row_bytes;
    device const float* x0=input;
    device const float* x1=(device const float*)((device const uchar*)input+a.input_stride_bytes);
    device const float* x2=(device const float*)((device const uchar*)input+2*a.input_stride_bytes);
    float sums[3]={0,0,0};const uint blocks=a.columns/32;
    if(!Packed){
        for(uint block=0;block<blocks;++block){device const uchar* q=data+ulong(block)*34;
            const float scale=float(*reinterpret_cast<device const half*>(q));
            three_block(block,scale,uint(q[2+lane]),x0,x1,x2,lane,sums);
        }
    }else{
        const uint full=blocks/16,rest=blocks%16;device const half* scales=(device const half*)data;device const uchar* quants=data+blocks*2;
        for(uint run=0;run<full;++run){const uint4 words=*reinterpret_cast<device const uint4*>(quants+run*512+lane*16);
            for(uint part=0;part<4;++part)for(uint k=0;k<4;++k){const uint block=run*16+part*4+k;
                three_block(block,float(scales[block]),(words[part]>>(8*k))&255,x0,x1,x2,lane,sums);
            }
        }
        device const uchar* tail=quants+full*512+lane*rest;
        for(uint k=0;k<rest;++k){const uint block=full*16+k;three_block(block,float(scales[block]),uint(tail[k]),x0,x1,x2,lane,sums);}
    }
    for(ushort i=0;i<3;++i){const float total=simd_sum(sums[i]);if(!lane){
        if(!isfinite(total))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        device float* y=(device float*)((device uchar*)output+ulong(i)*a.output_stride_bytes);y[row]=total;
    }}
}
#define THREE_KERNEL(NAME,PACKED) \
kernel void NAME(device const uchar* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]], \
    constant Q8ThreeArgs& a [[buffer(3)]],device atomic_uint* error [[buffer(7)]],uint group [[threadgroup_position_in_grid]], \
    ushort subgroup [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]){three_rows<PACKED>(w,x,y,a,error,group,subgroup,lane);}
THREE_KERNEL(graph_q8_three_rows,false)
THREE_KERNEL(graph_lanes_q8_three_rows,true)
