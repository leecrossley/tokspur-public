// Raw-FP32 projection tile follows Tokspur qsa_prefill_bf16_project, adapted from llama.cpp 6050b467.
// Copyright (c) 2023-2026 The ggml authors. MIT notice: ../third_party/llama-cpp/LICENSE.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct MoeRouterF32Args {
    uint columns,experts,tokens,reserved;
    ulong input_stride_bytes,output_stride_bytes;
};
static_assert(sizeof(MoeRouterF32Args)==32,"router FP32 argument ABI");
kernel void graph_moe_router_f32_batch(device const float* weights [[buffer(0)]],device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],constant MoeRouterF32Args& p [[buffer(3)]],device atomic_uint* error [[buffer(7)]],
    uint2 block [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]]) {
    if(p.columns!=2560 || p.experts!=512 || !p.tokens || p.tokens>512 || p.reserved || p.input_stride_bytes<10240 || p.output_stride_bytes<2048 || p.input_stride_bytes%16 || p.output_stride_bytes%16) {
        if(!tid)atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
        return;
    }
    threadgroup float l[1024],r[1024];
    auto left=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(l,dextents<int32_t,2>(32,32));
    auto right=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(r,dextents<int32_t,2>(32,32));
    static_assert(is_same_v<typename decltype(left)::value_type,float> && is_same_v<typename decltype(right)::value_type,float>,"router operands remain FP32");
    constexpr auto descriptor=matmul2d_descriptor(32,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision,"router TensorOps full FP32 precision");
    matmul2d<descriptor,execution_simdgroups<4>> op;
    auto result=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(auto it=result.begin();it!=result.end();++it)*it=0;
    for(uint at=0;at<p.columns;at+=32) {
        for(uint i=tid;i<1024;i+=128) {
            const uint expert=block.x*32+i/32,channel=at+i%32;
            const float a=expert<p.experts?weights[ulong(expert)*p.columns+channel]:0;
            const uint token=block.y*32+i%32,k=at+i/32;
            const float b=token<p.tokens?input[ulong(token)*(p.input_stride_bytes/4)+k]:0;
            if(!isfinite(a)||!isfinite(b))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            l[i]=a;r[i]=b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,result);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(auto it=result.begin();it!=result.end();++it) {
        if(!result.is_valid_element(it))continue;
        const auto c=result.get_multidimensional_index(it);
        const uint expert=block.x*32+c[1],token=block.y*32+c[0];
        if(c[0]<32 && c[1]<32 && expert<p.experts && token<p.tokens) {
            const float value=*it;
            if(!isfinite(value))atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            output[ulong(token)*(p.output_stride_bytes/4)+expert]=value;
        }
    }
}
