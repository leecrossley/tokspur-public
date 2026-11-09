// Bounded tile adaptation of llama.cpp 6050b467. Tokspur retains routing and graph ownership.
// Copyright (c) 2023-2026 The ggml authors. MIT notice: ../third_party/llama-cpp/LICENSE.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant uint grouped_f32_encoding [[function_constant(0)]];
constant uint grouped_f32_layout [[function_constant(1)]];
struct GroupedMmF32Args {
    uint columns, rows, row_bytes, experts;
    uint selected, tokens, broadcast, maximum_tasks;
    ulong expert_stride_bytes, input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(GroupedMmF32Args) == 56, "grouped FP32 argument ABI");
inline uint grouped_f32_dword(device const uchar* p, uint offset) {
    return uint(p[offset]) | (uint(p[offset+1])<<8) | (uint(p[offset+2])<<16) | (uint(p[offset+3])<<24);
}
inline float grouped_f32_half(device const uchar* p, uint offset) {
    return float(as_type<half>(ushort(uint(p[offset]) | (uint(p[offset+1])<<8))));
}
// Decode a represented FP32 weight before multiplying input, as in graph_grouped_serial.
inline float grouped_f32_element(device const uchar* row, uint column) {
    if (grouped_f32_layout == 2 && grouped_f32_encoding == 7) {
        constexpr uint blocks = 20;
        const uint b = column / 32;
        const float d = grouped_f32_half(row + 8 * b, 0);
        const float m = grouped_f32_half(row + 8 * b, 2);
        const uint qh = grouped_f32_dword(row + 8 * b, 4);
        const uint run = b / 16;
        const uint width = min(16u, blocks - run * 16u);
        const uint qi = column % 32;
        const ulong at = 8 * blocks + ulong(run) * 256 + (qi % 16) * width + b % 16;
        const uint q = ((uint(row[at]) >> (4 * (qi / 16))) & 15u) | (((qh >> qi) & 1u) << 4);
        return d * float(q) + m;
    }
    if (grouped_f32_encoding == 8) {
        device const uchar* block = row + (column/32)*34;
        return grouped_f32_half(block,0)*float(as_type<char>(block[2+column%32]));
    }
    if (grouped_f32_encoding == 7) {
        device const uchar* block = row + (column/32)*24;
        const uint i = column%32;
        const uint low = (uint(block[8+i%16])>>(4*(i/16)))&15;
        const uint q = low | (((grouped_f32_dword(block,4)>>i)&1)<<4);
        return grouped_f32_half(block,0)*float(q)+grouped_f32_half(block,2);
    }
    device const uchar* block = row + (column/256)*(grouped_f32_encoding == 12 ? 144 : 176);
    const uint i = column%256, group = i/32;
    uint scale, minimum;
    if (group < 4) { scale = uint(block[4+group])&63; minimum = uint(block[8+group])&63; }
    else {
        const uint slot = group-4;
        scale = (uint(block[12+slot])&15) | ((uint(block[4+slot])>>6)<<4);
        minimum = (uint(block[12+slot])>>4) | ((uint(block[8+slot])>>6)<<4);
    }
    const uint payload = grouped_f32_encoding == 12 ? 16 : 48;
    uint q = (uint(block[payload+(i/64)*32+i%32])>>(4*((i%64)/32)))&15;
    if (grouped_f32_encoding == 13) q |= ((uint(block[16+i%32])>>(i/32))&1)<<4;
    return grouped_f32_half(block,0)*float(scale)*float(q)-grouped_f32_half(block,2)*float(minimum);
}
// Four packed byte vectors have byte alignment, including Q8_0 payloads at block+2 and rows with a 34-byte stride.
inline uchar4 grouped_f32_bytes4(device const uchar* bytes) {
    return uchar4(*reinterpret_cast<device const packed_uchar4*>(bytes));
}
inline float4x4 grouped_f32_decode16(device const uchar* row, uint first) {
    // Callers provide a run of16 aligned elements inside a quant group of32. Coefficients are decoded once, with the scalar reference's FP32 operation order and contraction disabled.
    float4x4 decoded;
    if (grouped_f32_layout == 2 && grouped_f32_encoding == 7) {
        constexpr uint blocks = 20;
        const uint b = first / 32;
        const float d = grouped_f32_half(row + 8 * b, 0);
        const float m = grouped_f32_half(row + 8 * b, 2);
        const uint qh = grouped_f32_dword(row + 8 * b, 4);
        const uint run = b / 16;
        const uint width = min(16u, blocks - run * 16u);
        for (uint i = 0; i < 16; ++i) {
            const uint qi = (first + i) % 32;
            const ulong at = 8 * blocks + ulong(run) * 256 + (qi % 16) * width + b % 16;
            const uint q = ((uint(row[at]) >> (4 * (qi / 16))) & 15u) | (((qh >> qi) & 1u) << 4);
            decoded[i / 4][i % 4] = d * float(q) + m;
        }
        return decoded;
    }
    if (grouped_f32_encoding == 8) {
        device const uchar* block = row + (first/32)*34;
        const float d = grouped_f32_half(block,0);
        for (uint j = 0; j < 4; ++j) {
            const char4 q = char4(*reinterpret_cast<device const packed_char4*>(block+2+first%32+j*4));
            decoded[j] = d*float4(q);
        }
        return decoded;
    }
    if (grouped_f32_encoding == 7) {
        device const uchar* block = row + (first/32)*24;
        const uint at = first%32, shift = 4*(at/16), high = grouped_f32_dword(block,4);
        const float d = grouped_f32_half(block,0), m = grouped_f32_half(block,2);
        for (uint j = 0; j < 4; ++j) {
            const uint4 low = (uint4(grouped_f32_bytes4(block+8+j*4))>>shift)&15u;
            const uint4 bit = (uint4(high)>>uint4(at+j*4,at+j*4+1,at+j*4+2,at+j*4+3))&1u;
            decoded[j] = d*float4(low|(bit<<4))+m;
        }
        return decoded;
    }
    device const uchar* block = row + (first/256)*(grouped_f32_encoding == 12 ? 144 : 176);
    const uint at = first%256, group = at/32;
    uint scale, minimum;
    if (group < 4) { scale = uint(block[4+group])&63; minimum = uint(block[8+group])&63; }
    else {
        const uint slot = group-4;
        scale = (uint(block[12+slot])&15) | ((uint(block[4+slot])>>6)<<4);
        minimum = (uint(block[12+slot])>>4) | ((uint(block[8+slot])>>6)<<4);
    }
    const float scaled = grouped_f32_half(block,0)*float(scale);
    const float minimum_product = grouped_f32_half(block,2)*float(minimum);
    const uint payload = grouped_f32_encoding == 12 ? 16 : 48;
    device const uchar* low = block+payload+(at/64)*32+at%32;
    for (uint j = 0; j < 4; ++j) {
        uint4 q = (uint4(grouped_f32_bytes4(low+j*4))>>(4*((at%64)/32)))&15u;
        if (grouped_f32_encoding == 13) q |= ((uint4(grouped_f32_bytes4(block+16+at%32+j*4))>>group)&1u)<<4;
        decoded[j] = scaled*float4(q)-minimum_product;
    }
    return decoded;
}
inline uint grouped_f32_route(device const uint* ids, device const uint* map, uint expert, uint at,
    constant GroupedMmF32Args& a, device atomic_uint* error) {
    const uint rid = map[ulong(expert)*a.tokens+at];
    if (rid >= a.tokens*a.selected || ids[rid] != expert) {
        atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return UINT_MAX;
    }
    return rid;
}
inline uint2 grouped_f32_swizzle_2d(uint2 group, uint num_gx, uint num_gy, uint swizzle) {
    if (swizzle <= 1 || num_gx <= 1 || num_gy <= 1) return group;
    const uint linear = group.y * num_gx + group.x;
    const uint band_tiles = num_gx * swizzle;
    const uint band_idx = linear / band_tiles;
    const uint first_y = band_idx * swizzle;
    const uint band_h = min(swizzle, num_gy - first_y);
    const uint in_band = linear - band_idx * band_tiles;
    return uint2(in_band / band_h, first_y + (in_band % band_h));
}
kernel void graph_grouped_mm_f32(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant GroupedMmF32Args& a [[buffer(3)]], device const uint* ids [[buffer(4)]],
    device const uint* map [[buffer(6)]], device atomic_uint* error [[buffer(7)]], device const uint* tasks [[buffer(8)]],
    device const uint* counts [[buffer(9)]], threadgroup float* shared [[threadgroup(0)]],
    uint2 raw_group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    if ((grouped_f32_layout != 0 && (grouped_f32_layout != 2 || grouped_f32_encoding != 7)) || (grouped_f32_encoding != 7 && grouped_f32_encoding != 8 && grouped_f32_encoding != 12 && grouped_f32_encoding != 13)) {
        if (!tid) atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
    }
    if (tasks[0] > a.maximum_tasks || tasks[1] != a.tokens*a.selected) {
        if (!tid) atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
    }
    if (raw_group.y >= tasks[0]) return;
    constexpr uint NK = 32, NR0 = 64, HALF_ROWS = 16;
    const uint num_gx = (a.rows + NR0 - 1) / NR0;
    if (raw_group.x >= num_gx) return;
    const uint2 group = grouped_f32_swizzle_2d(raw_group, num_gx, tasks[0], (a.broadcast & 2u) ? 8u : 0u);
    const uint expert = tasks[2+2*group.y], start = tasks[3+2*group.y];
    if (expert >= a.experts || start >= a.tokens || counts[expert] > a.tokens || start >= counts[expert]) {
        if (!tid) atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
    }
    const uint n = min(32u,counts[expert]-start), first = group.x*NR0;
    if (first >= a.rows) return;
    const uint nr0 = min(NR0,a.rows-first), ar = min(uint(tid/2),nr0-1), ai = tid%2;
    const uint br = min(uint(tid/4),n-1), bi = 8*(tid%4);
    const uint rid = grouped_f32_route(ids,map,expert,start+br,a,error);
    const uint input_row = rid == UINT_MAX ? 0 : ((a.broadcast & 1u) ? rid/a.selected : rid);
    device const uchar* w = weights+ulong(expert)*a.expert_stride_bytes+ulong(first+ar)*a.row_bytes;
    device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input)+ulong(input_row)*a.input_stride_bytes);
    threadgroup float* sa = shared;
    threadgroup float* sb = shared+NR0*NK;
    auto tA = tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(sa,dextents<int32_t,2>(NK,NR0));
    auto tB0 = tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(sb,dextents<int32_t,2>(NK,HALF_ROWS));
    auto tB1 = tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(sb+HALF_ROWS*NK,dextents<int32_t,2>(NK,HALF_ROWS));
    static_assert(is_same_v<typename decltype(tA)::value_type,float> && is_same_v<typename decltype(tB0)::value_type,float>, "full FP32 operands");
    constexpr auto descriptor = matmul2d_descriptor(HALF_ROWS,NR0,NK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "full FP32 TensorOps precision");
    matmul2d<descriptor,execution_simdgroups<4>> op;
    auto c0 = op.template get_destination_cooperative_tensor<decltype(tA),decltype(tB0),float>();
    auto c1 = op.template get_destination_cooperative_tensor<decltype(tA),decltype(tB1),float>();
    for (auto i = c0.begin(); i != c0.end(); ++i) *i = 0;
    for (auto i = c1.begin(); i != c1.end(); ++i) *i = 0;
    const bool has_hi = n > HALF_ROWS;
    for (uint k = 0; k < a.columns; k += NK) {
        const float4x4 decoded = grouped_f32_decode16(w,k+ai*16);
        for (uint i = 0; i < 16; ++i) {
            const float value = decoded[i/4][i%4];
            if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            sa[(tid/2)*NK+ai*16+i] = value;
        }
        for (uint i = 0; i < 8; ++i) {
            const float value = rid == UINT_MAX ? 0 : x[k+bi+i];
            if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            sb[(tid/4)*NK+bi+i] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto left = tA.slice(0,0), low = tB0.slice(0,0);
        op.run(low,left,c0);
        if (has_hi) { auto high = tB1.slice(0,0); op.run(high,left,c1); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Inputs are dead after the final barrier. The 8 KiB output aliases 12 KiB operand storage.
    auto out0 = tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(shared,dextents<int32_t,2>(NR0,HALF_ROWS));
    c0.store(out0);
    if (has_hi) {
        auto out1 = tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(shared+HALF_ROWS*NR0,dextents<int32_t,2>(NR0,HALF_ROWS));
        c1.store(out1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = sg; j < n; j += 4) {
        const uint dest = grouped_f32_route(ids,map,expert,start+j,a,error);
        if (dest == UINT_MAX) continue;
        device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output)+ulong(dest)*a.output_stride_bytes);
        for (uint row = lane; row < nr0; row += 32) {
            const float value = shared[j*NR0+row];
            if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            y[first+row] = value;
        }
    }
}
// Generated decoder check only. It is not exposed as a Graph or serving operation.
kernel void graph_grouped_mm_f32_decode(device const uchar* weights [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint4& shape [[buffer(2)]], device atomic_uint* error [[buffer(7)]], uint at [[thread_position_in_grid]]) {
    if (at >= shape.x*shape.y) return;
    if ((grouped_f32_layout != 0 && (grouped_f32_layout != 2 || grouped_f32_encoding != 7)) || (grouped_f32_encoding != 7 && grouped_f32_encoding != 8 && grouped_f32_encoding != 12 && grouped_f32_encoding != 13)) {
        atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
    }
    const float value = grouped_f32_element(weights+ulong(at/shape.x)*shape.z,at%shape.x);
    if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
    output[at] = value;
}
// Generated vector decoder check. The unchanged scalar kernel above remains its GPU byte reference.
kernel void graph_grouped_mm_f32_decode16(device const uchar* weights [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint4& shape [[buffer(2)]], device atomic_uint* error [[buffer(7)]], uint at [[thread_position_in_grid]]) {
    const uint runs = shape.x/16;
    if (at >= runs*shape.y) return;
    if (shape.x%16 || (grouped_f32_layout != 0 && (grouped_f32_layout != 2 || grouped_f32_encoding != 7)) || (grouped_f32_encoding != 7 && grouped_f32_encoding != 8 && grouped_f32_encoding != 12 && grouped_f32_encoding != 13)) {
        atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
    }
    const uint row = at/runs, first = (at%runs)*16;
    const float4x4 values = grouped_f32_decode16(weights+ulong(row)*shape.z,first);
    for (uint j = 0; j < 4; ++j) for (uint k = 0; k < 4; ++k) {
        const float value = values[j][k];
        if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        output[ulong(row)*shape.x+first+j*4+k] = value;
    }
}

#if defined(TOKSPUR_GROUPED_F32_COMPACT) && TOKSPUR_GROUPED_F32_COMPACT
// Component experiment retaining the credited FP32 decoder and TensorOps arithmetic. Each task uses two route panels and 32 output rows per group, with 6 KiB of staged operands.
kernel void graph_grouped_mm_f32_compact(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant GroupedMmF32Args& a [[buffer(3)]], device const uint* ids [[buffer(4)]],
    device const uint* map [[buffer(6)]], device atomic_uint* error [[buffer(7)]], device const uint* tasks [[buffer(8)]],
    device const uint* counts [[buffer(9)]], threadgroup float* shared [[threadgroup(0)]],
    uint3 raw_group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    if ((grouped_f32_layout != 0 && (grouped_f32_layout != 2 || grouped_f32_encoding != 7)) ||
        (grouped_f32_encoding != 7 && grouped_f32_encoding != 8 && grouped_f32_encoding != 12 && grouped_f32_encoding != 13) ||
        tasks[0] > a.maximum_tasks || tasks[1] != a.tokens*a.selected) {
        if (!tid) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
        return;
    }
    constexpr uint NK = 32, WEIGHT_ROWS = 32, ROUTES = 16;
    const uint groups_x = (a.rows + WEIGHT_ROWS - 1) / WEIGHT_ROWS;
    if (raw_group.x >= groups_x || raw_group.y >= tasks[0] || raw_group.z >= 2) return;
    const uint2 group = grouped_f32_swizzle_2d(raw_group.xy,groups_x,tasks[0],(a.broadcast & 2u) ? 8u : 0u);
    const uint expert = tasks[2+2*group.y], task_start = tasks[3+2*group.y];
    if (expert >= a.experts || task_start >= a.tokens || counts[expert] > a.tokens || task_start >= counts[expert]) {
        if (!tid) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
        return;
    }
    const uint task_routes = min(32u,counts[expert]-task_start), panel = raw_group.z*ROUTES;
    if (panel >= task_routes) return;
    const uint start = task_start+panel, n = min(ROUTES,task_routes-panel);
    const uint first = group.x*WEIGHT_ROWS, nr = min(WEIGHT_ROWS,a.rows-first);
    const uint rid = grouped_f32_route(ids,map,expert,start+min(uint(tid/8),n-1),a,error);
    const uint input_row = rid == UINT_MAX ? 0 : ((a.broadcast & 1u) ? rid/a.selected : rid);
    device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input)+ulong(input_row)*a.input_stride_bytes);
    threadgroup float* inputs = shared+WEIGHT_ROWS*NK;
    auto weights_tile = tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(shared,dextents<int32_t,2>(NK,WEIGHT_ROWS));
    auto inputs_tile = tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(inputs,dextents<int32_t,2>(NK,ROUTES));
    constexpr auto descriptor = matmul2d_descriptor(ROUTES,WEIGHT_ROWS,NK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "compact grouped FP32 precision");
    matmul2d<descriptor,execution_simdgroups<4>> op;
    auto sums = op.template get_destination_cooperative_tensor<decltype(weights_tile),decltype(inputs_tile),float>();
    for (auto i = sums.begin(); i != sums.end(); ++i) *i = 0;
    for (uint k = 0; k < a.columns; k += NK) {
        if (tid < 64) {
            const uint row = min(uint(tid/2),nr-1), column = (tid%2)*16;
            device const uchar* w = weights+ulong(expert)*a.expert_stride_bytes+ulong(first+row)*a.row_bytes;
            const float4x4 decoded = grouped_f32_decode16(w,k+column);
            for (uint i = 0; i < 16; ++i) {
                const float value = decoded[i/4][i%4];
                if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
                shared[(tid/2)*NK+column+i] = value;
            }
        }
        for (uint i = 0; i < 4; ++i) {
            const float value = rid == UINT_MAX ? 0 : x[k+(tid%8)*4+i];
            if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            inputs[(tid/8)*NK+(tid%8)*4+i] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto w = weights_tile.slice(0,0), x_tile = inputs_tile.slice(0,0);
        op.run(x_tile,w,sums);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto output_tile = tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(shared,dextents<int32_t,2>(WEIGHT_ROWS,ROUTES));
    sums.store(output_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint at = tid; at < n*WEIGHT_ROWS; at += 128) {
        const uint row = at%WEIGHT_ROWS;
        if (row >= nr) continue;
        const uint dest = grouped_f32_route(ids,map,expert,start+at/WEIGHT_ROWS,a,error);
        if (dest == UINT_MAX) continue;
        device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output)+ulong(dest)*a.output_stride_bytes);
        const float value = shared[at];
        if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        y[first+row] = value;
    }
}
#endif

#if defined(TOKSPUR_GROUPED_F32_COMPENSATED) && TOKSPUR_GROUPED_F32_COMPENSATED
#include "../experiments/fp32_compensated.h"

// Separate arithmetic experiment on the retained 64-row task grid. Each SIMD group compensates products and sums for one output at a time, using the unchanged scalar represented-weight decoder.
kernel void graph_grouped_mm_f32_compensated(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant GroupedMmF32Args& a [[buffer(3)]], device const uint* ids [[buffer(4)]],
    device const uint* map [[buffer(6)]], device atomic_uint* error [[buffer(7)]], device const uint* tasks [[buffer(8)]],
    device const uint* counts [[buffer(9)]], uint2 raw_group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    if ((grouped_f32_layout != 0 && (grouped_f32_layout != 2 || grouped_f32_encoding != 7)) ||
        (grouped_f32_encoding != 7 && grouped_f32_encoding != 8 && grouped_f32_encoding != 12 && grouped_f32_encoding != 13) ||
        tasks[0] > a.maximum_tasks || tasks[1] != a.tokens*a.selected) {
        if (!sg && !lane) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
        return;
    }
    const uint groups_x = (a.rows+63)/64;
    if (raw_group.x >= groups_x || raw_group.y >= tasks[0]) return;
    const uint2 group = grouped_f32_swizzle_2d(raw_group,groups_x,tasks[0],(a.broadcast & 2u) ? 8u : 0u);
    const uint expert = tasks[2+2*group.y], start = tasks[3+2*group.y];
    if (expert >= a.experts || start >= a.tokens || counts[expert] > a.tokens || start >= counts[expert]) {
        if (!sg && !lane) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
        return;
    }
    const uint n = min(32u,counts[expert]-start), first = group.x*64;
    for (uint row = first+sg; row < min(first+64,a.rows); row += 4) {
        device const uchar* w = weights+ulong(expert)*a.expert_stride_bytes+ulong(row)*a.row_bytes;
        for (uint j = 0; j < n; ++j) {
            const uint rid = grouped_f32_route(ids,map,expert,start+j,a,error);
            if (rid == UINT_MAX) continue;
            const uint input_row = (a.broadcast & 1u) ? rid/a.selected : rid;
            device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input)+ulong(input_row)*a.input_stride_bytes);
            tokspur_fp32::Sum sum;
            for (uint column = lane; column < a.columns; column += 32)
                tokspur_fp32::product(sum,grouped_f32_element(w,column),x[column]);
            const float value = tokspur_fp32::reduce(sum,lane);
            if (!lane) {
                if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
                device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output)+ulong(rid)*a.output_stride_bytes);
                y[row] = value;
            }
        }
    }
}
#endif

#if defined(TOKSPUR_GROUPED_F32_REUSE) && TOKSPUR_GROUPED_F32_REUSE
#include "../experiments/fp32_compensated.h"

// A partial panel instantiates only its active routes. Each output keeps the scalar compensated component's column order, product residuals and shuffle reduction.
template<uint ROUTES>
inline void grouped_f32_reuse_panel(device const uchar* weights, device const float* input, device float* output,
    constant GroupedMmF32Args& a, device const uint* ids, device const uint* map, device atomic_uint* error,
    uint expert, uint first, uint start, ushort lane) {
    constexpr uint ROWS = 2;
    uint routes[ROUTES];
    ulong input_rows[ROUTES];
    #pragma unroll
    for (uint j = 0; j < ROUTES; ++j) {
        routes[j] = grouped_f32_route(ids,map,expert,start+j,a,error);
        input_rows[j] = routes[j] == UINT_MAX ? 0 : ((a.broadcast & 1u) ? routes[j]/a.selected : routes[j]);
    }
    tokspur_fp32::Sum sums[ROWS][ROUTES];
    for (uint column = lane; column < a.columns; column += 32) {
        float x[ROUTES];
        #pragma unroll
        for (uint j = 0; j < ROUTES; ++j) {
            device const float* row = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input)+input_rows[j]*a.input_stride_bytes);
            x[j] = routes[j] == UINT_MAX ? 0 : row[column];
        }
        #pragma unroll
        for (uint i = 0; i < ROWS; ++i) {
            device const uchar* row = weights+ulong(expert)*a.expert_stride_bytes+ulong(min(first+i,a.rows-1))*a.row_bytes;
            const float w = grouped_f32_element(row,column);
            #pragma unroll
            for (uint j = 0; j < ROUTES; ++j) tokspur_fp32::product(sums[i][j],w,x[j]);
        }
    }
    #pragma unroll
    for (uint i = 0; i < ROWS; ++i) {
        #pragma unroll
        for (uint j = 0; j < ROUTES; ++j) {
            const float value = tokspur_fp32::reduce(sums[i][j],lane);
            if (!lane && first+i < a.rows && routes[j] != UINT_MAX) {
                if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
                device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output)+ulong(routes[j])*a.output_stride_bytes);
                y[first+i] = value;
            }
        }
    }
}

kernel void graph_grouped_mm_f32_reuse(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant GroupedMmF32Args& a [[buffer(3)]], device const uint* ids [[buffer(4)]],
    device const uint* map [[buffer(6)]], device atomic_uint* error [[buffer(7)]], device const uint* tasks [[buffer(8)]],
    device const uint* counts [[buffer(9)]], uint3 raw_group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    if ((grouped_f32_layout != 0 && (grouped_f32_layout != 2 || grouped_f32_encoding != 7)) ||
        (grouped_f32_encoding != 7 && grouped_f32_encoding != 8 && grouped_f32_encoding != 12 && grouped_f32_encoding != 13) ||
        tasks[0] > a.maximum_tasks || tasks[1] != a.tokens*a.selected) {
        if (!sg && !lane) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
        return;
    }
    constexpr uint ROWS = 2, ROUTES = 4, GROUP_ROWS = 4*ROWS;
    const uint groups_x = (a.rows+GROUP_ROWS-1)/GROUP_ROWS;
    if (raw_group.x >= groups_x || raw_group.y >= tasks[0] || raw_group.z >= 32/ROUTES) return;
    const uint2 group = grouped_f32_swizzle_2d(raw_group.xy,groups_x,tasks[0],(a.broadcast & 2u) ? 8u : 0u);
    const uint expert = tasks[2+2*group.y], start = tasks[3+2*group.y];
    if (expert >= a.experts || start >= a.tokens || counts[expert] > a.tokens || start >= counts[expert]) {
        if (!sg && !lane) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
        return;
    }
    const uint first = group.x*GROUP_ROWS+sg*ROWS, panel = raw_group.z*ROUTES;
    const uint n = min(32u,counts[expert]-start);
    if (first >= a.rows || panel >= n) return;
    switch (min(ROUTES,n-panel)) {
        case 1: grouped_f32_reuse_panel<1>(weights,input,output,a,ids,map,error,expert,first,start+panel,lane); break;
        case 2: grouped_f32_reuse_panel<2>(weights,input,output,a,ids,map,error,expert,first,start+panel,lane); break;
        case 3: grouped_f32_reuse_panel<3>(weights,input,output,a,ids,map,error,expert,first,start+panel,lane); break;
        default: grouped_f32_reuse_panel<4>(weights,input,output,a,ids,map,error,expert,first,start+panel,lane); break;
    }
}
#endif
