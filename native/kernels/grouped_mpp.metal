// Tokspur grouping/decoding with an attributed staged TensorOps adaptation.
// Quant decoding follows owned gemv.metal. See the staged block's MIT notice.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant uint grouped_encoding [[function_constant(0)]];

// Per expert, stable compaction over token order. Each token has at most one
// slot for an expert. No host counts/readback and no nondeterministic appends.
kernel void graph_grouped_routes(device const uint* indices [[buffer(0)]],
    device uint* token_map [[buffer(1)]], device uint* counts [[buffer(2)]],
    constant uint4& s [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint expert [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint totals[4]; uint accumulated = 0;
    for (uint base = 0; base < s.z; base += 128) {
        const uint token = base + tid; uint slot = UINT_MAX, matches = 0;
        if (token < s.z) for (uint i = 0; i < s.y; ++i) {
            const uint id = indices[token*s.y+i];
            if (id >= s.x) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
            if (id == expert) { slot = i; ++matches; }
        }
        if (matches > 1) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
        const uint found = matches == 1;
        const uint prefix = simd_prefix_exclusive_sum(found), total = simd_sum(found);
        if (!lane) totals[sg] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint before = 0, chunk = 0;
        for (uint i = 0; i < 4; ++i) { if (i < sg) before += totals[i]; chunk += totals[i]; }
        if (found) token_map[expert*s.z+accumulated+before+prefix] = token*s.y+slot;
        accumulated += chunk;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) counts[expert] = accumulated;
}

// Prefix only 512 expert counts, then build a compact list of occupied tiles.
// The conservative grid is O(routes/tile + experts), not experts*batches/tile.
kernel void graph_grouped_tasks(device const uint* counts [[buffer(0)]],
    device uint* tasks [[buffer(1)]], constant uint4& s [[buffer(2)]],
    device atomic_uint* error [[buffer(7)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint offsets[513];
    const uint bound = s.z*s.y/s.w+s.x;
    if (!tid) {
        uint total = 0, routes = 0;
        for (uint e = 0; e < s.x; ++e) {
            offsets[e] = total;
            if (counts[e] > s.z) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
            total += (min(counts[e],s.z)+s.w-1)/s.w; routes += counts[e];
        }
        offsets[s.x] = total; tasks[0] = min(total,bound); tasks[1] = routes;
        if (total > bound || routes != s.z*s.y) atomic_fetch_or_explicit(error,4u,memory_order_relaxed);
    }
    for (uint i = tid; i < bound; i += 128) { tasks[2+2*i] = UINT_MAX; tasks[3+2*i] = UINT_MAX; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < s.x; e += 128) {
        const uint n = (min(counts[e],s.z)+s.w-1)/s.w;
        for (uint i = 0; i < n; ++i) if (offsets[e]+i < bound) {
            tasks[2+2*(offsets[e]+i)] = e; tasks[3+2*(offsets[e]+i)] = i*s.w;
        }
    }
}

float grouped_half(device const uchar* p, uint at) {
    return float(as_type<half>(ushort(uint(p[at]) | (uint(p[at+1]) << 8))));
}
float grouped_element(device const uchar* row, uint column, device atomic_uint* error) {
    const bool large = grouped_encoding == 12 || grouped_encoding == 13;
    const uint block_bytes = grouped_encoding == 7 ? 24 : grouped_encoding == 8 ? 34 : grouped_encoding == 12 ? 144 : 176;
    const uint block_width = large ? 256 : 32;
    device const uchar* p = row + (column/block_width)*block_bytes;
    const float d = grouped_half(p,0), m = grouped_encoding == 8 ? 0 : grouped_half(p,2);
    if (!isfinite(d) || !isfinite(m)) { atomic_fetch_or_explicit(error,1u,memory_order_relaxed); return 0; }
    const uint i = column%block_width;
    if (grouped_encoding == 8) return d*float(as_type<char>(p[2+i]));
    if (grouped_encoding == 7) {
        const uint high = (uint(p[4+i/8]) >> (i%8)) & 1;
        const uint low = (uint(p[8+i%16]) >> (4*(i/16))) & 15;
        return d*float(low | (high << 4)) + m;
    }
    const uint group = i/32; uint scale, minimum;
    if (group < 4) { scale = uint(p[4+group]) & 63; minimum = uint(p[8+group]) & 63; }
    else { const uint slot = group-4;
        scale = (uint(p[12+slot]) & 15) | ((uint(p[4+slot]) >> 6) << 4);
        minimum = (uint(p[12+slot]) >> 4) | ((uint(p[8+slot]) >> 6) << 4); }
    const uint payload = grouped_encoding == 12 ? 16 : 48;
    uint q = (uint(p[payload+(i/64)*32+i%32]) >> (4*((i%64)/32))) & 15;
    if (grouped_encoding == 13) q |= ((uint(p[16+i%32]) >> (i/32)) & 1) << 4;
    return d*float(scale)*float(q) - m*float(minimum);
}

template<uint Tile>
void grouped_tile(device const uchar* weights, device const float* input, device float* output,
    constant uint4& s, device const uint* indices, constant uint4& route,
    device const uint* token_map, device const uint* tasks, device const uint* counts,
    device atomic_uint* error, uint2 block) {
    const uint bound = route.y*route.x/Tile+s.w;
    if (tasks[0] > bound || tasks[1] != route.y*route.x) { atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return; }
    if (block.y >= tasks[0]) return;
    const uint expert = tasks[2+2*block.y], start = tasks[3+2*block.y];
    if (expert >= s.w || start >= route.y || counts[expert] > route.y || start >= counts[expert]) {
        atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
    }
    constexpr auto descriptor = matmul2d_descriptor(Tile,Tile,32,false,false,false,
        matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision);
    matmul2d<descriptor,execution_simdgroup> op;
    auto left = op.template get_left_input_cooperative_tensor<float,float,float>();
    auto right = op.template get_right_input_cooperative_tensor<float,float,float>();
    auto result = op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for (auto it = result.begin(); it != result.end(); ++it) *it = 0;
    const float input_bound = 0x1.fffffep127f/(8.0f*float(s.x)*134217728.0f);
    for (uint at = 0; at < s.x; at += 32) {
        for (auto it = left.begin(); it != left.end(); ++it) {
            *it = 0; if (!left.is_valid_element(it)) continue;
            const auto c = left.get_multidimensional_index(it); const uint row = block.x*Tile+c[1], column = at+c[0];
            if (c[0] < 32 && c[1] < int(Tile) && row < s.y && column < s.x)
                *it = grouped_element(weights+(ulong(expert)*s.y+row)*s.z,column,error);
        }
        for (auto it = right.begin(); it != right.end(); ++it) {
            *it = 0; if (!right.is_valid_element(it)) continue;
            const auto c = right.get_multidimensional_index(it); const uint slot = start+c[0];
            if (c[0] < int(Tile) && c[1] < 32 && slot < counts[expert] && at+c[1] < s.x) {
                const uint rid = token_map[expert*route.y+slot];
                if (rid >= route.y*route.x || indices[rid] != expert) { atomic_fetch_or_explicit(error,4u,memory_order_relaxed); continue; }
                const float value = input[ulong(route.w ? rid/route.x : rid)*s.x+at+c[1]];
                if (!isfinite(value) || abs(value) > input_bound) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
                *it = value;
            }
        }
        op.run(left,right,result);
    }
    for (auto it = result.begin(); it != result.end(); ++it) {
        if (!result.is_valid_element(it)) continue;
        const auto c = result.get_multidimensional_index(it); const uint slot = start+c[0], row = block.x*Tile+c[1];
        if (c[0] < int(Tile) && c[1] < int(Tile) && slot < counts[expert] && row < s.y) {
            const uint rid = token_map[expert*route.y+slot];
            if (rid >= route.y*route.x || indices[rid] != expert) { atomic_fetch_or_explicit(error,4u,memory_order_relaxed); continue; }
            const float value = *it; if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            output[ulong(rid)*s.y+row] = value;
        }
    }
}
#define GROUPED_KERNEL(TILE,NAME) \
kernel void NAME(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]], \
    device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], device const uint* indices [[buffer(4)]], \
    constant uint4& route [[buffer(5)]], device const uint* token_map [[buffer(6)]], device atomic_uint* error [[buffer(7)]], \
    device const uint* tasks [[buffer(8)]], device const uint* counts [[buffer(9)]], uint2 block [[threadgroup_position_in_grid]]) { \
    grouped_tile<TILE>(weights,input,output,shape,indices,route,token_map,tasks,counts,error,block); \
}
GROUPED_KERNEL(16,graph_grouped_mpp16)
GROUPED_KERNEL(32,graph_grouped_mpp32)

// Threadgroup tensor / four-SIMD TensorOps pattern adapted from llama.cpp
// 6050b467, mul_mm.metal kernel_mul_mm_id (MIT, ggml authors 2023-2026).
// See ../third_party/llama-cpp/LICENSE.
// Tokspur grouping/decoding/scatter stay in use. Unlike the upstream half
// staging, both local tensor inputs remain FP32, with relaxed precision off.
template<uint K>
void grouped_staged(device const uchar* weights, device const float* input, device float* output,
    constant uint4& s, device const uint* indices, constant uint4& route, device const uint* token_map,
    device const uint* tasks, device const uint* counts, device atomic_uint* error, uint2 block,
    ushort tid, threadgroup float* l, threadgroup float* r) {
    constexpr uint Tile = 32; const uint bound = route.y*route.x/Tile+s.w;
    if (tasks[0] > bound || tasks[1] != route.y*route.x) { atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return; }
    if (block.y >= tasks[0]) return;
    const uint expert = tasks[2+2*block.y], start = tasks[3+2*block.y];
    if (expert >= s.w || start >= route.y || counts[expert] > route.y || start >= counts[expert]) {
        atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
    }
    auto left = tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(l,dextents<int32_t,2>(K,Tile));
    auto right = tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(r,dextents<int32_t,2>(Tile,K));
    constexpr auto descriptor = matmul2d_descriptor(Tile,Tile,K,false,false,false,
        matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision);
    matmul2d<descriptor,execution_simdgroups<4>> op;
    auto result = op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for (auto it = result.begin(); it != result.end(); ++it) *it = 0;
    const float input_bound = 0x1.fffffep127f/(8.0f*float(s.x)*134217728.0f);
    for (uint at = 0; at < s.x; at += K) {
        for (uint i = tid; i < Tile*K; i += 128) {
            const uint row = block.x*Tile+i/K, column = at+i%K;
            l[i] = row < s.y && column < s.x ? grouped_element(weights+(ulong(expert)*s.y+row)*s.z,column,error) : 0;
        }
        for (uint i = tid; i < Tile*K; i += 128) {
            r[i] = 0; const uint slot = start+i%Tile, column = at+i/Tile;
            if (slot < counts[expert] && column < s.x) {
                const uint rid = token_map[expert*route.y+slot];
                if (rid >= route.y*route.x || indices[rid] != expert) { atomic_fetch_or_explicit(error,4u,memory_order_relaxed); continue; }
                const float value = input[ulong(route.w ? rid/route.x : rid)*s.x+column];
                if (!isfinite(value) || abs(value) > input_bound) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
                r[i] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (auto it = result.begin(); it != result.end(); ++it) {
        if (!result.is_valid_element(it)) continue;
        const auto c = result.get_multidimensional_index(it); const uint slot = start+c[0], row = block.x*Tile+c[1];
        if (c[0] < int(Tile) && c[1] < int(Tile) && slot < counts[expert] && row < s.y) {
            const uint rid = token_map[expert*route.y+slot];
            if (rid >= route.y*route.x || indices[rid] != expert) { atomic_fetch_or_explicit(error,4u,memory_order_relaxed); continue; }
            const float value = *it; if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            output[ulong(rid)*s.y+row] = value;
        }
    }
}
#define GROUPED_STAGED(K,NAME) \
kernel void NAME(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]], \
    device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], device const uint* indices [[buffer(4)]], \
    constant uint4& route [[buffer(5)]], device const uint* token_map [[buffer(6)]], device atomic_uint* error [[buffer(7)]], \
    device const uint* tasks [[buffer(8)]], device const uint* counts [[buffer(9)]], uint2 block [[threadgroup_position_in_grid]], \
    ushort tid [[thread_index_in_threadgroup]]) { \
    threadgroup float left[32*K], right[32*K]; \
    grouped_staged<K>(weights,input,output,shape,indices,route,token_map,tasks,counts,error,block,tid,left,right); \
}
GROUPED_STAGED(32,graph_grouped_staged32)
GROUPED_STAGED(128,graph_grouped_staged128)
