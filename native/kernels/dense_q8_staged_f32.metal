// Bounded dense adaptation of Tokspur's grouped_mm_f32 tile and packed Q8 decoder from llama.cpp 6050b467.
// Copyright (c) 2023-2026 The ggml authors. MIT notice: ../third_party/llama-cpp/LICENSE.
// Grid band ordering follows m5-max-kernels a6d92a3bdd7c784fde2e3e86b303d533354f4603 (MIT, Copyright 2026 9Tempest). The untraced multi-segment kernels are removed.
// Tokspur retains the graph, views, scheduling and original raw GGUF Q8_0 representation.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct DenseQ8StagedBlock { half d; char qs[32]; };
static_assert(sizeof(DenseQ8StagedBlock) == 34, "original GGUF Q8_0 bytes");

inline uint2 dense_q8_swizzle_2d(uint2 group, uint num_gx, uint num_gy, uint swizzle) {
    if (swizzle <= 1 || num_gx <= 1 || num_gy <= 1) return group;
    const uint linear = group.y * num_gx + group.x;
    const uint band_tiles = num_gx * swizzle;
    const uint band_idx = linear / band_tiles;
    const uint first_y = band_idx * swizzle;
    const uint band_h = min(swizzle, num_gy - first_y);
    const uint in_band = linear - band_idx * band_tiles;
    return uint2(in_band / band_h, first_y + (in_band % band_h));
}

inline float4x4 dense_q8_staged_decode16(device const DenseQ8StagedBlock* block, uint half_block) {
    device const packed_char4* values = reinterpret_cast<device const packed_char4*>(block->qs);
    const float scale = float(block->d); float4x4 decoded;
    for (uint i = 0; i < 4; ++i) decoded[i] = float4(values[4 * half_block + i]) * scale;
    return decoded;
}

inline float4x4 dense_q8_load_weight16(device const uchar* weights, uint columns, uint row_bytes,
    uint row_idx, uint at, uint half_block, uint layout) {
    const ulong row_ptr = ulong(row_idx) * row_bytes;
    const uint blk = at / 32;
    if (layout == 1) {
        float4x4 decoded;
        const uint blocks = columns / 32;
        const float scale = float(*reinterpret_cast<device const half*>(weights + row_ptr + ulong(blk) * 2));
        const uint run = blk / 16;
        const uint run_width = min(16u, blocks - run * 16u);
        const ulong quant_base = row_ptr + ulong(blocks) * 2 + ulong(run) * 512 + (blk % 16);
        for (uint i = 0; i < 4; ++i) {
            for (uint j = 0; j < 4; ++j) {
                const uint elem = half_block * 16 + i * 4 + j;
                const float q = float(*reinterpret_cast<device const char*>(weights + quant_base + ulong(elem) * run_width));
                decoded[i][j] = q * scale;
            }
        }
        return decoded;
    }
    device const DenseQ8StagedBlock* block = reinterpret_cast<device const DenseQ8StagedBlock*>(weights + row_ptr + ulong(blk) * 34);
    return dense_q8_staged_decode16(block, half_block);
}

inline void dense_q8_staged_tile32(device const uchar* weights, device const float* input, device float* output,
    uint4 shape, uint layout, device atomic_uint* error, threadgroup float* shared, uint2 group,
    ushort tid, ushort lane, ushort sg) {
    constexpr uint K = 32, Rows = 64, Tokens = 32, HalfTokens = 16;
    const uint first_row = group.x * Rows, first_token = group.y * Tokens;
    if (first_row >= shape.y || first_token >= shape.w) return;
    const uint active_rows = min(Rows, shape.y - first_row), active_tokens = min(Tokens, shape.w - first_token);
    const uint local_row = tid / 2, half_block = tid % 2, local_token = tid / 4, input_column = 8 * (tid % 4);
    threadgroup float* a = shared; threadgroup float* b = shared + Rows * K;
    auto tA = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(K, Rows));
    auto tB0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(K, HalfTokens));
    auto tB1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(b + HalfTokens * K, dextents<int32_t, 2>(K, HalfTokens));
    static_assert(is_same_v<typename decltype(tA)::value_type, float> && is_same_v<typename decltype(tB0)::value_type, float> && is_same_v<typename decltype(tB1)::value_type, float>, "full FP32 staging");
    constexpr auto descriptor = matmul2d_descriptor(HalfTokens, Rows, K, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "full FP32 TensorOps precision");
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto low = operation.template get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto high = operation.template get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();
    for (auto i = low.begin(); i != low.end(); ++i) *i = 0;
    for (auto i = high.begin(); i != high.end(); ++i) *i = 0;
    const float input_bound = 0x1.fffffep127f / (8.0f * float(shape.x) * 8388608.0f);
    for (uint at = 0; at < shape.x; at += K) {
        const float4x4 decoded = local_row < active_rows
            ? dense_q8_load_weight16(weights, shape.x, shape.z, first_row + local_row, at, half_block, layout)
            : float4x4(0.0f);
        for (uint i = 0; i < 16; ++i) {
            const float value = decoded[i / 4][i % 4];
            if (!isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            a[local_row * K + half_block * 16 + i] = value;
        }
        for (uint i = 0; i < 8; ++i) {
            const float value = local_token < active_tokens ? input[ulong(first_token + local_token) * shape.x + at + input_column + i] : 0;
            if (!isfinite(value) || abs(value) > input_bound) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            b[local_token * K + input_column + i] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto left = tA.slice(0, 0), lower = tB0.slice(0, 0); operation.run(lower, left, low);
        if (active_tokens > HalfTokens) { auto upper = tB1.slice(0, 0); operation.run(upper, left, high); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto out0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(shared, dextents<int32_t, 2>(Rows, HalfTokens)); low.store(out0);
    if (active_tokens > HalfTokens) {
        auto out1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(shared + HalfTokens * Rows, dextents<int32_t, 2>(Rows, HalfTokens)); high.store(out1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint token = sg; token < active_tokens; token += 4) for (uint row = lane; row < active_rows; row += 32) {
        const float value = shared[token * Rows + row];
        if (!isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[ulong(first_token + token) * shape.y + first_row + row] = value;
    }
}

inline void dense_q8_staged_tile64(device const uchar* weights, device const float* input, device float* output,
    uint4 shape, uint layout, device atomic_uint* error, threadgroup float* shared, uint2 group,
    ushort tid, ushort lane, ushort sg) {
    constexpr uint K = 32, Rows = 64, Tokens = 64, SubTokens = 16;
    const uint first_row = group.x * Rows, first_token = group.y * Tokens;
    if (first_row >= shape.y || first_token >= shape.w) return;
    const uint active_rows = min(Rows, shape.y - first_row), active_tokens = min(Tokens, shape.w - first_token);
    const uint local_row = tid / 2, half_block = tid % 2, local_token = tid / 2, input_column = 16 * (tid % 2);
    threadgroup float* a = shared; threadgroup float* b = shared + Rows * K;
    auto tA = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(K, Rows));
    auto tB0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(K, SubTokens));
    auto tB1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(b + SubTokens * K, dextents<int32_t, 2>(K, SubTokens));
    auto tB2 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(b + 2 * SubTokens * K, dextents<int32_t, 2>(K, SubTokens));
    auto tB3 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(b + 3 * SubTokens * K, dextents<int32_t, 2>(K, SubTokens));
    constexpr auto descriptor = matmul2d_descriptor(SubTokens, Rows, K, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "full FP32 TensorOps precision");
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto c0 = operation.template get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto c1 = operation.template get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();
    auto c2 = operation.template get_destination_cooperative_tensor<decltype(tA), decltype(tB2), float>();
    auto c3 = operation.template get_destination_cooperative_tensor<decltype(tA), decltype(tB3), float>();
    for (auto i = c0.begin(); i != c0.end(); ++i) *i = 0;
    for (auto i = c1.begin(); i != c1.end(); ++i) *i = 0;
    for (auto i = c2.begin(); i != c2.end(); ++i) *i = 0;
    for (auto i = c3.begin(); i != c3.end(); ++i) *i = 0;
    const float input_bound = 0x1.fffffep127f / (8.0f * float(shape.x) * 8388608.0f);
    for (uint at = 0; at < shape.x; at += K) {
        const float4x4 decoded = local_row < active_rows
            ? dense_q8_load_weight16(weights, shape.x, shape.z, first_row + local_row, at, half_block, layout)
            : float4x4(0.0f);
        for (uint i = 0; i < 16; ++i) {
            const float value = decoded[i / 4][i % 4];
            if (!isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            a[local_row * K + half_block * 16 + i] = value;
        }
        for (uint i = 0; i < 16; ++i) {
            const float value = local_token < active_tokens ? input[ulong(first_token + local_token) * shape.x + at + input_column + i] : 0;
            if (!isfinite(value) || abs(value) > input_bound) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            b[local_token * K + input_column + i] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto left = tA.slice(0, 0);
        auto b0 = tB0.slice(0, 0); operation.run(b0, left, c0);
        if (active_tokens > 16) { auto b1 = tB1.slice(0, 0); operation.run(b1, left, c1); }
        if (active_tokens > 32) { auto b2 = tB2.slice(0, 0); operation.run(b2, left, c2); }
        if (active_tokens > 48) { auto b3 = tB3.slice(0, 0); operation.run(b3, left, c3); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto out0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(shared, dextents<int32_t, 2>(Rows, SubTokens)); c0.store(out0);
    if (active_tokens > 16) {
        auto out1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(shared + SubTokens * Rows, dextents<int32_t, 2>(Rows, SubTokens)); c1.store(out1);
    }
    if (active_tokens > 32) {
        auto out2 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(shared + 2 * SubTokens * Rows, dextents<int32_t, 2>(Rows, SubTokens)); c2.store(out2);
    }
    if (active_tokens > 48) {
        auto out3 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(shared + 3 * SubTokens * Rows, dextents<int32_t, 2>(Rows, SubTokens)); c3.store(out3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint token = sg; token < active_tokens; token += 4) for (uint row = lane; row < active_rows; row += 32) {
        const float value = shared[token * Rows + row];
        if (!isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[ulong(first_token + token) * shape.y + first_row + row] = value;
    }
}

kernel void graph_dense_q8_staged_f32(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], constant uint& layout_kind [[buffer(4)]], device atomic_uint* error [[buffer(7)]],
    threadgroup float* shared [[threadgroup(0)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint num_gx = (shape.y + 63) / 64, num_gy = (shape.w + 31) / 32;
    const uint2 g = dense_q8_swizzle_2d(group, num_gx, num_gy, (layout_kind & 2u) ? 8u : 0u);
    dense_q8_staged_tile32(weights, input, output, shape, layout_kind & 1u, error, shared, g, tid, lane, sg);
}

kernel void graph_dense_q8_staged_f32_m64(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], constant uint& layout_kind [[buffer(4)]], device atomic_uint* error [[buffer(7)]],
    threadgroup float* shared [[threadgroup(0)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint num_gx = (shape.y + 63) / 64, num_gy = (shape.w + 63) / 64;
    const uint2 g = dense_q8_swizzle_2d(group, num_gx, num_gy, (layout_kind & 2u) ? 8u : 0u);
    dense_q8_staged_tile64(weights, input, output, shape, layout_kind & 1u, error, shared, g, tid, lane, sg);
}
