#include <metal_stdlib>
using namespace metal;

// Original bounded experiment. Each SIMD group owns one (row, split), with
// contiguous Q8_0 block partitions. Split 1 preserves the owned fast block-Q8
// loop/lane/simd_sum arithmetic. Larger splits deliberately reassociate FP32
// addition; there is no numerical atomic or higher-precision accumulation.
kernel void graph_q8_split_partials(device const uchar* weights [[buffer(0)]],
                                    device const float* input [[buffer(1)]],
                                    device float* partials [[buffer(2)]],
                                    constant uint4& shape [[buffer(3)]],
                                    device atomic_uint* error_flag [[buffer(7)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    ushort subgroup [[simdgroup_index_in_threadgroup]],
                                    ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x * 4 + subgroup, split = group.y;
    if (row >= shape.y) return;
    const uint blocks = shape.x / 32;
    const uint first = blocks * split / shape.z, last = blocks * (split + 1) / shape.z;
    device const uchar* encoded_row = weights + ulong(row) * blocks * 34;
    float accumulator = 0;
    for (uint b = first; b < last; ++b) {
        device const uchar* block = encoded_row + ulong(b) * 34;
        const float scale = float(*reinterpret_cast<device const half*>(block));
        const float q = float(*reinterpret_cast<device const char*>(block + 2 + lane));
        accumulator += (scale * q) * input[b * 32 + lane];
    }
    const float total = simd_sum(accumulator);
    if (!lane) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error_flag, 1u, memory_order_relaxed);
        partials[row * shape.z + split] = total;
    }
}

kernel void graph_q8_split_reduce(device const float* partials [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  constant uint4& shape [[buffer(2)]],
                                  device atomic_uint* error_flag [[buffer(7)]],
                                  uint row [[thread_position_in_grid]]) {
    if (row >= shape.y) return;
    // Starting from partial 0 avoids an extra +0 in the split-1 exact comparator.
    float total = partials[row * shape.z];
    bool finite = isfinite(total);
    for (uint split = 1; split < shape.z; ++split) {
        const float partial = partials[row * shape.z + split];
        finite = finite && isfinite(partial);
        total += partial;
    }
    if (!finite || !isfinite(total)) atomic_fetch_or_explicit(error_flag, 1u, memory_order_relaxed);
    output[row] = total;
}
