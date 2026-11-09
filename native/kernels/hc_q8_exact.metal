#include <metal_stdlib>
using namespace metal;

// Original launch/prefetch experiment over unchanged owned Q8_0 bytes. There
// are no partition sums: one lane accumulator visits blocks0..319 in order.
// Load-ahead arrays retain float scale/q/x; the accumulation expression and
// single simd_sum match graph_block_gemv. Compile with -fno-fast-math.
template<ushort R, ushort U>
void hc_q8_exact(device const uchar* weights, device const float* input, device float* output,
                 device atomic_uint* error, uint group, ushort subgroup, ushort lane) {
    const uint row = group * R + subgroup;
    if (row >= 320) return;
    device const uchar* data = weights + ulong(row) * 320 * 34;
    float accumulator = 0;
    for (uint base = 0; base < 320; base += U) {
        float scales[U], quants[U], values[U];
        #pragma unroll
        for (ushort i = 0; i < U; ++i) {
            device const uchar* block = data + ulong(base + i) * 34;
            scales[i] = float(*reinterpret_cast<device const half*>(block));
            quants[i] = float(*reinterpret_cast<device const char*>(block + 2 + lane));
            values[i] = input[(base + i) * 32 + lane];
        }
        #pragma unroll
        for (ushort i = 0; i < U; ++i) accumulator += (scales[i] * quants[i]) * values[i];
    }
    const float total = simd_sum(accumulator);
    if (!lane) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[row] = total;
    }
}
#define HC_Q8_EXACT(R, U) \
kernel void graph_hc_q8_exact_r##R##_u##U(device const uchar* weights [[buffer(0)]], \
                                        device const float* input [[buffer(1)]], \
                                        device float* output [[buffer(2)]], \
                                        device atomic_uint* error [[buffer(7)]], \
                                        uint group [[threadgroup_position_in_grid]], \
                                        ushort subgroup [[simdgroup_index_in_threadgroup]], \
                                        ushort lane [[thread_index_in_simdgroup]]) { \
    hc_q8_exact<R, U>(weights, input, output, error, group, subgroup, lane); \
}
#define HC_Q8_EXACT_ROW(R) \
HC_Q8_EXACT(R, 1) HC_Q8_EXACT(R, 2) HC_Q8_EXACT(R, 4) HC_Q8_EXACT(R, 8) HC_Q8_EXACT(R, 16)
HC_Q8_EXACT_ROW(1)
HC_Q8_EXACT_ROW(2)
HC_Q8_EXACT_ROW(4)
HC_Q8_EXACT(8, 1) HC_Q8_EXACT(8, 2) HC_Q8_EXACT(8, 4) HC_Q8_EXACT(8, 8)
// Partial-library fixture only: verify failed all-PSO preparation never admits
// the nineteen cached variants. Normal component builds always include all20.
#ifndef TOKSPUR_HC_EXACT_OMIT_LAST
HC_Q8_EXACT(8, 16)
#endif
