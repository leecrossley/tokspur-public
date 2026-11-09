#include <metal_stdlib>
using namespace metal;

// Generalization of OWN hc_q8_exact: four independent SIMD rows per group,
// fixed original Q8 bytes, one ordered accumulator and one unchanged simd_sum.
template<uint C, uint R, ushort U>
void q8_wide(device const uchar* weights, device const float* input, device float* output,
             device atomic_uint* error, uint group, ushort subgroup, ushort lane) {
    const uint row = group * 4 + subgroup;
    if (row >= R) return;
    device const uchar* data = weights + ulong(row) * (C / 32) * 34;
    float accumulator = 0;
    for (uint base = 0; base < C / 32; base += U) {
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
#define WIDE_Q8(C, R, U) \
kernel void q8_wide_c##C##_r##R##_u##U(device const uchar* weights [[buffer(0)]], \
                                      device const float* input [[buffer(1)]], \
                                      device float* output [[buffer(2)]], \
                                      device atomic_uint* error [[buffer(7)]], \
                                      uint group [[threadgroup_position_in_grid]], \
                                      ushort subgroup [[simdgroup_index_in_threadgroup]], \
                                      ushort lane [[thread_index_in_simdgroup]]) { \
    q8_wide<C, R, U>(weights, input, output, error, group, subgroup, lane); \
}
WIDE_Q8(2560, 10240, 8)
WIDE_Q8(2560, 10240, 16)
WIDE_Q8(6144, 2560, 8)
#ifndef TOKSPUR_WIDE_OMIT_LAST
WIDE_Q8(6144, 2560, 16)
#endif
