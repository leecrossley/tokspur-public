#include <metal_stdlib>
using namespace metal;

// Independently authored fixed-HC load-ahead. Each token keeps the ordinary
// q8_many lane accumulator's block0..319 order and one final simd_sum. Only
// independent loads move earlier; neither partition sums nor FMA rewrites.
template<ushort N>
void hc_exact_many(device const uchar* weights, device const float* input, device float* output,
                   device atomic_uint* error, uint row, ushort lane) {
    if (row >= 320) return;
    device const uchar* data = weights + ulong(row) * 320 * 34;
    float sums[N];
    for (ushort token = 0; token < N; ++token) sums[token] = 0;
    for (uint base = 0; base < 320; base += 16) {
        float scales[16], quants[16], values[N][16];
        #pragma unroll
        for (ushort i = 0; i < 16; ++i) {
            device const uchar* block = data + ulong(base + i) * 34;
            scales[i] = float(*reinterpret_cast<device const half*>(block));
            quants[i] = float(*reinterpret_cast<device const char*>(block + 2 + lane));
            for (ushort token = 0; token < N; ++token)
                values[token][i] = input[ulong(token) * 10240 + (base + i) * 32 + lane];
        }
        #pragma unroll
        for (ushort i = 0; i < 16; ++i) {
            const float weight = scales[i] * quants[i];
            for (ushort token = 0; token < N; ++token) sums[token] += weight * values[token][i];
        }
    }
    for (ushort token = 0; token < N; ++token) {
        const float total = simd_sum(sums[token]);
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[ulong(token) * 320 + row] = total;
        }
    }
}
#define HC_EXACT_MANY(N) \
kernel void graph_hc_exact_many_##N(device const uchar* weights [[buffer(0)]], \
                                    device const float* input [[buffer(1)]], \
                                    device float* output [[buffer(2)]], \
                                    device atomic_uint* error [[buffer(7)]], \
                                    uint row [[threadgroup_position_in_grid]], \
                                    ushort lane [[thread_index_in_simdgroup]]) { \
    hc_exact_many<N>(weights, input, output, error, row, lane); \
}
HC_EXACT_MANY(1)
HC_EXACT_MANY(2)
HC_EXACT_MANY(4)
#ifndef TOKSPUR_HC_EXACT_MANY_OMIT_LAST
HC_EXACT_MANY(8)
#endif
