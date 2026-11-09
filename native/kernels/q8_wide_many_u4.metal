#include <metal_stdlib>
using namespace metal;

// OWN f2e5c4d ordered wide-many body, unchanged. Only its six U4 entries are
// registered here; the exploratory U8 entries (including the N8 losses) are not.
template<uint C, uint R, ushort N, ushort U>
void wide_many(device const uchar* weights, device const float* input, device float* output,
               device atomic_uint* error, uint group, ushort subgroup, ushort lane) {
    const uint row = group * 4 + subgroup;
    if (row >= R) return;
    device const uchar* data = weights + ulong(row) * (C / 32) * 34;
    float sums[N];
    for (ushort token = 0; token < N; ++token) sums[token] = 0;
    for (uint base = 0; base < C / 32; base += U) {
        float scales[U], quants[U], values[N][U];
        #pragma unroll
        for (ushort i = 0; i < U; ++i) {
            device const uchar* block = data + ulong(base + i) * 34;
            scales[i] = float(*reinterpret_cast<device const half*>(block));
            quants[i] = float(*reinterpret_cast<device const char*>(block + 2 + lane));
            #pragma unroll
            for (ushort token = 0; token < N; ++token) values[token][i] = input[ulong(token) * C + (base + i) * 32 + lane];
        }
        #pragma unroll
        for (ushort i = 0; i < U; ++i) {
            const float weight = scales[i] * quants[i];
            #pragma unroll
            for (ushort token = 0; token < N; ++token) sums[token] += weight * values[token][i];
        }
    }
    for (ushort token = 0; token < N; ++token) {
        const float total = simd_sum(sums[token]);
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[ulong(token) * R + row] = total;
        }
    }
}
#define WIDE_MANY(C, R, N, U) \
kernel void q8_wide_many_c##C##_r##R##_n##N##_u##U(device const uchar* weights [[buffer(0)]], \
    device const float* input [[buffer(1)]], device float* output [[buffer(2)]], device atomic_uint* error [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) { \
    wide_many<C, R, N, U>(weights, input, output, error, group, subgroup, lane); \
}
WIDE_MANY(2560, 10240, 2, 4)
WIDE_MANY(2560, 10240, 4, 4)
WIDE_MANY(2560, 10240, 8, 4)
WIDE_MANY(6144, 2560, 2, 4)
WIDE_MANY(6144, 2560, 4, 4)
#ifndef TOKSPUR_WIDE_MANY_U4_OMIT_LAST
WIDE_MANY(6144, 2560, 8, 4)
#endif
