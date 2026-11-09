#include <metal_stdlib>
using namespace metal;

// One SIMD group owns a row, as in the ordinary Q8 block path. Decode each
// weight ONCE, then update N independent accumulators. Every token retains
// lane, lane+32, ... order and the final simd_sum boundary. No layout conversion.
template<ushort N>
void q8_many(device const uchar* weights, device const float* input, device float* output,
             constant uint4& s, device atomic_uint* error, uint group, ushort subgroup, ushort lane) {
    const uint row = group * 4 + subgroup;
    if (row >= s.y) return;
    device const uchar* data = weights + ulong(row) * s.z;
    float sums[N];
    for (ushort token = 0; token < N; ++token) sums[token] = 0;
    for (uint block = 0; block < s.x / 32; ++block) {
        device const uchar* encoded = data + ulong(block) * 34;
        const float scale = float(*reinterpret_cast<device const half*>(encoded));
        const float quantized = float(*reinterpret_cast<device const char*>(encoded + 2 + lane));
        const float weight = scale * quantized;
        for (ushort token = 0; token < N; ++token)
            sums[token] += weight * input[ulong(token) * s.x + block * 32 + lane];
    }
    for (ushort token = 0; token < N; ++token) {
        const float total = simd_sum(sums[token]);
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[ulong(token) * s.y + row] = total;
        }
    }
}

#define Q8_MANY_KERNEL(N) \
kernel void graph_q8_many_##N(device const uchar* weights [[buffer(0)]], \
                              device const float* input [[buffer(1)]], \
                              device float* output [[buffer(2)]], \
                              constant uint4& shape [[buffer(3)]], \
                              device atomic_uint* error [[buffer(7)]], \
                              uint group [[threadgroup_position_in_grid]], \
                              ushort subgroup [[simdgroup_index_in_threadgroup]], \
                              ushort lane [[thread_index_in_simdgroup]]) { \
    q8_many<N>(weights, input, output, shape, error, group, subgroup, lane); \
}
Q8_MANY_KERNEL(1)
Q8_MANY_KERNEL(2)
Q8_MANY_KERNEL(4)
Q8_MANY_KERNEL(8)
