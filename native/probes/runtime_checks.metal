#include <metal_stdlib>
using namespace metal;

// Probe-only functions. This source is never part of tokspur-owned.metallib.
kernel void probe_checked(device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
                          device atomic_uint* error [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float value = input[i];
    if (!isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    output[i] = value;
}

kernel void probe_unchecked(device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
                            device atomic_uint* error [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    (void)error;
    if (i < count) output[i] = input[i];
}

// The serving graph_rms arithmetic and reduction order, extracted from Tokspur's
// own graph.metal. Structural bounds remain in both probe variants. Only the
// finite-value checks differ; no variant is linked into the serving library.
template<bool Checked>
static inline void probe_rms_impl(device const float* input, device const float* gains, device float* output,
    constant uint4& shape, constant float& epsilon, device atomic_uint* error,
    uint group, ushort subgroup, ushort lane) {
    const uint block = group * 4 + subgroup;
    if (block >= shape.x / shape.y) return;
    const uint base = block * shape.y;
    float squared = 0;
    for (uint i = lane; i < shape.y; i += 32) squared += input[base + i] * input[base + i];
    const float total = simd_sum(squared);
    const float denominator = total / float(shape.y) + epsilon;
    if (Checked && (!isfinite(total) || !isfinite(denominator))) {
        if (lane == 0) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    const float inverse = rsqrt(denominator);
    for (uint i = lane; i < shape.y; i += 32) {
        const float value = input[base + i] * inverse * gains[(shape.z ? 0 : base) + i];
        if (Checked && !isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[base + i] = value;
    }
}

kernel void probe_rms_checked(device const float* input [[buffer(0)]], device const float* gains [[buffer(1)]],
    device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], constant float& epsilon [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    probe_rms_impl<true>(input, gains, output, shape, epsilon, error, group, subgroup, lane);
}

kernel void probe_rms_unchecked(device const float* input [[buffer(0)]], device const float* gains [[buffer(1)]],
    device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], constant float& epsilon [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    probe_rms_impl<false>(input, gains, output, shape, epsilon, error, group, subgroup, lane);
}
