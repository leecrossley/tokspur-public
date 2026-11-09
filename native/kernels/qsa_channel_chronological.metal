#include <metal_stdlib>
using namespace metal;

// OWN experiment only. Pass1 is the unchanged qsa_score_stage function from the
// existing library. No target/graph caller selects these new kernels.
enum : uint { chronological_chunk = 9, chronological_threads = 256 };

kernel void qsa_chronological_factors(device const float* scores [[buffer(0)]],
    device float2* factors [[buffer(1)]], constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (head >= s.x) return;
    const uint tokens = min(p.x / s.w, p.y) * s.w + p.x % s.w;
    if (!tokens || tokens > 2051 || s.x > 24 || s.z > 256) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); return;
    }
    // Snapshot once: a different head can set the device flag during THIS
    // dispatch. Per-thread early reads could diverge around the barriers below.
    threadgroup atomic_uint entry_error;
    threadgroup float prefix[chronological_threads];
    threadgroup atomic_uint invalid;
    if (!tid) {
        atomic_store_explicit(&entry_error, atomic_load_explicit(error, memory_order_relaxed), memory_order_relaxed);
        atomic_store_explicit(&invalid, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&entry_error, memory_order_relaxed)) return; // Immutable group-uniform snapshot.
    const uint first = tid * chronological_chunk, end = min(tokens, first + chronological_chunk);
    float chunk_max = -INFINITY;
    for (uint item = first; item < end; ++item) {
        const float score = scores[head * tokens + item];
        if (!isfinite(score)) atomic_fetch_or_explicit(&invalid, 1u, memory_order_relaxed);
        chunk_max = max(chunk_max, score);
    }
    prefix[tid] = chunk_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&invalid, memory_order_relaxed)) {
        if (!tid) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return;
    }
    // Read old values before any write at each level. The left/older operand
    // stays left, including equal scores and signed-zero boundary fixtures.
    for (uint distance = 1; distance < chronological_threads; distance <<= 1) {
        const float current = prefix[tid], older = tid >= distance ? prefix[tid - distance] : -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix[tid] = max(older, current);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = tid ? prefix[tid - 1] : -INFINITY;
    for (uint item = first; item < end; ++item) {
        const float score = scores[head * tokens + item];
        const float next_max = max(maximum, score), old_scale = exp(maximum - next_max), scale = exp(score - next_max);
        factors[head * tokens + item] = float2(old_scale, scale);
        maximum = next_max;
    }
}

kernel void qsa_chronological_channels(device const float2* factors [[buffer(0)]],
    device const ushort* values [[buffer(1)]], device const uint* ids [[buffer(2)]], device float* output [[buffer(3)]],
    constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], uint channel [[thread_index_in_threadgroup]]) {
    if (head >= s.x || channel >= s.z || atomic_load_explicit(error, memory_order_relaxed)) return;
    const uint blocks = p.x / s.w, selected = min(blocks, p.y), tokens = selected * s.w + p.x % s.w;
    const uint key_head = head / (s.x / s.y), row_width = s.y * s.z;
    float denominator = 0, sum = 0;
    for (uint item = 0; item < tokens; ++item) {
        uint token;
        if (item < selected * s.w) {
            const uint slot = item / s.w, id = blocks <= p.y ? slot : ids[slot];
            if (id >= blocks || (blocks > p.y && slot && id <= ids[slot - 1])) {
                atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); return;
            }
            token = id * s.w + item % s.w;
        } else token = blocks * s.w + item - selected * s.w;
        const float2 pair = factors[head * tokens + item];
        const float value = as_type<float>(uint(values[token * row_width + key_head * s.z + channel]) << 16);
        if (!all(isfinite(pair)) || !isfinite(value)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        // Same operations, token chronology and final divide as the original.
        // There is no parallel sum/reduction, denominator reassociation or sum[8].
        denominator = denominator * pair.x + pair.y;
        sum = sum * pair.x + pair.y * value;
    }
    const float result = sum / denominator;
    if (!isfinite(result)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    output[head * s.z + channel] = result;
}

// TEST-ONLY serial factor oracle. This is never part of the three-pass schedule
// or its timing: it exposes the original max/exp chronology for byte admission.
kernel void qsa_chronological_serial_factors(device const float* scores [[buffer(0)]],
    device float2* factors [[buffer(1)]], constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (head >= s.x || lane || atomic_load_explicit(error, memory_order_relaxed)) return;
    const uint tokens = min(p.x / s.w, p.y) * s.w + p.x % s.w;
    float maximum = -INFINITY;
    for (uint item = 0; item < tokens; ++item) {
        const float score = scores[head * tokens + item];
        if (!isfinite(score)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        const float next_max = max(maximum, score), old_scale = exp(maximum - next_max), scale = exp(score - next_max);
        factors[head * tokens + item] = float2(old_scale, scale); maximum = next_max;
    }
}
