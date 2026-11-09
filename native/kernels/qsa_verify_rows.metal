#include <metal_stdlib>
using namespace metal;

// Owned source adaptation: score row axis from 4f56424; factor/channel
// expressions from corrected ccb35f6. Fixed physical row strides do not alter
// any query's SIMD dot, prefix-max operands or chronological channel updates.
// This private component has no sampling, cache append or publication policy.
enum : uint { chronological_chunk = 9, chronological_threads = 256 };

float qsa_rows_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
bool qsa_rows_token(uint item, device const uint* ids, uint4 s, uint4 p,
                     thread uint& token, device atomic_uint* error) {
    const uint blocks = p.x / s.w, selected = min(blocks, p.y);
    if (item < selected * s.w) {
        const uint slot = item / s.w, id = blocks <= p.y ? slot : ids[slot];
        if (id >= blocks || (blocks > p.y && slot && id <= ids[slot - 1])) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); return false;
        }
        token = id * s.w + item % s.w;
    } else token = blocks * s.w + item - selected * s.w;
    return true;
}

kernel void qsa_verify_rows_score(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
                            device const uint* ids [[buffer(2)]], device float* scores [[buffer(3)]],
                            constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]],
                            device atomic_uint* error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
                            ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.z, head = group.y, item = group.x * 4 + subgroup;
    if (row >= p.z || head >= s.x) return;
    const uint4 causal = uint4(p.x + row + 1, p.y, 0, 0);
    const uint blocks = causal.x / s.w, tokens = min(blocks, p.y) * s.w + causal.x % s.w;
    if (item >= tokens) return;
    uint token;
    if (!qsa_rows_token(item, ids + row * p.y, s, causal, token, error)) return;
    const uint key_head = head / (s.x / s.y), row_width = s.y * s.z;
    const uint base = token * row_width + key_head * s.z, query_base = row * s.x * s.z;
    float dot = 0;
    for (uint d = lane; d < s.z; d += 32) dot += query[query_base + head * s.z + d] * qsa_rows_bf16(keys[base + d]);
    const float score = simd_sum(dot) * rsqrt(float(s.z));
    if (!isfinite(score)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    if (lane == 0) scores[row * 24 * p.w + head * p.w + item] = score;
}

kernel void qsa_verify_rows_factors(device const float* scores [[buffer(0)]],
    device float2* factors [[buffer(1)]], constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint head = group.x, row = group.y;
    if (head >= s.x || row >= p.z) return;
    const uint causal_prefix = p.x + row + 1, score_base = (row * 24 + head) * p.w;
    const uint tokens = min(causal_prefix / s.w, p.y) * s.w + causal_prefix % s.w;
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
        const float score = scores[score_base + item];
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
        const float score = scores[score_base + item];
        const float next_max = max(maximum, score), old_scale = exp(maximum - next_max), scale = exp(score - next_max);
        factors[score_base + item] = float2(old_scale, scale);
        maximum = next_max;
    }
}

kernel void qsa_verify_rows_channels(device const float2* factors [[buffer(0)]],
    device const ushort* values [[buffer(1)]], device const uint* ids [[buffer(2)]], device float* output [[buffer(3)]],
    constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint channel [[thread_index_in_threadgroup]]) {
    const uint head = group.x, row = group.y;
    if (head >= s.x || row >= p.z || channel >= s.z || atomic_load_explicit(error, memory_order_relaxed)) return;
    const uint causal_prefix = p.x + row + 1, score_base = (row * 24 + head) * p.w;
    ids += row * p.y;
    const uint blocks = causal_prefix / s.w, selected = min(blocks, p.y), tokens = selected * s.w + causal_prefix % s.w;
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
        const float2 pair = factors[score_base + item];
        const float value = as_type<float>(uint(values[token * row_width + key_head * s.z + channel]) << 16);
        if (!all(isfinite(pair)) || !isfinite(value)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        // Same operations, token chronology and final divide as the original.
        // There is no parallel sum/reduction, denominator reassociation or sum[8].
        denominator = denominator * pair.x + pair.y;
        sum = sum * pair.x + pair.y * value;
    }
    const float result = sum / denominator;
    if (!isfinite(result)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    output[row * s.x * s.z + head * s.z + channel] = result;
}
