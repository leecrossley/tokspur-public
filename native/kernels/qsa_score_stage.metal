#include <metal_stdlib>
using namespace metal;

// Original composition of the owned ordinary QSA dot and online recurrence.
// No score/softmax partition sums, altered reductions, exp specialization or
// new numerical tier. Generated compiled byte-equivalence is a mandatory gate.
float qsa_stage_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
bool qsa_stage_token(uint item, device const uint* ids, uint4 s, uint4 p,
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

// Four independent attended items/TG, one SIMD group per(head,item).
kernel void qsa_score_stage(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
                            device const uint* ids [[buffer(2)]], device float* scores [[buffer(3)]],
                            constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]],
                            device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
                            ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group.y, item = group.x * 4 + subgroup;
    const uint blocks = p.x / s.w, tokens = min(blocks, p.y) * s.w + p.x % s.w;
    if (head >= s.x || item >= tokens) return;
    uint token;
    if (!qsa_stage_token(item, ids, s, p, token, error)) return;
    const uint key_head = head / (s.x / s.y), row_width = s.y * s.z;
    const uint base = token * row_width + key_head * s.z;
    float dot = 0;
    for (uint d = lane; d < s.z; d += 32) dot += query[head * s.z + d] * qsa_stage_bf16(keys[base + d]);
    const float score = simd_sum(dot) * rsqrt(float(s.z));
    if (!isfinite(score)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    if (lane == 0) scores[head * tokens + item] = score;
}

#ifndef TOKSPUR_QSA_STAGE_OMIT_ACCUMULATE
// One SIMD group/head, exactly the ordinary chronological recurrence and divide.
kernel void qsa_score_accumulate(device const float* scores [[buffer(0)]], device const ushort* values [[buffer(1)]],
                                 device const uint* ids [[buffer(2)]], device float* output [[buffer(3)]],
                                 constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]],
                                 device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
                                 ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group * 4 + subgroup;
    if (head >= s.x) return;
    // The graph's inter-dispatch buffer barrier makes pass1's error visible.
    // Never read an unwritten score after malformed IDs/nonfinite first-pass input.
    if (atomic_load_explicit(error, memory_order_relaxed)) return;
    const uint key_head = head / (s.x / s.y), row_width = s.y * s.z, blocks = p.x / s.w;
    const uint selected = min(blocks, p.y), tail = p.x % s.w, tokens = selected * s.w + tail;
    float maximum = -INFINITY, denominator = 0, sum[8];
    for (uint c = 0; c < 8; ++c) sum[c] = 0;
    for (uint item = 0; item < tokens; ++item) {
        uint token;
        if (!qsa_stage_token(item, ids, s, p, token, error)) return;
        const uint base = token * row_width + key_head * s.z;
        const float score = scores[head * tokens + item];
        if (!isfinite(score)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        const float next_max = max(maximum, score), old_scale = exp(maximum - next_max), scale = exp(score - next_max);
        denominator = denominator * old_scale + scale;
        for (uint c = 0, d = lane; d < s.z; ++c, d += 32) {
            const float value = qsa_stage_bf16(values[base + d]);
            if (!isfinite(value)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
            sum[c] = sum[c] * old_scale + scale * value;
        }
        maximum = next_max;
    }
    for (uint c = 0, d = lane; d < s.z; ++c, d += 32) {
        const float value = sum[c] / denominator;
        if (!isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[head * s.z + d] = value;
    }
}
#endif
