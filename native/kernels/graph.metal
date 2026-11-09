#include <metal_stdlib>
using namespace metal;

float sigmoid_value(float x) {
    if (x >= 0) return 1.0f / (1.0f + exp(-x));
    const float e = exp(x);
    return e / (1.0f + e);
}

float softplus_value(float x) {
    const float z = exp(-abs(x));
    const float sum = 1.0f + z;
    // Metal has no scalar log1p. Compensate rounding of 1+z; retain tiny tails
    // when the sum rounds to 1 rather than flushing a negative gate to zero.
    return max(x, 0.0f) + (sum == 1.0f ? z : log(sum) * (z / (sum - 1.0f)));
}

void checked_store(device float* output, uint index, float value, device atomic_uint* error) {
    if (!isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    output[index] = value;
}

// The GGUF value-head permutation is ratio-major. Normalize Q/K once per
// expanded head; all three results retain that stored head order.
kernel void graph_gdn_prepare(device const float* input [[buffer(0)]],
                              device float* query [[buffer(1)]],
                              device float* key [[buffer(2)]],
                              device float* value [[buffer(3)]],
                              constant uint4& s [[buffer(4)]],
                              constant float2& p [[buffer(5)]],
                              constant uint& mode [[buffer(6)]],
                              device atomic_uint* error [[buffer(7)]],
                              uint group [[threadgroup_position_in_grid]],
                              ushort subgroup [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group * 4 + subgroup;
    if (head >= s.y) return;
    const uint block = s.x * s.z, base = (head % s.x) * s.z;
    float qs = 0, ks = 0;
    for (uint i = lane; i < s.z; i += 32) {
        qs += input[base + i] * input[base + i];
        ks += input[block + base + i] * input[block + base + i];
    }
    qs = simd_sum(qs); ks = simd_sum(ks);
    const float qd = mode ? max(qs, p.y) : qs + p.y;
    const float kd = mode ? max(ks, p.y) : ks + p.y;
    if (!isfinite(qs) || !isfinite(ks) || !isfinite(qd) || !isfinite(kd)) {
        if (lane == 0) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    const float qi = rsqrt(qd), ki = rsqrt(kd);
    for (uint i = lane; i < s.z; i += 32) {
        checked_store(query, head * s.z + i, (input[base + i] * qi) * p.x, error);
        checked_store(key, head * s.z + i, input[block + base + i] * ki, error);
    }
    for (uint i = lane; i < s.w; i += 32) checked_store(value, head * s.w + i, input[2 * block + head * s.w + i], error);
}

kernel void graph_gdn_gates(device const float* alpha [[buffer(0)]],
                            device const float* beta_logits [[buffer(1)]],
                            device const float* rates [[buffer(2)]],
                            device const float* bias [[buffer(3)]],
                            device float* decay [[buffer(4)]],
                            device float* beta [[buffer(5)]],
                            constant uint& heads [[buffer(6)]],
                            device atomic_uint* error [[buffer(7)]],
                            uint h [[thread_position_in_grid]]) {
    if (h >= heads) return;
    const float x = alpha[h] + bias[h];
    if (!isfinite(alpha[h]) || !isfinite(bias[h]) || !isfinite(x) || !isfinite(beta_logits[h]) ||
        !isfinite(rates[h]) || rates[h] > 0) { atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return; }
    const float exponent = rates[h] * softplus_value(x);
    if (!isfinite(exponent)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    checked_store(decay, h, exp(exponent), error);
    checked_store(beta, h, sigmoid_value(beta_logits[h]), error);
}

// Four independent RMS groups per threadgroup. Weights may be shared across
// groups or supplied per channel; checkpoint +1 transforms are NOT implicit.
kernel void graph_rms(device const float* input [[buffer(0)]],
                      device const float* gains [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant uint4& shape [[buffer(3)]],
                      constant float& epsilon [[buffer(4)]],
                      device atomic_uint* error [[buffer(7)]],
                      uint group [[threadgroup_position_in_grid]],
                      ushort subgroup [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    const uint block = group * 4 + subgroup;
    if (block >= shape.x / shape.y) return;
    const uint base = block * shape.y;
    float squared = 0;
    for (uint i = lane; i < shape.y; i += 32) squared += input[base + i] * input[base + i];
    const float total = simd_sum(squared);
    const float denominator = total / float(shape.y) + epsilon;
    if (!isfinite(total) || !isfinite(denominator)) {
        if (lane == 0) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    const float inverse = rsqrt(denominator);
    for (uint i = lane; i < shape.y; i += 32) {
        checked_store(output, base + i, input[base + i] * inverse * gains[(shape.z ? 0 : base) + i], error);
    }
}

kernel void graph_activate(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint2& shape [[buffer(2)]],
                           constant float& scale [[buffer(3)]],
                           device atomic_uint* error [[buffer(7)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= shape.x) return;
    const float x = input[i] * scale;
    // A bounded activation must not hide a bad upstream value by mapping an
    // infinity to 0 or 1. Refuse the batch, even if its eventual output is finite.
    if (!isfinite(x)) { checked_store(output, i, x, error); return; }
    float value = x;
    if (shape.y == 1) value = x * sigmoid_value(x);
    else if (shape.y == 2) value = sigmoid_value(x);
    else if (shape.y == 3) value = softplus_value(x);
    checked_store(output, i, value, error);
}

kernel void graph_multiply(device const float* input [[buffer(0)]],
                           device const float* gate [[buffer(1)]],
                           device float* output [[buffer(2)]],
                           constant uint2& shape [[buffer(3)]],
                           device atomic_uint* error [[buffer(7)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= shape.x) return;
    if (!isfinite(input[i]) || !isfinite(gate[i])) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    checked_store(output, i, input[i] * (shape.y ? sigmoid_value(gate[i]) : gate[i]), error);
}

// Fused SwiGLU: exact composition of graph_activate(silu) and graph_multiply(sigmoid_gate=false).
kernel void graph_swiglu(device const float* gate [[buffer(0)]],
                         device const float* up [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         constant float& scale [[buffer(4)]],
                         device atomic_uint* error [[buffer(7)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float x = gate[i] * scale;
    const float u = up[i];
    if (!isfinite(x)) { checked_store(output, i, x, error); return; }
    const float activated = x * sigmoid_value(x);
    if (!isfinite(activated) || !isfinite(u)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    checked_store(output, i, activated * u, error);
}

kernel void graph_residual_read(device const float* normalized [[buffer(0)]],
                                device const float* scores [[buffer(1)]],
                                device const float* inject [[buffer(2)]],
                                device float* output [[buffer(3)]],
                                device float* scales [[buffer(4)]],
                                constant uint2& shape [[buffer(5)]],
                                device atomic_uint* error [[buffer(7)]],
                                uint channel [[thread_position_in_grid]]) {
    if (channel < shape.y) {
        float total = 0;
        for (uint branch = 0; branch < shape.x; ++branch) {
            const uint index = branch * shape.y + channel;
            if (!isfinite(normalized[index]) || !isfinite(scores[index])) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
            total += normalized[index] * sigmoid_value(scores[index]);
        }
        checked_store(output, channel, total / float(shape.x), error);
    }
    if (channel < shape.x) {
        if (!isfinite(inject[channel])) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        checked_store(scales, channel, 2.0f * sigmoid_value(inject[channel] / float(shape.x)), error);
    }
}

kernel void graph_residual_write(device const float* streams [[buffer(0)]],
                                 device const float* input [[buffer(1)]],
                                 device const float* scales [[buffer(2)]],
                                 device float* output [[buffer(3)]],
                                 constant uint2& shape [[buffer(4)]],
                                 device atomic_uint* error [[buffer(7)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i < shape.x * shape.y) checked_store(output, i, streams[i] + scales[i / shape.y] * input[i % shape.y], error);
}

// One thread owns a channel across all time taps, including the history shift.
// Both histories are separate, so live state isn't mutated before publication.
kernel void graph_convolution(device const float* history [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device const float* weights [[buffer(2)]],
                              device float* next_history [[buffer(3)]],
                              device float* output [[buffer(4)]],
                              constant uint4& shape [[buffer(5)]],
                              device atomic_uint* error [[buffer(7)]],
                              uint channel [[thread_position_in_grid]]) {
    if (channel >= shape.x) return;
    if (!isfinite(input[channel])) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    float result = input[channel] * weights[channel * shape.y + shape.y - 1];
    for (uint t = 0; t + 1 < shape.y; ++t) {
        result += history[t * shape.z * shape.x + channel] * weights[channel * shape.y + t];
    }
    const uint depth = (shape.y - 1) * shape.z;
    for (uint t = 0; t < depth; ++t) {
        if (!isfinite(history[t * shape.x + channel])) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        next_history[t * shape.x + channel] = t + 1 < depth ? history[(t + 1) * shape.x + channel] : input[channel];
    }
    checked_store(output, channel, result, error);
}

kernel void graph_ple_gate(device const float* query [[buffer(0)]], device const float* key [[buffer(1)]],
                           device const float* value [[buffer(2)]], device float* gated [[buffer(3)]], device float* gates [[buffer(4)]],
                           constant uint4& shape [[buffer(5)]], device atomic_uint* error [[buffer(7)]],
                           uint group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint branch = group * 4 + subgroup;
    if (shape.z > 1) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    if (branch >= shape.x) return;
    float dot = 0;
    for (uint c = lane; c < shape.y; c += 32) dot += query[branch * shape.y + c] * key[branch * shape.y + c];
    float score = simd_sum(dot) * rsqrt(float(shape.y));
    if (!isfinite(score)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    if (shape.z && score != 0.0f)
        score = copysign(sqrt(max(abs(score), 1e-6f)), score);
    const float gate = sigmoid_value(score);
    if (lane == 0) gates[branch] = gate;
    for (uint c = lane; c < shape.y; c += 32) checked_store(gated, branch * shape.y + c, gate * value[c], error);
}
kernel void graph_ple_write(device const float* streams [[buffer(0)]], device const float* gated [[buffer(1)]],
                            device const float* convolved [[buffer(2)]], device float* output [[buffer(3)]], constant uint& count [[buffer(4)]],
                            device atomic_uint* error [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float x = convolved[i];
    if (!isfinite(x) || !isfinite(streams[i]) || !isfinite(gated[i])) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    checked_store(output, i, streams[i] + (gated[i] + x * sigmoid_value(x)), error);
}
kernel void graph_gated_mix(device const float* normalized [[buffer(0)]], device const float* scores [[buffer(1)]],
                            device float* output [[buffer(2)]], constant uint2& shape [[buffer(3)]], device atomic_uint* error [[buffer(7)]], uint c [[thread_position_in_grid]]) {
    if (c >= shape.y) return;
    float result = 0;
    for (uint b = 0; b < shape.x; ++b) {
        const uint at = b * shape.y + c;
        if (!isfinite(normalized[at]) || !isfinite(scores[at])) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        result += normalized[at] * sigmoid_value(scores[at]);
    }
    checked_store(output, c, result / float(shape.x), error);
}

// State is [head,key,value]. Adjacent threads cover values, coalescing each
// state row. Read and next state are distinct, permitting explicit publication.
kernel void graph_delta(device const float* state [[buffer(0)]],
                        device const float* query [[buffer(1)]],
                        device const float* key [[buffer(2)]],
                        device const float* value [[buffer(3)]],
                        device const float* decay [[buffer(4)]],
                        device const float* beta [[buffer(5)]],
                        device float* next_state [[buffer(6)]],
                        device atomic_uint* error [[buffer(7)]],
                        device float* output [[buffer(8)]],
                        constant uint4& shape [[buffer(9)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= shape.x * shape.z) return;
    const uint head = i / shape.z;
    const uint column = i % shape.z;
    const float a = decay[head], b = beta[head];
    if (!isfinite(a) || !isfinite(b) || a < 0 || a > 1 || b < 0 || b > 1) {
        atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    const uint base = head * shape.y * shape.z + column;
    float predicted = 0;
    for (uint k = 0; k < shape.y; ++k) predicted += (a * state[base + k * shape.z]) * key[head * shape.y + k];
    const float delta = value[i] - predicted;
    float result = 0;
    for (uint k = 0; k < shape.y; ++k) {
        const uint index = base + k * shape.z;
        const float updated = a * state[index] + b * key[head * shape.y + k] * delta;
        next_state[index] = updated;
        result += updated * query[head * shape.y + k];
    }
    checked_store(output, i, result, error);
}

// One 128-thread group owns a router. Four SIMD reductions select each maximum;
// exact ties use the smaller original expert ID. The original logits remain
// unchanged for full-softmax denominators and for checking independent results.
float route_activation(float logit, float maximum, uint mode, uint normalize) {
    if (!mode) return exp(logit - maximum);
    if (normalize) return exp(-softplus_value(-logit) + softplus_value(-maximum));
    return sigmoid_value(logit);
}

kernel void graph_route(device const float* logits [[buffer(0)]],
                        device uint* indices [[buffer(1)]],
                        device float* weights [[buffer(2)]],
                        constant uint4& shape [[buffer(3)]],
                        device atomic_uint* error [[buffer(7)]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort subgroup [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float candidates[4096];
    threadgroup float chosen_logits[64];
    threadgroup float maxima[4];
    threadgroup uint maximum_ids[4];
    threadgroup float denominator;
    for (uint i = tid; i < shape.x; i += 128) {
        const float value = logits[i];
        if (!isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        candidates[i] = isfinite(value) ? value : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint selected = 0; selected < shape.y; ++selected) {
        float best = -INFINITY;
        uint expert = 0xffffffffu;
        for (uint i = tid; i < shape.x; i += 128) {
            if (candidates[i] > best || (candidates[i] == best && i < expert)) { best = candidates[i]; expert = i; }
        }
        const float simd_best = simd_max(best);
        const uint simd_id = simd_min(best == simd_best ? expert : 0xffffffffu);
        if (lane == 0) { maxima[subgroup] = simd_best; maximum_ids[subgroup] = simd_id; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float score = -INFINITY;
            uint chosen = 0xffffffffu;
            for (uint s = 0; s < 4; ++s) {
                if (maxima[s] > score || (maxima[s] == score && maximum_ids[s] < chosen)) {
                    score = maxima[s]; chosen = maximum_ids[s];
                }
            }
            if (chosen >= shape.x) { atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); chosen = 0; }
            indices[selected] = chosen;
            chosen_logits[selected] = logits[chosen];
            candidates[chosen] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float sum = shape.z ? 1.0f : 0.0f;
        if (shape.w) {
            sum = 0;
            for (uint i = 0; i < shape.y; ++i) sum += route_activation(chosen_logits[i], chosen_logits[0], shape.z, shape.w);
        } else if (!shape.z) {
            for (uint i = 0; i < shape.x; ++i) sum += exp(logits[i] - chosen_logits[0]);
        }
        if (!isfinite(sum) || sum <= 0) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); sum = 1; }
        denominator = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < shape.y) {
        const float value = route_activation(chosen_logits[tid], chosen_logits[0], shape.z, shape.w);
        checked_store(weights, tid, value / denominator, error);
    }
}

kernel void graph_mixture(device const float* experts [[buffer(0)]],
                          device const float* weights [[buffer(1)]],
                          device const float* shared [[buffer(2)]],
                          device const float* shared_logit [[buffer(3)]],
                          device float* output [[buffer(4)]],
                          constant uint2& shape [[buffer(5)]],
                          device atomic_uint* error [[buffer(7)]],
                          uint channel [[thread_position_in_grid]]) {
    if (channel >= shape.x) return;
    if (!isfinite(shared_logit[0])) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    float value = shared[channel] * sigmoid_value(shared_logit[0]);
    for (uint route = 0; route < shape.y; ++route) {
        if (!isfinite(weights[route]) || weights[route] < 0) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        value += weights[route] * experts[route * shape.x + channel];
    }
    checked_store(output, channel, value, error);
}
