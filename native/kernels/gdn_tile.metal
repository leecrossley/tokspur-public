#include <metal_stdlib>
using namespace metal;

bool gdn_tile_count(uint capacity, uint steps) {
    return (capacity == 2 || capacity == 4 || capacity == 8 || capacity == 16 || capacity == 32
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        || capacity == 512
#endif
        ) && steps && steps <= capacity;
}

// Actual GDN convolution only: kernel four, dilation one. A channel owns its
// three chronological histories and four coefficients. No state publication.
kernel void graph_gdn_convolution_tile(device const float* history [[buffer(0)]],
                                       device const float* projected [[buffer(1)]],
                                       device const float* weights [[buffer(2)]],
                                       device float* next_history [[buffer(3)]],
                                       device float* output [[buffer(4)]],
                                       constant uint4& s [[buffer(5)]],
                                       device atomic_uint* error [[buffer(7)]],
                                       uint channel [[thread_position_in_grid]]) {
    if (!s.x || s.x > 65536 || !gdn_tile_count(s.y, s.z)) {
        if (!channel) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return;
    }
    if (channel >= s.x) return;
    float h0 = history[channel], h1 = history[s.x + channel], h2 = history[2 * s.x + channel];
    const float w0 = weights[4 * channel], w1 = weights[4 * channel + 1], w2 = weights[4 * channel + 2], w3 = weights[4 * channel + 3];
    if (!isfinite(h0) || !isfinite(h1) || !isfinite(h2) || !isfinite(w0) || !isfinite(w1) || !isfinite(w2) || !isfinite(w3)) {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return;
    }
    for (uint step = 0; step < s.z; ++step) {
        const float x = projected[ulong(step) * s.x + channel];
        if (!isfinite(x)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        // Same current-product-first sequence as graph_convolution. Do not
        // rearrange tap order or fuse the following SiLU/normalization stages.
        float result = x * w3;
        result += h0 * w0; result += h1 * w1; result += h2 * w2;
        if (!isfinite(result)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        output[ulong(step) * s.x + channel] = result;
        h0 = h1; h1 = h2; h2 = x;
    }
    next_history[channel] = h0; next_history[s.x + channel] = h1; next_history[2 * s.x + channel] = h2;
}

// Fixed key width 128; one thread per (head,value column), with coalesced value
// lanes exactly as graph_delta. Private-array register residency is an INTENT,
// not a hardware guarantee: compiler spills/occupancy need separate evidence.
kernel void graph_gdn_delta_tile(device const float* accepted [[buffer(0)]],
                                device const float* query [[buffer(1)]],
                                device const float* key [[buffer(2)]],
                                device const float* value [[buffer(3)]],
                                device const float* decay [[buffer(4)]],
                                device const float* beta [[buffer(5)]],
                                device float* candidate [[buffer(6)]],
                                device atomic_uint* error [[buffer(7)]],
                                device float* output [[buffer(8)]],
                                constant uint4& s [[buffer(9)]],
                                constant uint& steps [[buffer(10)]],
                                uint i [[thread_position_in_grid]]) {
    // s={heads,value_width,capacity,128}; steps may be a strict prefix for replay.
    if (!s.x || s.x > 48 || !s.y || s.y > 128 || s.w != 128 || !gdn_tile_count(s.z, steps)) {
        if (!i) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return;
    }
    if (i >= s.x * s.y) return;
    const uint head = i / s.y, column = i % s.y, base = head * 128 * s.y + column;
    float state[128];
    for (uint k = 0; k < 128; ++k) {
        state[k] = accepted[base + k * s.y];
        if (!isfinite(state[k])) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    }
    for (uint step = 0; step < steps; ++step) {
        const uint gate = step * s.x + head, qbase = (step * s.x + head) * 128;
        const float a = decay[gate], b = beta[gate];
        if (!isfinite(a) || !isfinite(b) || a < 0 || a > 1 || b < 0 || b > 1) {
            atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return;
        }
        float predicted = 0;
        for (uint k = 0; k < 128; ++k) predicted += (a * state[k]) * key[qbase + k];
        const float delta = value[ulong(step) * s.x * s.y + i] - predicted;
        float result = 0;
        for (uint k = 0; k < 128; ++k) {
            // Keep the old expression and order. Pre-decaying state, FMA
            // substitutions or a parallel key reduction are different tiers.
            const float updated = a * state[k] + b * key[qbase + k] * delta;
            state[k] = updated; result += updated * query[qbase + k];
            if (!isfinite(updated)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        }
        if (!isfinite(result)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        output[ulong(step) * s.x * s.y + i] = result;
    }
    for (uint k = 0; k < 128; ++k) candidate[base + k * s.y] = state[k];
}
