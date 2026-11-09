#include <metal_stdlib>
using namespace metal;

// Identical ABI to GdnChunkParams. This isolated recurrence accepts the current chunk builder's stored FP32 log decay, not a recomputed alpha/bias projection.
struct GdnLegacyBatchParams {
    uint rows, columns, outputs, heads;
    uint key_heads, key_width, value_width, mode;
    uint offset, count, l2_mode, reserved;
    ulong input_stride, output_stride, auxiliary_stride, gate_stride;
    float query_scale, l2_epsilon, rms_epsilon, reserved_float;
};
static_assert(sizeof(GdnLegacyBatchParams) == 96, "legacy-order GDN batch ABI");

inline device const float* legacy_gdn_row(device const float* p, uint row, ulong stride) {
    return (device const float*)((device const char*)p + ulong(row) * stride);
}
inline device float* legacy_gdn_row(device float* p, uint row, ulong stride) {
    return (device float*)((device char*)p + ulong(row) * stride);
}

// One thread owns a value column exactly as graph_delta and graph_gdn_delta_tile. Retain their sequential key sums and updated=a*state+(b*key)*delta expression, including the normal Metal3.2 -fno-fast-math contraction policy. No SIMD key reduction or beta reassociation.
kernel void gdn_chunk_legacy_batch(device const float* accepted [[buffer(0)]],
                                   device const float* query [[buffer(1)]],
                                   device const float* key [[buffer(2)]],
                                   device const float* value [[buffer(3)]],
                                   device const float* log_decay [[buffer(4)]],
                                   device const float* beta [[buffer(5)]],
                                   device float* candidate [[buffer(6)]],
                                   device atomic_uint* error [[buffer(7)]],
                                   device float* output [[buffer(8)]],
                                   constant GdnLegacyBatchParams& p [[buffer(14)]],
                                   uint i [[thread_position_in_grid]]) {
    if (!p.rows || p.rows > 512 || !p.heads || p.heads > 48 || p.key_width != 128 || !p.value_width || p.value_width > 128) {
        if (!i) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
        return;
    }
    if (i >= p.heads * p.value_width) return;
    const uint head = i / p.value_width, column = i % p.value_width;
    const uint base = head * 128 * p.value_width + column;
    float state[128];
    for (uint k = 0; k < 128; ++k) {
        state[k] = accepted[base + k * p.value_width];
        if (!isfinite(state[k])) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    }
    for (uint step = 0; step < p.rows; ++step) {
        const float exponent = legacy_gdn_row(log_decay, step, p.gate_stride)[head];
        const float b = legacy_gdn_row(beta, step, p.gate_stride)[head];
        if (!isfinite(exponent) || exponent > 0 || !isfinite(b) || b < 0 || b > 1) {
            atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return;
        }
        const float a = exp(exponent);
        const auto qrow = legacy_gdn_row(query, step, p.input_stride) + head * 128;
        const auto krow = legacy_gdn_row(key, step, p.input_stride) + head * 128;
        float predicted = 0;
        for (uint k = 0; k < 128; ++k) predicted += (a * state[k]) * krow[k];
        const float delta = legacy_gdn_row(value, step, p.auxiliary_stride)[i] - predicted;
        float result = 0;
        for (uint k = 0; k < 128; ++k) {
            const float updated = a * state[k] + b * krow[k] * delta;
            state[k] = updated;
            result += updated * qrow[k];
            if (!isfinite(updated)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        }
        if (!isfinite(result)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        legacy_gdn_row(output, step, p.output_stride)[i] = result;
    }
    for (uint k = 0; k < 128; ++k) candidate[base + k * p.value_width] = state[k];
}
