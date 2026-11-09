// Batch independent token rows while retaining the existing split16 and residual arithmetic.
#include <metal_stdlib>
using namespace metal;
float sigmoid_value(float x);
void checked_store(device float* output, uint index, float value, device atomic_uint* error);

inline uint nonfin_hc(float x) { return (as_type<uint>(x) & 0x7f800000u) == 0x7f800000u ? 1u : 0u; }

kernel void graph_hc_rows_inject_partials(device const float* weights [[buffer(0)]], device const float* input [[buffer(1)]], device float* partials [[buffer(2)]], constant uint4& shape [[buffer(3)]], device atomic_uint* error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]], ushort row [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint tok0 = group.z * 16u;
    if (row >= shape.y || tok0 >= shape.w) return;
    const uint nt = min(16u, shape.w - tok0);
    const uint width = ((shape.x + shape.z * 32u - 1u) / (shape.z * 32u)) * 32u;
    const uint begin = group.y * width, end = min(begin + width, shape.x);
    device const float* w_row = weights + ulong(row) * shape.x;
    device const float* x_base = input + ulong(tok0) * shape.x;
    float sums[16] = {0.0f};
    if (nt == 16u) {
        for (uint column = begin + lane; column < end; column += 32u) {
            const float w = w_row[column];
            device const float* x_col = x_base + column;
            #pragma unroll
            for (uint m = 0u; m < 16u; ++m) sums[m] += w * x_col[ulong(m) * shape.x];
        }
    } else {
        for (uint column = begin + lane; column < end; column += 32u) {
            const float w = w_row[column];
            device const float* x_col = x_base + column;
            for (uint m = 0u; m < nt; ++m) sums[m] += w * x_col[ulong(m) * shape.x];
        }
    }
    for (uint m = 0u; m < nt; ++m) {
        const float total = simd_sum(sums[m]);
        if (!lane) checked_store(partials, ((tok0 + m) * shape.y + row) * shape.z + group.y, total, error);
    }
}
kernel void graph_hc_rows_inject_partials_nax(device const float* weights [[buffer(0)]], device const float* input [[buffer(1)]], device float* partials [[buffer(2)]], constant uint4& shape [[buffer(3)]], device atomic_uint* error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]], ushort row [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint tok0 = group.z * 16u;
    const uint width = ((shape.x + shape.z * 32u - 1u) / (shape.z * 32u)) * 32u;
    const uint begin = group.y * width, end = min(begin + width, shape.x);
    if (shape.y == 4u) {
        const uint my_tok0 = tok0 + uint(row) * 4u;
        if (row >= 4u || my_tok0 >= shape.w) return;
        const uint my_nt = min(4u, shape.w - my_tok0);
        device const float* w0 = weights;
        device const float* w1 = w0 + shape.x;
        device const float* w2 = w1 + shape.x;
        device const float* w3 = w2 + shape.x;
        device const float* x_base = input + ulong(my_tok0) * shape.x;
        float4 s0(0.0f), s1(0.0f), s2(0.0f), s3(0.0f);
        if (my_nt == 4u) {
            for (uint column = begin + lane; column < end; column += 32u) {
                const float4 wv(w0[column], w1[column], w2[column], w3[column]);
                device const float* x_col = x_base + column;
                s0 += wv * x_col[0ul * shape.x];
                s1 += wv * x_col[1ul * shape.x];
                s2 += wv * x_col[2ul * shape.x];
                s3 += wv * x_col[3ul * shape.x];
            }
        } else {
            for (uint column = begin + lane; column < end; column += 32u) {
                const float4 wv(w0[column], w1[column], w2[column], w3[column]);
                device const float* x_col = x_base + column;
                s0 += wv * x_col[0ul * shape.x];
                if (my_nt > 1u) s1 += wv * x_col[1ul * shape.x];
                if (my_nt > 2u) s2 += wv * x_col[2ul * shape.x];
            }
        }
        const auto emit4 = [&](uint m, float4 sm) {
            const float t0 = simd_sum(sm.x), t1 = simd_sum(sm.y), t2 = simd_sum(sm.z), t3 = simd_sum(sm.w);
            if (!lane) {
                const uint out_base = (my_tok0 + m) * 4u * shape.z + group.y;
                checked_store(partials, out_base + 0u * shape.z, t0, error);
                checked_store(partials, out_base + 1u * shape.z, t1, error);
                checked_store(partials, out_base + 2u * shape.z, t2, error);
                checked_store(partials, out_base + 3u * shape.z, t3, error);
            }
        };
        emit4(0u, s0);
        if (my_nt > 1u) emit4(1u, s1);
        if (my_nt > 2u) emit4(2u, s2);
        if (my_nt > 3u) emit4(3u, s3);
        return;
    }
    if (row >= shape.y || tok0 >= shape.w) return;
    const uint nt = min(16u, shape.w - tok0);
    device const float* w_row = weights + ulong(row) * shape.x;
    device const float* x_base = input + ulong(tok0) * shape.x;
    float sums[16] = {0.0f};
    if (nt == 16u) {
        for (uint column = begin + lane; column < end; column += 32u) {
            const float w = w_row[column];
            device const float* x_col = x_base + column;
            #pragma unroll
            for (uint m = 0u; m < 16u; ++m) sums[m] += w * x_col[ulong(m) * shape.x];
        }
    } else {
        for (uint column = begin + lane; column < end; column += 32u) {
            const float w = w_row[column];
            device const float* x_col = x_base + column;
            for (uint m = 0u; m < nt; ++m) sums[m] += w * x_col[ulong(m) * shape.x];
        }
    }
    for (uint m = 0u; m < nt; ++m) {
        const float total = simd_sum(sums[m]);
        if (!lane) checked_store(partials, ((tok0 + m) * shape.y + row) * shape.z + group.y, total, error);
    }
}
kernel void graph_hc_rows_inject_reduce(device const float* partials [[buffer(0)]], device float* output [[buffer(1)]], constant uint4& shape [[buffer(2)]], device atomic_uint* error [[buffer(7)]], uint row [[thread_position_in_grid]]) {
    if (row >= shape.w * shape.y) return;
    float sum = 0.0f;
    if (shape.z == 16u) {
        device const packed_float4* p4 = reinterpret_cast<device const packed_float4*>(partials + ulong(row) * 16ul);
        #pragma unroll
        for (uint q = 0u; q < 4u; ++q) {
            const float4 v = float4(p4[q]);
            sum = (((sum + v.x) + v.y) + v.z) + v.w;
        }
    } else {
        for (uint i = 0u; i < shape.z; ++i) sum += partials[row * shape.z + i];
    }
    checked_store(output, row, sum, error);
}
kernel void graph_hc_rows_read(device const float* normalized [[buffer(0)]], device const float* scores [[buffer(1)]], device const float* inject [[buffer(2)]], device float* output [[buffer(3)]], device float* scales [[buffer(4)]], constant uint4& shape [[buffer(5)]], device atomic_uint* error [[buffer(7)]], uint index [[thread_position_in_grid]]) {
    const uint quads = shape.y >> 2u;
    const uint token = index / quads, col4 = (index % quads) << 2u;
    if (token >= shape.z) return;
    float4 total = float4(0.0f);
    bool4 ok = bool4(true);
    for (uint branch = 0u; branch < shape.x; ++branch) {
        const ulong at = (ulong(token) * shape.x + branch) * shape.y + col4;
        const float4 n = float4(*reinterpret_cast<device const packed_float4*>(normalized + at));
        const float4 s = float4(*reinterpret_cast<device const packed_float4*>(scores + at));
        ok = ok && isfinite(n) && isfinite(s);
        total.x += n.x * sigmoid_value(s.x);
        total.y += n.y * sigmoid_value(s.y);
        total.z += n.z * sigmoid_value(s.z);
        total.w += n.w * sigmoid_value(s.w);
    }
    const float4 out4 = total / float(shape.x);
    if (all(ok) && all(isfinite(out4))) {
        *reinterpret_cast<device packed_float4*>(output + ulong(index) * 4ul) = packed_float4(out4);
    } else {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        for (uint k = 0u; k < 4u; ++k) {
            if (ok[k]) checked_store(output, index * 4u + k, out4[k], error);
        }
    }
    if (col4 < shape.x) {
        if (col4 == 0u && shape.x == 4u && all(ok)) {
            const float4 inj = float4(*reinterpret_cast<device const packed_float4*>(inject + ulong(token) * 4ul));
            if (all(isfinite(inj))) {
                const float4 sc4(2.0f * sigmoid_value(inj.x / 4.0f),
                                 2.0f * sigmoid_value(inj.y / 4.0f),
                                 2.0f * sigmoid_value(inj.z / 4.0f),
                                 2.0f * sigmoid_value(inj.w / 4.0f));
                if (!all(isfinite(sc4))) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                *reinterpret_cast<device packed_float4*>(scales + ulong(token) * 4ul) = packed_float4(sc4);
                return;
            }
        }
        for (uint k = 0u; k < 4u && col4 + k < shape.x; ++k) {
            if (!ok[k]) continue;
            const uint at = token * shape.x + col4 + k;
            if (!isfinite(inject[at])) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); continue; }
            checked_store(scales, at, 2.0f * sigmoid_value(inject[at] / float(shape.x)), error);
        }
    }
}
kernel void graph_hc_rows_write(device const float* streams [[buffer(0)]], device const float* input [[buffer(1)]], device const float* scales [[buffer(2)]], device float* output [[buffer(3)]], constant uint4& shape [[buffer(4)]], device atomic_uint* error [[buffer(7)]], uint index [[thread_position_in_grid]]) {
    const uint ch_quads = shape.y >> 2u;
    const uint width_quads = shape.x * ch_quads;
    const uint token = index / width_quads, rem = index % width_quads;
    if (token >= shape.z) return;
    const uint branch = rem / ch_quads, col4 = (rem % ch_quads) << 2u;
    const float4 st = float4(*reinterpret_cast<device const packed_float4*>(streams + ulong(index) * 4ul));
    const float4 xv = float4(*reinterpret_cast<device const packed_float4*>(input + ulong(token) * shape.y + col4));
    const float sc = scales[ulong(token) * shape.x + branch];
    const float4 out4 = st + sc * xv;
    if (!all(isfinite(out4))) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    *reinterpret_cast<device packed_float4*>(output + ulong(index) * 4ul) = packed_float4(out4);
}
kernel void graph_hc_rows_write_rms(device const float* streams [[buffer(0)]], device const float* input [[buffer(1)]],
    device const float* scales [[buffer(2)]], device const float* gains [[buffer(3)]], device float* output [[buffer(4)]],
    device float* normalized [[buffer(5)]], constant uint4& shape [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    if (group >= shape.z || sg >= shape.x) return;
    const uint token = group, branch = sg, channels = shape.y;
    const ulong base = (ulong(token) * shape.x + branch) * channels;
    const float scale = scales[ulong(token) * shape.x + branch];
    float sum = 0.0f;
    if (channels == 2560u) {
        float rv[80];
        uint bad = 0u;
        #pragma unroll 4
        for (uint k = 0u; k < 80u; ++k) {
            const uint j = uint(lane) + k * 32u;
            const float v = streams[base + j] + scale * input[ulong(token) * 2560ul + j];
            if (!isfinite(v)) bad = 1u;
            output[base + j] = v;
            rv[k] = v;
            sum += v * v;
        }
        sum = simd_sum(sum);
        const float denominator = sum / 2560.0f + as_type<float>(shape.w);
        if (simd_or(bad) || !isfinite(sum) || !isfinite(denominator)) {
            if (!lane) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            return;
        }
        const float inv = rsqrt(denominator);
        uint bad2 = 0u;
        #pragma unroll 4
        for (uint k = 0u; k < 80u; ++k) {
            const uint j = uint(lane) + k * 32u;
            const float g = gains[ulong(branch) * 2560ul + j];
            const float n = (rv[k] * inv) * g;
            if (!isfinite(g) || !isfinite(n)) bad2 = 1u;
            normalized[base + j] = n;
        }
        if (simd_or(bad2) && !lane) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    for (uint j = lane; j < channels; j += 32) {
        const float v = streams[base + j] + scale * input[ulong(token) * channels + j];
        checked_store(output, base + j, v, error);
        sum += v * v;
    }
    sum = simd_sum(sum);
    const float denominator = sum / float(channels) + as_type<float>(shape.w);
    if (!isfinite(sum) || !isfinite(denominator)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    const float inv = rsqrt(denominator);
    for (uint j = lane; j < channels; j += 32) {
        const float g = gains[branch * channels + j];
        if (!isfinite(g)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        checked_store(normalized, base + j, (output[base + j] * inv) * g, error);
    }
}
kernel void graph_hc_rows_write_rms_nax(device const float* streams [[buffer(0)]], device const float* input [[buffer(1)]],
    device const float* scales [[buffer(2)]], device const float* gains [[buffer(3)]], device float* output [[buffer(4)]],
    device float* normalized [[buffer(5)]], constant uint4& shape [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup float v_tile[80][32];
    threadgroup float shared_inv;
    threadgroup uint shared_abort;
    if (!tid) { shared_inv = 0.0f; shared_abort = 0u; }
    if (group >= shape.z * shape.x) return;
    const uint token = group / shape.x, branch = group % shape.x, channels = shape.y;
    const ulong base = ulong(group) * channels, in_base = ulong(token) * channels, g_base = ulong(branch) * channels;
    const float scale = scales[token * shape.x + branch];
    if (channels == 2560u) {
        float4 rv4[5];
        uint bad = 0u;
        #pragma unroll
        for (uint q = 0u; q < 5u; ++q) {
            const uint idx4 = (q * 128u + uint(tid)) * 4u;
            const float4 st4 = float4(*reinterpret_cast<device const packed_float4*>(streams + base + idx4));
            const float4 in4 = float4(*reinterpret_cast<device const packed_float4*>(input + in_base + idx4));
            const float4 r4 = st4 + scale * in4;
            rv4[q] = r4;
            bad |= nonfin_hc(r4.x) | nonfin_hc(r4.y) | nonfin_hc(r4.z) | nonfin_hc(r4.w);
            *reinterpret_cast<device packed_float4*>(output + base + idx4) = packed_float4(r4);
            *reinterpret_cast<threadgroup float4*>(&v_tile[idx4 >> 5u][idx4 & 31u]) = r4;
        }
        if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float sum = 0.0f;
            for (uint m = 0u; m < 80u; ++m) { const float vm = v_tile[m][lane]; sum += vm * vm; }
            sum = simd_sum(sum);
            if (!lane) {
                const float denominator = sum / 2560.0f + as_type<float>(shape.w);
                const uint abort_flag = (nonfin_hc(sum) || nonfin_hc(denominator)) ? 1u : 0u;
                shared_abort = abort_flag;
                shared_inv = abort_flag ? 0.0f : rsqrt(denominator);
                if (abort_flag) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared_abort) return;
        const float inv = shared_inv;
        uint nbad = 0u;
        #pragma unroll
        for (uint q = 0u; q < 5u; ++q) {
            const uint idx4 = (q * 128u + uint(tid)) * 4u;
            const float4 g4 = float4(*reinterpret_cast<device const packed_float4*>(gains + g_base + idx4));
            nbad |= nonfin_hc(g4.x) | nonfin_hc(g4.y) | nonfin_hc(g4.z) | nonfin_hc(g4.w);
            const float4 n4 = (rv4[q] * inv) * g4;
            nbad |= nonfin_hc(n4.x) | nonfin_hc(n4.y) | nonfin_hc(n4.z) | nonfin_hc(n4.w);
            *reinterpret_cast<device packed_float4*>(normalized + base + idx4) = packed_float4(n4);
        }
        if (nbad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    if (sg == 0u) {
        float sum = 0.0f;
        for (uint j = lane; j < channels; j += 32u) {
            const float v = streams[base + j] + scale * input[in_base + j];
            checked_store(output, uint(base + j), v, error);
            sum += v * v;
        }
        sum = simd_sum(sum);
        const float denominator = sum / float(channels) + as_type<float>(shape.w);
        if (!isfinite(sum) || !isfinite(denominator)) { if (!lane) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        const float inv = rsqrt(denominator);
        for (uint j = lane; j < channels; j += 32u) {
            const float g = gains[g_base + j];
            if (!isfinite(g)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            checked_store(normalized, uint(base + j), (output[base + j] * inv) * g, error);
        }
    }
}
