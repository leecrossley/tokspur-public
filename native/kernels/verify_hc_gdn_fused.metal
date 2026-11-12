#include <metal_stdlib>
using namespace metal;

constant uint kVerifyHcEncoding [[function_constant(0)]];
constant uint kVerifyHcLayout [[function_constant(1)]];

namespace verify_fused_detail {

struct VerifyFusedHcQ8Args {
    uint columns;
    uint rows;
    uint row_bytes;
    uint tokens;
    uint silu_flag;
    float scale;
    uint reserved0;
    uint reserved1;
    ulong input_stride_bytes;
    ulong output_stride_bytes;
};
static_assert(sizeof(VerifyFusedHcQ8Args) == 48, "VerifyFusedHcQ8Args ABI");

struct VerifyFusedGdnGatesF32Args {
    uint rows;
    uint columns;
    uint outputs;
    uint half_groups;
    ulong input_stride_bytes;
    ulong alpha_stride_bytes;
    ulong beta_stride_bytes;
    ulong reserved_stride;
};
static_assert(sizeof(VerifyFusedGdnGatesF32Args) == 48, "VerifyFusedGdnGatesF32Args ABI");

struct VerifyFusedRowsF32Args {
    uint tokens;
    uint columns;
    uint outputs;
    uint splits;
    ulong input_stride_bytes;
    ulong output_stride_bytes;
};
static_assert(sizeof(VerifyFusedRowsF32Args) == 32, "VerifyFusedRowsF32Args ABI");

inline float sigmoid_fused(float x) {
    const float e = exp(x >= 0.0f ? -x : x);
    return (x >= 0.0f ? 1.0f : e) / (1.0f + e);
}

inline void flag_nonfinite(device atomic_uint* error) {
    atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

inline void checked_write(device float* dst, ulong idx, float val, device atomic_uint* error) {
    if (!isfinite(val)) flag_nonfinite(error);
    dst[idx] = val;
}

template <uint T>
inline void hc_staged_q8_block(
    float scale,
    uint byte_val,
    const threadgroup float (&sx)[T][512],
    uint col_local,
    thread float (&sums)[T]) {
    const float w = scale * float(as_type<char>(uchar(byte_val)));
    #pragma unroll
    for (uint i = 0u; i < T; ++i) {
        sums[i] += w * sx[i][col_local];
    }
}

template <uint T>
inline void hc_direct_q8_block(
    float scale,
    uint byte_val,
    device const float* const (&x_rows)[T],
    uint col,
    thread uint& bad,
    thread float (&sums)[T]) {
    const float w = scale * float(as_type<char>(uchar(byte_val)));
    #pragma unroll
    for (uint i = 0u; i < T; ++i) {
        const float xv = x_rows[i][col];
        if (!isfinite(xv)) bad = 1u;
        sums[i] += w * xv;
    }
}

template <uint T>
inline void verify_fused_hc_q8_impl(
    device const uchar* weights,
    device const uchar* input_bytes,
    device uchar* output_bytes,
    constant VerifyFusedHcQ8Args& a,
    device atomic_uint* error,
    uint group,
    ushort tid,
    ushort sg,
    ushort lane,
    threadgroup float (&sx)[T][512]) {
    const bool layout_ok = (kVerifyHcLayout == 0u) ||
        (kVerifyHcLayout == 1u && (a.columns < 512u || (a.columns % 256u) == 0u));
    if (kVerifyHcEncoding != 8u || !layout_ok ||
        a.columns < 32u || a.columns > 65536u || (a.columns % 32u) != 0u ||
        !a.rows || a.rows > 65536u || a.tokens != T ||
        a.row_bytes != (a.columns / 32u) * 34u ||
        (a.input_stride_bytes % 4u) != 0u || (a.output_stride_bytes % 4u) != 0u) {
        if (!group && !sg && !lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint row = group * 4u + uint(sg);
    device const float* x_rows[T];
    float sums[T];
    #pragma unroll
    for (uint i = 0u; i < T; ++i) {
        x_rows[i] = reinterpret_cast<device const float*>(input_bytes + ulong(i) * a.input_stride_bytes);
        sums[i] = 0.0f;
    }

    const uint blocks = a.columns / 32u;
    const uint full = blocks / 16u;
    uint bad = 0u;

    // Small output row counts (e.g. 10240 -> 320, 80 threadgroups) keep input hot in SLC/L1; skip barriers completely.
    if (a.rows <= 512u) {
        if (row >= a.rows) return;
        device const uchar* data = weights + ulong(row) * a.row_bytes;
        if (kVerifyHcLayout == 0u) {
            for (uint block = 0u; block < blocks; ++block) {
                device const uchar* encoded = data + ulong(block) * 34ul;
                const float w_scale = float(*reinterpret_cast<device const half*>(encoded));
                const float q = float(*reinterpret_cast<device const char*>(encoded + 2u + lane));
                const float w = w_scale * q;
                const uint col = block * 32u + uint(lane);
                #pragma unroll
                for (uint i = 0u; i < T; ++i) {
                    const float xv = x_rows[i][col];
                    if (!isfinite(xv)) bad = 1u;
                    sums[i] += w * xv;
                }
            }
        } else {
            device const half* scales = reinterpret_cast<device const half*>(data);
            device const uchar* quants = data + blocks * 2u;
            for (uint run = 0u; run < full; ++run) {
                const uint b0 = run * 16u;
                const uint4 words = uint4(*reinterpret_cast<device const packed_uint4*>(quants + ulong(run) * 512ul + uint(lane) * 16u));
                #pragma unroll
                for (uint part = 0u; part < 4u; ++part) {
                    const half4 sc4 = half4(*reinterpret_cast<device const packed_half4*>(scales + b0 + part * 4u));
                    #pragma unroll
                    for (uint k = 0u; k < 4u; ++k) {
                        const uint col = (b0 + part * 4u + k) * 32u + uint(lane);
                        hc_direct_q8_block<T>(float(sc4[k]), (words[part] >> (8u * k)) & 255u, x_rows, col, bad, sums);
                    }
                }
            }
            const uint rest = blocks - full * 16u;
            if (rest) {
                const uint b0 = full * 16u;
                device const uchar* tail = quants + ulong(full) * 512ul + uint(lane) * rest;
                for (uint k = 0u; k < rest; ++k) {
                    hc_direct_q8_block<T>(float(scales[b0 + k]), uint(tail[k]), x_rows, (b0 + k) * 32u + uint(lane), bad, sums);
                }
            }
        }
        if (simd_or(bad) && !lane) flag_nonfinite(error);
        #pragma unroll
        for (uint i = 0u; i < T; ++i) {
            const float total = simd_sum(sums[i]);
            if (!lane) {
                device float* y_row = reinterpret_cast<device float*>(output_bytes + ulong(i) * a.output_stride_bytes);
                if (a.silu_flag != 0u) {
                    const float x = total * a.scale;
                    if (!isfinite(total) || !isfinite(x)) {
                        flag_nonfinite(error);
                        y_row[row] = x;
                    } else {
                        checked_write(y_row, row, x * sigmoid_fused(x), error);
                    }
                } else {
                    checked_write(y_row, row, total, error);
                }
            }
        }
        return;
    }

    for (uint b0 = 0u; b0 < blocks; b0 += 16u) {
        const uint cnt = min(16u, blocks - b0);
        const uint elems = cnt * 32u;
        #pragma unroll
        for (uint step = 0u; step < 4u; ++step) {
            const uint c = uint(tid) + step * 128u;
            if (c < elems) {
                const uint col = b0 * 32u + c;
                #pragma unroll
                for (uint i = 0u; i < T; ++i) {
                    const float xv = x_rows[i][col];
                    if (!isfinite(xv)) bad = 1u;
                    sx[i][c] = xv;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < a.rows) {
            device const uchar* data = weights + ulong(row) * a.row_bytes;
            if (kVerifyHcLayout == 0u) {
                for (uint k = 0u; k < cnt; ++k) {
                    const uint block = b0 + k;
                    device const uchar* encoded = data + ulong(block) * 34ul;
                    const float w_scale = float(*reinterpret_cast<device const half*>(encoded));
                    const float q = float(*reinterpret_cast<device const char*>(encoded + 2u + lane));
                    const float w = w_scale * q;
                    const uint col_local = k * 32u + uint(lane);
                    #pragma unroll
                    for (uint i = 0u; i < T; ++i) {
                        sums[i] += w * sx[i][col_local];
                    }
                }
            } else if (cnt == 16u) {
                const uint run = b0 >> 4u;
                device const half* scales = reinterpret_cast<device const half*>(data);
                device const uchar* quants = data + blocks * 2u;
                const uint4 words = uint4(*reinterpret_cast<device const packed_uint4*>(quants + ulong(run) * 512ul + uint(lane) * 16u));
                #pragma unroll
                for (uint part = 0u; part < 4u; ++part) {
                    const half4 sc4 = half4(*reinterpret_cast<device const packed_half4*>(scales + b0 + part * 4u));
                    #pragma unroll
                    for (uint k = 0u; k < 4u; ++k) {
                        const uint col_local = (part * 4u + k) * 32u + uint(lane);
                        hc_staged_q8_block<T>(float(sc4[k]), (words[part] >> (8u * k)) & 255u, sx, col_local, sums);
                    }
                }
            } else {
                const uint rest = blocks - full * 16u;
                const uint tail_off = b0 - full * 16u;
                device const half* scales = reinterpret_cast<device const half*>(data);
                device const uchar* tail = data + blocks * 2u + ulong(full) * 512ul + uint(lane) * rest + tail_off;
                if ((cnt % 4u) == 0u) {
                    for (uint k = 0u; k < cnt; k += 4u) {
                        const uint word = *reinterpret_cast<device const uint*>(tail + k);
                        #pragma unroll
                        for (uint sub = 0u; sub < 4u; ++sub) {
                            hc_staged_q8_block<T>(float(scales[b0 + k + sub]), (word >> (8u * sub)) & 255u, sx, (k + sub) * 32u + uint(lane), sums);
                        }
                    }
                } else if ((cnt % 2u) == 0u) {
                    for (uint k = 0u; k < cnt; k += 2u) {
                        const uint word = uint(tail[k]) | (uint(tail[k + 1u]) << 8u);
                        #pragma unroll
                        for (uint sub = 0u; sub < 2u; ++sub) {
                            hc_staged_q8_block<T>(float(scales[b0 + k + sub]), (word >> (8u * sub)) & 255u, sx, (k + sub) * 32u + uint(lane), sums);
                        }
                    }
                } else {
                    for (uint k = 0u; k < cnt; ++k) {
                        hc_staged_q8_block<T>(float(scales[b0 + k]), uint(tail[k]), sx, k * 32u + uint(lane), sums);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_or(bad) && !lane) flag_nonfinite(error);
    if (row >= a.rows) return;

    #pragma unroll
    for (uint i = 0u; i < T; ++i) {
        const float total = simd_sum(sums[i]);
        if (!lane) {
            device float* y_row = reinterpret_cast<device float*>(output_bytes + ulong(i) * a.output_stride_bytes);
            if (a.silu_flag != 0u) {
                const float x = total * a.scale;
                if (!isfinite(total) || !isfinite(x)) {
                    flag_nonfinite(error);
                    y_row[row] = x;
                } else {
                    checked_write(y_row, row, x * sigmoid_fused(x), error);
                }
            } else {
                checked_write(y_row, row, total, error);
            }
        }
    }
}

} // namespace verify_fused_detail

#define VERIFY_FUSED_HC_Q8_KERNEL(T) \
kernel void verify_fused_hc_q8_##T( \
    device const uchar* weights [[buffer(0)]], \
    device const uchar* input_bytes [[buffer(1)]], \
    device uchar* output_bytes [[buffer(2)]], \
    constant verify_fused_detail::VerifyFusedHcQ8Args& a [[buffer(3)]], \
    device atomic_uint* error [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], \
    ushort tid [[thread_index_in_threadgroup]], \
    ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sx[T][512]; \
    verify_fused_detail::verify_fused_hc_q8_impl<T>(weights, input_bytes, output_bytes, a, error, group, tid, sg, lane, sx); \
}

VERIFY_FUSED_HC_Q8_KERNEL(1)
VERIFY_FUSED_HC_Q8_KERNEL(2)
VERIFY_FUSED_HC_Q8_KERNEL(3)
VERIFY_FUSED_HC_Q8_KERNEL(4)
VERIFY_FUSED_HC_Q8_KERNEL(5)
VERIFY_FUSED_HC_Q8_KERNEL(6)
VERIFY_FUSED_HC_Q8_KERNEL(7)
VERIFY_FUSED_HC_Q8_KERNEL(8)
#undef VERIFY_FUSED_HC_Q8_KERNEL

// Full-grid multi-row FP32 inject projection + exact split-partial reduction + scale emission (256 threads/TG across (branches, tokens)).
kernel void verify_fused_hc_inject_partials(
    device const float* weights [[buffer(0)]],
    device const float* normalized [[buffer(1)]],
    device float* inject_logits [[buffer(2)]],
    device float* scales [[buffer(3)]],
    constant uint4& shape [[buffer(4)]], // {columns, branches, splits, tokens}
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float s_part[32];
    const uint columns = shape.x;
    const uint branches = shape.y;
    const uint splits = shape.z;
    const uint tokens = shape.w;
    if (columns < 32u || columns > 65536u || (columns % 32u) != 0u ||
        !branches || branches > 16u || !splits || splits > 32u || !tokens || tokens > 64u) {
        if (!group.x && !group.y && !tid) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        }
        return;
    }

    const uint branch = group.x;
    const uint token = group.y;
    if (branch >= branches || token >= tokens) return;

    device const float* w_row = weights + ulong(branch) * columns;
    device const float* x_row = normalized + ulong(token) * columns;
    const ulong out_idx = ulong(token) * branches + branch;

    if (splits == 1u) {
        if (sg != 0u) return;
        float sum = 0.0f;
        uint bad = 0u;
        #pragma unroll 4
        for (uint col = uint(lane); col < columns; col += 32u) {
            const float w = w_row[col];
            const float xv = x_row[col];
            if (!isfinite(w) || !isfinite(xv)) bad = 1u;
            sum += w * xv;
        }
        if (simd_or(bad) && !lane) verify_fused_detail::flag_nonfinite(error);
        const float inj = simd_sum(sum);
        if (!lane) {
            verify_fused_detail::checked_write(inject_logits, out_idx, inj, error);
            if (isfinite(inj)) {
                verify_fused_detail::checked_write(scales, out_idx, 2.0f * verify_fused_detail::sigmoid_fused(inj / float(branches)), error);
            }
        }
        return;
    }

    const uint width = ((columns + splits * 32u - 1u) / (splits * 32u)) * 32u;
    uint bad = 0u;
    for (uint sp = uint(sg); sp < splits; sp += 8u) {
        const uint begin = sp * width;
        const uint end = begin < columns ? min(begin + width, columns) : columns;
        float sum = 0.0f;
        #pragma unroll 4
        for (uint col = begin + uint(lane); col < end; col += 32u) {
            const float w = w_row[col];
            const float xv = x_row[col];
            if (!isfinite(w) || !isfinite(xv)) bad = 1u;
            sum += w * xv;
        }
        const float part = simd_sum(sum);
        if (!lane) {
            if (!isfinite(part)) verify_fused_detail::flag_nonfinite(error);
            s_part[sp] = part;
        }
    }
    if (simd_or(bad) && !lane) verify_fused_detail::flag_nonfinite(error);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float inj = 0.0f;
        for (uint sp = 0u; sp < splits; ++sp) {
            inj += s_part[sp];
        }
        verify_fused_detail::checked_write(inject_logits, out_idx, inj, error);
        if (isfinite(inj)) {
            verify_fused_detail::checked_write(scales, out_idx, 2.0f * verify_fused_detail::sigmoid_fused(inj / float(branches)), error);
        }
    }
}

// Coalesced vectorised multi-row residual stream read fusing split partial reduction + scales on channel < branches.
kernel void verify_fused_hc_read_rows(
    device const float* normalized [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device const float* partials [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* inject_logits [[buffer(4)]],
    device float* scales [[buffer(5)]],
    constant uint4& shape [[buffer(6)]], // {branches, channels, tokens, splits}
    device atomic_uint* error [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const uint branches = shape.x;
    const uint channels = shape.y;
    const uint tokens = shape.z;
    const uint splits = shape.w;
    if (!branches || branches > 16u || channels < branches || channels > 65536u || !tokens || tokens > 64u ||
        !splits || splits > 32u) {
        if (!index) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint quads = (channels + 3u) >> 2u;
    const uint token = index / quads;
    const uint col4 = (index % quads) << 2u;
    if (token >= tokens) return;

    if ((channels & 3u) == 0u) {
        float4 total = float4(0.0f);
        bool4 ok = bool4(true);
        if (branches == 4u) {
            #pragma unroll
            for (uint b = 0u; b < 4u; ++b) {
                const ulong at = (ulong(token) * 4ul + b) * channels + col4;
                const float4 n = float4(*reinterpret_cast<device const packed_float4*>(normalized + at));
                const float4 s = float4(*reinterpret_cast<device const packed_float4*>(scores + at));
                ok = ok && isfinite(n) && isfinite(s);
                total.x += n.x * verify_fused_detail::sigmoid_fused(s.x);
                total.y += n.y * verify_fused_detail::sigmoid_fused(s.y);
                total.z += n.z * verify_fused_detail::sigmoid_fused(s.z);
                total.w += n.w * verify_fused_detail::sigmoid_fused(s.w);
            }
        } else {
            for (uint b = 0u; b < branches; ++b) {
                const ulong at = (ulong(token) * branches + b) * channels + col4;
                const float4 n = float4(*reinterpret_cast<device const packed_float4*>(normalized + at));
                const float4 s = float4(*reinterpret_cast<device const packed_float4*>(scores + at));
                ok = ok && isfinite(n) && isfinite(s);
                total.x += n.x * verify_fused_detail::sigmoid_fused(s.x);
                total.y += n.y * verify_fused_detail::sigmoid_fused(s.y);
                total.z += n.z * verify_fused_detail::sigmoid_fused(s.z);
                total.w += n.w * verify_fused_detail::sigmoid_fused(s.w);
            }
        }
        const float4 out4 = total / float(branches);
        const ulong out_base = ulong(token) * channels + col4;
        if (all(ok) && all(isfinite(out4))) {
            *reinterpret_cast<device packed_float4*>(output + out_base) = packed_float4(out4);
        } else {
            verify_fused_detail::flag_nonfinite(error);
            for (uint k = 0u; k < 4u; ++k) {
                if (ok[k]) verify_fused_detail::checked_write(output, out_base + k, out4[k], error);
            }
        }
        if (col4 < branches) {
            for (uint k = 0u; k < 4u && col4 + k < branches; ++k) {
                if (!ok[k]) continue;
                const ulong at = ulong(token) * branches + col4 + k;
                float inj = 0.0f;
                if (splits == 1u) {
                    inj = partials[at];
                } else if (splits == 16u) {
                    device const packed_float4* p4 = reinterpret_cast<device const packed_float4*>(partials + at * 16ul);
                    #pragma unroll
                    for (uint q = 0u; q < 4u; ++q) {
                        const float4 pv = float4(p4[q]);
                        inj = (((inj + pv.x) + pv.y) + pv.z) + pv.w;
                    }
                } else if (splits == 32u) {
                    device const packed_float4* p4 = reinterpret_cast<device const packed_float4*>(partials + at * 32ul);
                    #pragma unroll
                    for (uint q = 0u; q < 8u; ++q) {
                        const float4 pv = float4(p4[q]);
                        inj = (((inj + pv.x) + pv.y) + pv.z) + pv.w;
                    }
                } else {
                    for (uint i = 0u; i < splits; ++i) inj += partials[at * splits + i];
                }
                verify_fused_detail::checked_write(inject_logits, at, inj, error);
                if (isfinite(inj)) {
                    verify_fused_detail::checked_write(scales, at, 2.0f * verify_fused_detail::sigmoid_fused(inj / float(branches)), error);
                }
            }
        }
        return;
    }

    const uint cnt = min(4u, channels - col4);
    for (uint k = 0u; k < cnt; ++k) {
        const uint channel = col4 + k;
        float total = 0.0f;
        bool lane_ok = true;
        for (uint b = 0u; b < branches; ++b) {
            const ulong at = (ulong(token) * branches + b) * channels + channel;
            const float n = normalized[at];
            const float s = scores[at];
            if (!isfinite(n) || !isfinite(s)) {
                verify_fused_detail::flag_nonfinite(error);
                lane_ok = false;
                break;
            }
            total += n * verify_fused_detail::sigmoid_fused(s);
        }
        if (!lane_ok) continue;
        verify_fused_detail::checked_write(output, ulong(token) * channels + channel, total / float(branches), error);
        if (channel < branches) {
            const ulong at = ulong(token) * branches + channel;
            float inj = (splits == 1u) ? partials[at] : 0.0f;
            if (splits > 1u) {
                for (uint i = 0u; i < splits; ++i) inj += partials[at * splits + i];
            }
            verify_fused_detail::checked_write(inject_logits, at, inj, error);
            if (isfinite(inj)) {
                verify_fused_detail::checked_write(scales, at, 2.0f * verify_fused_detail::sigmoid_fused(inj / float(branches)), error);
            }
        }
    }
}

// Coalesced 1-float-per-thread multi-row gated_mix (256 threads/TG).
kernel void verify_fused_hc_gated_mix_rows(
    device const float* normalized [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint4& shape [[buffer(3)]], // {branches, channels, tokens, 0}
    device atomic_uint* error [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const uint branches = shape.x;
    const uint channels = shape.y;
    const uint tokens = shape.z;
    if (!branches || branches > 16u || !channels || channels > 65536u || !tokens || tokens > 64u) {
        if (!index) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint token = index / channels;
    const uint channel = index % channels;
    if (token >= tokens) return;

    float total = 0.0f;
    bool lane_ok = true;
    if (branches == 4u) {
        #pragma unroll
        for (uint b = 0u; b < 4u; ++b) {
            const ulong at = (ulong(token) * 4ul + b) * channels + channel;
            const float n = normalized[at];
            const float s = scores[at];
            if (!isfinite(n) || !isfinite(s)) {
                verify_fused_detail::flag_nonfinite(error);
                lane_ok = false;
                break;
            }
            total += n * verify_fused_detail::sigmoid_fused(s);
        }
    } else {
        for (uint b = 0u; b < branches; ++b) {
            const ulong at = (ulong(token) * branches + b) * channels + channel;
            const float n = normalized[at];
            const float s = scores[at];
            if (!isfinite(n) || !isfinite(s)) {
                verify_fused_detail::flag_nonfinite(error);
                lane_ok = false;
                break;
            }
            total += n * verify_fused_detail::sigmoid_fused(s);
        }
    }
    if (lane_ok) {
        verify_fused_detail::checked_write(output, ulong(token) * channels + channel, total / float(branches), error);
    }
}

// Coalesced vectorised multi-row residual stream write without hc_q8 split restrictions.
kernel void verify_fused_hc_write_rows(
    device const float* streams [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint4& shape [[buffer(4)]], // {branches, channels, tokens, 0}
    device atomic_uint* error [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const uint branches = shape.x;
    const uint channels = shape.y;
    const uint tokens = shape.z;
    const uint width = branches * channels;
    if (!branches || branches > 16u || !channels || channels > 65536u || !tokens || tokens > 64u) {
        if (!index) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint w_quads = (width + 3u) >> 2u;
    const uint token = index / w_quads;
    const uint at0 = (index % w_quads) << 2u;
    if (token >= tokens) return;

    if ((channels & 3u) == 0u) {
        const uint branch = at0 / channels;
        const uint col4 = at0 % channels;
        const ulong base = ulong(token) * width + at0;
        const float4 st = float4(*reinterpret_cast<device const packed_float4*>(streams + base));
        const float4 xv = float4(*reinterpret_cast<device const packed_float4*>(input + ulong(token) * channels + col4));
        const float sc = scales[ulong(token) * branches + branch];
        const float4 out4 = st + sc * xv;
        if (!all(isfinite(st)) || !isfinite(sc) || !all(isfinite(xv)) || !all(isfinite(out4))) {
            verify_fused_detail::flag_nonfinite(error);
        }
        *reinterpret_cast<device packed_float4*>(output + base) = packed_float4(out4);
        return;
    }

    const uint cnt = min(4u, width - at0);
    for (uint k = 0u; k < cnt; ++k) {
        const uint at = at0 + k;
        const ulong dst_idx = ulong(token) * width + at;
        const float st = streams[dst_idx];
        const float sc = scales[ulong(token) * branches + (at / channels)];
        const float xv = input[ulong(token) * channels + (at % channels)];
        if (!isfinite(st) || !isfinite(sc) || !isfinite(xv)) {
            verify_fused_detail::flag_nonfinite(error);
        }
        verify_fused_detail::checked_write(output, dst_idx, st + sc * xv, error);
    }
}

// Fused multi-row residual stream write + next-sublayer per-branch RMS normalisation (256 threads/TG).
kernel void verify_fused_hc_write_rms_rows(
    device const float* streams [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* gains [[buffer(3)]],
    device float* output [[buffer(4)]],
    device float* normalized [[buffer(5)]],
    constant uint4& shape [[buffer(6)]], // {branches, channels, tokens | (shared_gains << 16), eps_bits}
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint branches = shape.x;
    const uint channels = shape.y;
    const uint tokens = shape.z & 0xffffu;
    const uint shared_gains = shape.z >> 16u;
    const float eps = as_type<float>(shape.w);
    if (!branches || branches > 16u || channels < 32u || channels > 65536u || (channels % 32u) != 0u ||
        !tokens || tokens > 64u || !isfinite(eps) || eps < 0.0f) {
        if (!group.x && !group.y && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    threadgroup float sram_v[80][32];
    threadgroup uint sg_bad[8];
    threadgroup float shared_inv;
    threadgroup uint shared_abort;

    const uint branch = group.x;
    const uint token = group.y;
    if (branch >= branches || token >= tokens) return;

    const ulong base = (ulong(token) * branches + branch) * channels;
    const ulong x_base = ulong(token) * channels;
    const float sc = scales[ulong(token) * branches + branch];

    if (channels <= 2560u) {
        if (!tid) {
            shared_inv = 0.0f;
            shared_abort = 0u;
        }
        const uint steps32 = channels >> 5u;
        uint bad = isfinite(sc) ? 0u : 1u;
        if (channels == 2560u) {
            #pragma unroll
            for (uint m = 0u; m < 10u; ++m) {
                const uint idx = m * 256u + uint(tid);
                const float st = streams[base + idx];
                const float xv = input[x_base + idx];
                const float v = st + sc * xv;
                if (!isfinite(st) || !isfinite(xv) || !isfinite(v)) bad = 1u;
                output[base + idx] = v;
                sram_v[m * 8u + uint(sg)][lane] = v;
            }
        } else {
            for (uint k = uint(sg); k < steps32; k += 8u) {
                const uint idx = k * 32u + uint(lane);
                const float st = streams[base + idx];
                const float xv = input[x_base + idx];
                const float v = st + sc * xv;
                if (!isfinite(st) || !isfinite(xv) || !isfinite(v)) bad = 1u;
                output[base + idx] = v;
                sram_v[k][lane] = v;
            }
        }
        const uint sbad = simd_or(bad);
        if (!lane) sg_bad[sg] = sbad;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0u) {
            float sum = 0.0f;
            #pragma unroll 4
            for (uint k = 0u; k < steps32; ++k) {
                const float vm = sram_v[k][lane];
                sum += vm * vm;
            }
            sum = simd_sum(sum);
            if (!lane) {
                const uint all_bad = sg_bad[0] | sg_bad[1] | sg_bad[2] | sg_bad[3] |
                                     sg_bad[4] | sg_bad[5] | sg_bad[6] | sg_bad[7];
                const float denominator = sum / float(channels) + eps;
                const uint abort_flag = (all_bad || !isfinite(sum) || !isfinite(denominator) || denominator <= 0.0f) ? 1u : 0u;
                shared_abort = abort_flag;
                shared_inv = abort_flag ? 0.0f : rsqrt(denominator);
                if (abort_flag) verify_fused_detail::flag_nonfinite(error);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared_abort) return;

        const float inv = shared_inv;
        const ulong gain_base = shared_gains ? 0ul : ulong(branch) * channels;
        uint bad2 = 0u;
        if (channels == 2560u) {
            #pragma unroll
            for (uint m = 0u; m < 10u; ++m) {
                const uint idx = m * 256u + uint(tid);
                const float rv = sram_v[m * 8u + uint(sg)][lane];
                const float g = gains[gain_base + idx];
                const float n = (rv * inv) * g;
                if (!isfinite(g) || !isfinite(n)) bad2 = 1u;
                normalized[base + idx] = n;
            }
        } else {
            for (uint k = uint(sg); k < steps32; k += 8u) {
                const uint idx = k * 32u + uint(lane);
                const float rv = sram_v[k][lane];
                const float g = gains[gain_base + idx];
                const float n = (rv * inv) * g;
                if (!isfinite(g) || !isfinite(n)) bad2 = 1u;
                normalized[base + idx] = n;
            }
        }
        if (simd_or(bad2) && !lane) verify_fused_detail::flag_nonfinite(error);
        return;
    }

    if (sg != 0u) return;
    uint bad = isfinite(sc) ? 0u : 1u;
    float sum = 0.0f;
    for (uint j = uint(lane); j < channels; j += 32u) {
        const float st = streams[base + j];
        const float xv = input[x_base + j];
        if (!isfinite(st) || !isfinite(xv)) bad = 1u;
        const float v = st + sc * xv;
        verify_fused_detail::checked_write(output, base + j, v, error);
        sum += v * v;
    }
    sum = simd_sum(sum);
    const float denominator = sum / float(channels) + eps;
    const uint any_bad = simd_or(bad);
    if (any_bad || !isfinite(sum) || !isfinite(denominator) || denominator <= 0.0f) {
        if (!lane) verify_fused_detail::flag_nonfinite(error);
        return;
    }
    const float inv = rsqrt(denominator);
    const ulong gain_base = shared_gains ? 0ul : ulong(branch) * channels;
    for (uint j = uint(lane); j < channels; j += 32u) {
        const float g = gains[gain_base + j];
        if (!isfinite(g)) verify_fused_detail::flag_nonfinite(error);
        verify_fused_detail::checked_write(normalized, base + j, (output[base + j] * inv) * g, error);
    }
}

// Single-dispatch GDN alpha + beta FP32 gate projections (2560 -> 16) with pure scalar accumulators.
kernel void verify_fused_gdn_alpha_beta_f32(
    device const float* w_alpha [[buffer(0)]],
    device const float* w_beta [[buffer(1)]],
    device const uchar* input_bytes [[buffer(2)]],
    device uchar* alpha_bytes [[buffer(3)]],
    device uchar* beta_bytes [[buffer(4)]],
    constant verify_fused_detail::VerifyFusedGdnGatesF32Args& a [[buffer(5)]],
    device atomic_uint* error [[buffer(7)]],
    uint2 g [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    static_cast<void>(tid);
    if (!a.rows || a.rows > 64u || a.columns < 32u || a.columns > 65536u || (a.columns % 32u) != 0u ||
        !a.outputs || a.outputs > 256u || a.half_groups != (a.outputs + 3u) / 4u ||
        (a.input_stride_bytes % 4u) != 0u || (a.alpha_stride_bytes % 4u) != 0u || (a.beta_stride_bytes % 4u) != 0u) {
        if (!g.x && !g.y && !sg && !lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint first = g.y * 4u;
    if (first >= a.rows) return;
    const uint nt = min(4u, a.rows - first);

    const bool is_beta = g.x >= a.half_groups;
    const uint local_gx = is_beta ? (g.x - a.half_groups) : g.x;
    const uint out = local_gx * 4u + uint(sg);
    if (out >= a.outputs) return;

    device const float* w_row = (is_beta ? w_beta : w_alpha) + ulong(out) * a.columns;
    device uchar* out_base = is_beta ? beta_bytes : alpha_bytes;
    const ulong out_stride = is_beta ? a.beta_stride_bytes : a.alpha_stride_bytes;

    device const float* x0_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(first) * a.input_stride_bytes);
    device const float* x1_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(min(first + 1u, a.rows - 1u)) * a.input_stride_bytes);
    device const float* x2_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(min(first + 2u, a.rows - 1u)) * a.input_stride_bytes);
    device const float* x3_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(min(first + 3u, a.rows - 1u)) * a.input_stride_bytes);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    uint bad = 0u;

    #pragma unroll 4
    for (uint k = uint(lane); k < a.columns; k += 32u) {
        const float w = w_row[k];
        const float x0 = x0_ptr[k];
        const float x1 = x1_ptr[k];
        const float x2 = x2_ptr[k];
        const float x3 = x3_ptr[k];
        if (!isfinite(w) || !isfinite(x0) || !isfinite(x1) || !isfinite(x2) || !isfinite(x3)) bad = 1u;
        s0 += w * x0;
        s1 += w * x1;
        s2 += w * x2;
        s3 += w * x3;
    }

    if (simd_or(bad) && !lane) verify_fused_detail::flag_nonfinite(error);

    const auto emit_row = [&](uint j, float sm) {
        const float total = simd_sum(sm);
        if (!lane) {
            device float* y_row = reinterpret_cast<device float*>(out_base + ulong(first + j) * out_stride);
            verify_fused_detail::checked_write(y_row, out, total, error);
        }
    };
    emit_row(0u, s0);
    if (nt > 1u) emit_row(1u, s1);
    if (nt > 2u) emit_row(2u, s2);
    if (nt > 3u) emit_row(3u, s3);
}

// Single-dispatch multi-row FP32 expert-gate projection (pure scalar accumulators across 1..8 tokens, zero stack spill).
kernel void verify_fused_expert_gate_f32(
    device const float* weights [[buffer(0)]],
    device const uchar* input_bytes [[buffer(1)]],
    device uchar* output_bytes [[buffer(2)]],
    constant verify_fused_detail::VerifyFusedRowsF32Args& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (!a.tokens || a.tokens > 8u || a.columns < 32u || a.columns > 65536u || (a.columns % 32u) != 0u ||
        !a.outputs || a.outputs > 4096u ||
        (a.input_stride_bytes % 4u) != 0u || (a.output_stride_bytes % 4u) != 0u) {
        if (!group && !sg && !lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint out = group * 4u + uint(sg);
    if (out >= a.outputs) return;

    const uint nt = a.tokens;
    device const float* w_row = weights + ulong(out) * a.columns;
    device const float* x0_ptr = reinterpret_cast<device const float*>(input_bytes);
    device const float* x1_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(min(1u, nt - 1u)) * a.input_stride_bytes);
    device const float* x2_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(min(2u, nt - 1u)) * a.input_stride_bytes);
    device const float* x3_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(min(3u, nt - 1u)) * a.input_stride_bytes);
    device const float* x4_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(min(4u, nt - 1u)) * a.input_stride_bytes);
    device const float* x5_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(min(5u, nt - 1u)) * a.input_stride_bytes);
    device const float* x6_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(min(6u, nt - 1u)) * a.input_stride_bytes);
    device const float* x7_ptr = reinterpret_cast<device const float*>(input_bytes + ulong(min(7u, nt - 1u)) * a.input_stride_bytes);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    uint bad = 0u;

    if (nt == 8u) {
        #pragma unroll 4
        for (uint k = uint(lane); k < a.columns; k += 32u) {
            const float w = w_row[k];
            const float x0 = x0_ptr[k], x1 = x1_ptr[k], x2 = x2_ptr[k], x3 = x3_ptr[k];
            const float x4 = x4_ptr[k], x5 = x5_ptr[k], x6 = x6_ptr[k], x7 = x7_ptr[k];
            if (!isfinite(w) || !isfinite(x0) || !isfinite(x1) || !isfinite(x2) || !isfinite(x3) ||
                !isfinite(x4) || !isfinite(x5) || !isfinite(x6) || !isfinite(x7)) {
                bad = 1u;
            }
            s0 += w * x0; s1 += w * x1; s2 += w * x2; s3 += w * x3;
            s4 += w * x4; s5 += w * x5; s6 += w * x6; s7 += w * x7;
        }
    } else {
        #pragma unroll 4
        for (uint k = uint(lane); k < a.columns; k += 32u) {
            const float w = w_row[k];
            const float x0 = x0_ptr[k], x1 = x1_ptr[k], x2 = x2_ptr[k], x3 = x3_ptr[k];
            const float x4 = x4_ptr[k], x5 = x5_ptr[k], x6 = x6_ptr[k];
            if (!isfinite(w) || !isfinite(x0) || !isfinite(x1) || !isfinite(x2) || !isfinite(x3) ||
                !isfinite(x4) || !isfinite(x5) || !isfinite(x6)) {
                bad = 1u;
            }
            s0 += w * x0; s1 += w * x1; s2 += w * x2; s3 += w * x3;
            s4 += w * x4; s5 += w * x5; s6 += w * x6;
        }
    }

    if (simd_or(bad) && !lane) verify_fused_detail::flag_nonfinite(error);

    const auto emit_tok = [&](uint m, float sm) {
        const float total = simd_sum(sm);
        if (!lane) {
            device float* y_row = reinterpret_cast<device float*>(output_bytes + ulong(m) * a.output_stride_bytes);
            verify_fused_detail::checked_write(y_row, out, total, error);
        }
    };
    emit_tok(0u, s0);
    if (nt > 1u) emit_tok(1u, s1);
    if (nt > 2u) emit_tok(2u, s2);
    if (nt > 3u) emit_tok(3u, s3);
    if (nt > 4u) emit_tok(4u, s4);
    if (nt > 5u) emit_tok(5u, s5);
    if (nt > 6u) emit_tok(6u, s6);
    if (nt > 7u) emit_tok(7u, s7);
}

// Multi-threadgroup (1 TG of 256 threads per token) FP32 shared-expert scalar gate projection with exact split-partial reduction.
kernel void verify_fused_shared_logit_f32(
    device const float* weights [[buffer(0)]],
    device const uchar* input_bytes [[buffer(1)]],
    device uchar* output_bytes [[buffer(2)]],
    constant verify_fused_detail::VerifyFusedRowsF32Args& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float s_part[32];
    if (!a.tokens || a.tokens > 8u ||
        a.columns < 32u || a.columns > 65536u || (a.columns % 32u) != 0u ||
        a.outputs != 1u || a.splits > 32u ||
        (a.input_stride_bytes % 4u) != 0u || (a.output_stride_bytes % 4u) != 0u) {
        if (!group && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group >= a.tokens) return;

    device const float* x_row = reinterpret_cast<device const float*>(input_bytes + ulong(group) * a.input_stride_bytes);
    device float* y_row = reinterpret_cast<device float*>(output_bytes + ulong(group) * a.output_stride_bytes);

    if (a.splits <= 1u) {
        if (sg != 0u) return;
        float sum = 0.0f;
        uint bad = 0u;
        #pragma unroll 4
        for (uint k = uint(lane); k < a.columns; k += 32u) {
            const float w = weights[k];
            const float xv = x_row[k];
            if (!isfinite(w) || !isfinite(xv)) bad = 1u;
            sum += w * xv;
        }
        if (simd_or(bad) && !lane) verify_fused_detail::flag_nonfinite(error);
        const float total = simd_sum(sum);
        if (!lane) verify_fused_detail::checked_write(y_row, 0u, total, error);
        return;
    }

    const uint width = ((a.columns + a.splits * 32u - 1u) / (a.splits * 32u)) * 32u;
    uint bad = 0u;
    for (uint sp = uint(sg); sp < a.splits; sp += 8u) {
        const uint begin = sp * width;
        const uint end = begin < a.columns ? min(begin + width, a.columns) : a.columns;
        float sum = 0.0f;
        #pragma unroll 4
        for (uint k = begin + uint(lane); k < end; k += 32u) {
            const float w = weights[k];
            const float xv = x_row[k];
            if (!isfinite(w) || !isfinite(xv)) bad = 1u;
            sum += w * xv;
        }
        const float part = simd_sum(sum);
        if (!lane) {
            if (!isfinite(part)) verify_fused_detail::flag_nonfinite(error);
            s_part[sp] = part;
        }
    }
    if (simd_or(bad) && !lane) verify_fused_detail::flag_nonfinite(error);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float total = 0.0f;
        for (uint sp = 0u; sp < a.splits; ++sp) {
            total += s_part[sp];
        }
        verify_fused_detail::checked_write(y_row, 0u, total, error);
    }
}
