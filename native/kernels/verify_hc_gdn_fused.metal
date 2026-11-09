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

inline float sigmoid_fused(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    const float e = exp(x);
    return e / (1.0f + e);
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

// Full-grid multi-row FP32 inject split partials (tiles up to 8 tokens per threadgroup, loading weights once).
kernel void verify_fused_hc_inject_partials(
    device const float* weights [[buffer(0)]],
    device const float* normalized [[buffer(1)]],
    device float* partials [[buffer(2)]],
    constant uint4& shape [[buffer(3)]], // {columns, branches, splits, tokens}
    device atomic_uint* error [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint columns = shape.x;
    const uint branches = shape.y;
    const uint splits = shape.z;
    const uint tokens = shape.w;
    if (columns < 32u || columns > 65536u || (columns % 32u) != 0u ||
        !branches || branches > 16u || !splits || splits > 32u || !tokens || tokens > 64u) {
        if (!group.x && !group.y && !group.z && !sg && !lane) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        }
        return;
    }

    const uint tok0 = group.z * 8u;
    const uint split = group.y * 4u + uint(sg);
    const uint width = ((columns + splits * 32u - 1u) / (splits * 32u)) * 32u;
    const uint begin = split * width;
    const uint end = begin < columns ? min(begin + width, columns) : columns;

    if (branches == 4u) {
        if (group.x != 0u || sg >= 4u || split >= splits || tok0 >= tokens) return;
        const uint nt = min(8u, tokens - tok0);
        device const float* w0 = weights;
        device const float* w1 = w0 + columns;
        device const float* w2 = w1 + columns;
        device const float* w3 = w2 + columns;
        device const float* x_base = normalized + ulong(tok0) * columns;
        float4 s0(0.0f), s1(0.0f), s2(0.0f), s3(0.0f), s4(0.0f), s5(0.0f), s6(0.0f), s7(0.0f);
        if (nt == 8u) {
            for (uint col = begin + uint(lane); col < end; col += 32u) {
                const float4 wv(w0[col], w1[col], w2[col], w3[col]);
                device const float* xc = x_base + col;
                s0 += wv * xc[0ul * columns];
                s1 += wv * xc[1ul * columns];
                s2 += wv * xc[2ul * columns];
                s3 += wv * xc[3ul * columns];
                s4 += wv * xc[4ul * columns];
                s5 += wv * xc[5ul * columns];
                s6 += wv * xc[6ul * columns];
                s7 += wv * xc[7ul * columns];
            }
        } else {
            for (uint col = begin + uint(lane); col < end; col += 32u) {
                const float4 wv(w0[col], w1[col], w2[col], w3[col]);
                device const float* xc = x_base + col;
                s0 += wv * xc[0ul * columns];
                if (nt > 1u) s1 += wv * xc[1ul * columns];
                if (nt > 2u) s2 += wv * xc[2ul * columns];
                if (nt > 3u) s3 += wv * xc[3ul * columns];
                if (nt > 4u) s4 += wv * xc[4ul * columns];
                if (nt > 5u) s5 += wv * xc[5ul * columns];
                if (nt > 6u) s6 += wv * xc[6ul * columns];
            }
        }
        const auto emit_tok4 = [&](uint m, float4 sm) {
            const float t0 = simd_sum(sm.x), t1 = simd_sum(sm.y), t2 = simd_sum(sm.z), t3 = simd_sum(sm.w);
            if (!lane) {
                const ulong base_out = ulong(tok0 + m) * 4ul * splits + split;
                verify_fused_detail::checked_write(partials, base_out + 0ul * splits, t0, error);
                verify_fused_detail::checked_write(partials, base_out + 1ul * splits, t1, error);
                verify_fused_detail::checked_write(partials, base_out + 2ul * splits, t2, error);
                verify_fused_detail::checked_write(partials, base_out + 3ul * splits, t3, error);
            }
        };
        emit_tok4(0u, s0);
        if (nt > 1u) emit_tok4(1u, s1);
        if (nt > 2u) emit_tok4(2u, s2);
        if (nt > 3u) emit_tok4(3u, s3);
        if (nt > 4u) emit_tok4(4u, s4);
        if (nt > 5u) emit_tok4(5u, s5);
        if (nt > 6u) emit_tok4(6u, s6);
        if (nt > 7u) emit_tok4(7u, s7);
        return;
    }

    const uint branch = group.x;
    if (sg >= 4u || branch >= branches || split >= splits || tok0 >= tokens) return;
    const uint nt = min(8u, tokens - tok0);
    device const float* w_row = weights + ulong(branch) * columns;
    device const float* x_base = normalized + ulong(tok0) * columns;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (uint col = begin + uint(lane); col < end; col += 32u) {
        const float w = w_row[col];
        device const float* x_col = x_base + col;
        a0 += w * x_col[0ul * columns];
        if (nt > 1u) a1 += w * x_col[1ul * columns];
        if (nt > 2u) a2 += w * x_col[2ul * columns];
        if (nt > 3u) a3 += w * x_col[3ul * columns];
        if (nt > 4u) a4 += w * x_col[4ul * columns];
        if (nt > 5u) a5 += w * x_col[5ul * columns];
        if (nt > 6u) a6 += w * x_col[6ul * columns];
        if (nt > 7u) a7 += w * x_col[7ul * columns];
    }
    const auto emit_one = [&](uint m, float sm) {
        const float total = simd_sum(sm);
        if (!lane) {
            verify_fused_detail::checked_write(partials, (ulong(tok0 + m) * branches + branch) * splits + split, total, error);
        }
    };
    emit_one(0u, a0);
    if (nt > 1u) emit_one(1u, a1);
    if (nt > 2u) emit_one(2u, a2);
    if (nt > 3u) emit_one(3u, a3);
    if (nt > 4u) emit_one(4u, a4);
    if (nt > 5u) emit_one(5u, a5);
    if (nt > 6u) emit_one(6u, a6);
    if (nt > 7u) emit_one(7u, a7);
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

// Coalesced vectorised multi-row gated_mix (final head stream mixer without inject/scales).
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
    const uint quads = (channels + 3u) >> 2u;
    const uint token = index / quads;
    const uint col4 = (index % quads) << 2u;
    if (token >= tokens) return;

    if ((channels & 3u) == 0u) {
        float4 total = float4(0.0f);
        bool4 ok = bool4(true);
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
        if (lane_ok) verify_fused_detail::checked_write(output, ulong(token) * channels + channel, total / float(branches), error);
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

// Fused multi-row residual stream write + next-sublayer per-branch RMS normalisation.
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
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint branches = shape.x;
    const uint channels = shape.y;
    const uint tokens = shape.z & 0xffffu;
    const uint shared_gains = shape.z >> 16u;
    const float eps = as_type<float>(shape.w);
    if (!branches || branches > 16u || channels < 32u || channels > 65536u || (channels % 32u) != 0u ||
        !tokens || tokens > 64u || !isfinite(eps) || eps < 0.0f) {
        if (!group.x && !group.y && !sg && !lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    threadgroup float sram_row[2560];
    const uint branch = group.x;
    const uint token = group.y;
    if (sg != 0u || branch >= branches || token >= tokens) return;

    const ulong base = (ulong(token) * branches + branch) * channels;
    const ulong x_base = ulong(token) * channels;
    const float sc = scales[ulong(token) * branches + branch];
    uint bad = isfinite(sc) ? 0u : 1u;
    float sum = 0.0f;

    if (channels == 2560u) {
        #pragma unroll 4
        for (uint k = 0u; k < 80u; ++k) {
            const uint j = uint(lane) + k * 32u;
            const float st = streams[base + j];
            const float xv = input[x_base + j];
            const float v = st + sc * xv;
            if (!isfinite(st) || !isfinite(xv) || !isfinite(v)) bad = 1u;
            output[base + j] = v;
            sram_row[j] = v;
            sum += v * v;
        }
        sum = simd_sum(sum);
        const float denominator = sum / 2560.0f + eps;
        if (simd_or(bad) || !isfinite(sum) || !isfinite(denominator) || denominator <= 0.0f) {
            if (!lane) verify_fused_detail::flag_nonfinite(error);
            return;
        }
        const float inv = rsqrt(denominator);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const ulong gain_base = shared_gains ? 0ul : ulong(branch) * 2560ul;
        uint bad2 = 0u;
        #pragma unroll 4
        for (uint k = 0u; k < 20u; ++k) {
            const uint col4 = uint(lane) * 4u + k * 128u;
            const float4 rv4 = *reinterpret_cast<threadgroup const float4*>(sram_row + col4);
            const float4 g4 = float4(*reinterpret_cast<device const packed_float4*>(gains + gain_base + col4));
            const float4 n4 = (rv4 * inv) * g4;
            if (!all(isfinite(g4)) || !all(isfinite(n4))) bad2 = 1u;
            *reinterpret_cast<device packed_float4*>(normalized + base + col4) = packed_float4(n4);
        }
        if (simd_or(bad2) && !lane) verify_fused_detail::flag_nonfinite(error);
        return;
    }

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

// Single-dispatch GDN alpha + beta FP32 gate projections (2560 -> 16) with threadgroup-staged input rows.
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

    device const float* x_rows[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const uint r = min(first + j, a.rows - 1u);
        x_rows[j] = reinterpret_cast<device const float*>(input_bytes + ulong(r) * a.input_stride_bytes);
    }

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint bad = 0u;

    if (nt == 4u) {
        for (uint k = uint(lane); k < a.columns; k += 32u) {
            const float w = w_row[k];
            const float x0 = x_rows[0][k];
            const float x1 = x_rows[1][k];
            const float x2 = x_rows[2][k];
            const float x3 = x_rows[3][k];
            if (!isfinite(w) || !isfinite(x0) || !isfinite(x1) || !isfinite(x2) || !isfinite(x3)) bad = 1u;
            sums[0] += w * x0;
            sums[1] += w * x1;
            sums[2] += w * x2;
            sums[3] += w * x3;
        }
    } else {
        for (uint k = uint(lane); k < a.columns; k += 32u) {
            const float w = w_row[k];
            if (!isfinite(w)) bad = 1u;
            for (uint j = 0u; j < nt; ++j) {
                const float xv = x_rows[j][k];
                if (!isfinite(xv)) bad = 1u;
                sums[j] += w * xv;
            }
        }
    }

    if (simd_or(bad) && !lane) verify_fused_detail::flag_nonfinite(error);

    for (uint j = 0u; j < nt; ++j) {
        const float total = simd_sum(sums[j]);
        if (!lane) {
            device float* y_row = reinterpret_cast<device float*>(out_base + ulong(first + j) * out_stride);
            verify_fused_detail::checked_write(y_row, out, total, error);
        }
    }
}
