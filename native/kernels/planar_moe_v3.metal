// Lossless planar weight pack v3 (tokspur-pack-v3) Metal 4.0 kernels.
// The pack keeps quantised bytes losslessly. Kernel paths choose FP32 or narrow
// operands and different reductions, so each path needs separate numerical qualification.
// Sticky flags check finite values across inputs, scales, conversions and outputs.
#pragma clang fp reassociate(off)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint planar_encoding [[function_constant(0)]]; // 7 (Q5_1), 8 (Q8_0), 12 (Q4_K), 13 (Q5_K)

// Extracted independently from the retained narrow-row prototype. Only the
// union kernels change channel rows per SIMD group; operands and the ascending
// reduction within each represented output retain their existing order.
#ifndef TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS
#define TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS 0
#endif
#if TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS != 0 && TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS != 1
#error TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS must be 0 or 1
#endif
constant uint planar_union_gate_rows = TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS ? 1u : 2u;
constant uint planar_union_down_rows = TOKSPUR_PRIVATE_VERIFY_NARROW_ROWS ? 2u : 4u;

#ifndef TOKSPUR_PRIVATE_PLANAR_COLUMN32
#define TOKSPUR_PRIVATE_PLANAR_COLUMN32 0
#endif

#ifndef TOKSPUR_PRIVATE_PLANAR_GU_STAGE64
#define TOKSPUR_PRIVATE_PLANAR_GU_STAGE64 0
#endif
#if TOKSPUR_PRIVATE_PLANAR_GU_STAGE64 && !TOKSPUR_PRIVATE_PLANAR_COLUMN32
#error TOKSPUR_PRIVATE_PLANAR_GU_STAGE64 requires TOKSPUR_PRIVATE_PLANAR_COLUMN32
#endif

namespace planar_v3_detail {

struct PlanarUnionArgs {
    uint experts, selected, tokens, hidden;
    uint channels, reserved0, reserved1, reserved2;
};
static_assert(sizeof(PlanarUnionArgs) == 32, "planar union args ABI");

struct PlanarSmallMoeArgs {
    uint columns, rows, experts, selected;
    uint tokens, use_union, reserved0, reserved1;
    ulong planar_expert_bytes, qs_offset, qh_offset, sm_offset, dm_offset;
    ulong input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(PlanarSmallMoeArgs) == 88, "planar small MoE args ABI");

struct PlanarNaxMoeArgs {
    uint columns, rows, experts, selected;
    uint tokens, maximum_tasks, write_bf16_gathered, reserved0;
    ulong planar_expert_bytes, qs_offset, qh_offset, sm_offset, dm_offset;
    ulong input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(PlanarNaxMoeArgs) == 88, "planar NAX MoE args ABI");

static inline uint nonfinite_bit(float v) {
    return (as_type<uint>(v) & 0x7f800000u) == 0x7f800000u ? 1u : 0u;
}

static inline uint nonfinite_vec2(float2 v) {
    return nonfinite_bit(v.x) | nonfinite_bit(v.y);
}

static inline uint nonfinite_vec4(float4 v) {
    return nonfinite_bit(v.x) | nonfinite_bit(v.y) | nonfinite_bit(v.z) | nonfinite_bit(v.w);
}

static inline uint nonfinite_half_bit(half v) {
    return (as_type<ushort>(v) & 0x7c00u) == 0x7c00u ? 1u : 0u;
}

static inline float sigmoid_f32(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x));
}

static inline ushort round_bf16_bits(float v, thread uint& bad) {
    const uint bits = as_type<uint>(v);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        bad |= 1u;
        return 0;
    }
    const ushort rounded = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
    if ((rounded & 0x7f80u) == 0x7f80u) {
        bad |= 1u;
        return 0;
    }
    return rounded;
}

static inline bfloat clean_bf16(float v, thread uint& bad) {
    return as_type<bfloat>(round_bf16_bits(v, bad));
}

static inline void publish_sticky_error(uint bad, device atomic_uint* error) {
    const uint sg_bad = simd_or(bad);
    if (simd_is_first() && sg_bad) {
        atomic_fetch_or_explicit(error, sg_bad, memory_order_relaxed);
    }
}

static inline uint route_index_v3(device const uint* ids, device const uint* map, uint expert, uint at,
                                  uint tokens, uint selected, thread uint& bad) {
    const uint rid = map[ulong(expert) * tokens + at];
    if (rid >= tokens * selected || ids[rid] != expert) {
        bad |= 4u;
        return UINT_MAX;
    }
    return rid;
}

#if !TOKSPUR_PRIVATE_PLANAR_COLUMN32
static inline half4x4 decode16_planar_v3(device const uchar* exp_base, uint out_row, uint ci,
                                         uint blocks_per_row, ulong qs_offset, ulong qh_offset,
                                         ulong sm_offset, ulong dm_offset, bool live_col,
                                         thread uint& bad) {
    half4x4 result(0);
    if (!live_col) return result;

    if (planar_encoding == 12u || planar_encoding == 13u) {
        const uint sb = ci >> 4u;
        if (sb >= blocks_per_row) return result;
        const ulong blk = ulong(out_row) * blocks_per_row + sb;
        const uint il16 = ci & 15u;
        const uint grp = il16 >> 1u;
        const uint pair = grp >> 1u;
        const uint shift = (grp & 1u) * 4u;
        const uint half32 = il16 & 1u;

        const half2 dm = reinterpret_cast<device const half2*>(exp_base + dm_offset)[blk];
        float d = float(dm.x), dmin = float(dm.y);
        uint local_bad = 0u;
        if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
            local_bad = 1u;
            d = 0.0f;
            dmin = 0.0f;
        }
        device const uchar* sm = exp_base + sm_offset + blk * 16ul;
        const float dl = d * float(sm[grp]);
        const float ml = dmin * float(sm[8u + grp]);
        if (nonfinite_bit(dl) || nonfinite_bit(ml)) {
            local_bad = 1u;
        }
        bad |= local_bad;

        const uint4 q4 = reinterpret_cast<device const uint4*>(exp_base + qs_offset + blk * 128ul)[pair * 2u + half32];
        const uint4 qh4 = (planar_encoding == 13u)
            ? reinterpret_cast<device const uint4*>(exp_base + qh_offset + blk * 32ul)[half32]
            : uint4(0u);
        const float q_max = (planar_encoding == 13u) ? 31.0f : 15.0f;
        const bool block_clean = !local_bad && (fabs(dl) * q_max + fabs(ml) <= 65504.0f);

        if (block_clean) {
            #pragma unroll
            for (ushort w = 0; w < 4; ++w) {
                uchar4 qb = (as_type<uchar4>(q4[w]) >> uchar4(shift)) & uchar4(15u);
                if (planar_encoding == 13u) {
                    qb |= ((as_type<uchar4>(qh4[w]) >> uchar4(grp)) & uchar4(1u)) << uchar4(4u);
                }
                result[w] = half4(float4(qb) * dl - ml);
            }
        } else {
            #pragma unroll
            for (ushort w = 0; w < 4; ++w) {
                uchar4 qb = (as_type<uchar4>(q4[w]) >> uchar4(shift)) & uchar4(15u);
                if (planar_encoding == 13u) {
                    qb |= ((as_type<uchar4>(qh4[w]) >> uchar4(grp)) & uchar4(1u)) << uchar4(4u);
                }
                const float4 wf = float4(qb) * dl - ml;
                bad |= nonfinite_bit(wf.x) | nonfinite_bit(wf.y) | nonfinite_bit(wf.z) | nonfinite_bit(wf.w);
                const half4 wh = half4(wf);
                bad |= nonfinite_half_bit(wh.x) | nonfinite_half_bit(wh.y) |
                       nonfinite_half_bit(wh.z) | nonfinite_half_bit(wh.w);
                result[w] = wh;
            }
        }
    } else if (planar_encoding == 7u) {
        const uint b_idx = ci >> 1u;
        if (b_idx >= blocks_per_row) return result;
        const uint half32 = ci & 1u;
        const ulong blk = ulong(out_row) * blocks_per_row + b_idx;

        const half2 dm = reinterpret_cast<device const half2*>(exp_base + dm_offset)[blk];
        float d = float(dm.x), m = float(dm.y);
        uint local_bad = 0u;
        if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
            local_bad = 1u;
            d = 0.0f;
            m = 0.0f;
        }
        bad |= local_bad;
        const uint4 q4 = reinterpret_cast<device const uint4*>(exp_base + qs_offset)[blk];
        const uint qh = reinterpret_cast<device const uint*>(exp_base + qh_offset)[blk] >> (half32 * 16u);
        const uint shift = half32 * 4u;
        const bool block_clean = !local_bad && (fabs(d) * 31.0f + fabs(m) <= 65504.0f);

        if (block_clean) {
            #pragma unroll
            for (ushort w = 0; w < 4; ++w) {
                const uint qh4 = qh >> (uint(w) * 4u);
                const uchar4 hb = uchar4(qh4 & 1u, (qh4 >> 1u) & 1u, (qh4 >> 2u) & 1u, (qh4 >> 3u) & 1u) << uchar4(4u);
                const uchar4 qb = ((as_type<uchar4>(q4[w]) >> uchar4(shift)) & uchar4(15u)) | hb;
                result[w] = half4(float4(qb) * d + m);
            }
        } else {
            #pragma unroll
            for (ushort w = 0; w < 4; ++w) {
                const uint qh4 = qh >> (uint(w) * 4u);
                const uchar4 hb = uchar4(qh4 & 1u, (qh4 >> 1u) & 1u, (qh4 >> 2u) & 1u, (qh4 >> 3u) & 1u) << uchar4(4u);
                const uchar4 qb = ((as_type<uchar4>(q4[w]) >> uchar4(shift)) & uchar4(15u)) | hb;
                const float4 wf = float4(qb) * d + m;
                bad |= nonfinite_bit(wf.x) | nonfinite_bit(wf.y) | nonfinite_bit(wf.z) | nonfinite_bit(wf.w);
                const half4 wh = half4(wf);
                bad |= nonfinite_half_bit(wh.x) | nonfinite_half_bit(wh.y) |
                       nonfinite_half_bit(wh.z) | nonfinite_half_bit(wh.w);
                result[w] = wh;
            }
        }
    } else if (planar_encoding == 8u) {
        const uint b_idx = ci >> 1u;
        if (b_idx >= blocks_per_row) return result;
        const uint half32 = ci & 1u;
        const ulong blk = ulong(out_row) * blocks_per_row + b_idx;

        const half dh = reinterpret_cast<device const half*>(exp_base + dm_offset)[blk];
        const ushort abs_sc = as_type<ushort>(dh) & 0x7fffu;
        const uint4 q4 = reinterpret_cast<device const uint4*>(exp_base + qs_offset)[blk * 2ul + half32];
        if (abs_sc >= 0x0400u && abs_sc < 0x6000u) {
            result[0] = half4(as_type<char4>(q4.x)) * dh;
            result[1] = half4(as_type<char4>(q4.y)) * dh;
            result[2] = half4(as_type<char4>(q4.z)) * dh;
            result[3] = half4(as_type<char4>(q4.w)) * dh;
        } else if (abs_sc != 0u) {
            float d = float(dh);
            if (nonfinite_half_bit(dh)) {
                bad |= 1u;
                return result;
            }
            #pragma unroll
            for (ushort w = 0; w < 4; ++w) {
                const float4 f4 = float4(as_type<char4>(q4[w])) * d;
                bad |= nonfinite_bit(f4.x) | nonfinite_bit(f4.y) | nonfinite_bit(f4.z) | nonfinite_bit(f4.w);
                const half4 h4 = half4(f4);
                bad |= nonfinite_half_bit(h4.x) | nonfinite_half_bit(h4.y) |
                       nonfinite_half_bit(h4.z) | nonfinite_half_bit(h4.w);
                result[w] = h4;
            }
        }
    }
    return result;
}
#endif

#if TOKSPUR_PRIVATE_PLANAR_COLUMN32
// Decode the current 32-input group for one output column per lane, sharing its stored scale and packed Q5_1 codes across both halves.
static inline void decode32_planar_v3(device const uchar* exp_base, uint out_row, uint group32,
    uint blocks_per_row, ulong qs_offset, ulong qh_offset, ulong sm_offset, ulong dm_offset,
    bool live_col, thread half4x4& lo, thread half4x4& hi, thread uint& bad) {
    lo = half4x4(0);
    hi = half4x4(0);
    if (!live_col) return;

    if (planar_encoding == 12u || planar_encoding == 13u) {
        const uint sb = group32 >> 3u;
        if (sb >= blocks_per_row) return;
        const ulong blk = ulong(out_row) * blocks_per_row + sb;
        const uint grp = group32 & 7u;
        const uint pair = grp >> 1u;
        const uint shift = (grp & 1u) * 4u;
        const half2 dm = reinterpret_cast<device const half2*>(exp_base + dm_offset)[blk];
        float d = float(dm.x), dmin = float(dm.y);
        uint local_bad = 0u;
        if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
            local_bad = 1u;
            d = 0.0f;
            dmin = 0.0f;
        }
        device const uchar* sm = exp_base + sm_offset + blk * 16ul;
        const float dl = d * float(sm[grp]);
        const float ml = dmin * float(sm[8u + grp]);
        if (nonfinite_bit(dl) || nonfinite_bit(ml)) local_bad = 1u;
        bad |= local_bad;
        device const uint4* qs = reinterpret_cast<device const uint4*>(exp_base + qs_offset + blk * 128ul);
        const uint4 qlo = qs[pair * 2u], qhi = qs[pair * 2u + 1u];
        device const uint4* qh = reinterpret_cast<device const uint4*>(exp_base + qh_offset + blk * 32ul);
        const uint4 hlo = planar_encoding == 13u ? qh[0] : uint4(0u);
        const uint4 hhi = planar_encoding == 13u ? qh[1] : uint4(0u);
        const float q_max = planar_encoding == 13u ? 31.0f : 15.0f;
        const bool block_clean = !local_bad && (fabs(dl) * q_max + fabs(ml) <= 65504.0f);
        #pragma unroll
        for (ushort w = 0; w < 4; ++w) {
            uchar4 q0 = (as_type<uchar4>(qlo[w]) >> uchar4(shift)) & uchar4(15u);
            uchar4 q1 = (as_type<uchar4>(qhi[w]) >> uchar4(shift)) & uchar4(15u);
            if (planar_encoding == 13u) {
                q0 |= ((as_type<uchar4>(hlo[w]) >> uchar4(grp)) & uchar4(1u)) << uchar4(4u);
                q1 |= ((as_type<uchar4>(hhi[w]) >> uchar4(grp)) & uchar4(1u)) << uchar4(4u);
            }
            const float4 f0 = float4(q0) * dl - ml;
            const float4 f1 = float4(q1) * dl - ml;
            lo[w] = half4(f0);
            hi[w] = half4(f1);
            if (!block_clean) {
                bad |= nonfinite_vec4(f0) | nonfinite_vec4(f1);
                bad |= nonfinite_half_bit(lo[w].x) | nonfinite_half_bit(lo[w].y) | nonfinite_half_bit(lo[w].z) | nonfinite_half_bit(lo[w].w);
                bad |= nonfinite_half_bit(hi[w].x) | nonfinite_half_bit(hi[w].y) | nonfinite_half_bit(hi[w].z) | nonfinite_half_bit(hi[w].w);
            }
        }
    } else if (planar_encoding == 7u) {
        if (group32 >= blocks_per_row) return;
        const ulong blk = ulong(out_row) * blocks_per_row + group32;
        const half2 dm = reinterpret_cast<device const half2*>(exp_base + dm_offset)[blk];
        float d = float(dm.x), m = float(dm.y);
        uint local_bad = 0u;
        if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
            local_bad = 1u;
            d = 0.0f;
            m = 0.0f;
        }
        bad |= local_bad;
        const uint4 q4 = reinterpret_cast<device const uint4*>(exp_base + qs_offset)[blk];
        const uint qh = reinterpret_cast<device const uint*>(exp_base + qh_offset)[blk];
        const bool block_clean = !local_bad && (fabs(d) * 31.0f + fabs(m) <= 65504.0f);
        #pragma unroll
        for (ushort w = 0; w < 4; ++w) {
            const uint h0 = qh >> (uint(w) * 4u), h1 = qh >> (16u + uint(w) * 4u);
            const uchar4 b0 = uchar4(h0 & 1u, (h0 >> 1u) & 1u, (h0 >> 2u) & 1u, (h0 >> 3u) & 1u) << uchar4(4u);
            const uchar4 b1 = uchar4(h1 & 1u, (h1 >> 1u) & 1u, (h1 >> 2u) & 1u, (h1 >> 3u) & 1u) << uchar4(4u);
            const uchar4 q0 = (as_type<uchar4>(q4[w]) & uchar4(15u)) | b0;
            const uchar4 q1 = (as_type<uchar4>(q4[w]) >> uchar4(4u)) | b1;
            const float4 f0 = float4(q0) * d + m;
            const float4 f1 = float4(q1) * d + m;
            lo[w] = half4(f0);
            hi[w] = half4(f1);
            if (!block_clean) {
                bad |= nonfinite_vec4(f0) | nonfinite_vec4(f1);
                bad |= nonfinite_half_bit(lo[w].x) | nonfinite_half_bit(lo[w].y) | nonfinite_half_bit(lo[w].z) | nonfinite_half_bit(lo[w].w);
                bad |= nonfinite_half_bit(hi[w].x) | nonfinite_half_bit(hi[w].y) | nonfinite_half_bit(hi[w].z) | nonfinite_half_bit(hi[w].w);
            }
        }
    } else if (planar_encoding == 8u) {
        if (group32 >= blocks_per_row) return;
        const ulong blk = ulong(out_row) * blocks_per_row + group32;
        const half dh = reinterpret_cast<device const half*>(exp_base + dm_offset)[blk];
        const ushort abs_sc = as_type<ushort>(dh) & 0x7fffu;
        const uint4 q0 = reinterpret_cast<device const uint4*>(exp_base + qs_offset)[blk * 2ul];
        const uint4 q1 = reinterpret_cast<device const uint4*>(exp_base + qs_offset)[blk * 2ul + 1ul];
        if (abs_sc >= 0x0400u && abs_sc < 0x6000u) {
            #pragma unroll
            for (ushort w = 0; w < 4; ++w) {
                lo[w] = half4(as_type<char4>(q0[w])) * dh;
                hi[w] = half4(as_type<char4>(q1[w])) * dh;
            }
        } else if (abs_sc != 0u) {
            if (nonfinite_half_bit(dh)) { bad |= 1u; return; }
            const float d = float(dh);
            #pragma unroll
            for (ushort w = 0; w < 4; ++w) {
                const float4 f0 = float4(as_type<char4>(q0[w])) * d;
                const float4 f1 = float4(as_type<char4>(q1[w])) * d;
                bad |= nonfinite_vec4(f0) | nonfinite_vec4(f1);
                lo[w] = half4(f0);
                hi[w] = half4(f1);
                bad |= nonfinite_half_bit(lo[w].x) | nonfinite_half_bit(lo[w].y) | nonfinite_half_bit(lo[w].z) | nonfinite_half_bit(lo[w].w);
                bad |= nonfinite_half_bit(hi[w].x) | nonfinite_half_bit(hi[w].y) | nonfinite_half_bit(hi[w].z) | nonfinite_half_bit(hi[w].w);
            }
        }
    }
}
#endif

static inline void unpack_q4k_u8_pair_v3(uint2 q_u2, uint2 qh_u2, uint grp0, uint grp1,
                                         thread uchar4& q0_lo, thread uchar4& q0_hi,
                                         thread uchar4& q1_lo, thread uchar4& q1_hi) {
    const uchar4 b0 = as_type<uchar4>(q_u2.x);
    const uchar4 b1 = as_type<uchar4>(q_u2.y);
    q0_lo = b0 & uchar4(15u);
    q1_lo = b0 >> uchar4(4u);
    q0_hi = b1 & uchar4(15u);
    q1_hi = b1 >> uchar4(4u);
    if (planar_encoding == 13u) {
        const uchar4 h0 = as_type<uchar4>(qh_u2.x);
        const uchar4 h1 = as_type<uchar4>(qh_u2.y);
        q0_lo |= ((h0 >> uchar4(grp0)) & uchar4(1u)) << uchar4(4u);
        q1_lo |= ((h0 >> uchar4(grp1)) & uchar4(1u)) << uchar4(4u);
        q0_hi |= ((h1 >> uchar4(grp0)) & uchar4(1u)) << uchar4(4u);
        q1_hi |= ((h1 >> uchar4(grp1)) & uchar4(1u)) << uchar4(4u);
    }
}

static inline float dot8_u8_f32_v3(uchar4 q_lo, uchar4 q_hi, float4 x_lo, float4 x_hi) {
    float acc = 0.0f;
    acc += float(q_lo.x) * x_lo.x;
    acc += float(q_lo.y) * x_lo.y;
    acc += float(q_lo.z) * x_lo.z;
    acc += float(q_lo.w) * x_lo.w;
    acc += float(q_hi.x) * x_hi.x;
    acc += float(q_hi.y) * x_hi.y;
    acc += float(q_hi.z) * x_hi.z;
    acc += float(q_hi.w) * x_hi.w;
    return acc;
}

} // namespace planar_v3_detail

// Deduplicates up to 8 * 16 = 128 routes into U unique experts and writes indirect dispatch grids:
// Four SIMD groups times the selected channel rows per group form each grid tile.
// meta[0..2] = {gate_tiles, U, 1}, meta[3..5] = {down_tiles, U, 1}, meta[6] = U.
kernel void planar_moe_build_union_m8(device const uint* indices [[buffer(0)]],
                                      device uint* meta [[buffer(1)]],
                                      constant planar_v3_detail::PlanarUnionArgs& a [[buffer(2)]],
                                      device atomic_uint* error [[buffer(7)]],
                                      ushort lane [[thread_index_in_simdgroup]],
                                      ushort tid [[thread_index_in_threadgroup]]) {
    if (tid >= 32u) return;
    if (!a.experts || a.experts > 512u || a.experts % 2u != 0u ||
        !a.selected || a.selected > 16u || a.selected > a.experts ||
        !a.tokens || a.tokens > 8u || !a.hidden || !a.channels) {
        if (!lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        if (lane < 8u) meta[lane] = 0u;
        return;
    }

    const uint total = a.tokens * a.selected;
    uint any_oob = 0u;
    for (uint flat = lane; flat < total; flat += 32u) {
        if (indices[flat] >= a.experts) any_oob = 1u;
    }
    if (simd_or(any_oob) != 0u) {
        if (!lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        if (lane < 8u) meta[lane] = 0u;
        return;
    }

    threadgroup uint tg_exp[128];
    threadgroup uint tg_cnt[128];
    device uint* u_experts = meta + 8u;
    device uint* u_counts = meta + 136u;
    device uint* u_tokens = meta + 264u;
    device uint* u_routes = meta + 1288u;

    const uint m_cap = (a.tokens <= 4u) ? 4u : 8u;
    uint U = 0u;
    for (uint flat = 0u; flat < total; ++flat) {
        const uint e = indices[flat];
        const uint tok = flat / a.selected;
        int found = -1;
        for (uint base = 0u; base < U; base += 32u) {
            const uint u = base + uint(lane);
            const bool match = (u < U) && (tg_exp[u] == e) && (tg_cnt[u] < m_cap);
            const uint best = simd_min(match ? u : 0xffffffffu);
            if (best != 0xffffffffu) {
                found = int(best);
                break;
            }
        }
        if (found >= 0) {
            if (!lane) {
                const uint slot = uint(found);
                const uint k = tg_cnt[slot];
                tg_cnt[slot] = k + 1u;
                u_tokens[slot * 8u + k] = tok;
                u_routes[slot * 8u + k] = flat;
            }
        } else {
            if (!lane) {
                tg_exp[U] = e;
                tg_cnt[U] = 1u;
            }
            if (lane < 8u) {
                u_tokens[U * 8u + lane] = tok;
                u_routes[U * 8u + lane] = flat;
            }
            ++U;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint u = lane; u < U; u += 32u) {
        u_experts[u] = tg_exp[u];
        u_counts[u] = tg_cnt[u];
    }
    if (!lane) {
        meta[0] = (a.hidden + 4u * planar_union_gate_rows - 1u) / (4u * planar_union_gate_rows);
        meta[1] = U;
        meta[2] = 1u;
        meta[3] = (a.channels + 4u * planar_union_down_rows - 1u) / (4u * planar_union_down_rows);
        meta[4] = U;
        meta[5] = 1u;
        meta[6] = U;
        meta[7] = 0u;
    }
}

// Dedicated m = 1 direct co-located gate + up + SwiGLU kernel (zero threadgroup SRAM, zero barriers,
// 1 output hidden row = 2 gate/up rows per simdgroup, 4 output rows per 128-thread group).
kernel void planar_moe_gate_up_swiglu_m1(device const uchar* pair_lo [[buffer(0)]],
                                         device const uchar* pair_hi [[buffer(1)]],
                                         device const float* input [[buffer(2)]],
                                         device float* expert_hidden [[buffer(3)]],
                                         device const uint* indices [[buffer(4)]],
                                         constant planar_v3_detail::PlanarSmallMoeArgs& a [[buffer(5)]],
                                         device atomic_uint* error [[buffer(7)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         ushort sg [[simdgroup_index_in_threadgroup]],
                                         ushort lane [[thread_index_in_simdgroup]],
                                         ushort tid [[thread_index_in_threadgroup]]) {
    const uint block_cols = (planar_encoding == 12u || planar_encoding == 13u) ? 256u : 32u;
    if (a.tokens != 1u || a.use_union != 0u || !a.selected || a.selected > 16u ||
        !a.experts || a.experts > 512u || a.experts % 2u != 0u ||
        !a.columns || a.columns % block_cols != 0u || !a.rows || a.rows % 4u != 0u ||
        a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    constexpr ushort kRows = 2;
    const uint first_row = (group.x * 4u + uint(sg)) * uint(kRows);
    if (group.y >= a.selected || first_row >= a.rows) return;
    const uint active_rows = min(uint(kRows), a.rows - first_row);

    const uint expert = indices[group.y];
    if (expert >= a.experts) {
        if (!lane) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            device float* out = reinterpret_cast<device float*>(
                reinterpret_cast<device uchar*>(expert_hidden) + ulong(group.y) * a.output_stride_bytes);
            for (uint r = 0u; r < active_rows; ++r) out[first_row + r] = 0.0f;
        }
        return;
    }

    const uint half_experts = a.experts >> 1u;
    device const uchar* half_base = (expert < half_experts) ? pair_lo : pair_hi;
    const uint local_e = (expert < half_experts) ? expert : (expert - half_experts);
    device const uchar* gate_exp = half_base + ulong(2u * local_e) * a.planar_expert_bytes;
    device const uchar* up_exp = gate_exp + a.planar_expert_bytes;

    const uint blocks_per_row = a.columns / block_cols;

    float sum_g[kRows] = {0.0f};
    float sum_u[kRows] = {0.0f};
    uint bad = 0u;

    if (planar_encoding == 12u || planar_encoding == 13u) {
        const ushort ix = lane >> 4u;
        const ushort sub16 = lane & 15u;
        const ushort pair = sub16 >> 2u;
        const ushort ir = sub16 & 3u;
        const uint grp0 = uint(pair) * 2u;
        const uint grp1 = grp0 + 1u;

        for (uint sb0 = 0u; sb0 < blocks_per_row; sb0 += 2u) {
            const uint sblocks = min(2u, blocks_per_row - sb0);
            if (ix < sblocks) {
                const uint col_base0 = (sb0 + uint(ix)) * 256u + uint(pair) * 64u + uint(ir) * 8u;
                const float4 x0_lo = float4(*reinterpret_cast<device const packed_float4*>(input + col_base0));
                const float4 x0_hi = float4(*reinterpret_cast<device const packed_float4*>(input + col_base0 + 4u));
                const float4 x1_lo = float4(*reinterpret_cast<device const packed_float4*>(input + col_base0 + 32u));
                const float4 x1_hi = float4(*reinterpret_cast<device const packed_float4*>(input + col_base0 + 36u));
                bad |= planar_v3_detail::nonfinite_vec4(x0_lo) | planar_v3_detail::nonfinite_vec4(x0_hi) |
                       planar_v3_detail::nonfinite_vec4(x1_lo) | planar_v3_detail::nonfinite_vec4(x1_hi);

                const float sum_x0 = (x0_lo.x + x0_lo.y + x0_lo.z + x0_lo.w) + (x0_hi.x + x0_hi.y + x0_hi.z + x0_hi.w);
                const float sum_x1 = (x1_lo.x + x1_lo.y + x1_lo.z + x1_lo.w) + (x1_hi.x + x1_hi.y + x1_hi.z + x1_hi.w);
                const float x0[8] = {x0_lo.x, x0_lo.y, x0_lo.z, x0_lo.w, x0_hi.x, x0_hi.y, x0_hi.z, x0_hi.w};
                const float x1[8] = {x1_lo.x, x1_lo.y, x1_lo.z, x1_lo.w, x1_hi.x, x1_hi.y, x1_hi.z, x1_hi.w};

                #pragma unroll
                for (ushort r = 0; r < kRows; ++r) {
                    if (r < active_rows) {
                        const ulong blk_base = ulong(first_row + uint(r)) * blocks_per_row + sb0;
                        const ulong blk = blk_base + uint(ix);

                        const half2 gdm = reinterpret_cast<device const half2*>(gate_exp + a.dm_offset)[blk];
                        const half2 udm = reinterpret_cast<device const half2*>(up_exp + a.dm_offset)[blk];
                        float gd = float(gdm.x), gm = float(gdm.y);
                        float ud = float(udm.x), um = float(udm.y);
                        if (planar_v3_detail::nonfinite_half_bit(gdm.x) || planar_v3_detail::nonfinite_half_bit(gdm.y)) {
                            bad |= 1u; gd = 0.0f; gm = 0.0f;
                        }
                        if (planar_v3_detail::nonfinite_half_bit(udm.x) || planar_v3_detail::nonfinite_half_bit(udm.y)) {
                            bad |= 1u; ud = 0.0f; um = 0.0f;
                        }

                        const ushort gsc01 = reinterpret_cast<device const ushort*>(gate_exp + a.sm_offset + blk * 16ul)[pair];
                        const ushort gmn01 = reinterpret_cast<device const ushort*>(gate_exp + a.sm_offset + blk * 16ul + 8ul)[pair];
                        const ushort usc01 = reinterpret_cast<device const ushort*>(up_exp + a.sm_offset + blk * 16ul)[pair];
                        const ushort umn01 = reinterpret_cast<device const ushort*>(up_exp + a.sm_offset + blk * 16ul + 8ul)[pair];

                        const uint2 gq_u2 = reinterpret_cast<device const uint2*>(gate_exp + a.qs_offset + blk_base * 128ul)[lane];
                        const uint2 uq_u2 = reinterpret_cast<device const uint2*>(up_exp + a.qs_offset + blk_base * 128ul)[lane];
                        const uint2 gqh_u2 = (planar_encoding == 13u)
                            ? reinterpret_cast<device const uint2*>(gate_exp + a.qh_offset + blk * 32ul)[ir]
                            : uint2(0u);
                        const uint2 uqh_u2 = (planar_encoding == 13u)
                            ? reinterpret_cast<device const uint2*>(up_exp + a.qh_offset + blk * 32ul)[ir]
                            : uint2(0u);

                        float g_dot0 = 0.0f, g_dot1 = 0.0f;
                        float u_dot0 = 0.0f, u_dot1 = 0.0f;
                        #pragma unroll
                        for (ushort w = 0; w < 2; ++w) {
                            const uint gqw = gq_u2[w];
                            const uint uqw = uq_u2[w];
                            const uint ghw = gqh_u2[w];
                            const uint uhw = uqh_u2[w];
                            #pragma unroll
                            for (ushort b = 0; b < 4; ++b) {
                                const ushort idx = w * 4u + b;
                                const uint gb = (gqw >> (8u * b)) & 0xffu;
                                const uint ub = (uqw >> (8u * b)) & 0xffu;
                                uint gq0 = gb & 15u, gq1 = gb >> 4u;
                                uint uq0 = ub & 15u, uq1 = ub >> 4u;
                                if (planar_encoding == 13u) {
                                    const uint ghb = (ghw >> (8u * b)) & 0xffu;
                                    const uint uhb = (uhw >> (8u * b)) & 0xffu;
                                    gq0 |= ((ghb >> grp0) & 1u) << 4u;
                                    gq1 |= ((ghb >> grp1) & 1u) << 4u;
                                    uq0 |= ((uhb >> grp0) & 1u) << 4u;
                                    uq1 |= ((uhb >> grp1) & 1u) << 4u;
                                }
                                g_dot0 += float(gq0) * x0[idx];
                                g_dot1 += float(gq1) * x1[idx];
                                u_dot0 += float(uq0) * x0[idx];
                                u_dot1 += float(uq1) * x1[idx];
                            }
                        }
                        sum_g[r] += (gd * float(gsc01 & 0xffu) * g_dot0 - gm * float(gmn01 & 0xffu) * sum_x0) +
                                    (gd * float(gsc01 >> 8u) * g_dot1 - gm * float(gmn01 >> 8u) * sum_x1);
                        sum_u[r] += (ud * float(usc01 & 0xffu) * u_dot0 - um * float(umn01 & 0xffu) * sum_x0) +
                                    (ud * float(usc01 >> 8u) * u_dot1 - um * float(umn01 >> 8u) * sum_x1);
                    }
                }
            }
        }
    } else if (planar_encoding == 8u) {
        const ushort ib4 = lane >> 3u;
        uint k_blk = 0u;
        for (; k_blk + 4u <= blocks_per_row; k_blk += 4u) {
            const float4 x4 = float4(reinterpret_cast<device const packed_float4*>(input + k_blk * 32u)[lane]);
            bad |= planar_v3_detail::nonfinite_vec4(x4);
            #pragma unroll
            for (ushort r = 0; r < kRows; ++r) {
                if (r < active_rows) {
                    const ulong blk_base = ulong(first_row + uint(r)) * blocks_per_row + k_blk;
                    const ulong blk_lane = blk_base + uint(ib4);
                    const half gd_h = reinterpret_cast<device const half*>(gate_exp + a.dm_offset)[blk_lane];
                    const half ud_h = reinterpret_cast<device const half*>(up_exp + a.dm_offset)[blk_lane];
                    float gd_f = float(gd_h), ud_f = float(ud_h);
                    if (planar_v3_detail::nonfinite_half_bit(gd_h)) { bad |= 1u; gd_f = 0.0f; }
                    if (planar_v3_detail::nonfinite_half_bit(ud_h)) { bad |= 1u; ud_f = 0.0f; }
                    const char4 gq4 = reinterpret_cast<device const char4*>(gate_exp + a.qs_offset + blk_base * 32ul)[lane];
                    const char4 uq4 = reinterpret_cast<device const char4*>(up_exp + a.qs_offset + blk_base * 32ul)[lane];
                    sum_g[r] += dot(float4(gq4) * gd_f, x4);
                    sum_u[r] += dot(float4(uq4) * ud_f, x4);
                }
            }
        }
        for (; k_blk < blocks_per_row; ++k_blk) {
            const float xv = input[k_blk * 32u + lane];
            bad |= planar_v3_detail::nonfinite_bit(xv);
            #pragma unroll
            for (ushort r = 0; r < kRows; ++r) {
                if (r < active_rows) {
                    const ulong blk = ulong(first_row + uint(r)) * blocks_per_row + k_blk;
                    const half gd_h = reinterpret_cast<device const half*>(gate_exp + a.dm_offset)[blk];
                    const half ud_h = reinterpret_cast<device const half*>(up_exp + a.dm_offset)[blk];
                    float gd_f = float(gd_h), ud_f = float(ud_h);
                    if (planar_v3_detail::nonfinite_half_bit(gd_h)) { bad |= 1u; gd_f = 0.0f; }
                    if (planar_v3_detail::nonfinite_half_bit(ud_h)) { bad |= 1u; ud_f = 0.0f; }
                    const char gq = reinterpret_cast<device const char*>(gate_exp + a.qs_offset + blk * 32ul)[lane];
                    const char uq = reinterpret_cast<device const char*>(up_exp + a.qs_offset + blk * 32ul)[lane];
                    sum_g[r] += (float(gq) * gd_f) * xv;
                    sum_u[r] += (float(uq) * ud_f) * xv;
                }
            }
        }
    }

    device float* out = reinterpret_cast<device float*>(
        reinterpret_cast<device uchar*>(expert_hidden) + ulong(group.y) * a.output_stride_bytes);
    #pragma unroll
    for (ushort r = 0; r < kRows; ++r) {
        const float g_tot = simd_sum(sum_g[r]);
        const float u_tot = simd_sum(sum_u[r]);
        if (r < active_rows && !lane) {
            bad |= planar_v3_detail::nonfinite_bit(g_tot) | planar_v3_detail::nonfinite_bit(u_tot);
            const float act = (g_tot * planar_v3_detail::sigmoid_f32(g_tot)) * u_tot;
            bad |= planar_v3_detail::nonfinite_bit(act);
            out[first_row + r] = act;
        }
    }
    planar_v3_detail::publish_sticky_error(bad, error);
}

namespace planar_v3_detail {

template <ushort MCap, ushort MaxTokens = MCap>
static inline void planar_moe_gate_up_swiglu_union_impl(
    device const uchar* pair_lo,
    device const uchar* pair_hi,
    device const float* input,
    device float* expert_hidden,
    device const uint* meta,
    constant PlanarSmallMoeArgs& a,
    device atomic_uint* error,
    uint2 group,
    ushort sg,
    ushort lane,
    ushort tid
) {
    const uint block_cols = (planar_encoding == 12u || planar_encoding == 13u) ? 256u : 32u;
    if (!a.tokens || a.tokens > MaxTokens || !a.selected || a.selected > 16u || !a.experts || a.experts > 512u ||
        a.experts % 2u != 0u || !a.columns || a.columns % block_cols != 0u || !a.rows || a.rows % 4u != 0u ||
        a.use_union != 1u || a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint U = meta[6];
    if (group.y >= U) return;
    const uint expert = meta[8u + group.y];
    const uint n = meta[136u + group.y];
    if (expert >= a.experts || n == 0u || n > MCap) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    device const uint* tok_meta = meta + 264u + group.y * 8u;
    device const uint* flat_meta = meta + 1288u + group.y * 8u;
    for (uint k = 0u; k < n; ++k) {
        if (tok_meta[k] >= a.tokens || flat_meta[k] >= a.tokens * a.selected) {
            if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            return;
        }
    }

    constexpr ushort kRows = planar_union_gate_rows;
    const uint first_row = (group.x * 4u + uint(sg)) * uint(kRows);
    const bool sg_live = first_row < a.rows;
    const uint active_rows = sg_live ? min(uint(kRows), a.rows - first_row) : 0u;

    const uint half_experts = a.experts >> 1u;
    device const uchar* half_base = (expert < half_experts) ? pair_lo : pair_hi;
    const uint local_e = (expert < half_experts) ? expert : (expert - half_experts);
    device const uchar* gate_exp = half_base + ulong(2u * local_e) * a.planar_expert_bytes;
    device const uchar* up_exp = gate_exp + a.planar_expert_bytes;

    const uint blocks_per_row = a.columns / block_cols;
    const ulong in_stride_f32 = a.input_stride_bytes / sizeof(float);

    uint bad = 0u;
    if (n == 1u) {
        float sum_g0[kRows] = {0.0f};
        float sum_u0[kRows] = {0.0f};
        const uint tok0 = tok_meta[0];
        const uint flat0 = flat_meta[0];
        if (sg_live) {
            device const float* x_src = input + ulong(tok0) * in_stride_f32;
            if (planar_encoding == 12u || planar_encoding == 13u) {
                const ushort ix = lane >> 4u;
                const ushort sub16 = lane & 15u;
                const ushort pair = sub16 >> 2u;
                const ushort ir = sub16 & 3u;
                const uint grp0 = uint(pair) * 2u;
                const uint grp1 = grp0 + 1u;

                for (uint sb0 = 0u; sb0 < blocks_per_row; sb0 += 2u) {
                    const uint sblocks = min(2u, blocks_per_row - sb0);
                    if (ix < sblocks) {
                        const uint col_base0 = (sb0 + uint(ix)) * 256u + uint(pair) * 64u + uint(ir) * 8u;
                        const float4 x0_lo = float4(*reinterpret_cast<device const packed_float4*>(x_src + col_base0));
                        const float4 x0_hi = float4(*reinterpret_cast<device const packed_float4*>(x_src + col_base0 + 4u));
                        const float4 x1_lo = float4(*reinterpret_cast<device const packed_float4*>(x_src + col_base0 + 32u));
                        const float4 x1_hi = float4(*reinterpret_cast<device const packed_float4*>(x_src + col_base0 + 36u));
                        bad |= nonfinite_vec4(x0_lo) | nonfinite_vec4(x0_hi) |
                               nonfinite_vec4(x1_lo) | nonfinite_vec4(x1_hi);
                        const float sum_x0 = (x0_lo.x + x0_lo.y + x0_lo.z + x0_lo.w) + (x0_hi.x + x0_hi.y + x0_hi.z + x0_hi.w);
                        const float sum_x1 = (x1_lo.x + x1_lo.y + x1_lo.z + x1_lo.w) + (x1_hi.x + x1_hi.y + x1_hi.z + x1_hi.w);

                        #pragma unroll
                        for (ushort r = 0; r < kRows; ++r) {
                            if (r < active_rows) {
                                const uint prow = first_row + uint(r);
                                const ulong blk_base = ulong(prow) * blocks_per_row + sb0;
                                const ulong blk = blk_base + uint(ix);

                                const half2 gdm = reinterpret_cast<device const half2*>(gate_exp + a.dm_offset)[blk];
                                const half2 udm = reinterpret_cast<device const half2*>(up_exp + a.dm_offset)[blk];
                                float gd_f = float(gdm.x), gm_f = float(gdm.y);
                                float ud_f = float(udm.x), um_f = float(udm.y);
                                if (nonfinite_half_bit(gdm.x) || nonfinite_half_bit(gdm.y)) {
                                    bad |= 1u; gd_f = 0.0f; gm_f = 0.0f;
                                }
                                if (nonfinite_half_bit(udm.x) || nonfinite_half_bit(udm.y)) {
                                    bad |= 1u; ud_f = 0.0f; um_f = 0.0f;
                                }

                                const ushort gsc01 = reinterpret_cast<device const ushort*>(gate_exp + a.sm_offset + blk * 16ul)[pair];
                                const ushort gmn01 = reinterpret_cast<device const ushort*>(gate_exp + a.sm_offset + blk * 16ul + 8ul)[pair];
                                const ushort usc01 = reinterpret_cast<device const ushort*>(up_exp + a.sm_offset + blk * 16ul)[pair];
                                const ushort umn01 = reinterpret_cast<device const ushort*>(up_exp + a.sm_offset + blk * 16ul + 8ul)[pair];

                                const uint2 gq_u2 = reinterpret_cast<device const uint2*>(gate_exp + a.qs_offset + blk_base * 128ul)[lane];
                                const uint2 uq_u2 = reinterpret_cast<device const uint2*>(up_exp + a.qs_offset + blk_base * 128ul)[lane];
                                const uint2 gqh_u2 = (planar_encoding == 13u)
                                    ? reinterpret_cast<device const uint2*>(gate_exp + a.qh_offset + blk * 32ul)[ir]
                                    : uint2(0u);
                                const uint2 uqh_u2 = (planar_encoding == 13u)
                                    ? reinterpret_cast<device const uint2*>(up_exp + a.qh_offset + blk * 32ul)[ir]
                                    : uint2(0u);

                                uchar4 gq0_lo, gq0_hi, gq1_lo, gq1_hi;
                                uchar4 uq0_lo, uq0_hi, uq1_lo, uq1_hi;
                                unpack_q4k_u8_pair_v3(gq_u2, gqh_u2, grp0, grp1, gq0_lo, gq0_hi, gq1_lo, gq1_hi);
                                unpack_q4k_u8_pair_v3(uq_u2, uqh_u2, grp0, grp1, uq0_lo, uq0_hi, uq1_lo, uq1_hi);
                                const float g_dot0 = dot8_u8_f32_v3(gq0_lo, gq0_hi, x0_lo, x0_hi);
                                const float g_dot1 = dot8_u8_f32_v3(gq1_lo, gq1_hi, x1_lo, x1_hi);
                                const float u_dot0 = dot8_u8_f32_v3(uq0_lo, uq0_hi, x0_lo, x0_hi);
                                const float u_dot1 = dot8_u8_f32_v3(uq1_lo, uq1_hi, x1_lo, x1_hi);
                                sum_g0[r] += (gd_f * float(gsc01 & 0xffu) * g_dot0 - gm_f * float(gmn01 & 0xffu) * sum_x0) +
                                             (gd_f * float(gsc01 >> 8u) * g_dot1 - gm_f * float(gmn01 >> 8u) * sum_x1);
                                sum_u0[r] += (ud_f * float(usc01 & 0xffu) * u_dot0 - um_f * float(umn01 & 0xffu) * sum_x0) +
                                             (ud_f * float(usc01 >> 8u) * u_dot1 - um_f * float(umn01 >> 8u) * sum_x1);
                            }
                        }
                    }
                }
            } else if (planar_encoding == 8u) {
                const ushort ib4 = lane >> 3u;
                uint k_blk = 0u;
                for (; k_blk + 4u <= blocks_per_row; k_blk += 4u) {
                    const float4 x4 = float4(reinterpret_cast<device const packed_float4*>(x_src + k_blk * 32u)[lane]);
                    bad |= nonfinite_vec4(x4);
                    #pragma unroll
                    for (ushort r = 0; r < kRows; ++r) {
                        if (r < active_rows) {
                            const uint prow = first_row + uint(r);
                            const ulong blk_base = ulong(prow) * blocks_per_row + k_blk;
                            const ulong blk_lane = blk_base + uint(ib4);
                            const half gd_h = reinterpret_cast<device const half*>(gate_exp + a.dm_offset)[blk_lane];
                            const half ud_h = reinterpret_cast<device const half*>(up_exp + a.dm_offset)[blk_lane];
                            float gd_f = float(gd_h), ud_f = float(ud_h);
                            if (nonfinite_half_bit(gd_h)) { bad |= 1u; gd_f = 0.0f; }
                            if (nonfinite_half_bit(ud_h)) { bad |= 1u; ud_f = 0.0f; }
                            const char4 gq4 = reinterpret_cast<device const char4*>(gate_exp + a.qs_offset + blk_base * 32ul)[lane];
                            const char4 uq4 = reinterpret_cast<device const char4*>(up_exp + a.qs_offset + blk_base * 32ul)[lane];
                            sum_g0[r] += dot(float4(gq4) * gd_f, x4);
                            sum_u0[r] += dot(float4(uq4) * ud_f, x4);
                        }
                    }
                }
                for (; k_blk < blocks_per_row; ++k_blk) {
                    const float xv = x_src[k_blk * 32u + lane];
                    bad |= nonfinite_bit(xv);
                    #pragma unroll
                    for (ushort r = 0; r < kRows; ++r) {
                        if (r < active_rows) {
                            const uint prow = first_row + uint(r);
                            const ulong blk = ulong(prow) * blocks_per_row + k_blk;
                            const half gd_h = reinterpret_cast<device const half*>(gate_exp + a.dm_offset)[blk];
                            const half ud_h = reinterpret_cast<device const half*>(up_exp + a.dm_offset)[blk];
                            float gd_f = float(gd_h), ud_f = float(ud_h);
                            if (nonfinite_half_bit(gd_h)) { bad |= 1u; gd_f = 0.0f; }
                            if (nonfinite_half_bit(ud_h)) { bad |= 1u; ud_f = 0.0f; }
                            const char gq = reinterpret_cast<device const char*>(gate_exp + a.qs_offset + blk * 32ul)[lane];
                            const char uq = reinterpret_cast<device const char*>(up_exp + a.qs_offset + blk * 32ul)[lane];
                            sum_g0[r] += (float(gq) * gd_f) * xv;
                            sum_u0[r] += (float(uq) * ud_f) * xv;
                        }
                    }
                }
            }
            device float* out = reinterpret_cast<device float*>(
                reinterpret_cast<device uchar*>(expert_hidden) + ulong(flat0) * a.output_stride_bytes);
            #pragma unroll
            for (ushort r = 0; r < kRows; ++r) {
                const float g_tot = simd_sum(sum_g0[r]);
                const float u_tot = simd_sum(sum_u0[r]);
                if (r < active_rows && !lane) {
                    bad |= nonfinite_bit(g_tot) | nonfinite_bit(u_tot);
                    const float act = (g_tot * sigmoid_f32(g_tot)) * u_tot;
                    bad |= nonfinite_bit(act);
                    out[first_row + r] = act;
                }
            }
        }
        publish_sticky_error(bad, error);
        return;
    }

    if (sg_live) {
        uint tok_ids[MCap];
        uint flat_ids[MCap];
        float sum_g[MCap][kRows] = {{0.0f}};
        float sum_u[MCap][kRows] = {{0.0f}};
        #pragma unroll
        for (ushort m = 0; m < MCap; ++m) {
            tok_ids[m] = (m < n) ? tok_meta[m] : 0u;
            flat_ids[m] = (m < n) ? flat_meta[m] : 0u;
        }
        if (planar_encoding == 12u || planar_encoding == 13u) {
            const ushort ix = lane >> 4u;
            const ushort sub16 = lane & 15u;
            const ushort pair = sub16 >> 2u;
            const ushort ir = sub16 & 3u;
            const uint grp0 = uint(pair) * 2u;
            const uint grp1 = grp0 + 1u;

            for (uint sb0 = 0u; sb0 < blocks_per_row; sb0 += 2u) {
                const uint sblocks = min(2u, blocks_per_row - sb0);
                if (ix < sblocks) {
                    float g_dl0[kRows], g_dl1[kRows], g_ml0[kRows], g_ml1[kRows];
                    float u_dl0[kRows], u_dl1[kRows], u_ml0[kRows], u_ml1[kRows];
                    uchar4 gq0_lo[kRows], gq0_hi[kRows], gq1_lo[kRows], gq1_hi[kRows];
                    uchar4 uq0_lo[kRows], uq0_hi[kRows], uq1_lo[kRows], uq1_hi[kRows];

                    #pragma unroll
                    for (ushort r = 0; r < kRows; ++r) {
                        const uint prow = min(first_row + uint(r), a.rows - 1u);
                        const ulong blk_base = ulong(prow) * blocks_per_row + sb0;
                        const ulong blk = blk_base + uint(ix);

                        const half2 gdm = reinterpret_cast<device const half2*>(gate_exp + a.dm_offset)[blk];
                        const half2 udm = reinterpret_cast<device const half2*>(up_exp + a.dm_offset)[blk];
                        float gd_f = float(gdm.x), gm_f = float(gdm.y);
                        float ud_f = float(udm.x), um_f = float(udm.y);
                        if (nonfinite_half_bit(gdm.x) || nonfinite_half_bit(gdm.y)) {
                            bad |= 1u; gd_f = 0.0f; gm_f = 0.0f;
                        }
                        if (nonfinite_half_bit(udm.x) || nonfinite_half_bit(udm.y)) {
                            bad |= 1u; ud_f = 0.0f; um_f = 0.0f;
                        }

                        const ushort gsc01 = reinterpret_cast<device const ushort*>(gate_exp + a.sm_offset + blk * 16ul)[pair];
                        const ushort gmn01 = reinterpret_cast<device const ushort*>(gate_exp + a.sm_offset + blk * 16ul + 8ul)[pair];
                        const ushort usc01 = reinterpret_cast<device const ushort*>(up_exp + a.sm_offset + blk * 16ul)[pair];
                        const ushort umn01 = reinterpret_cast<device const ushort*>(up_exp + a.sm_offset + blk * 16ul + 8ul)[pair];

                        g_dl0[r] = gd_f * float(gsc01 & 0xffu);
                        g_dl1[r] = gd_f * float(gsc01 >> 8u);
                        g_ml0[r] = gm_f * float(gmn01 & 0xffu);
                        g_ml1[r] = gm_f * float(gmn01 >> 8u);
                        u_dl0[r] = ud_f * float(usc01 & 0xffu);
                        u_dl1[r] = ud_f * float(usc01 >> 8u);
                        u_ml0[r] = um_f * float(umn01 & 0xffu);
                        u_ml1[r] = um_f * float(umn01 >> 8u);

                        const uint2 gq_u2 = reinterpret_cast<device const uint2*>(gate_exp + a.qs_offset + blk_base * 128ul)[lane];
                        const uint2 uq_u2 = reinterpret_cast<device const uint2*>(up_exp + a.qs_offset + blk_base * 128ul)[lane];
                        const uint2 gqh_u2 = (planar_encoding == 13u)
                            ? reinterpret_cast<device const uint2*>(gate_exp + a.qh_offset + blk * 32ul)[ir]
                            : uint2(0u);
                        const uint2 uqh_u2 = (planar_encoding == 13u)
                            ? reinterpret_cast<device const uint2*>(up_exp + a.qh_offset + blk * 32ul)[ir]
                            : uint2(0u);
                        unpack_q4k_u8_pair_v3(gq_u2, gqh_u2, grp0, grp1, gq0_lo[r], gq0_hi[r], gq1_lo[r], gq1_hi[r]);
                        unpack_q4k_u8_pair_v3(uq_u2, uqh_u2, grp0, grp1, uq0_lo[r], uq0_hi[r], uq1_lo[r], uq1_hi[r]);
                    }

                    const uint col_base0 = (sb0 + uint(ix)) * 256u + uint(pair) * 64u + uint(ir) * 8u;
                    #pragma unroll
                    for (ushort m = 0; m < MCap; ++m) {
                        if (m < n) {
                            device const float* x_src = input + ulong(tok_ids[m]) * in_stride_f32 + col_base0;
                            const float4 x0_lo = float4(*reinterpret_cast<device const packed_float4*>(x_src));
                            const float4 x0_hi = float4(*reinterpret_cast<device const packed_float4*>(x_src + 4u));
                            const float4 x1_lo = float4(*reinterpret_cast<device const packed_float4*>(x_src + 32u));
                            const float4 x1_hi = float4(*reinterpret_cast<device const packed_float4*>(x_src + 36u));
                            bad |= nonfinite_vec4(x0_lo) | nonfinite_vec4(x0_hi) |
                                   nonfinite_vec4(x1_lo) | nonfinite_vec4(x1_hi);
                            const float sum_x0 = (x0_lo.x + x0_lo.y + x0_lo.z + x0_lo.w) + (x0_hi.x + x0_hi.y + x0_hi.z + x0_hi.w);
                            const float sum_x1 = (x1_lo.x + x1_lo.y + x1_lo.z + x1_lo.w) + (x1_hi.x + x1_hi.y + x1_hi.z + x1_hi.w);
                            #pragma unroll
                            for (ushort r = 0; r < kRows; ++r) {
                                if (r < active_rows) {
                                    const float g_dot0 = dot8_u8_f32_v3(gq0_lo[r], gq0_hi[r], x0_lo, x0_hi);
                                    const float g_dot1 = dot8_u8_f32_v3(gq1_lo[r], gq1_hi[r], x1_lo, x1_hi);
                                    const float u_dot0 = dot8_u8_f32_v3(uq0_lo[r], uq0_hi[r], x0_lo, x0_hi);
                                    const float u_dot1 = dot8_u8_f32_v3(uq1_lo[r], uq1_hi[r], x1_lo, x1_hi);
                                    sum_g[m][r] += (g_dl0[r] * g_dot0 - g_ml0[r] * sum_x0) + (g_dl1[r] * g_dot1 - g_ml1[r] * sum_x1);
                                    sum_u[m][r] += (u_dl0[r] * u_dot0 - u_ml0[r] * sum_x0) + (u_dl1[r] * u_dot1 - u_ml1[r] * sum_x1);
                                }
                            }
                        }
                    }
                }
            }
        } else if (planar_encoding == 8u) {
            uint k_blk = 0u;
            const ushort ib4 = lane >> 3u;
            for (; k_blk + 4u <= blocks_per_row; k_blk += 4u) {
                const uint b_cur = k_blk + uint(ib4);
                float4 gw4[kRows], uw4[kRows];
                #pragma unroll
                for (ushort r = 0; r < kRows; ++r) {
                    const uint prow = min(first_row + uint(r), a.rows - 1u);
                    const ulong blk_base = ulong(prow) * blocks_per_row + k_blk;
                    const ulong blk_lane = ulong(prow) * blocks_per_row + b_cur;
                    const half gd_h = reinterpret_cast<device const half*>(gate_exp + a.dm_offset)[blk_lane];
                    const half ud_h = reinterpret_cast<device const half*>(up_exp + a.dm_offset)[blk_lane];
                    float gd_f = float(gd_h), ud_f = float(ud_h);
                    if (nonfinite_half_bit(gd_h)) { bad |= 1u; gd_f = 0.0f; }
                    if (nonfinite_half_bit(ud_h)) { bad |= 1u; ud_f = 0.0f; }
                    const char4 gq4 = reinterpret_cast<device const char4*>(gate_exp + a.qs_offset + blk_base * 32ul)[lane];
                    const char4 uq4 = reinterpret_cast<device const char4*>(up_exp + a.qs_offset + blk_base * 32ul)[lane];
                    gw4[r] = float4(gq4) * gd_f;
                    uw4[r] = float4(uq4) * ud_f;
                }
                #pragma unroll
                for (ushort m = 0; m < MCap; ++m) {
                    if (m < n) {
                        device const float* x_src = input + ulong(tok_ids[m]) * in_stride_f32;
                        const float4 x4 = float4(reinterpret_cast<device const packed_float4*>(x_src + k_blk * 32u)[lane]);
                        bad |= nonfinite_vec4(x4);
                        #pragma unroll
                        for (ushort r = 0; r < kRows; ++r) {
                            if (r < active_rows) {
                                sum_g[m][r] += dot(gw4[r], x4);
                                sum_u[m][r] += dot(uw4[r], x4);
                            }
                        }
                    }
                }
            }
            for (; k_blk < blocks_per_row; ++k_blk) {
                float gw[kRows], uw[kRows];
                #pragma unroll
                for (ushort r = 0; r < kRows; ++r) {
                    const uint prow = min(first_row + uint(r), a.rows - 1u);
                    const ulong blk = ulong(prow) * blocks_per_row + k_blk;
                    const half gd_h = reinterpret_cast<device const half*>(gate_exp + a.dm_offset)[blk];
                    const half ud_h = reinterpret_cast<device const half*>(up_exp + a.dm_offset)[blk];
                    float gd_f = float(gd_h), ud_f = float(ud_h);
                    if (nonfinite_half_bit(gd_h)) { bad |= 1u; gd_f = 0.0f; }
                    if (nonfinite_half_bit(ud_h)) { bad |= 1u; ud_f = 0.0f; }
                    const char gq = reinterpret_cast<device const char*>(gate_exp + a.qs_offset + blk * 32ul)[lane];
                    const char uq = reinterpret_cast<device const char*>(up_exp + a.qs_offset + blk * 32ul)[lane];
                    gw[r] = float(gq) * gd_f;
                    uw[r] = float(uq) * ud_f;
                }
                #pragma unroll
                for (ushort m = 0; m < MCap; ++m) {
                    if (m < n) {
                        device const float* x_src = input + ulong(tok_ids[m]) * in_stride_f32;
                        const float xv = x_src[k_blk * 32u + lane];
                        bad |= nonfinite_bit(xv);
                        #pragma unroll
                        for (ushort r = 0; r < kRows; ++r) {
                            if (r < active_rows) {
                                sum_g[m][r] += gw[r] * xv;
                                sum_u[m][r] += uw[r] * xv;
                            }
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (ushort m = 0; m < MCap; ++m) {
            if (m < n) {
                device float* out = reinterpret_cast<device float*>(
                    reinterpret_cast<device uchar*>(expert_hidden) + ulong(flat_ids[m]) * a.output_stride_bytes);
                #pragma unroll
                for (ushort r = 0; r < kRows; ++r) {
                    const float g_tot = simd_sum(sum_g[m][r]);
                    const float u_tot = simd_sum(sum_u[m][r]);
                    if (r < active_rows && !lane) {
                        bad |= nonfinite_bit(g_tot) | nonfinite_bit(u_tot);
                        const float act = (g_tot * sigmoid_f32(g_tot)) * u_tot;
                        bad |= nonfinite_bit(act);
                        out[first_row + r] = act;
                    }
                }
            }
        }
    }
    publish_sticky_error(bad, error);
}

template <ushort MCap, ushort MaxTokens = MCap>
static inline void planar_moe_down_union_impl(
    device const uchar* down_weights,
    device const float* expert_hidden,
    device float* expert_down,
    device const uint* meta,
    constant PlanarSmallMoeArgs& a,
    device atomic_uint* error,
    uint2 group,
    ushort sg,
    ushort lane,
    ushort tid
) {
    if ((planar_encoding != 7u && planar_encoding != 8u) ||
        !a.tokens || a.tokens > MaxTokens || !a.selected || a.selected > 16u || !a.experts || a.experts > 512u ||
        !a.columns || a.columns % 32u != 0u || !a.rows || a.rows % 4u != 0u ||
        a.use_union != 1u || a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint U = meta[6];
    if (group.y >= U) return;
    const uint expert = meta[8u + group.y];
    const uint n = meta[136u + group.y];
    if (expert >= a.experts || n == 0u || n > MCap) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    device const uint* flat_meta = meta + 1288u + group.y * 8u;
    for (uint k = 0u; k < n; ++k) {
        if (flat_meta[k] >= a.tokens * a.selected) {
            if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            return;
        }
    }

    constexpr ushort kRows = planar_union_down_rows;
    const uint first_row = (group.x * 4u + uint(sg)) * uint(kRows);
    const bool sg_live = first_row < a.rows;
    const uint active_rows = sg_live ? min(uint(kRows), a.rows - first_row) : 0u;

    device const uchar* down_exp = down_weights + ulong(expert) * a.planar_expert_bytes;
    const uint blocks_per_row = a.columns / 32u;
    const ulong in_stride_f32 = a.input_stride_bytes / sizeof(float);

    uint bad = 0u;
    if (n == 1u) {
        float sums0[kRows] = {0.0f};
        const uint flat0 = flat_meta[0];
        if (sg_live) {
            device const float* x_src = expert_hidden + ulong(flat0) * in_stride_f32;
            uint k_blk = 0u;
            if (planar_encoding == 7u) {
                const ushort ib8 = lane >> 2u;
                const ushort l4 = lane & 3u;
                const ushort j_base = l4 * 4u;
                for (; k_blk + 8u <= blocks_per_row; k_blk += 8u) {
                    const uint x_col = (k_blk + uint(ib8)) * 32u + uint(j_base);
                    const float4 x_lo = float4(*reinterpret_cast<device const packed_float4*>(x_src + x_col));
                    const float4 x_hi = float4(*reinterpret_cast<device const packed_float4*>(x_src + x_col + 16u));
                    bad |= nonfinite_vec4(x_lo) | nonfinite_vec4(x_hi);
                    #pragma unroll
                    for (ushort r = 0; r < kRows; ++r) {
                        if (r < active_rows) {
                            const uint prow = first_row + uint(r);
                            const ulong blk_base = ulong(prow) * blocks_per_row + k_blk;
                            const ulong blk_lane = blk_base + uint(ib8);
                            const half2 dm = reinterpret_cast<device const half2*>(down_exp + a.dm_offset)[blk_lane];
                            float d = float(dm.x), mv = float(dm.y);
                            if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
                                bad |= 1u; d = 0.0f; mv = 0.0f;
                            }
                            const uint q4b = reinterpret_cast<device const uint*>(down_exp + a.qs_offset + blk_base * 16ul)[lane];
                            const uint qh = reinterpret_cast<device const uint*>(down_exp + a.qh_offset)[blk_lane];
                            const uint qh_lo = qh >> j_base;
                            const uint qh_hi = qh >> (16u + j_base);
                            const float4 q_lo(float((q4b & 15u) | ((qh_lo & 1u) << 4u)),
                                              float(((q4b >> 8u) & 15u) | (((qh_lo >> 1u) & 1u) << 4u)),
                                              float(((q4b >> 16u) & 15u) | (((qh_lo >> 2u) & 1u) << 4u)),
                                              float(((q4b >> 24u) & 15u) | (((qh_lo >> 3u) & 1u) << 4u)));
                            const float4 q_hi(float(((q4b >> 4u) & 15u) | ((qh_hi & 1u) << 4u)),
                                              float(((q4b >> 12u) & 15u) | (((qh_hi >> 1u) & 1u) << 4u)),
                                              float(((q4b >> 20u) & 15u) | (((qh_hi >> 2u) & 1u) << 4u)),
                                              float(((q4b >> 28u) & 15u) | (((qh_hi >> 3u) & 1u) << 4u)));
                            sums0[r] += dot(d * q_lo + mv, x_lo) + dot(d * q_hi + mv, x_hi);
                        }
                    }
                }
                const ushort ib4 = lane >> 3u;
                const ushort l8 = lane & 7u;
                for (; k_blk + 4u <= blocks_per_row; k_blk += 4u) {
                    const uint x_blk = (k_blk + uint(ib4)) * 32u;
                    const float2 x01 = float2(reinterpret_cast<device const packed_float2*>(x_src + x_blk)[l8]);
                    const float2 x23 = float2(reinterpret_cast<device const packed_float2*>(x_src + x_blk + 16u)[l8]);
                    bad |= nonfinite_vec2(x01) | nonfinite_vec2(x23);
                    const float4 x4(x01.x, x01.y, x23.x, x23.y);
                    #pragma unroll
                    for (ushort r = 0; r < kRows; ++r) {
                        if (r < active_rows) {
                            const uint prow = first_row + uint(r);
                            const ulong blk_base = ulong(prow) * blocks_per_row + k_blk;
                            const ulong blk_lane = blk_base + uint(ib4);
                            const half2 dm = reinterpret_cast<device const half2*>(down_exp + a.dm_offset)[blk_lane];
                            float d = float(dm.x), mv = float(dm.y);
                            if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
                                bad |= 1u; d = 0.0f; mv = 0.0f;
                            }
                            const ushort q2 = reinterpret_cast<device const ushort*>(down_exp + a.qs_offset + blk_base * 16ul)[lane];
                            const uint qh = reinterpret_cast<device const uint*>(down_exp + a.qh_offset)[blk_lane];
                            const ushort j0 = l8 * 2u;
                            const ushort j1 = j0 + 1u;
                            const uint q_idx0 = (uint(q2 & 0x000fu)) | (((qh >> j0) & 1u) << 4u);
                            const uint q_idx1 = (uint((q2 >> 8u) & 0x000fu)) | (((qh >> j1) & 1u) << 4u);
                            const uint q_idx2 = (uint((q2 >> 4u) & 0x000fu)) | (((qh >> (16u + j0)) & 1u) << 4u);
                            const uint q_idx3 = (uint((q2 >> 12u) & 0x000fu)) | (((qh >> (16u + j1)) & 1u) << 4u);
                            const float4 w4(d * float(q_idx0) + mv,
                                            d * float(q_idx1) + mv,
                                            d * float(q_idx2) + mv,
                                            d * float(q_idx3) + mv);
                            sums0[r] += dot(w4, x4);
                        }
                    }
                }
                const uint shift_tail = 4u * (uint(lane) / 16u);
                const uint j_tail = uint(lane) % 16u;
                for (; k_blk < blocks_per_row; ++k_blk) {
                    const float xv = x_src[k_blk * 32u + lane];
                    bad |= nonfinite_bit(xv);
                    #pragma unroll
                    for (ushort r = 0; r < kRows; ++r) {
                        if (r < active_rows) {
                            const uint prow = first_row + uint(r);
                            const ulong blk = ulong(prow) * blocks_per_row + k_blk;
                            const half2 dm = reinterpret_cast<device const half2*>(down_exp + a.dm_offset)[blk];
                            float d = float(dm.x), mv = float(dm.y);
                            if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
                                bad |= 1u; d = 0.0f; mv = 0.0f;
                            }
                            const uchar qb = (down_exp + a.qs_offset + blk * 16ul)[j_tail];
                            const uint qh = reinterpret_cast<device const uint*>(down_exp + a.qh_offset)[blk];
                            const uint q = ((uint(qb) >> shift_tail) & 15u) | (((qh >> lane) & 1u) << 4u);
                            sums0[r] += (d * float(q) + mv) * xv;
                        }
                    }
                }
            } else {
                const ushort ib4 = lane >> 3u;
                for (; k_blk + 4u <= blocks_per_row; k_blk += 4u) {
                    const float4 x4 = float4(reinterpret_cast<device const packed_float4*>(x_src + k_blk * 32u)[lane]);
                    bad |= nonfinite_vec4(x4);
                    #pragma unroll
                    for (ushort r = 0; r < kRows; ++r) {
                        if (r < active_rows) {
                            const uint prow = first_row + uint(r);
                            const ulong blk_base = ulong(prow) * blocks_per_row + k_blk;
                            const ulong blk_lane = blk_base + uint(ib4);
                            const half dh = reinterpret_cast<device const half*>(down_exp + a.dm_offset)[blk_lane];
                            float d = float(dh);
                            if (nonfinite_half_bit(dh)) { bad |= 1u; d = 0.0f; }
                            const char4 q4 = reinterpret_cast<device const char4*>(down_exp + a.qs_offset + blk_base * 32ul)[lane];
                            sums0[r] += dot(float4(q4) * d, x4);
                        }
                    }
                }
                for (; k_blk < blocks_per_row; ++k_blk) {
                    const float xv = x_src[k_blk * 32u + lane];
                    bad |= nonfinite_bit(xv);
                    #pragma unroll
                    for (ushort r = 0; r < kRows; ++r) {
                        if (r < active_rows) {
                            const uint prow = first_row + uint(r);
                            const ulong blk = ulong(prow) * blocks_per_row + k_blk;
                            const half dh = reinterpret_cast<device const half*>(down_exp + a.dm_offset)[blk];
                            float d = float(dh);
                            if (nonfinite_half_bit(dh)) { bad |= 1u; d = 0.0f; }
                            const char qc = reinterpret_cast<device const char*>(down_exp + a.qs_offset + blk * 32ul)[lane];
                            sums0[r] += (float(qc) * d) * xv;
                        }
                    }
                }
            }
            device float* out = reinterpret_cast<device float*>(
                reinterpret_cast<device uchar*>(expert_down) + ulong(flat0) * a.output_stride_bytes);
            #pragma unroll
            for (ushort r = 0; r < kRows; ++r) {
                const float tot = simd_sum(sums0[r]);
                if (r < active_rows && !lane) {
                    bad |= nonfinite_bit(tot);
                    out[first_row + r] = tot;
                }
            }
        }
        publish_sticky_error(bad, error);
        return;
    }

    if (sg_live) {
        uint flat_ids[MCap];
        float sums[MCap][kRows] = {{0.0f}};
        #pragma unroll
        for (ushort m = 0; m < MCap; ++m) {
            flat_ids[m] = (m < n) ? flat_meta[m] : 0u;
        }
        uint k_blk = 0u;
        if (planar_encoding == 7u) {
            const ushort ib8 = lane >> 2u;
            const ushort l4 = lane & 3u;
            const ushort j_base = l4 * 4u;
            for (; k_blk + 8u <= blocks_per_row; k_blk += 8u) {
                const uint b_cur = k_blk + uint(ib8);
                float4 w_lo[kRows], w_hi[kRows];
                #pragma unroll
                for (ushort r = 0; r < kRows; ++r) {
                    const uint prow = min(first_row + uint(r), a.rows - 1u);
                    const ulong blk_base = ulong(prow) * blocks_per_row + k_blk;
                    const ulong blk_lane = ulong(prow) * blocks_per_row + b_cur;

                    const half2 dm = reinterpret_cast<device const half2*>(down_exp + a.dm_offset)[blk_lane];
                    float d = float(dm.x), mv = float(dm.y);
                    if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
                        bad |= 1u; d = 0.0f; mv = 0.0f;
                    }
                    const uint q4b = reinterpret_cast<device const uint*>(down_exp + a.qs_offset + blk_base * 16ul)[lane];
                    const uint qh = reinterpret_cast<device const uint*>(down_exp + a.qh_offset)[blk_lane];
                    const uint qh_lo = qh >> j_base;
                    const uint qh_hi = qh >> (16u + j_base);
                    const float4 q_lo(float((q4b & 15u) | ((qh_lo & 1u) << 4u)),
                                      float(((q4b >> 8u) & 15u) | (((qh_lo >> 1u) & 1u) << 4u)),
                                      float(((q4b >> 16u) & 15u) | (((qh_lo >> 2u) & 1u) << 4u)),
                                      float(((q4b >> 24u) & 15u) | (((qh_lo >> 3u) & 1u) << 4u)));
                    const float4 q_hi(float(((q4b >> 4u) & 15u) | ((qh_hi & 1u) << 4u)),
                                      float(((q4b >> 12u) & 15u) | (((qh_hi >> 1u) & 1u) << 4u)),
                                      float(((q4b >> 20u) & 15u) | (((qh_hi >> 2u) & 1u) << 4u)),
                                      float(((q4b >> 28u) & 15u) | (((qh_hi >> 3u) & 1u) << 4u)));
                    w_lo[r] = d * q_lo + mv;
                    w_hi[r] = d * q_hi + mv;
                }
                const uint x_col = (k_blk + uint(ib8)) * 32u + uint(j_base);
                #pragma unroll
                for (ushort m = 0; m < MCap; ++m) {
                    if (m < n) {
                        device const float* x_src = expert_hidden + ulong(flat_ids[m]) * in_stride_f32;
                        const float4 x_lo = float4(*reinterpret_cast<device const packed_float4*>(x_src + x_col));
                        const float4 x_hi = float4(*reinterpret_cast<device const packed_float4*>(x_src + x_col + 16u));
                        bad |= nonfinite_vec4(x_lo) | nonfinite_vec4(x_hi);
                        #pragma unroll
                        for (ushort r = 0; r < kRows; ++r) {
                            if (r < active_rows) sums[m][r] += dot(w_lo[r], x_lo) + dot(w_hi[r], x_hi);
                        }
                    }
                }
            }
            const ushort ib4 = lane >> 3u;
            const ushort l8 = lane & 7u;
            for (; k_blk + 4u <= blocks_per_row; k_blk += 4u) {
                const uint b_cur = k_blk + uint(ib4);
                float4 w4[kRows];
                #pragma unroll
                for (ushort r = 0; r < kRows; ++r) {
                    const uint prow = min(first_row + uint(r), a.rows - 1u);
                    const ulong blk_base = ulong(prow) * blocks_per_row + k_blk;
                    const ulong blk_lane = ulong(prow) * blocks_per_row + b_cur;

                    const half2 dm = reinterpret_cast<device const half2*>(down_exp + a.dm_offset)[blk_lane];
                    float d = float(dm.x), mv = float(dm.y);
                    if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
                        bad |= 1u; d = 0.0f; mv = 0.0f;
                    }
                    const ushort q2 = reinterpret_cast<device const ushort*>(down_exp + a.qs_offset + blk_base * 16ul)[lane];
                    const uint qh = reinterpret_cast<device const uint*>(down_exp + a.qh_offset)[blk_lane];
                    const ushort j0 = l8 * 2u;
                    const ushort j1 = j0 + 1u;
                    const uint q_idx0 = (uint(q2 & 0x000fu)) | (((qh >> j0) & 1u) << 4u);
                    const uint q_idx1 = (uint((q2 >> 8u) & 0x000fu)) | (((qh >> j1) & 1u) << 4u);
                    const uint q_idx2 = (uint((q2 >> 4u) & 0x000fu)) | (((qh >> (16u + j0)) & 1u) << 4u);
                    const uint q_idx3 = (uint((q2 >> 12u) & 0x000fu)) | (((qh >> (16u + j1)) & 1u) << 4u);
                    w4[r] = float4(d * float(q_idx0) + mv,
                                   d * float(q_idx1) + mv,
                                   d * float(q_idx2) + mv,
                                   d * float(q_idx3) + mv);
                }
                const uint x_blk = (k_blk + uint(ib4)) * 32u;
                #pragma unroll
                for (ushort m = 0; m < MCap; ++m) {
                    if (m < n) {
                        device const float* x_src = expert_hidden + ulong(flat_ids[m]) * in_stride_f32;
                        const float2 x01 = float2(reinterpret_cast<device const packed_float2*>(x_src + x_blk)[l8]);
                        const float2 x23 = float2(reinterpret_cast<device const packed_float2*>(x_src + x_blk + 16u)[l8]);
                        bad |= nonfinite_vec2(x01) | nonfinite_vec2(x23);
                        const float4 x4(x01.x, x01.y, x23.x, x23.y);
                        #pragma unroll
                        for (ushort r = 0; r < kRows; ++r) {
                            if (r < active_rows) sums[m][r] += dot(w4[r], x4);
                        }
                    }
                }
            }
            const uint shift_tail = 4u * (uint(lane) / 16u);
            const uint j_tail = uint(lane) % 16u;
            for (; k_blk < blocks_per_row; ++k_blk) {
                float w1[kRows];
                #pragma unroll
                for (ushort r = 0; r < kRows; ++r) {
                    const uint prow = min(first_row + uint(r), a.rows - 1u);
                    const ulong blk = ulong(prow) * blocks_per_row + k_blk;
                    const half2 dm = reinterpret_cast<device const half2*>(down_exp + a.dm_offset)[blk];
                    float d = float(dm.x), mv = float(dm.y);
                    if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
                        bad |= 1u; d = 0.0f; mv = 0.0f;
                    }
                    const uchar qb = (down_exp + a.qs_offset + blk * 16ul)[j_tail];
                    const uint qh = reinterpret_cast<device const uint*>(down_exp + a.qh_offset)[blk];
                    const uint q = ((uint(qb) >> shift_tail) & 15u) | (((qh >> lane) & 1u) << 4u);
                    w1[r] = d * float(q) + mv;
                }
                #pragma unroll
                for (ushort m = 0; m < MCap; ++m) {
                    if (m < n) {
                        device const float* x_src = expert_hidden + ulong(flat_ids[m]) * in_stride_f32;
                        const float xv = x_src[k_blk * 32u + lane];
                        bad |= nonfinite_bit(xv);
                        #pragma unroll
                        for (ushort r = 0; r < kRows; ++r) {
                            if (r < active_rows) sums[m][r] += w1[r] * xv;
                        }
                    }
                }
            }
        } else {
            const ushort ib4 = lane >> 3u;
            for (; k_blk + 4u <= blocks_per_row; k_blk += 4u) {
                const uint b_cur = k_blk + uint(ib4);
                float4 w4[kRows];
                #pragma unroll
                for (ushort r = 0; r < kRows; ++r) {
                    const uint prow = min(first_row + uint(r), a.rows - 1u);
                    const ulong blk_base = ulong(prow) * blocks_per_row + k_blk;
                    const ulong blk_lane = ulong(prow) * blocks_per_row + b_cur;
                    const half dh = reinterpret_cast<device const half*>(down_exp + a.dm_offset)[blk_lane];
                    float d = float(dh);
                    if (nonfinite_half_bit(dh)) { bad |= 1u; d = 0.0f; }
                    const char4 q4 = reinterpret_cast<device const char4*>(down_exp + a.qs_offset + blk_base * 32ul)[lane];
                    w4[r] = float4(q4) * d;
                }
                #pragma unroll
                for (ushort m = 0; m < MCap; ++m) {
                    if (m < n) {
                        device const float* x_src = expert_hidden + ulong(flat_ids[m]) * in_stride_f32;
                        const float4 x4 = float4(reinterpret_cast<device const packed_float4*>(x_src + k_blk * 32u)[lane]);
                        bad |= nonfinite_vec4(x4);
                        #pragma unroll
                        for (ushort r = 0; r < kRows; ++r) {
                            if (r < active_rows) sums[m][r] += dot(w4[r], x4);
                        }
                    }
                }
            }
            for (; k_blk < blocks_per_row; ++k_blk) {
                float w1[kRows];
                #pragma unroll
                for (ushort r = 0; r < kRows; ++r) {
                    const uint prow = min(first_row + uint(r), a.rows - 1u);
                    const ulong blk = ulong(prow) * blocks_per_row + k_blk;
                    const half dh = reinterpret_cast<device const half*>(down_exp + a.dm_offset)[blk];
                    float d = float(dh);
                    if (nonfinite_half_bit(dh)) { bad |= 1u; d = 0.0f; }
                    const char qc = reinterpret_cast<device const char*>(down_exp + a.qs_offset + blk * 32ul)[lane];
                    w1[r] = float(qc) * d;
                }
                #pragma unroll
                for (ushort m = 0; m < MCap; ++m) {
                    if (m < n) {
                        device const float* x_src = expert_hidden + ulong(flat_ids[m]) * in_stride_f32;
                        const float xv = x_src[k_blk * 32u + lane];
                        bad |= nonfinite_bit(xv);
                        #pragma unroll
                        for (ushort r = 0; r < kRows; ++r) {
                            if (r < active_rows) sums[m][r] += w1[r] * xv;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (ushort m = 0; m < MCap; ++m) {
            if (m < n) {
                device float* out = reinterpret_cast<device float*>(
                    reinterpret_cast<device uchar*>(expert_down) + ulong(flat_ids[m]) * a.output_stride_bytes);
                #pragma unroll
                for (ushort r = 0; r < kRows; ++r) {
                    const float tot = simd_sum(sums[m][r]);
                    if (r < active_rows && !lane) {
                        bad |= nonfinite_bit(tot);
                        out[first_row + r] = tot;
                    }
                }
            }
        }
    }
    publish_sticky_error(bad, error);
}

} // namespace planar_v3_detail

kernel void planar_moe_gate_up_swiglu_m4(device const uchar* pair_lo [[buffer(0)]],
                                         device const uchar* pair_hi [[buffer(1)]],
                                         device const float* input [[buffer(2)]],
                                         device float* expert_hidden [[buffer(3)]],
                                         device const uint* meta [[buffer(4)]],
                                         constant planar_v3_detail::PlanarSmallMoeArgs& a [[buffer(5)]],
                                         device atomic_uint* error [[buffer(7)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         ushort sg [[simdgroup_index_in_threadgroup]],
                                         ushort lane [[thread_index_in_simdgroup]],
                                         ushort tid [[thread_index_in_threadgroup]]) {
    const uint n_est = (a.tokens >= 1u && a.tokens <= 4u && a.use_union == 1u && group.y < meta[6]) ? meta[136u + group.y] : 4u;
    if (n_est == 2u) {
        planar_v3_detail::planar_moe_gate_up_swiglu_union_impl<2, 4>(
            pair_lo, pair_hi, input, expert_hidden, meta, a, error, group, sg, lane, tid);
    } else {
        planar_v3_detail::planar_moe_gate_up_swiglu_union_impl<4, 4>(
            pair_lo, pair_hi, input, expert_hidden, meta, a, error, group, sg, lane, tid);
    }
}

kernel void planar_moe_gate_up_swiglu_m8(device const uchar* pair_lo [[buffer(0)]],
                                         device const uchar* pair_hi [[buffer(1)]],
                                         device const float* input [[buffer(2)]],
                                         device float* expert_hidden [[buffer(3)]],
                                         device const uint* meta [[buffer(4)]],
                                         constant planar_v3_detail::PlanarSmallMoeArgs& a [[buffer(5)]],
                                         device atomic_uint* error [[buffer(7)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         ushort sg [[simdgroup_index_in_threadgroup]],
                                         ushort lane [[thread_index_in_simdgroup]],
                                         ushort tid [[thread_index_in_threadgroup]]) {
    const uint n_est = (a.tokens >= 1u && a.tokens <= 8u && a.use_union == 1u && group.y < meta[6]) ? meta[136u + group.y] : 8u;
    if (n_est == 2u) {
        planar_v3_detail::planar_moe_gate_up_swiglu_union_impl<2, 8>(
            pair_lo, pair_hi, input, expert_hidden, meta, a, error, group, sg, lane, tid);
    } else if (n_est >= 3u && n_est <= 4u) {
        planar_v3_detail::planar_moe_gate_up_swiglu_union_impl<4, 8>(
            pair_lo, pair_hi, input, expert_hidden, meta, a, error, group, sg, lane, tid);
    } else {
        planar_v3_detail::planar_moe_gate_up_swiglu_union_impl<8, 8>(
            pair_lo, pair_hi, input, expert_hidden, meta, a, error, group, sg, lane, tid);
    }
}

// Dedicated m = 1 direct planar Q5_1 / Q8_0 down projection (zero threadgroup SRAM, zero barriers,
// 1 output row per simdgroup, 4 output rows per 128-thread group, 128-byte coalesced 8-block Q5_1 loads).
kernel void planar_moe_down_m1(device const uchar* down_weights [[buffer(0)]],
                               device const float* expert_hidden [[buffer(1)]],
                               device float* expert_down [[buffer(2)]],
                               device const uint* indices [[buffer(3)]],
                               constant planar_v3_detail::PlanarSmallMoeArgs& a [[buffer(4)]],
                               device atomic_uint* error [[buffer(7)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]],
                               ushort tid [[thread_index_in_threadgroup]]) {
    if ((planar_encoding != 7u && planar_encoding != 8u) ||
        a.tokens != 1u || a.use_union != 0u || !a.selected || a.selected > 16u || !a.experts || a.experts > 512u ||
        !a.columns || a.columns % 32u != 0u || !a.rows || a.rows % 4u != 0u ||
        a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    constexpr ushort kRows = 4;
    const uint first_row = (group.x * 4u + uint(sg)) * uint(kRows);
    if (group.y >= a.selected || first_row >= a.rows) return;
    const uint active_rows = min(uint(kRows), a.rows - first_row);

    const uint expert = indices[group.y];
    if (expert >= a.experts) {
        if (!lane) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            device float* out = reinterpret_cast<device float*>(
                reinterpret_cast<device uchar*>(expert_down) + ulong(group.y) * a.output_stride_bytes);
            for (uint r = 0u; r < active_rows; ++r) out[first_row + r] = 0.0f;
        }
        return;
    }

    device const uchar* down_exp = down_weights + ulong(expert) * a.planar_expert_bytes;
    device const float* x = expert_hidden + ulong(group.y) * (a.input_stride_bytes / sizeof(float));
    const uint blocks_per_row = a.columns / 32u;

    float sums[kRows] = {0.0f};
    uint bad = 0u;
    uint k_blk = 0u;

    if (planar_encoding == 7u) {
        const ushort ib8 = lane >> 2u;
        const ushort l4 = lane & 3u;
        const ushort j_base = l4 * 4u;
        for (; k_blk + 8u <= blocks_per_row; k_blk += 8u) {
            const uint x_col = (k_blk + uint(ib8)) * 32u + uint(j_base);
            const float4 x_lo = float4(*reinterpret_cast<device const packed_float4*>(x + x_col));
            const float4 x_hi = float4(*reinterpret_cast<device const packed_float4*>(x + x_col + 16u));
            bad |= planar_v3_detail::nonfinite_vec4(x_lo) | planar_v3_detail::nonfinite_vec4(x_hi);
            #pragma unroll
            for (ushort r = 0; r < kRows; ++r) {
                if (r < active_rows) {
                    const ulong blk_base = ulong(first_row + uint(r)) * blocks_per_row + k_blk;
                    const ulong blk_lane = blk_base + uint(ib8);

                    const half2 dm = reinterpret_cast<device const half2*>(down_exp + a.dm_offset)[blk_lane];
                    float d = float(dm.x), mv = float(dm.y);
                    if (planar_v3_detail::nonfinite_half_bit(dm.x) || planar_v3_detail::nonfinite_half_bit(dm.y)) {
                        bad |= 1u; d = 0.0f; mv = 0.0f;
                    }
                    const uint q4b = reinterpret_cast<device const uint*>(down_exp + a.qs_offset + blk_base * 16ul)[lane];
                    const uint qh = reinterpret_cast<device const uint*>(down_exp + a.qh_offset)[blk_lane];
                    const uint qh_lo = qh >> j_base;
                    const uint qh_hi = qh >> (16u + j_base);
                    const float4 q_lo(float((q4b & 15u) | ((qh_lo & 1u) << 4u)),
                                      float(((q4b >> 8u) & 15u) | (((qh_lo >> 1u) & 1u) << 4u)),
                                      float(((q4b >> 16u) & 15u) | (((qh_lo >> 2u) & 1u) << 4u)),
                                      float(((q4b >> 24u) & 15u) | (((qh_lo >> 3u) & 1u) << 4u)));
                    const float4 q_hi(float(((q4b >> 4u) & 15u) | ((qh_hi & 1u) << 4u)),
                                      float(((q4b >> 12u) & 15u) | (((qh_hi >> 1u) & 1u) << 4u)),
                                      float(((q4b >> 20u) & 15u) | (((qh_hi >> 2u) & 1u) << 4u)),
                                      float(((q4b >> 28u) & 15u) | (((qh_hi >> 3u) & 1u) << 4u)));
                    sums[r] += dot(d * q_lo + mv, x_lo) + dot(d * q_hi + mv, x_hi);
                }
            }
        }
        const ushort ib4 = lane >> 3u;
        const ushort l8 = lane & 7u;
        for (; k_blk + 4u <= blocks_per_row; k_blk += 4u) {
            const uint x_blk = (k_blk + uint(ib4)) * 32u;
            const float2 x01 = float2(reinterpret_cast<device const packed_float2*>(x + x_blk)[l8]);
            const float2 x23 = float2(reinterpret_cast<device const packed_float2*>(x + x_blk + 16u)[l8]);
            bad |= planar_v3_detail::nonfinite_vec2(x01) | planar_v3_detail::nonfinite_vec2(x23);
            const float4 x4(x01.x, x01.y, x23.x, x23.y);
            #pragma unroll
            for (ushort r = 0; r < kRows; ++r) {
                if (r < active_rows) {
                    const ulong blk_base = ulong(first_row + uint(r)) * blocks_per_row + k_blk;
                    const ulong blk_lane = blk_base + uint(ib4);

                    const half2 dm = reinterpret_cast<device const half2*>(down_exp + a.dm_offset)[blk_lane];
                    float d = float(dm.x), mv = float(dm.y);
                    if (planar_v3_detail::nonfinite_half_bit(dm.x) || planar_v3_detail::nonfinite_half_bit(dm.y)) {
                        bad |= 1u; d = 0.0f; mv = 0.0f;
                    }
                    const ushort q2 = reinterpret_cast<device const ushort*>(down_exp + a.qs_offset + blk_base * 16ul)[lane];
                    const uint qh = reinterpret_cast<device const uint*>(down_exp + a.qh_offset)[blk_lane];
                    const ushort j0 = l8 * 2u;
                    const ushort j1 = j0 + 1u;
                    const uint q_idx0 = (uint(q2 & 0x000fu)) | (((qh >> j0) & 1u) << 4u);
                    const uint q_idx1 = (uint((q2 >> 8u) & 0x000fu)) | (((qh >> j1) & 1u) << 4u);
                    const uint q_idx2 = (uint((q2 >> 4u) & 0x000fu)) | (((qh >> (16u + j0)) & 1u) << 4u);
                    const uint q_idx3 = (uint((q2 >> 12u) & 0x000fu)) | (((qh >> (16u + j1)) & 1u) << 4u);
                    const float4 w4(d * float(q_idx0) + mv,
                                    d * float(q_idx1) + mv,
                                    d * float(q_idx2) + mv,
                                    d * float(q_idx3) + mv);
                    sums[r] += dot(w4, x4);
                }
            }
        }
        const uint shift_tail = 4u * (uint(lane) / 16u);
        const uint j_tail = uint(lane) % 16u;
        for (; k_blk < blocks_per_row; ++k_blk) {
            const float xv = x[k_blk * 32u + lane];
            bad |= planar_v3_detail::nonfinite_bit(xv);
            #pragma unroll
            for (ushort r = 0; r < kRows; ++r) {
                if (r < active_rows) {
                    const ulong blk = ulong(first_row + uint(r)) * blocks_per_row + k_blk;
                    const half2 dm = reinterpret_cast<device const half2*>(down_exp + a.dm_offset)[blk];
                    float d = float(dm.x), mv = float(dm.y);
                    if (planar_v3_detail::nonfinite_half_bit(dm.x) || planar_v3_detail::nonfinite_half_bit(dm.y)) {
                        bad |= 1u; d = 0.0f; mv = 0.0f;
                    }
                    const uchar qb = (down_exp + a.qs_offset + blk * 16ul)[j_tail];
                    const uint qh = reinterpret_cast<device const uint*>(down_exp + a.qh_offset)[blk];
                    const uint q = ((uint(qb) >> shift_tail) & 15u) | (((qh >> lane) & 1u) << 4u);
                    sums[r] += (d * float(q) + mv) * xv;
                }
            }
        }
    } else {
        const ushort ib4 = lane >> 3u;
        for (; k_blk + 4u <= blocks_per_row; k_blk += 4u) {
            const float4 x4 = float4(reinterpret_cast<device const packed_float4*>(x + k_blk * 32u)[lane]);
            bad |= planar_v3_detail::nonfinite_vec4(x4);
            #pragma unroll
            for (ushort r = 0; r < kRows; ++r) {
                if (r < active_rows) {
                    const ulong blk_base = ulong(first_row + uint(r)) * blocks_per_row + k_blk;
                    const ulong blk_lane = blk_base + uint(ib4);
                    const half dh = reinterpret_cast<device const half*>(down_exp + a.dm_offset)[blk_lane];
                    float d = float(dh);
                    if (planar_v3_detail::nonfinite_half_bit(dh)) { bad |= 1u; d = 0.0f; }
                    const char4 q4 = reinterpret_cast<device const char4*>(down_exp + a.qs_offset + blk_base * 32ul)[lane];
                    sums[r] += dot(float4(q4) * d, x4);
                }
            }
        }
        for (; k_blk < blocks_per_row; ++k_blk) {
            const float xv = x[k_blk * 32u + lane];
            bad |= planar_v3_detail::nonfinite_bit(xv);
            #pragma unroll
            for (ushort r = 0; r < kRows; ++r) {
                if (r < active_rows) {
                    const ulong blk = ulong(first_row + uint(r)) * blocks_per_row + k_blk;
                    const half dh = reinterpret_cast<device const half*>(down_exp + a.dm_offset)[blk];
                    float d = float(dh);
                    if (planar_v3_detail::nonfinite_half_bit(dh)) { bad |= 1u; d = 0.0f; }
                    const char qc = reinterpret_cast<device const char*>(down_exp + a.qs_offset + blk * 32ul)[lane];
                    sums[r] += (float(qc) * d) * xv;
                }
            }
        }
    }

    device float* out = reinterpret_cast<device float*>(
        reinterpret_cast<device uchar*>(expert_down) + ulong(group.y) * a.output_stride_bytes);
    #pragma unroll
    for (ushort r = 0; r < kRows; ++r) {
        const float tot = simd_sum(sums[r]);
        if (r < active_rows && !lane) {
            bad |= planar_v3_detail::nonfinite_bit(tot);
            out[first_row + r] = tot;
        }
    }
    planar_v3_detail::publish_sticky_error(bad, error);
}

kernel void planar_moe_down_m4(device const uchar* down_weights [[buffer(0)]],
                               device const float* expert_hidden [[buffer(1)]],
                               device float* expert_down [[buffer(2)]],
                               device const uint* meta [[buffer(3)]],
                               constant planar_v3_detail::PlanarSmallMoeArgs& a [[buffer(4)]],
                               device atomic_uint* error [[buffer(7)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]],
                               ushort tid [[thread_index_in_threadgroup]]) {
    const uint n_est = (a.tokens >= 1u && a.tokens <= 4u && a.use_union == 1u && group.y < meta[6]) ? meta[136u + group.y] : 4u;
    if (n_est == 2u) {
        planar_v3_detail::planar_moe_down_union_impl<2, 4>(
            down_weights, expert_hidden, expert_down, meta, a, error, group, sg, lane, tid);
    } else {
        planar_v3_detail::planar_moe_down_union_impl<4, 4>(
            down_weights, expert_hidden, expert_down, meta, a, error, group, sg, lane, tid);
    }
}

kernel void planar_moe_down_m8(device const uchar* down_weights [[buffer(0)]],
                               device const float* expert_hidden [[buffer(1)]],
                               device float* expert_down [[buffer(2)]],
                               device const uint* meta [[buffer(3)]],
                               constant planar_v3_detail::PlanarSmallMoeArgs& a [[buffer(4)]],
                               device atomic_uint* error [[buffer(7)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]],
                               ushort tid [[thread_index_in_threadgroup]]) {
    const uint n_est = (a.tokens >= 1u && a.tokens <= 8u && a.use_union == 1u && group.y < meta[6]) ? meta[136u + group.y] : 8u;
    if (n_est == 2u) {
        planar_v3_detail::planar_moe_down_union_impl<2, 8>(
            down_weights, expert_hidden, expert_down, meta, a, error, group, sg, lane, tid);
    } else if (n_est >= 3u && n_est <= 4u) {
        planar_v3_detail::planar_moe_down_union_impl<4, 8>(
            down_weights, expert_hidden, expert_down, meta, a, error, group, sg, lane, tid);
    } else {
        planar_v3_detail::planar_moe_down_union_impl<8, 8>(
            down_weights, expert_hidden, expert_down, meta, a, error, group, sg, lane, tid);
    }
}

namespace planar_v3_detail {

union PlanarNaxGateUpSmemV3 {
    alignas(16) half b_stage[2][2][32 * 32];
#if TOKSPUR_PRIVATE_PLANAR_GU_STAGE64
    alignas(16) half k64_stage[2][32 * 64];
#endif
    struct {
        float gate_epi[32][33];
        uint rids[32];
    } epi;
};
static_assert(sizeof(PlanarNaxGateUpSmemV3) == 8192, "PlanarNaxGateUpSmemV3 must remain 8 KiB per 2-SG TG");

union PlanarNaxSgSmemV3 {
    alignas(16) half b_stage[2][32 * 32];
    uint rids[32];
};
static_assert(sizeof(PlanarNaxSgSmemV3) == 4096, "PlanarNaxSgSmemV3 must remain 4 KiB per simdgroup (8 KiB per 2-SG TG)");

__attribute__((noinline))
static void run_col_grouped_moe_gate_up_swiglu_v3(
    device const uchar* gate_exp,
    device const uchar* up_exp,
    device const bfloat* task_input,
    device uchar* output_buf,
    constant PlanarNaxMoeArgs& a,
    uint group_y,
    uint first_col,
    uint n,
    uint rid0,
    uint rid1,
    ushort tid,
    thread uint& bad
) {
    const ushort quad_id = tid >> 2u;
    const ushort qlane = tid & 3u;
    const uint blocks_per_row = a.columns / 256u;
    const uint grp0 = uint(qlane) * 2u;
    const uint grp1 = grp0 + 1u;

    for (ushort pass = 0; pass < 2; ++pass) {
        const uint col_out = first_col + uint(pass) * 16u + uint(quad_id);
        const bool col_valid = col_out < a.rows;
        float g_sum0 = 0.0f, u_sum0 = 0.0f;
        float g_sum1 = 0.0f, u_sum1 = 0.0f;
        if (col_valid) {
            for (uint sb = 0u; sb < blocks_per_row; ++sb) {
                const ulong blk = ulong(col_out) * blocks_per_row + sb;
                const half2 gdm = reinterpret_cast<device const half2*>(gate_exp + a.dm_offset)[blk];
                const half2 udm = reinterpret_cast<device const half2*>(up_exp + a.dm_offset)[blk];
                float gd_f = float(gdm.x), gm_f = float(gdm.y);
                float ud_f = float(udm.x), um_f = float(udm.y);
                if (nonfinite_half_bit(gdm.x) || nonfinite_half_bit(gdm.y)) { bad |= 1u; gd_f = 0.0f; gm_f = 0.0f; }
                if (nonfinite_half_bit(udm.x) || nonfinite_half_bit(udm.y)) { bad |= 1u; ud_f = 0.0f; um_f = 0.0f; }

                const ushort gsc01 = reinterpret_cast<device const ushort*>(gate_exp + a.sm_offset + blk * 16ul)[qlane];
                const ushort gmn01 = reinterpret_cast<device const ushort*>(gate_exp + a.sm_offset + blk * 16ul + 8ul)[qlane];
                const ushort usc01 = reinterpret_cast<device const ushort*>(up_exp + a.sm_offset + blk * 16ul)[qlane];
                const ushort umn01 = reinterpret_cast<device const ushort*>(up_exp + a.sm_offset + blk * 16ul + 8ul)[qlane];
                const float g_dl0 = gd_f * float(gsc01 & 0xffu), g_dl1 = gd_f * float(gsc01 >> 8u);
                const float g_ml0 = gm_f * float(gmn01 & 0xffu), g_ml1 = gm_f * float(gmn01 >> 8u);
                const float u_dl0 = ud_f * float(usc01 & 0xffu), u_dl1 = ud_f * float(usc01 >> 8u);
                const float u_ml0 = um_f * float(umn01 & 0xffu), u_ml1 = um_f * float(umn01 >> 8u);

                device const uint2* gq_ptr = reinterpret_cast<device const uint2*>(
                    gate_exp + a.qs_offset + blk * 128ul + ulong(qlane) * 32ul);
                device const uint2* uq_ptr = reinterpret_cast<device const uint2*>(
                    up_exp + a.qs_offset + blk * 128ul + ulong(qlane) * 32ul);
                device const uint2* gqh_ptr = reinterpret_cast<device const uint2*>(gate_exp + a.qh_offset + blk * 32ul);
                device const uint2* uqh_ptr = reinterpret_cast<device const uint2*>(up_exp + a.qh_offset + blk * 32ul);

                const uint x_off = sb * 256u + uint(qlane) * 64u;
                device const bfloat* x_r0 = task_input + x_off;
                device const bfloat* x_r1 = task_input + a.columns + x_off;

                float gd0_0 = 0.0f, gd1_0 = 0.0f, ud0_0 = 0.0f, ud1_0 = 0.0f, sx0_0 = 0.0f, sx1_0 = 0.0f;
                float gd0_1 = 0.0f, gd1_1 = 0.0f, ud0_1 = 0.0f, ud1_1 = 0.0f, sx0_1 = 0.0f, sx1_1 = 0.0f;

                #pragma unroll
                for (ushort ir = 0; ir < 4; ++ir) {
                    uchar4 gq0_lo, gq0_hi, gq1_lo, gq1_hi;
                    uchar4 uq0_lo, uq0_hi, uq1_lo, uq1_hi;
                    const uint2 gqh_u2 = (planar_encoding == 13u) ? gqh_ptr[ir] : uint2(0u);
                    const uint2 uqh_u2 = (planar_encoding == 13u) ? uqh_ptr[ir] : uint2(0u);
                    unpack_q4k_u8_pair_v3(gq_ptr[ir], gqh_u2, grp0, grp1, gq0_lo, gq0_hi, gq1_lo, gq1_hi);
                    unpack_q4k_u8_pair_v3(uq_ptr[ir], uqh_u2, grp0, grp1, uq0_lo, uq0_hi, uq1_lo, uq1_hi);

                    const uint sub_off = uint(ir) * 8u;
                    const float4 x00_lo = float4(*reinterpret_cast<device const bfloat4*>(x_r0 + sub_off));
                    const float4 x00_hi = float4(*reinterpret_cast<device const bfloat4*>(x_r0 + sub_off + 4u));
                    const float4 x01_lo = float4(*reinterpret_cast<device const bfloat4*>(x_r0 + 32u + sub_off));
                    const float4 x01_hi = float4(*reinterpret_cast<device const bfloat4*>(x_r0 + 36u + sub_off));
                    sx0_0 += (x00_lo.x + x00_lo.y + x00_lo.z + x00_lo.w) + (x00_hi.x + x00_hi.y + x00_hi.z + x00_hi.w);
                    sx1_0 += (x01_lo.x + x01_lo.y + x01_lo.z + x01_lo.w) + (x01_hi.x + x01_hi.y + x01_hi.z + x01_hi.w);
                    gd0_0 += dot8_u8_f32_v3(gq0_lo, gq0_hi, x00_lo, x00_hi);
                    gd1_0 += dot8_u8_f32_v3(gq1_lo, gq1_hi, x01_lo, x01_hi);
                    ud0_0 += dot8_u8_f32_v3(uq0_lo, uq0_hi, x00_lo, x00_hi);
                    ud1_0 += dot8_u8_f32_v3(uq1_lo, uq1_hi, x01_lo, x01_hi);

                    if (n == 2u) {
                        const float4 x10_lo = float4(*reinterpret_cast<device const bfloat4*>(x_r1 + sub_off));
                        const float4 x10_hi = float4(*reinterpret_cast<device const bfloat4*>(x_r1 + sub_off + 4u));
                        const float4 x11_lo = float4(*reinterpret_cast<device const bfloat4*>(x_r1 + 32u + sub_off));
                        const float4 x11_hi = float4(*reinterpret_cast<device const bfloat4*>(x_r1 + 36u + sub_off));
                        sx0_1 += (x10_lo.x + x10_lo.y + x10_lo.z + x10_lo.w) + (x10_hi.x + x10_hi.y + x10_hi.z + x10_hi.w);
                        sx1_1 += (x11_lo.x + x11_lo.y + x11_lo.z + x11_lo.w) + (x11_hi.x + x11_hi.y + x11_hi.z + x11_hi.w);
                        gd0_1 += dot8_u8_f32_v3(gq0_lo, gq0_hi, x10_lo, x10_hi);
                        gd1_1 += dot8_u8_f32_v3(gq1_lo, gq1_hi, x11_lo, x11_hi);
                        ud0_1 += dot8_u8_f32_v3(uq0_lo, uq0_hi, x10_lo, x10_hi);
                        ud1_1 += dot8_u8_f32_v3(uq1_lo, uq1_hi, x11_lo, x11_hi);
                    }
                }

                g_sum0 += (g_dl0 * gd0_0 - g_ml0 * sx0_0) + (g_dl1 * gd1_0 - g_ml1 * sx1_0);
                u_sum0 += (u_dl0 * ud0_0 - u_ml0 * sx0_0) + (u_dl1 * ud1_0 - u_ml1 * sx1_0);
                if (n == 2u) {
                    g_sum1 += (g_dl0 * gd0_1 - g_ml0 * sx0_1) + (g_dl1 * gd1_1 - g_ml1 * sx1_1);
                    u_sum1 += (u_dl0 * ud0_1 - u_ml0 * sx0_1) + (u_dl1 * ud1_1 - u_ml1 * sx1_1);
                }
            }
        }

        g_sum0 += simd_shuffle_xor(g_sum0, 1u); g_sum0 += simd_shuffle_xor(g_sum0, 2u);
        u_sum0 += simd_shuffle_xor(u_sum0, 1u); u_sum0 += simd_shuffle_xor(u_sum0, 2u);
        if (n == 2u) {
            g_sum1 += simd_shuffle_xor(g_sum1, 1u); g_sum1 += simd_shuffle_xor(g_sum1, 2u);
            u_sum1 += simd_shuffle_xor(u_sum1, 1u); u_sum1 += simd_shuffle_xor(u_sum1, 2u);
        }

        if (qlane == 0u && col_valid) {
            if (a.write_bf16_gathered) {
                device bfloat* gathered_h = reinterpret_cast<device bfloat*>(output_buf);
                device bfloat* dst0 = gathered_h + (ulong(group_y) * 32ul + 0ul) * a.rows;
                if (rid0 != UINT_MAX) {
                    bad |= nonfinite_bit(g_sum0) | nonfinite_bit(u_sum0);
                    const float act0 = (g_sum0 * sigmoid_f32(g_sum0)) * u_sum0;
                    bad |= nonfinite_bit(act0);
                    dst0[col_out] = clean_bf16(act0, bad);
                } else {
                    dst0[col_out] = bfloat(0);
                }
                device bfloat* dst1 = gathered_h + (ulong(group_y) * 32ul + 1ul) * a.rows;
                if (n == 2u && rid1 != UINT_MAX) {
                    bad |= nonfinite_bit(g_sum1) | nonfinite_bit(u_sum1);
                    const float act1 = (g_sum1 * sigmoid_f32(g_sum1)) * u_sum1;
                    bad |= nonfinite_bit(act1);
                    dst1[col_out] = clean_bf16(act1, bad);
                } else {
                    dst1[col_out] = bfloat(0);
                }
                #pragma unroll
                for (ushort r = 2; r < 8; ++r) {
                    (gathered_h + (ulong(group_y) * 32ul + r) * a.rows)[col_out] = bfloat(0);
                }
            } else {
                if (rid0 != UINT_MAX) {
                    bad |= nonfinite_bit(g_sum0) | nonfinite_bit(u_sum0);
                    const float act0 = (g_sum0 * sigmoid_f32(g_sum0)) * u_sum0;
                    bad |= nonfinite_bit(act0);
                    reinterpret_cast<device float*>(output_buf + ulong(rid0) * a.output_stride_bytes)[col_out] = act0;
                }
                if (n == 2u && rid1 != UINT_MAX) {
                    bad |= nonfinite_bit(g_sum1) | nonfinite_bit(u_sum1);
                    const float act1 = (g_sum1 * sigmoid_f32(g_sum1)) * u_sum1;
                    bad |= nonfinite_bit(act1);
                    reinterpret_cast<device float*>(output_buf + ulong(rid1) * a.output_stride_bytes)[col_out] = act1;
                }
            }
        }
    }
}

#if TOKSPUR_PRIVATE_PLANAR_GU_STAGE64
// A current 64-input window shares native Q4_K/Q5_K payload bytes across its two ordered K32 products. No future window lives in registers.
static inline void stage64_planar_gu_v3(device const uchar* exp_base, uint out_row, uint window64,
    uint blocks_per_row, ulong qs_offset, ulong qh_offset, ulong sm_offset, ulong dm_offset,
    bool live_col, threadgroup half* destination, thread uint& bad) {
    const uint sb = window64 >> 2u;
    if (!live_col || sb >= blocks_per_row) {
        threadgroup half4* words = reinterpret_cast<threadgroup half4*>(destination);
        #pragma unroll
        for (ushort w = 0; w < 16; ++w) words[w] = half4(0);
        return;
    }
    const ulong blk = ulong(out_row) * blocks_per_row + sb;
    const uint pair = window64 & 3u;
    const half2 dm = reinterpret_cast<device const half2*>(exp_base + dm_offset)[blk];
    float d = float(dm.x), dmin = float(dm.y);
    uint scale_bad = 0u;
    if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) {
        scale_bad = 1u;
        d = 0.0f;
        dmin = 0.0f;
    }
    device const uchar* sm = exp_base + sm_offset + blk * 16ul;
    device const uint4* qs = reinterpret_cast<device const uint4*>(exp_base + qs_offset + blk * 128ul);
    const uint4 qlo = qs[pair * 2u], qhi = qs[pair * 2u + 1u];
    device const uint4* qh = reinterpret_cast<device const uint4*>(exp_base + qh_offset + blk * 32ul);
    const uint4 hlo = planar_encoding == 13u ? qh[0] : uint4(0u);
    const uint4 hhi = planar_encoding == 13u ? qh[1] : uint4(0u);
    const float q_max = planar_encoding == 13u ? 31.0f : 15.0f;
    #pragma unroll
    for (ushort part = 0; part < 2; ++part) {
        const uint grp = pair * 2u + uint(part);
        const uint shift = uint(part) * 4u;
        const float dl = d * float(sm[grp]);
        const float ml = dmin * float(sm[8u + grp]);
        uint local_bad = scale_bad;
        if (nonfinite_bit(dl) || nonfinite_bit(ml)) local_bad = 1u;
        bad |= local_bad;
        const bool block_clean = !local_bad && (fabs(dl) * q_max + fabs(ml) <= 65504.0f);
        threadgroup half4* low = reinterpret_cast<threadgroup half4*>(destination + uint(part) * 32u);
        threadgroup half4* high = low + 4u;
        #pragma unroll
        for (ushort w = 0; w < 4; ++w) {
            uchar4 q0 = (as_type<uchar4>(qlo[w]) >> uchar4(shift)) & uchar4(15u);
            uchar4 q1 = (as_type<uchar4>(qhi[w]) >> uchar4(shift)) & uchar4(15u);
            if (planar_encoding == 13u) {
                q0 |= ((as_type<uchar4>(hlo[w]) >> uchar4(grp)) & uchar4(1u)) << uchar4(4u);
                q1 |= ((as_type<uchar4>(hhi[w]) >> uchar4(grp)) & uchar4(1u)) << uchar4(4u);
            }
            const float4 f0 = float4(q0) * dl - ml;
            const float4 f1 = float4(q1) * dl - ml;
            const half4 h0 = half4(f0), h1 = half4(f1);
            if (!block_clean) {
                bad |= nonfinite_vec4(f0) | nonfinite_vec4(f1);
                bad |= nonfinite_half_bit(h0.x) | nonfinite_half_bit(h0.y) | nonfinite_half_bit(h0.z) | nonfinite_half_bit(h0.w);
                bad |= nonfinite_half_bit(h1.x) | nonfinite_half_bit(h1.y) | nonfinite_half_bit(h1.z) | nonfinite_half_bit(h1.w);
            }
            low[w] = h0;
            high[w] = h1;
        }
    }
}
#endif

template <ushort MTile>
static inline void run_nax_grouped_moe_gate_up_swiglu_v3(
    threadgroup PlanarNaxGateUpSmemV3& smem,
    device const uchar* gate_exp,
    device const uchar* up_exp,
    device bfloat* task_input,
    device uchar* output_buf,
    constant PlanarNaxMoeArgs& a,
    uint group_y,
    uint first_col,
    uint n,
    uint my_rid,
    ushort sg,
    ushort lane,
    thread uint& bad
) {
    constexpr ushort ColsPerSG = 32, KS = 32;
#if TOKSPUR_PRIVATE_PLANAR_COLUMN32
    const ushort local_col = lane;
    const ushort ai = 0u;
#else
    const ushort local_col = lane >> 1u;
    const ushort ai = lane & 1u;
#endif
    const uint out_row0 = first_col + uint(local_col);
#if !TOKSPUR_PRIVATE_PLANAR_COLUMN32
    const uint out_row1 = first_col + 16u + uint(local_col);
    const bool live1 = out_row1 < a.rows;
#endif
    const bool live0 = out_row0 < a.rows;
    const uint block_cols = (planar_encoding == 12u || planar_encoding == 13u) ? 256u : 32u;
    const uint blocks_per_row = a.columns / block_cols;
    device const uchar* exp_src = sg ? up_exp : gate_exp;

    auto a_tensor = tensor(task_input, dextents<int, 2>{int(a.columns), MTile}, array<int, 2>{1, int(a.columns)});
    constexpr auto descriptor = matmul2d_descriptor(
        MTile, ColsPerSG, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(
        smem.b_stage[sg][0], dextents<int, 2>{KS, ColsPerSG}, array<int, 2>{1, KS});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(
        smem.b_stage[sg][1], dextents<int, 2>{KS, ColsPerSG}, array<int, 2>{1, KS});
    auto b0 = bt0.slice<KS, ColsPerSG>(0, 0);
    auto b1 = bt1.slice<KS, ColsPerSG>(0, 0);
    auto a0 = a_tensor.template slice<KS, MTile>(0, 0);
    auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    const uint steps = a.columns / KS;
#if TOKSPUR_PRIVATE_PLANAR_GU_STAGE64
    if (planar_encoding == 12u || planar_encoding == 13u) {
        constexpr ushort WindowK = 64;
        threadgroup half* stage = smem.k64_stage[sg];
        tensor<threadgroup half, dextents<int, 2>, tensor_inline> window(
            stage, dextents<int, 2>{WindowK, ColsPerSG}, array<int, 2>{1, WindowK});
        auto first = window.slice<KS, ColsPerSG>(0, 0);
        auto second = window.slice<KS, ColsPerSG>(KS, 0);
        for (uint at = 0u; at < a.columns; at += WindowK) {
            stage64_planar_gu_v3(exp_src, live0 ? out_row0 : 0u, at / WindowK, blocks_per_row,
                a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live0, stage + uint(local_col) * WindowK, bad);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            auto a_first = a_tensor.template slice<KS, MTile>(int(at), 0);
            auto a_second = a_tensor.template slice<KS, MTile>(int(at + KS), 0);
            operation.run(a_first, first, acc);
            operation.run(a_second, second, acc);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else
#endif
    if (steps > 0u) {
#if TOKSPUR_PRIVATE_PLANAR_COLUMN32
        half4x4 w0, w1;
        decode32_planar_v3(exp_src, live0 ? out_row0 : 0u, 0u, blocks_per_row,
            a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live0, w0, w1, bad);
#else
        half4x4 w0 = decode16_planar_v3(exp_src, live0 ? out_row0 : 0u, uint(ai), blocks_per_row,
            a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live0, bad);
        half4x4 w1 = decode16_planar_v3(exp_src, live1 ? out_row1 : 0u, uint(ai), blocks_per_row,
            a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live1, bad);
#endif

        for (uint step = 0u; step < steps; ++step) {
            threadgroup half* plane = smem.b_stage[sg][step & 1u];
            threadgroup half4* dst0 = reinterpret_cast<threadgroup half4*>(plane + uint(local_col) * KS + uint(ai) * 16u);
#if TOKSPUR_PRIVATE_PLANAR_COLUMN32
            threadgroup half4* dst1 = dst0 + 4u;
#else
            threadgroup half4* dst1 = reinterpret_cast<threadgroup half4*>(plane + uint(16u + local_col) * KS + uint(ai) * 16u);
#endif
            #pragma unroll
            for (ushort i = 0; i < 4; ++i) { dst0[i] = w0[i]; dst1[i] = w1[i]; }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            if (step + 1u < steps) {
#if TOKSPUR_PRIVATE_PLANAR_COLUMN32
                decode32_planar_v3(exp_src, live0 ? out_row0 : 0u, step + 1u, blocks_per_row,
                    a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live0, w0, w1, bad);
#else
                const uint next_ci = (step + 1u) * 2u + uint(ai);
                w0 = decode16_planar_v3(exp_src, live0 ? out_row0 : 0u, next_ci, blocks_per_row,
                    a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live0, bad);
                w1 = decode16_planar_v3(exp_src, live1 ? out_row1 : 0u, next_ci, blocks_per_row,
                    a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live1, bad);
#endif
            }
            auto a_slice = a_tensor.template slice<KS, MTile>(int(step * KS), 0);
            if ((step & 1u) == 0u) operation.run(a_slice, b0, acc);
            else operation.run(a_slice, b1, acc);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        smem.epi.rids[lane] = my_rid;
        #pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            const auto idx = acc.get_multidimensional_index(i);
            smem.epi.gate_epi[uint(idx[1])][uint(idx[0])] = float(acc[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 1u) {
        if (a.write_bf16_gathered) {
            device bfloat* gathered_h = reinterpret_cast<device bfloat*>(output_buf);
            #pragma unroll
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto idx = acc.get_multidimensional_index(i);
                const uint col_idx = uint(idx[0]), r = uint(idx[1]);
                const uint col_out = first_col + col_idx;
                if (col_out < a.rows) {
                    device bfloat* dst_row = gathered_h + (ulong(group_y) * 32ul + r) * a.rows;
                    const uint rid = smem.epi.rids[r];
                    if (r < n && rid != UINT_MAX) {
                        const float g = smem.epi.gate_epi[r][col_idx];
                        const float u = float(acc[i]);
                        bad |= nonfinite_bit(g) | nonfinite_bit(u);
                        const float act = (g * sigmoid_f32(g)) * u;
                        bad |= nonfinite_bit(act);
                        dst_row[col_out] = clean_bf16(act, bad);
                    } else {
                        dst_row[col_out] = bfloat(0);
                    }
                }
            }
        } else {
            #pragma unroll
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto idx = acc.get_multidimensional_index(i);
                const uint col_idx = uint(idx[0]), r = uint(idx[1]);
                const uint col_out = first_col + col_idx;
                const uint rid = smem.epi.rids[r];
                if (r < n && rid != UINT_MAX && col_out < a.rows) {
                    const float g = smem.epi.gate_epi[r][col_idx];
                    const float u = float(acc[i]);
                    bad |= nonfinite_bit(g) | nonfinite_bit(u);
                    const float act = (g * sigmoid_f32(g)) * u;
                    bad |= nonfinite_bit(act);
                    device float* y = reinterpret_cast<device float*>(output_buf + ulong(rid) * a.output_stride_bytes);
                    y[col_out] = act;
                }
            }
        }
    }
}

__attribute__((noinline))
static void run_col_grouped_moe_down_q51_v3(
    device const uchar* exp_base,
    device const bfloat* task_input,
    device float* output,
    constant PlanarNaxMoeArgs& a,
    uint first_col,
    uint n,
    uint rid0,
    uint rid1,
    ushort tid,
    thread uint& bad
) {
    const ushort quad_id = tid >> 2u;
    const ushort qlane = tid & 3u;
    const uint blocks_per_row = a.columns / 32u;

    for (ushort pass = 0; pass < 4; ++pass) {
        const uint col_out = first_col + uint(pass) * 16u + uint(quad_id);
        const bool col_valid = col_out < a.rows;
        float sum0 = 0.0f, sum1 = 0.0f;
        if (col_valid) {
            for (uint b0 = 0u; b0 < blocks_per_row; b0 += 4u) {
                const uint b_idx = b0 + uint(qlane);
                if (b_idx < blocks_per_row) {
                    const ulong blk = ulong(col_out) * blocks_per_row + b_idx;
                    const half2 dm = reinterpret_cast<device const half2*>(exp_base + a.dm_offset)[blk];
                    float d = float(dm.x), m = float(dm.y);
                    if (nonfinite_half_bit(dm.x) || nonfinite_half_bit(dm.y)) { bad |= 1u; d = 0.0f; m = 0.0f; }
                    const uint4 q4 = reinterpret_cast<device const uint4*>(exp_base + a.qs_offset)[blk];
                    const uint qh = reinterpret_cast<device const uint*>(exp_base + a.qh_offset)[blk];

                    device const bfloat* x0 = task_input + b_idx * 32u;
                    device const bfloat* x1 = task_input + a.columns + b_idx * 32u;
                    float dot0 = 0.0f, sx0 = 0.0f;
                    float dot1 = 0.0f, sx1 = 0.0f;
                    #pragma unroll
                    for (ushort w = 0; w < 4; ++w) {
                        const uchar4 qb = as_type<uchar4>(q4[w]);
                        const uint qh_lo = qh >> (uint(w) * 4u);
                        const uint qh_hi = qh >> (16u + uint(w) * 4u);
                        const uchar4 q_lo = (qb & uchar4(15u)) |
                            (uchar4(qh_lo & 1u, (qh_lo >> 1u) & 1u, (qh_lo >> 2u) & 1u, (qh_lo >> 3u) & 1u) << uchar4(4u));
                        const uchar4 q_hi = (qb >> uchar4(4u)) |
                            (uchar4(qh_hi & 1u, (qh_hi >> 1u) & 1u, (qh_hi >> 2u) & 1u, (qh_hi >> 3u) & 1u) << uchar4(4u));

                        const float4 xv0_lo = float4(*reinterpret_cast<device const bfloat4*>(x0 + uint(w) * 4u));
                        const float4 xv0_hi = float4(*reinterpret_cast<device const bfloat4*>(x0 + 16u + uint(w) * 4u));
                        sx0 += (xv0_lo.x + xv0_lo.y + xv0_lo.z + xv0_lo.w) + (xv0_hi.x + xv0_hi.y + xv0_hi.z + xv0_hi.w);
                        dot0 += dot8_u8_f32_v3(q_lo, q_hi, xv0_lo, xv0_hi);

                        if (n == 2u) {
                            const float4 xv1_lo = float4(*reinterpret_cast<device const bfloat4*>(x1 + uint(w) * 4u));
                            const float4 xv1_hi = float4(*reinterpret_cast<device const bfloat4*>(x1 + 16u + uint(w) * 4u));
                            sx1 += (xv1_lo.x + xv1_lo.y + xv1_lo.z + xv1_lo.w) + (xv1_hi.x + xv1_hi.y + xv1_hi.z + xv1_hi.w);
                            dot1 += dot8_u8_f32_v3(q_lo, q_hi, xv1_lo, xv1_hi);
                        }
                    }
                    sum0 += d * dot0 + m * sx0;
                    if (n == 2u) sum1 += d * dot1 + m * sx1;
                }
            }
        }
        sum0 += simd_shuffle_xor(sum0, 1u); sum0 += simd_shuffle_xor(sum0, 2u);
        if (n == 2u) { sum1 += simd_shuffle_xor(sum1, 1u); sum1 += simd_shuffle_xor(sum1, 2u); }
        if (qlane == 0u && col_valid) {
            if (rid0 != UINT_MAX) {
                bad |= nonfinite_bit(sum0);
                reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(rid0) * a.output_stride_bytes)[col_out] = sum0;
            }
            if (n == 2u && rid1 != UINT_MAX) {
                bad |= nonfinite_bit(sum1);
                reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(rid1) * a.output_stride_bytes)[col_out] = sum1;
            }
        }
    }
}

template <ushort MTile>
static inline void run_nax_grouped_moe_down_v3(
    threadgroup PlanarNaxSgSmemV3& smem_sg,
    device const uchar* exp_base,
    device bfloat* task_input,
    device float* output,
    constant PlanarNaxMoeArgs& a,
    uint sg_first_col,
    uint n,
    uint my_rid,
    ushort lane,
    thread uint& bad
) {
    constexpr ushort ColsPerSG = 32, KS = 32;
#if TOKSPUR_PRIVATE_PLANAR_COLUMN32
    const ushort local_col = lane;
    const ushort ai = 0u;
#else
    const ushort local_col = lane >> 1u;
    const ushort ai = lane & 1u;
#endif
    const uint out_row0 = sg_first_col + uint(local_col);
#if !TOKSPUR_PRIVATE_PLANAR_COLUMN32
    const uint out_row1 = sg_first_col + 16u + uint(local_col);
    const bool live1 = out_row1 < a.rows;
#endif
    const bool live0 = out_row0 < a.rows;
    const uint blocks_per_row = a.columns / 32u;

    auto a_tensor = tensor(task_input, dextents<int, 2>{int(a.columns), MTile}, array<int, 2>{1, int(a.columns)});
    constexpr auto descriptor = matmul2d_descriptor(
        MTile, ColsPerSG, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(
        smem_sg.b_stage[0], dextents<int, 2>{KS, ColsPerSG}, array<int, 2>{1, KS});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(
        smem_sg.b_stage[1], dextents<int, 2>{KS, ColsPerSG}, array<int, 2>{1, KS});
    auto b0 = bt0.slice<KS, ColsPerSG>(0, 0);
    auto b1 = bt1.slice<KS, ColsPerSG>(0, 0);
    auto a0 = a_tensor.template slice<KS, MTile>(0, 0);
    auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    const uint steps = a.columns / KS;
    if (steps > 0u) {
#if TOKSPUR_PRIVATE_PLANAR_COLUMN32
        half4x4 w0, w1;
        decode32_planar_v3(exp_base, live0 ? out_row0 : 0u, 0u, blocks_per_row,
            a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live0, w0, w1, bad);
#else
        half4x4 w0 = decode16_planar_v3(exp_base, live0 ? out_row0 : 0u, uint(ai), blocks_per_row,
            a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live0, bad);
        half4x4 w1 = decode16_planar_v3(exp_base, live1 ? out_row1 : 0u, uint(ai), blocks_per_row,
            a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live1, bad);
#endif

        for (uint step = 0u; step < steps; ++step) {
            threadgroup half* plane = smem_sg.b_stage[step & 1u];
            threadgroup half4* dst0 = reinterpret_cast<threadgroup half4*>(plane + uint(local_col) * KS + uint(ai) * 16u);
#if TOKSPUR_PRIVATE_PLANAR_COLUMN32
            threadgroup half4* dst1 = dst0 + 4u;
#else
            threadgroup half4* dst1 = reinterpret_cast<threadgroup half4*>(plane + uint(16u + local_col) * KS + uint(ai) * 16u);
#endif
            #pragma unroll
            for (ushort i = 0; i < 4; ++i) { dst0[i] = w0[i]; dst1[i] = w1[i]; }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            if (step + 1u < steps) {
#if TOKSPUR_PRIVATE_PLANAR_COLUMN32
                decode32_planar_v3(exp_base, live0 ? out_row0 : 0u, step + 1u, blocks_per_row,
                    a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live0, w0, w1, bad);
#else
                const uint next_ci = (step + 1u) * 2u + uint(ai);
                w0 = decode16_planar_v3(exp_base, live0 ? out_row0 : 0u, next_ci, blocks_per_row,
                    a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live0, bad);
                w1 = decode16_planar_v3(exp_base, live1 ? out_row1 : 0u, next_ci, blocks_per_row,
                    a.qs_offset, a.qh_offset, a.sm_offset, a.dm_offset, live1, bad);
#endif
            }
            auto a_slice = a_tensor.template slice<KS, MTile>(int(step * KS), 0);
            if ((step & 1u) == 0u) operation.run(a_slice, b0, acc);
            else operation.run(a_slice, b1, acc);
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    smem_sg.rids[lane] = my_rid;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto idx = acc.get_multidimensional_index(i);
        const uint col_out = sg_first_col + uint(idx[0]);
        const uint r = uint(idx[1]);
        const uint rid = smem_sg.rids[r];
        if (r < n && rid != UINT_MAX && col_out < a.rows) {
            const float val = float(acc[i]);
            bad |= nonfinite_bit(val);
            device float* y = reinterpret_cast<device float*>(
                reinterpret_cast<device uchar*>(output) + ulong(rid) * a.output_stride_bytes);
            y[col_out] = val;
        }
    }
}

} // namespace planar_v3_detail

// Pre-gather routed MoE input rows into contiguous BF16 task tiles [live_tiles * 32, channels]
// so all column tiles of nax_grouped_moe_gate_up_swiglu_v3 stream A via NAX tensor ops with zero a_stage SRAM or TG barriers.
kernel void nax_moe_gather_bf16_v3(device const uchar* src [[buffer(0)]],
                                   device bfloat* gathered_x [[buffer(1)]],
                                   constant planar_v3_detail::PlanarNaxMoeArgs& a [[buffer(2)]],
                                   device const uint* ids [[buffer(3)]],
                                   device const uint* map [[buffer(4)]],
                                   device const uint* tasks [[buffer(5)]],
                                   device const uint* counts [[buffer(6)]],
                                   device atomic_uint* error [[buffer(7)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint rids[32];
    if (!a.tokens || !a.selected || !a.experts || !a.columns || a.columns % 32u != 0u ||
        tasks[0] > a.maximum_tasks || tasks[1] != a.tokens * a.selected) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y >= tasks[0]) return;
    const uint expert = tasks[2u + 2u * group.y];
    const uint start = tasks[3u + 2u * group.y];
    if (expert >= a.experts || start >= a.tokens || counts[expert] > a.tokens || start >= counts[expert]) {
        if (!tid && !group.x) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint n = min(32u, counts[expert] - start);
    uint bad = 0u;
    if (tid < 32u) {
        rids[tid] = (tid < n)
            ? planar_v3_detail::route_index_v3(ids, map, expert, start + tid, a.tokens, a.selected, bad)
            : UINT_MAX;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort padded_n = (n <= 8u) ? 8u : ((n <= 16u) ? 16u : 32u);
    if (a.reserved0 != 0u) {
        const uint octs = a.columns >> 3u;
        device const bfloat* x_bf16 = reinterpret_cast<device const bfloat*>(src);
        for (ushort r = 0; r < padded_n; ++r) {
            device uint4* dst8 = reinterpret_cast<device uint4*>(gathered_x + (ulong(group.y) * 32ul + r) * a.columns);
            const uint rid = rids[r];
            const uint row_tok = (rid != UINT_MAX) ? (rid / a.selected) : UINT_MAX;
            if (uint(r) < n && row_tok < a.tokens) {
                device const uint4* src8 = reinterpret_cast<device const uint4*>(x_bf16 + ulong(row_tok) * a.columns);
                for (uint col8 = group.x * 128u + tid; col8 < octs; col8 += 512u) {
                    dst8[col8] = src8[col8];
                }
            } else {
                for (uint col8 = group.x * 128u + tid; col8 < octs; col8 += 512u) {
                    dst8[col8] = uint4(0u);
                }
            }
        }
    } else {
        const uint quads = a.columns >> 2u;
        const ulong stride_f32 = a.input_stride_bytes / sizeof(float);
        device const float* input = reinterpret_cast<device const float*>(src);
        for (ushort r = 0; r < padded_n; ++r) {
            device bfloat4* dst4 = reinterpret_cast<device bfloat4*>(gathered_x + (ulong(group.y) * 32ul + r) * a.columns);
            const uint rid = rids[r];
            const uint row_tok = (rid != UINT_MAX) ? (rid / a.selected) : UINT_MAX;
            if (uint(r) < n && row_tok < a.tokens) {
                device const float* src_row = input + ulong(row_tok) * stride_f32;
                for (uint col4 = group.x * 128u + tid; col4 < quads; col4 += 512u) {
                    const float4 v = float4(*reinterpret_cast<device const packed_float4*>(src_row + col4 * 4u));
                    dst4[col4] = bfloat4(
                        planar_v3_detail::clean_bf16(v.x, bad),
                        planar_v3_detail::clean_bf16(v.y, bad),
                        planar_v3_detail::clean_bf16(v.z, bad),
                        planar_v3_detail::clean_bf16(v.w, bad));
                }
            } else {
                for (uint col4 = group.x * 128u + tid; col4 < quads; col4 += 512u) {
                    dst4[col4] = bfloat4(0);
                }
            }
        }
    }
    planar_v3_detail::publish_sticky_error(bad, error);
}

// NAX grouped MoE co-located gate + up + SwiGLU kernel (m = 1..2048).
// 64 threads (2 SGs: SG 0 = gate, SG 1 = up, 32 hidden cols/TG, 8.0 KiB SRAM = 4 TGs/core = 8 SGs/core)
// with n <= 2 quad_sum column-dot fast path and 2-barrier direct-register SwiGLU epilogue.
kernel void nax_grouped_moe_gate_up_swiglu_v3(device const uchar* pair_lo [[buffer(0)]],
                                              device const uchar* pair_hi [[buffer(1)]],
                                              device bfloat* gathered_x [[buffer(2)]],
                                              device uchar* output_buf [[buffer(3)]],
                                              constant planar_v3_detail::PlanarNaxMoeArgs& a [[buffer(4)]],
                                              device const uint* ids [[buffer(5)]],
                                              device const uint* map [[buffer(6)]],
                                              device atomic_uint* error [[buffer(7)]],
                                              device const uint* tasks [[buffer(8)]],
                                              device const uint* counts [[buffer(9)]],
                                              uint2 group [[threadgroup_position_in_grid]],
                                              ushort tid [[thread_index_in_threadgroup]],
                                              ushort sg [[simdgroup_index_in_threadgroup]],
                                              ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup planar_v3_detail::PlanarNaxGateUpSmemV3 smem;
    static_assert(sizeof(smem) <= 8192, "nax_grouped_moe_gate_up_swiglu_v3 8 KiB 4-TG/core occupancy budget");

    const uint block_cols = (planar_encoding == 12u || planar_encoding == 13u) ? 256u : 32u;
    if (!a.tokens || !a.selected || !a.experts || a.experts % 2u != 0u ||
        !a.columns || a.columns % block_cols != 0u || !a.rows || a.rows % 32u != 0u ||
        tasks[0] > a.maximum_tasks || tasks[1] != a.tokens * a.selected) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint first_col = group.x * 32u;
    if (group.y >= tasks[0] || first_col >= a.rows) return;

    const uint expert = tasks[2u + 2u * group.y];
    const uint start = tasks[3u + 2u * group.y];
    if (expert >= a.experts || start >= a.tokens || counts[expert] > a.tokens || start >= counts[expert]) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint n = min(32u, counts[expert] - start);
    uint bad = 0u;
    const uint my_rid = (uint(lane) < n)
        ? planar_v3_detail::route_index_v3(ids, map, expert, start + uint(lane), a.tokens, a.selected, bad)
        : UINT_MAX;

    const uint half_experts = a.experts >> 1u;
    device const uchar* half_base = (expert < half_experts) ? pair_lo : pair_hi;
    const uint local_e = (expert < half_experts) ? expert : (expert - half_experts);
    device const uchar* gate_exp = half_base + ulong(2u * local_e) * a.planar_expert_bytes;
    device const uchar* up_exp = gate_exp + a.planar_expert_bytes;
    device bfloat* task_input = gathered_x + (ulong(group.y) * 32ul) * a.columns;

    if ((planar_encoding == 12u || planar_encoding == 13u) && n <= 2u) {
        const uint rid0 = simd_broadcast(my_rid, 0u);
        const uint rid1 = (n == 2u) ? simd_broadcast(my_rid, 1u) : UINT_MAX;
        planar_v3_detail::run_col_grouped_moe_gate_up_swiglu_v3(
            gate_exp, up_exp, task_input, output_buf, a, group.y, first_col, n, rid0, rid1, tid, bad);
    } else if (n <= 8u) {
        planar_v3_detail::run_nax_grouped_moe_gate_up_swiglu_v3<8>(
            smem, gate_exp, up_exp, task_input, output_buf, a, group.y, first_col, n, my_rid, sg, lane, bad);
    } else if (n <= 16u) {
        planar_v3_detail::run_nax_grouped_moe_gate_up_swiglu_v3<16>(
            smem, gate_exp, up_exp, task_input, output_buf, a, group.y, first_col, n, my_rid, sg, lane, bad);
    } else {
        planar_v3_detail::run_nax_grouped_moe_gate_up_swiglu_v3<32>(
            smem, gate_exp, up_exp, task_input, output_buf, a, group.y, first_col, n, my_rid, sg, lane, bad);
    }
    planar_v3_detail::publish_sticky_error(bad, error);
}

// NAX grouped MoE planar Q5_1 / Q8_0 down projection (m = 1..2048).
// 64 threads (2 independent SGs, 64 output cols/TG, 8.0 KiB SRAM = 4 TGs/core = 8 SGs/core)
// with n <= 2 quad_sum column-dot fast path and direct-register epilogue.
kernel void nax_grouped_moe_down_v3(device const uchar* down_weights [[buffer(0)]],
                                    device bfloat* gathered_h [[buffer(1)]],
                                    device float* output [[buffer(2)]],
                                    constant planar_v3_detail::PlanarNaxMoeArgs& a [[buffer(3)]],
                                    device const uint* ids [[buffer(4)]],
                                    device const uint* map [[buffer(6)]],
                                    device atomic_uint* error [[buffer(7)]],
                                    device const uint* tasks [[buffer(8)]],
                                    device const uint* counts [[buffer(9)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    ushort tid [[thread_index_in_threadgroup]],
                                    ushort sg [[simdgroup_index_in_threadgroup]],
                                    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup planar_v3_detail::PlanarNaxSgSmemV3 smem[2];
    static_assert(sizeof(smem) <= 8192, "nax_grouped_moe_down_v3 8 KiB 4-TG/core occupancy budget");

    if ((planar_encoding != 7u && planar_encoding != 8u) ||
        !a.tokens || !a.selected || !a.experts ||
        !a.columns || a.columns % 32u != 0u || !a.rows || a.rows % 4u != 0u ||
        tasks[0] > a.maximum_tasks || tasks[1] != a.tokens * a.selected) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint first_col = group.x * 64u;
    if (group.y >= tasks[0] || first_col >= a.rows) return;

    const uint expert = tasks[2u + 2u * group.y];
    const uint start = tasks[3u + 2u * group.y];
    if (expert >= a.experts || start >= a.tokens || counts[expert] > a.tokens || start >= counts[expert]) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint n = min(32u, counts[expert] - start);
    uint bad = 0u;
    const uint my_rid = (uint(lane) < n)
        ? planar_v3_detail::route_index_v3(ids, map, expert, start + uint(lane), a.tokens, a.selected, bad)
        : UINT_MAX;

    device const uchar* exp_base = down_weights + ulong(expert) * a.planar_expert_bytes;
    device bfloat* task_input = gathered_h + (ulong(group.y) * 32ul) * a.columns;

    if (planar_encoding == 7u && n <= 2u) {
        const uint rid0 = simd_broadcast(my_rid, 0u);
        const uint rid1 = (n == 2u) ? simd_broadcast(my_rid, 1u) : UINT_MAX;
        planar_v3_detail::run_col_grouped_moe_down_q51_v3(
            exp_base, task_input, output, a, first_col, n, rid0, rid1, tid, bad);
    } else {
        const uint sg_first_col = first_col + uint(sg) * 32u;
        if (sg_first_col < a.rows) {
            if (n <= 8u) {
                planar_v3_detail::run_nax_grouped_moe_down_v3<8>(
                    smem[sg], exp_base, task_input, output, a, sg_first_col, n, my_rid, lane, bad);
            } else if (n <= 16u) {
                planar_v3_detail::run_nax_grouped_moe_down_v3<16>(
                    smem[sg], exp_base, task_input, output, a, sg_first_col, n, my_rid, lane, bad);
            } else {
                planar_v3_detail::run_nax_grouped_moe_down_v3<32>(
                    smem[sg], exp_base, task_input, output, a, sg_first_col, n, my_rid, lane, bad);
            }
        }
    }
    planar_v3_detail::publish_sticky_error(bad, error);
}
