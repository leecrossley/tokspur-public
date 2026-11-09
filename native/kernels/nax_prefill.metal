// Bounded Metal 4.0 Neural Accelerator (MPP TensorOps) prefill kernels.
// Dequantisation and staged tile structure adapt llama.cpp 6050b467 (MIT, ggml authors, see ../third_party/llama-cpp/LICENSE). The three-part BF16 weight split, matmul over its planes and split reduction adapt Splash 1.4.0 (Apache-2.0, see NOTICE).
// Tokspur owns graph state, row views, route compaction, finite error reporting and scheduling.
#pragma clang fp reassociate(off)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#ifndef TOKSPUR_PRIVATE_DENSE_M32_SINGLE_TILE
#define TOKSPUR_PRIVATE_DENSE_M32_SINGLE_TILE 0
#endif
#if TOKSPUR_PRIVATE_DENSE_M32_SINGLE_TILE != 0 && TOKSPUR_PRIVATE_DENSE_M32_SINGLE_TILE != 1
#error "TOKSPUR_PRIVATE_DENSE_M32_SINGLE_TILE must be 0 or 1"
#endif

#ifndef TOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE
#define TOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE 0
#endif
#if TOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE != 0 && TOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE != 1
#error "TOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE must be 0 or 1"
#endif

using namespace metal;
using namespace mpp::tensor_ops;

constant uint nax_encoding [[function_constant(0)]];
constant uint nax_layout [[function_constant(1)]]; // 0 GGUF, 1 Q8 lanes-v1, 2 Q5_1 lanes-v1.

namespace nax_detail {

struct block_q5_1 { half d, m; uchar qh[4], qs[16]; };
struct block_q8_0 { half d; char qs[32]; };
struct block_q4_K { half d, dmin; uchar scales[12], qs[128]; };
struct block_q5_K { half d, dmin; uchar scales[12], qh[32], qs[128]; };
static_assert(sizeof(block_q5_1) == 24 && sizeof(block_q8_0) == 34 && sizeof(block_q4_K) == 144 && sizeof(block_q5_K) == 176, "GGUF block bytes");

struct GroupedMmArgs {
    uint columns, rows, row_bytes, experts;
    uint selected, tokens, broadcast, maximum_tasks;
    ulong expert_stride_bytes, input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(GroupedMmArgs) == 56, "grouped mm ABI");

struct MoeRouterWindowArgs {
    uint columns, experts, tokens, reserved;
    ulong input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(MoeRouterWindowArgs) == 32, "router window ABI");

struct NaxConvertBf16Args {
    uint columns, rows, padded_rows, reserved;
    ulong input_stride_bytes;
};
static_assert(sizeof(NaxConvertBf16Args) == 24, "NAX convert BF16 ABI");

struct NaxQsaAttentionArgs {
    uint rows, base, query_heads, kv_heads;
    uint head_width, block_budget, q_stride, id_stride;
};
static_assert(sizeof(NaxQsaAttentionArgs) == 32, "NAX QSA attention ABI");

struct NaxQsaG12Args {
    uint rows, base, query_heads, kv_heads;
    uint head_width, block_budget, q_stride, id_stride;
    uint num_tiles, list_stride, splits, sparse_mode;
};
static_assert(sizeof(NaxQsaG12Args) == 48, "NAX QSA G12 attention ABI");

struct NaxVerifyMoeArgs {
    uint columns, rows, row_bytes, experts;
    uint selected, tokens, broadcast, swiglu;
    ulong expert_stride_bytes, input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(NaxVerifyMoeArgs) == 56, "NAX verify MoE ABI");

static inline uint nonfinite_bit(float v) {
    return (as_type<uint>(v) & 0x7f800000u) == 0x7f800000u ? 1u : 0u;
}

static inline uint nonfinite_half_bit(half v) {
    return (as_type<ushort>(v) & 0x7c00u) == 0x7c00u ? 1u : 0u;
}

static inline float stored_half(device const uchar* p) {
    return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
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

static inline ushort4 clean_bf16_bits4(ushort4 h, thread uint& bad) {
    const bool4 nf = (h & ushort4(0x7f80u)) == ushort4(0x7f80u);
    if (any(nf)) {
        bad |= 1u;
        return select(h, ushort4(0), nf);
    }
    return h;
}

// Sterbenz exact 3-part FP32 -> BF16 split: hi + mid + lo == w in FP32.
static inline ushort3 split_f32_bf16(float w, thread uint& bad) {
    if (nonfinite_bit(w)) {
        bad |= 1u;
        return ushort3(0);
    }
    const uint hi = as_type<uint>(w) & 0xFFFF0000u;
    const float r = w - as_type<float>(hi);
    const uint mid = as_type<uint>(r) & 0xFFFF0000u;
    const float lo = r - as_type<float>(mid);
    return ushort3(ushort(hi >> 16), ushort(mid >> 16), ushort(as_type<uint>(lo) >> 16));
}

struct Half32 {
    half4 v[8];
};

static inline Half32 decode_q8_pair64(device const uchar* row, uint k_step, ushort half_idx, uint columns, bool live_col, thread uint& bad) {
    Half32 out;
    if (!live_col) {
        #pragma unroll
        for (ushort i = 0; i < 8; ++i) out.v[i] = half4(0);
        return out;
    }
    if (nax_layout == 0) {
        device const block_q8_0* xb = reinterpret_cast<device const block_q8_0*>(row) + (2u * k_step + uint(half_idx));
        float d = float(xb->d);
        if (nonfinite_bit(d)) { bad |= 1u; d = 0.0f; }
        device const packed_char4* qs = reinterpret_cast<device const packed_char4*>(xb->qs);
        #pragma unroll
        for (ushort i = 0; i < 8; ++i) {
            const float4 f = float4(qs[i]) * d;
            const half4 h = half4(f);
            bad |= nonfinite_half_bit(h.x) | nonfinite_half_bit(h.y) | nonfinite_half_bit(h.z) | nonfinite_half_bit(h.w);
            out.v[i] = h;
        }
        return out;
    }
    const uint blocks = columns / 32u;
    const half2 d01 = *reinterpret_cast<device const half2*>(row + 4u * k_step);
    float d0 = float(d01.x), d1 = float(d01.y);
    if (nonfinite_bit(d0)) { bad |= 1u; d0 = 0.0f; }
    if (nonfinite_bit(d1)) { bad |= 1u; d1 = 0.0f; }
    const uint b0 = 2u * k_step;
    const uint run = b0 / 16u, width = min(16u, blocks - run * 16u), lane_off = b0 % 16u;
    const ulong base = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(lane_off);
    const uint q_base = uint(half_idx) * 16u;
    #pragma unroll
    for (ushort i = 0; i < 4; ++i) {
        const uint q0 = q_base + i * 4u;
        const char2 c0 = *reinterpret_cast<device const char2*>(row + base + ulong(q0 + 0u) * width);
        const char2 c1 = *reinterpret_cast<device const char2*>(row + base + ulong(q0 + 1u) * width);
        const char2 c2 = *reinterpret_cast<device const char2*>(row + base + ulong(q0 + 2u) * width);
        const char2 c3 = *reinterpret_cast<device const char2*>(row + base + ulong(q0 + 3u) * width);
        const half4 h0 = half4(float4(float(c0.x), float(c1.x), float(c2.x), float(c3.x)) * d0);
        const half4 h1 = half4(float4(float(c0.y), float(c1.y), float(c2.y), float(c3.y)) * d1);
        bad |= nonfinite_half_bit(h0.x) | nonfinite_half_bit(h0.y) | nonfinite_half_bit(h0.z) | nonfinite_half_bit(h0.w);
        bad |= nonfinite_half_bit(h1.x) | nonfinite_half_bit(h1.y) | nonfinite_half_bit(h1.z) | nonfinite_half_bit(h1.w);
        out.v[i] = h0;
        out.v[i + 4] = h1;
    }
    return out;
}

static inline uchar2 scale_min_k4(int j, int k, device const uchar* q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2)),
                          uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

static inline half4x4 decode16_any(device const uchar* row, uint ci, uint columns, bool live_col, thread uint& bad) {
    half4x4 result(0);
    if (!live_col) return result;
    if (nax_layout == 0) {
        if (nax_encoding == 7) {
            device const block_q5_1* xb = reinterpret_cast<device const block_q5_1*>(row) + ci / 2;
            const short il = short(ci % 2);
            device const ushort* qs = reinterpret_cast<device const ushort*>(xb) + 4;
            float d = float(xb->d), m = float(xb->m);
            if (nonfinite_bit(d) || nonfinite_bit(m)) { bad |= 1u; d = 0.0f; m = 0.0f; }
            const ushort mask = il ? 0x00F0 : 0x000F;
            const uint qh = *reinterpret_cast<device const uint*>(xb->qh);
            const int x_mv = il ? 4 : 0, gh_mv = il ? 12 : 0, gh_bk = il ? 0 : 4;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uchar xh_0 = ((qh >> (gh_mv + 2 * i)) << gh_bk) & 0x10;
                const uchar xh_1 = ((qh >> (gh_mv + 2 * i + 1)) << gh_bk) & 0x10;
                const int x0 = (((qs[i]) & mask) >> x_mv) | xh_0;
                const int x1 = (((qs[i] >> 8) & mask) >> x_mv) | xh_1;
                result[i / 2][2 * (i % 2) + 0] = half(d * float(x0) + m);
                result[i / 2][2 * (i % 2) + 1] = half(d * float(x1) + m);
            }
        } else if (nax_encoding == 8) {
            device const block_q8_0* xb = reinterpret_cast<device const block_q8_0*>(row) + ci / 2;
            const short il = short(ci % 2);
            device const packed_char4* qs = reinterpret_cast<device const packed_char4*>(xb->qs);
            float d = float(xb->d);
            if (nonfinite_bit(d)) { bad |= 1u; d = 0.0f; }
            #pragma unroll
            for (int i = 0; i < 4; ++i) result[i] = half4(float4(qs[4 * il + i]) * d);
        } else if (nax_encoding == 12) {
            device const block_q4_K* xb = reinterpret_cast<device const block_q4_K*>(row) + ci / 16;
            short il = short(ci % 16);
            device const uchar* q = xb->qs + (il / 4) * 32 + 16 * (il & 1);
            const short is = (il / 4) * 2;
            il = il & 3;
            const uchar2 sc = scale_min_k4(is, il / 2, xb->scales);
            float stored_d = float(xb->d), min_v = float(xb->dmin);
            if (nonfinite_bit(stored_d) || nonfinite_bit(min_v)) { bad |= 1u; stored_d = 0.0f; min_v = 0.0f; }
            const float d = il < 2 ? stored_d : stored_d / 16.0f;
            const float dl = d * float(sc[0]), ml = min_v * float(sc[1]);
            const ushort mask = il < 2 ? 0x0F : 0xF0;
            #pragma unroll
            for (int i = 0; i < 16; ++i) result[i / 4][i % 4] = half(dl * float(q[i] & mask) - ml);
        } else {
            device const block_q5_K* xb = reinterpret_cast<device const block_q5_K*>(row) + ci / 16;
            short il = short(ci % 16);
            device const uchar* q = xb->qs + 32 * (il / 4) + 16 * (il & 1);
            device const uchar* qh = xb->qh + 16 * (il & 1);
            const short is = (il / 4) * 2;
            const uchar ul = uchar(1u << (il / 2));
            il = il & 3;
            const uchar2 sc = scale_min_k4(is, il / 2, xb->scales);
            float stored_d = float(xb->d), min_v = float(xb->dmin);
            if (nonfinite_bit(stored_d) || nonfinite_bit(min_v)) { bad |= 1u; stored_d = 0.0f; min_v = 0.0f; }
            const float d = il < 2 ? stored_d : stored_d / 16.0f;
            const float dl = d * float(sc[0]), ml = min_v * float(sc[1]);
            const ushort mask = il < 2 ? 0x0F : 0xF0;
            const float qh_val = il < 2 ? 16.0f : 256.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i)
                result[i / 4][i % 4] = half(dl * (float(q[i] & mask) + (qh[i] & ul ? qh_val : 0.0f)) - ml);
        }
    } else if (nax_layout == 1) {
        const uint blocks = columns / 32u, column = 16u * ci, b = column / 32u;
        float d = stored_half(row + 2u * b);
        if (nonfinite_bit(d)) { bad |= 1u; d = 0.0f; }
        const uint run = b / 16u, width = min(16u, blocks - run * 16u);
        const uint q_base = (ci & 1u) * 16u;
        const ulong base = 2ul * ulong(blocks) + ulong(run) * 512ul + (b % 16u);
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
            const ulong at = base + ulong(q_base + i) * width;
            result[i / 4][i % 4] = half(d * float(as_type<char>(row[at])));
        }
    } else {
        const uint blocks = columns / 32u, b = ci >> 1;
        const half2 dm = *reinterpret_cast<device const half2*>(row + 8u * b);
        float d = float(dm.x), m = float(dm.y);
        if (nonfinite_bit(d) || nonfinite_bit(m)) { bad |= 1u; d = 0.0f; m = 0.0f; }
        const uint qh_shifted = (*reinterpret_cast<device const uint*>(row + 8u * b + 4u)) >> ((ci & 1u) * 16u);
        const uint shift = (ci & 1u) * 4u;
        const uint run = b / 16u, width = min(16u, blocks - run * 16u);
        const ulong base = 8ul * ulong(blocks) + ulong(run) * 256ul + (b % 16u);
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
            const uint q = ((uint(row[base + ulong(i) * width]) >> shift) & 15u) | (((qh_shifted >> i) & 1u) << 4);
            result[i / 4][i % 4] = half(d * float(q) + m);
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        bad |= nonfinite_half_bit(result[r].x) | nonfinite_half_bit(result[r].y) |
               nonfinite_half_bit(result[r].z) | nonfinite_half_bit(result[r].w);
    return result;
}

static inline uint route_index(device const uint* ids, device const uint* map, uint expert, uint at,
    constant GroupedMmArgs& a, thread uint& bad_route) {
    const uint rid = map[ulong(expert) * a.tokens + at];
    if (rid >= a.tokens * a.selected || ids[rid] != expert) {
        bad_route |= 4u;
        return UINT_MAX;
    }
    return rid;
}

static inline half4 decode4_gguf(device const uchar* row, uint step, ushort sub, bool live_col, thread uint& bad) {
    if (!live_col) return half4(0);
    half4 res;
    if (nax_encoding == 8) {
        device const block_q8_0* xb = reinterpret_cast<device const block_q8_0*>(row) + step;
        float d = float(xb->d);
        if (nonfinite_bit(d)) { bad |= 1u; d = 0.0f; }
        const char4 q4 = char4(*reinterpret_cast<device const packed_char4*>(xb->qs + uint(sub) * 4u));
        res = half4(float4(q4) * d);
    } else if (nax_encoding == 7) {
        device const block_q5_1* xb = reinterpret_cast<device const block_q5_1*>(row) + step;
        float d = float(xb->d), m = float(xb->m);
        if (nonfinite_bit(d) || nonfinite_bit(m)) { bad |= 1u; d = 0.0f; m = 0.0f; }
        const uint at = uint(sub) * 4u, shift = (uint(sub) >> 2u) * 4u, qs_off = (uint(sub) & 3u) * 4u;
        const uchar4 qb = uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->qs + qs_off));
        const uint qh = as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->qh)));
        const uint4 low = (uint4(qb) >> shift) & 15u;
        const uint4 bit = ((uint4(qh) >> uint4(at, at + 1u, at + 2u, at + 3u)) & 1u) << 4u;
        res = half4(d * float4(low | bit) + m);
    } else if (nax_encoding == 12) {
        const uint ci = 2u * step + (uint(sub) >> 2u);
        device const block_q4_K* xb = reinterpret_cast<device const block_q4_K*>(row) + ci / 16u;
        short il = short(ci % 16u);
        device const uchar* q = xb->qs + (il / 4) * 32 + 16 * (il & 1) + (uint(sub) & 3u) * 4u;
        const short is = (il / 4) * 2;
        il = il & 3;
        const uchar2 sc = scale_min_k4(is, il / 2, xb->scales);
        float stored_d = float(xb->d), min_v = float(xb->dmin);
        if (nonfinite_bit(stored_d) || nonfinite_bit(min_v)) { bad |= 1u; stored_d = 0.0f; min_v = 0.0f; }
        const float d = il < 2 ? stored_d : stored_d / 16.0f;
        const float dl = d * float(sc[0]), ml = min_v * float(sc[1]);
        const uchar mask = uchar(il < 2 ? 0x0Fu : 0xF0u);
        const uchar4 q4 = uchar4(*reinterpret_cast<device const packed_uchar4*>(q)) & uchar4(mask);
        res = half4(dl * float4(q4) - ml);
    } else {
        const uint ci = 2u * step + (uint(sub) >> 2u);
        device const block_q5_K* xb = reinterpret_cast<device const block_q5_K*>(row) + ci / 16u;
        short il = short(ci % 16u);
        const uint q_off = (uint(sub) & 3u) * 4u;
        device const uchar* q = xb->qs + 32 * (il / 4) + 16 * (il & 1) + q_off;
        device const uchar* qh = xb->qh + 16 * (il & 1) + q_off;
        const short is = (il / 4) * 2;
        const uchar ul = uchar(1u << (il / 2));
        il = il & 3;
        const uchar2 sc = scale_min_k4(is, il / 2, xb->scales);
        float stored_d = float(xb->d), min_v = float(xb->dmin);
        if (nonfinite_bit(stored_d) || nonfinite_bit(min_v)) { bad |= 1u; stored_d = 0.0f; min_v = 0.0f; }
        const float d = il < 2 ? stored_d : stored_d / 16.0f;
        const float dl = d * float(sc[0]), ml = min_v * float(sc[1]);
        const uchar mask = uchar(il < 2 ? 0x0Fu : 0xF0u);
        const float qh_val = il < 2 ? 16.0f : 256.0f;
        const uchar4 q4 = uchar4(*reinterpret_cast<device const packed_uchar4*>(q)) & uchar4(mask);
        const uchar4 qh4 = uchar4(*reinterpret_cast<device const packed_uchar4*>(qh)) & uchar4(ul);
        const float4 hi_add = select(float4(0.0f), float4(qh_val), qh4 != uchar4(0u));
        res = half4(dl * (float4(q4) + hi_add) - ml);
    }
    bad |= nonfinite_half_bit(res.x) | nonfinite_half_bit(res.y) | nonfinite_half_bit(res.z) | nonfinite_half_bit(res.w);
    return res;
}

template <ushort SM>
static inline void prefill_dense_q8_tile(device const uchar* weights, device bfloat* x_bf16, device float* output,
    constant GroupedMmArgs& a, device atomic_uint* error, uint2 group, ushort tid, ushort sg, threadgroup half* stage) {
    constexpr ushort BN = 64, SN = 32, KS = 64, StageStride = 68;
    const ushort sg_m = sg >> 1u, sg_n = sg & 1u, lane = tid & 31u;
    const uint first_col = group.x * BN;
    const uint tile_row0 = group.y * (2u * uint(SM));
    const uint row_base = tile_row0 + uint(sg_m) * uint(SM);
    if (first_col >= a.rows || tile_row0 >= a.tokens) return;

    const bool owns_rows = row_base < a.tokens;
    device bfloat* sg_input = x_bf16 + ulong(owns_rows ? row_base : 0u) * a.columns;
    auto a_tensor = tensor(sg_input, dextents<int, 2>{int(a.columns), SM}, array<int, 2>{1, int(a.columns)});
    constexpr auto descriptor = matmul2d_descriptor(SM, SN, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt(stage + uint(sg_n) * uint(SN) * StageStride,
        dextents<int, 2>{KS, SN}, array<int, 2>{1, StageStride});
    auto b_tile = bt.slice<KS, SN>(0, 0);
    auto a0 = a_tensor.template slice<KS, SM>(0, 0);
    auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b_tile), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    uint bad = 0;
    const uint steps = a.columns / KS;
    if (nax_layout == 1) {
        const uint blocks = a.columns / 32u;
        const uint sg_col_base = first_col + uint(sg) * 16u;
        const uint sc_col = sg_col_base + uint(lane);
        ushort q_next[16];
        half2 sc_next = half2(0);

        const auto load_step_q8_lanes = [&](uint k_step, thread ushort (&q_dst)[16], thread half2& sc_dst) {
            const uint b0 = 2u * k_step;
            const uint run = b0 / 16u;
            const uint width = min(16u, blocks - run * 16u);
            const ulong quant_base = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(b0 % 16u) + ulong(lane) * width;
            if (lane < 16u && sc_col < a.rows) {
                half2 s2 = *reinterpret_cast<device const half2*>(weights + ulong(sc_col) * a.row_bytes + 2ul * ulong(b0));
                if (nonfinite_half_bit(s2.x)) { bad |= 1u; s2.x = half(0); }
                if (nonfinite_half_bit(s2.y)) { bad |= 1u; s2.y = half(0); }
                sc_dst = s2;
            } else {
                sc_dst = half2(0);
            }
            #pragma unroll
            for (ushort c = 0; c < 16; ++c) {
                const uint w_col = sg_col_base + uint(c);
                q_dst[c] = (w_col < a.rows)
                    ? *reinterpret_cast<device const ushort*>(weights + ulong(w_col) * a.row_bytes + quant_base)
                    : ushort(0);
            }
        };

        load_step_q8_lanes(0u, q_next, sc_next);
        for (uint k_step = 0u; k_step < steps; ++k_step) {
            threadgroup half* sg_stage = stage + (uint(sg) * 16u) * StageStride;
            #pragma unroll
            for (ushort c = 0; c < 16; ++c) {
                const half2 sc2 = simd_broadcast(sc_next, c);
                const ushort qw = q_next[c];
                const half h0 = half(float(as_type<char>(uchar(qw & 0xffu))) * float(sc2.x));
                const half h1 = half(float(as_type<char>(uchar(qw >> 8u))) * float(sc2.y));
                bad |= nonfinite_half_bit(h0) | nonfinite_half_bit(h1);
                sg_stage[uint(c) * StageStride + uint(lane)] = h0;
                sg_stage[uint(c) * StageStride + 32u + uint(lane)] = h1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k_step + 1u < steps) {
                load_step_q8_lanes(k_step + 1u, q_next, sc_next);
            }
            if (owns_rows) {
                auto a_slice = a_tensor.template slice<KS, SM>(int(k_step * KS), 0);
                operation.run(a_slice, b_tile, acc);
            }
            if (k_step + 1u < steps) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    } else {
        const ushort sub = lane & 7u, r_in_pass = tid >> 3u;
        half4 next0[4], next1[4];
        const auto load_step_gguf = [&](uint k_step, thread half4 (&dst0)[4], thread half4 (&dst1)[4]) {
            #pragma unroll
            for (ushort p = 0; p < 4; ++p) {
                const uint c_local = uint(p) * 16u + uint(r_in_pass);
                const bool live = first_col + c_local < a.rows;
                device const uchar* r_ptr = weights + ulong(live ? first_col + c_local : 0u) * a.row_bytes;
                dst0[p] = decode4_gguf(r_ptr, 2u * k_step, sub, live, bad);
                dst1[p] = decode4_gguf(r_ptr, 2u * k_step + 1u, sub, live, bad);
            }
        };
        load_step_gguf(0u, next0, next1);
        for (uint k_step = 0u; k_step < steps; ++k_step) {
            #pragma unroll
            for (ushort p = 0; p < 4; ++p) {
                const uint c_local = uint(p) * 16u + uint(r_in_pass);
                *reinterpret_cast<threadgroup half4*>(stage + c_local * StageStride + uint(sub) * 4u) = next0[p];
                *reinterpret_cast<threadgroup half4*>(stage + c_local * StageStride + 32u + uint(sub) * 4u) = next1[p];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k_step + 1u < steps) {
                load_step_gguf(k_step + 1u, next0, next1);
            }
            if (owns_rows) {
                auto a_slice = a_tensor.template slice<KS, SM>(int(k_step * KS), 0);
                operation.run(a_slice, b_tile, acc);
            }
            if (k_step + 1u < steps) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    if (owns_rows) {
        #pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            const auto idx = acc.get_multidimensional_index(i);
            const uint col_out = first_col + uint(sg_n) * uint(SN) + uint(idx[0]);
            const uint row_out = row_base + uint(idx[1]);
            if (row_out < a.tokens && col_out < a.rows) {
                const float val = float(acc[i]);
                bad |= nonfinite_bit(val);
                device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(row_out) * a.output_stride_bytes);
                y[col_out] = val;
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

union DenseQ8M32Smem {
    alignas(16) half stage[2 * 64 * 64];
    float epi[4][8][65];
};
static_assert(sizeof(DenseQ8M32Smem) == 16384, "DenseQ8M32Smem must remain 16 KiB");

union SwigluTileSmem {
    alignas(16) half stage[2 * 64 * 64];
    float epi[4][8][65];
};
static_assert(sizeof(SwigluTileSmem) == 16384, "SwigluTileSmem must remain 16 KiB");

union Bf16TileSmem {
    alignas(16) bfloat stage[2 * 64 * 64];
    float epi[4][8][65];
};
static_assert(sizeof(Bf16TileSmem) == 16384, "Bf16TileSmem must remain 16 KiB");

static inline void prefill_dense_q8_m256_tile(device const uchar* weights, device bfloat* x_bf16, device float* output,
    constant GroupedMmArgs& a, device atomic_uint* error, uint2 group, ushort tid, ushort sg, threadgroup DenseQ8M32Smem& smem) {
    constexpr ushort SM = 32, BN = 64, SN = 64, KS = 64, StageStride = 64;
    constexpr uint PlaneElems = uint(BN) * uint(StageStride);
    const ushort lane = tid & 31u;
    const uint gx = (a.rows + BN - 1u) / BN;
    const uint gy = (a.tokens + 127u) / 128u;
    if (!gx || !gy || group.x >= gx || group.y >= gy) return;
    const uint linear = group.y * gx + group.x;
    const uint col_tile = linear / gy;
    const uint row_tile = linear % gy;
    const uint first_col = col_tile * BN;
    const uint tile_row0 = row_tile * 128u;
    if (first_col >= a.rows || tile_row0 >= a.tokens) return;

    threadgroup half* stage = smem.stage;
    const uint row_base0 = tile_row0 + uint(sg) * 32u;
    const bool owns_rows0 = row_base0 < a.tokens;
    device bfloat* sg_input0 = x_bf16 + ulong(owns_rows0 ? row_base0 : 0u) * a.columns;
    auto a_tensor0 = tensor(sg_input0, dextents<int, 2>{int(a.columns), SM}, array<int, 2>{1, int(a.columns)});
    constexpr auto descriptor = matmul2d_descriptor(SM, SN, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, SN}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + PlaneElems, dextents<int, 2>{KS, SN}, array<int, 2>{1, StageStride});
    auto b_tile0 = bt0.slice<KS, SN>(0, 0);
    auto b_tile1 = bt1.slice<KS, SN>(0, 0);
    auto a0_init = a_tensor0.template slice<KS, SM>(0, 0);
    auto acc0 = operation.template get_destination_cooperative_tensor<decltype(a0_init), decltype(b_tile0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) acc0[i] = 0.0f;

    uint bad = 0;
    const uint steps = a.columns / KS;
    if (nax_layout == 1) {
        const uint blocks = a.columns / 32u;
        const uint full_runs = blocks / 16u;
        const uint full_quarters = full_runs * 4u;
        const uint full_steps = full_runs * 8u;
        const uint q0 = 0u;
        const ushort sc_c = lane >> 1u, sc_sub = lane & 1u;
        const uint sg_col_base = first_col + uint(sg) * 16u;
        const uint sc_col = sg_col_base + uint(lane);
        const uint sc_run_col = sg_col_base + uint(sc_c);

        const auto run_mma = [&](uint plane_step, uint k_actual) {
            if (!owns_rows0) return;
            auto a_s0 = a_tensor0.template slice<KS, SM>(int(k_actual * KS), 0);
            if ((plane_step & 1u) == 0u) operation.run(a_s0, b_tile0, acc0);
            else operation.run(a_s0, b_tile1, acc0);
        };

        const auto sanitize_qtr_scales = [&](half2 raw_s2, thread half2& sc_dst, thread bool& qtr_clean) {
            bool lane_clean = true;
            if (nonfinite_half_bit(raw_s2.x)) { bad |= 1u; raw_s2.x = half(0); lane_clean = false; }
            if (nonfinite_half_bit(raw_s2.y)) { bad |= 1u; raw_s2.y = half(0); lane_clean = false; }
            if (fmax(fabs(float(raw_s2.x)), fabs(float(raw_s2.y))) * 128.0f > 65504.0f) lane_clean = false;
            sc_dst = raw_s2;
            qtr_clean = simd_all(lane_clean);
        };

        const auto load_qtr_raw = [&](uint q_idx, thread uint (&q_dst)[16], thread half2& raw_sc_dst) {
            raw_sc_dst = (sc_run_col < a.rows)
                ? half2(*reinterpret_cast<device const packed_half2*>(
                    weights + ulong(sc_run_col) * a.row_bytes + ulong(q_idx) * 8ul + ulong(sc_sub) * 4ul))
                : half2(0);
            const uint run = q_idx >> 2u;
            const ulong qtr_off = ulong(q_idx & 3u) * 4ul;
            const ulong quant_off = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(lane) * 16ul + qtr_off;
            #pragma unroll
            for (ushort c = 0; c < 16; ++c) {
                const uint w_col = sg_col_base + uint(c);
                q_dst[c] = (w_col < a.rows)
                    ? as_type<uint>(ushort2(*reinterpret_cast<device const packed_ushort2*>(weights + ulong(w_col) * a.row_bytes + quant_off)))
                    : 0u;
            }
        };

        const auto store_qtr_sub_q8_lanes = [&](thread const uint (&q_src)[16], half2 sc_src, bool qtr_clean, ushort sub, threadgroup half* plane) {
            threadgroup half* sg_stage = plane + (uint(sg) * 16u) * StageStride;
            const bool hi_half = (sub & 1u) != 0u;
            if (qtr_clean) {
                #pragma unroll
                for (ushort c = 0; c < 16; ++c) {
                    const half2 sc2 = simd_broadcast(sc_src, ushort(c * 2u + sub));
                    const ushort qw = ushort(hi_half ? (q_src[c] >> 16u) : (q_src[c] & 0xffffu));
                    sg_stage[uint(c) * StageStride + uint(lane)] = half(as_type<char>(uchar(qw & 0xffu))) * sc2.x;
                    sg_stage[uint(c) * StageStride + 32u + uint(lane)] = half(as_type<char>(uchar(qw >> 8u))) * sc2.y;
                }
            } else {
                #pragma unroll
                for (ushort c = 0; c < 16; ++c) {
                    const half2 sc2 = simd_broadcast(sc_src, ushort(c * 2u + sub));
                    const ushort qw = ushort(hi_half ? (q_src[c] >> 16u) : (q_src[c] & 0xffffu));
                    const half h0_v = half(float(as_type<char>(uchar(qw & 0xffu))) * float(sc2.x));
                    const half h1_v = half(float(as_type<char>(uchar(qw >> 8u))) * float(sc2.y));
                    bad |= nonfinite_half_bit(h0_v) | nonfinite_half_bit(h1_v);
                    sg_stage[uint(c) * StageStride + uint(lane)] = h0_v;
                    sg_stage[uint(c) * StageStride + 32u + uint(lane)] = h1_v;
                }
            }
        };

        const auto load_step_q8_lanes_raw = [&](uint k_step, thread ushort (&q_dst)[16], thread half2& raw_sc_dst) {
            const uint b0 = 2u * k_step;
            const uint run = b0 / 16u;
            const uint width = min(16u, blocks - run * 16u);
            const ulong quant_base = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(b0 % 16u) + ulong(lane) * width;
            raw_sc_dst = (lane < 16u && sc_col < a.rows)
                ? half2(*reinterpret_cast<device const packed_half2*>(weights + ulong(sc_col) * a.row_bytes + 2ul * ulong(b0)))
                : half2(0);
            #pragma unroll
            for (ushort c = 0; c < 16; ++c) {
                const uint w_col = sg_col_base + uint(c);
                q_dst[c] = (w_col < a.rows)
                    ? *reinterpret_cast<device const ushort*>(weights + ulong(w_col) * a.row_bytes + quant_base)
                    : ushort(0);
            }
        };

        const auto store_plane_q8_lanes = [&](thread const ushort (&q_src)[16], half2 raw_sc2, threadgroup half* plane) {
            if (nonfinite_half_bit(raw_sc2.x)) { bad |= 1u; raw_sc2.x = half(0); }
            if (nonfinite_half_bit(raw_sc2.y)) { bad |= 1u; raw_sc2.y = half(0); }
            threadgroup half* sg_stage = plane + (uint(sg) * 16u) * StageStride;
            #pragma unroll
            for (ushort c = 0; c < 16; ++c) {
                const half2 sc2 = simd_broadcast(raw_sc2, c);
                const ushort qw = q_src[c];
                const half h0_v = half(float(as_type<char>(uchar(qw & 0xffu))) * float(sc2.x));
                const half h1_v = half(float(as_type<char>(uchar(qw >> 8u))) * float(sc2.y));
                bad |= nonfinite_half_bit(h0_v) | nonfinite_half_bit(h1_v);
                sg_stage[uint(c) * StageStride + uint(lane)] = h0_v;
                sg_stage[uint(c) * StageStride + 32u + uint(lane)] = h1_v;
            }
        };

        ushort q_next[16];
        half2 raw_sc_next = half2(0);
        if (full_quarters > 0u) {
            uint q_qtr[16];
            half2 raw_sc_qtr = half2(0);
            half2 sc_qtr = half2(0);
            bool qtr_clean = true;
            load_qtr_raw(q0, q_qtr, raw_sc_qtr);
            sanitize_qtr_scales(raw_sc_qtr, sc_qtr, qtr_clean);
            store_qtr_sub_q8_lanes(q_qtr, sc_qtr, qtr_clean, 0u, stage);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint step = 0u; step < full_steps; ++step) {
                const ushort sub = ushort(step & 1u);
                const uint q_cur = ((step >> 1u) + q0) % full_quarters;
                const uint k_actual = q_cur * 2u + uint(sub);
                const uint next_q_iter = (step >> 1u) + 1u;
                threadgroup half* next_plane = stage + ((step + 1u) & 1u) * PlaneElems;
                if (sub == 0u) {
                    store_qtr_sub_q8_lanes(q_qtr, sc_qtr, qtr_clean, 1u, next_plane);
                    if (next_q_iter < full_quarters) {
                        const uint q_nxt = (next_q_iter + q0) % full_quarters;
                        load_qtr_raw(q_nxt, q_qtr, raw_sc_qtr);
                    } else if (full_steps < steps) {
                        load_step_q8_lanes_raw(full_steps, q_next, raw_sc_next);
                    }
                    run_mma(step, k_actual);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                } else {
                    run_mma(step, k_actual);
                    if (next_q_iter < full_quarters) {
                        sanitize_qtr_scales(raw_sc_qtr, sc_qtr, qtr_clean);
                        store_qtr_sub_q8_lanes(q_qtr, sc_qtr, qtr_clean, 0u, next_plane);
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    } else if (full_steps < steps) {
                        store_plane_q8_lanes(q_next, raw_sc_next, next_plane);
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }
            }
        } else if (steps > 0u) {
            load_step_q8_lanes_raw(0u, q_next, raw_sc_next);
            store_plane_q8_lanes(q_next, raw_sc_next, stage);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint k_step = full_steps; k_step < steps; ++k_step) {
            if (k_step + 1u < steps) {
                load_step_q8_lanes_raw(k_step + 1u, q_next, raw_sc_next);
            }
            run_mma(k_step, k_step);
            if (k_step + 1u < steps) {
                store_plane_q8_lanes(q_next, raw_sc_next, stage + ((k_step + 1u) & 1u) * PlaneElems);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    } else {
        const ushort sub = tid & 7u, r_in_pass = tid >> 3u;
        half4 next0[4], next1[4];
        const auto load_step_gguf = [&](uint k_step, thread half4 (&dst0)[4], thread half4 (&dst1)[4]) {
            #pragma unroll
            for (ushort p = 0; p < 4; ++p) {
                const uint c_local = uint(p) * 16u + uint(r_in_pass);
                const bool live = first_col + c_local < a.rows;
                device const uchar* r_ptr = weights + ulong(live ? first_col + c_local : 0u) * a.row_bytes;
                dst0[p] = decode4_gguf(r_ptr, 2u * k_step, sub, live, bad);
                dst1[p] = decode4_gguf(r_ptr, 2u * k_step + 1u, sub, live, bad);
            }
        };
        const auto store_plane_gguf = [&](thread const half4 (&src0)[4], thread const half4 (&src1)[4], threadgroup half* plane) {
            #pragma unroll
            for (ushort p = 0; p < 4; ++p) {
                const uint c_local = uint(p) * 16u + uint(r_in_pass);
                *reinterpret_cast<threadgroup half4*>(plane + c_local * StageStride + uint(sub) * 4u) = src0[p];
                *reinterpret_cast<threadgroup half4*>(plane + c_local * StageStride + 32u + uint(sub) * 4u) = src1[p];
            }
        };

        load_step_gguf(0u, next0, next1);
        store_plane_gguf(next0, next1, stage);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k_step = 0u; k_step < steps; ++k_step) {
            if (k_step + 1u < steps) {
                load_step_gguf(k_step + 1u, next0, next1);
            }
            const int k_off = int(k_step * KS);
            if ((k_step & 1u) == 0u) {
                if (owns_rows0) {
                    auto a_s0 = a_tensor0.template slice<KS, SM>(k_off, 0);
                    operation.run(a_s0, b_tile0, acc0);
                }
                if (k_step + 1u < steps) {
                    store_plane_gguf(next0, next1, stage + PlaneElems);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            } else {
                if (owns_rows0) {
                    auto a_s0 = a_tensor0.template slice<KS, SM>(k_off, 0);
                    operation.run(a_s0, b_tile1, acc0);
                }
                if (k_step + 1u < steps) {
                    store_plane_gguf(next0, next1, stage);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owns_rows0) {
        const uint c0 = first_col + uint(lane);
        const uint c1 = first_col + 32u + uint(lane);
        const bool col0_valid = c0 < a.rows;
        const bool col1_valid = c1 < a.rows;
        for (ushort r_base = 0; r_base < SM; r_base += 8) {
            #pragma unroll
            for (ushort i = 0; i < acc0.get_capacity(); ++i) {
                if (!acc0.is_valid_element(i)) continue;
                const auto idx = acc0.get_multidimensional_index(i);
                const ushort col_idx = ushort(idx[0]), row_idx = ushort(idx[1]);
                if (row_idx >= r_base && row_idx < r_base + 8) {
                    smem.epi[sg][row_idx - r_base][col_idx] = float(acc0[i]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (ushort r_sub = 0; r_sub < 8; ++r_sub) {
                const uint row_out = row_base0 + uint(r_base + r_sub);
                if (row_out < a.tokens) {
                    device float* y = reinterpret_cast<device float*>(
                        reinterpret_cast<device uchar*>(output) + ulong(row_out) * a.output_stride_bytes);
                    if (col0_valid) {
                        const float v0 = smem.epi[sg][r_sub][lane];
                        bad |= nonfinite_bit(v0);
                        y[c0] = v0;
                    }
                    if (col1_valid) {
                        const float v1 = smem.epi[sg][r_sub][32u + lane];
                        bad |= nonfinite_bit(v1);
                        y[c1] = v1;
                    }
                }
            }
            if (r_base + 8 < SM) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

struct Ushort32 {
    ushort4 v[8];
};

static inline Ushort32 load_bf16_half32(device const ushort* row, uint k_step, ushort half_k, bool live_col, thread uint& bad) {
    Ushort32 out;
    if (!live_col) {
        #pragma unroll
        for (ushort i = 0; i < 8; ++i) out.v[i] = ushort4(0);
        return out;
    }
    device const ushort* src = row + ulong(k_step) * 64ul + ulong(half_k) * 32ul;
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) {
        const ushort4 h = ushort4(*reinterpret_cast<device const packed_ushort4*>(src + ulong(i) * 4ul));
        out.v[i] = clean_bf16_bits4(h, bad);
    }
    return out;
}

template <ushort RowsPerSG>
static inline void prefill_dense_bf16_tile(device const uchar* weights, device bfloat* x_bf16, device float* output,
    constant GroupedMmArgs& a, device atomic_uint* error, uint2 group, ushort tid, ushort sg, threadgroup Bf16TileSmem& smem) {
    constexpr ushort Cols = 64, KS = 64, Simdgroups = 4;
    constexpr uint TileM = uint(Simdgroups) * uint(RowsPerSG);
    const uint gx = (a.rows + Cols - 1u) / Cols;
    const uint gy = (a.tokens + TileM - 1u) / TileM;
    if (!gx || !gy || group.x >= gx || group.y >= gy) return;
    const uint linear = group.y * gx + group.x;
    const uint col_tile = linear / gy;
    const uint row_tile = linear % gy;
    const uint first_col = col_tile * Cols;
    const uint row_base = row_tile * TileM + sg * RowsPerSG;
    if (first_col >= a.rows || row_tile * TileM >= a.tokens) return;

    threadgroup bfloat* stage = smem.stage;
    const bool owns_rows = row_base < a.tokens;
    const ushort col = tid >> 1u, half_k = tid & 1u;
    const bool live_col = first_col + col < a.rows;
    device const ushort* w_row = reinterpret_cast<device const ushort*>(weights) + ulong(live_col ? first_col + col : 0u) * a.columns;

    device bfloat* sg_input = x_bf16 + ulong(owns_rows ? row_base : 0u) * a.columns;
    auto a_tensor = tensor(sg_input, dextents<int, 2>{int(a.columns), RowsPerSG}, array<int, 2>{1, int(a.columns)});
    constexpr auto descriptor = matmul2d_descriptor(RowsPerSG, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> bt1(stage + KS * Cols, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
    auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
    auto a0 = a_tensor.template slice<KS, RowsPerSG>(0, 0);
    auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    uint bad = 0u;
    const uint steps = a.columns / KS;
    Ushort32 next_w = load_bf16_half32(w_row, 0u, half_k, live_col, bad);

    for (uint step = 0u; step < steps; ++step) {
        threadgroup ushort4* dst = reinterpret_cast<threadgroup ushort4*>(
            stage + (step & 1u) * (KS * Cols) + col * KS + half_k * 32u);
        #pragma unroll
        for (ushort i = 0; i < 8; ++i) dst[i] = next_w.v[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (step + 1u < steps) {
            next_w = load_bf16_half32(w_row, step + 1u, half_k, live_col, bad);
        }
        if (owns_rows) {
            auto a_slice = a_tensor.template slice<KS, RowsPerSG>(int(step * KS), 0);
            if (step & 1u) operation.run(a_slice, b1, acc);
            else operation.run(a_slice, b0, acc);
        }
    }

    if (owns_rows) {
        #pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            const auto idx = acc.get_multidimensional_index(i);
            const uint col_out = first_col + uint(idx[0]);
            const uint row_out = row_base + uint(idx[1]);
            if (row_out < a.tokens && col_out < a.rows) {
                const float val = float(acc[i]);
                bad |= nonfinite_bit(val);
                device float* y = reinterpret_cast<device float*>(
                    reinterpret_cast<device uchar*>(output) + ulong(row_out) * a.output_stride_bytes);
                y[col_out] = val;
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

template <ushort RowsPerSG>
static inline void prefill_dense_q8_swiglu_tile(device const uchar* gate_weights, device const uchar* up_weights,
    device bfloat* x_bf16, device float* output, constant GroupedMmArgs& a, device atomic_uint* error,
    uint2 group, ushort tid, ushort sg, threadgroup SwigluTileSmem& smem) {
    constexpr ushort Cols = 64, HalfCols = 32, KS = 64, Simdgroups = 4;
    constexpr uint TileM = uint(Simdgroups) * uint(RowsPerSG);
    const uint gx = (a.rows + HalfCols - 1u) / HalfCols;
    const uint gy = (a.tokens + TileM - 1u) / TileM;
    if (!gx || !gy || group.x >= gx || group.y >= gy) return;
    const uint linear = group.y * gx + group.x;
    const uint col_tile = linear / gy;
    const uint row_tile = linear % gy;
    const uint first_col = col_tile * HalfCols;
    const uint row_base = row_tile * TileM + sg * RowsPerSG;
    if (first_col >= a.rows || row_tile * TileM >= a.tokens) return;

    const bool owns_rows = row_base < a.tokens;
    const ushort col = tid >> 1, blk = tid & 1;
    const bool is_up = col >= HalfCols;
    const uint local_col = is_up ? (uint(col) - HalfCols) : uint(col);
    const uint out_col = first_col + local_col;
    const bool live_col = out_col < a.rows;
    device const uchar* w_row = (is_up ? up_weights : gate_weights) + ulong(live_col ? out_col : 0u) * a.row_bytes;

    device bfloat* sg_input = x_bf16 + ulong(owns_rows ? row_base : 0u) * a.columns;
    auto a_tensor = tensor(sg_input, dextents<int, 2>{int(a.columns), RowsPerSG}, array<int, 2>{1, int(a.columns)});
    constexpr auto descriptor = matmul2d_descriptor(RowsPerSG, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(smem.stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(smem.stage + KS * Cols, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
    auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
    auto a0 = a_tensor.template slice<KS, RowsPerSG>(0, 0);
    auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    uint bad = 0;
    const uint steps = a.columns / KS;
    if (nax_layout == 1 && RowsPerSG == 32) {
        constexpr uint PlaneElems = uint(KS) * uint(Cols);
        const uint blocks = a.columns / 32u;
        const uint full_runs = blocks / 16u;
        const uint full_quarters = full_runs * 4u;
        const uint full_steps = full_runs * 8u;
        const uint q0 = 0u;
        const ushort lane = tid & 31u;
        const ushort sc_c = lane >> 1u, sc_sub = lane & 1u;
        device const uchar* sg_w_base = (sg >= 2u) ? up_weights : gate_weights;
        const uint sg_col_base = first_col + uint(sg & 1u) * 16u;
        const uint sc_col = sg_col_base + uint(lane);
        const uint sc_run_col = sg_col_base + uint(sc_c);

        const auto run_mma = [&](uint plane_step, uint k_actual) {
            if (!owns_rows) return;
            auto a_slice = a_tensor.template slice<KS, RowsPerSG>(int(k_actual * KS), 0);
            if ((plane_step & 1u) == 0u) operation.run(a_slice, b0, acc);
            else operation.run(a_slice, b1, acc);
        };

        const auto sanitize_qtr_scales = [&](half2 raw_s2, thread half2& sc_dst, thread bool& qtr_clean) {
            bool lane_clean = true;
            if (nonfinite_half_bit(raw_s2.x)) { bad |= 1u; raw_s2.x = half(0); lane_clean = false; }
            if (nonfinite_half_bit(raw_s2.y)) { bad |= 1u; raw_s2.y = half(0); lane_clean = false; }
            if (fmax(fabs(float(raw_s2.x)), fabs(float(raw_s2.y))) * 128.0f > 65504.0f) lane_clean = false;
            sc_dst = raw_s2;
            qtr_clean = simd_all(lane_clean);
        };

        const auto load_qtr_raw = [&](uint q_idx, thread uint (&q_dst)[16], thread half2& raw_sc_dst) {
            raw_sc_dst = (sc_run_col < a.rows)
                ? half2(*reinterpret_cast<device const packed_half2*>(
                    sg_w_base + ulong(sc_run_col) * a.row_bytes + ulong(q_idx) * 8ul + ulong(sc_sub) * 4ul))
                : half2(0);
            const uint run = q_idx >> 2u;
            const ulong qtr_off = ulong(q_idx & 3u) * 4ul;
            const ulong quant_off = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(lane) * 16ul + qtr_off;
            #pragma unroll
            for (ushort c = 0; c < 16; ++c) {
                const uint w_col = sg_col_base + uint(c);
                q_dst[c] = (w_col < a.rows)
                    ? as_type<uint>(ushort2(*reinterpret_cast<device const packed_ushort2*>(sg_w_base + ulong(w_col) * a.row_bytes + quant_off)))
                    : 0u;
            }
        };

        const auto store_qtr_sub_q8_lanes = [&](thread const uint (&q_src)[16], half2 sc_src, bool qtr_clean, ushort sub, threadgroup half* plane) {
            threadgroup half* sg_stage = plane + (uint(sg) * 16u) * KS;
            const bool hi_half = (sub & 1u) != 0u;
            if (qtr_clean) {
                #pragma unroll
                for (ushort c = 0; c < 16; ++c) {
                    const half2 sc2 = simd_broadcast(sc_src, ushort(c * 2u + sub));
                    const ushort qw = ushort(hi_half ? (q_src[c] >> 16u) : (q_src[c] & 0xffffu));
                    sg_stage[uint(c) * KS + uint(lane)] = half(as_type<char>(uchar(qw & 0xffu))) * sc2.x;
                    sg_stage[uint(c) * KS + 32u + uint(lane)] = half(as_type<char>(uchar(qw >> 8u))) * sc2.y;
                }
            } else {
                #pragma unroll
                for (ushort c = 0; c < 16; ++c) {
                    const half2 sc2 = simd_broadcast(sc_src, ushort(c * 2u + sub));
                    const ushort qw = ushort(hi_half ? (q_src[c] >> 16u) : (q_src[c] & 0xffffu));
                    const half h0_v = half(float(as_type<char>(uchar(qw & 0xffu))) * float(sc2.x));
                    const half h1_v = half(float(as_type<char>(uchar(qw >> 8u))) * float(sc2.y));
                    bad |= nonfinite_half_bit(h0_v) | nonfinite_half_bit(h1_v);
                    sg_stage[uint(c) * KS + uint(lane)] = h0_v;
                    sg_stage[uint(c) * KS + 32u + uint(lane)] = h1_v;
                }
            }
        };

        const auto load_step_q8_lanes_raw = [&](uint k_step, thread ushort (&q_dst)[16], thread half2& raw_sc_dst) {
            const uint b0_idx = 2u * k_step;
            const uint run = b0_idx / 16u;
            const uint width = min(16u, blocks - run * 16u);
            const ulong quant_base = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(b0_idx % 16u) + ulong(lane) * width;
            raw_sc_dst = (lane < 16u && sc_col < a.rows)
                ? half2(*reinterpret_cast<device const packed_half2*>(sg_w_base + ulong(sc_col) * a.row_bytes + 2ul * ulong(b0_idx)))
                : half2(0);
            #pragma unroll
            for (ushort c = 0; c < 16; ++c) {
                const uint w_col = sg_col_base + uint(c);
                q_dst[c] = (w_col < a.rows)
                    ? *reinterpret_cast<device const ushort*>(sg_w_base + ulong(w_col) * a.row_bytes + quant_base)
                    : ushort(0);
            }
        };

        const auto store_plane_q8_lanes = [&](thread const ushort (&q_src)[16], half2 raw_sc, threadgroup half* plane) {
            if (nonfinite_half_bit(raw_sc.x)) { bad |= 1u; raw_sc.x = half(0); }
            if (nonfinite_half_bit(raw_sc.y)) { bad |= 1u; raw_sc.y = half(0); }
            threadgroup half* sg_stage = plane + (uint(sg) * 16u) * KS;
            #pragma unroll
            for (ushort c = 0; c < 16; ++c) {
                const half2 sc2 = simd_broadcast(raw_sc, c);
                const ushort qw = q_src[c];
                const half h0_v = half(float(as_type<char>(uchar(qw & 0xffu))) * float(sc2.x));
                const half h1_v = half(float(as_type<char>(uchar(qw >> 8u))) * float(sc2.y));
                bad |= nonfinite_half_bit(h0_v) | nonfinite_half_bit(h1_v);
                sg_stage[uint(c) * KS + uint(lane)] = h0_v;
                sg_stage[uint(c) * KS + 32u + uint(lane)] = h1_v;
            }
        };

        ushort q_next[16];
        half2 raw_sc_next = half2(0);
        if (full_quarters > 0u) {
            uint q_qtr[16];
            half2 raw_sc_qtr = half2(0);
            half2 sc_qtr = half2(0);
            bool qtr_clean = true;
            load_qtr_raw(q0, q_qtr, raw_sc_qtr);
            sanitize_qtr_scales(raw_sc_qtr, sc_qtr, qtr_clean);
            store_qtr_sub_q8_lanes(q_qtr, sc_qtr, qtr_clean, 0u, smem.stage);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint step = 0u; step < full_steps; ++step) {
                const ushort sub = ushort(step & 1u);
                const uint q_cur = ((step >> 1u) + q0) % full_quarters;
                const uint k_actual = q_cur * 2u + uint(sub);
                const uint next_q_iter = (step >> 1u) + 1u;
                threadgroup half* next_plane = smem.stage + ((step + 1u) & 1u) * PlaneElems;
                if (sub == 0u) {
                    store_qtr_sub_q8_lanes(q_qtr, sc_qtr, qtr_clean, 1u, next_plane);
                    if (next_q_iter < full_quarters) {
                        const uint q_nxt = (next_q_iter + q0) % full_quarters;
                        load_qtr_raw(q_nxt, q_qtr, raw_sc_qtr);
                    } else if (full_steps < steps) {
                        load_step_q8_lanes_raw(full_steps, q_next, raw_sc_next);
                    }
                    run_mma(step, k_actual);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                } else {
                    run_mma(step, k_actual);
                    if (next_q_iter < full_quarters) {
                        sanitize_qtr_scales(raw_sc_qtr, sc_qtr, qtr_clean);
                        store_qtr_sub_q8_lanes(q_qtr, sc_qtr, qtr_clean, 0u, next_plane);
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    } else if (full_steps < steps) {
                        store_plane_q8_lanes(q_next, raw_sc_next, next_plane);
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }
            }
        } else if (steps > 0u) {
            load_step_q8_lanes_raw(0u, q_next, raw_sc_next);
            store_plane_q8_lanes(q_next, raw_sc_next, smem.stage);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint k_step = full_steps; k_step < steps; ++k_step) {
            if (k_step + 1u < steps) {
                load_step_q8_lanes_raw(k_step + 1u, q_next, raw_sc_next);
            }
            run_mma(k_step, k_step);
            if (k_step + 1u < steps) {
                store_plane_q8_lanes(q_next, raw_sc_next, smem.stage + ((k_step + 1u) & 1u) * PlaneElems);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    } else {
        Half32 next_w = decode_q8_pair64(w_row, 0u, blk, a.columns, live_col, bad);

        for (uint step = 0; step < steps; ++step) {
            threadgroup half* stage_col = smem.stage + (step & 1u) * (KS * Cols) + col * KS;
            if (nax_layout == 0) {
                threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(stage_col + blk * 32u);
                #pragma unroll
                for (ushort i = 0; i < 8; ++i) dst[i] = next_w.v[i];
            } else {
                threadgroup half4* dst0 = reinterpret_cast<threadgroup half4*>(stage_col + blk * 16u);
                threadgroup half4* dst1 = reinterpret_cast<threadgroup half4*>(stage_col + 32u + blk * 16u);
                #pragma unroll
                for (ushort i = 0; i < 4; ++i) {
                    dst0[i] = next_w.v[i];
                    dst1[i] = next_w.v[i + 4];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (step + 1u < steps) {
                next_w = decode_q8_pair64(w_row, step + 1u, blk, a.columns, live_col, bad);
            }
            if (owns_rows) {
                auto a_slice = a_tensor.template slice<KS, RowsPerSG>(int(step * KS), 0);
                if (step & 1u) operation.run(a_slice, b1, acc);
                else operation.run(a_slice, b0, acc);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owns_rows) {
        const ushort lane = tid & 31u;
        const uint col_out = first_col + uint(lane);
        const bool col_valid = col_out < a.rows;
        for (ushort r_base = 0; r_base < RowsPerSG; r_base += 8) {
            #pragma unroll
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto idx = acc.get_multidimensional_index(i);
                const ushort col_idx = ushort(idx[0]), row_idx = ushort(idx[1]);
                if (row_idx >= r_base && row_idx < r_base + 8) {
                    smem.epi[sg][row_idx - r_base][col_idx] = float(acc[i]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (col_valid) {
                #pragma unroll
                for (ushort r_sub = 0; r_sub < 8; ++r_sub) {
                    const uint row_out = row_base + uint(r_base + r_sub);
                    if (row_out < a.tokens) {
                        const float g = smem.epi[sg][r_sub][lane];
                        const float u = smem.epi[sg][r_sub][32u + lane];
                        bad |= nonfinite_bit(g) | nonfinite_bit(u);
                        const float act = (g * sigmoid_f32(g)) * u;
                        bad |= nonfinite_bit(act);
                        device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(row_out) * a.output_stride_bytes);
                        y[col_out] = act;
                    }
                }
            }
            if (r_base + 8 < RowsPerSG) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

struct RepackQ8Tile64Args {
    uint columns;
    uint rows;
    uint row_bytes;
    uint src_layout;
};

struct Q8Tile64Ipt1 {
    uint4 q0;
    uint4 q1;
    ushort raw_sc;
};

static inline Q8Tile64Ipt1 load_q8_t64_ipt1(device const uchar* plane0, device const ushort* meta,
    uint G, uint tile_n, uint g, uint col_in_tile) {
    const ulong tile_idx = ulong(tile_n) * ulong(G) + ulong(g);
    device const uint4* pq = reinterpret_cast<device const uint4*>(
        plane0 + tile_idx * 2048ul + ulong(col_in_tile) * 32ul);
    Q8Tile64Ipt1 out;
    out.q0 = pq[0];
    out.q1 = pq[1];
    out.raw_sc = meta[tile_idx * 64ul + ulong(col_in_tile)];
    return out;
}

static inline void store_q8_t64_ipt1(thread const Q8Tile64Ipt1& w, threadgroup half* dst_32, thread uint& bad) {
    threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(dst_32);
    const ushort abs_sc = w.raw_sc & 0x7fffu;
    if (abs_sc >= 0x0400u && abs_sc < 0x6000u) {
        const half sc = as_type<half>(w.raw_sc);
        dst[0] = half4(as_type<char4>(w.q0.x)) * sc;
        dst[1] = half4(as_type<char4>(w.q0.y)) * sc;
        dst[2] = half4(as_type<char4>(w.q0.z)) * sc;
        dst[3] = half4(as_type<char4>(w.q0.w)) * sc;
        dst[4] = half4(as_type<char4>(w.q1.x)) * sc;
        dst[5] = half4(as_type<char4>(w.q1.y)) * sc;
        dst[6] = half4(as_type<char4>(w.q1.z)) * sc;
        dst[7] = half4(as_type<char4>(w.q1.w)) * sc;
    } else if (abs_sc == 0u) {
        #pragma unroll
        for (ushort i = 0; i < 8; ++i) dst[i] = half4(0);
    } else {
        half sc_h = as_type<half>(w.raw_sc);
        if (nonfinite_half_bit(sc_h)) {
            bad |= 1u;
            #pragma unroll
            for (ushort i = 0; i < 8; ++i) dst[i] = half4(0);
            return;
        }
        const float sc_f = float(sc_h);
        const uint qw[8] = {w.q0.x, w.q0.y, w.q0.z, w.q0.w, w.q1.x, w.q1.y, w.q1.z, w.q1.w};
        #pragma unroll
        for (ushort i = 0; i < 8; ++i) {
            const char4 c4 = as_type<char4>(qw[i]);
            const half4 hv = half4(float4(c4) * sc_f);
            bad |= nonfinite_half_bit(hv.x) | nonfinite_half_bit(hv.y) | nonfinite_half_bit(hv.z) | nonfinite_half_bit(hv.w);
            dst[i] = hv;
        }
    }
}

static inline void prefill_dense_q8_t64_m32_tile(device const uchar* plane0, device const ushort* meta,
    device bfloat* x_bf16, device float* output, constant GroupedMmArgs& a, device atomic_uint* error,
    uint2 group, ushort tid, ushort sg, threadgroup half* stage) {
    constexpr ushort SM = 32, BN = 64, SN = 64, KS = 64, StageStride = 64;
    constexpr uint PlaneElems = uint(BN) * uint(StageStride);
    const uint gx = (a.rows + BN - 1u) / BN;
    const uint gy = (a.tokens + 127u) / 128u;
    if (!gx || !gy || group.x >= gx || group.y >= gy) return;
    const uint linear = group.y * gx + group.x;
    const uint col_tile = linear / gy;
    const uint row_tile = linear % gy;
    const uint first_col = col_tile * BN;
    const uint tile_row0 = row_tile * 128u;
    if (first_col >= a.rows || tile_row0 >= a.tokens) return;

    const uint row_base0 = tile_row0 + uint(sg) * 32u;
    const bool owns_rows0 = row_base0 < a.tokens;
    device bfloat* sg_input0 = x_bf16 + ulong(owns_rows0 ? row_base0 : 0u) * a.columns;
    auto a_tensor0 = tensor(sg_input0, dextents<int, 2>{int(a.columns), SM}, array<int, 2>{1, int(a.columns)});
    constexpr auto descriptor = matmul2d_descriptor(SM, SN, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, SN}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + (TOKSPUR_PRIVATE_DENSE_M32_SINGLE_TILE ? 0u : PlaneElems), dextents<int, 2>{KS, SN}, array<int, 2>{1, StageStride});
    auto b_tile0 = bt0.slice<KS, SN>(0, 0);
    auto b_tile1 = bt1.slice<KS, SN>(0, 0);
    auto a0_init = a_tensor0.template slice<KS, SM>(0, 0);
    auto acc0 = operation.template get_destination_cooperative_tensor<decltype(a0_init), decltype(b_tile0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) acc0[i] = 0.0f;

    uint bad = 0u;
    const uint G = a.columns / 32u;
    const uint steps = a.columns / KS;
    const ushort col = tid >> 1u;
    const ushort gi = tid & 1u;
    const uint stage_off = uint(col) * StageStride + uint(gi) * 32u;

    if (steps > 0u) {
        Q8Tile64Ipt1 w_next = load_q8_t64_ipt1(plane0, meta, G, col_tile, uint(gi), uint(col));
        for (uint step = 0u; step < steps; ++step) {
            const uint plane = TOKSPUR_PRIVATE_DENSE_M32_SINGLE_TILE ? 0u : (step & 1u);
            store_q8_t64_ipt1(w_next, stage + plane * PlaneElems + stage_off, bad);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (step + 1u < steps) {
                w_next = load_q8_t64_ipt1(plane0, meta, G, col_tile, 2u * (step + 1u) + uint(gi), uint(col));
            }
            if (owns_rows0) {
                auto a_s0 = a_tensor0.template slice<KS, SM>(int(step * KS), 0);
                if (plane == 0u) operation.run(a_s0, b_tile0, acc0);
                else operation.run(a_s0, b_tile1, acc0);
            }
#if TOKSPUR_PRIVATE_DENSE_M32_SINGLE_TILE
            // Every SIMD group must finish reading the shared tile before its next overwrite.
            if (step + 1u < steps) threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
        }
    }

    if (owns_rows0) {
        #pragma unroll
        for (ushort i = 0; i < acc0.get_capacity(); ++i) {
            if (!acc0.is_valid_element(i)) continue;
            const auto idx = acc0.get_multidimensional_index(i);
            const uint col_out = first_col + uint(idx[0]);
            const uint row_out = row_base0 + uint(idx[1]);
            if (row_out < a.tokens && col_out < a.rows) {
                const float val = float(acc0[i]);
                bad |= nonfinite_bit(val);
                device float* y = reinterpret_cast<device float*>(
                    reinterpret_cast<device uchar*>(output) + ulong(row_out) * a.output_stride_bytes);
                y[col_out] = val;
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

static inline void prefill_dense_q8_t64_m8_tile(device const uchar* plane0, device const ushort* meta,
    device bfloat* x_bf16, device float* output, constant GroupedMmArgs& a, device atomic_uint* error,
    uint2 group, ushort tid, ushort sg, threadgroup half* stage) {
    constexpr ushort SM = 16, BN = 64, SN = 32, KS = 64, StageStride = 64;
    constexpr uint PlaneElems = uint(BN) * uint(StageStride);
    const ushort sg_m = sg >> 1u, sg_n = sg & 1u;
    const uint gx = (a.rows + BN - 1u) / BN;
    const uint gy = (a.tokens + 31u) / 32u;
    if (!gx || !gy || group.x >= gx || group.y >= gy) return;
    const uint linear = group.y * gx + group.x;
    const uint col_tile = linear / gy;
    const uint row_tile = linear % gy;
    const uint first_col = col_tile * BN;
    const uint tile_row0 = row_tile * 32u;
    const uint row_base = tile_row0 + uint(sg_m) * uint(SM);
    if (first_col >= a.rows || tile_row0 >= a.tokens) return;

    const bool owns_rows = row_base < a.tokens;
    device bfloat* sg_input = x_bf16 + ulong(owns_rows ? row_base : 0u) * a.columns;
    auto a_tensor = tensor(sg_input, dextents<int, 2>{int(a.columns), SM}, array<int, 2>{1, int(a.columns)});
    constexpr auto descriptor = matmul2d_descriptor(SM, SN, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    const uint sg_n_off = uint(sg_n) * uint(SN) * StageStride;
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage + sg_n_off, dextents<int, 2>{KS, SN}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + PlaneElems + sg_n_off, dextents<int, 2>{KS, SN}, array<int, 2>{1, StageStride});
    auto b_tile0 = bt0.slice<KS, SN>(0, 0);
    auto b_tile1 = bt1.slice<KS, SN>(0, 0);
    auto a0 = a_tensor.template slice<KS, SM>(0, 0);
    auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b_tile0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    uint bad = 0u;
    const uint G = a.columns / 32u;
    const uint steps = a.columns / KS;
    const ushort col = tid >> 1u;
    const ushort gi = tid & 1u;
    const uint stage_off = uint(col) * StageStride + uint(gi) * 32u;

    if (steps > 0u) {
        Q8Tile64Ipt1 w_next = load_q8_t64_ipt1(plane0, meta, G, col_tile, uint(gi), uint(col));
        for (uint step = 0u; step < steps; ++step) {
            store_q8_t64_ipt1(w_next, stage + (step & 1u) * PlaneElems + stage_off, bad);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (step + 1u < steps) {
                w_next = load_q8_t64_ipt1(plane0, meta, G, col_tile, 2u * (step + 1u) + uint(gi), uint(col));
            }
            if (owns_rows) {
                auto a_slice = a_tensor.template slice<KS, SM>(int(step * KS), 0);
                if ((step & 1u) == 0u) operation.run(a_slice, b_tile0, acc);
                else operation.run(a_slice, b_tile1, acc);
            }
        }
    }

    if (owns_rows) {
        #pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            const auto idx = acc.get_multidimensional_index(i);
            const uint col_out = first_col + uint(sg_n) * uint(SN) + uint(idx[0]);
            const uint row_out = row_base + uint(idx[1]);
            if (row_out < a.tokens && col_out < a.rows) {
                const float val = float(acc[i]);
                bad |= nonfinite_bit(val);
                device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(row_out) * a.output_stride_bytes);
                y[col_out] = val;
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

template <ushort RowsPerSG>
static inline void prefill_dense_q8_swiglu_t64_tile(device const uchar* gate_plane0, device const ushort* gate_meta,
    device const uchar* up_plane0, device const ushort* up_meta,
    device bfloat* x_bf16, device float* output, constant GroupedMmArgs& a, device atomic_uint* error,
    uint2 group, ushort tid, ushort sg, threadgroup SwigluTileSmem& smem) {
    constexpr ushort Cols = 64, HalfCols = 32, KS = 64, Simdgroups = 4;
    constexpr uint PlaneElems = uint(KS) * uint(Cols);
    constexpr uint TileM = uint(Simdgroups) * uint(RowsPerSG);
    const uint gx = (a.rows + HalfCols - 1u) / HalfCols;
    const uint gy = (a.tokens + TileM - 1u) / TileM;
    if (!gx || !gy || group.x >= gx || group.y >= gy) return;
    const uint linear = group.y * gx + group.x;
    const uint col_tile = linear / gy;
    const uint row_tile = linear % gy;
    const uint first_col = col_tile * HalfCols;
    const uint row_base = row_tile * TileM + sg * RowsPerSG;
    if (first_col >= a.rows || row_tile * TileM >= a.tokens) return;

    const bool owns_rows = row_base < a.tokens;
    const ushort lane = tid & 31u;
    device bfloat* sg_input = x_bf16 + ulong(owns_rows ? row_base : 0u) * a.columns;
    auto a_tensor = tensor(sg_input, dextents<int, 2>{int(a.columns), RowsPerSG}, array<int, 2>{1, int(a.columns)});
    constexpr auto descriptor = matmul2d_descriptor(RowsPerSG, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(smem.stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(smem.stage + PlaneElems, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
    auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
    auto a0 = a_tensor.template slice<KS, RowsPerSG>(0, 0);
    auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    uint bad = 0u;
    const uint G = a.columns / 32u;
    const uint steps = a.columns / KS;
    const uint t64_tile = col_tile >> 1u;
    const uint t64_col_base = (col_tile & 1u) * 32u;
    const ushort col = tid >> 1u;
    const ushort gi = tid & 1u;
    const bool is_up = col >= 32u;
    const uint src_col = t64_col_base + uint(col & 31u);
    device const uchar* sg_p0 = is_up ? up_plane0 : gate_plane0;
    device const ushort* sg_meta = is_up ? up_meta : gate_meta;
    const uint stage_off = uint(col) * KS + uint(gi) * 32u;

    if (steps > 0u) {
        Q8Tile64Ipt1 w_next = load_q8_t64_ipt1(sg_p0, sg_meta, G, t64_tile, uint(gi), src_col);
        for (uint step = 0u; step < steps; ++step) {
            store_q8_t64_ipt1(w_next, smem.stage + (step & 1u) * PlaneElems + stage_off, bad);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (step + 1u < steps) {
                w_next = load_q8_t64_ipt1(sg_p0, sg_meta, G, t64_tile, 2u * (step + 1u) + uint(gi), src_col);
            }
            if (owns_rows) {
                auto a_slice = a_tensor.template slice<KS, RowsPerSG>(int(step * KS), 0);
                if ((step & 1u) == 0u) operation.run(a_slice, b0, acc);
                else operation.run(a_slice, b1, acc);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owns_rows) {
        const uint col_out = first_col + uint(lane);
        const bool col_valid = col_out < a.rows;
        for (ushort r_base = 0; r_base < RowsPerSG; r_base += 8) {
            #pragma unroll
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto idx = acc.get_multidimensional_index(i);
                const ushort col_idx = ushort(idx[0]), row_idx = ushort(idx[1]);
                if (row_idx >= r_base && row_idx < r_base + 8) {
                    smem.epi[sg][row_idx - r_base][col_idx] = float(acc[i]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (col_valid) {
                #pragma unroll
                for (ushort r_sub = 0; r_sub < 8; ++r_sub) {
                    const uint row_out = row_base + uint(r_base + r_sub);
                    if (row_out < a.tokens) {
                        const float g = smem.epi[sg][r_sub][lane];
                        const float u = smem.epi[sg][r_sub][32u + lane];
                        bad |= nonfinite_bit(g) | nonfinite_bit(u);
                        const float act = (g * sigmoid_f32(g)) * u;
                        bad |= nonfinite_bit(act);
                        device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(row_out) * a.output_stride_bytes);
                        y[col_out] = act;
                    }
                }
            }
            if (r_base + 8 < RowsPerSG) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

} // namespace nax_detail

kernel void nax_f32_to_bf16(device const float* input [[buffer(0)]], device bfloat* output [[buffer(1)]],
    constant nax_detail::NaxConvertBf16Args& p [[buffer(2)]], device atomic_uint* error [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
    if (!p.columns || p.columns % 4 != 0 || !p.rows || p.padded_rows < p.rows || p.input_stride_bytes % 4 != 0 || p.input_stride_bytes < ulong(p.columns) * 4ul) {
        if (!idx) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint quads_per_row = p.columns / 4;
    const uint total_quads = p.padded_rows * quads_per_row;
    if (idx >= total_quads) return;
    const uint row = idx / quads_per_row, col4 = idx % quads_per_row;
    device bfloat4* dst = reinterpret_cast<device bfloat4*>(output + ulong(row) * p.columns + col4 * 4);
    if (row >= p.rows) {
        *dst = bfloat4(0);
        return;
    }
    const ulong stride_floats = p.input_stride_bytes / sizeof(float);
    const float4 src = float4(*reinterpret_cast<device const packed_float4*>(input + ulong(row) * stride_floats + col4 * 4));
    uint bad = 0;
    const bfloat b0 = nax_detail::clean_bf16(src.x, bad);
    const bfloat b1 = nax_detail::clean_bf16(src.y, bad);
    const bfloat b2 = nax_detail::clean_bf16(src.z, bad);
    const bfloat b3 = nax_detail::clean_bf16(src.w, bad);
    *dst = bfloat4(b0, b1, b2, b3);
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

kernel void nax_prefill_dense_q8_m32(device const uchar* w [[buffer(0)]], device bfloat* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup nax_detail::DenseQ8M32Smem smem;
    nax_detail::prefill_dense_q8_m256_tile(w, x, y, a, error, group, tid, sg, smem);
}

kernel void nax_prefill_dense_q8_m8(device const uchar* w [[buffer(0)]], device bfloat* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    alignas(16) threadgroup half stage[64 * 68];
    nax_detail::prefill_dense_q8_tile<16>(w, x, y, a, error, group, tid, sg, stage);
}

kernel void nax_prefill_dense_q8_swiglu_m32(device const uchar* wg [[buffer(0)]], device const uchar* wu [[buffer(1)]],
    device bfloat* x [[buffer(2)]], device float* y [[buffer(3)]], constant nax_detail::GroupedMmArgs& a [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup nax_detail::SwigluTileSmem smem;
    nax_detail::prefill_dense_q8_swiglu_tile<32>(wg, wu, x, y, a, error, group, tid, sg, smem);
}

kernel void nax_prefill_dense_q8_swiglu_m8(device const uchar* wg [[buffer(0)]], device const uchar* wu [[buffer(1)]],
    device bfloat* x [[buffer(2)]], device float* y [[buffer(3)]], constant nax_detail::GroupedMmArgs& a [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup nax_detail::SwigluTileSmem smem;
    nax_detail::prefill_dense_q8_swiglu_tile<8>(wg, wu, x, y, a, error, group, tid, sg, smem);
}

kernel void nax_repack_q8_tile64(device const uchar* src [[buffer(0)]], device uchar* plane0 [[buffer(1)]],
    device ushort* meta [[buffer(2)]], constant nax_detail::RepackQ8Tile64Args& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    if (!a.columns || a.columns % 32u != 0u || !a.rows || a.row_bytes != (a.columns / 32u) * 34u || a.src_layout > 1u) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint G = a.columns / 32u;
    const uint NT = (a.rows + 63u) / 64u;
    const uint g = group.x;
    const uint t = group.y;
    const uint c = uint(tid);
    if (g >= G || t >= NT || c >= 64u) return;

    const uint row = t * 64u + c;
    device uchar* dst_q = plane0 + ((ulong(t) * ulong(G) + ulong(g)) * 64ul + ulong(c)) * 32ul;
    device ushort* dst_m = meta + (ulong(t) * ulong(G) + ulong(g)) * 64ul + ulong(c);
    if (row >= a.rows) {
        device uint4* dq = reinterpret_cast<device uint4*>(dst_q);
        dq[0] = uint4(0u);
        dq[1] = uint4(0u);
        dst_m[0] = ushort(0);
        return;
    }

    device const uchar* w_row = src + ulong(row) * a.row_bytes;
    if (a.src_layout == 0u) {
        device const uchar* blk = w_row + ulong(g) * 34ul;
        dst_m[0] = *reinterpret_cast<device const ushort*>(blk);
        device const packed_ushort4* sq = reinterpret_cast<device const packed_ushort4*>(blk + 2ul);
        device ushort4* dq = reinterpret_cast<device ushort4*>(dst_q);
        #pragma unroll
        for (ushort i = 0; i < 4; ++i) dq[i] = ushort4(sq[i]);
    } else {
        dst_m[0] = reinterpret_cast<device const ushort*>(w_row)[g];
        const uint run = g / 16u;
        const uint width = min(16u, G - run * 16u);
        device const uchar* q_base = w_row + 2ul * ulong(G) + ulong(run) * 512ul + ulong(g % 16u);
        uchar4 qv[8];
        #pragma unroll
        for (ushort i = 0; i < 8; ++i) {
            const uint k0 = uint(i) * 4u;
            qv[i] = uchar4(q_base[(k0 + 0u) * width], q_base[(k0 + 1u) * width],
                           q_base[(k0 + 2u) * width], q_base[(k0 + 3u) * width]);
        }
        device uchar4* dq = reinterpret_cast<device uchar4*>(dst_q);
        #pragma unroll
        for (ushort i = 0; i < 8; ++i) dq[i] = qv[i];
    }
}

kernel void nax_prefill_dense_q8_t64_m32(device const uchar* plane0 [[buffer(0)]], device const ushort* meta [[buffer(1)]],
    device bfloat* x [[buffer(2)]], device float* y [[buffer(3)]], constant nax_detail::GroupedMmArgs& a [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (!a.columns || a.columns % 64u != 0u || !a.rows || !a.tokens) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    alignas(16) threadgroup half stage[(TOKSPUR_PRIVATE_DENSE_M32_SINGLE_TILE ? 1 : 2) * 64 * 64];
    nax_detail::prefill_dense_q8_t64_m32_tile(plane0, meta, x, y, a, error, group, tid, sg, stage);
}

kernel void nax_prefill_dense_q8_t64_m8(device const uchar* plane0 [[buffer(0)]], device const ushort* meta [[buffer(1)]],
    device bfloat* x [[buffer(2)]], device float* y [[buffer(3)]], constant nax_detail::GroupedMmArgs& a [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (!a.columns || a.columns % 64u != 0u || !a.rows || !a.tokens) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    alignas(16) threadgroup half stage[2 * 64 * 64];
    nax_detail::prefill_dense_q8_t64_m8_tile(plane0, meta, x, y, a, error, group, tid, sg, stage);
}

kernel void nax_prefill_dense_q8_swiglu_t64_m32(device const uchar* gp0 [[buffer(0)]], device const ushort* gm [[buffer(1)]],
    device const uchar* up0 [[buffer(2)]], device const ushort* um [[buffer(3)]],
    device bfloat* x [[buffer(4)]], device float* y [[buffer(5)]], constant nax_detail::GroupedMmArgs& a [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (!a.columns || a.columns % 64u != 0u || !a.rows || !a.tokens) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    threadgroup nax_detail::SwigluTileSmem smem;
    nax_detail::prefill_dense_q8_swiglu_t64_tile<32>(gp0, gm, up0, um, x, y, a, error, group, tid, sg, smem);
}

kernel void nax_prefill_dense_q8_swiglu_t64_m8(device const uchar* gp0 [[buffer(0)]], device const ushort* gm [[buffer(1)]],
    device const uchar* up0 [[buffer(2)]], device const ushort* um [[buffer(3)]],
    device bfloat* x [[buffer(4)]], device float* y [[buffer(5)]], constant nax_detail::GroupedMmArgs& a [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (!a.columns || a.columns % 64u != 0u || !a.rows || !a.tokens) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    threadgroup nax_detail::SwigluTileSmem smem;
    nax_detail::prefill_dense_q8_swiglu_t64_tile<8>(gp0, gm, up0, um, x, y, a, error, group, tid, sg, smem);
}

kernel void nax_prefill_dense_bf16_m32(device const uchar* w [[buffer(0)]], device bfloat* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (!a.columns || a.columns % 64u != 0u || a.row_bytes != a.columns * 2u || !a.rows || !a.tokens) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    threadgroup nax_detail::Bf16TileSmem smem;
    nax_detail::prefill_dense_bf16_tile<32>(w, x, y, a, error, group, tid, sg, smem);
}

kernel void nax_prefill_dense_bf16_m8(device const uchar* w [[buffer(0)]], device bfloat* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (!a.columns || a.columns % 64u != 0u || a.row_bytes != a.columns * 2u || !a.rows || !a.tokens) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    threadgroup nax_detail::Bf16TileSmem smem;
    nax_detail::prefill_dense_bf16_tile<8>(w, x, y, a, error, group, tid, sg, smem);
}

namespace nax_verify_detail {

template <uint T>
static inline void stage_verify_x256(device const float* const (&in_rows)[T], uint c_base, uint chunk_cols,
    ushort tid, threadgroup float (*sx)[256], thread uint& bad) {
    const uint quads = chunk_cols >> 2u;
    for (uint idx = uint(tid); idx < T * quads; idx += 128u) {
        const uint t = idx / quads;
        const uint q = idx % quads;
        const float4 v = float4(*reinterpret_cast<device const packed_float4*>(in_rows[t] + c_base + q * 4u));
        bad |= nax_detail::nonfinite_bit(v.x) | nax_detail::nonfinite_bit(v.y) |
               nax_detail::nonfinite_bit(v.z) | nax_detail::nonfinite_bit(v.w);
        *reinterpret_cast<threadgroup float4*>(&sx[t][q * 4u]) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template<uint T, uint NR>
static inline void verify_dense_q8_rows(device const uchar* weights, device const float* input, device float* output,
    constant nax_detail::GroupedMmArgs& a, device atomic_uint* error, uint group_x, ushort sg, ushort lane) {
    const uint row0 = (group_x * 4u + uint(sg)) * NR;
    if (row0 >= a.rows) return;
    const uint active_rows = min(NR, a.rows - row0);
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const ulong out_stride = a.output_stride_bytes / sizeof(float);
    device const float* in_rows[T];
    #pragma unroll
    for (uint t = 0u; t < T; ++t) in_rows[t] = input + ulong(t) * in_stride;

    float sums[NR][T];
    #pragma unroll
    for (uint r = 0u; r < NR; ++r) {
        #pragma unroll
        for (uint t = 0u; t < T; ++t) sums[r][t] = 0.0f;
    }
    uint bad = 0u;
    const uint blocks = a.columns / 32u;
    const uint full = blocks / 16u;

    for (uint b0 = 0u; b0 < blocks; b0 += 16u) {
        const uint count = min(16u, blocks - b0);
        if (nax_layout == 0) {
            device const uchar* base = weights + ulong(row0) * a.row_bytes + ulong(b0) * 34ul;
            for (uint k = 0u; k < count; ++k) {
                float xv[T];
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float v = in_rows[t][(b0 + k) * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(v);
                    xv[t] = v;
                }
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    if (r < active_rows) {
                        device const uchar* enc = base + ulong(r) * a.row_bytes + ulong(k) * 34ul;
                        float sc = nax_detail::stored_half(enc);
                        if (nax_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                        const float w = sc * float(as_type<char>(enc[2u + lane]));
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) sums[r][t] += w * xv[t];
                    }
                }
            }
        } else if (count == 16u) {
            const uint run = b0 / 16u;
            uint4 words[NR];
            float sc_lane[NR];
            #pragma unroll
            for (uint r = 0u; r < NR; ++r) {
                if (r < active_rows) {
                    device const uchar* row_ptr = weights + ulong(row0 + r) * a.row_bytes;
                    device const half* scales = reinterpret_cast<device const half*>(row_ptr);
                    float sc = (lane < 16u) ? float(scales[b0 + uint(lane)]) : 0.0f;
                    if (nax_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                    sc_lane[r] = sc;
                    device const uchar* quants = row_ptr + ulong(blocks) * 2ul;
                    words[r] = uint4(*reinterpret_cast<device const packed_uint4*>(quants + ulong(run) * 512ul + ulong(lane) * 16ul));
                } else {
                    sc_lane[r] = 0.0f;
                    words[r] = uint4(0u);
                }
            }
            #pragma unroll
            for (uint part = 0u; part < 4u; ++part) {
                #pragma unroll
                for (uint sub = 0u; sub < 4u; ++sub) {
                    const uint k = part * 4u + sub;
                    float xv[T];
                    #pragma unroll
                    for (uint t = 0u; t < T; ++t) {
                        const float v = in_rows[t][(b0 + k) * 32u + uint(lane)];
                        bad |= nax_detail::nonfinite_bit(v);
                        xv[t] = v;
                    }
                    #pragma unroll
                    for (uint r = 0u; r < NR; ++r) {
                        if (r < active_rows) {
                            const float sc = simd_broadcast(sc_lane[r], ushort(k));
                            const float w = sc * float(as_type<char>(uchar((words[r][part] >> (8u * sub)) & 255u)));
                            #pragma unroll
                            for (uint t = 0u; t < T; ++t) sums[r][t] += w * xv[t];
                        }
                    }
                }
            }
        } else if (count % 4u == 0u) {
            for (uint k0 = 0u; k0 < count; k0 += 4u) {
                uchar4 w4_r[NR];
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    if (r < active_rows) {
                        device const uchar* tail = weights + ulong(row0 + r) * a.row_bytes + ulong(blocks) * 2ul + ulong(full) * 512ul + ulong(lane) * count;
                        w4_r[r] = uchar4(*reinterpret_cast<device const packed_uchar4*>(tail + k0));
                    } else {
                        w4_r[r] = uchar4(0u);
                    }
                }
                #pragma unroll
                for (uint i = 0u; i < 4u; ++i) {
                    const uint k = k0 + i;
                    float xv[T];
                    #pragma unroll
                    for (uint t = 0u; t < T; ++t) {
                        const float v = in_rows[t][(b0 + k) * 32u + uint(lane)];
                        bad |= nax_detail::nonfinite_bit(v);
                        xv[t] = v;
                    }
                    #pragma unroll
                    for (uint r = 0u; r < NR; ++r) {
                        if (r < active_rows) {
                            device const half* scales = reinterpret_cast<device const half*>(weights + ulong(row0 + r) * a.row_bytes);
                            float sc = float(scales[b0 + k]);
                            if (nax_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                            const float w = sc * float(as_type<char>(w4_r[r][i]));
                            #pragma unroll
                            for (uint t = 0u; t < T; ++t) sums[r][t] += w * xv[t];
                        }
                    }
                }
            }
        } else {
            for (uint k = 0u; k < count; ++k) {
                float xv[T];
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float v = in_rows[t][(b0 + k) * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(v);
                    xv[t] = v;
                }
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    if (r < active_rows) {
                        device const uchar* row_ptr = weights + ulong(row0 + r) * a.row_bytes;
                        device const half* scales = reinterpret_cast<device const half*>(row_ptr);
                        device const uchar* tail = row_ptr + ulong(blocks) * 2ul + ulong(full) * 512ul + ulong(lane) * count;
                        float sc = float(scales[b0 + k]);
                        if (nax_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                        const float w = sc * float(as_type<char>(tail[k]));
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) sums[r][t] += w * xv[t];
                    }
                }
            }
        }
    }

    const uint mode = a.maximum_tasks;
    const float silu_scale = as_type<float>(a.experts);
    #pragma unroll
    for (uint r = 0u; r < NR; ++r) {
        if (r < active_rows) {
            #pragma unroll
            for (uint t = 0u; t < T; ++t) {
                float tot = simd_sum(sums[r][t]);
                if (!lane) {
                    device float* out_t = output + ulong(t) * out_stride;
                    bad |= nax_detail::nonfinite_bit(tot);
                    if (mode == 1u) {
                        const float g0 = out_t[row0 + r];
                        bad |= nax_detail::nonfinite_bit(g0);
                        tot = (g0 * nax_detail::sigmoid_f32(g0)) * tot;
                        bad |= nax_detail::nonfinite_bit(tot);
                    } else if (mode == 2u) {
                        const float x = tot * silu_scale;
                        bad |= nax_detail::nonfinite_bit(silu_scale) | nax_detail::nonfinite_bit(x);
                        tot = x * nax_detail::sigmoid_f32(x);
                        bad |= nax_detail::nonfinite_bit(tot);
                    }
                    out_t[row0 + r] = tot;
                }
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

template<uint T>
static inline void dispatch_verify_dense_q8(device const uchar* weights, device const float* input, device float* output,
    constant nax_detail::GroupedMmArgs& a, device atomic_uint* error, uint group_x, ushort sg, ushort lane) {
    if (a.rows <= 1024u) {
        verify_dense_q8_rows<T, 1u>(weights, input, output, a, error, group_x, sg, lane);
    } else if (T > 4u || a.rows <= 4096u) {
        verify_dense_q8_rows<T, 2u>(weights, input, output, a, error, group_x, sg, lane);
    } else {
        verify_dense_q8_rows<T, 4u>(weights, input, output, a, error, group_x, sg, lane);
    }
}

template<uint T>
static inline void verify_dense_q8_swiglu_rows(device const uchar* gate_weights, device const uchar* up_weights,
    device const float* input, device float* output, constant nax_detail::GroupedMmArgs& a,
    device atomic_uint* error, uint group_x, ushort sg, ushort lane) {
    const uint row = group_x * 4u + uint(sg);
    if (row >= a.rows) return;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const ulong out_stride = a.output_stride_bytes / sizeof(float);
    device const float* in_rows[T];
    #pragma unroll
    for (uint t = 0u; t < T; ++t) in_rows[t] = input + ulong(t) * in_stride;

    float sum_g[T], sum_u[T];
    #pragma unroll
    for (uint t = 0u; t < T; ++t) {
        sum_g[t] = 0.0f;
        sum_u[t] = 0.0f;
    }
    uint bad = 0u;
    const uint blocks = a.columns / 32u;
    const uint full = blocks / 16u;
    device const uchar* g_row = gate_weights + ulong(row) * a.row_bytes;
    device const uchar* u_row = up_weights + ulong(row) * a.row_bytes;

    for (uint b0 = 0u; b0 < blocks; b0 += 16u) {
        const uint count = min(16u, blocks - b0);
        if (nax_layout == 0) {
            device const uchar* g_base = g_row + ulong(b0) * 34ul;
            device const uchar* u_base = u_row + ulong(b0) * 34ul;
            for (uint k = 0u; k < count; ++k) {
                float xv[T];
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float v = in_rows[t][(b0 + k) * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(v);
                    xv[t] = v;
                }
                device const uchar* g_enc = g_base + ulong(k) * 34ul;
                device const uchar* u_enc = u_base + ulong(k) * 34ul;
                float g_sc = nax_detail::stored_half(g_enc);
                float u_sc = nax_detail::stored_half(u_enc);
                if (nax_detail::nonfinite_bit(g_sc)) { bad |= 1u; g_sc = 0.0f; }
                if (nax_detail::nonfinite_bit(u_sc)) { bad |= 1u; u_sc = 0.0f; }
                const float gw = g_sc * float(as_type<char>(g_enc[2u + lane]));
                const float uw = u_sc * float(as_type<char>(u_enc[2u + lane]));
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    sum_g[t] += gw * xv[t];
                    sum_u[t] += uw * xv[t];
                }
            }
        } else if (count == 16u) {
            const uint run = b0 / 16u;
            device const half* g_scales = reinterpret_cast<device const half*>(g_row);
            device const half* u_scales = reinterpret_cast<device const half*>(u_row);
            float g_sc_lane = (lane < 16u) ? float(g_scales[b0 + uint(lane)]) : 0.0f;
            float u_sc_lane = (lane < 16u) ? float(u_scales[b0 + uint(lane)]) : 0.0f;
            if (nax_detail::nonfinite_bit(g_sc_lane)) { bad |= 1u; g_sc_lane = 0.0f; }
            if (nax_detail::nonfinite_bit(u_sc_lane)) { bad |= 1u; u_sc_lane = 0.0f; }
            const uint4 g_words = uint4(*reinterpret_cast<device const packed_uint4*>(
                g_row + ulong(blocks) * 2ul + ulong(run) * 512ul + ulong(lane) * 16ul));
            const uint4 u_words = uint4(*reinterpret_cast<device const packed_uint4*>(
                u_row + ulong(blocks) * 2ul + ulong(run) * 512ul + ulong(lane) * 16ul));
            #pragma unroll
            for (uint part = 0u; part < 4u; ++part) {
                #pragma unroll
                for (uint sub = 0u; sub < 4u; ++sub) {
                    const uint k = part * 4u + sub;
                    float xv[T];
                    #pragma unroll
                    for (uint t = 0u; t < T; ++t) {
                        const float v = in_rows[t][(b0 + k) * 32u + uint(lane)];
                        bad |= nax_detail::nonfinite_bit(v);
                        xv[t] = v;
                    }
                    const float g_sc = simd_broadcast(g_sc_lane, ushort(k));
                    const float u_sc = simd_broadcast(u_sc_lane, ushort(k));
                    const float gw = g_sc * float(as_type<char>(uchar((g_words[part] >> (8u * sub)) & 255u)));
                    const float uw = u_sc * float(as_type<char>(uchar((u_words[part] >> (8u * sub)) & 255u)));
                    #pragma unroll
                    for (uint t = 0u; t < T; ++t) {
                        sum_g[t] += gw * xv[t];
                        sum_u[t] += uw * xv[t];
                    }
                }
            }
        } else {
            device const half* g_scales = reinterpret_cast<device const half*>(g_row);
            device const half* u_scales = reinterpret_cast<device const half*>(u_row);
            device const uchar* g_tail = g_row + ulong(blocks) * 2ul + ulong(full) * 512ul + ulong(lane) * count;
            device const uchar* u_tail = u_row + ulong(blocks) * 2ul + ulong(full) * 512ul + ulong(lane) * count;
            for (uint k = 0u; k < count; ++k) {
                float xv[T];
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float v = in_rows[t][(b0 + k) * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(v);
                    xv[t] = v;
                }
                float g_sc = float(g_scales[b0 + k]);
                float u_sc = float(u_scales[b0 + k]);
                if (nax_detail::nonfinite_bit(g_sc)) { bad |= 1u; g_sc = 0.0f; }
                if (nax_detail::nonfinite_bit(u_sc)) { bad |= 1u; u_sc = 0.0f; }
                const float gw = g_sc * float(as_type<char>(g_tail[k]));
                const float uw = u_sc * float(as_type<char>(u_tail[k]));
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    sum_g[t] += gw * xv[t];
                    sum_u[t] += uw * xv[t];
                }
            }
        }
    }

    #pragma unroll
    for (uint t = 0u; t < T; ++t) {
        const float g0 = simd_sum(sum_g[t]);
        const float u0 = simd_sum(sum_u[t]);
        if (!lane) {
            device float* out_t = output + ulong(t) * out_stride;
            bad |= nax_detail::nonfinite_bit(g0) | nax_detail::nonfinite_bit(u0);
            const float tot = (g0 * nax_detail::sigmoid_f32(g0)) * u0;
            bad |= nax_detail::nonfinite_bit(tot);
            out_t[row] = tot;
        }
    }
    bad = simd_or(bad);
    if (!lane && bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

template<uint T, uint MCap, uint RowsPerSG>
static inline void verify_dense_bf16_rows_staged(device const ushort* weights, device const float* input, device float* output,
    constant nax_detail::GroupedMmArgs& a, device atomic_uint* error, uint group_x, ushort sg, ushort lane, ushort tid,
    threadgroup float (&sx)[MCap][256]) {
    const uint row0 = (group_x * 8u + uint(sg)) * RowsPerSG;
    const uint active_rows = (row0 < a.rows) ? min(RowsPerSG, a.rows - row0) : 0u;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const ulong out_stride = a.output_stride_bytes / sizeof(float);
    float sums[RowsPerSG][T];
    #pragma unroll
    for (uint r = 0u; r < RowsPerSG; ++r) {
        #pragma unroll
        for (uint t = 0u; t < T; ++t) sums[r][t] = 0.0f;
    }
    uint bad = 0u;
    for (uint b_col = 0u; b_col < a.columns; b_col += 256u) {
        const uint elems = min(256u, a.columns - b_col);
        #pragma unroll
        for (uint t = 0u; t < T; ++t) {
            float xv = 0.0f;
            if (uint(tid) < elems) {
                xv = input[ulong(t) * in_stride + b_col + uint(tid)];
                if (nax_detail::nonfinite_bit(xv)) { bad |= 1u; xv = 0.0f; }
            }
            sx[t][tid] = xv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active_rows > 0u) {
            const uint c0 = uint(lane) * 4u;
            const uint c1 = 128u + c0;
            #pragma unroll
            for (uint r = 0u; r < RowsPerSG; ++r) {
                if (r < active_rows) {
                    device const ushort* row_ptr = weights + ulong(row0 + r) * a.columns + b_col;
                    if (c0 < elems) {
                        const ushort4 h0 = nax_detail::clean_bf16_bits4(
                            ushort4(*reinterpret_cast<device const packed_ushort4*>(row_ptr + c0)), bad);
                        const float4 w0 = float4(as_type<bfloat4>(h0));
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) {
                            const float4 x0 = *reinterpret_cast<threadgroup const float4*>(&sx[t][c0]);
                            sums[r][t] += w0.x * x0.x + w0.y * x0.y + w0.z * x0.z + w0.w * x0.w;
                        }
                    }
                    if (c1 < elems) {
                        const ushort4 h1 = nax_detail::clean_bf16_bits4(
                            ushort4(*reinterpret_cast<device const packed_ushort4*>(row_ptr + c1)), bad);
                        const float4 w1 = float4(as_type<bfloat4>(h1));
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) {
                            const float4 x1 = *reinterpret_cast<threadgroup const float4*>(&sx[t][c1]);
                            sums[r][t] += w1.x * x1.x + w1.y * x1.y + w1.z * x1.z + w1.w * x1.w;
                        }
                    }
                }
            }
        }
        if (b_col + 256u < a.columns) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    #pragma unroll
    for (uint r = 0u; r < RowsPerSG; ++r) {
        if (r < active_rows) {
            #pragma unroll
            for (uint t = 0u; t < T; ++t) {
                const float tot = simd_sum(sums[r][t]);
                if (!lane) {
                    bad |= nax_detail::nonfinite_bit(tot);
                    output[ulong(t) * out_stride + row0 + r] = tot;
                }
            }
        }
    }
    bad = simd_or(bad);
    if (!lane && bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

template<uint T, uint NR, uint SGsPerTG>
static inline void verify_bf16_rows(device const ushort* weights, device const float* input, device float* output,
    constant nax_detail::GroupedMmArgs& a, device atomic_uint* error, uint group_x, ushort tid, ushort sg, ushort lane,
    threadgroup float (*sx)[256]) {
    if (group_x * (SGsPerTG * NR) >= a.rows) return;
    const uint row0 = (group_x * SGsPerTG + uint(sg)) * NR;
    const bool sg_active = (uint(sg) < SGsPerTG) && (row0 < a.rows);
    const bool full_rows = sg_active && (row0 + NR <= a.rows);
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const ulong out_stride = a.output_stride_bytes / sizeof(float);
    device const float* in_rows[T];
    #pragma unroll
    for (uint t = 0u; t < T; ++t) in_rows[t] = input + ulong(t) * in_stride;

    float sums[NR][T];
    #pragma unroll
    for (uint r = 0u; r < NR; ++r) {
        #pragma unroll
        for (uint t = 0u; t < T; ++t) sums[r][t] = 0.0f;
    }
    uint bad = 0u;
    for (uint c_chunk = 0u; c_chunk < a.columns; c_chunk += 256u) {
        const uint chunk_cols = min(256u, a.columns - c_chunk);
        stage_verify_x256<T>(in_rows, c_chunk, chunk_cols, tid, sx, bad);
        if (sg_active) {
            const uint vec_chunk = chunk_cols & ~127u;
            for (uint c_off = 0u; c_off < vec_chunk; c_off += 128u) {
                const uint s_off = c_off + uint(lane) * 4u;
                const uint col = c_chunk + s_off;
                float4 xv[T];
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    xv[t] = *reinterpret_cast<threadgroup const float4*>(&sx[t][s_off]);
                }
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    if (full_rows || row0 + r < a.rows) {
                        ushort4 wb = ushort4(*reinterpret_cast<device const packed_ushort4*>(weights + ulong(row0 + r) * a.columns + col));
                        #pragma unroll
                        for (uint j = 0u; j < 4u; ++j) {
                            if ((wb[j] & 0x7f80u) == 0x7f80u) { bad |= 1u; wb[j] = 0u; }
                        }
                        const float4 wf(as_type<float>(uint(wb.x) << 16), as_type<float>(uint(wb.y) << 16),
                                        as_type<float>(uint(wb.z) << 16), as_type<float>(uint(wb.w) << 16));
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) sums[r][t] += dot(wf, xv[t]);
                    }
                }
            }
            for (uint c_off = vec_chunk + uint(lane); c_off < chunk_cols; c_off += 32u) {
                const uint col = c_chunk + c_off;
                float xv[T];
                #pragma unroll
                for (uint t = 0u; t < T; ++t) xv[t] = sx[t][c_off];
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    if (full_rows || row0 + r < a.rows) {
                        ushort wb = weights[ulong(row0 + r) * a.columns + col];
                        if ((wb & 0x7f80u) == 0x7f80u) { bad |= 1u; wb = 0u; }
                        const float wv = as_type<float>(uint(wb) << 16);
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) sums[r][t] += wv * xv[t];
                    }
                }
            }
        }
        if (c_chunk + 256u < a.columns) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg_active) {
        #pragma unroll
        for (uint r = 0u; r < NR; ++r) {
            const uint row = row0 + r;
            if (row >= a.rows) continue;
            #pragma unroll
            for (uint t = 0u; t < T; ++t) {
                const float tot = simd_sum(sums[r][t]);
                if (!lane) {
                    bad |= nax_detail::nonfinite_bit(tot);
                    output[ulong(t) * out_stride + row] = tot;
                }
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

template<uint NR, uint SGsPerTG>
static inline void dispatch_verify_bf16(device const ushort* weights, device const float* input, device float* output,
    constant nax_detail::GroupedMmArgs& a, device atomic_uint* error, uint group_x, ushort tid, ushort sg, ushort lane,
    threadgroup float (*sx)[256]) {
    switch (a.tokens) {
        case 1u: verify_bf16_rows<1, NR, SGsPerTG>(weights, input, output, a, error, group_x, tid, sg, lane, sx); break;
        case 2u: verify_bf16_rows<2, NR, SGsPerTG>(weights, input, output, a, error, group_x, tid, sg, lane, sx); break;
        case 3u: verify_bf16_rows<3, NR, SGsPerTG>(weights, input, output, a, error, group_x, tid, sg, lane, sx); break;
        case 4u: verify_bf16_rows<4, NR, SGsPerTG>(weights, input, output, a, error, group_x, tid, sg, lane, sx); break;
        case 5u: verify_bf16_rows<5, NR, SGsPerTG>(weights, input, output, a, error, group_x, tid, sg, lane, sx); break;
        case 6u: verify_bf16_rows<6, NR, SGsPerTG>(weights, input, output, a, error, group_x, tid, sg, lane, sx); break;
        case 7u: verify_bf16_rows<7, NR, SGsPerTG>(weights, input, output, a, error, group_x, tid, sg, lane, sx); break;
        default: verify_bf16_rows<8, NR, SGsPerTG>(weights, input, output, a, error, group_x, tid, sg, lane, sx); break;
    }
}

static inline void write_moe_row(float val, device float* out, uint row, uint swiglu, device atomic_uint* error, ushort lane) {
    if (!lane) {
        uint bad = nax_detail::nonfinite_bit(val);
        if (swiglu) {
            const float g = out[row];
            bad |= nax_detail::nonfinite_bit(g);
            val = (g * nax_detail::sigmoid_f32(g)) * val;
            bad |= nax_detail::nonfinite_bit(val);
        }
        if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
        out[row] = val;
    }
}

} // namespace nax_verify_detail

kernel void nax_verify_dense_bf16_r1_m4(device const ushort* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (!a.tokens || a.tokens > 4u || !a.columns || a.columns % 32u != 0u || a.row_bytes != a.columns * 2u || !a.rows) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y > 0u) return;
    threadgroup float sx[4][256];
    switch (a.tokens) {
        case 1u: nax_verify_detail::verify_dense_bf16_rows_staged<1, 4, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 2u: nax_verify_detail::verify_dense_bf16_rows_staged<2, 4, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 3u: nax_verify_detail::verify_dense_bf16_rows_staged<3, 4, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        default: nax_verify_detail::verify_dense_bf16_rows_staged<4, 4, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
    }
}

kernel void nax_verify_dense_bf16_r1_m8(device const ushort* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (!a.tokens || a.tokens > 8u || !a.columns || a.columns % 32u != 0u || a.row_bytes != a.columns * 2u || !a.rows) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y > 0u) return;
    threadgroup float sx[8][256];
    switch (a.tokens) {
        case 1u: nax_verify_detail::verify_dense_bf16_rows_staged<1, 8, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 2u: nax_verify_detail::verify_dense_bf16_rows_staged<2, 8, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 3u: nax_verify_detail::verify_dense_bf16_rows_staged<3, 8, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 4u: nax_verify_detail::verify_dense_bf16_rows_staged<4, 8, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 5u: nax_verify_detail::verify_dense_bf16_rows_staged<5, 8, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 6u: nax_verify_detail::verify_dense_bf16_rows_staged<6, 8, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 7u: nax_verify_detail::verify_dense_bf16_rows_staged<7, 8, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        default: nax_verify_detail::verify_dense_bf16_rows_staged<8, 8, 1>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
    }
}

kernel void nax_verify_dense_bf16_m4(device const ushort* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (!a.tokens || a.tokens > 4u || !a.columns || a.columns % 32u != 0u || a.row_bytes != a.columns * 2u || !a.rows) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y > 0u) return;
    threadgroup float sx[4][256];
    switch (a.tokens) {
        case 1u: nax_verify_detail::verify_dense_bf16_rows_staged<1, 4, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 2u: nax_verify_detail::verify_dense_bf16_rows_staged<2, 4, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 3u: nax_verify_detail::verify_dense_bf16_rows_staged<3, 4, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        default: nax_verify_detail::verify_dense_bf16_rows_staged<4, 4, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
    }
}

kernel void nax_verify_dense_bf16_m8(device const ushort* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (!a.tokens || a.tokens > 8u || !a.columns || a.columns % 32u != 0u || a.row_bytes != a.columns * 2u || !a.rows) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y > 0u) return;
    threadgroup float sx[8][256];
    switch (a.tokens) {
        case 1u: nax_verify_detail::verify_dense_bf16_rows_staged<1, 8, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 2u: nax_verify_detail::verify_dense_bf16_rows_staged<2, 8, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 3u: nax_verify_detail::verify_dense_bf16_rows_staged<3, 8, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 4u: nax_verify_detail::verify_dense_bf16_rows_staged<4, 8, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 5u: nax_verify_detail::verify_dense_bf16_rows_staged<5, 8, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 6u: nax_verify_detail::verify_dense_bf16_rows_staged<6, 8, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        case 7u: nax_verify_detail::verify_dense_bf16_rows_staged<7, 8, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
        default: nax_verify_detail::verify_dense_bf16_rows_staged<8, 8, 4>(weights, input, output, a, error, group.x, sg, lane, tid, sx); break;
    }
}

kernel void nax_verify_dense_q8_m4(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (!a.tokens || a.tokens > 4u || !a.rows || !a.columns || a.columns % 32u != 0u || a.row_bytes != (a.columns / 32u) * 34u ||
        a.maximum_tasks > 2u || (nax_layout == 1u && a.columns >= 512u && a.columns % 256u != 0u)) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y > 0u) return;
    switch (a.tokens) {
        case 1u: nax_verify_detail::dispatch_verify_dense_q8<1>(weights, input, output, a, error, group.x, sg, lane); break;
        case 2u: nax_verify_detail::dispatch_verify_dense_q8<2>(weights, input, output, a, error, group.x, sg, lane); break;
        case 3u: nax_verify_detail::dispatch_verify_dense_q8<3>(weights, input, output, a, error, group.x, sg, lane); break;
        default: nax_verify_detail::dispatch_verify_dense_q8<4>(weights, input, output, a, error, group.x, sg, lane); break;
    }
}

kernel void nax_verify_dense_q8_m8(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (!a.tokens || a.tokens > 8u || !a.rows || !a.columns || a.columns % 32u != 0u || a.row_bytes != (a.columns / 32u) * 34u ||
        a.maximum_tasks > 2u || (nax_layout == 1u && a.columns >= 512u && a.columns % 256u != 0u)) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y > 0u) return;
    switch (a.tokens) {
        case 1u: nax_verify_detail::dispatch_verify_dense_q8<1>(weights, input, output, a, error, group.x, sg, lane); break;
        case 2u: nax_verify_detail::dispatch_verify_dense_q8<2>(weights, input, output, a, error, group.x, sg, lane); break;
        case 3u: nax_verify_detail::dispatch_verify_dense_q8<3>(weights, input, output, a, error, group.x, sg, lane); break;
        case 4u: nax_verify_detail::dispatch_verify_dense_q8<4>(weights, input, output, a, error, group.x, sg, lane); break;
        case 5u: nax_verify_detail::dispatch_verify_dense_q8<5>(weights, input, output, a, error, group.x, sg, lane); break;
        case 6u: nax_verify_detail::dispatch_verify_dense_q8<6>(weights, input, output, a, error, group.x, sg, lane); break;
        case 7u: nax_verify_detail::dispatch_verify_dense_q8<7>(weights, input, output, a, error, group.x, sg, lane); break;
        default: nax_verify_detail::dispatch_verify_dense_q8<8>(weights, input, output, a, error, group.x, sg, lane); break;
    }
}

kernel void nax_verify_dense_q8_swiglu_m4(device const uchar* gate_weights [[buffer(0)]],
    device const uchar* up_weights [[buffer(1)]], device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]], constant nax_detail::GroupedMmArgs& a [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (!a.tokens || a.tokens > 4u || !a.rows || !a.columns || a.columns % 32u != 0u || a.row_bytes != (a.columns / 32u) * 34u ||
        (nax_layout == 1u && a.columns >= 512u && a.columns % 256u != 0u)) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y > 0u) return;
    switch (a.tokens) {
        case 1u: nax_verify_detail::verify_dense_q8_swiglu_rows<1>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
        case 2u: nax_verify_detail::verify_dense_q8_swiglu_rows<2>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
        case 3u: nax_verify_detail::verify_dense_q8_swiglu_rows<3>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
        default: nax_verify_detail::verify_dense_q8_swiglu_rows<4>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
    }
}

kernel void nax_verify_dense_q8_swiglu_m8(device const uchar* gate_weights [[buffer(0)]],
    device const uchar* up_weights [[buffer(1)]], device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]], constant nax_detail::GroupedMmArgs& a [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (!a.tokens || a.tokens > 8u || !a.rows || !a.columns || a.columns % 32u != 0u || a.row_bytes != (a.columns / 32u) * 34u ||
        (nax_layout == 1u && a.columns >= 512u && a.columns % 256u != 0u)) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y > 0u) return;
    switch (a.tokens) {
        case 1u: nax_verify_detail::verify_dense_q8_swiglu_rows<1>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
        case 2u: nax_verify_detail::verify_dense_q8_swiglu_rows<2>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
        case 3u: nax_verify_detail::verify_dense_q8_swiglu_rows<3>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
        case 4u: nax_verify_detail::verify_dense_q8_swiglu_rows<4>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
        case 5u: nax_verify_detail::verify_dense_q8_swiglu_rows<5>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
        case 6u: nax_verify_detail::verify_dense_q8_swiglu_rows<6>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
        case 7u: nax_verify_detail::verify_dense_q8_swiglu_rows<7>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
        default: nax_verify_detail::verify_dense_q8_swiglu_rows<8>(gate_weights, up_weights, input, output, a, error, group.x, sg, lane); break;
    }
}

kernel void nax_verify_bf16_m8(device const ushort* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup float sx[8][256];
    if (!a.tokens || a.tokens > 8u || !a.columns || a.columns % 32u != 0u || !a.rows || a.row_bytes != a.columns * 2u) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y > 0u) return;
    if (a.rows <= 512u) {
        nax_verify_detail::dispatch_verify_bf16<1, 4>(weights, input, output, a, error, group.x, tid, sg, lane, sx);
    } else {
        nax_verify_detail::dispatch_verify_bf16<2, 4>(weights, input, output, a, error, group.x, tid, sg, lane, sx);
    }
}

// Fused multi-core FP32 alpha+beta projection for GDN verify (one 128-thread group per output row across 2 * heads groups).
kernel void nax_verify_gdn_gates_f32(device const float* alpha_w [[buffer(0)]], device const float* beta_w [[buffer(1)]],
    device const float* input [[buffer(2)]], device float* alpha_out [[buffer(3)]], device float* beta_out [[buffer(4)]],
    constant uint4& shape [[buffer(5)]], device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    // shape = {columns, heads, steps, 0}
    threadgroup float sg_sum[4][8];
    const uint cols = shape.x, heads = shape.y, steps = shape.z;
    if (!cols || !heads || !steps || steps > 8u) {
        if (!tid && !group) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group >= heads * 2u) return;
    const bool is_beta = group >= heads;
    const uint head = is_beta ? (group - heads) : group;
    device const float* wr = (is_beta ? beta_w : alpha_w) + ulong(head) * cols;
    device float* dst = is_beta ? beta_out : alpha_out;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint bad = 0u;
    const uint vec_cols = cols & ~3u;
    for (uint k = uint(tid) * 4u; k < vec_cols; k += 512u) {
        const float4 w4 = float4(*reinterpret_cast<device const packed_float4*>(wr + k));
        bad |= nax_detail::nonfinite_bit(w4.x) | nax_detail::nonfinite_bit(w4.y) |
               nax_detail::nonfinite_bit(w4.z) | nax_detail::nonfinite_bit(w4.w);
        #pragma unroll
        for (uint t = 0u; t < 8u; ++t) {
            if (t < steps) {
                const float4 x4 = float4(*reinterpret_cast<device const packed_float4*>(input + ulong(t) * cols + k));
                bad |= nax_detail::nonfinite_bit(x4.x) | nax_detail::nonfinite_bit(x4.y) |
                       nax_detail::nonfinite_bit(x4.z) | nax_detail::nonfinite_bit(x4.w);
                sums[t] += dot(w4, x4);
            }
        }
    }
    for (uint k = vec_cols + uint(tid); k < cols; k += 128u) {
        const float wv = wr[k];
        bad |= nax_detail::nonfinite_bit(wv);
        #pragma unroll
        for (uint t = 0u; t < 8u; ++t) {
            if (t < steps) {
                const float xv = input[ulong(t) * cols + k];
                bad |= nax_detail::nonfinite_bit(xv);
                sums[t] += wv * xv;
            }
        }
    }
    #pragma unroll
    for (uint t = 0u; t < 8u; ++t) {
        const float s = simd_sum(sums[t]);
        if (!lane) sg_sum[sg][t] = s;
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane < steps) {
        const float tot = sg_sum[0][lane] + sg_sum[1][lane] + sg_sum[2][lane] + sg_sum[3][lane];
        if (nax_detail::nonfinite_bit(tot)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        dst[ulong(lane) * heads + head] = tot;
    }
}

// Batched final gated mix across verify token rows in a single dispatch.
kernel void nax_verify_gated_mix_rows(device const float* normalized [[buffer(0)]], device const float* scores [[buffer(1)]],
    device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    // shape = {branches, channels, steps, 0}
    const uint branches = shape.x, channels = shape.y, steps = shape.z;
    if (index >= steps * channels) return;
    const uint token = index / channels, c = index % channels;
    const ulong base = ulong(token) * branches * channels + c;
    float result = 0.0f;
    uint bad = 0u;
    if (branches == 4u) {
        #pragma unroll
        for (uint b = 0u; b < 4u; ++b) {
            const ulong at = base + ulong(b) * channels;
            const float nv = normalized[at], sv = scores[at];
            bad |= nax_detail::nonfinite_bit(nv) | nax_detail::nonfinite_bit(sv);
            result = fma(nv, nax_detail::sigmoid_f32(sv), result);
        }
        result *= 0.25f;
    } else {
        for (uint b = 0u; b < branches; ++b) {
            const ulong at = base + ulong(b) * channels;
            const float nv = normalized[at], sv = scores[at];
            bad |= nax_detail::nonfinite_bit(nv) | nax_detail::nonfinite_bit(sv);
            result = fma(nv, nax_detail::sigmoid_f32(sv), result);
        }
        result /= float(branches);
    }
    bad |= nax_detail::nonfinite_bit(result);
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    output[index] = result;
}

kernel void nax_verify_moe_routed(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::NaxVerifyMoeArgs& a [[buffer(3)]],
    device const uint* ids [[buffer(4)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint block_cols = (nax_encoding == 12u || nax_encoding == 13u) ? 256u : 32u;
    const uint block_bytes = nax_encoding == 12u ? 144u : (nax_encoding == 13u ? 176u : (nax_encoding == 8u ? 34u : 24u));
    if (!a.tokens || a.tokens > 8u || !a.selected || a.selected > 16u || !a.experts || a.experts > 512u ||
        !a.columns || a.columns % block_cols != 0u || !a.rows || a.row_bytes != (a.columns / block_cols) * block_bytes ||
        a.broadcast > 1u || a.swiglu > 1u || a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        a.expert_stride_bytes != ulong(a.row_bytes) * a.rows) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint routes = a.tokens * a.selected;
    if (group.y >= routes) return;
    const uint RowsPerSG = (nax_encoding == 12u || nax_encoding == 13u) ? 2u : 4u;
    const uint first_row = (group.x * 2u + uint(sg)) * RowsPerSG;
    if (first_row >= a.rows) return;
    const uint expert = ids[group.y];
    if (expert >= a.experts) {
        if (!lane) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            device float* out = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(group.y) * a.output_stride_bytes);
            for (uint r = 0u; r < RowsPerSG && first_row + r < a.rows; ++r) out[first_row + r] = 0.0f;
        }
        return;
    }
    uint prior_matches = 0u;
    if (!lane) {
        for (uint j = 0u; j < group.y; ++j) {
            if (ids[j] == expert) ++prior_matches;
        }
    }
    prior_matches = simd_broadcast_first(prior_matches);
    if ((prior_matches & 7u) != 0u) return;
    uint match_routes[8];
    uint match_count = 0u;
    for (uint j = group.y; j < routes && match_count < 8u; ++j) {
        if (ids[j] == expert) match_routes[match_count++] = j;
    }
    device const uchar* matrix = weights + ulong(expert) * a.expert_stride_bytes;

    if (nax_encoding == 12u || nax_encoding == 13u) {
        const ushort ix = lane / 8u, iq = (lane % 8u) / 4u, ir = lane % 4u;
        const uint nb = a.columns / 256u;
        float sumf[8][2];
        #pragma unroll
        for (ushort m = 0; m < 8; ++m) { sumf[m][0] = 0.0f; sumf[m][1] = 0.0f; }
        uint bad = 0u;
        for (uint ib = ix; ib < nb; ib += 4u) {
            ushort sc16_r[2][4];
            ushort q1_r[2][4];
            ushort q2_r[2][4];
            ushort qh_r[2][4];
            float d_r[2], min_r[2];
            #pragma unroll
            for (ushort row = 0; row < 2; ++row) {
                const uint physical_row = min(first_row + uint(row), a.rows - 1u);
                device const uchar* block = matrix + ulong(physical_row) * a.row_bytes + ulong(ib) * block_bytes;
                device const ushort* sc = reinterpret_cast<device const ushort*>(block + 4) + iq;
                float d = float(*reinterpret_cast<device const half*>(block));
                float minimum = float(*reinterpret_cast<device const half*>(block + 2));
                if (nax_detail::nonfinite_bit(d) || nax_detail::nonfinite_bit(minimum)) { bad |= 1u; d = 0.0f; minimum = 0.0f; }
                d_r[row] = d;
                min_r[row] = minimum;
                sc16_r[row][0] = ushort(sc[0] & 0x3f3fu);
                sc16_r[row][1] = ushort(sc[2] & 0x3f3fu);
                sc16_r[row][2] = ushort((sc[4] & 0x0f0fu) | ((sc[0] & 0xc0c0u) >> 2));
                sc16_r[row][3] = ushort(((sc[4] >> 4) & 0x0f0fu) | ((sc[2] & 0xc0c0u) >> 2));
                if (nax_encoding == 12u) {
                    device const ushort* q1 = reinterpret_cast<device const ushort*>(block + 16) + 16 * iq + 4 * ir;
                    device const ushort* q2 = q1 + 32;
                    #pragma unroll
                    for (ushort i = 0; i < 4; ++i) { q1_r[row][i] = q1[i]; q2_r[row][i] = q2[i]; qh_r[row][i] = 0u; }
                } else {
                    device const ushort* qh = reinterpret_cast<device const ushort*>(block + 16) + 4 * ir;
                    device const ushort* q1 = reinterpret_cast<device const ushort*>(block + 48) + 16 * iq + 4 * ir;
                    device const ushort* q2 = q1 + 32;
                    #pragma unroll
                    for (ushort i = 0; i < 4; ++i) { qh_r[row][i] = qh[i]; q1_r[row][i] = q1[i]; q2_r[row][i] = q2[i]; }
                }
            }
            for (uint m = 0u; m < match_count; ++m) {
                const uint r_idx = match_routes[m];
                const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
                device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + ulong(in_row) * a.input_stride_bytes);
                device const float* y4 = x + ib * 256u + 64u * iq + 8u * ir;
                float yl[16], yh[16];
                float4 sumy = 0.0f;
                #pragma unroll
                for (ushort i = 0; i < 8; ++i) {
                    yl[i] = y4[i];           bad |= nax_detail::nonfinite_bit(yl[i]);     sumy[0] += yl[i];
                    yl[i + 8] = y4[i + 32];  bad |= nax_detail::nonfinite_bit(yl[i + 8]); sumy[1] += yl[i + 8];
                    yh[i] = y4[i + 128];     bad |= nax_detail::nonfinite_bit(yh[i]);     sumy[2] += yh[i];
                    yh[i + 8] = y4[i + 160]; bad |= nax_detail::nonfinite_bit(yh[i + 8]); sumy[3] += yh[i + 8];
                }
                #pragma unroll
                for (ushort row = 0; row < 2; ++row) {
                    thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16_r[row]);
                    float4 acc1 = 0.0f, acc2 = 0.0f;
                    if (nax_encoding == 12u) {
                        #pragma unroll
                        for (ushort i = 0; i < 4; ++i) {
                            const ushort q1v = q1_r[row][i];
                            const ushort q2v = q2_r[row][i];
                            acc1[0] += yl[2*i]   * float(q1v & 0x000fu); acc1[1] += yl[2*i+1] * float(q1v & 0x0f00u);
                            acc1[2] += yl[2*i+8] * float(q1v & 0x00f0u); acc1[3] += yl[2*i+9] * float(q1v & 0xf000u);
                            acc2[0] += yh[2*i]   * float(q2v & 0x000fu); acc2[1] += yh[2*i+1] * float(q2v & 0x0f00u);
                            acc2[2] += yh[2*i+8] * float(q2v & 0x00f0u); acc2[3] += yh[2*i+9] * float(q2v & 0xf000u);
                        }
                    } else {
                        #pragma unroll
                        for (ushort i = 0; i < 4; ++i) {
                            const uint h = uint(qh_r[row][i]);
                            const uint h_lo = h >> (2u * iq);
                            const uint h_hi = h >> (4u + 2u * iq);
                            const uint q1u = uint(q1_r[row][i]);
                            const uint q2u = uint(q2_r[row][i]);
                            acc1[0] += yl[2*i]   * float((q1u & 0x000fu) | ((h_lo << 4) & 0x0010u));
                            acc1[1] += yl[2*i+1] * float((q1u & 0x0f00u) | ((h_lo << 4) & 0x1000u));
                            acc1[2] += yl[2*i+8] * float((q1u & 0x00f0u) | ((h_lo << 7) & 0x0100u));
                            acc1[3] += yl[2*i+9] * float((q1u & 0xf000u) | ((h_lo << 7) & 0x10000u));
                            acc2[0] += yh[2*i]   * float((q2u & 0x000fu) | ((h_hi << 4) & 0x0010u));
                            acc2[1] += yh[2*i+1] * float((q2u & 0x0f00u) | ((h_hi << 4) & 0x1000u));
                            acc2[2] += yh[2*i+8] * float((q2u & 0x00f0u) | ((h_hi << 7) & 0x0100u));
                            acc2[3] += yh[2*i+9] * float((q2u & 0xf000u) | ((h_hi << 7) & 0x10000u));
                        }
                    }
                    sumf[m][row] += d_r[row] * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * float(sc8[0]) +
                                                (acc1[2] + (1.0f / 256.0f) * acc1[3]) * float(sc8[1]) * (1.0f / 16.0f) +
                                                (acc2[0] + (1.0f / 256.0f) * acc2[1]) * float(sc8[4]) +
                                                (acc2[2] + (1.0f / 256.0f) * acc2[3]) * float(sc8[5]) * (1.0f / 16.0f)) -
                                    min_r[row] * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
                }
            }
        }
        if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
        for (uint m = 0u; m < match_count; ++m) {
            device float* out = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(match_routes[m]) * a.output_stride_bytes);
            for (ushort r = 0; r < 2; ++r) {
                const float tot = simd_sum(sumf[m][r]);
                if (first_row + r < a.rows) nax_verify_detail::write_moe_row(tot, out, first_row + r, a.swiglu, error, lane);
            }
        }
        return;
    }

    const uint rows = min(4u, a.rows - first_row);
    const uint blocks = a.columns / 32u;
    device const uchar* data = matrix + ulong(first_row) * a.row_bytes;
    float sums[8][4];
    #pragma unroll
    for (ushort m = 0; m < 8; ++m) {
        #pragma unroll
        for (ushort r = 0; r < 4; ++r) sums[m][r] = 0.0f;
    }
    uint bad = 0u;
    if (nax_encoding == 7u) {
        const uint shift = 4u * (uint(lane) / 16u), j = uint(lane) % 16u;
        for (uint run = 0u; run * 16u < blocks; ++run) {
            const uint count = min(16u, blocks - run * 16u);
            uint4 words[4];
            #pragma unroll
            for (uint r = 0u; r < 4u; ++r) {
                words[r] = uint4(0u);
                if (r >= rows || nax_layout == 0u) continue;
                device const uchar* q = data + ulong(r) * a.row_bytes + ulong(blocks) * 8ul + ulong(run) * 256ul + ulong(j) * count;
                if (count == 16u) {
                    words[r] = uint4(*reinterpret_cast<device const packed_uint4*>(q));
                } else {
                    for (uint k = 0u; k < count; ++k) words[r][k / 4u] |= uint(q[k]) << (8u * (k % 4u));
                }
            }
            for (uint k = 0u; k < count; ++k) {
                const uint b = run * 16u + k;
                float w_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (uint r = 0u; r < 4u; ++r) {
                    if (r >= rows) continue;
                    device const uchar* row_ptr = data + ulong(r) * a.row_bytes;
                    const uint2 h = nax_layout == 2u
                        ? uint2(*reinterpret_cast<device const packed_uint2*>(row_ptr + ulong(b) * 8ul))
                        : uint2(*reinterpret_cast<device const packed_uint2*>(row_ptr + ulong(b) * 24ul));
                    const uint byte = nax_layout == 2u
                        ? ((words[r][k / 4u] >> (8u * (k % 4u))) & 255u)
                        : uint(row_ptr[ulong(b) * 24ul + 8ul + j]);
                    float dd = float(as_type<half2>(h.x).x), mn = float(as_type<half2>(h.x).y);
                    if (nax_detail::nonfinite_bit(dd) || nax_detail::nonfinite_bit(mn)) { bad |= 1u; dd = 0.0f; mn = 0.0f; }
                    const uint low = (byte >> shift) & 15u;
                    const uint q = low | (((h.y >> lane) & 1u) << 4);
                    w_val[r] = dd * float(q) + mn;
                }
                if (match_count == 1u) {
                    const uint r_idx = match_routes[0];
                    const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
                    device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + ulong(in_row) * a.input_stride_bytes);
                    const float xv = x[b * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(xv);
                    #pragma unroll
                    for (uint r = 0u; r < 4u; ++r) {
                        if (r < rows) sums[0][r] += w_val[r] * xv;
                    }
                } else {
                    for (uint m = 0u; m < match_count; ++m) {
                        const uint r_idx = match_routes[m];
                        const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
                        device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + ulong(in_row) * a.input_stride_bytes);
                        const float xv = x[b * 32u + uint(lane)];
                        bad |= nax_detail::nonfinite_bit(xv);
                        #pragma unroll
                        for (uint r = 0u; r < 4u; ++r) {
                            if (r < rows) sums[m][r] += w_val[r] * xv;
                        }
                    }
                }
            }
        }
    } else {
        if (nax_layout == 0u) {
            for (uint b = 0u; b < blocks; ++b) {
                float w_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (uint r = 0u; r < 4u; ++r) {
                    if (r >= rows) continue;
                    device const uchar* enc = data + ulong(r) * a.row_bytes + ulong(b) * 34ul;
                    float sc = nax_detail::stored_half(enc);
                    if (nax_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                    w_val[r] = sc * float(as_type<char>(enc[2u + lane]));
                }
                if (match_count == 1u) {
                    const uint r_idx = match_routes[0];
                    const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
                    device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + ulong(in_row) * a.input_stride_bytes);
                    const float xv = x[b * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(xv);
                    #pragma unroll
                    for (uint r = 0u; r < 4u; ++r) {
                        if (r < rows) sums[0][r] += w_val[r] * xv;
                    }
                } else {
                    for (uint m = 0u; m < match_count; ++m) {
                        const uint r_idx = match_routes[m];
                        const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
                        device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + ulong(in_row) * a.input_stride_bytes);
                        const float xv = x[b * 32u + uint(lane)];
                        bad |= nax_detail::nonfinite_bit(xv);
                        #pragma unroll
                        for (uint r = 0u; r < 4u; ++r) {
                            if (r < rows) sums[m][r] += w_val[r] * xv;
                        }
                    }
                }
            }
        } else {
            const uint full = blocks / 16u, rest = blocks % 16u;
            for (uint run = 0u; run < full; ++run) {
                uint4 words[4];
                #pragma unroll
                for (uint r = 0u; r < 4u; ++r) {
                    words[r] = r < rows
                        ? uint4(*reinterpret_cast<device const packed_uint4*>(data + ulong(r) * a.row_bytes + ulong(blocks) * 2ul + ulong(run) * 512ul + ulong(lane) * 16ul))
                        : uint4(0u);
                }
                #pragma unroll
                for (uint part = 0u; part < 4u; ++part) {
                    #pragma unroll
                    for (uint k = 0u; k < 4u; ++k) {
                        const uint b = run * 16u + part * 4u + k;
                        float w_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        #pragma unroll
                        for (uint r = 0u; r < 4u; ++r) {
                            if (r >= rows) continue;
                            float sc = nax_detail::stored_half(data + ulong(r) * a.row_bytes + ulong(b) * 2ul);
                            if (nax_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                            w_val[r] = sc * float(as_type<char>(uchar((words[r][part] >> (8u * k)) & 255u)));
                        }
                        if (match_count == 1u) {
                            const uint r_idx = match_routes[0];
                            const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
                            device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + ulong(in_row) * a.input_stride_bytes);
                            const float xv = x[b * 32u + uint(lane)];
                            bad |= nax_detail::nonfinite_bit(xv);
                            #pragma unroll
                            for (uint r = 0u; r < 4u; ++r) {
                                if (r < rows) sums[0][r] += w_val[r] * xv;
                            }
                        } else {
                            for (uint m = 0u; m < match_count; ++m) {
                                const uint r_idx = match_routes[m];
                                const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
                                device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + ulong(in_row) * a.input_stride_bytes);
                                const float xv = x[b * 32u + uint(lane)];
                                bad |= nax_detail::nonfinite_bit(xv);
                                #pragma unroll
                                for (uint r = 0u; r < 4u; ++r) {
                                    if (r < rows) sums[m][r] += w_val[r] * xv;
                                }
                            }
                        }
                    }
                }
            }
            for (uint k = 0u; k < rest; ++k) {
                const uint b = full * 16u + k;
                float w_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (uint r = 0u; r < 4u; ++r) {
                    if (r >= rows) continue;
                    device const uchar* row_ptr = data + ulong(r) * a.row_bytes;
                    float sc = nax_detail::stored_half(row_ptr + ulong(b) * 2ul);
                    if (nax_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                    const uchar qb = row_ptr[ulong(blocks) * 2ul + ulong(full) * 512ul + ulong(lane) * rest + k];
                    w_val[r] = sc * float(as_type<char>(qb));
                }
                if (match_count == 1u) {
                    const uint r_idx = match_routes[0];
                    const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
                    device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + ulong(in_row) * a.input_stride_bytes);
                    const float xv = x[b * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(xv);
                    #pragma unroll
                    for (uint r = 0u; r < 4u; ++r) {
                        if (r < rows) sums[0][r] += w_val[r] * xv;
                    }
                } else {
                    for (uint m = 0u; m < match_count; ++m) {
                        const uint r_idx = match_routes[m];
                        const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
                        device const float* x = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + ulong(in_row) * a.input_stride_bytes);
                        const float xv = x[b * 32u + uint(lane)];
                        bad |= nax_detail::nonfinite_bit(xv);
                        #pragma unroll
                        for (uint r = 0u; r < 4u; ++r) {
                            if (r < rows) sums[m][r] += w_val[r] * xv;
                        }
                    }
                }
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
    for (uint m = 0u; m < match_count; ++m) {
        device float* out = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(match_routes[m]) * a.output_stride_bytes);
        for (ushort r = 0; r < 4; ++r) {
            const float tot = simd_sum(sums[m][r]);
            if (first_row + r < a.rows) nax_verify_detail::write_moe_row(tot, out, first_row + r, a.swiglu, error, lane);
        }
    }
}

namespace nax_verify_union_detail {

struct NaxVerifyMoeUnionArgs {
    uint experts, selected, tokens, grid_x_gate_up;
    uint grid_x_down, reserved0, reserved1, reserved2;
};
static_assert(sizeof(NaxVerifyMoeUnionArgs) == 32, "NAX verify MoE union args ABI");

struct QkScales {
    float d, minimum;
    uchar4 sc_lo, sc_hi;
};

static inline QkScales decode_qk_scales(device const uchar* block, ushort iq, thread uint& bad) {
    QkScales s;
    device const ushort* sc = reinterpret_cast<device const ushort*>(block + 4) + iq;
    float d = float(*reinterpret_cast<device const half*>(block));
    float minimum = float(*reinterpret_cast<device const half*>(block + 2));
    if (nax_detail::nonfinite_bit(d) || nax_detail::nonfinite_bit(minimum)) { bad |= 1u; d = 0.0f; minimum = 0.0f; }
    s.d = d;
    s.minimum = minimum;
    ushort sc16[4];
    sc16[0] = ushort(sc[0] & 0x3f3fu);
    sc16[1] = ushort(sc[2] & 0x3f3fu);
    sc16[2] = ushort((sc[4] & 0x0f0fu) | ((sc[0] & 0xc0c0u) >> 2));
    sc16[3] = ushort(((sc[4] >> 4) & 0x0f0fu) | ((sc[2] & 0xc0c0u) >> 2));
    thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);
    s.sc_lo = uchar4(sc8[0], sc8[1], sc8[4], sc8[5]);
    s.sc_hi = uchar4(sc8[2], sc8[3], sc8[6], sc8[7]);
    return s;
}

static inline void decode_qk_quants(device const uchar* block, ushort iq, ushort ir,
    thread uchar4& q_l0, thread uchar4& q_l1, thread uchar4& q_h0, thread uchar4& q_h1) {
    if (nax_encoding == 12u) {
        const uchar4 q1 = uchar4(*reinterpret_cast<device const packed_uchar4*>(block + 16 + 32u * iq + 4u * ir));
        const uchar4 q2 = uchar4(*reinterpret_cast<device const packed_uchar4*>(block + 16 + 64u + 32u * iq + 4u * ir));
        q_l0 = q1 & uchar4(0x0fu);
        q_l1 = q1 >> 4u;
        q_h0 = q2 & uchar4(0x0fu);
        q_h1 = q2 >> 4u;
    } else {
        const uchar4 qh4 = uchar4(*reinterpret_cast<device const packed_uchar4*>(block + 16 + 4u * ir));
        const uchar4 q1 = uchar4(*reinterpret_cast<device const packed_uchar4*>(block + 48 + 32u * iq + 4u * ir));
        const uchar4 q2 = uchar4(*reinterpret_cast<device const packed_uchar4*>(block + 48 + 64u + 32u * iq + 4u * ir));
        const uchar4 h_l0 = ((qh4 >> uchar(2u * iq)) << 4u) & uchar4(0x10u);
        const uchar4 h_l1 = ((qh4 >> uchar(2u * iq + 1u)) << 4u) & uchar4(0x10u);
        const uchar4 h_h0 = ((qh4 >> uchar(4u + 2u * iq)) << 4u) & uchar4(0x10u);
        const uchar4 h_h1 = ((qh4 >> uchar(5u + 2u * iq)) << 4u) & uchar4(0x10u);
        q_l0 = (q1 & uchar4(0x0fu)) | h_l0;
        q_l1 = (q1 >> 4u) | h_l1;
        q_h0 = (q2 & uchar4(0x0fu)) | h_h0;
        q_h1 = (q2 >> 4u) | h_h1;
    }
}

static inline float eval_qk_half(thread const QkScales& s, uchar4 q_l0, uchar4 q_l1, uchar4 q_h0, uchar4 q_h1,
    float4 xl0, float4 xl1, float4 xh0, float4 xh1, float4 sumy) {
    const float4 dots(dot(xl0, float4(q_l0)), dot(xl1, float4(q_l1)),
                      dot(xh0, float4(q_h0)), dot(xh1, float4(q_h1)));
    return s.d * dot(dots, float4(s.sc_lo)) - s.minimum * dot(sumy, float4(s.sc_hi));
}

static inline uint check_f4_nonfinite(float4 v) {
    return nax_detail::nonfinite_bit(v.x) | nax_detail::nonfinite_bit(v.y) |
           nax_detail::nonfinite_bit(v.z) | nax_detail::nonfinite_bit(v.w);
}

template <uint MC, uint NThreads = 128u>
static inline void stage_moe_input_chunk(device const float* const* x_rows, uint c_base, uint chunk_cols,
    ushort tid, threadgroup float (*sx)[512], thread uint& bad) {
    const uint chunk_quads = chunk_cols >> 2u;
    for (uint idx = uint(tid); idx < MC * chunk_quads; idx += NThreads) {
        const uint m = idx / chunk_quads;
        const uint q = idx % chunk_quads;
        const float4 v = float4(*reinterpret_cast<device const packed_float4*>(x_rows[m] + c_base + q * 4u));
        bad |= check_f4_nonfinite(v);
        *reinterpret_cast<threadgroup float4*>(&sx[m][q * 4u]) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint MC, uint NR, uint NThreads = 128u>
static inline void moe_gate_up_pass(device const uchar* gate_mat, device const uchar* up_mat,
    device const float* input, device float* output, constant nax_detail::NaxVerifyMoeArgs& a,
    device const uint* rec, uint first_row, ushort tid, ushort lane, threadgroup float (*sx)[512], thread uint& bad) {
    const uint rows_live = first_row < a.rows ? min(NR, a.rows - first_row) : 0u;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    device const float* x_rows[MC];
    uint routes[MC];
    #pragma unroll
    for (uint m = 0u; m < MC; ++m) {
        const uint r_idx = rec[2u + m];
        routes[m] = r_idx;
        const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
        x_rows[m] = input + ulong(in_row) * in_stride;
    }

    float sum_g[MC][NR], sum_u[MC][NR];
    #pragma unroll
    for (uint m = 0u; m < MC; ++m) {
        #pragma unroll
        for (uint r = 0u; r < NR; ++r) { sum_g[m][r] = 0.0f; sum_u[m][r] = 0.0f; }
    }

    const uint block_bytes = nax_encoding == 12u ? 144u : (nax_encoding == 13u ? 176u : (nax_encoding == 8u ? 34u : 24u));
    const uint blocks = a.columns / 32u;
    const uint full = blocks / 16u, rest = blocks % 16u;

    if (nax_encoding == 12u || nax_encoding == 13u) {
        const ushort ix = lane / 16u, sl = lane % 16u, iq = sl / 8u, ir = sl % 8u;
        for (uint c_base = 0u; c_base < a.columns; c_base += 512u) {
            const uint chunk_cols = min(512u, a.columns - c_base);
            stage_moe_input_chunk<MC, NThreads>(x_rows, c_base, chunk_cols, tid, sx, bad);
            const uint chunk_nb = chunk_cols / 256u;
            if (rows_live > 0u && ix < chunk_nb) {
                const uint ib = (c_base / 256u) + uint(ix);
                const uint s_off = uint(ix) * 256u + 64u * uint(iq) + 4u * uint(ir);
                float4 xl0[MC], xl1[MC], xh0[MC], xh1[MC], sy[MC];
                #pragma unroll
                for (uint m = 0u; m < MC; ++m) {
                    threadgroup const float* xp = &sx[m][s_off];
                    xl0[m] = *reinterpret_cast<threadgroup const float4*>(xp);
                    xl1[m] = *reinterpret_cast<threadgroup const float4*>(xp + 32u);
                    xh0[m] = *reinterpret_cast<threadgroup const float4*>(xp + 128u);
                    xh1[m] = *reinterpret_cast<threadgroup const float4*>(xp + 160u);
                    sy[m] = float4(xl0[m].x + xl0[m].y + xl0[m].z + xl0[m].w,
                                   xl1[m].x + xl1[m].y + xl1[m].z + xl1[m].w,
                                   xh0[m].x + xh0[m].y + xh0[m].z + xh0[m].w,
                                   xh1[m].x + xh1[m].y + xh1[m].z + xh1[m].w);
                }
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    if (r < rows_live) {
                        device const uchar* gb = gate_mat + ulong(first_row + r) * a.row_bytes + ulong(ib) * block_bytes;
                        device const uchar* ub = up_mat + ulong(first_row + r) * a.row_bytes + ulong(ib) * block_bytes;
                        const auto gs = decode_qk_scales(gb, iq, bad);
                        const auto us = decode_qk_scales(ub, iq, bad);
                        uchar4 gql0, gql1, gqh0, gqh1, uql0, uql1, uqh0, uqh1;
                        decode_qk_quants(gb, iq, ir, gql0, gql1, gqh0, gqh1);
                        decode_qk_quants(ub, iq, ir, uql0, uql1, uqh0, uqh1);
                        for (uint m = 0u; m < MC; ++m) {
                            sum_g[m][r] += eval_qk_half(gs, gql0, gql1, gqh0, gqh1, xl0[m], xl1[m], xh0[m], xh1[m], sy[m]);
                            sum_u[m][r] += eval_qk_half(us, uql0, uql1, uqh0, uqh1, xl0[m], xl1[m], xh0[m], xh1[m], sy[m]);
                        }
                    }
                }
            }
            if (c_base + chunk_cols < a.columns) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else if (nax_encoding == 7u) {
        const uint shift = 4u * (uint(lane) / 16u), j = uint(lane) % 16u;
        for (uint c_base = 0u; c_base < a.columns; c_base += 512u) {
            const uint chunk_cols = min(512u, a.columns - c_base);
            stage_moe_input_chunk<MC, NThreads>(x_rows, c_base, chunk_cols, tid, sx, bad);
            const uint run = c_base / 512u, count = chunk_cols / 32u;
            if (rows_live > 0u) {
                if (nax_layout == 2u && count == 16u) {
                    uint2 g_half[NR], u_half[NR];
                    uint g_hx[NR], g_hy[NR], u_hx[NR], u_hy[NR];
                    #pragma unroll
                    for (uint r = 0u; r < NR; ++r) {
                        if (r < rows_live) {
                            device const uchar* gr = gate_mat + ulong(first_row + r) * a.row_bytes;
                            device const uchar* ur = up_mat + ulong(first_row + r) * a.row_bytes;
                            g_half[r] = uint2(*reinterpret_cast<device const packed_uint2*>(gr + ulong(blocks) * 8ul + ulong(run) * 256ul + ulong(lane) * 8ul));
                            u_half[r] = uint2(*reinterpret_cast<device const packed_uint2*>(ur + ulong(blocks) * 8ul + ulong(run) * 256ul + ulong(lane) * 8ul));
                            if (lane < 16u) {
                                uint2 gh = uint2(*reinterpret_cast<device const packed_uint2*>(gr + ulong(run * 16u + uint(lane)) * 8ul));
                                uint2 uh = uint2(*reinterpret_cast<device const packed_uint2*>(ur + ulong(run * 16u + uint(lane)) * 8ul));
                                if (((gh.x & 0x7c00u) == 0x7c00u) || ((gh.x & 0x7c000000u) == 0x7c000000u)) { bad |= 1u; gh.x = 0u; }
                                if (((uh.x & 0x7c00u) == 0x7c00u) || ((uh.x & 0x7c000000u) == 0x7c000000u)) { bad |= 1u; uh.x = 0u; }
                                g_hx[r] = gh.x; g_hy[r] = gh.y; u_hx[r] = uh.x; u_hy[r] = uh.y;
                            } else {
                                g_hx[r] = 0u; g_hy[r] = 0u; u_hx[r] = 0u; u_hy[r] = 0u;
                            }
                        } else {
                            g_half[r] = uint2(0u); u_half[r] = uint2(0u);
                            g_hx[r] = 0u; g_hy[r] = 0u; u_hx[r] = 0u; u_hy[r] = 0u;
                        }
                    }
                    #pragma unroll
                    for (uint part = 0u; part < 4u; ++part) {
                        const ushort src = ushort(2u * j + (part >> 1u));
                        uint gw_part[NR], uw_part[NR];
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            gw_part[r] = simd_shuffle((part & 1u) ? g_half[r].y : g_half[r].x, src);
                            uw_part[r] = simd_shuffle((part & 1u) ? u_half[r].y : u_half[r].x, src);
                        }
                        #pragma unroll
                        for (uint sub = 0u; sub < 4u; ++sub) {
                            const uint k = part * 4u + sub;
                            const uint s_col = k * 32u + uint(lane);
                            float xv[MC];
                            for (uint m = 0u; m < MC; ++m) xv[m] = sx[m][s_col];
                            #pragma unroll
                            for (uint r = 0u; r < NR; ++r) {
                                if (r < rows_live) {
                                    const uint ghx = simd_broadcast(g_hx[r], ushort(k)), ghy = simd_broadcast(g_hy[r], ushort(k));
                                    const uint uhx = simd_broadcast(u_hx[r], ushort(k)), uhy = simd_broadcast(u_hy[r], ushort(k));
                                    const uint gb = (gw_part[r] >> (8u * sub)) & 255u;
                                    const uint ub = (uw_part[r] >> (8u * sub)) & 255u;
                                    const uint gq = ((gb >> shift) & 15u) | (((ghy >> lane) & 1u) << 4);
                                    const uint uq = ((ub >> shift) & 15u) | (((uhy >> lane) & 1u) << 4);
                                    const half2 gdm = as_type<half2>(ghx), udm = as_type<half2>(uhx);
                                    const float gw = float(gdm.x) * float(gq) + float(gdm.y);
                                    const float uw = float(udm.x) * float(uq) + float(udm.y);
                                    for (uint m = 0u; m < MC; ++m) {
                                        sum_g[m][r] += gw * xv[m];
                                        sum_u[m][r] += uw * xv[m];
                                    }
                                }
                            }
                        }
                    }
                } else {
                    uint4 g_words[NR], u_words[NR];
                    #pragma unroll
                    for (uint r = 0u; r < NR; ++r) {
                        g_words[r] = uint4(0u); u_words[r] = uint4(0u);
                        if (r >= rows_live || nax_layout == 0u) continue;
                        device const uchar* gq = gate_mat + ulong(first_row + r) * a.row_bytes + ulong(blocks) * 8ul + ulong(run) * 256ul + ulong(j) * count;
                        device const uchar* uq = up_mat + ulong(first_row + r) * a.row_bytes + ulong(blocks) * 8ul + ulong(run) * 256ul + ulong(j) * count;
                        if (count == 4u) {
                            g_words[r].x = as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(gq)));
                            u_words[r].x = as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(uq)));
                        } else {
                            for (uint k = 0u; k < count; ++k) {
                                g_words[r][k / 4u] |= uint(gq[k]) << (8u * (k % 4u));
                                u_words[r][k / 4u] |= uint(uq[k]) << (8u * (k % 4u));
                            }
                        }
                    }
                    for (uint k = 0u; k < count; ++k) {
                        const uint b = run * 16u + k;
                        const uint s_col = k * 32u + uint(lane);
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            if (r >= rows_live) continue;
                            for (uint which = 0u; which < 2u; ++which) {
                                device const uchar* r_ptr = (which ? up_mat : gate_mat) + ulong(first_row + r) * a.row_bytes;
                                uint2 h = nax_layout == 2u
                                    ? uint2(*reinterpret_cast<device const packed_uint2*>(r_ptr + ulong(b) * 8ul))
                                    : uint2(*reinterpret_cast<device const packed_uint2*>(r_ptr + ulong(b) * 24ul));
                                if (((h.x & 0x7c00u) == 0x7c00u) || ((h.x & 0x7c000000u) == 0x7c000000u)) { bad |= 1u; h.x = 0u; }
                                const uint4 w4 = which ? u_words[r] : g_words[r];
                                const uint byte = nax_layout == 2u
                                    ? ((w4[k / 4u] >> (8u * (k % 4u))) & 255u)
                                    : uint(r_ptr[ulong(b) * 24ul + 8ul + j]);
                                const half2 dm = as_type<half2>(h.x);
                                const uint q = ((byte >> shift) & 15u) | (((h.y >> lane) & 1u) << 4);
                                const float wv = float(dm.x) * float(q) + float(dm.y);
                                for (uint m = 0u; m < MC; ++m) {
                                    const float xv = sx[m][s_col];
                                    if (which) sum_u[m][r] += wv * xv;
                                    else sum_g[m][r] += wv * xv;
                                }
                            }
                        }
                    }
                }
            }
            if (c_base + chunk_cols < a.columns) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        if (nax_layout == 0u) {
            const uint b_sub = uint(lane) >> 3u, q_sub = (uint(lane) & 7u) * 4u;
            for (uint c_base = 0u; c_base < a.columns; c_base += 512u) {
                const uint chunk_cols = min(512u, a.columns - c_base);
                stage_moe_input_chunk<MC, NThreads>(x_rows, c_base, chunk_cols, tid, sx, bad);
                if (rows_live > 0u) {
                    const uint chunk_blocks = chunk_cols >> 5u;
                    const uint vec_blocks = chunk_blocks & ~3u;
                    const uint b_start = c_base >> 5u;
                    for (uint b_off = 0u; b_off < vec_blocks; b_off += 4u) {
                        const uint blk = b_start + b_off + b_sub;
                        const uint s_off = (b_off + b_sub) * 32u + q_sub;
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            if (r < rows_live) {
                                device const uchar* ge = gate_mat + ulong(first_row + r) * a.row_bytes + ulong(blk) * 34ul;
                                device const uchar* ue = up_mat + ulong(first_row + r) * a.row_bytes + ulong(blk) * 34ul;
                                float gs = nax_detail::stored_half(ge), us = nax_detail::stored_half(ue);
                                if (nax_detail::nonfinite_bit(gs)) { bad |= 1u; gs = 0.0f; }
                                if (nax_detail::nonfinite_bit(us)) { bad |= 1u; us = 0.0f; }
                                const float4 gwf = float4(char4(*reinterpret_cast<device const packed_char4*>(ge + 2u + q_sub)));
                                const float4 uwf = float4(char4(*reinterpret_cast<device const packed_char4*>(ue + 2u + q_sub)));
                                for (uint m = 0u; m < MC; ++m) {
                                    const float4 xv = *reinterpret_cast<threadgroup const float4*>(&sx[m][s_off]);
                                    sum_g[m][r] += gs * dot(gwf, xv);
                                    sum_u[m][r] += us * dot(uwf, xv);
                                }
                            }
                        }
                    }
                    for (uint b_off = vec_blocks; b_off < chunk_blocks; ++b_off) {
                        const uint b = b_start + b_off;
                        const uint s_col = b_off * 32u + uint(lane);
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            if (r < rows_live) {
                                device const uchar* ge = gate_mat + ulong(first_row + r) * a.row_bytes + ulong(b) * 34ul;
                                device const uchar* ue = up_mat + ulong(first_row + r) * a.row_bytes + ulong(b) * 34ul;
                                float gs = nax_detail::stored_half(ge), us = nax_detail::stored_half(ue);
                                if (nax_detail::nonfinite_bit(gs)) { bad |= 1u; gs = 0.0f; }
                                if (nax_detail::nonfinite_bit(us)) { bad |= 1u; us = 0.0f; }
                                const float gw = gs * float(as_type<char>(ge[2u + lane]));
                                const float uw = us * float(as_type<char>(ue[2u + lane]));
                                for (uint m = 0u; m < MC; ++m) {
                                    const float xv = sx[m][s_col];
                                    sum_g[m][r] += gw * xv;
                                    sum_u[m][r] += uw * xv;
                                }
                            }
                        }
                    }
                }
                if (c_base + chunk_cols < a.columns) threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            for (uint run = 0u; run < full; ++run) {
                stage_moe_input_chunk<MC, NThreads>(x_rows, run * 512u, 512u, tid, sx, bad);
                if (rows_live > 0u) {
                    uint4 gw4[NR], uw4[NR];
                    float gsc_lane[NR], usc_lane[NR];
                    #pragma unroll
                    for (uint r = 0u; r < NR; ++r) {
                        if (r < rows_live) {
                            device const uchar* gr = gate_mat + ulong(first_row + r) * a.row_bytes;
                            device const uchar* ur = up_mat + ulong(first_row + r) * a.row_bytes;
                            gw4[r] = uint4(*reinterpret_cast<device const packed_uint4*>(gr + ulong(blocks) * 2ul + ulong(run) * 512ul + ulong(lane) * 16ul));
                            uw4[r] = uint4(*reinterpret_cast<device const packed_uint4*>(ur + ulong(blocks) * 2ul + ulong(run) * 512ul + ulong(lane) * 16ul));
                            if (lane < 16u) {
                                const ushort gsb = *reinterpret_cast<device const ushort*>(gr + ulong(run * 16u + uint(lane)) * 2ul);
                                const ushort usb = *reinterpret_cast<device const ushort*>(ur + ulong(run * 16u + uint(lane)) * 2ul);
                                if ((gsb & 0x7c00u) == 0x7c00u) { bad |= 1u; gsc_lane[r] = 0.0f; }
                                else gsc_lane[r] = float(as_type<half>(gsb));
                                if ((usb & 0x7c00u) == 0x7c00u) { bad |= 1u; usc_lane[r] = 0.0f; }
                                else usc_lane[r] = float(as_type<half>(usb));
                            } else {
                                gsc_lane[r] = 0.0f; usc_lane[r] = 0.0f;
                            }
                        } else {
                            gw4[r] = uint4(0u); uw4[r] = uint4(0u);
                            gsc_lane[r] = 0.0f; usc_lane[r] = 0.0f;
                        }
                    }
                    #pragma unroll
                    for (uint part = 0u; part < 4u; ++part) {
                        uint gwp[NR], uwp[NR];
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            gwp[r] = part == 0u ? gw4[r].x : (part == 1u ? gw4[r].y : (part == 2u ? gw4[r].z : gw4[r].w));
                            uwp[r] = part == 0u ? uw4[r].x : (part == 1u ? uw4[r].y : (part == 2u ? uw4[r].z : uw4[r].w));
                        }
                        #pragma unroll
                        for (uint k = 0u; k < 4u; ++k) {
                            const uint b_local = part * 4u + k;
                            const uint s_col = b_local * 32u + uint(lane);
                            float xv[MC];
                            for (uint m = 0u; m < MC; ++m) xv[m] = sx[m][s_col];
                            #pragma unroll
                            for (uint r = 0u; r < NR; ++r) {
                                if (r < rows_live) {
                                    const float gs = simd_broadcast(gsc_lane[r], ushort(b_local));
                                    const float us = simd_broadcast(usc_lane[r], ushort(b_local));
                                    const float gw = gs * float(as_type<char>(uchar((gwp[r] >> (8u * k)) & 255u)));
                                    const float uw = us * float(as_type<char>(uchar((uwp[r] >> (8u * k)) & 255u)));
                                    for (uint m = 0u; m < MC; ++m) {
                                        sum_g[m][r] += gw * xv[m];
                                        sum_u[m][r] += uw * xv[m];
                                    }
                                }
                            }
                        }
                    }
                }
                if (run + 1u < full || rest > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (rest > 0u) {
                stage_moe_input_chunk<MC, NThreads>(x_rows, full * 512u, rest * 32u, tid, sx, bad);
                if (rows_live > 0u) {
                    for (uint k = 0u; k < rest; ++k) {
                        const uint b = full * 16u + k;
                        const uint s_col = k * 32u + uint(lane);
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            if (r < rows_live) {
                                device const uchar* gr = gate_mat + ulong(first_row + r) * a.row_bytes;
                                device const uchar* ur = up_mat + ulong(first_row + r) * a.row_bytes;
                                float gs = nax_detail::stored_half(gr + ulong(b) * 2ul);
                                float us = nax_detail::stored_half(ur + ulong(b) * 2ul);
                                if (nax_detail::nonfinite_bit(gs)) { bad |= 1u; gs = 0.0f; }
                                if (nax_detail::nonfinite_bit(us)) { bad |= 1u; us = 0.0f; }
                                const ulong q_off = ulong(blocks) * 2ul + ulong(full) * 512ul + ulong(lane) * rest + k;
                                const float gw = gs * float(as_type<char>(gr[q_off]));
                                const float uw = us * float(as_type<char>(ur[q_off]));
                                for (uint m = 0u; m < MC; ++m) {
                                    const float xv = sx[m][s_col];
                                    sum_g[m][r] += gw * xv;
                                    sum_u[m][r] += uw * xv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (rows_live > 0u) {
        for (uint m = 0u; m < MC; ++m) {
            device float* out_row = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(routes[m]) * a.output_stride_bytes);
            #pragma unroll
            for (uint r = 0u; r < NR; ++r) {
                const float g0 = simd_sum(sum_g[m][r]);
                const float u0 = simd_sum(sum_u[m][r]);
                if (!lane && r < rows_live) {
                    const float val = (g0 * nax_detail::sigmoid_f32(g0)) * u0;
                    bad |= nax_detail::nonfinite_bit(g0) | nax_detail::nonfinite_bit(u0) | nax_detail::nonfinite_bit(val);
                    out_row[first_row + r] = val;
                }
            }
        }
    }
}

template <uint MC, uint NR, uint NThreads = 128u>
static inline void moe_down_pass(device const uchar* matrix, device const float* input,
    device float* output, constant nax_detail::NaxVerifyMoeArgs& a,
    device const uint* rec, uint first_row, ushort tid, ushort lane, threadgroup float (*sx)[512], thread uint& bad) {
    const uint rows_live = first_row < a.rows ? min(NR, a.rows - first_row) : 0u;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    device const float* x_rows[MC];
    uint routes[MC];
    for (uint m = 0u; m < MC; ++m) {
        const uint r_idx = rec[2u + m];
        routes[m] = r_idx;
        const uint in_row = a.broadcast ? (r_idx / a.selected) : r_idx;
        x_rows[m] = input + ulong(in_row) * in_stride;
    }

    float sums[MC][NR];
    for (uint m = 0u; m < MC; ++m) {
        #pragma unroll
        for (uint r = 0u; r < NR; ++r) sums[m][r] = 0.0f;
    }

    const uint block_bytes = nax_encoding == 12u ? 144u : (nax_encoding == 13u ? 176u : (nax_encoding == 8u ? 34u : 24u));
    const uint blocks = a.columns / 32u;
    const uint full = blocks / 16u, rest = blocks % 16u;
    device const uchar* data = matrix + ulong(first_row < a.rows ? first_row : 0u) * a.row_bytes;

    if (nax_encoding == 12u || nax_encoding == 13u) {
        const ushort ix = lane / 16u, sl = lane % 16u, iq = sl / 8u, ir = sl % 8u;
        for (uint c_base = 0u; c_base < a.columns; c_base += 512u) {
            const uint chunk_cols = min(512u, a.columns - c_base);
            stage_moe_input_chunk<MC, NThreads>(x_rows, c_base, chunk_cols, tid, sx, bad);
            const uint chunk_nb = chunk_cols / 256u;
            if (rows_live > 0u && ix < chunk_nb) {
                const uint ib = (c_base / 256u) + uint(ix);
                const uint s_off = uint(ix) * 256u + 64u * uint(iq) + 4u * uint(ir);
                float4 xl0[MC], xl1[MC], xh0[MC], xh1[MC], sy[MC];
                for (uint m = 0u; m < MC; ++m) {
                    threadgroup const float* xp = &sx[m][s_off];
                    xl0[m] = *reinterpret_cast<threadgroup const float4*>(xp);
                    xl1[m] = *reinterpret_cast<threadgroup const float4*>(xp + 32u);
                    xh0[m] = *reinterpret_cast<threadgroup const float4*>(xp + 128u);
                    xh1[m] = *reinterpret_cast<threadgroup const float4*>(xp + 160u);
                    sy[m] = float4(xl0[m].x + xl0[m].y + xl0[m].z + xl0[m].w,
                                   xl1[m].x + xl1[m].y + xl1[m].z + xl1[m].w,
                                   xh0[m].x + xh0[m].y + xh0[m].z + xh0[m].w,
                                   xh1[m].x + xh1[m].y + xh1[m].z + xh1[m].w);
                }
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    if (r < rows_live) {
                        device const uchar* blk = data + ulong(r) * a.row_bytes + ulong(ib) * block_bytes;
                        const auto sc = decode_qk_scales(blk, iq, bad);
                        uchar4 ql0, ql1, qh0, qh1;
                        decode_qk_quants(blk, iq, ir, ql0, ql1, qh0, qh1);
                        for (uint m = 0u; m < MC; ++m) {
                            sums[m][r] += eval_qk_half(sc, ql0, ql1, qh0, qh1, xl0[m], xl1[m], xh0[m], xh1[m], sy[m]);
                        }
                    }
                }
            }
            if (c_base + chunk_cols < a.columns) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else if (nax_encoding == 7u) {
        const uint shift = 4u * (uint(lane) / 16u), j = uint(lane) % 16u;
        for (uint c_base = 0u; c_base < a.columns; c_base += 512u) {
            const uint chunk_cols = min(512u, a.columns - c_base);
            stage_moe_input_chunk<MC, NThreads>(x_rows, c_base, chunk_cols, tid, sx, bad);
            const uint run = c_base / 512u, count = chunk_cols / 32u;
            if (rows_live > 0u) {
                if (nax_layout == 2u && count == 16u) {
                    uint2 w_half[NR];
                    uint hx[NR], hy[NR];
                    #pragma unroll
                    for (uint r = 0u; r < NR; ++r) {
                        if (r < rows_live) {
                            device const uchar* row_ptr = data + ulong(r) * a.row_bytes;
                            w_half[r] = uint2(*reinterpret_cast<device const packed_uint2*>(row_ptr + ulong(blocks) * 8ul + ulong(run) * 256ul + ulong(lane) * 8ul));
                            if (lane < 16u) {
                                uint2 h = uint2(*reinterpret_cast<device const packed_uint2*>(row_ptr + ulong(run * 16u + uint(lane)) * 8ul));
                                if (((h.x & 0x7c00u) == 0x7c00u) || ((h.x & 0x7c000000u) == 0x7c000000u)) { bad |= 1u; h.x = 0u; }
                                hx[r] = h.x; hy[r] = h.y;
                            } else {
                                hx[r] = 0u; hy[r] = 0u;
                            }
                        } else {
                            w_half[r] = uint2(0u); hx[r] = 0u; hy[r] = 0u;
                        }
                    }
                    #pragma unroll
                    for (uint part = 0u; part < 4u; ++part) {
                        const ushort src = ushort(2u * j + (part >> 1u));
                        uint w_part[NR];
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            w_part[r] = simd_shuffle((part & 1u) ? w_half[r].y : w_half[r].x, src);
                        }
                        #pragma unroll
                        for (uint sub = 0u; sub < 4u; ++sub) {
                            const uint k = part * 4u + sub;
                            const uint s_col = k * 32u + uint(lane);
                            float xv[MC];
                            for (uint m = 0u; m < MC; ++m) xv[m] = sx[m][s_col];
                            #pragma unroll
                            for (uint r = 0u; r < NR; ++r) {
                                if (r < rows_live) {
                                    const uint dhx = simd_broadcast(hx[r], ushort(k));
                                    const uint dhy = simd_broadcast(hy[r], ushort(k));
                                    const uint byte = (w_part[r] >> (8u * sub)) & 255u;
                                    const uint q = ((byte >> shift) & 15u) | (((dhy >> lane) & 1u) << 4);
                                    const half2 dm = as_type<half2>(dhx);
                                    const float wv = float(dm.x) * float(q) + float(dm.y);
                                    for (uint m = 0u; m < MC; ++m) sums[m][r] += wv * xv[m];
                                }
                            }
                        }
                    }
                } else if (nax_layout == 2u && count == 4u) {
                    uint w1[NR], hx[NR], hy[NR];
                    #pragma unroll
                    for (uint r = 0u; r < NR; ++r) {
                        if (r < rows_live) {
                            device const uchar* row_ptr = data + ulong(r) * a.row_bytes;
                            w1[r] = as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(row_ptr + ulong(blocks) * 8ul + ulong(run) * 256ul + ulong(j) * 4ul)));
                            if (lane < 4u) {
                                uint2 h = uint2(*reinterpret_cast<device const packed_uint2*>(row_ptr + ulong(run * 16u + uint(lane)) * 8ul));
                                if (((h.x & 0x7c00u) == 0x7c00u) || ((h.x & 0x7c000000u) == 0x7c000000u)) { bad |= 1u; h.x = 0u; }
                                hx[r] = h.x; hy[r] = h.y;
                            } else {
                                hx[r] = 0u; hy[r] = 0u;
                            }
                        } else {
                            w1[r] = 0u; hx[r] = 0u; hy[r] = 0u;
                        }
                    }
                    #pragma unroll
                    for (uint sub = 0u; sub < 4u; ++sub) {
                        const uint s_col = sub * 32u + uint(lane);
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            if (r < rows_live) {
                                const uint dhx = simd_broadcast(hx[r], ushort(sub));
                                const uint dhy = simd_broadcast(hy[r], ushort(sub));
                                const uint byte = (w1[r] >> (8u * sub)) & 255u;
                                const uint q = ((byte >> shift) & 15u) | (((dhy >> lane) & 1u) << 4);
                                const half2 dm = as_type<half2>(dhx);
                                const float wv = float(dm.x) * float(q) + float(dm.y);
                                for (uint m = 0u; m < MC; ++m) sums[m][r] += wv * sx[m][s_col];
                            }
                        }
                    }
                } else {
                    uint4 words[NR];
                    #pragma unroll
                    for (uint r = 0u; r < NR; ++r) {
                        words[r] = uint4(0u);
                        if (r >= rows_live || nax_layout == 0u) continue;
                        device const uchar* q = data + ulong(r) * a.row_bytes + ulong(blocks) * 8ul + ulong(run) * 256ul + ulong(j) * count;
                        for (uint k = 0u; k < count; ++k) words[r][k / 4u] |= uint(q[k]) << (8u * (k % 4u));
                    }
                    for (uint k = 0u; k < count; ++k) {
                        const uint b = run * 16u + k;
                        const uint s_col = k * 32u + uint(lane);
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            if (r >= rows_live) continue;
                            device const uchar* row_ptr = data + ulong(r) * a.row_bytes;
                            uint2 h = nax_layout == 2u
                                ? uint2(*reinterpret_cast<device const packed_uint2*>(row_ptr + ulong(b) * 8ul))
                                : uint2(*reinterpret_cast<device const packed_uint2*>(row_ptr + ulong(b) * 24ul));
                            if (((h.x & 0x7c00u) == 0x7c00u) || ((h.x & 0x7c000000u) == 0x7c000000u)) { bad |= 1u; h.x = 0u; }
                            const uint byte = nax_layout == 2u
                                ? ((words[r][k / 4u] >> (8u * (k % 4u))) & 255u)
                                : uint(row_ptr[ulong(b) * 24ul + 8ul + j]);
                            const half2 dm = as_type<half2>(h.x);
                            const uint q = ((byte >> shift) & 15u) | (((h.y >> lane) & 1u) << 4);
                            const float wv = float(dm.x) * float(q) + float(dm.y);
                            for (uint m = 0u; m < MC; ++m) sums[m][r] += wv * sx[m][s_col];
                        }
                    }
                }
            }
            if (c_base + chunk_cols < a.columns) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        if (nax_layout == 0u) {
            const uint b_sub = uint(lane) >> 3u, q_sub = (uint(lane) & 7u) * 4u;
            for (uint c_base = 0u; c_base < a.columns; c_base += 512u) {
                const uint chunk_cols = min(512u, a.columns - c_base);
                stage_moe_input_chunk<MC, NThreads>(x_rows, c_base, chunk_cols, tid, sx, bad);
                if (rows_live > 0u) {
                    const uint chunk_blocks = chunk_cols >> 5u;
                    const uint vec_blocks = chunk_blocks & ~3u;
                    const uint b_start = c_base >> 5u;
                    for (uint b_off = 0u; b_off < vec_blocks; b_off += 4u) {
                        const uint blk = b_start + b_off + b_sub;
                        const uint s_off = (b_off + b_sub) * 32u + q_sub;
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            if (r < rows_live) {
                                device const uchar* enc = data + ulong(r) * a.row_bytes + ulong(blk) * 34ul;
                                float sc = nax_detail::stored_half(enc);
                                if (nax_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                                const float4 wf = float4(char4(*reinterpret_cast<device const packed_char4*>(enc + 2u + q_sub)));
                                for (uint m = 0u; m < MC; ++m) {
                                    const float4 xv = *reinterpret_cast<threadgroup const float4*>(&sx[m][s_off]);
                                    sums[m][r] += sc * dot(wf, xv);
                                }
                            }
                        }
                    }
                    for (uint b_off = vec_blocks; b_off < chunk_blocks; ++b_off) {
                        const uint b = b_start + b_off;
                        const uint s_col = b_off * 32u + uint(lane);
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            if (r < rows_live) {
                                device const uchar* enc = data + ulong(r) * a.row_bytes + ulong(b) * 34ul;
                                float sc = nax_detail::stored_half(enc);
                                if (nax_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                                const float wv = sc * float(as_type<char>(enc[2u + lane]));
                                for (uint m = 0u; m < MC; ++m) sums[m][r] += wv * sx[m][s_col];
                            }
                        }
                    }
                }
                if (c_base + chunk_cols < a.columns) threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            for (uint run = 0u; run < full; ++run) {
                stage_moe_input_chunk<MC, NThreads>(x_rows, run * 512u, 512u, tid, sx, bad);
                if (rows_live > 0u) {
                    uint4 words[NR];
                    float sc_lane[NR];
                    #pragma unroll
                    for (uint r = 0u; r < NR; ++r) {
                        if (r < rows_live) {
                            device const uchar* row_ptr = data + ulong(r) * a.row_bytes;
                            words[r] = uint4(*reinterpret_cast<device const packed_uint4*>(row_ptr + ulong(blocks) * 2ul + ulong(run) * 512ul + ulong(lane) * 16ul));
                            if (lane < 16u) {
                                const ushort sb = *reinterpret_cast<device const ushort*>(row_ptr + ulong(run * 16u + uint(lane)) * 2ul);
                                if ((sb & 0x7c00u) == 0x7c00u) { bad |= 1u; sc_lane[r] = 0.0f; }
                                else sc_lane[r] = float(as_type<half>(sb));
                            } else {
                                sc_lane[r] = 0.0f;
                            }
                        } else {
                            words[r] = uint4(0u); sc_lane[r] = 0.0f;
                        }
                    }
                    #pragma unroll
                    for (uint part = 0u; part < 4u; ++part) {
                        uint w_part[NR];
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            w_part[r] = part == 0u ? words[r].x : (part == 1u ? words[r].y : (part == 2u ? words[r].z : words[r].w));
                        }
                        #pragma unroll
                        for (uint k = 0u; k < 4u; ++k) {
                            const uint b_local = part * 4u + k;
                            const uint s_col = b_local * 32u + uint(lane);
                            float xv[MC];
                            for (uint m = 0u; m < MC; ++m) xv[m] = sx[m][s_col];
                            #pragma unroll
                            for (uint r = 0u; r < NR; ++r) {
                                if (r < rows_live) {
                                    const float sc = simd_broadcast(sc_lane[r], ushort(b_local));
                                    const float wv = sc * float(as_type<char>(uchar((w_part[r] >> (8u * k)) & 255u)));
                                    for (uint m = 0u; m < MC; ++m) sums[m][r] += wv * xv[m];
                                }
                            }
                        }
                    }
                }
                if (run + 1u < full || rest > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (rest > 0u) {
                stage_moe_input_chunk<MC, NThreads>(x_rows, full * 512u, rest * 32u, tid, sx, bad);
                if (rows_live > 0u) {
                    for (uint k = 0u; k < rest; ++k) {
                        const uint b = full * 16u + k;
                        const uint s_col = k * 32u + uint(lane);
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            if (r < rows_live) {
                                device const uchar* row_ptr = data + ulong(r) * a.row_bytes;
                                float sc = nax_detail::stored_half(row_ptr + ulong(b) * 2ul);
                                if (nax_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                                const uchar qb = row_ptr[ulong(blocks) * 2ul + ulong(full) * 512ul + ulong(lane) * rest + k];
                                const float wv = sc * float(as_type<char>(qb));
                                for (uint m = 0u; m < MC; ++m) sums[m][r] += wv * sx[m][s_col];
                            }
                        }
                    }
                }
            }
        }
    }

    if (rows_live > 0u) {
        for (uint m = 0u; m < MC; ++m) {
            device float* out_row = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(routes[m]) * a.output_stride_bytes);
            #pragma unroll
            for (uint r = 0u; r < NR; ++r) {
                const float tot = simd_sum(sums[m][r]);
                if (!lane && r < rows_live) {
                    bad |= nax_detail::nonfinite_bit(tot);
                    out_row[first_row + r] = tot;
                }
            }
        }
    }
}

} // namespace nax_verify_union_detail

kernel void nax_verify_moe_build_union(device const uint* ids [[buffer(0)]], device uint* meta [[buffer(1)]],
    constant nax_verify_union_detail::NaxVerifyMoeUnionArgs& u [[buffer(2)]], device atomic_uint* error [[buffer(7)]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup ushort expert_tile[512];
    for (ushort i = lane; i < 512u; i += 32u) expert_tile[i] = 0xffffu;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (!lane) {
        if (!u.tokens || u.tokens > 8u || !u.selected || u.selected > 16u || !u.experts || u.experts > 512u) {
            meta[0] = 0u; meta[1] = 0u; meta[2] = 1u; meta[3] = 0u;
            meta[4] = 0u; meta[5] = 0u; meta[6] = 1u; meta[7] = 0u;
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            return;
        }
        const uint max_routes = (u.reserved0 >= 1u && u.reserved0 <= 8u) ? u.reserved0 : 8u;
        const uint routes = u.tokens * u.selected;
        uint num_tiles = 0u;
        uint bad = 0u;
        for (uint r = 0u; r < routes; ++r) {
            const uint ex = ids[r];
            if (ex >= u.experts) { bad |= 4u; continue; }
            ushort t = expert_tile[ex];
            if (t == 0xffffu || meta[8u + uint(t) * 10u + 1u] == max_routes) {
                t = ushort(num_tiles++);
                expert_tile[ex] = t;
                device uint* rec = meta + 8u + uint(t) * 10u;
                rec[0] = ex;
                rec[1] = 1u;
                rec[2] = r;
            } else {
                device uint* rec = meta + 8u + uint(t) * 10u;
                const uint cnt = rec[1];
                rec[2u + cnt] = r;
                rec[1] = cnt + 1u;
            }
        }
        meta[0] = u.grid_x_gate_up; meta[1] = num_tiles; meta[2] = 1u; meta[3] = 0u;
        meta[4] = u.grid_x_down;    meta[5] = num_tiles; meta[6] = 1u; meta[7] = 0u;
        if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
    }
}

kernel void nax_verify_moe_gate_up_swiglu(device const uchar* gate_weights [[buffer(0)]], device const uchar* up_weights [[buffer(1)]],
    device const float* input [[buffer(2)]], device float* output [[buffer(3)]],
    constant nax_detail::NaxVerifyMoeArgs& a [[buffer(4)]], device const uint* meta [[buffer(5)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup float sx[8][512];
    const uint block_cols = (nax_encoding == 12u || nax_encoding == 13u) ? 256u : 32u;
    const uint block_bytes = nax_encoding == 12u ? 144u : (nax_encoding == 13u ? 176u : (nax_encoding == 8u ? 34u : 24u));
    if (!a.tokens || a.tokens > 8u || !a.selected || a.selected > 16u || !a.experts || a.experts > 512u ||
        !a.columns || a.columns % block_cols != 0u || !a.rows || a.row_bytes != (a.columns / block_cols) * block_bytes ||
        a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        a.expert_stride_bytes != ulong(a.row_bytes) * a.rows) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint num_tiles = meta[1];
    if (group.y >= num_tiles || group.x * 8u >= a.rows) return;
    device const uint* rec = meta + 8u + group.y * 10u;
    const uint expert = rec[0], match_count = rec[1];
    if (expert >= a.experts || !match_count || match_count > 8u) {
        if (!tid && !group.x) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint first_row = (group.x * 4u + uint(sg)) * 2u;
    device const uchar* gate_mat = gate_weights + ulong(expert) * a.expert_stride_bytes;
    device const uchar* up_mat = up_weights + ulong(expert) * a.expert_stride_bytes;
    uint bad = 0u;
    switch (match_count) {
        case 1u: nax_verify_union_detail::moe_gate_up_pass<1, 2>(gate_mat, up_mat, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 2u: nax_verify_union_detail::moe_gate_up_pass<2, 2>(gate_mat, up_mat, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 3u: nax_verify_union_detail::moe_gate_up_pass<3, 2>(gate_mat, up_mat, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 4u: nax_verify_union_detail::moe_gate_up_pass<4, 2>(gate_mat, up_mat, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 5u: nax_verify_union_detail::moe_gate_up_pass<5, 2>(gate_mat, up_mat, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 6u: nax_verify_union_detail::moe_gate_up_pass<6, 2>(gate_mat, up_mat, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 7u: nax_verify_union_detail::moe_gate_up_pass<7, 2>(gate_mat, up_mat, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        default: nax_verify_union_detail::moe_gate_up_pass<8, 2>(gate_mat, up_mat, input, output, a, rec, first_row, tid, lane, sx, bad); break;
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_verify_moe_down(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::NaxVerifyMoeArgs& a [[buffer(3)]],
    device const uint* meta [[buffer(4)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup float sx[8][512];
    const uint block_cols = (nax_encoding == 12u || nax_encoding == 13u) ? 256u : 32u;
    const uint block_bytes = nax_encoding == 12u ? 144u : (nax_encoding == 13u ? 176u : (nax_encoding == 8u ? 34u : 24u));
    if (!a.tokens || a.tokens > 8u || !a.selected || a.selected > 16u || !a.experts || a.experts > 512u ||
        !a.columns || a.columns % block_cols != 0u || !a.rows || a.row_bytes != (a.columns / block_cols) * block_bytes ||
        a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        a.expert_stride_bytes != ulong(a.row_bytes) * a.rows) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint num_tiles = meta[5];
    if (group.y >= num_tiles || group.x * 16u >= a.rows) return;
    device const uint* rec = meta + 8u + group.y * 10u;
    const uint expert = rec[0], match_count = rec[1];
    if (expert >= a.experts || !match_count || match_count > 8u) {
        if (!tid && !group.x) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint first_row = (group.x * 4u + uint(sg)) * 4u;
    device const uchar* matrix = weights + ulong(expert) * a.expert_stride_bytes;
    uint bad = 0u;
    switch (match_count) {
        case 1u: nax_verify_union_detail::moe_down_pass<1, 4>(matrix, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 2u: nax_verify_union_detail::moe_down_pass<2, 4>(matrix, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 3u: nax_verify_union_detail::moe_down_pass<3, 4>(matrix, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 4u: nax_verify_union_detail::moe_down_pass<4, 4>(matrix, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 5u: nax_verify_union_detail::moe_down_pass<5, 4>(matrix, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 6u: nax_verify_union_detail::moe_down_pass<6, 4>(matrix, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        case 7u: nax_verify_union_detail::moe_down_pass<7, 4>(matrix, input, output, a, rec, first_row, tid, lane, sx, bad); break;
        default: nax_verify_union_detail::moe_down_pass<8, 4>(matrix, input, output, a, rec, first_row, tid, lane, sx, bad); break;
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_router_f32(device const float* weights [[buffer(0)]], device bfloat* x_bf16 [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::MoeRouterWindowArgs& p [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort RowsPerSG = 16, Simdgroups = 4, Rows = RowsPerSG * Simdgroups, Cols = 32, KS = 32, StageStride = 40, Parts = 3;
    constexpr uint Plane = uint(Cols) * StageStride, Buffer = Parts * Plane;
    alignas(16) threadgroup bfloat stage[2 * Buffer];
    const ushort tid = sg * 32 + lane;
    if (p.columns != 2560 || p.experts != 512 || !p.tokens || p.tokens > 2048 || p.reserved || p.output_stride_bytes < 2048 || p.output_stride_bytes % 16) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint col0 = group.x * Cols;
    const uint row0 = group.y * Rows + sg * RowsPerSG;
    if (col0 >= p.experts || group.y * Rows >= p.tokens) return;

    const bool owns_rows = row0 < p.tokens;
    const ushort col = tid >> 2, first_k = (tid & 3) * 8;
    const bool live_col = col0 + col < p.experts;
    device const float4* w_ptr = reinterpret_cast<device const float4*>(weights + ulong(live_col ? col0 + col : 0) * p.columns + first_k);

    device bfloat* sg_input = x_bf16 + ulong(owns_rows ? row0 : 0) * p.columns;
    auto a = tensor(sg_input, dextents<int, 2>{int(p.columns), RowsPerSG}, array<int, 2>{1, int(p.columns)});
    constexpr auto descriptor = matmul2d_descriptor(RowsPerSG, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "exact Sterbenz BF16/FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    const auto make_plane = [&](uint buf_idx, uint part) {
        tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> b(
            stage + buf_idx * Buffer + part * Plane, dextents<int, 2>{KS, Cols}, array<int, 2>{1, StageStride});
        return b.slice<KS, Cols>(0, 0);
    };
    auto h0 = make_plane(0, 0), m0 = make_plane(0, 1), l0 = make_plane(0, 2);
    auto h1 = make_plane(1, 0), m1 = make_plane(1, 1), l1 = make_plane(1, 2);
    auto a0 = a.template slice<KS, RowsPerSG>(0, 0);
    auto high = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(h0), float>();
    auto low = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(h0), float>();
    #pragma unroll
    for (ushort i = 0; i < high.get_capacity(); ++i) high[i] = low[i] = 0.0f;

    uint bad = 0;
    const uint steps = p.columns / KS;
    float4 next[2];
    next[0] = live_col ? w_ptr[0] : float4(0);
    next[1] = live_col ? w_ptr[1] : float4(0);

    for (uint step = 0; step < steps; ++step) {
        threadgroup ushort* parts = reinterpret_cast<threadgroup ushort*>(stage + (step & 1) * Buffer) + uint(col) * StageStride + first_k;
        #pragma unroll
        for (ushort j = 0; j < 2; ++j) {
            const ushort3 sx = nax_detail::split_f32_bf16(next[j].x, bad);
            const ushort3 sy = nax_detail::split_f32_bf16(next[j].y, bad);
            const ushort3 sz = nax_detail::split_f32_bf16(next[j].z, bad);
            const ushort3 sw = nax_detail::split_f32_bf16(next[j].w, bad);
            #pragma unroll
            for (ushort part = 0; part < Parts; ++part)
                *reinterpret_cast<threadgroup ushort4*>(parts + part * Plane + 4 * j) = ushort4(sx[part], sy[part], sz[part], sw[part]);
        }
        if (step + 1 < steps && live_col) {
            next[0] = w_ptr[(step + 1) * 8 + 0];
            next[1] = w_ptr[(step + 1) * 8 + 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (owns_rows) {
            auto a_slice = a.template slice<KS, RowsPerSG>(int(step * KS), 0);
            if (step & 1) {
                operation.run(a_slice, h1, high);
                operation.run(a_slice, m1, low);
                operation.run(a_slice, l1, low);
            } else {
                operation.run(a_slice, h0, high);
                operation.run(a_slice, m0, low);
                operation.run(a_slice, l0, low);
            }
        }
    }

    if (owns_rows) {
        const ulong out_stride = p.output_stride_bytes / sizeof(float);
        #pragma unroll
        for (ushort i = 0; i < high.get_capacity(); ++i) {
            if (!high.is_valid_element(i)) continue;
            const auto idx = high.get_multidimensional_index(i);
            const uint col_out = col0 + uint(idx[0]), row_out = row0 + uint(idx[1]);
            if (row_out < p.tokens && col_out < p.experts) {
                const float val = float(high[i]) + float(low[i]);
                bad |= nax_detail::nonfinite_bit(val);
                output[ulong(row_out) * out_stride + col_out] = val;
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_grouped_routes(device const uint* indices [[buffer(0)]],
    device uint* token_map [[buffer(1)]], device uint* counts [[buffer(2)]],
    constant uint4& s [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint expert = group * 8u + uint(sg);
    if (expert >= s.x) return;
    uint accumulated = 0u;
    uint bad = 0u;
    for (uint base = 0u; base < s.z; base += 32u) {
        const uint token = base + uint(lane);
        uint slot = UINT_MAX, matches = 0u;
        if (token < s.z) {
            const uint row_off = token * s.y;
            for (uint i = 0u; i < s.y; ++i) {
                const uint id = indices[row_off + i];
                if (id >= s.x) bad |= 4u;
                if (id == expert) { slot = i; ++matches; }
            }
        }
        if (matches > 1u) bad |= 4u;
        const uint found = (matches == 1u) ? 1u : 0u;
        const uint prefix = simd_prefix_exclusive_sum(found);
        const uint total = simd_sum(found);
        if (found) token_map[expert * s.z + accumulated + prefix] = token * s.y + slot;
        accumulated += total;
    }
    if (!lane) counts[expert] = accumulated;
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_router_bf16_fast(device const float* weights [[buffer(0)]], device bfloat* x_bf16 [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::MoeRouterWindowArgs& p [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort RowsPerSG = 16, Simdgroups = 4, Rows = RowsPerSG * Simdgroups, Cols = 32, KS = 32;
    constexpr uint Plane = Cols * KS;
    threadgroup bfloat stage[2 * Plane];
    const ushort tid = sg * 32 + lane;
    if (p.columns != 2560 || p.experts != 512 || !p.tokens || p.tokens > 2048 || p.reserved || p.output_stride_bytes < 2048 || p.output_stride_bytes % 16) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint col0 = group.x * Cols;
    const uint row0 = group.y * Rows + sg * RowsPerSG;
    if (col0 >= p.experts || group.y * Rows >= p.tokens) return;

    const bool owns_rows = row0 < p.tokens;
    const ushort col = tid >> 2, first_k = (tid & 3) * 8;
    const bool live_col = col0 + col < p.experts;
    device const float4* w_ptr = reinterpret_cast<device const float4*>(weights + ulong(live_col ? col0 + col : 0) * p.columns + first_k);

    device bfloat* sg_input = x_bf16 + ulong(owns_rows ? row0 : 0) * p.columns;
    auto a = tensor(sg_input, dextents<int, 2>{int(p.columns), RowsPerSG}, array<int, 2>{1, int(p.columns)});
    constexpr auto descriptor = matmul2d_descriptor(RowsPerSG, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> b0_t(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> b1_t(stage + Plane, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
    auto b0 = b0_t.slice<KS, Cols>(0, 0), b1 = b1_t.slice<KS, Cols>(0, 0);
    auto a0 = a.template slice<KS, RowsPerSG>(0, 0);
    auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    uint bad = 0;
    const uint steps = p.columns / KS;
    float4 next[2];
    next[0] = live_col ? w_ptr[0] : float4(0);
    next[1] = live_col ? w_ptr[1] : float4(0);

    for (uint step = 0; step < steps; ++step) {
        threadgroup ushort* dst = reinterpret_cast<threadgroup ushort*>(stage + (step & 1) * Plane) + col * KS + first_k;
        #pragma unroll
        for (ushort j = 0; j < 2; ++j) {
            const ushort bx = nax_detail::round_bf16_bits(next[j].x, bad);
            const ushort by = nax_detail::round_bf16_bits(next[j].y, bad);
            const ushort bz = nax_detail::round_bf16_bits(next[j].z, bad);
            const ushort bw = nax_detail::round_bf16_bits(next[j].w, bad);
            *reinterpret_cast<threadgroup ushort4*>(dst + 4 * j) = ushort4(bx, by, bz, bw);
        }
        if (step + 1 < steps && live_col) {
            next[0] = w_ptr[(step + 1) * 8 + 0];
            next[1] = w_ptr[(step + 1) * 8 + 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (owns_rows) {
            auto a_slice = a.template slice<KS, RowsPerSG>(int(step * KS), 0);
            if (step & 1) operation.run(a_slice, b1, acc);
            else operation.run(a_slice, b0, acc);
        }
    }

    if (owns_rows) {
        const ulong out_stride = p.output_stride_bytes / sizeof(float);
        #pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            const auto idx = acc.get_multidimensional_index(i);
            const uint col_out = col0 + uint(idx[0]), row_out = row0 + uint(idx[1]);
            if (row_out < p.tokens && col_out < p.experts) {
                const float val = float(acc[i]);
                bad |= nax_detail::nonfinite_bit(val);
                output[ulong(row_out) * out_stride + col_out] = val;
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

struct NaxGdnChunkParams {
    uint rows, columns, outputs, heads;
    uint key_heads, key_width, value_width, mode;
    uint offset, count, l2_mode, reserved;
    ulong input_stride, output_stride, auxiliary_stride, gate_stride;
    float query_scale, l2_epsilon, rms_epsilon, reserved_float;
};
static_assert(sizeof(NaxGdnChunkParams) == 96, "GDN chunk ABI");

kernel void nax_gdn_proj_gates_f32(device const float* w_alpha [[buffer(0)]],
    device const float* w_beta [[buffer(1)]],
    device bfloat* x_bf16 [[buffer(2)]],
    device const float* rate [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* logg [[buffer(5)]],
    device float* beta [[buffer(6)]],
    constant NaxGdnChunkParams& p [[buffer(14)]],
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort RowsPerSG = 16, Simdgroups = 4, Rows = RowsPerSG * Simdgroups;
    constexpr ushort Cols = 32, HalfCols = 16, KS = 32, StageStride = 36, Parts = 3;
    constexpr uint Plane = uint(Cols) * uint(StageStride), Buffer = uint(Parts) * Plane;
    alignas(16) threadgroup bfloat stage[2 * Buffer];
    const ushort tid = sg * 32u + lane;
    if (!p.rows || p.rows > 2048u || !p.columns || p.columns % 32u != 0u ||
        !p.heads || p.heads != p.outputs || p.gate_stride % 4u != 0u || p.gate_stride < ulong(p.heads) * 4ul) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    const uint h0 = group.x * HalfCols;
    const uint row0 = group.y * Rows + uint(sg) * RowsPerSG;
    if (h0 >= p.heads || group.y * Rows >= p.rows) return;

    const bool owns_rows = row0 < p.rows;
    const ushort col = tid >> 2u;
    const ushort first_k = (tid & 3u) * 8u;
    const bool is_beta = col >= HalfCols;
    const uint h_col = h0 + uint(col & 15u);
    const bool live_col = h_col < p.heads;
    device const float* w_mat = is_beta ? w_beta : w_alpha;
    device const float4* w_ptr = reinterpret_cast<device const float4*>(
        w_mat + ulong(live_col ? h_col : 0u) * p.columns + first_k);

    device bfloat* sg_input = x_bf16 + ulong(owns_rows ? row0 : 0u) * p.columns;
    auto a = tensor(sg_input, dextents<int, 2>{int(p.columns), RowsPerSG}, array<int, 2>{1, int(p.columns)});
    constexpr auto descriptor = matmul2d_descriptor(RowsPerSG, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;

    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> h0_t(stage + 0 * Plane, dextents<int, 2>{KS, Cols}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> m0_t(stage + 1 * Plane, dextents<int, 2>{KS, Cols}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> l0_t(stage + 2 * Plane, dextents<int, 2>{KS, Cols}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> h1_t(stage + Buffer + 0 * Plane, dextents<int, 2>{KS, Cols}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> m1_t(stage + Buffer + 1 * Plane, dextents<int, 2>{KS, Cols}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> l1_t(stage + Buffer + 2 * Plane, dextents<int, 2>{KS, Cols}, array<int, 2>{1, StageStride});
    auto bh0 = h0_t.slice<KS, Cols>(0, 0), bm0 = m0_t.slice<KS, Cols>(0, 0), bl0 = l0_t.slice<KS, Cols>(0, 0);
    auto bh1 = h1_t.slice<KS, Cols>(0, 0), bm1 = m1_t.slice<KS, Cols>(0, 0), bl1 = l1_t.slice<KS, Cols>(0, 0);
    auto a0 = a.template slice<KS, RowsPerSG>(0, 0);
    auto high = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(bh0), float>();
    auto low = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(bh0), float>();
    #pragma unroll
    for (ushort i = 0; i < high.get_capacity(); ++i) {
        high[i] = 0.0f;
        low[i] = 0.0f;
    }

    uint bad = 0u;
    const uint steps = p.columns / KS;
    float4 next[2];
    next[0] = live_col ? w_ptr[0] : float4(0);
    next[1] = live_col ? w_ptr[1] : float4(0);

    for (uint step = 0u; step < steps; ++step) {
        threadgroup ushort* parts = reinterpret_cast<threadgroup ushort*>(stage + (step & 1u) * Buffer) + uint(col) * StageStride + first_k;
        #pragma unroll
        for (ushort j = 0; j < 2; ++j) {
            const ushort3 sx = nax_detail::split_f32_bf16(next[j].x, bad);
            const ushort3 sy = nax_detail::split_f32_bf16(next[j].y, bad);
            const ushort3 sz = nax_detail::split_f32_bf16(next[j].z, bad);
            const ushort3 sw = nax_detail::split_f32_bf16(next[j].w, bad);
            #pragma unroll
            for (ushort part = 0; part < Parts; ++part) {
                *reinterpret_cast<threadgroup ushort4*>(parts + uint(part) * Plane + 4u * j) = ushort4(sx[part], sy[part], sz[part], sw[part]);
            }
        }
        if (step + 1u < steps && live_col) {
            next[0] = w_ptr[(step + 1u) * 8u + 0u];
            next[1] = w_ptr[(step + 1u) * 8u + 1u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (owns_rows) {
            auto a_slice = a.template slice<KS, RowsPerSG>(int(step * KS), 0);
            if (step & 1u) {
                operation.run(a_slice, bh1, high);
                operation.run(a_slice, bm1, low);
                operation.run(a_slice, bl1, low);
            } else {
                operation.run(a_slice, bh0, high);
                operation.run(a_slice, bm0, low);
                operation.run(a_slice, bl0, low);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float (*epi)[16][33] = reinterpret_cast<threadgroup float (*)[16][33]>(stage);
    if (owns_rows) {
        #pragma unroll
        for (ushort i = 0; i < high.get_capacity(); ++i) {
            if (!high.is_valid_element(i)) continue;
            const auto idx = high.get_multidimensional_index(i);
            epi[sg][uint(idx[1])][uint(idx[0])] = float(high[i]) + float(low[i]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const ushort h_local = lane & 15u;
        const ushort r_odd = lane >> 4u;
        const uint h = h0 + uint(h_local);
        if (h < p.heads) {
            const float r_val = rate[h];
            const float d_val = bias[h];
            bad |= nax_detail::nonfinite_bit(r_val) | nax_detail::nonfinite_bit(d_val);
            if (r_val > 0.0f) {
                bad |= 1u;
            } else {
                #pragma unroll
                for (ushort p_sub = 0; p_sub < 8; ++p_sub) {
                    const ushort r_sub = p_sub * 2u + r_odd;
                    const uint row_out = row0 + uint(r_sub);
                    if (row_out < p.rows) {
                        const float a_val = epi[sg][r_sub][h_local];
                        const float b_val = epi[sg][r_sub][16u + h_local];
                        const float sum = a_val + d_val;
                        bad |= nax_detail::nonfinite_bit(a_val) | nax_detail::nonfinite_bit(b_val) | nax_detail::nonfinite_bit(sum);
                        if (isfinite(sum)) {
                            const float z = exp(-abs(sum));
                            const float sp_s = 1.0f + z;
                            const float sp = max(sum, 0.0f) + (sp_s == 1.0f ? z : log(sp_s) * (z / (sp_s - 1.0f)));
                            const float lg = r_val * sp;
                            const float bt = nax_detail::sigmoid_f32(b_val);
                            bad |= nax_detail::nonfinite_bit(lg) | nax_detail::nonfinite_bit(bt);
                            device float* logg_row = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(logg) + ulong(row_out) * p.gate_stride);
                            device float* beta_row = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(beta) + ulong(row_out) * p.gate_stride);
                            logg_row[h] = lg;
                            beta_row[h] = bt;
                        }
                    }
                }
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

kernel void nax_grouped_tasks_indirect(device const uint* counts [[buffer(0)]], device uint* tasks [[buffer(1)]],
    constant uint4& s [[buffer(2)]], device uint* indirect [[buffer(3)]], constant uint2& grid_x [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint offsets[513];
    const uint bound = s.z * s.y / s.w + s.x;
    if (!tid) {
        uint total = 0, routes = 0;
        for (uint e = 0; e < s.x; ++e) {
            offsets[e] = total;
            if (counts[e] > s.z) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            total += (min(counts[e], s.z) + s.w - 1) / s.w;
            routes += counts[e];
        }
        offsets[s.x] = total;
        const uint live_tiles = min(total, bound);
        tasks[0] = live_tiles;
        tasks[1] = routes;
        if (total > bound || routes != s.z * s.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        indirect[0]  = grid_x.x; indirect[1]  = live_tiles; indirect[2]  = 1; indirect[3]  = 0;
        indirect[4]  = grid_x.y; indirect[5]  = live_tiles; indirect[6]  = 1; indirect[7]  = 0;
        indirect[8]  = 4;        indirect[9]  = live_tiles; indirect[10] = 1; indirect[11] = 0;
        indirect[12] = 16;       indirect[13] = live_tiles; indirect[14] = 1; indirect[15] = 0;
        indirect[16] = 10;       indirect[17] = live_tiles; indirect[18] = 1; indirect[19] = 0;
        indirect[20] = 20;       indirect[21] = live_tiles; indirect[22] = 1; indirect[23] = 0;
    }
    for (uint i = tid; i < bound; i += 128) { tasks[2 + 2 * i] = UINT_MAX; tasks[3 + 2 * i] = UINT_MAX; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < s.x; e += 128) {
        const uint n = (min(counts[e], s.z) + s.w - 1) / s.w;
        for (uint i = 0; i < n; ++i) if (offsets[e] + i < bound) {
            tasks[2 + 2 * (offsets[e] + i)] = e;
            tasks[3 + 2 * (offsets[e] + i)] = i * s.w;
        }
    }
}

kernel void nax_moe_gather_bf16(device const float* input [[buffer(0)]], device bfloat* gathered_x [[buffer(1)]],
    constant nax_detail::GroupedMmArgs& a [[buffer(2)]], device const uint* ids [[buffer(3)]],
    device const uint* map [[buffer(4)]], device const uint* tasks [[buffer(5)]], device const uint* counts [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    if (tasks[0] > a.maximum_tasks || tasks[1] != a.tokens * a.selected) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.y >= tasks[0]) return;
    const uint expert = tasks[2 + 2 * group.y], start = tasks[3 + 2 * group.y];
    if (expert >= a.experts || start >= a.tokens || counts[expert] > a.tokens || start >= counts[expert]) {
        if (!tid && !group.x) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint n = min(32u, counts[expert] - start);
    threadgroup uint rids[32];
    uint bad = 0;
    if (tid < 32u) {
        rids[tid] = tid < n ? nax_detail::route_index(ids, map, expert, start + tid, a, bad) : UINT_MAX;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint quads = a.columns / 4;
    const ulong stride_floats = a.input_stride_bytes / sizeof(float);
    const ushort padded_n = ushort((n + 7u) & ~7u);
    for (ushort r = 0; r < padded_n; ++r) {
        device bfloat4* dst_row = reinterpret_cast<device bfloat4*>(gathered_x + (ulong(group.y) * 32 + r) * a.columns);
        const uint rid = rids[r];
        if (rid != UINT_MAX) {
            const uint input_row = (a.broadcast & 1u) ? rid / a.selected : rid;
            device const float* src_row = input + ulong(input_row) * stride_floats;
            for (uint col4 = group.x * 128 + tid; col4 < quads; col4 += 512) {
                const float4 src = float4(*reinterpret_cast<device const packed_float4*>(src_row + col4 * 4u));
                const bfloat b0 = nax_detail::clean_bf16(src.x, bad);
                const bfloat b1 = nax_detail::clean_bf16(src.y, bad);
                const bfloat b2 = nax_detail::clean_bf16(src.z, bad);
                const bfloat b3 = nax_detail::clean_bf16(src.w, bad);
                dst_row[col4] = bfloat4(b0, b1, b2, b3);
            }
        } else {
            for (uint col4 = group.x * 128 + tid; col4 < quads; col4 += 512) dst_row[col4] = bfloat4(0);
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

namespace nax_grouped_moe_detail {

static inline void unpack_qk_scales_pair(uint3 sc, ushort sub_pair, thread uchar2& sc0, thread uchar2& sc1) {
    if (sub_pair == 0u) {
        sc0 = uchar2(uchar(sc.x & 63u), uchar(sc.y & 63u));
        sc1 = uchar2(uchar((sc.x >> 8u) & 63u), uchar((sc.y >> 8u) & 63u));
    } else if (sub_pair == 1u) {
        sc0 = uchar2(uchar((sc.x >> 16u) & 63u), uchar((sc.y >> 16u) & 63u));
        sc1 = uchar2(uchar((sc.x >> 24u) & 63u), uchar((sc.y >> 24u) & 63u));
    } else if (sub_pair == 2u) {
        const uint q0 = sc.x & 0xFFu, q1 = (sc.x >> 8u) & 0xFFu;
        const uint q4 = sc.y & 0xFFu, q5 = (sc.y >> 8u) & 0xFFu;
        const uint q8 = sc.z & 0xFFu, q9 = (sc.z >> 8u) & 0xFFu;
        sc0 = uchar2(uchar((q8 & 0xFu) | ((q0 & 0xC0u) >> 2u)), uchar((q8 >> 4u) | ((q4 & 0xC0u) >> 2u)));
        sc1 = uchar2(uchar((q9 & 0xFu) | ((q1 & 0xC0u) >> 2u)), uchar((q9 >> 4u) | ((q5 & 0xC0u) >> 2u)));
    } else {
        const uint q2 = (sc.x >> 16u) & 0xFFu, q3 = (sc.x >> 24u) & 0xFFu;
        const uint q6 = (sc.y >> 16u) & 0xFFu, q7 = (sc.y >> 24u) & 0xFFu;
        const uint q10 = (sc.z >> 16u) & 0xFFu, q11 = (sc.z >> 24u) & 0xFFu;
        sc0 = uchar2(uchar((q10 & 0xFu) | ((q2 & 0xC0u) >> 2u)), uchar((q10 >> 4u) | ((q6 & 0xC0u) >> 2u)));
        sc1 = uchar2(uchar((q11 & 0xFu) | ((q3 & 0xC0u) >> 2u)), uchar((q11 >> 4u) | ((q7 & 0xC0u) >> 2u)));
    }
}

static inline void decode4_qk_pair(device const uchar* row, uint pair, ushort sub, bool live_col,
    thread half4& w_even, thread half4& w_odd, thread uint& bad) {
    if (!live_col) {
        w_even = half4(0);
        w_odd = half4(0);
        return;
    }
    const uint sb_idx = pair >> 2u;
    const uint sub_pair = pair & 3u;
    const short is = short(sub_pair * 2u);
    const uint byte_off = sub_pair * 32u + uint(sub) * 4u;
    if (nax_encoding == 12u) {
        device const nax_detail::block_q4_K* xb = reinterpret_cast<device const nax_detail::block_q4_K*>(row) + sb_idx;
        const uchar2 sc0 = nax_detail::scale_min_k4(is, 0, xb->scales);
        const uchar2 sc1 = nax_detail::scale_min_k4(is, 1, xb->scales);
        float stored_d = float(xb->d), min_v = float(xb->dmin);
        if (nax_detail::nonfinite_bit(stored_d) || nax_detail::nonfinite_bit(min_v)) { bad |= 1u; stored_d = 0.0f; min_v = 0.0f; }
        const float dl0 = stored_d * float(sc0[0]), ml0 = min_v * float(sc0[1]);
        const float dl1 = (stored_d / 16.0f) * float(sc1[0]), ml1 = min_v * float(sc1[1]);
        const uchar4 raw_q = uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->qs + byte_off));
        w_even = half4(dl0 * float4(raw_q & uchar4(0x0Fu)) - ml0);
        w_odd = half4(dl1 * float4(raw_q & uchar4(0xF0u)) - ml1);
    } else {
        device const nax_detail::block_q5_K* xb = reinterpret_cast<device const nax_detail::block_q5_K*>(row) + sb_idx;
        const uchar2 sc0 = nax_detail::scale_min_k4(is, 0, xb->scales);
        const uchar2 sc1 = nax_detail::scale_min_k4(is, 1, xb->scales);
        float stored_d = float(xb->d), min_v = float(xb->dmin);
        if (nax_detail::nonfinite_bit(stored_d) || nax_detail::nonfinite_bit(min_v)) { bad |= 1u; stored_d = 0.0f; min_v = 0.0f; }
        const float dl0 = stored_d * float(sc0[0]), ml0 = min_v * float(sc0[1]);
        const float dl1 = (stored_d / 16.0f) * float(sc1[0]), ml1 = min_v * float(sc1[1]);
        const uchar4 raw_q = uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->qs + byte_off));
        const uchar4 raw_qh = uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->qh + uint(sub) * 4u));
        const uchar ul0 = uchar(1u << (2u * sub_pair));
        const uchar ul1 = uchar(1u << (2u * sub_pair + 1u));
        const float4 hi0 = select(float4(0.0f), float4(16.0f), (raw_qh & uchar4(ul0)) != uchar4(0u));
        const float4 hi1 = select(float4(0.0f), float4(256.0f), (raw_qh & uchar4(ul1)) != uchar4(0u));
        w_even = half4(dl0 * (float4(raw_q & uchar4(0x0Fu)) + hi0) - ml0);
        w_odd = half4(dl1 * (float4(raw_q & uchar4(0xF0u)) + hi1) - ml1);
    }
    bad |= nax_detail::nonfinite_half_bit(w_even.x) | nax_detail::nonfinite_half_bit(w_even.y) |
           nax_detail::nonfinite_half_bit(w_even.z) | nax_detail::nonfinite_half_bit(w_even.w) |
           nax_detail::nonfinite_half_bit(w_odd.x) | nax_detail::nonfinite_half_bit(w_odd.y) |
           nax_detail::nonfinite_half_bit(w_odd.z) | nax_detail::nonfinite_half_bit(w_odd.w);
}

template <bool IsGateUp, ushort StageStride, typename OpT, typename ATensorT, typename BSliceT, typename AccT>
static inline void run_grouped_moe_mma(device const uchar* w0_expert, device const uchar* w1_expert,
    uint first_col, constant nax_detail::GroupedMmArgs& a, bool sg_has_rows,
    ushort tid, ushort sg, ushort lane, threadgroup half* stage,
    thread OpT& op, thread ATensorT& a_tensor,
    thread BSliceT& b0_lo, thread BSliceT& b0_hi, thread BSliceT& b1_lo, thread BSliceT& b1_hi,
    thread AccT& c_lo, thread AccT& c_hi, thread uint& bad) {
    constexpr ushort KS = 32, SgRows = 8;
    constexpr uint PlaneElems = 64u * uint(StageStride);
    const auto col_live = [&](ushort c) -> bool {
        const uint local_c = IsGateUp ? (c < 32u ? uint(c) : uint(c - 32u)) : uint(c);
        return first_col + local_c < a.rows;
    };
    const auto col_row_ptr = [&](ushort c) -> device const uchar* {
        const uint local_c = IsGateUp ? (c < 32u ? uint(c) : uint(c - 32u)) : uint(c);
        const bool live = first_col + local_c < a.rows;
        device const uchar* base = (IsGateUp && c >= 32u) ? w1_expert : w0_expert;
        return base + ulong(live ? (first_col + local_c) : 0u) * a.row_bytes;
    };
    const auto mma_step = [&](uint s_idx) {
        if (!sg_has_rows) return;
        const int k_off = int(s_idx * KS);
        auto a_s = a_tensor.template slice<KS, SgRows>(k_off, 0);
        if (s_idx & 1u) {
            op.run(a_s, b1_lo, c_lo);
            op.run(a_s, b1_hi, c_hi);
        } else {
            op.run(a_s, b0_lo, c_lo);
            op.run(a_s, b0_hi, c_hi);
        }
    };
    const auto finish_staged_step = [&](uint s_idx) {
        if (s_idx > 0u) mma_step(s_idx - 1u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    };

    const uint steps = a.columns / KS;
    if (nax_layout == 2u) {
        const uint blocks = a.columns / 32u;
        const ushort j = lane >> 1u, odd = lane & 1u;
        const ushort h_row_sub = lane >> 2u, h_sub = lane & 3u;
        uint step = 0u;
        for (; step + 16u <= steps; step += 16u) {
            const uint run = step / 16u;
            const ulong quant_base = 8ul * ulong(blocks) + ulong(run) * 256ul + ulong(lane) * 8ul;
            uint2 q_lo8[8], q_hi8[8];
            #pragma unroll
            for (ushort m = 0; m < 8; ++m) {
                const ushort c0 = sg * 16u + 2u * m;
                const ushort c1 = c0 + 1u;
                const uint2 v0 = col_live(c0)
                    ? as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(col_row_ptr(c0) + quant_base)))
                    : uint2(0u);
                const uint2 v1 = col_live(c1)
                    ? as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(col_row_ptr(c1) + quant_base)))
                    : uint2(0u);
                const uint2 send = odd ? v0 : v1;
                const uint2 recv = simd_shuffle_xor(send, 1u);
                q_lo8[m] = odd ? recv : v0;
                q_hi8[m] = odd ? v1 : recv;
            }
            #pragma unroll
            for (ushort grp = 0; grp < 4; ++grp) {
                const uint h_blk = step + uint(grp) * 4u + uint(h_sub);
                const ushort hc_lo = sg * 16u + h_row_sub;
                const ushort hc_hi = hc_lo + 8u;
                uint2 hdr_lo = uint2(0u), hdr_hi = uint2(0u);
                if (col_live(hc_lo)) {
                    uint2 h = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(col_row_ptr(hc_lo) + ulong(h_blk) * 8ul)));
                    const half2 dm_chk = as_type<half2>(h.x);
                    if (nax_detail::nonfinite_half_bit(dm_chk.x) || nax_detail::nonfinite_half_bit(dm_chk.y)) { bad |= 1u; h.x = 0u; }
                    hdr_lo = h;
                }
                if (col_live(hc_hi)) {
                    uint2 h = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(col_row_ptr(hc_hi) + ulong(h_blk) * 8ul)));
                    const half2 dm_chk = as_type<half2>(h.x);
                    if (nax_detail::nonfinite_half_bit(dm_chk.x) || nax_detail::nonfinite_half_bit(dm_chk.y)) { bad |= 1u; h.x = 0u; }
                    hdr_hi = h;
                }
                #pragma unroll
                for (ushort sub_in_4 = 0; sub_in_4 < 4; ++sub_in_4) {
                    const ushort sub = grp * 4u + sub_in_4;
                    const uint cur_s = step + uint(sub);
                    threadgroup half* stage_s = stage + (cur_s & 1u) * PlaneElems;
                    const ushort word_idx = (sub & 7u) >> 2u;
                    const uint byte_shift = (uint(sub) & 3u) * 8u;
                    #pragma unroll
                    for (ushort m = 0; m < 8; ++m) {
                        const ushort col_c = sg * 16u + 2u * m + odd;
                        const ushort src_lane = ushort(((2u * (m & 3u) + odd) << 2u) + sub_in_4);
                        const uint2 hdr_src = m < 4u ? hdr_lo : hdr_hi;
                        const uint dhx = simd_shuffle(hdr_src.x, src_lane);
                        const uint dhy = simd_shuffle(hdr_src.y, src_lane);
                        const half2 dm = as_type<half2>(dhx);
                        const float d = float(dm.x), mn = float(dm.y);
                        const uint q_word = sub < 8u ? (word_idx ? q_lo8[m].y : q_lo8[m].x) : (word_idx ? q_hi8[m].y : q_hi8[m].x);
                        const uint byte = (q_word >> byte_shift) & 255u;
                        const uint q_lo = (byte & 15u) | (((dhy >> uint(j)) & 1u) << 4u);
                        const uint q_hi = ((byte >> 4u) & 15u) | (((dhy >> (uint(j) + 16u)) & 1u) << 4u);
                        const half h_lo = half(d * float(q_lo) + mn);
                        const half h_hi = half(d * float(q_hi) + mn);
                        bad |= nax_detail::nonfinite_half_bit(h_lo) | nax_detail::nonfinite_half_bit(h_hi);
                        stage_s[uint(col_c) * StageStride + uint(j)] = h_lo;
                        stage_s[uint(col_c) * StageStride + 16u + uint(j)] = h_hi;
                    }
                    finish_staged_step(cur_s);
                }
            }
        }
        for (; step + 4u <= steps; step += 4u) {
            const uint run = step / 16u;
            const uint width = min(16u, blocks - run * 16u);
            const ulong quant_off = 8ul * ulong(blocks) + ulong(run) * 256ul + ulong(j) * ulong(width) + ulong(step % 16u);
            uint q4_cache[8];
            #pragma unroll
            for (ushort m = 0; m < 8; ++m) {
                const ushort col_c = sg * 16u + 2u * m + odd;
                q4_cache[m] = col_live(col_c)
                    ? as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(col_row_ptr(col_c) + quant_off)))
                    : 0u;
            }
            const uint h_blk = step + uint(h_sub);
            const ushort hc_lo = sg * 16u + h_row_sub;
            const ushort hc_hi = hc_lo + 8u;
            uint2 hdr_lo = uint2(0u), hdr_hi = uint2(0u);
            if (col_live(hc_lo)) {
                uint2 h = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(col_row_ptr(hc_lo) + ulong(h_blk) * 8ul)));
                const half2 dm_chk = as_type<half2>(h.x);
                if (nax_detail::nonfinite_half_bit(dm_chk.x) || nax_detail::nonfinite_half_bit(dm_chk.y)) { bad |= 1u; h.x = 0u; }
                hdr_lo = h;
            }
            if (col_live(hc_hi)) {
                uint2 h = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(col_row_ptr(hc_hi) + ulong(h_blk) * 8ul)));
                const half2 dm_chk = as_type<half2>(h.x);
                if (nax_detail::nonfinite_half_bit(dm_chk.x) || nax_detail::nonfinite_half_bit(dm_chk.y)) { bad |= 1u; h.x = 0u; }
                hdr_hi = h;
            }
            #pragma unroll
            for (ushort sub = 0; sub < 4; ++sub) {
                const uint cur_s = step + uint(sub);
                threadgroup half* stage_s = stage + (cur_s & 1u) * PlaneElems;
                const uint byte_shift = uint(sub) * 8u;
                #pragma unroll
                for (ushort m = 0; m < 8; ++m) {
                    const ushort col_c = sg * 16u + 2u * m + odd;
                    const ushort src_lane = ushort(((2u * (m & 3u) + odd) << 2u) + sub);
                    const uint2 hdr_src = m < 4u ? hdr_lo : hdr_hi;
                    const uint dhx = simd_shuffle(hdr_src.x, src_lane);
                    const uint dhy = simd_shuffle(hdr_src.y, src_lane);
                    const half2 dm = as_type<half2>(dhx);
                    const float d = float(dm.x), mn = float(dm.y);
                    const uint byte = (q4_cache[m] >> byte_shift) & 255u;
                    const uint q_lo = (byte & 15u) | (((dhy >> uint(j)) & 1u) << 4u);
                    const uint q_hi = ((byte >> 4u) & 15u) | (((dhy >> (uint(j) + 16u)) & 1u) << 4u);
                    const half h_lo = half(d * float(q_lo) + mn);
                    const half h_hi = half(d * float(q_hi) + mn);
                    bad |= nax_detail::nonfinite_half_bit(h_lo) | nax_detail::nonfinite_half_bit(h_hi);
                    stage_s[uint(col_c) * StageStride + uint(j)] = h_lo;
                    stage_s[uint(col_c) * StageStride + 16u + uint(j)] = h_hi;
                }
                finish_staged_step(cur_s);
            }
        }
        const ushort col = tid >> 1u, ai = tid & 1u;
        const bool live = col_live(col);
        device const uchar* r_ptr = col_row_ptr(col);
        for (; step < steps; ++step) {
            threadgroup half* stage_s = stage + (step & 1u) * PlaneElems;
            const half4x4 w = nax_detail::decode16_any(r_ptr, step * 2u + uint(ai), a.columns, live, bad);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(stage_s + uint(col) * StageStride + uint(ai) * 16u);
            #pragma unroll
            for (ushort i = 0; i < 4; ++i) dst[i] = w[i];
            finish_staged_step(step);
        }
    } else if (nax_layout == 1u) {
        const uint blocks = a.columns / 32u;
        uint step = 0u;
        if ((a.row_bytes & 3u) == 0u) {
            for (; step + 16u <= steps; step += 16u) {
                const uint run = step / 16u;
                const ulong quant_off = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(lane) * 16ul;
                uint4 q_full[16];
                #pragma unroll
                for (ushort p = 0; p < 16; ++p) {
                    const ushort col_c = sg * 16u + p;
                    q_full[p] = col_live(col_c)
                        ? uint4(*reinterpret_cast<device const packed_uint4*>(col_row_ptr(col_c) + quant_off))
                        : uint4(0u);
                }
                #pragma unroll
                for (ushort grp = 0; grp < 4; ++grp) {
                    half4 my_scales = half4(0);
                    if (lane < 16u) {
                        const ushort sc_col = sg * 16u + lane;
                        if (col_live(sc_col)) {
                            my_scales = half4(*reinterpret_cast<device const packed_half4*>(col_row_ptr(sc_col) + 2ul * ulong(step + uint(grp) * 4u)));
                            if (nax_detail::nonfinite_half_bit(my_scales.x)) { bad |= 1u; my_scales.x = half(0); }
                            if (nax_detail::nonfinite_half_bit(my_scales.y)) { bad |= 1u; my_scales.y = half(0); }
                            if (nax_detail::nonfinite_half_bit(my_scales.z)) { bad |= 1u; my_scales.z = half(0); }
                            if (nax_detail::nonfinite_half_bit(my_scales.w)) { bad |= 1u; my_scales.w = half(0); }
                        }
                    }
                    #pragma unroll
                    for (ushort sub_in_4 = 0; sub_in_4 < 4; ++sub_in_4) {
                        const ushort sub = grp * 4u + sub_in_4;
                        const uint cur_s = step + uint(sub);
                        threadgroup half* stage_s = stage + (cur_s & 1u) * PlaneElems;
                        const half sc_sub = sub_in_4 == 0 ? my_scales.x : (sub_in_4 == 1 ? my_scales.y : (sub_in_4 == 2 ? my_scales.z : my_scales.w));
                        const ushort word_idx = sub >> 2u;
                        const uint byte_shift = (uint(sub) & 3u) * 8u;
                        #pragma unroll
                        for (ushort p = 0; p < 16; ++p) {
                            const float d = float(simd_broadcast(sc_sub, p));
                            const uint word = word_idx == 0 ? q_full[p].x : (word_idx == 1 ? q_full[p].y : (word_idx == 2 ? q_full[p].z : q_full[p].w));
                            const char q = as_type<char>(uchar((word >> byte_shift) & 255u));
                            const half h = half(float(q) * d);
                            bad |= nax_detail::nonfinite_half_bit(h);
                            stage_s[(uint(sg) * 16u + uint(p)) * StageStride + uint(lane)] = h;
                        }
                        finish_staged_step(cur_s);
                    }
                }
            }
        }
        for (; step + 4u <= steps; step += 4u) {
            const uint run = step / 16u;
            const uint width = min(16u, blocks - run * 16u);
            const ulong quant_off = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(lane) * ulong(width) + ulong(step % 16u);
            half4 my_scales = half4(0);
            if (lane < 16u) {
                const ushort sc_col = sg * 16u + lane;
                if (col_live(sc_col)) {
                    my_scales = half4(*reinterpret_cast<device const packed_half4*>(col_row_ptr(sc_col) + 2ul * ulong(step)));
                    if (nax_detail::nonfinite_half_bit(my_scales.x)) { bad |= 1u; my_scales.x = half(0); }
                    if (nax_detail::nonfinite_half_bit(my_scales.y)) { bad |= 1u; my_scales.y = half(0); }
                    if (nax_detail::nonfinite_half_bit(my_scales.z)) { bad |= 1u; my_scales.z = half(0); }
                    if (nax_detail::nonfinite_half_bit(my_scales.w)) { bad |= 1u; my_scales.w = half(0); }
                }
            }
            uint q4_cache[16];
            #pragma unroll
            for (ushort p = 0; p < 16; ++p) {
                const ushort col_c = sg * 16u + p;
                q4_cache[p] = col_live(col_c)
                    ? as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(col_row_ptr(col_c) + quant_off)))
                    : 0u;
            }
            #pragma unroll
            for (ushort sub = 0; sub < 4; ++sub) {
                const uint cur_s = step + uint(sub);
                threadgroup half* stage_s = stage + (cur_s & 1u) * PlaneElems;
                const half sc_sub = sub == 0 ? my_scales.x : (sub == 1 ? my_scales.y : (sub == 2 ? my_scales.z : my_scales.w));
                #pragma unroll
                for (ushort p = 0; p < 16; ++p) {
                    const float d = float(simd_broadcast(sc_sub, p));
                    const char q = as_type<char>(uchar((q4_cache[p] >> (8u * uint(sub))) & 255u));
                    const half h = half(float(q) * d);
                    bad |= nax_detail::nonfinite_half_bit(h);
                    stage_s[(uint(sg) * 16u + uint(p)) * StageStride + uint(lane)] = h;
                }
                finish_staged_step(cur_s);
            }
        }
        const ushort col = tid >> 1u, ai = tid & 1u;
        const bool live = col_live(col);
        device const uchar* r_ptr = col_row_ptr(col);
        for (; step < steps; ++step) {
            threadgroup half* stage_s = stage + (step & 1u) * PlaneElems;
            const half4x4 w = nax_detail::decode16_any(r_ptr, step * 2u + uint(ai), a.columns, live, bad);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(stage_s + uint(col) * StageStride + uint(ai) * 16u);
            #pragma unroll
            for (ushort i = 0; i < 4; ++i) dst[i] = w[i];
            finish_staged_step(step);
        }
    } else if ((nax_encoding == 12u || nax_encoding == 13u) && (steps % 8u == 0u)) {
        const ushort sub = lane & 7u, r_in_pass = tid >> 3u;
        const uint sbs = steps >> 3u;
        for (uint sb = 0u; sb < sbs; ++sb) {
            float2 dm[4];
            uint3 sc_u32[4];
            uchar4 qh4[4];
            uchar4 raw_q[4];
            #pragma unroll
            for (ushort p = 0; p < 4; ++p) {
                const ushort col_c = p * 16u + r_in_pass;
                if (col_live(col_c)) {
                    if (nax_encoding == 12u) {
                        device const nax_detail::block_q4_K* xb = reinterpret_cast<device const nax_detail::block_q4_K*>(col_row_ptr(col_c)) + sb;
                        float stored_d = float(xb->d), min_v = float(xb->dmin);
                        if (nax_detail::nonfinite_bit(stored_d) || nax_detail::nonfinite_bit(min_v)) { bad |= 1u; stored_d = 0.0f; min_v = 0.0f; }
                        dm[p] = float2(stored_d, min_v);
                        sc_u32[p] = uint3(
                            as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->scales))),
                            as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->scales + 4u))),
                            as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->scales + 8u))));
                        qh4[p] = uchar4(0u);
                        raw_q[p] = uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->qs + uint(sub) * 4u));
                    } else {
                        device const nax_detail::block_q5_K* xb = reinterpret_cast<device const nax_detail::block_q5_K*>(col_row_ptr(col_c)) + sb;
                        float stored_d = float(xb->d), min_v = float(xb->dmin);
                        if (nax_detail::nonfinite_bit(stored_d) || nax_detail::nonfinite_bit(min_v)) { bad |= 1u; stored_d = 0.0f; min_v = 0.0f; }
                        dm[p] = float2(stored_d, min_v);
                        sc_u32[p] = uint3(
                            as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->scales))),
                            as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->scales + 4u))),
                            as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->scales + 8u))));
                        qh4[p] = uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->qh + uint(sub) * 4u));
                        raw_q[p] = uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->qs + uint(sub) * 4u));
                    }
                } else {
                    dm[p] = float2(0.0f);
                    sc_u32[p] = uint3(0u);
                    qh4[p] = uchar4(0u);
                    raw_q[p] = uchar4(0u);
                }
            }
            #pragma unroll
            for (ushort sub_pair = 0; sub_pair < 4; ++sub_pair) {
                const uint s_even = sb * 8u + 2u * uint(sub_pair);
                const uint s_odd = s_even + 1u;
                threadgroup half* stage_even = stage + (s_even & 1u) * PlaneElems;
                threadgroup half* stage_odd = stage + (s_odd & 1u) * PlaneElems;
                half4 w_odd[4];
                #pragma unroll
                for (ushort p = 0; p < 4; ++p) {
                    const uint col_c = uint(p) * 16u + uint(r_in_pass);
                    uchar2 sc0, sc1;
                    unpack_qk_scales_pair(sc_u32[p], sub_pair, sc0, sc1);
                    const float dl0 = dm[p].x * float(sc0[0]), ml0 = dm[p].y * float(sc0[1]);
                    const float dl1 = (dm[p].x * (1.0f / 16.0f)) * float(sc1[0]), ml1 = dm[p].y * float(sc1[1]);
                    half4 w_even;
                    if (nax_encoding == 12u) {
                        w_even = half4(dl0 * float4(raw_q[p] & uchar4(0x0Fu)) - ml0);
                        w_odd[p] = half4(dl1 * float4(raw_q[p] & uchar4(0xF0u)) - ml1);
                    } else {
                        const uchar ul0 = uchar(1u << (2u * sub_pair));
                        const uchar ul1 = uchar(1u << (2u * sub_pair + 1u));
                        const float4 hi0 = select(float4(0.0f), float4(16.0f), (qh4[p] & uchar4(ul0)) != uchar4(0u));
                        const float4 hi1 = select(float4(0.0f), float4(256.0f), (qh4[p] & uchar4(ul1)) != uchar4(0u));
                        w_even = half4(dl0 * (float4(raw_q[p] & uchar4(0x0Fu)) + hi0) - ml0);
                        w_odd[p] = half4(dl1 * (float4(raw_q[p] & uchar4(0xF0u)) + hi1) - ml1);
                    }
                    bad |= nax_detail::nonfinite_half_bit(w_even.x) | nax_detail::nonfinite_half_bit(w_even.y) |
                           nax_detail::nonfinite_half_bit(w_even.z) | nax_detail::nonfinite_half_bit(w_even.w) |
                           nax_detail::nonfinite_half_bit(w_odd[p].x) | nax_detail::nonfinite_half_bit(w_odd[p].y) |
                           nax_detail::nonfinite_half_bit(w_odd[p].z) | nax_detail::nonfinite_half_bit(w_odd[p].w);
                    *reinterpret_cast<threadgroup half4*>(stage_even + col_c * StageStride + uint(sub) * 4u) = w_even;
                }
                finish_staged_step(s_even);
                #pragma unroll
                for (ushort p = 0; p < 4; ++p) {
                    const uint col_c = uint(p) * 16u + uint(r_in_pass);
                    *reinterpret_cast<threadgroup half4*>(stage_odd + col_c * StageStride + uint(sub) * 4u) = w_odd[p];
                }
                if (sub_pair + 1u < 4u) {
                    const uint next_byte_off = uint(sub_pair + 1u) * 32u + uint(sub) * 4u;
                    #pragma unroll
                    for (ushort p = 0; p < 4; ++p) {
                        const ushort col_c = p * 16u + r_in_pass;
                        if (col_live(col_c)) {
                            if (nax_encoding == 12u) {
                                device const nax_detail::block_q4_K* xb = reinterpret_cast<device const nax_detail::block_q4_K*>(col_row_ptr(col_c)) + sb;
                                raw_q[p] = uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->qs + next_byte_off));
                            } else {
                                device const nax_detail::block_q5_K* xb = reinterpret_cast<device const nax_detail::block_q5_K*>(col_row_ptr(col_c)) + sb;
                                raw_q[p] = uchar4(*reinterpret_cast<device const packed_uchar4*>(xb->qs + next_byte_off));
                            }
                        } else {
                            raw_q[p] = uchar4(0u);
                        }
                    }
                }
                finish_staged_step(s_odd);
            }
        }
    } else {
        const ushort sub = lane & 7u, r_in_pass = tid >> 3u;
        for (uint step = 0u; step < steps; ++step) {
            threadgroup half* stage_s = stage + (step & 1u) * PlaneElems;
            #pragma unroll
            for (ushort p = 0; p < 4; ++p) {
                const ushort col_c = p * 16u + r_in_pass;
                *reinterpret_cast<threadgroup half4*>(stage_s + uint(col_c) * StageStride + uint(sub) * 4u) =
                    nax_detail::decode4_gguf(col_row_ptr(col_c), step, sub, col_live(col_c), bad);
            }
            finish_staged_step(step);
        }
    }
    if (steps > 0u) mma_step(steps - 1u);
}

} // namespace nax_grouped_moe_detail

kernel void nax_grouped_moe(device const uchar* weights [[buffer(0)]], device bfloat* gathered_x [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device const uint* ids [[buffer(4)]], device const uint* map [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], device const uint* tasks [[buffer(8)]],
    device const uint* counts [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort SgRows = 8, HalfCols = 32, Cols = 64, KS = 32, StageStride = 36;
    constexpr uint PlaneElems = uint(Cols) * StageStride;
    alignas(16) threadgroup half stage[2 * PlaneElems]; // 9,216 B + 128 B rids = 9,344 B (3 TGs / 12 SGs per core).
    threadgroup uint rids[32];

    if (tasks[0] > a.maximum_tasks || tasks[1] != a.tokens * a.selected) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint first_col = group.x * Cols;
    if (group.y >= tasks[0] || first_col >= a.rows) return;
    const uint expert = tasks[2 + 2 * group.y], start = tasks[3 + 2 * group.y];
    if (expert >= a.experts || start >= a.tokens || counts[expert] > a.tokens || start >= counts[expert]) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint n = min(32u, counts[expert] - start);
    uint bad = 0u;
    if (tid < 32u) {
        rids[tid] = tid < n ? nax_detail::route_index(ids, map, expert, start + tid, a, bad) : UINT_MAX;
    }

    const uint sg_row0 = uint(sg) * SgRows;
    const bool sg_has_rows = sg_row0 < n;
    device const uchar* w_expert = weights + ulong(expert) * a.expert_stride_bytes;
    device bfloat* base_in = gathered_x + (ulong(group.y) * 32u + (sg_has_rows ? sg_row0 : 0u)) * a.columns;
    auto a_tensor = tensor(base_in, dextents<int, 2>{int(a.columns), SgRows}, array<int, 2>{1, int(a.columns)});

    constexpr auto descriptor = matmul2d_descriptor(SgRows, HalfCols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> op;

    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0_lo(stage, dextents<int, 2>{KS, HalfCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0_hi(stage + uint(HalfCols) * StageStride, dextents<int, 2>{KS, HalfCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1_lo(stage + PlaneElems, dextents<int, 2>{KS, HalfCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1_hi(stage + PlaneElems + uint(HalfCols) * StageStride, dextents<int, 2>{KS, HalfCols}, array<int, 2>{1, StageStride});
    auto b0_lo = bt0_lo.slice<KS, HalfCols>(0, 0), b0_hi = bt0_hi.slice<KS, HalfCols>(0, 0);
    auto b1_lo = bt1_lo.slice<KS, HalfCols>(0, 0), b1_hi = bt1_hi.slice<KS, HalfCols>(0, 0);
    auto a0 = a_tensor.slice<KS, SgRows>(0, 0);
    auto c_lo = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0_lo), float>();
    auto c_hi = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0_lo), float>();
    if (sg_has_rows) {
        #pragma unroll
        for (ushort i = 0; i < c_lo.get_capacity(); ++i) {
            c_lo[i] = 0.0f;
            c_hi[i] = 0.0f;
        }
    }

    nax_grouped_moe_detail::run_grouped_moe_mma<false, StageStride>(w_expert, w_expert, first_col, a, sg_has_rows,
        tid, sg, lane, stage, op, a_tensor, b0_lo, b0_hi, b1_lo, b1_hi, c_lo, c_hi, bad);

    if (sg_has_rows) {
        const auto write_half = [&](thread const decltype(c_lo)& ch, uint col_bias) {
            #pragma unroll
            for (ushort i = 0; i < ch.get_capacity(); ++i) {
                if (!ch.is_valid_element(i)) continue;
                const auto idx = ch.get_multidimensional_index(i);
                const uint out_col = first_col + col_bias + uint(idx[0]);
                const uint rid = rids[sg_row0 + uint(idx[1])];
                if (rid != UINT_MAX && out_col < a.rows) {
                    const float val = float(ch[i]);
                    bad |= nax_detail::nonfinite_bit(val);
                    device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(rid) * a.output_stride_bytes);
                    y[out_col] = val;
                }
            }
        };
        write_half(c_lo, 0u);
        write_half(c_hi, HalfCols);
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_grouped_moe_gate_up_swiglu(device const uchar* gate_weights [[buffer(0)]],
    device const uchar* up_weights [[buffer(1)]], device bfloat* gathered_x [[buffer(2)]],
    device float* output [[buffer(3)]], constant nax_detail::GroupedMmArgs& a [[buffer(4)]],
    device const uint* ids [[buffer(5)]], device const uint* map [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], device const uint* tasks [[buffer(8)]],
    device const uint* counts [[buffer(9)]], device bfloat* gathered_out [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort SgRows = 8, OutCols = 32, Cols = 64, KS = 32, StageStride = 36;
    constexpr uint PlaneElems = uint(Cols) * StageStride;
    alignas(16) threadgroup half stage[2 * PlaneElems]; // 9,216 B + 128 B rids = 9,344 B (3 TGs / 12 SGs per core).
    threadgroup uint rids[32];

    if (tasks[0] > a.maximum_tasks || tasks[1] != a.tokens * a.selected) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint first_col = group.x * OutCols;
    if (group.y >= tasks[0] || first_col >= a.rows) return;
    const uint expert = tasks[2 + 2 * group.y], start = tasks[3 + 2 * group.y];
    if (expert >= a.experts || start >= a.tokens || counts[expert] > a.tokens || start >= counts[expert]) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint n = min(32u, counts[expert] - start);
    const bool handoff_bf16 = (a.broadcast & 2u) != 0u;
    uint bad = 0u;
    if (!handoff_bf16 && tid < 32u) {
        rids[tid] = tid < n ? nax_detail::route_index(ids, map, expert, start + tid, a, bad) : UINT_MAX;
    }

    const uint sg_row0 = uint(sg) * SgRows;
    const bool sg_has_rows = sg_row0 < n;
    device const uchar* gate_expert = gate_weights + ulong(expert) * a.expert_stride_bytes;
    device const uchar* up_expert = up_weights + ulong(expert) * a.expert_stride_bytes;
    device bfloat* base_in = gathered_x + (ulong(group.y) * 32u + (sg_has_rows ? sg_row0 : 0u)) * a.columns;
    auto a_tensor = tensor(base_in, dextents<int, 2>{int(a.columns), SgRows}, array<int, 2>{1, int(a.columns)});

    constexpr auto descriptor = matmul2d_descriptor(SgRows, OutCols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> op;

    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0_lo(stage, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0_hi(stage + uint(OutCols) * StageStride, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1_lo(stage + PlaneElems, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1_hi(stage + PlaneElems + uint(OutCols) * StageStride, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    auto b0_lo = bt0_lo.slice<KS, OutCols>(0, 0), b0_hi = bt0_hi.slice<KS, OutCols>(0, 0);
    auto b1_lo = bt1_lo.slice<KS, OutCols>(0, 0), b1_hi = bt1_hi.slice<KS, OutCols>(0, 0);
    auto a0 = a_tensor.slice<KS, SgRows>(0, 0);
    auto c_lo = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0_lo), float>();
    auto c_hi = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0_lo), float>();
    if (sg_has_rows) {
        #pragma unroll
        for (ushort i = 0; i < c_lo.get_capacity(); ++i) {
            c_lo[i] = 0.0f;
            c_hi[i] = 0.0f;
        }
    }

    nax_grouped_moe_detail::run_grouped_moe_mma<true, StageStride>(gate_expert, up_expert, first_col, a, sg_has_rows,
        tid, sg, lane, stage, op, a_tensor, b0_lo, b0_hi, b1_lo, b1_hi, c_lo, c_hi, bad);

    if (sg_has_rows) {
        #pragma unroll
        for (ushort i = 0; i < c_lo.get_capacity(); ++i) {
            if (!c_lo.is_valid_element(i)) continue;
            const auto idx = c_lo.get_multidimensional_index(i);
            const uint out_col = first_col + uint(idx[0]);
            const uint tile_row = sg_row0 + uint(idx[1]);
            if (out_col >= a.rows) continue;
            if (handoff_bf16) {
                if (tile_row < n) {
                    const float g_val = float(c_lo[i]);
                    const float u_val = float(c_hi[i]);
                    const float val = (g_val * nax_detail::sigmoid_f32(g_val)) * u_val;
                    bad |= nax_detail::nonfinite_bit(g_val) | nax_detail::nonfinite_bit(u_val);
                    gathered_out[(ulong(group.y) * 32u + tile_row) * a.rows + out_col] = nax_detail::clean_bf16(val, bad);
                } else {
                    gathered_out[(ulong(group.y) * 32u + tile_row) * a.rows + out_col] = bfloat(0);
                }
            } else {
                const uint rid = rids[tile_row];
                if (rid != UINT_MAX) {
                    const float g_val = float(c_lo[i]);
                    const float u_val = float(c_hi[i]);
                    const float val = (g_val * nax_detail::sigmoid_f32(g_val)) * u_val;
                    bad |= nax_detail::nonfinite_bit(g_val) | nax_detail::nonfinite_bit(u_val) | nax_detail::nonfinite_bit(val);
                    device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(rid) * a.output_stride_bytes);
                    y[out_col] = val;
                }
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_ple_gate_rows(device const float* query [[buffer(0)]], device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]], device float* gated [[buffer(3)]], constant uint4& shape [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint branches = shape.x, channels = shape.y, gate_transform = shape.z, rows = shape.w;
    if (gate_transform > 1 || !branches || !channels || !rows) {
        if (!group && !subgroup && !lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint pair = group * 4 + subgroup;
    if (pair >= rows * branches) return;
    const uint row = pair / branches;
    const ulong base = ulong(pair) * channels;
    const ulong val_base = ulong(row) * channels;
    float dot = 0.0f;
    for (uint c = lane; c < channels; c += 32) dot += query[base + c] * key[base + c];
    float score = simd_sum(dot) * rsqrt(float(channels));
    if (nax_detail::nonfinite_bit(score)) {
        if (!lane) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    if (gate_transform && score != 0.0f) score = copysign(sqrt(max(abs(score), 1e-6f)), score);
    const float gate = nax_detail::sigmoid_f32(score);
    uint bad = 0;
    for (uint c = lane; c < channels; c += 32) {
        const float out = gate * value[val_base + c];
        bad |= nax_detail::nonfinite_bit(out);
        gated[base + c] = out;
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

kernel void nax_ple_conv_write_rows(device const float* history [[buffer(0)]], device const float* norm_gated [[buffer(1)]],
    device const float* weights [[buffer(2)]], device const float* streams [[buffer(3)]], device const float* gated [[buffer(4)]],
    device float* next_history [[buffer(5)]], device float* output [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], constant uint4& shape [[buffer(8)]], constant ulong2& strides [[buffer(9)]],
    uint channel [[thread_position_in_grid]]) {
    const uint width = shape.x, K = shape.y, dilation = shape.z, rows = shape.w;
    if (channel >= width) return;
    const uint depth = (K - 1) * dilation;
    uint bad = 0;
    for (uint t = 0; t < depth; ++t) bad |= nax_detail::nonfinite_bit(history[ulong(t) * width + channel]);
    const auto x_at = [&](int idx) -> float {
        return idx < 0 ? history[ulong(int(depth) + idx) * width + channel] : norm_gated[ulong(idx) * width + channel];
    };
    const ulong s_stride = strides.x / sizeof(float), out_stride = strides.y / sizeof(float);
    device const float* w = weights + ulong(channel) * K;
    for (uint r = 0; r < rows; ++r) {
        const float x_cur = norm_gated[ulong(r) * width + channel];
        if (nax_detail::nonfinite_bit(x_cur)) { bad |= 1u; break; }
        float conv = x_cur * w[K - 1];
        for (uint tap = 0; tap + 1 < K; ++tap) {
            conv += x_at(int(r) - int((K - 1 - tap) * dilation)) * w[tap];
        }
        const float s_val = streams[ulong(r) * s_stride + channel];
        const float g_val = gated[ulong(r) * width + channel];
        bad |= nax_detail::nonfinite_bit(conv) | nax_detail::nonfinite_bit(s_val) | nax_detail::nonfinite_bit(g_val);
        const float final_val = s_val + (g_val + conv * nax_detail::sigmoid_f32(conv));
        bad |= nax_detail::nonfinite_bit(final_val);
        output[ulong(r) * out_stride + channel] = final_val;
    }
    for (uint t = 0; t < depth; ++t) {
        const float h_next = x_at(int(rows) - int(depth) + int(t));
        bad |= nax_detail::nonfinite_bit(h_next);
        next_history[ulong(t) * width + channel] = h_next;
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

kernel void nax_mtp_concat_rows(device const float* embedding [[buffer(0)]], device const float* hidden [[buffer(1)]],
    device float* output [[buffer(2)]], constant uint3& s [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
    const uint branches = s.x, channels = s.y, rows = s.z;
    const uint row_elems = branches * 2 * channels;
    if (idx >= rows * row_elems) return;
    const uint row = idx / row_elems, rem = idx % row_elems;
    const uint branch = rem / (2 * channels), ch = rem % (2 * channels);
    const float val = ch < channels ? embedding[ulong(row) * channels + ch]
                                    : hidden[(ulong(row) * branches + branch) * channels + (ch - channels)];
    if (nax_detail::nonfinite_bit(val)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    output[idx] = val;
}

kernel void nax_mtp_rope_store_kv_bf16(device const float* norm_key [[buffer(0)]], device const float* raw_value [[buffer(1)]],
    device const float* frequency [[buffer(2)]], device ushort* keys_bf16 [[buffer(3)]], device ushort* values_bf16 [[buffer(4)]],
    constant uint4& s [[buffer(5)]], constant uint2& mode_rows [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
    const uint heads = s.x, head_width = s.y, rotary_half = s.z, first_pos = s.w;
    const uint mode = mode_rows.x, rows = mode_rows.y;
    const uint kv_width = heads * head_width;
    if (idx >= rows * kv_width) return;
    const uint row = idx / kv_width, rem = idx % kv_width;
    const uint d = rem % head_width, base = row * kv_width + (rem - d);
    uint bad = 0;
    values_bf16[idx] = nax_detail::round_bf16_bits(raw_value[idx], bad);
    if (d < rotary_half) {
        const float freq = frequency[d];
        if (nax_detail::nonfinite_bit(freq) || freq <= 0.0f) {
            atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            return;
        }
        const uint a = base + (mode ? 2 * d : d), b = a + (mode ? 1 : rotary_half);
        const float angle = float(first_pos + row) * freq, c = cos(angle), t = sin(angle);
        const float ka = norm_key[a] * c - norm_key[b] * t;
        const float kb = norm_key[b] * c + norm_key[a] * t;
        keys_bf16[a] = nax_detail::round_bf16_bits(ka, bad);
        keys_bf16[b] = nax_detail::round_bf16_bits(kb, bad);
    } else if (d >= 2 * rotary_half) {
        keys_bf16[idx] = nax_detail::round_bf16_bits(norm_key[idx], bad);
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

kernel void nax_qsa_attention(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* output [[buffer(4)]],
    constant nax_detail::NaxQsaAttentionArgs& p [[buffer(5)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat q_stage[16 * 256];
    threadgroup bfloat kv_stage[32 * 256];
    threadgroup float s_stage[16 * 32];
    threadgroup bfloat p_stage[16 * 32];
    threadgroup float row_max[16];
    threadgroup float row_sum[16];
    threadgroup float row_alpha[16];
    threadgroup uint token_ids[32];
    threadgroup uint tile_abort;

    if (!p.rows || !p.kv_heads || !p.query_heads || p.query_heads % p.kv_heads != 0 ||
        (p.query_heads / p.kv_heads) > 16u || p.head_width != 256u || !p.block_budget ||
        p.q_stride < p.query_heads * 256u || p.id_stride < p.block_budget) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint kv_head = group.x, row = group.y;
    if (kv_head >= p.kv_heads || row >= p.rows) return;

    const uint gqa = p.query_heads / p.kv_heads;
    const uint prefix = p.base + row + 1u;
    const uint blocks = prefix >> 2;
    const uint selected = min(blocks, p.block_budget);
    const uint total_tokens = selected * 4u + (prefix & 3u);
    const uint kv_width = p.kv_heads * 256u;

    uint bad = 0;
    if (tid < 16u) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
        row_alpha[tid] = 0.0f;
    }
    if (!tid) tile_abort = 0u;

    device const float* q_row = query + ulong(row) * p.q_stride + ulong(kv_head * gqa) * 256u;
    for (uint i = tid; i < 1024u; i += 128u) {
        const uint m = i >> 6, col4 = (i & 63u) * 4u;
        threadgroup bfloat4* dst = reinterpret_cast<threadgroup bfloat4*>(q_stage + m * 256u + col4);
        if (m < gqa) {
            const float4 src = float4(*reinterpret_cast<device const packed_float4*>(q_row + m * 256u + col4));
            *dst = bfloat4(
                nax_detail::clean_bf16(src.x, bad),
                nax_detail::clean_bf16(src.y, bad),
                nax_detail::clean_bf16(src.z, bad),
                nax_detail::clean_bf16(src.w, bad));
        } else {
            *dst = bfloat4(0);
        }
    }

    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> qt(q_stage, dextents<int, 2>{256, 16}, array<int, 2>{1, 256});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> kt(kv_stage, dextents<int, 2>{256, 32}, array<int, 2>{1, 256});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> pt(p_stage, dextents<int, 2>{32, 16}, array<int, 2>{1, 32});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> vt(kv_stage + uint(sg) * 64u, dextents<int, 2>{64, 32}, array<int, 2>{1, 256});

    constexpr auto qk_desc = matmul2d_descriptor(16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto pv_desc = matmul2d_descriptor(16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!qk_desc.relaxed_precision && !pv_desc.relaxed_precision, "BF16 FlashAttention with exact FP32 accumulation");
    matmul2d<qk_desc, execution_simdgroups<4>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<1>> pv_op;

    auto q0 = qt.slice<64, 16>(0, 0);
    auto k0 = kt.slice<64, 32>(0, 0);
    auto p_slice = pt.slice<32, 16>(0, 0);
    auto v_slice = vt.slice<64, 32>(0, 0);
    auto acc_pv = pv_op.get_destination_cooperative_tensor<decltype(p_slice), decltype(v_slice), float>();
    #pragma unroll
    for (ushort i = 0; i < acc_pv.get_capacity(); ++i) acc_pv[i] = 0.0f;

    for (uint tile_start = 0; tile_start < total_tokens; tile_start += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint tile_n = min(32u, total_tokens - tile_start);
        if (sg == 0) {
            uint tok = UINT_MAX;
            uint abort_lane = 0u;
            if (lane < tile_n) {
                const uint item = tile_start + lane;
                if (item < selected * 4u) {
                    const uint slot = item >> 2;
                    const uint id = blocks <= p.block_budget ? slot : ids[ulong(row) * p.id_stride + slot];
                    if (id >= blocks || (blocks > p.block_budget && slot && id <= ids[ulong(row) * p.id_stride + slot - 1u])) {
                        bad |= 4u;
                        abort_lane = 1u;
                    } else {
                        tok = id * 4u + (item & 3u);
                    }
                } else {
                    tok = blocks * 4u + item - selected * 4u;
                }
            }
            token_ids[lane] = tok;
            const uint any_abort = simd_or(abort_lane);
            if (lane == 0 && any_abort) tile_abort = 1u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tile_abort) break;

        for (uint i = tid; i < 1024u; i += 128u) {
            const uint t = i >> 5, vec8 = (i & 31u) * 8u;
            const uint tok = token_ids[t];
            threadgroup ushort4* dst = reinterpret_cast<threadgroup ushort4*>(kv_stage + t * 256u + vec8);
            if (tok != UINT_MAX) {
                device const ushort* k_ptr = keys + ulong(tok) * kv_width + kv_head * 256u + vec8;
                const ushort4 h0 = ushort4(*reinterpret_cast<device const packed_ushort4*>(k_ptr));
                const ushort4 h1 = ushort4(*reinterpret_cast<device const packed_ushort4*>(k_ptr + 4u));
                dst[0] = nax_detail::clean_bf16_bits4(h0, bad);
                dst[1] = nax_detail::clean_bf16_bits4(h1, bad);
            } else {
                dst[0] = ushort4(0);
                dst[1] = ushort4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto acc_qk = qk_op.get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
        #pragma unroll
        for (ushort i = 0; i < acc_qk.get_capacity(); ++i) acc_qk[i] = 0.0f;
        #pragma unroll
        for (ushort s = 0; s < 4; ++s) {
            auto q_slice = qt.slice<64, 16>(int(s * 64), 0);
            auto k_slice = kt.slice<64, 32>(int(s * 64), 0);
            qk_op.run(q_slice, k_slice, acc_qk);
        }
        #pragma unroll
        for (ushort i = 0; i < acc_qk.get_capacity(); ++i) {
            if (!acc_qk.is_valid_element(i)) continue;
            const auto idx = acc_qk.get_multidimensional_index(i);
            s_stage[uint(idx[1]) * 32u + uint(idx[0])] = float(acc_qk[i]) * 0.0625f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint m = sg; m < 16u; m += 4u) {
            if (m < gqa) {
                float score = -INFINITY;
                if (lane < tile_n) {
                    score = s_stage[m * 32u + lane];
                    if (nax_detail::nonfinite_bit(score)) { bad |= 1u; score = 0.0f; }
                }
                const float tile_max = simd_max(score);
                const float prev_max = row_max[m];
                const float new_max = max(prev_max, tile_max);
                const float alpha = prev_max == -INFINITY ? 0.0f : exp(prev_max - new_max);
                const float p_f32 = lane < tile_n ? exp(score - new_max) : 0.0f;
                const bfloat p_bf = nax_detail::clean_bf16(p_f32, bad);
                p_stage[m * 32u + lane] = p_bf;
                const float tile_sum = simd_sum(float(p_bf));
                if (lane == 0) {
                    row_max[m] = new_max;
                    row_sum[m] = row_sum[m] * alpha + tile_sum;
                    row_alpha[m] = alpha;
                }
            } else {
                p_stage[m * 32u + lane] = bfloat(0);
                if (lane == 0) row_alpha[m] = 0.0f;
            }
        }
        for (uint i = tid; i < 1024u; i += 128u) {
            const uint t = i >> 5, vec8 = (i & 31u) * 8u;
            const uint tok = token_ids[t];
            threadgroup ushort4* dst = reinterpret_cast<threadgroup ushort4*>(kv_stage + t * 256u + vec8);
            if (tok != UINT_MAX) {
                device const ushort* v_ptr = values + ulong(tok) * kv_width + kv_head * 256u + vec8;
                const ushort4 h0 = ushort4(*reinterpret_cast<device const packed_ushort4*>(v_ptr));
                const ushort4 h1 = ushort4(*reinterpret_cast<device const packed_ushort4*>(v_ptr + 4u));
                dst[0] = nax_detail::clean_bf16_bits4(h0, bad);
                dst[1] = nax_detail::clean_bf16_bits4(h1, bad);
            } else {
                dst[0] = ushort4(0);
                dst[1] = ushort4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (ushort i = 0; i < acc_pv.get_capacity(); ++i) {
            if (!acc_pv.is_valid_element(i)) continue;
            const auto idx = acc_pv.get_multidimensional_index(i);
            acc_pv[i] = float(acc_pv[i]) * row_alpha[ushort(idx[1])];
        }
        pv_op.run(p_slice, v_slice, acc_pv);
    }

    if (!tile_abort) {
        #pragma unroll
        for (ushort i = 0; i < acc_pv.get_capacity(); ++i) {
            if (!acc_pv.is_valid_element(i)) continue;
            const auto idx = acc_pv.get_multidimensional_index(i);
            const uint m = uint(idx[1]), d = uint(sg) * 64u + uint(idx[0]);
            if (m < gqa && d < 256u) {
                const float denom = row_sum[m];
                if (nax_detail::nonfinite_bit(denom) || denom <= 0.0f) {
                    bad |= 1u;
                } else {
                    const float val = float(acc_pv[i]) / denom;
                    bad |= nax_detail::nonfinite_bit(val);
                    output[ulong(row) * p.q_stride + ulong(kv_head * gqa + m) * 256u + d] = val;
                }
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

namespace nax_qsa_g12_detail {

constant uint kPartitionBlocks = 8u;
constant uint kSplitBlocks = 32u;
constant uint kMinSplits = 8u;
constant uint kNoEntry = 0xffffffffu;

static inline uint split_entries(uint count, uint splits) {
    const uint units = (count + kPartitionBlocks - 1u) / kPartitionBlocks;
    const uint used = clamp(max((count + kSplitBlocks - 1u) / kSplitBlocks, min(kMinSplits, units)), 1u, splits);
    const uint share = (count + used - 1u) / used;
    return ((share + kPartitionBlocks - 1u) / kPartitionBlocks) * kPartitionBlocks;
}

static inline uint written_splits(uint count, uint splits) {
    const uint per_split = split_entries(count, splits);
    return per_split ? (count + per_split - 1u) / per_split : 0u;
}

struct G12Scratch {
    uint entries[4];
    alignas(16) bfloat stage[16 * 256];
    alignas(16) float scores[96 * 16];
    float row_max[96];
    float row_sum[96];
    float prev_scale[96];
    atomic_uint rescale;
};
static_assert(sizeof(G12Scratch) <= 16384, "G12Scratch occupancy budget (<= 16 KiB)");

struct G12R4Scratch {
    alignas(16) bfloat stage[16 * 256];
    alignas(16) float scores[48 * 16];
    float row_max[48];
    float row_sum[48];
    float prev_scale[48];
    atomic_uint rescale;
};
static_assert(sizeof(G12R4Scratch) <= 16384, "G12R4Scratch occupancy budget (<= 16 KiB)");

static inline void softmax_lane_g12(threadgroup float* scores_and_probs,
    threadgroup float* row_max, threadgroup float* row_sum, threadgroup float* prev_scale,
    threadgroup atomic_uint* rescale, thread const uint (&entries)[4],
    uint first_pos, uint active_rows, uint lane_idx, thread uint& bad) {
    const uint fused_row = lane_idx >> 1;
    const uint col = (lane_idx & 1u) * 8u;
    const uint row = min(fused_row / 12u, active_rows - 1u);
    const uint causal_end = first_pos + row + 1u;
    uint allowed = 0u;
    #pragma unroll
    for (uint slot = 0u; slot < 2u; ++slot) {
        const uint entry = entries[(col >> 2) + slot];
        if (entry == kNoEntry || !((entry >> row) & 1u)) continue;
        const uint first = (entry >> 8u) * 4u;
        #pragma unroll
        for (uint t = 0u; t < 4u; ++t) {
            if (first + t < causal_end) allowed |= 1u << (slot * 4u + t);
        }
    }
    float score[8];
    float local_max = -INFINITY;
    #pragma unroll
    for (uint j = 0u; j < 8u; ++j) {
        if ((allowed >> j) & 1u) {
            float v = scores_and_probs[fused_row * 16u + col + j] * 0.0625f;
            if (nax_detail::nonfinite_bit(v)) { bad |= 1u; v = 0.0f; }
            score[j] = v;
            local_max = max(local_max, v);
        } else {
            score[j] = -INFINITY;
        }
    }
    local_max = max(local_max, simd_shuffle_xor(local_max, 1u));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup bfloat* probs_strided32 = reinterpret_cast<threadgroup bfloat*>(scores_and_probs);
    const float prev_max = row_max[fused_row];
    const float next_max = max(prev_max, local_max);
    float local_sum = 0.0f;
    #pragma unroll
    for (uint j = 0u; j < 8u; ++j) {
        const float p_f32 = ((allowed >> j) & 1u) ? exp(score[j] - next_max) : 0.0f;
        const bfloat p_bf = nax_detail::clean_bf16(p_f32, bad);
        probs_strided32[fused_row * 32u + col + j] = p_bf;
        local_sum += float(p_bf);
    }
    local_sum += simd_shuffle_xor(local_sum, 1u);
    if (col == 0u) {
        float scale = 1.0f;
        if (next_max != prev_max) {
            if (prev_max == -INFINITY) {
                scale = 0.0f;
            } else {
                scale = exp(prev_max - next_max);
                atomic_store_explicit(rescale, 1u, memory_order_relaxed);
            }
        }
        prev_scale[fused_row] = scale;
        row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
        row_max[fused_row] = next_max;
    }
}

} // namespace nax_qsa_g12_detail

kernel void nax_qsa_pack_queries_bf16(device const float* query [[buffer(0)]], device bfloat* q_bf16 [[buffer(1)]],
    constant nax_detail::NaxQsaG12Args& p [[buffer(2)]], device atomic_uint* error [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
    const uint total_vec4 = p.kv_heads * p.num_tiles * 96u * 64u;
    if (idx >= total_vec4) return;
    const uint col4 = (idx & 63u) * 4u;
    const uint tmp = idx >> 6;
    const uint m = tmp % 96u;
    const uint head_tile = tmp / 96u;
    const uint tile = head_tile % p.num_tiles;
    const uint kv_head = head_tile / p.num_tiles;
    const uint r = m / 12u, h_in_group = m % 12u;
    const uint row = tile * 8u + r;
    device bfloat4* dst = reinterpret_cast<device bfloat4*>(q_bf16 + ulong(idx) * 4ul);
    if (row < p.rows) {
        uint bad = 0u;
        device const float* src_ptr = query + ulong(row) * p.q_stride + ulong(kv_head * 12u + h_in_group) * 256u + col4;
        const float4 src = float4(*reinterpret_cast<device const packed_float4*>(src_ptr));
        *dst = bfloat4(
            nax_detail::clean_bf16(src.x, bad),
            nax_detail::clean_bf16(src.y, bad),
            nax_detail::clean_bf16(src.z, bad),
            nax_detail::clean_bf16(src.w, bad));
        if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
    } else {
        *dst = bfloat4(0);
    }
}

kernel void nax_qsa_build_tile_lists(device const uint* ids [[buffer(0)]], device uint* tile_lists [[buffer(1)]],
    device uint* tile_counts [[buffer(2)]], constant nax_detail::NaxQsaG12Args& p [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint mask_words[4096];
    threadgroup uint sg_totals[8];
    threadgroup atomic_uint tile_bad;

    if (group >= p.num_tiles) return;
    const uint first_row = group * 8u;
    if (first_row >= p.rows) return;
    const uint active_rows = min(8u, p.rows - first_row);
    const uint history_blocks = (p.base + first_row + active_rows + 3u) >> 2;
    if (history_blocks > 16384u) {
        if (!tid) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            tile_counts[group] = 0x80000000u;
        }
        return;
    }
    const uint num_words = (history_blocks + 3u) >> 2;
    for (uint w = tid; w < num_words; w += 256u) {
        atomic_store_explicit(&mask_words[w], 0u, memory_order_relaxed);
    }
    if (!tid) atomic_store_explicit(&tile_bad, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0u; r < active_rows; ++r) {
        const uint row = first_row + r;
        const uint prefix = p.base + row + 1u;
        const uint blocks = prefix >> 2;
        const uint rem = prefix & 3u;
        const uint bit = 1u << r;
        if (blocks <= p.block_budget) {
            for (uint b = tid; b < blocks; b += 256u) {
                atomic_fetch_or_explicit(&mask_words[b >> 2], bit << ((b & 3u) * 8u), memory_order_relaxed);
            }
        } else {
            device const uint* row_ids = ids + ulong(row) * p.id_stride;
            for (uint s = tid; s < p.block_budget; s += 256u) {
                const uint b = row_ids[s];
                if (b >= blocks || (s > 0u && b <= row_ids[s - 1u])) {
                    atomic_fetch_or_explicit(&tile_bad, 4u, memory_order_relaxed);
                } else {
                    atomic_fetch_or_explicit(&mask_words[b >> 2], bit << ((b & 3u) * 8u), memory_order_relaxed);
                }
            }
        }
        if (rem && tid == r) {
            atomic_fetch_or_explicit(&mask_words[blocks >> 2], bit << ((blocks & 3u) * 8u), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint bad_bits = atomic_load_explicit(&tile_bad, memory_order_relaxed);
    if (bad_bits) {
        if (!tid) {
            atomic_fetch_or_explicit(error, bad_bits, memory_order_relaxed);
            tile_counts[group] = 0x80000000u;
        }
        return;
    }

    const uint words_per_thread = (num_words + 255u) >> 8;
    const uint w0 = min(uint(tid) * words_per_thread, num_words);
    const uint w1 = min(w0 + words_per_thread, num_words);
    uint my_count = 0u;
    for (uint w = w0; w < w1; ++w) {
        const uint word = atomic_load_explicit(&mask_words[w], memory_order_relaxed);
        my_count += uint((word & 0xffu) != 0u) + uint(((word >> 8) & 0xffu) != 0u) +
                    uint(((word >> 16) & 0xffu) != 0u) + uint(((word >> 24) & 0xffu) != 0u);
    }
    const uint sg_incl = simd_prefix_inclusive_sum(my_count);
    if (lane == 31u) sg_totals[sg] = sg_incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint offset = sg_incl - my_count;
    for (ushort g = 0u; g < sg; ++g) offset += sg_totals[g];
    device uint* out_list = tile_lists + ulong(group) * p.list_stride;
    for (uint w = w0; w < w1; ++w) {
        const uint word = atomic_load_explicit(&mask_words[w], memory_order_relaxed);
        if (!word) continue;
        #pragma unroll
        for (uint byte_idx = 0u; byte_idx < 4u; ++byte_idx) {
            const uint row_mask = (word >> (byte_idx * 8u)) & 0xffu;
            if (row_mask) {
                const uint b = w * 4u + byte_idx;
                if (offset < p.list_stride) out_list[offset] = (b << 8u) | row_mask;
                ++offset;
            }
        }
    }
    if (tid == 255u) {
        if (offset > p.list_stride) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        tile_counts[group] = min(offset, p.list_stride);
    }
}

kernel void nax_qsa_attention_g12(device bfloat* q_bf16 [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* tile_lists [[buffer(3)]],
    device const uint* tile_counts [[buffer(4)]], device float* output [[buffer(5)]],
    device float* partials [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    device float* statistics [[buffer(8)]], constant nax_detail::NaxQsaG12Args& p [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr ushort M = 96, N = 16, D = 256;
    threadgroup nax_qsa_g12_detail::G12Scratch scratch;

    const uint kv_head = group.x, split = group.z;
    const uint tile = (!p.sparse_mode && p.splits == 1u && group.y < p.num_tiles) ? (p.num_tiles - 1u - group.y) : group.y;
    if (kv_head >= p.kv_heads || tile >= p.num_tiles || split >= p.splits) return;
    const uint first_row = tile * 8u;
    if (first_row >= p.rows) return;
    const uint active_rows = min(8u, p.rows - first_row);
    const uint first_pos = p.base + first_row;
    const uint max_valid_tok = first_pos + active_rows;
    const uint history_blocks = (max_valid_tok + 3u) >> 2;
    const uint all_rows_mask = (1u << active_rows) - 1u;

    uint count = history_blocks;
    device const uint* list = nullptr;
    if (p.sparse_mode) {
        const uint raw_count = tile_counts[tile];
        if (raw_count >= 0x80000000u) return;
        count = min(raw_count, p.list_stride);
        list = tile_lists + ulong(tile) * p.list_stride;
    }

    const uint per_split = nax_qsa_g12_detail::split_entries(count, p.splits);
    const uint begin = split * per_split;
    if (begin >= count) return;
    const uint end = min(count, begin + per_split);

    if (tid < M) {
        scratch.row_max[tid] = -INFINITY;
        scratch.row_sum[tid] = 0.0f;
    }

    device bfloat* tile_q = q_bf16 + (ulong(kv_head) * p.num_tiles + tile) * (96ul * 256ul);
    auto qt = tensor(tile_q, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    auto st = tensor(scratch.scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    auto pt = tensor(reinterpret_cast<threadgroup bfloat*>(scratch.scores), dextents<int, 2>{N, M}, array<int, 2>{1, 2 * N});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> kv_stage(scratch.stage, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto q0 = qt.slice<D, M>(0, 0);
    auto p0 = pt.slice<N, M>(0, 0);
    auto stage0 = kv_stage.slice<D, N>(0, 0);

    constexpr auto qk_desc = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(M, D, N, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!qk_desc.relaxed_precision && !pv_desc.relaxed_precision, "BF16 G12 FlashAttention with exact FP32 accumulation");
    matmul2d<qk_desc, execution_simdgroups<8>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<8>> pv_op;

    auto running = pv_op.get_destination_cooperative_tensor<decltype(p0), decltype(stage0), float>();
    const bool running_full = uint(running.get_capacity()) * 256u == uint(M) * D;
    #pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (running_full || running.is_valid_element(i)) running[i] = 0.0f;
    }

    const ushort lane = tid & 31u;
    const uint kv_width = p.kv_heads * 256u;
    uint bad = 0u;

    for (uint step = begin; step < end; step += 4u) {
        if (tid == 0u) atomic_store_explicit(&scratch.rescale, 0u, memory_order_relaxed);

        uint step_entries[4];
        if (list) {
            const uint my_e = (lane < 4u && step + lane < end) ? list[step + lane] : nax_qsa_g12_detail::kNoEntry;
            #pragma unroll
            for (ushort s = 0; s < 4; ++s) {
                const uint e = simd_broadcast(my_e, s);
                step_entries[s] = ((e >> 8u) < history_blocks) ? e : nax_qsa_g12_detail::kNoEntry;
            }
        } else {
            #pragma unroll
            for (ushort s = 0; s < 4; ++s) {
                const uint idx = step + uint(s);
                step_entries[s] = (idx < end && idx < history_blocks) ? ((idx << 8u) | all_rows_mask) : nax_qsa_g12_detail::kNoEntry;
            }
        }

        for (uint chunk = tid; chunk < 512u; chunk += 256u) {
            const uint token = chunk >> 5, vec8 = (chunk & 31u) * 8u;
            const uint entry = step_entries[token >> 2];
            const uint phys_tok = (entry >> 8u) * 4u + (token & 3u);
            threadgroup ushort4* dst = reinterpret_cast<threadgroup ushort4*>(scratch.stage + token * 256u + vec8);
            if (entry != nax_qsa_g12_detail::kNoEntry && phys_tok < max_valid_tok) {
                device const ushort* k_ptr = keys + ulong(phys_tok) * kv_width + kv_head * 256u + vec8;
                const uint4 raw = uint4(*reinterpret_cast<device const packed_uint4*>(k_ptr));
                dst[0] = nax_detail::clean_bf16_bits4(as_type<ushort4>(raw.xy), bad);
                dst[1] = nax_detail::clean_bf16_bits4(as_type<ushort4>(raw.zw), bad);
            } else {
                dst[0] = ushort4(0);
                dst[1] = ushort4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto step_scores = qk_op.get_destination_cooperative_tensor<decltype(q0), decltype(stage0), float>();
        qk_op.run(q0, stage0, step_scores);
        step_scores.store(st.slice<N, M>(0, 0));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint chunk = tid; chunk < 512u; chunk += 256u) {
            const uint token = chunk >> 5, vec8 = (chunk & 31u) * 8u;
            const uint entry = step_entries[token >> 2];
            const uint phys_tok = (entry >> 8u) * 4u + (token & 3u);
            threadgroup ushort4* dst = reinterpret_cast<threadgroup ushort4*>(scratch.stage + token * 256u + vec8);
            if (entry != nax_qsa_g12_detail::kNoEntry && phys_tok < max_valid_tok) {
                device const ushort* v_ptr = values + ulong(phys_tok) * kv_width + kv_head * 256u + vec8;
                const uint4 raw = uint4(*reinterpret_cast<device const packed_uint4*>(v_ptr));
                dst[0] = nax_detail::clean_bf16_bits4(as_type<ushort4>(raw.xy), bad);
                dst[1] = nax_detail::clean_bf16_bits4(as_type<ushort4>(raw.zw), bad);
            } else {
                dst[0] = ushort4(0);
                dst[1] = ushort4(0);
            }
        }
        if (tid < 192u) {
            nax_qsa_g12_detail::softmax_lane_g12(scratch.scores, scratch.row_max, scratch.row_sum,
                scratch.prev_scale, &scratch.rescale, step_entries, first_pos, active_rows, tid, bad);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (atomic_load_explicit(&scratch.rescale, memory_order_relaxed)) {
            #pragma unroll
            for (ushort i = 0; i < running.get_capacity(); ++i) {
                if (!running_full && !running.is_valid_element(i)) continue;
                const auto coord = running.get_multidimensional_index(i);
                running[i] = float(running[i]) * scratch.prev_scale[ushort(coord[1])];
            }
        }
        pv_op.run(p0, stage0, running);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.splits == 1u) {
        #pragma unroll
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            if (!running_full && !running.is_valid_element(i)) continue;
            const auto coord = running.get_multidimensional_index(i);
            const uint d = uint(coord[0]), m = uint(coord[1]);
            const uint r = m / 12u, h_in_group = m % 12u;
            if (r < active_rows && d < 256u) {
                const float denom = scratch.row_sum[m];
                if (nax_detail::nonfinite_bit(denom) || denom <= 0.0f) {
                    bad |= 1u;
                } else {
                    const float val = float(running[i]) / denom;
                    bad |= nax_detail::nonfinite_bit(val);
                    output[ulong(first_row + r) * p.q_stride + ulong(kv_head * 12u + h_in_group) * 256u + d] = val;
                }
            }
        }
    } else {
        const ulong slot = (ulong(tile) * p.kv_heads + kv_head) * p.splits + split;
        auto target = tensor(partials + slot * (96ul * 256ul), dextents<int, 2>{D, M}, array<int, 2>{1, D});
        running.store(target.slice<D, M>(0, 0));
        if (tid < M) {
            statistics[(slot * 96ul + tid) * 2ul] = scratch.row_max[tid];
            statistics[(slot * 96ul + tid) * 2ul + 1ul] = scratch.row_sum[tid];
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_qsa_attention_g12_reduce(device const float* partials [[buffer(0)]], device const float* statistics [[buffer(1)]],
    device const uint* tile_counts [[buffer(2)]], device float* output [[buffer(3)]],
    constant nax_detail::NaxQsaG12Args& p [[buffer(4)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float weights[16];
    threadgroup float sg_max_vals[8];

    const uint kv_head = group.x, row_head = group.y;
    if (kv_head >= p.kv_heads || row_head >= p.rows * 12u || tid >= 256u) return;
    const uint global_row = row_head / 12u, h_in_group = row_head % 12u;
    const uint tile = global_row >> 3, r = global_row & 7u;
    const uint first_row = tile * 8u;
    const uint active_rows = min(8u, p.rows - first_row);
    const uint fused_row = r * 12u + h_in_group;

    uint count = (p.base + first_row + active_rows + 3u) >> 2;
    if (p.sparse_mode) {
        const uint raw_count = tile_counts[tile];
        if (raw_count >= 0x80000000u) return;
        count = min(raw_count, p.list_stride);
    }
    const uint written = nax_qsa_g12_detail::written_splits(count, p.splits);
    const ulong head_slot = (ulong(tile) * p.kv_heads + kv_head) * p.splits;

    const float maximum = tid < written ? statistics[((head_slot + tid) * 96ul + fused_row) * 2ul] : -INFINITY;
    const float grp_max = simd_max(maximum);
    if (lane == 0u) sg_max_vals[sg] = grp_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_maximum = simd_max(lane < 8u ? sg_max_vals[lane] : -INFINITY);
    if (tid < written) {
        weights[tid] = (maximum == -INFINITY || row_maximum == -INFINITY) ? 0.0f : exp(maximum - row_maximum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float numerator = 0.0f, denominator = 0.0f;
    for (uint s = 0u; s < written; ++s) {
        const float w = weights[s];
        if (w > 0.0f) {
            const ulong stat = ((head_slot + s) * 96ul + fused_row) * 2ul;
            numerator += w * partials[((head_slot + s) * 96ul + fused_row) * 256ul + tid];
            denominator += w * statistics[stat + 1ul];
        }
    }
    uint bad = 0u;
    if (nax_detail::nonfinite_bit(denominator) || denominator <= 0.0f) {
        bad |= 1u;
    } else {
        const float val = numerator / denominator;
        bad |= nax_detail::nonfinite_bit(val);
        output[ulong(global_row) * p.q_stride + ulong(kv_head * 12u + h_in_group) * 256u + tid] = val;
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_qsa_pack_queries_bf16_r4(device const float* query [[buffer(0)]], device bfloat* q_bf16 [[buffer(1)]],
    constant nax_detail::NaxQsaG12Args& p [[buffer(2)]], device atomic_uint* error [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
    const uint total_vec4 = p.kv_heads * p.num_tiles * 48u * 64u;
    if (idx >= total_vec4) return;
    const uint col4 = (idx & 63u) * 4u;
    const uint tmp = idx >> 6;
    const uint m = tmp % 48u;
    const uint head_tile = tmp / 48u;
    const uint tile = head_tile % p.num_tiles;
    const uint kv_head = head_tile / p.num_tiles;
    const uint r = m / 12u, h_in_group = m % 12u;
    const uint row = tile * 4u + r;
    device bfloat4* dst = reinterpret_cast<device bfloat4*>(q_bf16 + ulong(idx) * 4ul);
    if (row < p.rows) {
        uint bad = 0u;
        device const float* src_ptr = query + ulong(row) * p.q_stride + ulong(kv_head * 12u + h_in_group) * 256u + col4;
        const float4 src = float4(*reinterpret_cast<device const packed_float4*>(src_ptr));
        *dst = bfloat4(
            nax_detail::clean_bf16(src.x, bad),
            nax_detail::clean_bf16(src.y, bad),
            nax_detail::clean_bf16(src.z, bad),
            nax_detail::clean_bf16(src.w, bad));
        if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
    } else {
        *dst = bfloat4(0);
    }
}

kernel void nax_qsa_build_tile_lists_r4(device const uint* ids [[buffer(0)]], device uint* tile_lists [[buffer(1)]],
    device uint* tile_counts [[buffer(2)]], constant nax_detail::NaxQsaG12Args& p [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint mask_words[4096];
    threadgroup uint sg_totals[8];
    threadgroup atomic_uint tile_bad;

    if (group >= p.num_tiles) return;
    const uint first_row = group * 4u;
    if (first_row >= p.rows) return;
    const uint active_rows = min(4u, p.rows - first_row);
    const uint history_blocks = (p.base + first_row + active_rows + 3u) >> 2;
    if (history_blocks > 16384u) {
        if (!tid) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            tile_counts[group] = 0x80000000u;
        }
        return;
    }
    const uint num_words = (history_blocks + 3u) >> 2;
    for (uint w = tid; w < num_words; w += 256u) {
        atomic_store_explicit(&mask_words[w], 0u, memory_order_relaxed);
    }
    if (!tid) atomic_store_explicit(&tile_bad, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0u; r < active_rows; ++r) {
        const uint row = first_row + r;
        const uint prefix = p.base + row + 1u;
        const uint blocks = prefix >> 2;
        const uint rem = prefix & 3u;
        const uint bit = 1u << r;
        if (blocks <= p.block_budget) {
            for (uint b = tid; b < blocks; b += 256u) {
                atomic_fetch_or_explicit(&mask_words[b >> 2], bit << ((b & 3u) * 8u), memory_order_relaxed);
            }
        } else {
            device const uint* row_ids = ids + ulong(row) * p.id_stride;
            for (uint s = tid; s < p.block_budget; s += 256u) {
                const uint b = row_ids[s];
                if (b >= blocks || (s > 0u && b <= row_ids[s - 1u])) {
                    atomic_fetch_or_explicit(&tile_bad, 4u, memory_order_relaxed);
                } else {
                    atomic_fetch_or_explicit(&mask_words[b >> 2], bit << ((b & 3u) * 8u), memory_order_relaxed);
                }
            }
        }
        if (rem && tid == r) {
            atomic_fetch_or_explicit(&mask_words[blocks >> 2], bit << ((blocks & 3u) * 8u), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint bad_bits = atomic_load_explicit(&tile_bad, memory_order_relaxed);
    if (bad_bits) {
        if (!tid) {
            atomic_fetch_or_explicit(error, bad_bits, memory_order_relaxed);
            tile_counts[group] = 0x80000000u;
        }
        return;
    }

    const uint words_per_thread = (num_words + 255u) >> 8;
    const uint w0 = min(uint(tid) * words_per_thread, num_words);
    const uint w1 = min(w0 + words_per_thread, num_words);
    uint my_count = 0u;
    for (uint w = w0; w < w1; ++w) {
        const uint word = atomic_load_explicit(&mask_words[w], memory_order_relaxed);
        my_count += uint((word & 0xffu) != 0u) + uint(((word >> 8) & 0xffu) != 0u) +
                    uint(((word >> 16) & 0xffu) != 0u) + uint(((word >> 24) & 0xffu) != 0u);
    }
    const uint sg_incl = simd_prefix_inclusive_sum(my_count);
    if (lane == 31u) sg_totals[sg] = sg_incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint offset = sg_incl - my_count;
    for (ushort g = 0u; g < sg; ++g) offset += sg_totals[g];
    device uint* out_list = tile_lists + ulong(group) * p.list_stride;
    for (uint w = w0; w < w1; ++w) {
        const uint word = atomic_load_explicit(&mask_words[w], memory_order_relaxed);
        if (!word) continue;
        #pragma unroll
        for (uint byte_idx = 0u; byte_idx < 4u; ++byte_idx) {
            const uint row_mask = (word >> (byte_idx * 8u)) & 0xffu;
            if (row_mask) {
                const uint b = w * 4u + byte_idx;
                if (offset < p.list_stride) out_list[offset] = (b << 8u) | row_mask;
                ++offset;
            }
        }
    }
    if (tid == 255u) {
        if (offset > p.list_stride) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        tile_counts[group] = min(offset, p.list_stride);
    }
}

kernel void nax_qsa_attention_g12_r4(device bfloat* q_bf16 [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* tile_lists [[buffer(3)]],
    device const uint* tile_counts [[buffer(4)]], device float* output [[buffer(5)]],
    device float* partials [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    device float* statistics [[buffer(8)]], constant nax_detail::NaxQsaG12Args& p [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr ushort M = 48, N = 16, D = 256;
    threadgroup nax_qsa_g12_detail::G12R4Scratch scratch;

    const uint kv_head = group.x, split = group.z;
    const uint tile = (!p.sparse_mode && p.splits == 1u && group.y < p.num_tiles) ? (p.num_tiles - 1u - group.y) : group.y;
    if (kv_head >= p.kv_heads || tile >= p.num_tiles || split >= p.splits) return;
    const uint first_row = tile * 4u;
    if (first_row >= p.rows) return;
    const uint active_rows = min(4u, p.rows - first_row);
    const uint first_pos = p.base + first_row;
    const uint max_valid_tok = first_pos + active_rows;
    const uint history_blocks = (max_valid_tok + 3u) >> 2;
    const uint all_rows_mask = (1u << active_rows) - 1u;

    uint count = history_blocks;
    device const uint* list = nullptr;
    if (p.sparse_mode) {
        const uint raw_count = tile_counts[tile];
        if (raw_count >= 0x80000000u) return;
        count = min(raw_count, p.list_stride);
        list = tile_lists + ulong(tile) * p.list_stride;
    }

    const uint per_split = nax_qsa_g12_detail::split_entries(count, p.splits);
    const uint begin = split * per_split;
    if (begin >= count) return;
    const uint end = min(count, begin + per_split);

    if (tid < M) {
        scratch.row_max[tid] = -INFINITY;
        scratch.row_sum[tid] = 0.0f;
    }

    device bfloat* tile_q = q_bf16 + (ulong(kv_head) * p.num_tiles + tile) * (48ul * 256ul);
    auto qt = tensor(tile_q, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    auto st = tensor(scratch.scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    auto pt = tensor(reinterpret_cast<threadgroup bfloat*>(scratch.scores), dextents<int, 2>{N, M}, array<int, 2>{1, 2 * N});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> kv_stage(scratch.stage, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto q0 = qt.slice<D, M>(0, 0);
    auto p0 = pt.slice<N, M>(0, 0);
    auto stage0 = kv_stage.slice<D, N>(0, 0);

    constexpr auto qk_desc = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(M, D, N, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!qk_desc.relaxed_precision && !pv_desc.relaxed_precision, "BF16 G12 R4 FlashAttention with exact FP32 accumulation");
    matmul2d<qk_desc, execution_simdgroups<4>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<4>> pv_op;

    auto running = pv_op.get_destination_cooperative_tensor<decltype(p0), decltype(stage0), float>();
    const bool running_full = uint(running.get_capacity()) * 128u == uint(M) * D;
    #pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (running_full || running.is_valid_element(i)) running[i] = 0.0f;
    }

    const ushort lane = tid & 31u;
    const uint kv_width = p.kv_heads * 256u;
    uint bad = 0u;

    for (uint step = begin; step < end; step += 4u) {
        if (tid == 0u) atomic_store_explicit(&scratch.rescale, 0u, memory_order_relaxed);

        uint step_entries[4];
        if (list) {
            const uint my_e = (lane < 4u && step + lane < end) ? list[step + lane] : nax_qsa_g12_detail::kNoEntry;
            #pragma unroll
            for (ushort s = 0; s < 4; ++s) {
                const uint e = simd_broadcast(my_e, s);
                step_entries[s] = ((e >> 8u) < history_blocks) ? e : nax_qsa_g12_detail::kNoEntry;
            }
        } else {
            #pragma unroll
            for (ushort s = 0; s < 4; ++s) {
                const uint idx = step + uint(s);
                step_entries[s] = (idx < end && idx < history_blocks) ? ((idx << 8u) | all_rows_mask) : nax_qsa_g12_detail::kNoEntry;
            }
        }

        for (uint chunk = tid; chunk < 512u; chunk += 128u) {
            const uint token = chunk >> 5, vec8 = (chunk & 31u) * 8u;
            const uint entry = step_entries[token >> 2];
            const uint phys_tok = (entry >> 8u) * 4u + (token & 3u);
            threadgroup ushort4* dst = reinterpret_cast<threadgroup ushort4*>(scratch.stage + token * 256u + vec8);
            if (entry != nax_qsa_g12_detail::kNoEntry && phys_tok < max_valid_tok) {
                device const ushort* k_ptr = keys + ulong(phys_tok) * kv_width + kv_head * 256u + vec8;
                const uint4 raw = uint4(*reinterpret_cast<device const packed_uint4*>(k_ptr));
                dst[0] = nax_detail::clean_bf16_bits4(as_type<ushort4>(raw.xy), bad);
                dst[1] = nax_detail::clean_bf16_bits4(as_type<ushort4>(raw.zw), bad);
            } else {
                dst[0] = ushort4(0);
                dst[1] = ushort4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto step_scores = qk_op.get_destination_cooperative_tensor<decltype(q0), decltype(stage0), float>();
        qk_op.run(q0, stage0, step_scores);
        step_scores.store(st.slice<N, M>(0, 0));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint chunk = tid; chunk < 512u; chunk += 128u) {
            const uint token = chunk >> 5, vec8 = (chunk & 31u) * 8u;
            const uint entry = step_entries[token >> 2];
            const uint phys_tok = (entry >> 8u) * 4u + (token & 3u);
            threadgroup ushort4* dst = reinterpret_cast<threadgroup ushort4*>(scratch.stage + token * 256u + vec8);
            if (entry != nax_qsa_g12_detail::kNoEntry && phys_tok < max_valid_tok) {
                device const ushort* v_ptr = values + ulong(phys_tok) * kv_width + kv_head * 256u + vec8;
                const uint4 raw = uint4(*reinterpret_cast<device const packed_uint4*>(v_ptr));
                dst[0] = nax_detail::clean_bf16_bits4(as_type<ushort4>(raw.xy), bad);
                dst[1] = nax_detail::clean_bf16_bits4(as_type<ushort4>(raw.zw), bad);
            } else {
                dst[0] = ushort4(0);
                dst[1] = ushort4(0);
            }
        }
        if (tid < 96u) {
            nax_qsa_g12_detail::softmax_lane_g12(scratch.scores, scratch.row_max, scratch.row_sum,
                scratch.prev_scale, &scratch.rescale, step_entries, first_pos, active_rows, tid, bad);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (atomic_load_explicit(&scratch.rescale, memory_order_relaxed)) {
            #pragma unroll
            for (ushort i = 0; i < running.get_capacity(); ++i) {
                if (!running_full && !running.is_valid_element(i)) continue;
                const auto coord = running.get_multidimensional_index(i);
                running[i] = float(running[i]) * scratch.prev_scale[ushort(coord[1])];
            }
        }
        pv_op.run(p0, stage0, running);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.splits == 1u) {
        #pragma unroll
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            if (!running_full && !running.is_valid_element(i)) continue;
            const auto coord = running.get_multidimensional_index(i);
            const uint d = uint(coord[0]), m = uint(coord[1]);
            const uint r = m / 12u, h_in_group = m % 12u;
            if (r < active_rows && d < 256u) {
                const float denom = scratch.row_sum[m];
                if (nax_detail::nonfinite_bit(denom) || denom <= 0.0f) {
                    bad |= 1u;
                } else {
                    const float val = float(running[i]) / denom;
                    bad |= nax_detail::nonfinite_bit(val);
                    output[ulong(first_row + r) * p.q_stride + ulong(kv_head * 12u + h_in_group) * 256u + d] = val;
                }
            }
        }
    } else {
        const ulong slot = (ulong(tile) * p.kv_heads + kv_head) * p.splits + split;
        auto target = tensor(partials + slot * (48ul * 256ul), dextents<int, 2>{D, M}, array<int, 2>{1, D});
        running.store(target.slice<D, M>(0, 0));
        if (tid < M) {
            statistics[(slot * 48ul + tid) * 2ul] = scratch.row_max[tid];
            statistics[(slot * 48ul + tid) * 2ul + 1ul] = scratch.row_sum[tid];
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_qsa_attention_g12_reduce_r4(device const float* partials [[buffer(0)]], device const float* statistics [[buffer(1)]],
    device const uint* tile_counts [[buffer(2)]], device float* output [[buffer(3)]],
    constant nax_detail::NaxQsaG12Args& p [[buffer(4)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float weights[16];
    threadgroup float sg_max_vals[8];

    const uint kv_head = group.x, row_head = group.y;
    if (kv_head >= p.kv_heads || row_head >= p.rows * 12u || tid >= 256u) return;
    const uint global_row = row_head / 12u, h_in_group = row_head % 12u;
    const uint tile = global_row >> 2, r = global_row & 3u;
    const uint first_row = tile * 4u;
    const uint active_rows = min(4u, p.rows - first_row);
    const uint fused_row = r * 12u + h_in_group;

    uint count = (p.base + first_row + active_rows + 3u) >> 2;
    if (p.sparse_mode) {
        const uint raw_count = tile_counts[tile];
        if (raw_count >= 0x80000000u) return;
        count = min(raw_count, p.list_stride);
    }
    const uint written = nax_qsa_g12_detail::written_splits(count, p.splits);
    const ulong head_slot = (ulong(tile) * p.kv_heads + kv_head) * p.splits;

    const float maximum = tid < written ? statistics[((head_slot + tid) * 48ul + fused_row) * 2ul] : -INFINITY;
    const float grp_max = simd_max(maximum);
    if (lane == 0u) sg_max_vals[sg] = grp_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_maximum = simd_max(lane < 8u ? sg_max_vals[lane] : -INFINITY);
    if (tid < written) {
        weights[tid] = (maximum == -INFINITY || row_maximum == -INFINITY) ? 0.0f : exp(maximum - row_maximum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float numerator = 0.0f, denominator = 0.0f;
    for (uint s = 0u; s < written; ++s) {
        const float w = weights[s];
        if (w > 0.0f) {
            const ulong stat = ((head_slot + s) * 48ul + fused_row) * 2ul;
            numerator += w * partials[((head_slot + s) * 48ul + fused_row) * 256ul + tid];
            denominator += w * statistics[stat + 1ul];
        }
    }
    uint bad = 0u;
    if (nax_detail::nonfinite_bit(denominator) || denominator <= 0.0f) {
        bad |= 1u;
    } else {
        const float val = numerator / denominator;
        bad |= nax_detail::nonfinite_bit(val);
        output[ulong(global_row) * p.q_stride + ulong(kv_head * 12u + h_in_group) * 256u + tid] = val;
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

namespace nax_m8_split_detail {

// Cross-threadgroup K-split arrival/reduction adapted from Splash 1.4.0 split_reduce.h.
static inline bool split_arrive_last(device atomic_uint* counter, uint splits, ushort tid, threadgroup uint* arrival) {
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    return *arrival == splits - 1u;
}

static inline void split_release(device atomic_uint* counter, ushort tid) {
    if (tid == 0u) {
        atomic_store_explicit(counter, 0u, memory_order_relaxed);
    }
}

static inline void stage_a_4slices(device const float* input, ulong in_stride, uint tokens, uint k_col0, uint active_slices,
    ushort tid, threadgroup bfloat* a_stage, thread uint& bad) {
    constexpr uint StageStride = 32u, APlane = 8u * StageStride;
    const uint m = uint(tid) >> 4u;
    const uint s = (uint(tid) >> 2u) & 3u;
    const uint k8 = (uint(tid) & 3u) * 8u;
    threadgroup bfloat4* dst = reinterpret_cast<threadgroup bfloat4*>(a_stage + s * APlane + m * StageStride + k8);
    if (m < tokens && s < active_slices) {
        device const float* p = input + ulong(m) * in_stride + ulong(k_col0 + s * 32u + k8);
        const float4 v0 = float4(*reinterpret_cast<device const packed_float4*>(p));
        const float4 v1 = float4(*reinterpret_cast<device const packed_float4*>(p + 4ul));
        dst[0] = bfloat4(nax_detail::clean_bf16(v0.x, bad), nax_detail::clean_bf16(v0.y, bad),
                         nax_detail::clean_bf16(v0.z, bad), nax_detail::clean_bf16(v0.w, bad));
        dst[1] = bfloat4(nax_detail::clean_bf16(v1.x, bad), nax_detail::clean_bf16(v1.y, bad),
                         nax_detail::clean_bf16(v1.z, bad), nax_detail::clean_bf16(v1.w, bad));
    } else {
        dst[0] = bfloat4(0);
        dst[1] = bfloat4(0);
    }
}

static inline void stage_a_m8_t64(device const float* input, ulong in_stride, uint tokens, uint k_col0,
    ushort tid, threadgroup bfloat* a_plane, thread uint& bad) {
    constexpr uint StageStride = 36u;
    const uint m = uint(tid) >> 3u;
    const uint k4 = (uint(tid) & 7u) * 4u;
    threadgroup bfloat4* dst = reinterpret_cast<threadgroup bfloat4*>(a_plane + m * StageStride + k4);
    if (m < tokens) {
        const float4 v = float4(*reinterpret_cast<device const packed_float4*>(input + ulong(m) * in_stride + ulong(k_col0 + k4)));
        *dst = bfloat4(nax_detail::clean_bf16(v.x, bad), nax_detail::clean_bf16(v.y, bad),
                       nax_detail::clean_bf16(v.z, bad), nax_detail::clean_bf16(v.w, bad));
    } else {
        *dst = bfloat4(0);
    }
}

static inline void load_q8_pair_m8(device const uchar* weights, uint rows, uint blocks, uint row_bytes,
    uint sg_col_base, uint b0, ushort lane, thread ushort (&q_dst)[32], thread half2& raw_sc2) {
    const uint run = b0 / 16u;
    const uint width = min(16u, blocks - run * 16u);
    const uint sc_col = sg_col_base + uint(lane);
    raw_sc2 = (sc_col < rows)
        ? half2(*reinterpret_cast<device const packed_half2*>(weights + ulong(sc_col) * row_bytes + 2ul * ulong(b0)))
        : half2(0);
    const ulong q_off = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(b0 % 16u) + ulong(lane) * ulong(width);
    #pragma unroll
    for (ushort c = 0; c < 32; ++c) {
        const uint w_col = sg_col_base + uint(c);
        q_dst[c] = (w_col < rows)
            ? *reinterpret_cast<device const ushort*>(weights + ulong(w_col) * row_bytes + q_off)
            : ushort(0);
    }
}

static inline void sanitize_pair_scales_m8(half2 raw_sc2, thread half2& sc_dst, thread bool& pair_clean, thread uint& bad) {
    bool lane_clean = true;
    if (nax_detail::nonfinite_half_bit(raw_sc2.x)) { bad |= 1u; raw_sc2.x = half(0); lane_clean = false; }
    if (nax_detail::nonfinite_half_bit(raw_sc2.y)) { bad |= 1u; raw_sc2.y = half(0); lane_clean = false; }
    if (fmax(fabs(float(raw_sc2.x)), fabs(float(raw_sc2.y))) * 128.0f > 65504.0f) lane_clean = false;
    sc_dst = raw_sc2;
    pair_clean = simd_all(lane_clean);
}

static inline void store_q8_sub_m8(thread const ushort (&q_src)[32], half2 sc_src, bool pair_clean, ushort sub,
    ushort lane, threadgroup half* sg_b, thread uint& bad) {
    constexpr uint StageStride = 36u;
    if (pair_clean) {
        #pragma unroll
        for (ushort c = 0; c < 32; ++c) {
            const half2 sc2 = simd_broadcast(sc_src, c);
            const half sc = sub ? sc2.y : sc2.x;
            const uchar qb = sub ? uchar(q_src[c] >> 8u) : uchar(q_src[c] & 0xffu);
            sg_b[uint(c) * StageStride + uint(lane)] = half(as_type<char>(qb)) * sc;
        }
    } else {
        #pragma unroll
        for (ushort c = 0; c < 32; ++c) {
            const half2 sc2 = simd_broadcast(sc_src, c);
            const half sc = sub ? sc2.y : sc2.x;
            const uchar qb = sub ? uchar(q_src[c] >> 8u) : uchar(q_src[c] & 0xffu);
            const half hv = half(float(as_type<char>(qb)) * float(sc));
            bad |= nax_detail::nonfinite_half_bit(hv);
            sg_b[uint(c) * StageStride + uint(lane)] = hv;
        }
    }
}

static inline void load_q8_step_gguf_m8(device const uchar* weights, uint rows, uint columns, uint row_bytes,
    uint sg_col_base, uint k_step, ushort lane, thread half4 (&w_dst)[8], thread uint& bad) {
    if (nax_layout == 0u) {
        const ushort sub_q = lane & 7u, r_in_pass = lane >> 3u;
        #pragma unroll
        for (ushort p = 0; p < 8; ++p) {
            const ushort col = p * 4u + r_in_pass;
            const bool live_col = sg_col_base + uint(col) < rows;
            device const uchar* w_row = weights + ulong(live_col ? sg_col_base + uint(col) : 0u) * row_bytes;
            w_dst[p] = nax_detail::decode4_gguf(w_row, k_step, sub_q, live_col, bad);
        }
    } else {
        #pragma unroll
        for (ushort p = 0; p < 2; ++p) {
            const ushort col = p * 16u + (lane >> 1u), ai = lane & 1u;
            const bool live_col = sg_col_base + uint(col) < rows;
            device const uchar* w_row = weights + ulong(live_col ? sg_col_base + uint(col) : 0u) * row_bytes;
            const half4x4 w16 = nax_detail::decode16_any(w_row, k_step * 2u + uint(ai), columns, live_col, bad);
            w_dst[p * 4u + 0u] = w16[0];
            w_dst[p * 4u + 1u] = w16[1];
            w_dst[p * 4u + 2u] = w16[2];
            w_dst[p * 4u + 3u] = w16[3];
        }
    }
}

static inline void store_q8_step_gguf_m8(thread const half4 (&w_src)[8], ushort lane, threadgroup half* sg_b) {
    constexpr uint StageStride = 36u;
    if (nax_layout == 0u) {
        const ushort sub_q = lane & 7u, r_in_pass = lane >> 3u;
        #pragma unroll
        for (ushort p = 0; p < 8; ++p) {
            const ushort col = p * 4u + r_in_pass;
            *reinterpret_cast<threadgroup half4*>(sg_b + uint(col) * StageStride + uint(sub_q) * 4u) = w_src[p];
        }
    } else {
        #pragma unroll
        for (ushort p = 0; p < 2; ++p) {
            const ushort col = p * 16u + (lane >> 1u), ai = lane & 1u;
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(sg_b + uint(col) * StageStride + uint(ai) * 16u);
            dst[0] = w_src[p * 4u + 0u];
            dst[1] = w_src[p * 4u + 1u];
            dst[2] = w_src[p * 4u + 2u];
            dst[3] = w_src[p * 4u + 3u];
        }
    }
}

} // namespace nax_m8_split_detail

template<uint T, uint NR>
static inline void qmv_q8_split_rows(device const uchar* weights, device const float* input, device float* output,
    constant nax_detail::GroupedMmArgs& a, device coherent(device) float* partials, device atomic_uint* counters,
    device atomic_uint* error, uint group_x, uint split, uint splits, ushort tid, ushort sg, ushort lane,
    threadgroup uint* arrival) {
    constexpr uint ColsPerTG = 4u * NR;
    const uint first_col = group_x * ColsPerTG;
    const uint sg_col0 = first_col + uint(sg) * NR;
    const uint active_cols = (sg_col0 < a.rows) ? min(NR, a.rows - sg_col0) : 0u;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const uint blocks = a.columns / 32u;
    const uint blocks_per_split = splits > 1u ? (blocks / splits) : blocks;
    uint b = split * blocks_per_split;
    const uint b_end = b + blocks_per_split;

    device const float* in_rows[T];
    #pragma unroll
    for (uint t = 0u; t < T; ++t) in_rows[t] = input + ulong(t) * in_stride;

    float sums[NR][T];
    #pragma unroll
    for (uint r = 0u; r < NR; ++r) {
        #pragma unroll
        for (uint t = 0u; t < T; ++t) sums[r][t] = 0.0f;
    }

    uint bad = 0u;
    if (active_cols > 0u) {
        if (nax_layout == 1u) {
            const ushort sc_r = lane >> 3u;
            const ushort sc_k2 = (lane & 7u) * 2u;
            for (; b + 16u <= b_end; b += 16u) {
                const uint run = b / 16u;
                half2 my_sc2 = half2(0);
                if (sc_r < active_cols) {
                    my_sc2 = half2(*reinterpret_cast<device const packed_half2*>(
                        weights + ulong(sg_col0 + sc_r) * a.row_bytes + ulong(b + sc_k2) * 2ul));
                    if (nax_detail::nonfinite_half_bit(my_sc2.x)) { bad |= 1u; my_sc2.x = half(0); }
                    if (nax_detail::nonfinite_half_bit(my_sc2.y)) { bad |= 1u; my_sc2.y = half(0); }
                }
                const ulong q_off = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(lane) * 16ul;
                uint4 qw[NR];
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    qw[r] = (r < active_cols)
                        ? uint4(*reinterpret_cast<device const packed_uint4*>(weights + ulong(sg_col0 + r) * a.row_bytes + q_off))
                        : uint4(0u);
                }
                #pragma unroll
                for (ushort part = 0; part < 4; ++part) {
                    #pragma unroll
                    for (ushort sub = 0; sub < 4; ++sub) {
                        const uint k = uint(part) * 4u + uint(sub);
                        float xv[T];
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) {
                            const float v = in_rows[t][(b + k) * 32u + uint(lane)];
                            bad |= nax_detail::nonfinite_bit(v);
                            xv[t] = v;
                        }
                        #pragma unroll
                        for (uint r = 0u; r < NR; ++r) {
                            if (r < active_cols) {
                                const half2 sc_pair = simd_broadcast(my_sc2, ushort((r << 3u) | (k >> 1u)));
                                const float sc_f = float((k & 1u) ? sc_pair.y : sc_pair.x);
                                const float wv = sc_f * float(as_type<char>(uchar((qw[r][part] >> (8u * sub)) & 0xffu)));
                                #pragma unroll
                                for (uint t = 0u; t < T; ++t) sums[r][t] += wv * xv[t];
                            }
                        }
                    }
                }
            }
            for (; b < b_end; ++b) {
                const uint run = b / 16u;
                const uint width = min(16u, blocks - run * 16u);
                float xv[T];
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float v = in_rows[t][b * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(v);
                    xv[t] = v;
                }
                const ulong q_off = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(lane) * ulong(width) + ulong(b % 16u);
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    if (r < active_cols) {
                        device const uchar* w_row = weights + ulong(sg_col0 + r) * a.row_bytes;
                        const half sc_h = reinterpret_cast<device const half*>(w_row)[b];
                        float sc_f = float(sc_h);
                        if (nax_detail::nonfinite_half_bit(sc_h)) { bad |= 1u; sc_f = 0.0f; }
                        const float wv = sc_f * float(as_type<char>(w_row[q_off]));
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) sums[r][t] += wv * xv[t];
                    }
                }
            }
        } else {
            for (; b < b_end; ++b) {
                float xv[T];
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float v = in_rows[t][b * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(v);
                    xv[t] = v;
                }
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    if (r < active_cols) {
                        device const uchar* enc = weights + ulong(sg_col0 + r) * a.row_bytes + ulong(b) * 34ul;
                        float sc_f = nax_detail::stored_half(enc);
                        if (nax_detail::nonfinite_bit(sc_f)) { bad |= 1u; sc_f = 0.0f; }
                        const float wv = sc_f * float(as_type<char>(enc[2u + lane]));
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) sums[r][t] += wv * xv[t];
                    }
                }
            }
        }
    }

    const uint mode = a.maximum_tasks;
    const float silu_scale = as_type<float>(a.experts);
    const auto write_out = [&](uint tok, uint col, float val) {
        bad |= nax_detail::nonfinite_bit(val);
        device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(tok) * a.output_stride_bytes);
        if (mode == 1u) {
            const float g0 = y[col];
            bad |= nax_detail::nonfinite_bit(g0);
            val = (g0 * nax_detail::sigmoid_f32(g0)) * val;
            bad |= nax_detail::nonfinite_bit(val);
        } else if (mode == 2u) {
            const float x = val * silu_scale;
            bad |= nax_detail::nonfinite_bit(silu_scale) | nax_detail::nonfinite_bit(x);
            val = x * nax_detail::sigmoid_f32(x);
            bad |= nax_detail::nonfinite_bit(val);
        }
        y[col] = val;
    };

    if (splits == 1u) {
        #pragma unroll
        for (uint r = 0u; r < NR; ++r) {
            if (r < active_cols) {
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float tot = simd_sum(sums[r][t]);
                    if (!lane) write_out(t, sg_col0 + r, tot);
                }
            }
        }
    } else {
        const ulong p_stride = ulong((a.rows + 15u) & ~15u);
        float my_tot[NR];
        #pragma unroll
        for (uint r = 0u; r < NR; ++r) {
            my_tot[r] = 0.0f;
            if (r < active_cols) {
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float tot = simd_sum(sums[r][t]);
                    if (lane == t) {
                        my_tot[r] = tot;
                        partials[(ulong(split) * 8ul + t) * p_stride + sg_col0 + r] = tot;
                    }
                }
            }
        }
        if (nax_m8_split_detail::split_arrive_last(counters + group_x, splits, tid, arrival)) {
            if (lane < T) {
                #pragma unroll
                for (uint r = 0u; r < NR; ++r) {
                    if (r < active_cols) {
                        const uint col = sg_col0 + r;
                        float sum = 0.0f;
                        for (uint s = 0u; s < splits; ++s) {
                            sum += (s == split) ? my_tot[r] : partials[(ulong(s) * 8ul + uint(lane)) * p_stride + col];
                        }
                        write_out(uint(lane), col, sum);
                    }
                }
            }
            nax_m8_split_detail::split_release(counters + group_x, tid);
        }
    }
    bad = simd_or(bad);
    if (!lane && bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

template<uint T>
static inline void dispatch_qmv_q8_split(device const uchar* weights, device const float* input, device float* output,
    constant nax_detail::GroupedMmArgs& a, device coherent(device) float* partials, device atomic_uint* counters,
    device atomic_uint* error, uint group_x, uint split, uint splits, ushort tid, ushort sg, ushort lane,
    threadgroup uint* arrival) {
    if (a.rows <= 1024u) {
        qmv_q8_split_rows<T, 1u>(weights, input, output, a, partials, counters, error, group_x, split, splits, tid, sg, lane, arrival);
    } else if (T > 4u || a.rows <= 4096u) {
        qmv_q8_split_rows<T, 2u>(weights, input, output, a, partials, counters, error, group_x, split, splits, tid, sg, lane, arrival);
    } else {
        qmv_q8_split_rows<T, 4u>(weights, input, output, a, partials, counters, error, group_x, split, splits, tid, sg, lane, arrival);
    }
}

template<uint T>
static inline void qmv_q8_swiglu_split_rows(device const uchar* gate_weights, device const uchar* up_weights,
    device const float* input, device float* output, constant nax_detail::GroupedMmArgs& a,
    device coherent(device) float* partials, device atomic_uint* counters, device atomic_uint* error,
    uint group_x, uint split, uint splits, ushort tid, ushort sg, ushort lane, threadgroup uint* arrival) {
    const uint col = group_x * 4u + uint(sg);
    const bool col_live = col < a.rows;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const uint blocks = a.columns / 32u;
    const uint blocks_per_split = splits > 1u ? (blocks / splits) : blocks;
    uint b = split * blocks_per_split;
    const uint b_end = b + blocks_per_split;

    device const float* in_rows[T];
    #pragma unroll
    for (uint t = 0u; t < T; ++t) in_rows[t] = input + ulong(t) * in_stride;

    float sum_g[T], sum_u[T];
    #pragma unroll
    for (uint t = 0u; t < T; ++t) {
        sum_g[t] = 0.0f;
        sum_u[t] = 0.0f;
    }

    uint bad = 0u;
    if (col_live) {
        device const uchar* g_row = gate_weights + ulong(col) * a.row_bytes;
        device const uchar* u_row = up_weights + ulong(col) * a.row_bytes;
        if (nax_layout == 1u) {
            device const half* g_scales = reinterpret_cast<device const half*>(g_row);
            device const half* u_scales = reinterpret_cast<device const half*>(u_row);
            for (; b + 16u <= b_end; b += 16u) {
                const uint run = b / 16u;
                float g_sc_lane = (lane < 16u) ? float(g_scales[b + uint(lane)]) : 0.0f;
                float u_sc_lane = (lane < 16u) ? float(u_scales[b + uint(lane)]) : 0.0f;
                if (nax_detail::nonfinite_bit(g_sc_lane)) { bad |= 1u; g_sc_lane = 0.0f; }
                if (nax_detail::nonfinite_bit(u_sc_lane)) { bad |= 1u; u_sc_lane = 0.0f; }
                const ulong q_off = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(lane) * 16ul;
                const uint4 g_words = uint4(*reinterpret_cast<device const packed_uint4*>(g_row + q_off));
                const uint4 u_words = uint4(*reinterpret_cast<device const packed_uint4*>(u_row + q_off));
                #pragma unroll
                for (uint part = 0u; part < 4u; ++part) {
                    #pragma unroll
                    for (uint sub = 0u; sub < 4u; ++sub) {
                        const uint k = part * 4u + sub;
                        float xv[T];
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) {
                            const float v = in_rows[t][(b + k) * 32u + uint(lane)];
                            bad |= nax_detail::nonfinite_bit(v);
                            xv[t] = v;
                        }
                        const float g_sc = simd_broadcast(g_sc_lane, ushort(k));
                        const float u_sc = simd_broadcast(u_sc_lane, ushort(k));
                        const float gw = g_sc * float(as_type<char>(uchar((g_words[part] >> (8u * sub)) & 255u)));
                        const float uw = u_sc * float(as_type<char>(uchar((u_words[part] >> (8u * sub)) & 255u)));
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) {
                            sum_g[t] += gw * xv[t];
                            sum_u[t] += uw * xv[t];
                        }
                    }
                }
            }
            for (; b < b_end; ++b) {
                const uint run = b / 16u;
                const uint width = min(16u, blocks - run * 16u);
                float xv[T];
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float v = in_rows[t][b * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(v);
                    xv[t] = v;
                }
                float g_sc = float(g_scales[b]);
                float u_sc = float(u_scales[b]);
                if (nax_detail::nonfinite_bit(g_sc)) { bad |= 1u; g_sc = 0.0f; }
                if (nax_detail::nonfinite_bit(u_sc)) { bad |= 1u; u_sc = 0.0f; }
                const ulong q_off = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(lane) * ulong(width) + ulong(b % 16u);
                const float gw = g_sc * float(as_type<char>(g_row[q_off]));
                const float uw = u_sc * float(as_type<char>(u_row[q_off]));
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    sum_g[t] += gw * xv[t];
                    sum_u[t] += uw * xv[t];
                }
            }
        } else {
            for (; b < b_end; ++b) {
                float xv[T];
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float v = in_rows[t][b * 32u + uint(lane)];
                    bad |= nax_detail::nonfinite_bit(v);
                    xv[t] = v;
                }
                device const uchar* g_enc = g_row + ulong(b) * 34ul;
                device const uchar* u_enc = u_row + ulong(b) * 34ul;
                float g_sc = nax_detail::stored_half(g_enc);
                float u_sc = nax_detail::stored_half(u_enc);
                if (nax_detail::nonfinite_bit(g_sc)) { bad |= 1u; g_sc = 0.0f; }
                if (nax_detail::nonfinite_bit(u_sc)) { bad |= 1u; u_sc = 0.0f; }
                const float gw = g_sc * float(as_type<char>(g_enc[2u + lane]));
                const float uw = u_sc * float(as_type<char>(u_enc[2u + lane]));
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    sum_g[t] += gw * xv[t];
                    sum_u[t] += uw * xv[t];
                }
            }
        }
    }

    const auto write_swiglu = [&](uint tok, float g_val, float u_val) {
        bad |= nax_detail::nonfinite_bit(g_val) | nax_detail::nonfinite_bit(u_val);
        const float val = (g_val * nax_detail::sigmoid_f32(g_val)) * u_val;
        bad |= nax_detail::nonfinite_bit(val);
        device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(tok) * a.output_stride_bytes);
        y[col] = val;
    };

    if (splits == 1u) {
        if (col_live) {
            #pragma unroll
            for (uint t = 0u; t < T; ++t) {
                const float g0 = simd_sum(sum_g[t]);
                const float u0 = simd_sum(sum_u[t]);
                if (!lane) write_swiglu(t, g0, u0);
            }
        }
    } else {
        const ulong p_stride = ulong((a.rows + 15u) & ~15u);
        float my_g = 0.0f, my_u = 0.0f;
        if (col_live) {
            #pragma unroll
            for (uint t = 0u; t < T; ++t) {
                const float g0 = simd_sum(sum_g[t]);
                const float u0 = simd_sum(sum_u[t]);
                if (lane == t) {
                    my_g = g0;
                    my_u = u0;
                    partials[((ulong(split) * 2ul + 0ul) * 8ul + t) * p_stride + col] = g0;
                    partials[((ulong(split) * 2ul + 1ul) * 8ul + t) * p_stride + col] = u0;
                }
            }
        }
        if (nax_m8_split_detail::split_arrive_last(counters + group_x, splits, tid, arrival)) {
            if (col_live && lane < T) {
                float gs = 0.0f, us = 0.0f;
                for (uint s = 0u; s < splits; ++s) {
                    gs += (s == split) ? my_g : partials[((ulong(s) * 2ul + 0ul) * 8ul + uint(lane)) * p_stride + col];
                    us += (s == split) ? my_u : partials[((ulong(s) * 2ul + 1ul) * 8ul + uint(lane)) * p_stride + col];
                }
                write_swiglu(uint(lane), gs, us);
            }
            nax_m8_split_detail::split_release(counters + group_x, tid);
        }
    }
    bad = simd_or(bad);
    if (!lane && bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

// Coalesced templated SIMT QMV for dense Q8_0 projections (m = 1..8):
// 128 threads = 4 SGs, adaptive NR = 1/2/4 cols/SG, zero weight SRAM, zero K-loop barriers,
// 512 B coalesced uint4 run loads on q8_lanes_v1 and cross-TG K-split reduction.
kernel void nax_m8_qmv_q8(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device coherent(device) float* partials [[buffer(4)]], device atomic_uint* counters [[buffer(5)]],
    device atomic_uint* error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint arrival;

    const uint splits = max(1u, a.selected);
    if (!a.tokens || a.tokens > 8u || !a.rows || !a.columns || a.columns % 32u != 0u || a.row_bytes != (a.columns / 32u) * 34u ||
        a.maximum_tasks > 2u || splits > 32u || a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        (splits > 1u && (a.columns % (splits * 512u)) != 0u) ||
        (nax_layout == 1u && a.columns >= 512u && a.columns % 256u != 0u)) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint max_nr = a.tokens <= 4u ? 4u : 2u;
    const uint nr = a.rows <= 1024u ? 1u : (a.rows <= 4096u ? 2u : max_nr);
    if (group.x * (4u * nr) >= a.rows || group.y >= splits) return;

    switch (a.tokens) {
        case 1u: dispatch_qmv_q8_split<1>(weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 2u: dispatch_qmv_q8_split<2>(weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 3u: dispatch_qmv_q8_split<3>(weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 4u: dispatch_qmv_q8_split<4>(weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 5u: dispatch_qmv_q8_split<5>(weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 6u: dispatch_qmv_q8_split<6>(weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 7u: dispatch_qmv_q8_split<7>(weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        default: dispatch_qmv_q8_split<8>(weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
    }
}

// 64-thread (2-SG), BN = 64 double-buffered NAX matmul2d(8, 32, 32) for wide dense Q8_0 projections (rows > 2560):
// StageStride = 36 (bank-conflict-free), 10,372 B SRAM = 3 TGs/core, 1 threadgroup_barrier/step, zero cross-SG reduction.
kernel void nax_m8_dense_q8(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device coherent(device) float* partials [[buffer(4)]], device atomic_uint* counters [[buffer(5)]],
    device atomic_uint* error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort Rows = 8, OutCols = 32, BN = 64, KS = 32, StageStride = 36;
    constexpr uint APlane = uint(Rows) * StageStride, BPlane = uint(OutCols) * StageStride;
    alignas(16) threadgroup bfloat a_stage[2 * APlane];
    alignas(16) threadgroup half b_stage[2][2 * BPlane];
    threadgroup uint arrival;

    const uint splits = max(1u, a.selected);
    if (!a.tokens || a.tokens > 8u || !a.columns || a.columns % 32u != 0u || a.row_bytes != (a.columns / 32u) * 34u ||
        a.maximum_tasks > 2u || splits > 32u || a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        (splits > 1u && (a.columns % (splits * 512u)) != 0u) ||
        (nax_layout == 1u && a.columns >= 512u && a.columns % 256u != 0u)) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint first_col = group.x * BN;
    const uint split = group.y;
    if (first_col >= a.rows || split >= splits) return;

    const uint sg_col_base = first_col + uint(sg) * OutCols;
    const bool sg_live = sg_col_base < a.rows;

    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at0(a_stage, dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at1(a_stage + APlane, dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(b_stage[0] + uint(sg) * BPlane, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(b_stage[1] + uint(sg) * BPlane, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    constexpr auto descriptor = matmul2d_descriptor(Rows, OutCols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;
    auto a_s0 = at0.slice<KS, Rows>(0, 0), a_s1 = at1.slice<KS, Rows>(0, 0);
    auto b_s0 = bt0.slice<KS, OutCols>(0, 0), b_s1 = bt1.slice<KS, OutCols>(0, 0);
    auto acc = operation.get_destination_cooperative_tensor<decltype(a_s0), decltype(b_s0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const uint total_steps = a.columns / KS;
    const uint steps_per_split = splits > 1u ? (total_steps / splits) : total_steps;
    const uint step_begin = split * steps_per_split;
    const uint step_end = step_begin + steps_per_split;
    uint bad = 0u;

    if (nax_layout == 1u && ((step_end - step_begin) & 1u) == 0u && step_begin < step_end) {
        const uint blocks = a.columns / 32u;
        ushort q_pair[32];
        half2 raw_sc = half2(0), sc_pair = half2(0);
        bool pair_clean = true;

        nax_m8_split_detail::stage_a_m8_t64(input, in_stride, a.tokens, step_begin * 32u, tid, a_stage, bad);
        if (sg_live) {
            nax_m8_split_detail::load_q8_pair_m8(weights, a.rows, blocks, a.row_bytes, sg_col_base, step_begin, lane, q_pair, raw_sc);
            nax_m8_split_detail::sanitize_pair_scales_m8(raw_sc, sc_pair, pair_clean, bad);
            nax_m8_split_detail::store_q8_sub_m8(q_pair, sc_pair, pair_clean, 0u, lane, b_stage[0] + uint(sg) * BPlane, bad);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s_rel = 0u; s_rel < steps_per_split; ++s_rel) {
            const uint step = step_begin + s_rel;
            const ushort sub = ushort(s_rel & 1u);
            const uint next_plane = (s_rel + 1u) & 1u;

            if (s_rel + 1u < steps_per_split) {
                nax_m8_split_detail::stage_a_m8_t64(input, in_stride, a.tokens, (step + 1u) * 32u, tid, a_stage + next_plane * APlane, bad);
                if (sub == 0u) {
                    if (sg_live) {
                        nax_m8_split_detail::store_q8_sub_m8(q_pair, sc_pair, pair_clean, 1u, lane, b_stage[next_plane] + uint(sg) * BPlane, bad);
                        if (s_rel + 2u < steps_per_split) {
                            nax_m8_split_detail::load_q8_pair_m8(weights, a.rows, blocks, a.row_bytes, sg_col_base, step + 2u, lane, q_pair, raw_sc);
                        }
                    }
                    if (sg_live) {
                        if ((s_rel & 1u) == 0u) operation.run(a_s0, b_s0, acc);
                        else operation.run(a_s1, b_s1, acc);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                } else {
                    if (sg_live) {
                        if ((s_rel & 1u) == 0u) operation.run(a_s0, b_s0, acc);
                        else operation.run(a_s1, b_s1, acc);
                        nax_m8_split_detail::sanitize_pair_scales_m8(raw_sc, sc_pair, pair_clean, bad);
                        nax_m8_split_detail::store_q8_sub_m8(q_pair, sc_pair, pair_clean, 0u, lane, b_stage[next_plane] + uint(sg) * BPlane, bad);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            } else if (sg_live) {
                if ((s_rel & 1u) == 0u) operation.run(a_s0, b_s0, acc);
                else operation.run(a_s1, b_s1, acc);
            }
        }
    } else if (step_begin < step_end) {
        half4 w_next[8];
        nax_m8_split_detail::stage_a_m8_t64(input, in_stride, a.tokens, step_begin * 32u, tid, a_stage, bad);
        if (sg_live) {
            nax_m8_split_detail::load_q8_step_gguf_m8(weights, a.rows, a.columns, a.row_bytes, sg_col_base, step_begin, lane, w_next, bad);
            nax_m8_split_detail::store_q8_step_gguf_m8(w_next, lane, b_stage[0] + uint(sg) * BPlane);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s_rel = 0u; s_rel < steps_per_split; ++s_rel) {
            const uint step = step_begin + s_rel;
            const uint next_plane = (s_rel + 1u) & 1u;
            if (s_rel + 1u < steps_per_split) {
                nax_m8_split_detail::stage_a_m8_t64(input, in_stride, a.tokens, (step + 1u) * 32u, tid, a_stage + next_plane * APlane, bad);
                if (sg_live) {
                    nax_m8_split_detail::load_q8_step_gguf_m8(weights, a.rows, a.columns, a.row_bytes, sg_col_base, step + 1u, lane, w_next, bad);
                }
            }
            if (sg_live) {
                if ((s_rel & 1u) == 0u) operation.run(a_s0, b_s0, acc);
                else operation.run(a_s1, b_s1, acc);
            }
            if (s_rel + 1u < steps_per_split) {
                if (sg_live) {
                    nax_m8_split_detail::store_q8_step_gguf_m8(w_next, lane, b_stage[next_plane] + uint(sg) * BPlane);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float (*epi)[8][33] = reinterpret_cast<threadgroup float (*)[8][33]>(b_stage);
    if (sg_live) {
        #pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            const auto idx = acc.get_multidimensional_index(i);
            epi[sg][uint(idx[1])][uint(idx[0])] = float(acc[i]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint col_out = sg_col_base + uint(lane);
    const bool col_live = sg_live && (col_out < a.rows);
    const uint mode = a.maximum_tasks;
    const float silu_scale = as_type<float>(a.experts);
    const auto write_out = [&](uint r, float val) {
        bad |= nax_detail::nonfinite_bit(val);
        device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(r) * a.output_stride_bytes);
        if (mode == 1u) {
            const float g0 = y[col_out];
            bad |= nax_detail::nonfinite_bit(g0);
            val = (g0 * nax_detail::sigmoid_f32(g0)) * val;
            bad |= nax_detail::nonfinite_bit(val);
        } else if (mode == 2u) {
            const float x = val * silu_scale;
            bad |= nax_detail::nonfinite_bit(silu_scale) | nax_detail::nonfinite_bit(x);
            val = x * nax_detail::sigmoid_f32(x);
            bad |= nax_detail::nonfinite_bit(val);
        }
        y[col_out] = val;
    };

    if (splits == 1u) {
        if (col_live) {
            #pragma unroll
            for (ushort r = 0; r < 8; ++r) {
                if (r < a.tokens) write_out(r, epi[sg][r][lane]);
            }
        }
    } else {
        const ulong p_stride = ulong((a.rows + 63u) & ~63u);
        if (col_live) {
            #pragma unroll
            for (ushort r = 0; r < 8; ++r) {
                if (r < a.tokens) partials[(ulong(split) * 8ul + r) * p_stride + col_out] = epi[sg][r][lane];
            }
        }
        if (nax_m8_split_detail::split_arrive_last(counters + group.x, splits, tid, &arrival)) {
            if (col_live) {
                #pragma unroll
                for (ushort r = 0; r < 8; ++r) {
                    if (r < a.tokens) {
                        float sum = 0.0f;
                        for (uint s = 0u; s < splits; ++s) {
                            sum += (s == split) ? epi[sg][r][lane] : partials[(ulong(s) * 8ul + r) * p_stride + col_out];
                        }
                        write_out(r, sum);
                    }
                }
            }
            nax_m8_split_detail::split_release(counters + group.x, tid);
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

// 128-thread (4-SG, 4 cols/TG) coalesced templated SIMT QMV fused gate + up + SwiGLU dense Q8_0 kernel:
// 512 B coalesced uint4 run loads on q8_lanes_v1, zero weight SRAM, zero K-loop barriers, cross-TG K-split reduction.
kernel void nax_m8_dense_q8_swiglu(device const uchar* gate_weights [[buffer(0)]], device const uchar* up_weights [[buffer(1)]],
    device const float* input [[buffer(2)]], device float* output [[buffer(3)]],
    constant nax_detail::GroupedMmArgs& a [[buffer(4)]],
    device coherent(device) float* partials [[buffer(5)]], device atomic_uint* counters [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint arrival;

    const uint splits = max(1u, a.selected);
    if (!a.tokens || a.tokens > 8u || !a.rows || !a.columns || a.columns % 32u != 0u || a.row_bytes != (a.columns / 32u) * 34u ||
        splits > 32u || a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        (splits > 1u && (a.columns % (splits * 512u)) != 0u) ||
        (nax_layout == 1u && a.columns >= 512u && a.columns % 256u != 0u)) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (group.x * 4u >= a.rows || group.y >= splits) return;

    switch (a.tokens) {
        case 1u: qmv_q8_swiglu_split_rows<1>(gate_weights, up_weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 2u: qmv_q8_swiglu_split_rows<2>(gate_weights, up_weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 3u: qmv_q8_swiglu_split_rows<3>(gate_weights, up_weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 4u: qmv_q8_swiglu_split_rows<4>(gate_weights, up_weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 5u: qmv_q8_swiglu_split_rows<5>(gate_weights, up_weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 6u: qmv_q8_swiglu_split_rows<6>(gate_weights, up_weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        case 7u: qmv_q8_swiglu_split_rows<7>(gate_weights, up_weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
        default: qmv_q8_swiglu_split_rows<8>(gate_weights, up_weights, input, output, a, partials, counters, error, group.x, group.y, splits, tid, sg, lane, &arrival); break;
    }
}

// 64-thread (2-SG), BN64 NAX matmul2d(8,32,32) on coalesced Q8_tile64 planes, retaining arbitrary 32-column K-splits.
// Default double staging declares 10,372 bytes and one TG barrier per step; the private single-stage candidate declares 5,188 bytes and two barriers. Compiler padding can increase the allocation.
kernel void nax_m8_dense_q8_t64(device const uchar* plane0 [[buffer(0)]], device const ushort* meta [[buffer(1)]],
    device const float* input [[buffer(2)]], device float* output [[buffer(3)]],
    constant nax_detail::GroupedMmArgs& a [[buffer(4)]],
    device coherent(device) float* partials [[buffer(5)]], device atomic_uint* counters [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort Rows = 8, OutCols = 32, BN = 64, KS = 32, StageStride = 36;
    constexpr uint APlane = uint(Rows) * StageStride, BPlane = uint(OutCols) * StageStride;
    constexpr uint StagePlanes = TOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE ? 1u : 2u;
    alignas(16) threadgroup bfloat a_stage[StagePlanes * APlane];
    alignas(16) threadgroup half b_stage[StagePlanes][2 * BPlane];
    threadgroup uint arrival;

    const uint splits = max(1u, a.selected);
    if (!a.tokens || a.tokens > 8u || !a.rows || !a.columns || a.columns % 32u != 0u ||
        a.maximum_tasks > 2u || splits > 32u || a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        ((a.columns / 32u) % splits) != 0u) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint first_col = group.x * BN;
    const uint split = group.y;
    if (first_col >= a.rows || split >= splits) return;

    const uint sg_col_base = first_col + uint(sg) * OutCols;
    const bool sg_live = sg_col_base < a.rows;

    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at0(a_stage, dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at1(a_stage + (TOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE ? 0u : APlane), dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(b_stage[0] + uint(sg) * BPlane, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(b_stage[TOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE ? 0u : 1u] + uint(sg) * BPlane, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    constexpr auto descriptor = matmul2d_descriptor(Rows, OutCols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;
    auto a_s0 = at0.slice<KS, Rows>(0, 0), a_s1 = at1.slice<KS, Rows>(0, 0);
    auto b_s0 = bt0.slice<KS, OutCols>(0, 0), b_s1 = bt1.slice<KS, OutCols>(0, 0);
    auto acc = operation.get_destination_cooperative_tensor<decltype(a_s0), decltype(b_s0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const uint G = a.columns / KS;
    const uint steps_per_split = splits > 1u ? (G / splits) : G;
    const uint step_begin = split * steps_per_split;
    const uint step_end = step_begin + steps_per_split;
    uint bad = 0u;

    const uint sg_col_in_tile = uint(sg) * 32u + uint(lane);
    if (step_begin < step_end) {
        nax_detail::Q8Tile64Ipt1 w_next{};
        if (sg_live) {
            w_next = nax_detail::load_q8_t64_ipt1(plane0, meta, G, group.x, step_begin, sg_col_in_tile);
        }
        for (uint s_rel = 0u; s_rel < steps_per_split; ++s_rel) {
            const uint step = step_begin + s_rel;
            const uint cur_plane = TOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE ? 0u : (s_rel & 1u);
            nax_m8_split_detail::stage_a_m8_t64(input, in_stride, a.tokens, step * 32u, tid, a_stage + cur_plane * APlane, bad);
            if (sg_live) {
                nax_detail::store_q8_t64_ipt1(w_next, b_stage[cur_plane] + uint(sg) * BPlane + uint(lane) * StageStride, bad);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (s_rel + 1u < steps_per_split && sg_live) {
                w_next = nax_detail::load_q8_t64_ipt1(plane0, meta, G, group.x, step + 1u, sg_col_in_tile);
            }
            if (sg_live) {
                if (cur_plane == 0u) operation.run(a_s0, b_s0, acc);
                else operation.run(a_s1, b_s1, acc);
            }
#if TOKSPUR_PRIVATE_DENSE_M8_SINGLE_TILE
            // Both SGs finish consuming the shared A tile before any thread reuses the staging storage.
            threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
        }
    }

    const uint mode = a.maximum_tasks;
    const float silu_scale = as_type<float>(a.experts);
    const auto write_out = [&](uint r, uint col_out, float val) {
        bad |= nax_detail::nonfinite_bit(val);
        device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(r) * a.output_stride_bytes);
        if (mode == 1u) {
            const float g0 = y[col_out];
            bad |= nax_detail::nonfinite_bit(g0);
            val = (g0 * nax_detail::sigmoid_f32(g0)) * val;
            bad |= nax_detail::nonfinite_bit(val);
        } else if (mode == 2u) {
            const float x = val * silu_scale;
            bad |= nax_detail::nonfinite_bit(silu_scale) | nax_detail::nonfinite_bit(x);
            val = x * nax_detail::sigmoid_f32(x);
            bad |= nax_detail::nonfinite_bit(val);
        }
        y[col_out] = val;
    };

    if (splits == 1u) {
        if (sg_live) {
            #pragma unroll
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto idx = acc.get_multidimensional_index(i);
                const uint col_out = sg_col_base + uint(idx[0]);
                const uint r = uint(idx[1]);
                if (r < a.tokens && col_out < a.rows) {
                    write_out(r, col_out, float(acc[i]));
                }
            }
        }
    } else {
        const ulong p_stride = ulong((a.rows + 63u) & ~63u);
        if (sg_live) {
            #pragma unroll
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto idx = acc.get_multidimensional_index(i);
                const uint col_out = sg_col_base + uint(idx[0]);
                const uint r = uint(idx[1]);
                if (r < a.tokens && col_out < a.rows) {
                    partials[(ulong(split) * 8ul + r) * p_stride + col_out] = float(acc[i]);
                }
            }
        }
        if (nax_m8_split_detail::split_arrive_last(counters + group.x, splits, tid, &arrival)) {
            if (sg_live) {
                #pragma unroll
                for (ushort i = 0; i < acc.get_capacity(); ++i) {
                    if (!acc.is_valid_element(i)) continue;
                    const auto idx = acc.get_multidimensional_index(i);
                    const uint col_out = sg_col_base + uint(idx[0]);
                    const uint r = uint(idx[1]);
                    if (r < a.tokens && col_out < a.rows) {
                        float sum = 0.0f;
                        for (uint s = 0u; s < splits; ++s) {
                            sum += (s == split) ? float(acc[i]) : partials[(ulong(s) * 8ul + r) * p_stride + col_out];
                        }
                        write_out(r, col_out, sum);
                    }
                }
            }
            nax_m8_split_detail::split_release(counters + group.x, tid);
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

// 64-thread (2-SG: SG0 = gate, SG1 = up), 32 cols/TG fused gate + up + SwiGLU NAX matmul2d(8, 32, 32) on Q8_tile64.
kernel void nax_m8_dense_q8_swiglu_t64(device const uchar* gate_plane0 [[buffer(0)]], device const ushort* gate_meta [[buffer(1)]],
    device const uchar* up_plane0 [[buffer(2)]], device const ushort* up_meta [[buffer(3)]],
    device const float* input [[buffer(4)]], device float* output [[buffer(5)]],
    constant nax_detail::GroupedMmArgs& a [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]],
    device coherent(device) float* partials [[buffer(8)]], device atomic_uint* counters [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort Rows = 8, OutCols = 32, KS = 32, StageStride = 36;
    constexpr uint APlane = uint(Rows) * StageStride, BPlane = uint(OutCols) * StageStride;
    alignas(16) threadgroup bfloat a_stage[2 * APlane];
    alignas(16) threadgroup half b_stage[2][2 * BPlane];
    threadgroup uint arrival;

    const uint splits = max(1u, a.selected);
    if (!a.tokens || a.tokens > 8u || !a.rows || !a.columns || a.columns % 32u != 0u ||
        splits > 32u || a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        ((a.columns / 32u) % splits) != 0u) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint first_col = group.x * OutCols;
    const uint split = group.y;
    if (first_col >= a.rows || split >= splits) return;

    const uint t64_tile = group.x >> 1u;
    const uint src_col_in_tile = (group.x & 1u) * 32u + uint(lane);
    device const uchar* sg_p0 = sg ? up_plane0 : gate_plane0;
    device const ushort* sg_meta = sg ? up_meta : gate_meta;

    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at0(a_stage, dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at1(a_stage + APlane, dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(b_stage[0] + uint(sg) * BPlane, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(b_stage[1] + uint(sg) * BPlane, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    constexpr auto descriptor = matmul2d_descriptor(Rows, OutCols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;
    auto a_s0 = at0.slice<KS, Rows>(0, 0), a_s1 = at1.slice<KS, Rows>(0, 0);
    auto b_s0 = bt0.slice<KS, OutCols>(0, 0), b_s1 = bt1.slice<KS, OutCols>(0, 0);
    auto acc = operation.get_destination_cooperative_tensor<decltype(a_s0), decltype(b_s0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const uint G = a.columns / KS;
    const uint steps_per_split = splits > 1u ? (G / splits) : G;
    const uint step_begin = split * steps_per_split;
    const uint step_end = step_begin + steps_per_split;
    uint bad = 0u;

    if (step_begin < step_end) {
        nax_detail::Q8Tile64Ipt1 w_next = nax_detail::load_q8_t64_ipt1(sg_p0, sg_meta, G, t64_tile, step_begin, src_col_in_tile);
        for (uint s_rel = 0u; s_rel < steps_per_split; ++s_rel) {
            const uint step = step_begin + s_rel;
            const uint cur_plane = s_rel & 1u;
            nax_m8_split_detail::stage_a_m8_t64(input, in_stride, a.tokens, step * 32u, tid, a_stage + cur_plane * APlane, bad);
            nax_detail::store_q8_t64_ipt1(w_next, b_stage[cur_plane] + uint(sg) * BPlane + uint(lane) * StageStride, bad);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (s_rel + 1u < steps_per_split) {
                w_next = nax_detail::load_q8_t64_ipt1(sg_p0, sg_meta, G, t64_tile, step + 1u, src_col_in_tile);
            }
            if (cur_plane == 0u) operation.run(a_s0, b_s0, acc);
            else operation.run(a_s1, b_s1, acc);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float (*epi)[8][33] = reinterpret_cast<threadgroup float (*)[8][33]>(b_stage);
    if (splits == 1u) {
        if (sg == 0u) {
            #pragma unroll
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto idx = acc.get_multidimensional_index(i);
                epi[0][uint(idx[1])][uint(idx[0])] = float(acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 1u) {
            #pragma unroll
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                const auto idx = acc.get_multidimensional_index(i);
                const uint col_out = first_col + uint(idx[0]);
                const uint r = uint(idx[1]);
                if (r < a.tokens && col_out < a.rows) {
                    const float g0 = epi[0][r][uint(idx[0])];
                    const float u0 = float(acc[i]);
                    bad |= nax_detail::nonfinite_bit(g0) | nax_detail::nonfinite_bit(u0);
                    const float val = (g0 * nax_detail::sigmoid_f32(g0)) * u0;
                    bad |= nax_detail::nonfinite_bit(val);
                    device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(r) * a.output_stride_bytes);
                    y[col_out] = val;
                }
            }
        }
    } else {
        #pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            const auto idx = acc.get_multidimensional_index(i);
            epi[sg][uint(idx[1])][uint(idx[0])] = float(acc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint col_out = first_col + uint(lane);
        const bool col_live = col_out < a.rows;
        const ulong p_stride = ulong((a.rows + 31u) & ~31u);
        if (col_live) {
            #pragma unroll
            for (ushort r = 0; r < 8; ++r) {
                if (r < a.tokens) {
                    partials[((ulong(split) * 2ul + ulong(sg)) * 8ul + ulong(r)) * p_stride + col_out] = epi[sg][r][lane];
                }
            }
        }
        if (nax_m8_split_detail::split_arrive_last(counters + group.x, splits, tid, &arrival)) {
            if (col_live) {
                #pragma unroll
                for (ushort r_sub = 0; r_sub < 4; ++r_sub) {
                    const uint r = uint(sg) * 4u + uint(r_sub);
                    if (r < a.tokens) {
                        float gs = 0.0f, us = 0.0f;
                        for (uint s = 0u; s < splits; ++s) {
                            gs += (s == split) ? epi[0][r][lane] : partials[((ulong(s) * 2ul + 0ul) * 8ul + ulong(r)) * p_stride + col_out];
                            us += (s == split) ? epi[1][r][lane] : partials[((ulong(s) * 2ul + 1ul) * 8ul + ulong(r)) * p_stride + col_out];
                        }
                        bad |= nax_detail::nonfinite_bit(gs) | nax_detail::nonfinite_bit(us);
                        const float val = (gs * nax_detail::sigmoid_f32(gs)) * us;
                        bad |= nax_detail::nonfinite_bit(val);
                        device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(r) * a.output_stride_bytes);
                        y[col_out] = val;
                    }
                }
            }
            nax_m8_split_detail::split_release(counters + group.x, tid);
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_m8_dense_bf16(device const ushort* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::GroupedMmArgs& a [[buffer(3)]],
    device coherent(device) float* partials [[buffer(4)]], device atomic_uint* counters [[buffer(5)]],
    device atomic_uint* error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort Rows = 8, OutCols = 32, KS = 32, StageStride = 32;
    constexpr uint APlane = uint(Rows) * StageStride, BPlane = uint(OutCols) * StageStride;
    alignas(16) threadgroup bfloat a_stage[4 * APlane];
    alignas(16) threadgroup bfloat b_stage[4 * BPlane];
    threadgroup uint arrival;

    const uint splits = max(1u, a.selected);
    if (!a.tokens || a.tokens > 8u || !a.columns || a.columns % 32u != 0u || a.row_bytes != a.columns * 2u ||
        splits > 16u || a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        (splits > 1u && (a.columns % (splits * 128u)) != 0u)) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint first_col = group.x * OutCols;
    const uint split = group.y;
    if (first_col >= a.rows || split >= splits) return;

    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at(a_stage + uint(sg) * APlane, dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> bt(b_stage + uint(sg) * BPlane, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    constexpr auto descriptor = matmul2d_descriptor(Rows, OutCols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;
    auto a_slice = at.slice<KS, Rows>(0, 0);
    auto b_slice = bt.slice<KS, OutCols>(0, 0);
    auto acc = operation.get_destination_cooperative_tensor<decltype(a_slice), decltype(b_slice), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const uint total_steps = a.columns / KS;
    const uint steps_per_split = splits > 1u ? (total_steps / splits) : total_steps;
    uint step = split * steps_per_split;
    const uint step_end = step + steps_per_split;
    const uint b_slice_idx = uint(lane) >> 3u;
    const uint b_k4 = (uint(lane) & 7u) * 4u;
    uint bad = 0u;

    const auto load_bf16_raw = [&](uint s_base, uint chunk_cnt, thread ushort4 (&raw_dst)[8]) {
        #pragma unroll
        for (ushort r = 0; r < 8; ++r) {
            const uint w_col = first_col + uint(sg) * 8u + uint(r);
            raw_dst[r] = (w_col < a.rows && b_slice_idx < chunk_cnt)
                ? ushort4(*reinterpret_cast<device const packed_ushort4*>(
                    weights + ulong(w_col) * a.columns + ulong(s_base * 32u + uint(lane) * 4u)))
                : ushort4(0);
        }
    };

    if (step < step_end) {
        ushort4 raw_h[8];
        load_bf16_raw(step, min(4u, step_end - step), raw_h);
        while (step < step_end) {
            const uint chunk = min(4u, step_end - step);
            nax_m8_split_detail::stage_a_4slices(input, in_stride, a.tokens, step * 32u, chunk, tid, a_stage, bad);
            #pragma unroll
            for (ushort r = 0; r < 8; ++r) {
                const uint col_local = uint(sg) * 8u + uint(r);
                const ushort4 h = (first_col + col_local < a.rows && b_slice_idx < chunk)
                    ? nax_detail::clean_bf16_bits4(raw_h[r], bad)
                    : ushort4(0);
                *reinterpret_cast<threadgroup ushort4*>(b_stage + b_slice_idx * BPlane + col_local * StageStride + b_k4) = h;
            }
            if (step + chunk < step_end) {
                load_bf16_raw(step + chunk, min(4u, step_end - (step + chunk)), raw_h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (uint(sg) < chunk) {
                operation.run(a_slice, b_slice, acc);
            }
            step += chunk;
            if (step < step_end) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float (*red)[8][32] = reinterpret_cast<threadgroup float (*)[8][32]>(b_stage);
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto idx = acc.get_multidimensional_index(i);
        red[sg][uint(idx[1])][uint(idx[0])] = float(acc[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint col_out = first_col + uint(lane);
    const bool col_live = col_out < a.rows;
    const uint r0_out = uint(sg), r1_out = uint(sg) + 4u;
    float own0 = 0.0f, own1 = 0.0f;
    if (col_live && r0_out < a.tokens) {
        own0 = (red[0][r0_out][lane] + red[1][r0_out][lane]) + (red[2][r0_out][lane] + red[3][r0_out][lane]);
    }
    if (col_live && r1_out < a.tokens) {
        own1 = (red[0][r1_out][lane] + red[1][r1_out][lane]) + (red[2][r1_out][lane] + red[3][r1_out][lane]);
    }

    const auto write_out = [&](uint r, float val) {
        bad |= nax_detail::nonfinite_bit(val);
        device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(r) * a.output_stride_bytes);
        y[col_out] = val;
    };

    if (splits == 1u) {
        if (col_live && r0_out < a.tokens) write_out(r0_out, own0);
        if (col_live && r1_out < a.tokens) write_out(r1_out, own1);
    } else {
        const ulong p_stride = ulong((a.rows + 31u) & ~31u);
        if (col_live && r0_out < a.tokens) partials[(ulong(split) * 8ul + r0_out) * p_stride + col_out] = own0;
        if (col_live && r1_out < a.tokens) partials[(ulong(split) * 8ul + r1_out) * p_stride + col_out] = own1;
        if (nax_m8_split_detail::split_arrive_last(counters + group.x, splits, tid, &arrival)) {
            if (col_live && r0_out < a.tokens) {
                float sum = 0.0f;
                for (uint s = 0u; s < splits; ++s) {
                    sum += s == split ? own0 : partials[(ulong(s) * 8ul + r0_out) * p_stride + col_out];
                }
                write_out(r0_out, sum);
            }
            if (col_live && r1_out < a.tokens) {
                float sum = 0.0f;
                for (uint s = 0u; s < splits; ++s) {
                    sum += s == split ? own1 : partials[(ulong(s) * 8ul + r1_out) * p_stride + col_out];
                }
                write_out(r1_out, sum);
            }
            nax_m8_split_detail::split_release(counters + group.x, tid);
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_m8_moe_gate_up_swiglu(device const uchar* gate_weights [[buffer(0)]], device const uchar* up_weights [[buffer(1)]],
    device const float* input [[buffer(2)]], device float* output [[buffer(3)]],
    constant nax_detail::NaxVerifyMoeArgs& a [[buffer(4)]], device const uint* meta [[buffer(5)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort Rows = 8, OutCols = 32, KS = 32, StageStride = 36;
    constexpr uint APlane = uint(Rows) * StageStride, SgBPlane = uint(OutCols) * StageStride, BPlane = 2u * SgBPlane;
    alignas(16) threadgroup bfloat a_stage[2 * APlane];
    alignas(16) threadgroup float4 raw_b[576]; // 9,216 B: stages 2*64*36 halfs in K-loop, then 8*65 floats in SwiGLU epilogue.
    threadgroup half* b_stage = reinterpret_cast<threadgroup half*>(raw_b);

    const uint block_cols = (nax_encoding == 12u || nax_encoding == 13u) ? 256u : 32u;
    const uint block_bytes = nax_encoding == 12u ? 144u : (nax_encoding == 13u ? 176u : (nax_encoding == 8u ? 34u : 24u));
    if (!a.tokens || a.tokens > 8u || !a.selected || a.selected > 16u || !a.experts || a.experts > 512u ||
        !a.columns || a.columns % block_cols != 0u || !a.rows || a.row_bytes != (a.columns / block_cols) * block_bytes ||
        a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        a.expert_stride_bytes != ulong(a.row_bytes) * a.rows) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint num_tiles = meta[1];
    const uint first_col = group.x * OutCols;
    if (group.y >= num_tiles || first_col >= a.rows) return;
    device const uint* rec = meta + 8u + group.y * 10u;
    const uint expert = rec[0], match_count = rec[1];
    if (expert >= a.experts || !match_count || match_count > 8u) {
        if (!tid && !group.x) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    if (match_count == 1u) {
        device const uchar* gate_mat = gate_weights + ulong(expert) * a.expert_stride_bytes;
        device const uchar* up_mat = up_weights + ulong(expert) * a.expert_stride_bytes;
        threadgroup float (*sx)[512] = reinterpret_cast<threadgroup float (*)[512]>(raw_b);
        uint bad = 0u;
        const uint sg_base = first_col + uint(sg) * 16u;
        #pragma unroll
        for (uint sub = 0u; sub < 4u; ++sub) {
            nax_verify_union_detail::moe_gate_up_pass<1, 4, 64u>(gate_mat, up_mat, input, output, a, rec, sg_base + sub * 4u, tid, lane, sx, bad);
        }
        if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
        return;
    }

    uint match_routes[8];
    for (uint m = 0u; m < match_count; ++m) match_routes[m] = rec[2u + m];
    device const uchar* mat_base = (sg == 0 ? gate_weights : up_weights) + ulong(expert) * a.expert_stride_bytes;
    threadgroup half* sg_b_base = b_stage + uint(sg) * SgBPlane;

    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at0(a_stage, dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at1(a_stage + APlane, dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(sg_b_base, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(sg_b_base + BPlane, dextents<int, 2>{KS, OutCols}, array<int, 2>{1, StageStride});
    constexpr auto descriptor = matmul2d_descriptor(Rows, OutCols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;
    auto a0 = at0.slice<KS, Rows>(0, 0), a1 = at1.slice<KS, Rows>(0, 0);
    auto b0 = bt0.slice<KS, OutCols>(0, 0), b1 = bt1.slice<KS, OutCols>(0, 0);
    auto acc = operation.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    const ushort in_r = tid >> 3u, in_k4 = (tid & 7u) * 4u;
    const uint in_src_row = in_r < match_count ? (a.broadcast ? (match_routes[in_r] / a.selected) : match_routes[in_r]) : 0u;
    uint bad = 0u;
    const auto load_a_step = [&](uint k_step, threadgroup bfloat* a_dst) {
        if (in_r < match_count) {
            const float4 src = float4(*reinterpret_cast<device const packed_float4*>(input + ulong(in_src_row) * in_stride + k_step * KS + in_k4));
            *reinterpret_cast<threadgroup bfloat4*>(a_dst + in_r * StageStride + in_k4) = bfloat4(
                nax_detail::clean_bf16(src.x, bad), nax_detail::clean_bf16(src.y, bad),
                nax_detail::clean_bf16(src.z, bad), nax_detail::clean_bf16(src.w, bad));
        } else {
            *reinterpret_cast<threadgroup bfloat4*>(a_dst + in_r * StageStride + in_k4) = bfloat4(0);
        }
    };

    const uint steps = a.columns / KS;
    if ((nax_encoding == 12u || nax_encoding == 13u) && (steps % 2u == 0u)) {
        const ushort sub = lane & 7u, r_in_pass = lane >> 3u;
        const uint pairs = steps >> 1u;
        for (uint pair = 0u; pair < pairs; ++pair) {
            const uint s_even = 2u * pair, s_odd = s_even + 1u;
            load_a_step(s_even, a_stage + (s_even & 1u) * APlane);
            threadgroup half* b_even = sg_b_base + (s_even & 1u) * BPlane;
            half4 w_odd[8];
            #pragma unroll
            for (ushort p = 0; p < 8; ++p) {
                const ushort col = p * 4u + r_in_pass;
                const bool live_col = first_col + col < a.rows;
                device const uchar* w_row = mat_base + ulong(live_col ? first_col + col : 0u) * a.row_bytes;
                half4 w_even;
                nax_grouped_moe_detail::decode4_qk_pair(w_row, pair, sub, live_col, w_even, w_odd[p], bad);
                *reinterpret_cast<threadgroup half4*>(b_even + uint(col) * StageStride + uint(sub) * 4u) = w_even;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (s_even & 1u) operation.run(a1, b1, acc);
            else operation.run(a0, b0, acc);

            load_a_step(s_odd, a_stage + (s_odd & 1u) * APlane);
            threadgroup half* b_odd = sg_b_base + (s_odd & 1u) * BPlane;
            #pragma unroll
            for (ushort p = 0; p < 8; ++p) {
                const ushort col = p * 4u + r_in_pass;
                *reinterpret_cast<threadgroup half4*>(b_odd + uint(col) * StageStride + uint(sub) * 4u) = w_odd[p];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (s_odd & 1u) operation.run(a1, b1, acc);
            else operation.run(a0, b0, acc);
        }
    } else {
        for (uint step = 0u; step < steps; ++step) {
            const uint buf = step & 1u;
            load_a_step(step, a_stage + buf * APlane);
            threadgroup half* b_dst = sg_b_base + buf * BPlane;
            #pragma unroll
            for (ushort c_half = 0; c_half < OutCols; c_half += 16) {
                const ushort col = c_half + (lane >> 1u), ai = lane & 1u;
                const bool live_col = first_col + col < a.rows;
                device const uchar* w_row = mat_base + ulong(live_col ? first_col + col : 0u) * a.row_bytes;
                const half4x4 w16 = nax_detail::decode16_any(w_row, step * 2u + ai, a.columns, live_col, bad);
                threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(b_dst + uint(col) * StageStride + ai * 16u);
                #pragma unroll
                for (ushort i = 0; i < 4; ++i) dst[i] = w16[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (buf) operation.run(a1, b1, acc);
            else operation.run(a0, b0, acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint EpilogueStride = 65u;
    threadgroup float* gu_sh = reinterpret_cast<threadgroup float*>(raw_b);
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto idx = acc.get_multidimensional_index(i);
        gu_sh[uint(idx[1]) * EpilogueStride + uint(sg) * OutCols + uint(idx[0])] = float(acc[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (first_col + uint(lane) < a.rows) {
        for (uint m = uint(sg); m < match_count; m += 2u) {
            const float g_val = gu_sh[m * EpilogueStride + uint(lane)];
            const float u_val = gu_sh[m * EpilogueStride + OutCols + uint(lane)];
            const float val = (g_val * nax_detail::sigmoid_f32(g_val)) * u_val;
            bad |= nax_detail::nonfinite_bit(g_val) | nax_detail::nonfinite_bit(u_val) | nax_detail::nonfinite_bit(val);
            device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(match_routes[m]) * a.output_stride_bytes);
            y[first_col + uint(lane)] = val;
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void nax_m8_moe_down(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant nax_detail::NaxVerifyMoeArgs& a [[buffer(3)]],
    device const uint* meta [[buffer(4)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr ushort Rows = 8, SgCols = 32, OutCols = 64, KS = 32, StageStride = 36;
    constexpr uint APlane = uint(Rows) * StageStride, SgBPlane = uint(SgCols) * StageStride, BPlane = 2u * SgBPlane;
    alignas(16) threadgroup bfloat a_stage[2 * APlane];
    alignas(16) threadgroup half b_stage[2 * BPlane];

    const uint block_cols = (nax_encoding == 12u || nax_encoding == 13u) ? 256u : 32u;
    const uint block_bytes = nax_encoding == 12u ? 144u : (nax_encoding == 13u ? 176u : (nax_encoding == 8u ? 34u : 24u));
    if (!a.tokens || a.tokens > 8u || !a.selected || a.selected > 16u || !a.experts || a.experts > 512u ||
        !a.columns || a.columns % block_cols != 0u || !a.rows || a.row_bytes != (a.columns / block_cols) * block_bytes ||
        a.input_stride_bytes % 4u != 0u || a.output_stride_bytes % 4u != 0u ||
        a.expert_stride_bytes != ulong(a.row_bytes) * a.rows) {
        if (!tid && !group.x && !group.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint num_tiles = meta[5];
    const uint first_col = group.x * OutCols;
    if (group.y >= num_tiles || first_col >= a.rows) return;
    device const uint* rec = meta + 8u + group.y * 10u;
    const uint expert = rec[0], match_count = rec[1];
    if (expert >= a.experts || !match_count || match_count > 8u) {
        if (!tid && !group.x) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    device const uchar* matrix = weights + ulong(expert) * a.expert_stride_bytes;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const uint sg_col_base = first_col + uint(sg) * SgCols;
    const bool sg_active = sg_col_base < a.rows;

    if (match_count == 1u) {
        threadgroup float (*sx)[512] = reinterpret_cast<threadgroup float (*)[512]>(b_stage);
        uint bad = 0u;
        if (nax_encoding == 12u || nax_encoding == 13u) {
            #pragma unroll
            for (uint sub = 0u; sub < 8u; ++sub) {
                nax_verify_union_detail::moe_down_pass<1, 4, 64u>(matrix, input, output, a, rec, sg_col_base + sub * 4u, tid, lane, sx, bad);
            }
        } else {
            #pragma unroll
            for (uint sub = 0u; sub < 4u; ++sub) {
                nax_verify_union_detail::moe_down_pass<1, 8, 64u>(matrix, input, output, a, rec, sg_col_base + sub * 8u, tid, lane, sx, bad);
            }
        }
        if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
        return;
    }

    uint match_routes[8];
    for (uint m = 0u; m < match_count; ++m) match_routes[m] = rec[2u + m];
    threadgroup half* sg_b_base = b_stage + uint(sg) * SgBPlane;

    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at0(a_stage, dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> at1(a_stage + APlane, dextents<int, 2>{KS, Rows}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(sg_b_base, dextents<int, 2>{KS, SgCols}, array<int, 2>{1, StageStride});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(sg_b_base + BPlane, dextents<int, 2>{KS, SgCols}, array<int, 2>{1, StageStride});
    constexpr auto descriptor = matmul2d_descriptor(Rows, SgCols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision, "BF16/FP16 inputs with exact FP32 accumulation");
    matmul2d<descriptor, execution_simdgroups<1>> operation;
    auto a0 = at0.slice<KS, Rows>(0, 0), a1 = at1.slice<KS, Rows>(0, 0);
    auto b0 = bt0.slice<KS, SgCols>(0, 0), b1 = bt1.slice<KS, SgCols>(0, 0);
    auto acc = operation.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    const ushort in_r = tid >> 3u, in_k4 = (tid & 7u) * 4u;
    const uint in_src_row = in_r < match_count ? (a.broadcast ? (match_routes[in_r] / a.selected) : match_routes[in_r]) : 0u;
    uint bad = 0u;
    const auto load_a_step = [&](uint k_step, threadgroup bfloat* a_dst) {
        if (in_r < match_count) {
            const float4 src = float4(*reinterpret_cast<device const packed_float4*>(input + ulong(in_src_row) * in_stride + k_step * KS + in_k4));
            *reinterpret_cast<threadgroup bfloat4*>(a_dst + in_r * StageStride + in_k4) = bfloat4(
                nax_detail::clean_bf16(src.x, bad), nax_detail::clean_bf16(src.y, bad),
                nax_detail::clean_bf16(src.z, bad), nax_detail::clean_bf16(src.w, bad));
        } else {
            *reinterpret_cast<threadgroup bfloat4*>(a_dst + in_r * StageStride + in_k4) = bfloat4(0);
        }
    };

    const uint steps = a.columns / KS;
    uint step = 0u;
    if (nax_layout == 2u) {
        const uint blocks = a.columns / 32u;
        const ushort j = lane >> 1u, odd = lane & 1u;
        const ushort h_row_sub = lane >> 2u, h_sub = lane & 3u;
        for (; step + 16u <= steps; step += 16u) {
            const uint run = step / 16u;
            const ulong quant_base = 8ul * ulong(blocks) + ulong(run) * 256ul + ulong(lane) * 8ul;
            uint2 q_lo8[16], q_hi8[16];
            if (sg_active) {
                #pragma unroll
                for (ushort m = 0; m < 16; ++m) {
                    const uint c0 = sg_col_base + 2u * uint(m);
                    const uint c1 = c0 + 1u;
                    const uint2 v0 = c0 < a.rows
                        ? as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(matrix + ulong(c0) * a.row_bytes + quant_base)))
                        : uint2(0u);
                    const uint2 v1 = c1 < a.rows
                        ? as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(matrix + ulong(c1) * a.row_bytes + quant_base)))
                        : uint2(0u);
                    const uint2 send = odd ? v0 : v1;
                    const uint2 recv = simd_shuffle_xor(send, 1u);
                    q_lo8[m] = odd ? recv : v0;
                    q_hi8[m] = odd ? v1 : recv;
                }
            }
            #pragma unroll
            for (ushort grp = 0; grp < 4; ++grp) {
                uint2 hdr[4] = {uint2(0u), uint2(0u), uint2(0u), uint2(0u)};
                if (sg_active) {
                    const uint h_blk = step + uint(grp) * 4u + uint(h_sub);
                    #pragma unroll
                    for (ushort g = 0; g < 4; ++g) {
                        const uint h_col = sg_col_base + uint(g) * 8u + uint(h_row_sub);
                        if (h_col < a.rows) {
                            uint2 h = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(matrix + ulong(h_col) * a.row_bytes + ulong(h_blk) * 8ul)));
                            const half2 dm_chk = as_type<half2>(h.x);
                            if (nax_detail::nonfinite_half_bit(dm_chk.x) || nax_detail::nonfinite_half_bit(dm_chk.y)) { bad |= 1u; h.x = 0u; }
                            hdr[g] = h;
                        }
                    }
                }
                #pragma unroll
                for (ushort sub_in_4 = 0; sub_in_4 < 4; ++sub_in_4) {
                    const ushort sub = grp * 4u + sub_in_4;
                    const uint k_step = step + uint(sub);
                    const uint buf = k_step & 1u;
                    load_a_step(k_step, a_stage + buf * APlane);
                    if (sg_active) {
                        threadgroup half* b_dst = sg_b_base + buf * BPlane;
                        const ushort word_idx = (sub & 7u) >> 2u;
                        const uint byte_shift = (uint(sub) & 3u) * 8u;
                        #pragma unroll
                        for (ushort m = 0; m < 16; ++m) {
                            const ushort c_local = 2u * m + odd;
                            const ushort g = m >> 2u;
                            const ushort src_lane = ushort(((2u * (m & 3u) + odd) << 2u) + sub_in_4);
                            const uint dhx = simd_shuffle(hdr[g].x, src_lane);
                            const uint dhy = simd_shuffle(hdr[g].y, src_lane);
                            const half2 dm = as_type<half2>(dhx);
                            const float d = float(dm.x), mn = float(dm.y);
                            const uint q_word = sub < 8u ? (word_idx ? q_lo8[m].y : q_lo8[m].x) : (word_idx ? q_hi8[m].y : q_hi8[m].x);
                            const uint byte = (q_word >> byte_shift) & 255u;
                            const uint q_lo = (byte & 15u) | (((dhy >> uint(j)) & 1u) << 4u);
                            const uint q_hi = ((byte >> 4u) & 15u) | (((dhy >> (uint(j) + 16u)) & 1u) << 4u);
                            const half h_lo = half(d * float(q_lo) + mn);
                            const half h_hi = half(d * float(q_hi) + mn);
                            bad |= nax_detail::nonfinite_half_bit(h_lo) | nax_detail::nonfinite_half_bit(h_hi);
                            b_dst[uint(c_local) * StageStride + uint(j)] = h_lo;
                            b_dst[uint(c_local) * StageStride + 16u + uint(j)] = h_hi;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (sg_active) {
                        if (buf) operation.run(a1, b1, acc);
                        else operation.run(a0, b0, acc);
                    }
                }
            }
        }
        for (; step + 4u <= steps; step += 4u) {
            const uint run = step / 16u;
            const uint width = min(16u, blocks - run * 16u);
            const ulong quant_off = 8ul * ulong(blocks) + ulong(run) * 256ul + ulong(j) * ulong(width) + ulong(step % 16u);
            uint w_tail[16];
            uint2 hdr[4] = {uint2(0u), uint2(0u), uint2(0u), uint2(0u)};
            if (sg_active) {
                #pragma unroll
                for (ushort m = 0; m < 16; ++m) {
                    const uint w_col = sg_col_base + 2u * uint(m) + uint(odd);
                    w_tail[m] = w_col < a.rows
                        ? as_type<uint>(uchar4(*reinterpret_cast<device const packed_uchar4*>(matrix + ulong(w_col) * a.row_bytes + quant_off)))
                        : 0u;
                }
                const uint h_blk = step + uint(h_sub);
                #pragma unroll
                for (ushort g = 0; g < 4; ++g) {
                    const uint h_col = sg_col_base + uint(g) * 8u + uint(h_row_sub);
                    if (h_col < a.rows) {
                        uint2 h = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(matrix + ulong(h_col) * a.row_bytes + ulong(h_blk) * 8ul)));
                        const half2 dm_chk = as_type<half2>(h.x);
                        if (nax_detail::nonfinite_half_bit(dm_chk.x) || nax_detail::nonfinite_half_bit(dm_chk.y)) { bad |= 1u; h.x = 0u; }
                        hdr[g] = h;
                    }
                }
            }
            #pragma unroll
            for (ushort sub = 0; sub < 4; ++sub) {
                const uint k_step = step + uint(sub);
                const uint buf = k_step & 1u;
                load_a_step(k_step, a_stage + buf * APlane);
                if (sg_active) {
                    threadgroup half* b_dst = sg_b_base + buf * BPlane;
                    const uint byte_shift = uint(sub) * 8u;
                    #pragma unroll
                    for (ushort m = 0; m < 16; ++m) {
                        const ushort c_local = 2u * m + odd;
                        const ushort g = m >> 2u;
                        const ushort src_lane = ushort(((2u * (m & 3u) + odd) << 2u) + sub);
                        const uint dhx = simd_shuffle(hdr[g].x, src_lane);
                        const uint dhy = simd_shuffle(hdr[g].y, src_lane);
                        const half2 dm = as_type<half2>(dhx);
                        const float d = float(dm.x), mn = float(dm.y);
                        const uint byte = (w_tail[m] >> byte_shift) & 255u;
                        const uint q_lo = (byte & 15u) | (((dhy >> uint(j)) & 1u) << 4u);
                        const uint q_hi = ((byte >> 4u) & 15u) | (((dhy >> (uint(j) + 16u)) & 1u) << 4u);
                        const half h_lo = half(d * float(q_lo) + mn);
                        const half h_hi = half(d * float(q_hi) + mn);
                        bad |= nax_detail::nonfinite_half_bit(h_lo) | nax_detail::nonfinite_half_bit(h_hi);
                        b_dst[uint(c_local) * StageStride + uint(j)] = h_lo;
                        b_dst[uint(c_local) * StageStride + 16u + uint(j)] = h_hi;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg_active) {
                    if (buf) operation.run(a1, b1, acc);
                    else operation.run(a0, b0, acc);
                }
            }
        }
    } else if (nax_layout == 1u && (a.columns & 255u) == 0u) {
        const uint blocks = a.columns / 32u;
        const uint sc_col = sg_col_base + uint(lane);
        for (; step + 8u <= steps; step += 8u) {
            const uint run = step / 16u;
            const uint width = min(16u, blocks - run * 16u);
            const ulong quant_off = 2ul * ulong(blocks) + ulong(run) * 512ul + ulong(lane) * ulong(width) + ulong(step % 16u);
            half sc8[8] = {half(0), half(0), half(0), half(0), half(0), half(0), half(0), half(0)};
            uint2 q_half[32];
            if (sg_active) {
                if (sc_col < a.rows) {
                    device const uchar* sc_ptr = matrix + ulong(sc_col) * a.row_bytes + 2ul * ulong(step);
                    const half4 s0 = half4(*reinterpret_cast<device const packed_half4*>(sc_ptr));
                    const half4 s1 = half4(*reinterpret_cast<device const packed_half4*>(sc_ptr + 8ul));
                    sc8[0] = s0.x; sc8[1] = s0.y; sc8[2] = s0.z; sc8[3] = s0.w;
                    sc8[4] = s1.x; sc8[5] = s1.y; sc8[6] = s1.z; sc8[7] = s1.w;
                    #pragma unroll
                    for (ushort i = 0; i < 8; ++i) {
                        if (nax_detail::nonfinite_half_bit(sc8[i])) { bad |= 1u; sc8[i] = half(0); }
                    }
                }
                #pragma unroll
                for (ushort c = 0; c < 32; ++c) {
                    const uint w_col = sg_col_base + uint(c);
                    q_half[c] = w_col < a.rows
                        ? uint2(*reinterpret_cast<device const packed_uint2*>(matrix + ulong(w_col) * a.row_bytes + quant_off))
                        : uint2(0u);
                }
            }
            #pragma unroll
            for (ushort sub = 0; sub < 8; ++sub) {
                const uint k_step = step + uint(sub);
                const uint buf = k_step & 1u;
                load_a_step(k_step, a_stage + buf * APlane);
                if (sg_active) {
                    threadgroup half* b_dst = sg_b_base + buf * BPlane;
                    const uint shift = (uint(sub) & 3u) * 8u;
                    const half my_sc = sc8[sub];
                    #pragma unroll
                    for (ushort c = 0; c < 32; ++c) {
                        const float d = float(simd_broadcast(my_sc, c));
                        const uint word = sub < 4u ? q_half[c].x : q_half[c].y;
                        const char q = as_type<char>(uchar((word >> shift) & 0xffu));
                        const half h = half(float(q) * d);
                        bad |= nax_detail::nonfinite_half_bit(h);
                        b_dst[uint(c) * StageStride + uint(lane)] = h;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg_active) {
                    if (buf) operation.run(a1, b1, acc);
                    else operation.run(a0, b0, acc);
                }
            }
        }
    }

    for (; step < steps; ++step) {
        const uint buf = step & 1u;
        load_a_step(step, a_stage + buf * APlane);
        if (sg_active) {
            threadgroup half* b_dst = sg_b_base + buf * BPlane;
            #pragma unroll
            for (ushort c_half = 0; c_half < SgCols; c_half += 16) {
                const ushort col = c_half + (lane >> 1u), ai = lane & 1u;
                const bool live_col = sg_col_base + col < a.rows;
                device const uchar* w_row = matrix + ulong(live_col ? sg_col_base + col : 0u) * a.row_bytes;
                const half4x4 w16 = nax_detail::decode16_any(w_row, step * 2u + ai, a.columns, live_col, bad);
                threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(b_dst + uint(col) * StageStride + ai * 16u);
                #pragma unroll
                for (ushort i = 0; i < 4; ++i) dst[i] = w16[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_active) {
            if (buf) operation.run(a1, b1, acc);
            else operation.run(a0, b0, acc);
        }
    }

    if (sg_active) {
        #pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            const auto idx = acc.get_multidimensional_index(i);
            const uint col_out = sg_col_base + uint(idx[0]), m = uint(idx[1]);
            if (m < match_count && col_out < a.rows) {
                const float val = float(acc[i]);
                bad |= nax_detail::nonfinite_bit(val);
                device float* y = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(match_routes[m]) * a.output_stride_bytes);
                y[col_out] = val;
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}
