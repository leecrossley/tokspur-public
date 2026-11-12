// Register top-k helpers (simd_best_head, top_insert and top_pop) adapted from Splash 1.4.0, commit 1482f6bbc2abf12375332725302019594150b6cb, runtime/metal/kernels/decode/sampling.metal (Apache-2.0, see NOTICE).
#include <metal_stdlib>
using namespace metal;

namespace logit_topk_detail {

struct LogitTopKArgs {
    uint rows;
    uint vocabulary;
    uint tiles;
    uint tile_span;
    ulong input_stride_bytes;
    ulong reserved;
};
static_assert(sizeof(LogitTopKArgs) == 32, "LogitTopKArgs ABI");

struct LogitTopKRowOut {
    uint vocabulary;
    uint count;
    uint first_nonfinite_col;
    uint first_subnormal_col;
    uint ids[64];
    float logits[64];
};
static_assert(sizeof(LogitTopKRowOut) == 528, "LogitTopKRowOut ABI");

inline bool topk_beats(float va, uint ida, float vb, uint idb) {
    if (ida == 0xffffffffu) return false;
    if (idb == 0xffffffffu) return true;
    const float ca = (va == 0.0f) ? 0.0f : va;
    const float cb = (vb == 0.0f) ? 0.0f : vb;
    return (ca > cb) || (ca == cb && ida < idb);
}

inline void simd_best_head(float val, uint id, ushort lane, thread float& out_val, thread uint& out_id) {
    const bool valid = (id != 0xffffffffu);
    if (!simd_any(valid)) {
        out_val = -INFINITY;
        out_id = 0xffffffffu;
        return;
    }
    const float clean = valid ? ((val == 0.0f) ? 0.0f : val) : -INFINITY;
    const float max_clean = simd_max(clean);
    out_id = simd_min((valid && clean == max_clean) ? id : 0xffffffffu);
    const ushort win_lane = simd_min((valid && id == out_id) ? lane : ushort(32));
    out_val = simd_shuffle(val, win_lane);
}

template <uint Count>
inline void top_insert_ascending_finite(thread float (&values)[Count], thread uint (&ids)[Count], float raw, uint col) {
    values[Count - 1u] = raw;
    ids[Count - 1u] = col;
    #pragma unroll
    for (int i = int(Count) - 2; i >= 0; --i) {
        if (raw > values[i]) {
            values[i + 1] = values[i];
            ids[i + 1] = ids[i];
            values[i] = raw;
            ids[i] = col;
        }
    }
}

template <uint Count>
inline void top_insert(thread float (&values)[Count], thread uint (&ids)[Count], float value, uint id) {
    #pragma unroll
    for (uint i = 0u; i < Count; ++i) {
        if (topk_beats(value, id, values[i], ids[i])) {
            const float prev_v = values[i];
            const uint prev_i = ids[i];
            values[i] = value;
            ids[i] = id;
            value = prev_v;
            id = prev_i;
        }
    }
}

template <uint Count>
inline void top_pop(thread float (&values)[Count], thread uint (&ids)[Count]) {
    #pragma unroll
    for (uint i = 0u; i + 1u < Count; ++i) {
        values[i] = values[i + 1u];
        ids[i] = ids[i + 1u];
    }
    values[Count - 1u] = -INFINITY;
    ids[Count - 1u] = 0xffffffffu;
}

} // namespace logit_topk_detail

// Stage 1: exact per-tile top-64 reduction plus full-row non-finite and FP_SUBNORMAL first-column tracking.
kernel void logit_topk_stage1(
    device const uchar* logits_bytes [[buffer(0)]],
    device logit_topk_detail::LogitTopKRowOut* partials [[buffer(1)]],
    constant logit_topk_detail::LogitTopKArgs& a [[buffer(2)]],
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint MaxVocab = 1u << 20;
    constexpr uint MaxTiles = 256u;
    constexpr uint MaxTileSpan = 4096u;
    constexpr uint Threads = 256u;
    constexpr uint MaxPerThread = MaxTileSpan / Threads;
    constexpr uint TopK = 64u;

    if (!a.rows || a.rows > 2048u || !a.vocabulary || a.vocabulary > MaxVocab ||
        !a.tiles || a.tiles > MaxTiles || !a.tile_span || a.tile_span > MaxTileSpan ||
        ulong(a.tiles) * ulong(a.tile_span) < ulong(a.vocabulary) ||
        (a.input_stride_bytes % 4u) != 0u || a.input_stride_bytes < ulong(a.vocabulary) * 4ul) {
        if (!group.x && !group.y && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint tile = group.x;
    const uint row = group.y;
    if (tile >= a.tiles || row >= a.rows) return;

    device logit_topk_detail::LogitTopKRowOut& out_part = partials[ulong(row) * a.tiles + tile];
    const uint begin = tile * a.tile_span;
    const uint end = begin < a.vocabulary ? min(a.vocabulary, begin + a.tile_span) : begin;
    const uint tile_len = end - begin;
    if (tile_len == 0u) {
        if (tid == 0u) {
            out_part.vocabulary = a.vocabulary;
            out_part.count = 0u;
            out_part.first_nonfinite_col = 0xffffffffu;
            out_part.first_subnormal_col = 0xffffffffu;
        }
        return;
    }

    device const float* row_logits = reinterpret_cast<device const float*>(logits_bytes + ulong(row) * a.input_stride_bytes);
    uint local_nf = 0xffffffffu;
    uint local_sn = 0xffffffffu;
    float local_vals[MaxPerThread];
    uint local_ids[MaxPerThread];
    #pragma unroll
    for (uint i = 0u; i < MaxPerThread; ++i) {
        local_vals[i] = -INFINITY;
        local_ids[i] = 0xffffffffu;
    }

    #pragma unroll
    for (uint p = 0u; p < MaxPerThread; ++p) {
        const uint col = begin + p * Threads + tid;
        if (col < end) {
            const float raw = row_logits[col];
            const uint bits = as_type<uint>(raw);
            const uint exp_bits = bits & 0x7f800000u;
            const uint mant_bits = bits & 0x007fffffu;
            if (exp_bits == 0x7f800000u) {
                local_nf = min(local_nf, col);
            } else {
                if (exp_bits == 0u && mant_bits != 0u) {
                    local_sn = min(local_sn, col);
                }
                if (raw > local_vals[MaxPerThread - 1u]) {
                    logit_topk_detail::top_insert_ascending_finite<MaxPerThread>(local_vals, local_ids, raw, col);
                }
            }
        }
    }

    threadgroup uint shared_bad[2][8];
    threadgroup float sg_vals[8][TopK];
    threadgroup uint sg_ids[8][TopK];
    threadgroup uint sg_cnt[8];

    const uint sg_nf = simd_min(local_nf);
    const uint sg_sn = simd_min(local_sn);
    if (!lane) {
        shared_bad[0][sg] = sg_nf;
        shared_bad[1][sg] = sg_sn;
    }

    const uint target_k = min(TopK, tile_len);
    uint sg_emitted = 0u;
    for (uint r = 0u; r < target_k; ++r) {
        const float cur_val = local_vals[0];
        const uint cur_id = local_ids[0];
        float simd_val;
        uint simd_id;
        logit_topk_detail::simd_best_head(cur_val, cur_id, lane, simd_val, simd_id);
        if (simd_id == 0xffffffffu) break;
        if (!lane) {
            sg_vals[sg][r] = simd_val;
            sg_ids[sg][r] = simd_id;
        }
        if (cur_id == simd_id) {
            logit_topk_detail::top_pop<MaxPerThread>(local_vals, local_ids);
        }
        ++sg_emitted;
    }
    if (!lane) {
        sg_cnt[sg] = sg_emitted;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        if (lane == 0u) {
            uint tile_nf = shared_bad[0][0];
            uint tile_sn = shared_bad[1][0];
            #pragma unroll
            for (uint s = 1u; s < 8u; ++s) {
                tile_nf = min(tile_nf, shared_bad[0][s]);
                tile_sn = min(tile_sn, shared_bad[1][s]);
            }
            out_part.vocabulary = a.vocabulary;
            out_part.first_nonfinite_col = tile_nf;
            out_part.first_subnormal_col = tile_sn;
        }
        const uint cnt = (lane < 8u) ? sg_cnt[lane] : 0u;
        uint cur = 0u;
        uint emitted = 0u;
        float out_v0 = -INFINITY, out_v1 = -INFINITY;
        uint out_i0 = 0u, out_i1 = 0u;
        for (uint rank = 0u; rank < target_k; ++rank) {
            const bool active = (cur < cnt);
            const float v = active ? sg_vals[lane][cur] : -INFINITY;
            const uint id = active ? sg_ids[lane][cur] : 0xffffffffu;
            float best_val;
            uint best_id;
            logit_topk_detail::simd_best_head(v, id, lane, best_val, best_id);
            if (best_id == 0xffffffffu) break;
            if (uint(lane) == rank) {
                out_i0 = best_id;
                out_v0 = best_val;
            }
            if (uint(lane) + 32u == rank) {
                out_i1 = best_id;
                out_v1 = best_val;
            }
            if (active && id == best_id) ++cur;
            ++emitted;
        }
        if (uint(lane) < emitted) {
            out_part.ids[lane] = out_i0;
            out_part.logits[lane] = out_v0;
        }
        if (uint(lane) + 32u < emitted) {
            out_part.ids[uint(lane) + 32u] = out_i1;
            out_part.logits[uint(lane) + 32u] = out_v1;
        }
        if (lane == 0u) {
            out_part.count = emitted;
        }
    }
}

// Stage 2: merge up to 256 pre-sorted tile summaries into one exact sorted top-64 summary per row.
kernel void logit_topk_stage2(
    device const logit_topk_detail::LogitTopKRowOut* partials [[buffer(0)]],
    device logit_topk_detail::LogitTopKRowOut* out_rows [[buffer(1)]],
    constant logit_topk_detail::LogitTopKArgs& a [[buffer(2)]],
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint MaxVocab = 1u << 20;
    constexpr uint MaxTiles = 256u;
    constexpr uint TopK = 64u;

    if (!a.rows || a.rows > 2048u || !a.vocabulary || a.vocabulary > MaxVocab ||
        !a.tiles || a.tiles > MaxTiles) {
        if (!group.x && !group.y && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }

    const uint row = group.y;
    if (group.x > 0u || row >= a.rows) return;

    device const logit_topk_detail::LogitTopKRowOut* row_parts = partials + ulong(row) * a.tiles;
    device logit_topk_detail::LogitTopKRowOut& out_row = out_rows[row];

    const bool active_tile = tid < a.tiles;
    const uint local_nf = active_tile ? row_parts[tid].first_nonfinite_col : 0xffffffffu;
    const uint local_sn = active_tile ? row_parts[tid].first_subnormal_col : 0xffffffffu;
    const uint tile_count = active_tile ? min(TopK, row_parts[tid].count) : 0u;

    threadgroup uint shared_bad[2][8];
    threadgroup float sg_vals[8][TopK];
    threadgroup uint sg_ids[8][TopK];
    threadgroup uint sg_cnt[8];

    const uint sg_nf = simd_min(local_nf);
    const uint sg_sn = simd_min(local_sn);
    if (!lane) {
        shared_bad[0][sg] = sg_nf;
        shared_bad[1][sg] = sg_sn;
    }

    const uint target_k = min(TopK, a.vocabulary);
    uint sg_emitted = 0u;
    if (uint(sg) * 32u < a.tiles) {
        uint cursor = 0u;
        for (uint r = 0u; r < target_k; ++r) {
            const float cur_val = (cursor < tile_count) ? row_parts[tid].logits[cursor] : -INFINITY;
            const uint cur_id = (cursor < tile_count) ? row_parts[tid].ids[cursor] : 0xffffffffu;
            float simd_val;
            uint simd_id;
            logit_topk_detail::simd_best_head(cur_val, cur_id, lane, simd_val, simd_id);
            if (simd_id == 0xffffffffu) break;
            if (!lane) {
                sg_vals[sg][r] = simd_val;
                sg_ids[sg][r] = simd_id;
            }
            if (cur_id == simd_id) ++cursor;
            ++sg_emitted;
        }
    }
    if (!lane) {
        sg_cnt[sg] = sg_emitted;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        if (lane == 0u) {
            uint row_nf = shared_bad[0][0];
            uint row_sn = shared_bad[1][0];
            #pragma unroll
            for (uint s = 1u; s < 8u; ++s) {
                row_nf = min(row_nf, shared_bad[0][s]);
                row_sn = min(row_sn, shared_bad[1][s]);
            }
            out_row.vocabulary = a.vocabulary;
            out_row.first_nonfinite_col = row_nf;
            out_row.first_subnormal_col = row_sn;
        }
        const uint cnt = (lane < 8u) ? sg_cnt[lane] : 0u;
        uint cur = 0u;
        uint emitted = 0u;
        float out_v0 = -INFINITY, out_v1 = -INFINITY;
        uint out_i0 = 0u, out_i1 = 0u;
        for (uint rank = 0u; rank < target_k; ++rank) {
            const bool active = (cur < cnt);
            const float v = active ? sg_vals[lane][cur] : -INFINITY;
            const uint id = active ? sg_ids[lane][cur] : 0xffffffffu;
            float best_val;
            uint best_id;
            logit_topk_detail::simd_best_head(v, id, lane, best_val, best_id);
            if (best_id == 0xffffffffu) break;
            if (uint(lane) == rank) {
                out_i0 = best_id;
                out_v0 = best_val;
            }
            if (uint(lane) + 32u == rank) {
                out_i1 = best_id;
                out_v1 = best_val;
            }
            if (active && id == best_id) ++cur;
            ++emitted;
        }
        out_row.ids[lane] = (uint(lane) < emitted) ? out_i0 : 0u;
        out_row.logits[lane] = (uint(lane) < emitted) ? out_v0 : -INFINITY;
        out_row.ids[uint(lane) + 32u] = (uint(lane) + 32u < emitted) ? out_i1 : 0u;
        out_row.logits[uint(lane) + 32u] = (uint(lane) + 32u < emitted) ? out_v1 : -INFINITY;
        if (lane == 0u) {
            out_row.count = emitted;
        }
    }
}
