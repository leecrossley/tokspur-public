// SPDX-License-Identifier: MIT
// Ported and adapted from mlx_vlm/models/qwen4_exp/qsa_kernel.py (mlx-vlm 0.7.1, lines 22-166, qwen4_exp_qsa_sparse_attention_*).
//
// Copyright (c) 2025 Prince Canuma
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <metal_stdlib>
using namespace metal;

inline float qsa_sparse_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

template <uint BN>
inline void qsa_sparse_core(
    device const float* query_head,
    device const ushort* keys,
    device const ushort* values,
    device const uint* row_ids,
    device float* output_head,
    uint4 s,
    uint prefix,
    uint block_budget,
    device atomic_uint* error,
    ushort subgroup,
    ushort lane,
    ushort tid,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    threadgroup uint* sg_errors,
    threadgroup float* partials) {
    const uint blocks = prefix / s.w;
    const uint selected = min(blocks, block_budget);
    const uint selected_tokens = selected * s.w;
    const uint tokens = selected_tokens + (prefix % s.w);
    const ulong row_width = ulong(s.y) * ulong(s.z);
    const float scale = rsqrt(float(s.z));

    float q[8];
    float o[8];
    uint local_err = 0u;
    #pragma unroll
    for (uint c = 0; c < 8; ++c) {
        const uint d = c * 32u + uint(lane);
        if (d < s.z) {
            const float qv = query_head[d];
            if (!isfinite(qv)) local_err |= 1u;
            q[c] = qv;
        } else {
            q[c] = 0.0f;
        }
        o[c] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;
    const bool dense_short = (blocks <= block_budget);

    if (s.z == 256u) {
        for (uint item = uint(subgroup); item < tokens; item += BN) {
            uint token;
            if (dense_short) {
                token = item;
            } else if (item < selected_tokens) {
                const uint slot = (s.w == 4u) ? (item >> 2) : (item / s.w);
                const uint rem = (s.w == 4u) ? (item & 3u) : (item % s.w);
                const uint id = row_ids[slot];
                if (id >= blocks || (slot && id <= row_ids[slot - 1u])) {
                    local_err |= 4u;
                    continue;
                }
                token = id * s.w + rem;
            } else {
                token = blocks * s.w + (item - selected_tokens);
            }

            device const ushort* k_ptr = keys + ulong(token) * row_width;
            device const ushort* v_ptr = values + ulong(token) * row_width;
            float dot = 0.0f;
            #pragma unroll
            for (uint c = 0; c < 8; ++c) {
                const ushort kbits = k_ptr[c * 32u + uint(lane)];
                if ((kbits & 0x7f80u) == 0x7f80u) local_err |= 1u;
                dot = fma(q[c], qsa_sparse_bf16(kbits), dot);
            }
            const float score = simd_sum(dot) * scale;
            if (!isfinite(score)) {
                local_err |= 1u;
                continue;
            }

            float v_reg[8];
            #pragma unroll
            for (uint c = 0; c < 8; ++c) {
                const ushort vbits = v_ptr[c * 32u + uint(lane)];
                if ((vbits & 0x7f80u) == 0x7f80u) local_err |= 1u;
                v_reg[c] = qsa_sparse_bf16(vbits);
            }

            if (score > max_score) {
                const float factor = exp(max_score - score);
                max_score = score;
                sum_exp_score = fma(sum_exp_score, factor, 1.0f);
                #pragma unroll
                for (uint c = 0; c < 8; ++c) o[c] = fma(o[c], factor, v_reg[c]);
            } else {
                const float factor = exp(score - max_score);
                sum_exp_score += factor;
                #pragma unroll
                for (uint c = 0; c < 8; ++c) o[c] = fma(factor, v_reg[c], o[c]);
            }
        }
    } else {
        for (uint item = uint(subgroup); item < tokens; item += BN) {
            uint token;
            if (dense_short) {
                token = item;
            } else if (item < selected_tokens) {
                const uint slot = item / s.w;
                const uint id = row_ids[slot];
                if (id >= blocks || (slot && id <= row_ids[slot - 1u])) {
                    local_err |= 4u;
                    continue;
                }
                token = id * s.w + (item % s.w);
            } else {
                token = blocks * s.w + (item - selected_tokens);
            }

            device const ushort* k_ptr = keys + ulong(token) * row_width;
            device const ushort* v_ptr = values + ulong(token) * row_width;
            float dot = 0.0f;
            for (uint c = 0; c < 8; ++c) {
                const uint d = c * 32u + uint(lane);
                if (d < s.z) {
                    const ushort kbits = k_ptr[d];
                    if ((kbits & 0x7f80u) == 0x7f80u) local_err |= 1u;
                    dot = fma(q[c], qsa_sparse_bf16(kbits), dot);
                }
            }
            const float score = simd_sum(dot) * scale;
            if (!isfinite(score)) {
                local_err |= 1u;
                continue;
            }

            float v_reg[8];
            for (uint c = 0; c < 8; ++c) {
                const uint d = c * 32u + uint(lane);
                if (d < s.z) {
                    const ushort vbits = v_ptr[d];
                    if ((vbits & 0x7f80u) == 0x7f80u) local_err |= 1u;
                    v_reg[c] = qsa_sparse_bf16(vbits);
                } else {
                    v_reg[c] = 0.0f;
                }
            }

            if (score > max_score) {
                const float factor = exp(max_score - score);
                max_score = score;
                sum_exp_score = fma(sum_exp_score, factor, 1.0f);
                for (uint c = 0; c < 8; ++c) o[c] = fma(o[c], factor, v_reg[c]);
            } else {
                const float factor = exp(score - max_score);
                sum_exp_score += factor;
                for (uint c = 0; c < 8; ++c) o[c] = fma(factor, v_reg[c], o[c]);
            }
        }
    }

    const uint sg_err = simd_or(local_err);
    if (lane == 0u) {
        max_scores[subgroup] = max_score;
        sum_exp_scores[subgroup] = sum_exp_score;
        sg_errors[subgroup] = sg_err;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_err = simd_or(uint(lane) < BN ? sg_errors[lane] : 0u);
    if (total_err != 0u) {
        if (tid == 0u) atomic_fetch_or_explicit(error, total_err, memory_order_relaxed);
        return;
    }

    const float m = (uint(lane) < BN) ? max_scores[lane] : -INFINITY;
    const float s_exp = (uint(lane) < BN) ? sum_exp_scores[lane] : 0.0f;
    const float global_max = simd_max(m);
    const float lane_factor = (uint(lane) < BN && m > -INFINITY) ? exp(m - global_max) : 0.0f;
    const float total_sum = simd_sum(s_exp * lane_factor);

    constexpr uint CH_PER_SG = 32u / BN;
    for (uint c = 0; c < 8; ++c) {
        const uint base_d = c * 32u;
        if (base_d >= s.z) break;
        partials[uint(subgroup) * 33u + uint(lane)] = (base_d + uint(lane) < s.z) ? o[c] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint k = 0; k < CH_PER_SG; ++k) {
            const uint ch = uint(subgroup) * CH_PER_SG + k;
            const uint d = base_d + ch;
            const float contrib = (uint(lane) < BN) ? (partials[uint(lane) * 33u + ch] * lane_factor) : 0.0f;
            const float acc = simd_sum(contrib);
            if (lane == 0u && d < s.z) {
                const float res = acc / total_sum;
                if (!isfinite(res)) {
                    atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                } else {
                    output_head[d] = res;
                }
            }
        }
        if (base_d + 32u < s.z) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template <uint BN>
inline void qsa_sparse_decode_entry(
    device const float* query,
    device const ushort* keys,
    device const ushort* values,
    device const uint* ids,
    device float* output,
    constant uint4& s,
    constant uint4& p,
    device atomic_uint* error,
    uint head,
    ushort subgroup,
    ushort lane,
    ushort tid,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    threadgroup uint* sg_errors,
    threadgroup float* partials) {
    if (!s.x || s.x > 24u || !s.y || (s.x % s.y) != 0u || !s.z || s.z > 256u || !s.w) {
        if (tid == 0u) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint tokens = min(p.x / s.w, p.y) * s.w + (p.x % s.w);
    if (!tokens || tokens > 2051u) {
        if (tid == 0u) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (head >= s.x || uint(subgroup) >= BN) return;
    const uint key_head = head / (s.x / s.y);
    const ulong kv_head_offset = ulong(key_head) * ulong(s.z);
    const ulong q_offset = ulong(head) * ulong(s.z);
    qsa_sparse_core<BN>(
        query + q_offset,
        keys + kv_head_offset,
        values + kv_head_offset,
        ids,
        output + q_offset,
        s,
        p.x,
        p.y,
        error,
        subgroup,
        lane,
        tid,
        max_scores,
        sum_exp_scores,
        sg_errors,
        partials);
}

template <uint BN>
inline void qsa_sparse_rows_entry(
    device const float* queries,
    device const ushort* keys,
    device const ushort* values,
    device const uint* ids,
    device float* outputs,
    constant uint4& s,
    constant uint4& p,
    device atomic_uint* error,
    uint2 group,
    ushort subgroup,
    ushort lane,
    ushort tid,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    threadgroup uint* sg_errors,
    threadgroup float* partials) {
    if (!p.z || p.z > 8u || !s.x || s.x > 24u || !s.y || (s.x % s.y) != 0u || !s.z || s.z > 256u || !s.w) {
        if (tid == 0u) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint head = group.x;
    const uint row = group.y;
    if (head >= s.x || row >= p.z || uint(subgroup) >= BN) return;
    const uint causal_prefix = p.x + row + 1u;
    const uint tokens = min(causal_prefix / s.w, p.y) * s.w + (causal_prefix % s.w);
    if (!tokens || tokens > 2051u) {
        if (tid == 0u) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint key_head = head / (s.x / s.y);
    const ulong kv_head_offset = ulong(key_head) * ulong(s.z);
    const ulong q_offset = (ulong(row) * ulong(s.x) + ulong(head)) * ulong(s.z);
    qsa_sparse_core<BN>(
        queries + q_offset,
        keys + kv_head_offset,
        values + kv_head_offset,
        ids + ulong(row) * ulong(p.y),
        outputs + q_offset,
        s,
        causal_prefix,
        p.y,
        error,
        subgroup,
        lane,
        tid,
        max_scores,
        sum_exp_scores,
        sg_errors,
        partials);
}

#ifndef TOKSPUR_QSA_SPARSE_DECODE_BN
#define TOKSPUR_QSA_SPARSE_DECODE_BN 32u
#endif
#ifndef TOKSPUR_QSA_SPARSE_VERIFY_BN
#define TOKSPUR_QSA_SPARSE_VERIFY_BN 32u
#endif

kernel void qsa_sparse_attention(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* output [[buffer(4)]],
    constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float max_scores[TOKSPUR_QSA_SPARSE_DECODE_BN];
    threadgroup float sum_exp_scores[TOKSPUR_QSA_SPARSE_DECODE_BN];
    threadgroup uint sg_errors[TOKSPUR_QSA_SPARSE_DECODE_BN];
    threadgroup float partials[TOKSPUR_QSA_SPARSE_DECODE_BN * 33u];
    qsa_sparse_decode_entry<TOKSPUR_QSA_SPARSE_DECODE_BN>(query, keys, values, ids, output, s, p, error, head, subgroup, lane, tid, max_scores, sum_exp_scores, sg_errors, partials);
}

kernel void qsa_sparse_attention_bn8(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* output [[buffer(4)]],
    constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float max_scores[8u];
    threadgroup float sum_exp_scores[8u];
    threadgroup uint sg_errors[8u];
    threadgroup float partials[8u * 33u];
    qsa_sparse_decode_entry<8u>(query, keys, values, ids, output, s, p, error, head, subgroup, lane, tid, max_scores, sum_exp_scores, sg_errors, partials);
}

kernel void qsa_sparse_attention_bn16(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* output [[buffer(4)]],
    constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float max_scores[16u];
    threadgroup float sum_exp_scores[16u];
    threadgroup uint sg_errors[16u];
    threadgroup float partials[16u * 33u];
    qsa_sparse_decode_entry<16u>(query, keys, values, ids, output, s, p, error, head, subgroup, lane, tid, max_scores, sum_exp_scores, sg_errors, partials);
}

kernel void qsa_sparse_attention_bn32(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* output [[buffer(4)]],
    constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float max_scores[32u];
    threadgroup float sum_exp_scores[32u];
    threadgroup uint sg_errors[32u];
    threadgroup float partials[32u * 33u];
    qsa_sparse_decode_entry<32u>(query, keys, values, ids, output, s, p, error, head, subgroup, lane, tid, max_scores, sum_exp_scores, sg_errors, partials);
}

kernel void qsa_sparse_attention_rows(device const float* queries [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* outputs [[buffer(4)]],
    constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float max_scores[TOKSPUR_QSA_SPARSE_VERIFY_BN];
    threadgroup float sum_exp_scores[TOKSPUR_QSA_SPARSE_VERIFY_BN];
    threadgroup uint sg_errors[TOKSPUR_QSA_SPARSE_VERIFY_BN];
    threadgroup float partials[TOKSPUR_QSA_SPARSE_VERIFY_BN * 33u];
    qsa_sparse_rows_entry<TOKSPUR_QSA_SPARSE_VERIFY_BN>(queries, keys, values, ids, outputs, s, p, error, group, subgroup, lane, tid, max_scores, sum_exp_scores, sg_errors, partials);
}

kernel void qsa_sparse_attention_rows_bn8(device const float* queries [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* outputs [[buffer(4)]],
    constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float max_scores[8u];
    threadgroup float sum_exp_scores[8u];
    threadgroup uint sg_errors[8u];
    threadgroup float partials[8u * 33u];
    qsa_sparse_rows_entry<8u>(queries, keys, values, ids, outputs, s, p, error, group, subgroup, lane, tid, max_scores, sum_exp_scores, sg_errors, partials);
}

kernel void qsa_sparse_attention_rows_bn16(device const float* queries [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* outputs [[buffer(4)]],
    constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float max_scores[16u];
    threadgroup float sum_exp_scores[16u];
    threadgroup uint sg_errors[16u];
    threadgroup float partials[16u * 33u];
    qsa_sparse_rows_entry<16u>(queries, keys, values, ids, outputs, s, p, error, group, subgroup, lane, tid, max_scores, sum_exp_scores, sg_errors, partials);
}

kernel void qsa_sparse_attention_rows_bn32(device const float* queries [[buffer(0)]], device const ushort* keys [[buffer(1)]],
    device const ushort* values [[buffer(2)]], device const uint* ids [[buffer(3)]], device float* outputs [[buffer(4)]],
    constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float max_scores[32u];
    threadgroup float sum_exp_scores[32u];
    threadgroup uint sg_errors[32u];
    threadgroup float partials[32u * 33u];
    qsa_sparse_rows_entry<32u>(queries, keys, values, ids, outputs, s, p, error, group, subgroup, lane, tid, max_scores, sum_exp_scores, sg_errors, partials);
}
