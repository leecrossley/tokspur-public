// Bounded Metal 4.0 non-causal block-8 DFlash2 speculative draft kernels.
// Dynamic convolution, ring KV attention and sharded top-16 candidate selector adapted from Splash 1.4.0 runtime/metal/kernels under the Apache License 2.0 (see NOTICE).
// Tokspur owns graph state, FP32 row views, finite error reporting, RNG laws and speculative verification.
#pragma clang fp reassociate(off)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint dflash2_q8_encoding [[function_constant(0)]];
constant uint dflash2_q8_layout [[function_constant(1)]];

namespace dflash2_detail {

struct DFlash2StreamMeanArgs {
    uint rows, width, slot_index, total_slots;
    ulong input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(DFlash2StreamMeanArgs) == 32, "DFlash2 stream mean ABI");

struct DFlash2ContextKvArgs {
    uint rows, start_pos, window;
    float eps;
};
static_assert(sizeof(DFlash2ContextKvArgs) == 16, "DFlash2 context KV ABI");

struct DFlash2ConvArgs {
    uint hidden, finish, rows, group_size;
};
static_assert(sizeof(DFlash2ConvArgs) == 16, "DFlash2 conv ABI");

struct DFlash2QkvArgs {
    uint cache_len, rows;
    float eps, reserved;
};
static_assert(sizeof(DFlash2QkvArgs) == 16, "DFlash2 QKV ABI");

struct DFlash2AttnArgs {
    uint cache_len, window, splits, reserved;
};
static_assert(sizeof(DFlash2AttnArgs) == 16, "DFlash2 attention ABI");

struct DFlash2SelectArgs {
    uint positions, vocabulary, anchor, reserved;
};
static_assert(sizeof(DFlash2SelectArgs) == 16, "DFlash2 selector ABI");

struct DFlash2DenseBf16Args {
    uint columns, rows, row_bytes, experts, selected, tokens, broadcast, swiglu;
    ulong matrix_bytes, input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(DFlash2DenseBf16Args) == 56, "DFlash2 dense BF16 ABI");

struct DFlash2ConvRmsArgs {
    uint hidden, rows, group_size;
    float eps;
};
static_assert(sizeof(DFlash2ConvRmsArgs) == 16, "DFlash2 conv+RMS ABI");

static inline float sigmoid_f32(float v) {
    return v >= 0.0f ? 1.0f / (1.0f + exp(-v)) : exp(v) / (1.0f + exp(v));
}

static inline uint nonfinite_bit(float v) {
    return (as_type<uint>(v) & 0x7f800000u) == 0x7f800000u ? 1u : 0u;
}

static inline bfloat clean_bf16(float v, thread uint& bad) {
    const uint bits = as_type<uint>(v);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        bad |= 1u;
        return as_type<bfloat>(ushort(0));
    }
    const ushort rounded = ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
    if ((rounded & 0x7f80u) == 0x7f80u) {
        bad |= 1u;
        return as_type<bfloat>(ushort(0));
    }
    return as_type<bfloat>(rounded);
}

static inline bool top_beats(float value, uint id, float other_val, uint other_id) {
    return value > other_val || (value == other_val && id < other_id);
}

template <uint Count>
static inline void top_insert(thread float (&values)[Count], thread uint (&ids)[Count], float value, uint id) {
    #pragma unroll
    for (uint i = 0; i < Count; ++i) {
        if (top_beats(value, id, values[i], ids[i])) {
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
static inline void top_pop(thread float (&values)[Count], thread uint (&ids)[Count]) {
    #pragma unroll
    for (uint i = 0; i + 1u < Count; ++i) {
        values[i] = values[i + 1u];
        ids[i] = ids[i + 1u];
    }
    values[Count - 1u] = -INFINITY;
    ids[Count - 1u] = 0xffffffffu;
}

static inline void simd_best_head(float value, uint token, thread float& best, thread uint& best_token) {
    best = simd_max(value);
    best_token = simd_min(value == best ? token : 0xffffffffu);
}

static inline void top16_merge_shards(device const uint* partial_ids, device const float* partial_values,
    uint row, uint lane, thread float& value, thread uint& token) {
    constexpr uint K = 16u, Shards = 8u, Entries = 4u;
    const uint origin = row * Shards * K + lane * Entries;
    float values[Entries];
    uint ids[Entries];
    #pragma unroll
    for (uint i = 0; i < Entries; ++i) {
        values[i] = partial_values[origin + i];
        ids[i] = partial_ids[origin + i];
    }
    value = -INFINITY;
    token = 0xffffffffu;
    for (uint rank = 0; rank < K; ++rank) {
        float best;
        uint best_id;
        simd_best_head(values[0], ids[0], best, best_id);
        if (lane == rank) {
            value = best;
            token = best_id;
        }
        if (values[0] == best && ids[0] == best_id) top_pop<Entries>(values, ids);
    }
}

} // namespace dflash2_detail

kernel void dflash2_stream_mean(device const float* streams [[buffer(0)]], device float* captured [[buffer(1)]],
    constant dflash2_detail::DFlash2StreamMeanArgs& a [[buffer(2)]], device atomic_uint* error [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
    if (!a.rows || !a.width || (a.width % 4u) != 0u || a.slot_index >= a.total_slots ||
        (a.input_stride_bytes % 4ul) != 0ul || (a.output_stride_bytes % 4ul) != 0ul) {
        if (!idx) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint quads = a.width / 4u;
    const uint total = a.rows * quads;
    if (idx >= total) return;
    const uint row = idx / quads, col4 = (idx % quads) * 4u;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const ulong out_stride = a.output_stride_bytes / sizeof(float);
    device const float* src = streams + ulong(row) * in_stride;
    const float4 s0 = float4(*reinterpret_cast<device const packed_float4*>(src + 0u * a.width + col4));
    const float4 s1 = float4(*reinterpret_cast<device const packed_float4*>(src + 1u * a.width + col4));
    const float4 s2 = float4(*reinterpret_cast<device const packed_float4*>(src + 2u * a.width + col4));
    const float4 s3 = float4(*reinterpret_cast<device const packed_float4*>(src + 3u * a.width + col4));
    const float4 mean = 0.25f * ((s0 + s1) + (s2 + s3));
    uint bad = dflash2_detail::nonfinite_bit(mean.x) | dflash2_detail::nonfinite_bit(mean.y) |
               dflash2_detail::nonfinite_bit(mean.z) | dflash2_detail::nonfinite_bit(mean.w);
    *reinterpret_cast<device packed_float4*>(captured + ulong(row) * out_stride + ulong(a.slot_index) * a.width + col4) = mean;
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

kernel void dflash2_context_kv_write(device const float* context_kv [[buffer(0)]],
    device const float* qk_norm_freq [[buffer(1)]], device bfloat* ring_k [[buffer(2)]],
    device bfloat* ring_v [[buffer(3)]], constant dflash2_detail::DFlash2ContextKvArgs& a [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint KVHeads = 8u, HeadDim = 128u, KWidth = KVHeads * HeadDim, RowWidth = 2u * KWidth;
    if (!a.rows || a.window != 2048u || !(a.eps > 0.0f)) {
        if (!group && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint row = group / KVHeads;
    if (row >= a.rows) return;
    const uint head = group % KVHeads;
    const uint position = a.start_pos + row;
    const uint slot = position % a.window;
    device const float* k_norm = qk_norm_freq + HeadDim;
    device const float* inv_freq = qk_norm_freq + 2u * HeadDim;
    device const float* src = context_kv + ulong(row) * RowWidth + head * HeadDim;

    uint bad = 0u;
    const float el = tid < HeadDim ? src[tid] : 0.0f;
    const float v_el = tid < HeadDim ? src[KWidth + tid] : 0.0f;
    bad |= dflash2_detail::nonfinite_bit(el) | dflash2_detail::nonfinite_bit(v_el);

    threadgroup float reductions[4];
    threadgroup bfloat normalized[HeadDim];
    const float sq = simd_sum(el * el);
    if (!lane && sg < 4u) reductions[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean_sq = (reductions[0] + reductions[1] + reductions[2] + reductions[3]) * (1.0f / float(HeadDim));
    const float inv_rms = rsqrt(mean_sq + a.eps);
    if (tid < HeadDim) {
        normalized[tid] = dflash2_detail::clean_bf16(el * inv_rms * k_norm[tid], bad);
        ring_v[(ulong(head) * HeadDim + tid) * a.window + slot] = dflash2_detail::clean_bf16(v_el, bad);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HeadDim / 2u) {
        const float first = float(normalized[tid]);
        const float second = float(normalized[tid + HeadDim / 2u]);
        const float angle = float(position) * inv_freq[tid];
        const float c = cos(angle), s = sin(angle);
        device bfloat* key = ring_k + (ulong(head) * a.window + slot) * HeadDim;
        key[tid] = dflash2_detail::clean_bf16(first * c - second * s, bad);
        key[tid + HeadDim / 2u] = dflash2_detail::clean_bf16(second * c + first * s, bad);
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

kernel void dflash2_conv_f32(device const float* input [[buffer(0)]], device const float* dynamic [[buffer(1)]],
    device const float* base [[buffer(2)]], device const float* residual [[buffer(3)]],
    device float* output [[buffer(4)]], constant dflash2_detail::DFlash2ConvArgs& a [[buffer(5)]],
    device atomic_uint* error [[buffer(7)]], uint idx [[thread_position_in_grid]]) {
    if (!a.hidden || !a.rows || a.group_size != 16u || (a.hidden % a.group_size) != 0u) {
        if (!idx) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint total = a.rows * a.hidden;
    if (idx >= total) return;
    const uint row = idx / a.hidden, ch = idx % a.hidden;
    const uint conv_groups = a.hidden / a.group_size;
    const uint dyn_width = 4u * conv_groups;
    const uint cg = ch / a.group_size;
    const uint kind = a.finish ? 1u : 0u;
    uint bad = 0u;
    const float x0 = input[idx];
    const float b0 = base[(kind * 2u) * a.hidden + ch];
    const float d0 = dynamic[row * dyn_width + (kind * 2u) * conv_groups + cg];
    bad |= dflash2_detail::nonfinite_bit(x0) | dflash2_detail::nonfinite_bit(b0) | dflash2_detail::nonfinite_bit(d0);
    float val = x0 * (b0 + d0);
    if (row > 0u) {
        const float x1 = input[(row - 1u) * a.hidden + ch];
        const float b1 = base[(kind * 2u + 1u) * a.hidden + ch];
        const float d1 = dynamic[row * dyn_width + (kind * 2u + 1u) * conv_groups + cg];
        bad |= dflash2_detail::nonfinite_bit(x1) | dflash2_detail::nonfinite_bit(b1) | dflash2_detail::nonfinite_bit(d1);
        val += x1 * (b1 + d1);
    }
    if (a.finish) {
        const float res = residual[idx];
        bad |= dflash2_detail::nonfinite_bit(res);
        val += res;
    }
    bad |= dflash2_detail::nonfinite_bit(val);
    output[idx] = val;
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

kernel void dflash2_qkv_prepare(device const float* proposal_qkv [[buffer(0)]],
    device const float* qk_norm_freq [[buffer(1)]], device bfloat* queries [[buffer(2)]],
    device bfloat* query_keys [[buffer(3)]], device bfloat* query_values [[buffer(4)]],
    constant dflash2_detail::DFlash2QkvArgs& a [[buffer(5)]], device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint Rows = 8u, QHeads = 32u, KVHeads = 8u, HeadDim = 128u;
    constexpr uint QWidth = QHeads * HeadDim, KWidth = KVHeads * HeadDim, PackedWidth = QWidth + 2u * KWidth;
    constexpr uint QueryTasks = Rows * QHeads, TotalTasks = QueryTasks + Rows * KVHeads;
    if (a.rows != Rows || !(a.eps > 0.0f)) {
        if (!group && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    uint bad = 0u;
    const uint elem = group * 128u + uint(tid);
    if (elem < Rows * KWidth) {
        const uint row = elem / KWidth;
        const uint rem = elem % KWidth;
        const uint head = rem / HeadDim;
        const uint dim = rem % HeadDim;
        const float v = proposal_qkv[ulong(row) * PackedWidth + QWidth + KWidth + rem];
        query_values[(head * HeadDim + dim) * Rows + row] = dflash2_detail::clean_bf16(v, bad);
    }

    threadgroup float reductions[4];
    threadgroup bfloat head_buf[HeadDim];
    if (group < TotalTasks) {
        const bool is_q = group < QueryTasks;
        const uint local_task = is_q ? group : (group - QueryTasks);
        const uint heads = is_q ? QHeads : KVHeads;
        const uint row = local_task / heads;
        const uint head = local_task % heads;
        device const float* src = is_q ? (proposal_qkv + ulong(row) * PackedWidth + head * HeadDim)
                                       : (proposal_qkv + ulong(row) * PackedWidth + QWidth + head * HeadDim);
        device const float* norm_w = is_q ? qk_norm_freq : (qk_norm_freq + HeadDim);
        device const float* inv_freq = qk_norm_freq + 2u * HeadDim;
        device bfloat* dst = (is_q ? queries : query_keys) + (head * Rows + row) * HeadDim;

        const float val = tid < HeadDim ? src[tid] : 0.0f;
        bad |= dflash2_detail::nonfinite_bit(val);
        const float sq = simd_sum(val * val);
        if (!lane && sg < 4u) reductions[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float inv_rms = rsqrt((reductions[0] + reductions[1] + reductions[2] + reductions[3]) * (1.0f / float(HeadDim)) + a.eps);
        if (tid < HeadDim) head_buf[tid] = dflash2_detail::clean_bf16(val * inv_rms * norm_w[tid], bad);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < HeadDim / 2u) {
            const float first = float(head_buf[tid]);
            const float second = float(head_buf[tid + HeadDim / 2u]);
            const float angle = float(a.cache_len + row) * inv_freq[tid];
            const float c = cos(angle), s = sin(angle);
            dst[tid] = dflash2_detail::clean_bf16(first * c - second * s, bad);
            dst[tid + HeadDim / 2u] = dflash2_detail::clean_bf16(second * c + first * s, bad);
        }
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

kernel void dflash2_attention_split(device bfloat* queries_all [[buffer(0)]],
    device bfloat* keys_all [[buffer(1)]], device bfloat* values_all [[buffer(2)]],
    device bfloat* query_keys_all [[buffer(3)]], device bfloat* query_values_all [[buffer(4)]],
    device float* partials_all [[buffer(5)]], constant dflash2_detail::DFlash2AttnArgs& a [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort M = 32, N = 128, D = 128, TileK = 64;
    constexpr uint Rows = 8u, Window = 2048u, Splits = 16u;
    constexpr ulong PartialFloats = ulong(M) * D + 2ul * M;
    if (a.window != Window || a.splits != Splits) {
        if (!group.x && !group.z && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint kv_head = group.x, split = group.z;
    device bfloat* queries = queries_all + ulong(kv_head) * M * D;
    device bfloat* keys = keys_all + ulong(kv_head) * Window * D;
    device bfloat* values = values_all + ulong(kv_head) * D * Window;
    device bfloat* query_keys = query_keys_all + ulong(kv_head) * Rows * D;
    device bfloat* query_values = query_values_all + ulong(kv_head) * D * Rows;
    device float* partial = partials_all + (ulong(kv_head) * Splits + split) * PartialFloats;

    const uint cache_length = a.cache_len;
    const uint common_start = cache_length >= (Window - 1u) ? (cache_length - (Window - 1u)) : 0u;
    const uint old_count = cache_length - common_start;
    const uint physical_start = common_start % Window;
    const uint physical_end = physical_start + old_count;
    const uint wrapped = physical_end > Window ? (physical_end - Window + N - 1u) / N : 0u;
    const uint tail_end = (min(physical_end, Window) + N - 1u) / N;
    const uint resume = max(physical_start / uint(N), wrapped);
    const uint live_tiles = wrapped + tail_end - resume;

    if (split >= live_tiles && split + 1u != Splits) {
        if (tid < M) {
            partial[M * D + tid] = -INFINITY;
            partial[M * D + M + tid] = 0.0f;
        }
        return;
    }

    threadgroup float workspace[uint(M) * uint(N) + 3u * uint(M)];
    threadgroup float* score_storage = workspace;
    threadgroup float* row_max = workspace + uint(M) * uint(N);
    threadgroup float* row_sum = row_max + M;
    threadgroup float* previous_scale = row_sum + M;

    auto qt = tensor(queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    constexpr auto qk_desc = matmul2d_descriptor(M, N, TileK, false, true, false);
    matmul2d<qk_desc, execution_simdgroups<8>> qk;
    auto q0 = qt.slice<TileK, M>(0, 0);
    auto pt = tensor(score_storage, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    auto p0 = pt.slice<TileK, M>(0, 0);
    auto first_v = tensor(values, dextents<int, 2>{N, D}, array<int, 2>{1, int(Window)});
    auto first_v0 = first_v.slice<TileK, D>(0, 0);
    constexpr auto pv_desc = matmul2d_descriptor(M, D, TileK, false, true, false);
    matmul2d<pv_desc, execution_simdgroups<8>> pv;
    auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(first_v0), float>();
    #pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0.0f;
    if (tid < M) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile = split; tile < live_tiles; tile += Splits) {
        const uint slot = (tile < wrapped ? tile : (resume + tile - wrapped)) * uint(N);
        auto kt = tensor(keys + ulong(slot) * D, dextents<int, 2>{D, N}, array<int, 2>{1, D});
        auto k0 = kt.slice<TileK, N>(0, 0);
        auto scores = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
        #pragma unroll
        for (ushort i = 0; i < scores.get_capacity(); ++i) scores[i] = 0.0f;
        #pragma unroll
        for (ushort chunk = 0; chunk < D / TileK; ++chunk) {
            auto qs = qt.slice<TileK, M>(chunk * TileK, 0);
            auto ks = kt.slice<TileK, N>(chunk * TileK, 0);
            auto ps_acc = qk.template get_destination_cooperative_tensor<decltype(qs), decltype(ks), float>();
            qk.run(qs, ks, ps_acc);
            #pragma unroll
            for (ushort i = 0; i < scores.get_capacity(); ++i) scores[i] += ps_acc[i];
        }
        scores.store(pt.slice<N, M>(0, 0));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort batch = 0; batch < M; batch += 8) {
            const ushort matrix_row = batch + sg;
            const uint proposal_row = uint(matrix_row) % Rows;
            const uint query_pos = cache_length + proposal_row;
            const uint row_start = query_pos >= (Window - 1u) ? (query_pos - (Window - 1u)) : 0u;
            const uint hidden_prefix = row_start - common_start;
            float local_scores[4];
            float tile_max = -INFINITY;
            #pragma unroll
            for (ushort i = 0; i < 4; ++i) {
                const uint col = uint(lane) + uint(i) * 32u;
                const uint key = slot + col;
                const uint logical_key = key >= physical_start ? (key - physical_start) : (key + Window - physical_start);
                const bool valid = logical_key < old_count && logical_key >= hidden_prefix;
                const float sc = score_storage[uint(matrix_row) * N + col] * 0.08838834765f;
                local_scores[i] = valid ? sc : -INFINITY;
                tile_max = max(tile_max, local_scores[i]);
            }
            tile_max = simd_max(tile_max);
            const float prev_max = row_max[matrix_row];
            const float next_max = max(prev_max, tile_max);
            const bool empty = (next_max == -INFINITY);
            const float scale = empty ? 1.0f : (prev_max == -INFINITY ? 0.0f : exp(prev_max - next_max));
            float tile_sum = 0.0f;
            #pragma unroll
            for (ushort i = 0; i < 4; ++i) {
                local_scores[i] = (empty || local_scores[i] == -INFINITY) ? 0.0f : exp(local_scores[i] - next_max);
                tile_sum += local_scores[i];
            }
            tile_sum = simd_sum(tile_sum);
            if (!lane) {
                previous_scale[matrix_row] = scale;
                row_sum[matrix_row] = row_sum[matrix_row] * scale + tile_sum;
                row_max[matrix_row] = next_max;
            }
            #pragma unroll
            for (ushort i = 0; i < 4; ++i) {
                score_storage[uint(matrix_row) * N + uint(lane) + uint(i) * 32u] = local_scores[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto vt = tensor(values + slot, dextents<int, 2>{N, D}, array<int, 2>{1, int(Window)});
        auto v0 = vt.slice<TileK, D>(0, 0);
        auto partial_out = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
        #pragma unroll
        for (ushort i = 0; i < partial_out.get_capacity(); ++i) partial_out[i] = 0.0f;
        #pragma unroll
        for (ushort chunk = 0; chunk < N / TileK; ++chunk) {
            auto ps = pt.slice<TileK, M>(chunk * TileK, 0);
            auto vs = vt.slice<TileK, D>(chunk * TileK, 0);
            auto pv_acc = pv.template get_destination_cooperative_tensor<decltype(ps), decltype(vs), float>();
            pv.run(ps, vs, pv_acc);
            #pragma unroll
            for (ushort i = 0; i < partial_out.get_capacity(); ++i) partial_out[i] += pv_acc[i];
        }
        #pragma unroll
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            const auto idx = running.get_multidimensional_index(i);
            running[i] = running[i] * previous_scale[idx[1]] + partial_out[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (split + 1u == Splits) {
        constexpr ushort CurrentN = 8, CurrentValueTile = 16;
        threadgroup float* current_raw_scores = score_storage + uint(M) * CurrentValueTile;
        threadgroup bfloat* current_values = reinterpret_cast<threadgroup bfloat*>(score_storage + uint(M) * CurrentValueTile + uint(M) * CurrentN);
        auto current_scores = tensor(current_raw_scores, dextents<int, 2>{CurrentN, M}, array<int, 2>{1, CurrentN});
        constexpr auto cur_qk_desc = matmul2d_descriptor(M, CurrentN, TileK, false, true, false);
        matmul2d<cur_qk_desc, execution_simdgroups<8>> cur_qk;
        auto cur_kt = tensor(query_keys, dextents<int, 2>{D, CurrentN}, array<int, 2>{1, D});
        auto cur_k0 = cur_kt.slice<TileK, CurrentN>(0, 0);
        auto cur = cur_qk.template get_destination_cooperative_tensor<decltype(q0), decltype(cur_k0), float>();
        #pragma unroll
        for (ushort i = 0; i < cur.get_capacity(); ++i) cur[i] = 0.0f;
        #pragma unroll
        for (ushort chunk = 0; chunk < D / TileK; ++chunk) {
            auto qs = qt.slice<TileK, M>(chunk * TileK, 0);
            auto ks = cur_kt.slice<TileK, CurrentN>(chunk * TileK, 0);
            auto ps_acc = cur_qk.template get_destination_cooperative_tensor<decltype(qs), decltype(ks), float>();
            cur_qk.run(qs, ks, ps_acc);
            #pragma unroll
            for (ushort i = 0; i < cur.get_capacity(); ++i) cur[i] += ps_acc[i];
        }
        cur.store(current_scores.slice<CurrentN, M>(0, 0));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort batch = 0; batch < M; batch += 8) {
            const ushort matrix_row = batch + sg;
            const float sc = lane < CurrentN ? (current_raw_scores[uint(matrix_row) * CurrentN + lane] * 0.08838834765f) : -INFINITY;
            const float tile_max = simd_max(sc);
            const float prev_max = row_max[matrix_row];
            const float next_max = max(prev_max, tile_max);
            const float scale = (prev_max == -INFINITY || next_max == -INFINITY) ? 0.0f : exp(prev_max - next_max);
            const float prob = (lane < CurrentN && next_max != -INFINITY) ? exp(sc - next_max) : 0.0f;
            const float tile_sum = simd_sum(prob);
            if (!lane) {
                previous_scale[matrix_row] = scale;
                row_sum[matrix_row] = row_sum[matrix_row] * scale + tile_sum;
                row_max[matrix_row] = next_max;
            }
            if (lane < CurrentValueTile) {
                score_storage[uint(matrix_row) * CurrentValueTile + lane] = lane < CurrentN ? prob : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < uint(D) * CurrentValueTile; idx += 256u) {
            const uint dim = idx / CurrentValueTile;
            const uint tok = idx % CurrentValueTile;
            current_values[idx] = tok < CurrentN ? query_values[dim * CurrentN + tok] : as_type<bfloat>(ushort(0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto cur_probs = tensor(score_storage, dextents<int, 2>{CurrentValueTile, M}, array<int, 2>{1, CurrentValueTile});
        auto cur_p0 = cur_probs.slice<CurrentValueTile, M>(0, 0);
        auto cur_vt = tensor(current_values, dextents<int, 2>{CurrentValueTile, D}, array<int, 2>{1, int(CurrentValueTile)});
        auto cur_v0 = cur_vt.slice<CurrentValueTile, D>(0, 0);
        constexpr auto cur_pv_desc = matmul2d_descriptor(M, D, CurrentValueTile, false, true, false);
        matmul2d<cur_pv_desc, execution_simdgroups<8>> cur_pv;
        auto cur_out = cur_pv.template get_destination_cooperative_tensor<decltype(cur_p0), decltype(cur_v0), float>();
        cur_pv.run(cur_p0, cur_v0, cur_out);
        #pragma unroll
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            const auto idx = running.get_multidimensional_index(i);
            running[i] = running[i] * previous_scale[idx[1]] + cur_out[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto acc_tensor = tensor(partial, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    running.store(acc_tensor.slice<D, M>(0, 0));
    if (tid < M) {
        partial[M * D + tid] = row_max[tid];
        partial[M * D + M + tid] = row_sum[tid];
    }
}

kernel void dflash2_attention_reduce(device const float* partials_all [[buffer(0)]],
    device float* output [[buffer(1)]], constant dflash2_detail::DFlash2AttnArgs& a [[buffer(2)]],
    device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint M = 32u, D = 128u, Splits = 16u, Rows = 8u, QHeads = 32u, Chunk = 16u;
    constexpr ulong Stride = ulong(M) * D + 2ul * M;
    if (a.splits != Splits) {
        if (!group && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint kv_head = group;
    device const float* partials = partials_all + ulong(kv_head) * Splits * Stride;
    const uint m_row = uint(tid) / 8u;
    const uint dim = (uint(tid) % 8u) * Chunk;

    float maximum = -INFINITY;
    for (uint s = 0u; s < Splits; ++s) maximum = max(maximum, partials[ulong(s) * Stride + M * D + m_row]);

    float total = 0.0f;
    float acc[Chunk];
    #pragma unroll
    for (uint i = 0u; i < Chunk; ++i) acc[i] = 0.0f;

    if (maximum != -INFINITY) {
        for (uint s = 0u; s < Splits; ++s) {
            device const float* p = partials + ulong(s) * Stride;
            const float split_max = p[M * D + m_row];
            if (split_max == -INFINITY) continue;
            const float scale = exp(split_max - maximum);
            total += scale * p[M * D + M + m_row];
            #pragma unroll
            for (uint i = 0u; i < Chunk; ++i) acc[i] += scale * p[m_row * D + dim + i];
        }
    }

    const uint local_q = m_row / Rows;
    const uint prop_row = m_row % Rows;
    const uint global_q = kv_head * 4u + local_q;
    device float* dst = output + (ulong(prop_row) * QHeads + global_q) * D + dim;
    uint bad = 0u;
    const float inv_total = total > 0.0f ? (1.0f / total) : 0.0f;
    #pragma unroll
    for (uint i = 0u; i < Chunk; ++i) {
        const float val = acc[i] * inv_total;
        bad |= dflash2_detail::nonfinite_bit(val);
        dst[i] = val;
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

kernel void dflash2_top16_sharded(device const float* logits [[buffer(0)]],
    device uint* partial_ids [[buffer(1)]], device float* partial_values [[buffer(2)]],
    constant dflash2_detail::DFlash2SelectArgs& a [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint Shards = 8u, K = 16u, VectorTokens = 4u;
    if (!a.positions || a.positions > 7u || !a.vocabulary) {
        if (!group && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint position = group / Shards;
    const uint shard = group % Shards;
    if (position >= a.positions) return;

    const ulong row_start = ulong(position) * a.vocabulary;
    const uint shard_tokens = (a.vocabulary + Shards - 1u) / Shards;
    const uint begin = min(shard * shard_tokens, a.vocabulary);
    const uint end = min(begin + shard_tokens, a.vocabulary);
    uint head = uint((VectorTokens - (row_start + begin) % VectorTokens) % VectorTokens);
    head = min(head, end - begin);
    const uint vectors = (end - begin - head) / VectorTokens;
    const uint vector_begin = begin + head;
    const uint tail_begin = vector_begin + vectors * VectorTokens;
    device const float* row = logits + row_start;
    device const packed_float4* vector_row = reinterpret_cast<device const packed_float4*>(row + vector_begin);

    float values[K];
    uint ids[K];
    #pragma unroll
    for (uint i = 0u; i < K; ++i) {
        values[i] = -INFINITY;
        ids[i] = 0xffffffffu;
    }
    uint bad = 0u;
    if (tid < head) {
        const uint tok = begin + tid;
        const float val = row[tok];
        bad |= dflash2_detail::nonfinite_bit(val);
        if (dflash2_detail::top_beats(val, tok, values[K - 1u], ids[K - 1u]))
            dflash2_detail::top_insert<K>(values, ids, val, tok);
    }
    if (tid < (end - tail_begin)) {
        const uint tok = tail_begin + tid;
        const float val = row[tok];
        bad |= dflash2_detail::nonfinite_bit(val);
        if (dflash2_detail::top_beats(val, tok, values[K - 1u], ids[K - 1u]))
            dflash2_detail::top_insert<K>(values, ids, val, tok);
    }

    for (uint idx = uint(tid); idx < vectors; idx += 256u) {
        const float4 loaded = float4(vector_row[idx]);
        bad |= dflash2_detail::nonfinite_bit(loaded.x) | dflash2_detail::nonfinite_bit(loaded.y) |
               dflash2_detail::nonfinite_bit(loaded.z) | dflash2_detail::nonfinite_bit(loaded.w);
        const uint tok = vector_begin + idx * VectorTokens;
        #pragma unroll
        for (uint j = 0u; j < VectorTokens; ++j) {
            const float val = loaded[j];
            if (dflash2_detail::top_beats(val, tok + j, values[K - 1u], ids[K - 1u]))
                dflash2_detail::top_insert<K>(values, ids, val, tok + j);
        }
    }

    threadgroup float sg_vals[8][K];
    threadgroup uint sg_ids[8][K];
    for (uint r = 0u; r < K; ++r) {
        const float head_val = values[0];
        const uint head_id = ids[0];
        float simd_val;
        uint simd_id;
        dflash2_detail::simd_best_head(head_val, head_id, simd_val, simd_id);
        const ushort win_lane = simd_min((head_id != 0xffffffffu && head_id == simd_id) ? lane : ushort(32));
        if (win_lane < 32u) simd_val = simd_shuffle(head_val, win_lane);
        if (!lane) {
            sg_vals[sg][r] = simd_val;
            sg_ids[sg][r] = simd_id;
        }
        if (simd_id == 0xffffffffu) {
            if (!lane) {
                for (uint rem = r + 1u; rem < K; ++rem) {
                    sg_vals[sg][rem] = -INFINITY;
                    sg_ids[sg][rem] = 0xffffffffu;
                }
            }
            break;
        }
        if (head_id == simd_id) dflash2_detail::top_pop<K>(values, ids);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        uint cur = 0u;
        float out_v = -INFINITY;
        uint out_i = 0xffffffffu;
        for (uint rank = 0u; rank < K; ++rank) {
            const bool active = (lane < 8u) && (cur < K);
            const float v = active ? sg_vals[lane][cur] : -INFINITY;
            const uint id = active ? sg_ids[lane][cur] : 0xffffffffu;
            float best;
            uint best_id;
            dflash2_detail::simd_best_head(v, id, best, best_id);
            const ushort win_lane = simd_min((id != 0xffffffffu && id == best_id) ? lane : ushort(32));
            if (win_lane < 8u) best = simd_shuffle(v, win_lane);
            if (uint(lane) == rank) {
                out_i = best_id;
                out_v = best;
            }
            if (active && best_id != 0xffffffffu && id == best_id) ++cur;
        }
        if (uint(lane) < K) {
            partial_ids[group * K + uint(lane)] = out_i;
            partial_values[group * K + uint(lane)] = out_v;
        }
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

// Writes packed selector_out layout: uint candidates[7*16], float unary[7*16], float edges[7*16*16].
kernel void dflash2_select_edges(device const uint* partial_ids [[buffer(0)]],
    device const float* partial_values [[buffer(1)]], device uchar* selector_out [[buffer(2)]],
    device const float* selector_hidden [[buffer(3)]], device const bfloat* predecessor_codebook [[buffer(4)]],
    device const bfloat* successor_codebook [[buffer(5)]], constant dflash2_detail::DFlash2SelectArgs& a [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]], uint row [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint MaxPos = 7u, Candidates = 16u, Rank = 256u, TaskCandidates = 8u, Dims = Rank / 32u;
    const uint cb_vocab = a.reserved ? a.reserved : a.vocabulary;
    if (!a.positions || a.positions > MaxPos || !a.vocabulary || cb_vocab < a.vocabulary || a.anchor >= cb_vocab) {
        if (!row && !sg && !lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (row >= a.positions) return;

    device uint* candidates = reinterpret_cast<device uint*>(selector_out);
    device float* unary = reinterpret_cast<device float*>(selector_out + MaxPos * Candidates * sizeof(uint));
    device float* tables = reinterpret_cast<device float*>(selector_out + MaxPos * Candidates * (sizeof(uint) + sizeof(float)));

    threadgroup uint successors[Candidates];
    threadgroup uint predecessors[Candidates];
    if (sg == 0u) {
        float val;
        uint tok;
        dflash2_detail::top16_merge_shards(partial_ids, partial_values, row, lane, val, tok);
        if (lane < Candidates) {
            candidates[row * Candidates + lane] = tok;
            unary[row * Candidates + lane] = val;
            successors[lane] = tok;
        }
    } else if (sg == 1u) {
        uint tok = a.anchor;
        if (row > 0u) {
            float val;
            dflash2_detail::top16_merge_shards(partial_ids, partial_values, row - 1u, lane, val, tok);
        }
        if (lane < Candidates) predecessors[lane] = tok;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint bad = 0u;
    const uint tasks = (row > 0u ? Candidates : 1u) * (Candidates / TaskCandidates);
    device float* table = tables + ulong(row) * Candidates * Candidates;
    device const float* row_hidden = selector_hidden + ulong(row) * Rank;
    for (uint task = sg; task < tasks; task += 8u) {
        const uint pred_idx = task / (Candidates / TaskCandidates);
        const uint first_cand = (task % (Candidates / TaskCandidates)) * TaskCandidates;
        const uint safe_pred = min(predecessors[pred_idx], cb_vocab - 1u);
        float ctx[Dims];
        float succ[TaskCandidates][Dims];
        #pragma unroll
        for (uint i = 0u; i < Dims; ++i) {
            const uint dim = uint(lane) + i * 32u;
            const float h = row_hidden[dim];
            const float p = float(predecessor_codebook[ulong(safe_pred) * Rank + dim]);
            bad |= dflash2_detail::nonfinite_bit(h) | dflash2_detail::nonfinite_bit(p);
            ctx[i] = p * h;
        }
        #pragma unroll
        for (uint j = 0u; j < TaskCandidates; ++j) {
            const uint safe_cand = min(successors[first_cand + j], cb_vocab - 1u);
            #pragma unroll
            for (uint i = 0u; i < Dims; ++i) {
                const float s = float(successor_codebook[ulong(safe_cand) * Rank + uint(lane) + i * 32u]);
                bad |= dflash2_detail::nonfinite_bit(s);
                succ[j][i] = s;
            }
        }
        #pragma unroll
        for (uint j = 0u; j < TaskCandidates; ++j) {
            float sc = 0.0f;
            #pragma unroll
            for (uint i = 0u; i < Dims; ++i) sc += ctx[i] * succ[j][i];
            sc = simd_sum(sc);
            if (!lane) {
                bad |= dflash2_detail::nonfinite_bit(sc);
                table[pred_idx * Candidates + first_cand + j] = sc;
            }
        }
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}

namespace dflash2_dense_detail {

static inline float stored_half_q8(device const uchar* p) {
    return float(as_type<half>(ushort(uint(p[0]) | (uint(p[1]) << 8u))));
}

static inline void write_dense_epilogue(float tot, device float* out, uint row, uint swiglu, thread uint& bad) {
    if (dflash2_detail::nonfinite_bit(tot)) {
        bad |= 1u;
        out[row] = NAN;
    } else if (swiglu) {
        const float g0 = out[row];
        if (dflash2_detail::nonfinite_bit(g0)) {
            bad |= 1u;
            out[row] = NAN;
        } else {
            const float act = g0 * dflash2_detail::sigmoid_f32(g0);
            const float prod = act * tot;
            if (dflash2_detail::nonfinite_bit(act) || dflash2_detail::nonfinite_bit(prod)) {
                bad |= 1u;
                out[row] = NAN;
            } else {
                out[row] = prod;
            }
        }
    } else {
        out[row] = tot;
    }
}

static inline void write_swiglu_pair_epilogue(float g0, float u0, device float* out, uint row, thread uint& bad) {
    if (dflash2_detail::nonfinite_bit(g0) || dflash2_detail::nonfinite_bit(u0)) {
        bad |= 1u;
        out[row] = NAN;
    } else {
        const float act = g0 * dflash2_detail::sigmoid_f32(g0);
        const float prod = act * u0;
        if (dflash2_detail::nonfinite_bit(act) || dflash2_detail::nonfinite_bit(prod)) {
            bad |= 1u;
            out[row] = NAN;
        } else {
            out[row] = prod;
        }
    }
}

template <uint RowsPerSG>
static inline void dense_bf16_staged(device const bfloat* matrix, device const float* input, device float* output,
    constant dflash2_detail::DFlash2DenseBf16Args& a, device atomic_uint* error, uint2 group,
    ushort tid, ushort sg, ushort lane, threadgroup float (&sx)[8][256]) {
    if (!a.columns || (a.columns % 32u) != 0u || !a.rows || !a.tokens || a.tokens > 2048u ||
        a.row_bytes != a.columns * 2u || (a.input_stride_bytes % 4ul) != 0ul || (a.output_stride_bytes % 4ul) != 0ul) {
        if (!group.x && !group.y && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    const uint t0 = group.y * 8u;
    if (t0 >= a.tokens) return;
    const uint nt = min(8u, a.tokens - t0);
    const uint row0 = (group.x * 8u + uint(sg)) * RowsPerSG;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const ulong out_stride = a.output_stride_bytes / sizeof(float);

    float sums[RowsPerSG][8];
    #pragma unroll
    for (uint r = 0u; r < RowsPerSG; ++r) {
        #pragma unroll
        for (uint t = 0u; t < 8u; ++t) sums[r][t] = 0.0f;
    }
    uint bad = 0u;

    for (uint col0 = 0u; col0 < a.columns; col0 += 256u) {
        const uint cnt = min(256u, a.columns - col0);
        for (uint t = 0u; t < nt; ++t) {
            if (tid < cnt) {
                float xv = input[ulong(t0 + t) * in_stride + col0 + tid];
                if (dflash2_detail::nonfinite_bit(xv)) { bad |= 1u; xv = 0.0f; }
                sx[t][tid] = xv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row0 < a.rows) {
            for (uint k = lane; k < cnt; k += 32u) {
                const uint col = col0 + k;
                #pragma unroll
                for (uint r = 0u; r < RowsPerSG; ++r) {
                    if (row0 + r < a.rows) {
                        float w = float(matrix[ulong(row0 + r) * a.columns + col]);
                        if (dflash2_detail::nonfinite_bit(w)) { bad |= 1u; w = 0.0f; }
                        for (uint t = 0u; t < nt; ++t) sums[r][t] += w * sx[t][k];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < a.rows) {
        #pragma unroll
        for (uint r = 0u; r < RowsPerSG; ++r) {
            if (row0 + r < a.rows) {
                for (uint t = 0u; t < nt; ++t) {
                    const float tot = simd_sum(sums[r][t]);
                    if (!lane) write_dense_epilogue(tot, output + ulong(t0 + t) * out_stride, row0 + r, a.swiglu, bad);
                }
            }
        }
    }
    bad = simd_or(bad);
    if (!lane && bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

template <uint T, uint RowsPerSG>
static inline void dense_q8_staged(device const uchar* weights, device const float* input, device float* output,
    constant dflash2_detail::DFlash2DenseBf16Args& a, device atomic_uint* error, uint group_x,
    ushort tid, ushort sg, ushort lane, threadgroup float (&sx)[8][256]) {
    const uint row0 = (group_x * 8u + uint(sg)) * RowsPerSG;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const ulong out_stride = a.output_stride_bytes / sizeof(float);
    float sums[RowsPerSG][T];
    #pragma unroll
    for (uint r = 0u; r < RowsPerSG; ++r) {
        #pragma unroll
        for (uint t = 0u; t < T; ++t) sums[r][t] = 0.0f;
    }
    uint bad = 0u;
    const uint blocks = a.columns / 32u;
    const uint full = blocks / 16u;

    for (uint b0 = 0u; b0 < blocks; b0 += 8u) {
        const uint cnt = min(8u, blocks - b0);
        const uint elems = cnt * 32u;
        if (uint(tid) < elems) {
            #pragma unroll
            for (uint t = 0u; t < T; ++t) {
                float xv = input[ulong(t) * in_stride + ulong(b0) * 32ul + uint(tid)];
                if (dflash2_detail::nonfinite_bit(xv)) { bad |= 1u; xv = 0.0f; }
                sx[t][tid] = xv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row0 < a.rows) {
            const uint active_rows = min(RowsPerSG, a.rows - row0);
            if (dflash2_q8_layout == 0u) {
                for (uint k = 0u; k < cnt; ++k) {
                    const uint b = b0 + k;
                    const uint col_local = k * 32u + uint(lane);
                    float xv[T];
                    #pragma unroll
                    for (uint t = 0u; t < T; ++t) xv[t] = sx[t][col_local];
                    #pragma unroll
                    for (uint r = 0u; r < RowsPerSG; ++r) {
                        if (r < active_rows) {
                            device const uchar* enc = weights + ulong(row0 + r) * a.row_bytes + ulong(b) * 34ul;
                            float sc = stored_half_q8(enc);
                            if (dflash2_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                            const float w = sc * float(as_type<char>(enc[2u + lane]));
                            #pragma unroll
                            for (uint t = 0u; t < T; ++t) sums[r][t] += w * xv[t];
                        }
                    }
                }
            } else if (b0 < full * 16u) {
                const uint run = b0 >> 4u;
                const uint half_run = (b0 >> 3u) & 1u;
                const ulong q_base = ulong(blocks) * 2ul + ulong(run) * 512ul + ulong(lane) * 16ul + ulong(half_run) * 8ul;
                #pragma unroll
                for (uint part = 0u; part < 2u; ++part) {
                    uint word_r[RowsPerSG];
                    half4 sc4_r[RowsPerSG];
                    #pragma unroll
                    for (uint r = 0u; r < RowsPerSG; ++r) {
                        if (r < active_rows) {
                            device const uchar* data = weights + ulong(row0 + r) * a.row_bytes;
                            device const half* scales = reinterpret_cast<device const half*>(data);
                            word_r[r] = *reinterpret_cast<device const uint*>(data + q_base + ulong(part) * 4ul);
                            sc4_r[r] = half4(*reinterpret_cast<device const packed_half4*>(scales + b0 + part * 4u));
                        }
                    }
                    #pragma unroll
                    for (uint k = 0u; k < 4u; ++k) {
                        const uint col_local = (part * 4u + k) * 32u + uint(lane);
                        float xv[T];
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) xv[t] = sx[t][col_local];
                        #pragma unroll
                        for (uint r = 0u; r < RowsPerSG; ++r) {
                            if (r < active_rows) {
                                float sc = float(sc4_r[r][k]);
                                if (dflash2_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                                const float w = sc * float(as_type<char>(uchar((word_r[r] >> (8u * k)) & 255u)));
                                #pragma unroll
                                for (uint t = 0u; t < T; ++t) sums[r][t] += w * xv[t];
                            }
                        }
                    }
                }
            } else {
                const uint tail_blocks = blocks - full * 16u;
                const uint tail_off = b0 - full * 16u;
                #pragma unroll
                for (uint r = 0u; r < RowsPerSG; ++r) {
                    if (r < active_rows) {
                        device const uchar* data = weights + ulong(row0 + r) * a.row_bytes;
                        device const half* scales = reinterpret_cast<device const half*>(data);
                        device const uchar* tail = data + ulong(blocks) * 2ul + ulong(full) * 512ul + ulong(lane) * tail_blocks + tail_off;
                        for (uint k = 0u; k < cnt; ++k) {
                            float sc = float(scales[b0 + k]);
                            if (dflash2_detail::nonfinite_bit(sc)) { bad |= 1u; sc = 0.0f; }
                            const float w = sc * float(as_type<char>(tail[k]));
                            const uint col_local = k * 32u + uint(lane);
                            #pragma unroll
                            for (uint t = 0u; t < T; ++t) sums[r][t] += w * sx[t][col_local];
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < a.rows) {
        #pragma unroll
        for (uint r = 0u; r < RowsPerSG; ++r) {
            if (row0 + r < a.rows) {
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float tot = simd_sum(sums[r][t]);
                    if (!lane) write_dense_epilogue(tot, output + ulong(t) * out_stride, row0 + r, a.swiglu, bad);
                }
            }
        }
    }
    bad = simd_or(bad);
    if (!lane && bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

template <uint T>
static inline void gate_up_swiglu_q8_staged(device const uchar* gate_w, device const uchar* up_w,
    device const float* input, device float* output, constant dflash2_detail::DFlash2DenseBf16Args& a,
    device atomic_uint* error, uint group_x, ushort tid, ushort sg, ushort lane, threadgroup float (&sx)[8][512]) {
    constexpr uint RowsPerSG = 2u;
    const uint row0 = (group_x * 8u + uint(sg)) * RowsPerSG;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const ulong out_stride = a.output_stride_bytes / sizeof(float);
    float g_sums[RowsPerSG][T];
    float u_sums[RowsPerSG][T];
    #pragma unroll
    for (uint r = 0u; r < RowsPerSG; ++r) {
        #pragma unroll
        for (uint t = 0u; t < T; ++t) {
            g_sums[r][t] = 0.0f;
            u_sums[r][t] = 0.0f;
        }
    }
    uint bad = 0u;
    const uint blocks = a.columns / 32u;
    const uint full = blocks / 16u;

    for (uint b0 = 0u; b0 < blocks; b0 += 16u) {
        const uint cnt = min(16u, blocks - b0);
        const uint elems = cnt * 32u;
        #pragma unroll
        for (uint t = 0u; t < T; ++t) {
            device const float* x_t = input + ulong(t) * in_stride + ulong(b0) * 32ul;
            if (uint(tid) < elems) {
                float xv = x_t[tid];
                if (dflash2_detail::nonfinite_bit(xv)) { bad |= 1u; xv = 0.0f; }
                sx[t][tid] = xv;
            }
            if (uint(tid) + 256u < elems) {
                float xv = x_t[uint(tid) + 256u];
                if (dflash2_detail::nonfinite_bit(xv)) { bad |= 1u; xv = 0.0f; }
                sx[t][uint(tid) + 256u] = xv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row0 < a.rows) {
            const uint active_rows = min(RowsPerSG, a.rows - row0);
            if (dflash2_q8_layout == 0u) {
                for (uint k = 0u; k < cnt; ++k) {
                    const uint b = b0 + k;
                    const uint col_local = k * 32u + uint(lane);
                    float xv[T];
                    #pragma unroll
                    for (uint t = 0u; t < T; ++t) xv[t] = sx[t][col_local];
                    #pragma unroll
                    for (uint r = 0u; r < RowsPerSG; ++r) {
                        if (r < active_rows) {
                            device const uchar* g_enc = gate_w + ulong(row0 + r) * a.row_bytes + ulong(b) * 34ul;
                            device const uchar* u_enc = up_w + ulong(row0 + r) * a.row_bytes + ulong(b) * 34ul;
                            float g_sc = stored_half_q8(g_enc), u_sc = stored_half_q8(u_enc);
                            if (dflash2_detail::nonfinite_bit(g_sc)) { bad |= 1u; g_sc = 0.0f; }
                            if (dflash2_detail::nonfinite_bit(u_sc)) { bad |= 1u; u_sc = 0.0f; }
                            const float gw = g_sc * float(as_type<char>(g_enc[2u + lane]));
                            const float uw = u_sc * float(as_type<char>(u_enc[2u + lane]));
                            #pragma unroll
                            for (uint t = 0u; t < T; ++t) {
                                g_sums[r][t] += gw * xv[t];
                                u_sums[r][t] += uw * xv[t];
                            }
                        }
                    }
                }
            } else if (cnt == 16u) {
                const uint run = b0 / 16u;
                const ulong q_off = ulong(blocks) * 2ul + ulong(run) * 512ul + ulong(lane) * 16ul;
                uint4 g_words[RowsPerSG];
                uint4 u_words[RowsPerSG];
                #pragma unroll
                for (uint r = 0u; r < RowsPerSG; ++r) {
                    if (r < active_rows) {
                        g_words[r] = uint4(*reinterpret_cast<device const packed_uint4*>(gate_w + ulong(row0 + r) * a.row_bytes + q_off));
                        u_words[r] = uint4(*reinterpret_cast<device const packed_uint4*>(up_w + ulong(row0 + r) * a.row_bytes + q_off));
                    }
                }
                #pragma unroll
                for (uint part = 0u; part < 4u; ++part) {
                    half4 g_sc4[RowsPerSG];
                    half4 u_sc4[RowsPerSG];
                    #pragma unroll
                    for (uint r = 0u; r < RowsPerSG; ++r) {
                        if (r < active_rows) {
                            device const half* g_scales = reinterpret_cast<device const half*>(gate_w + ulong(row0 + r) * a.row_bytes);
                            device const half* u_scales = reinterpret_cast<device const half*>(up_w + ulong(row0 + r) * a.row_bytes);
                            g_sc4[r] = half4(*reinterpret_cast<device const packed_half4*>(g_scales + b0 + part * 4u));
                            u_sc4[r] = half4(*reinterpret_cast<device const packed_half4*>(u_scales + b0 + part * 4u));
                        }
                    }
                    #pragma unroll
                    for (uint k = 0u; k < 4u; ++k) {
                        const uint col_local = (part * 4u + k) * 32u + uint(lane);
                        float xv[T];
                        #pragma unroll
                        for (uint t = 0u; t < T; ++t) xv[t] = sx[t][col_local];
                        #pragma unroll
                        for (uint r = 0u; r < RowsPerSG; ++r) {
                            if (r < active_rows) {
                                float g_sc = float(g_sc4[r][k]), u_sc = float(u_sc4[r][k]);
                                if (dflash2_detail::nonfinite_bit(g_sc)) { bad |= 1u; g_sc = 0.0f; }
                                if (dflash2_detail::nonfinite_bit(u_sc)) { bad |= 1u; u_sc = 0.0f; }
                                const float gw = g_sc * float(as_type<char>(uchar((g_words[r][part] >> (8u * k)) & 255u)));
                                const float uw = u_sc * float(as_type<char>(uchar((u_words[r][part] >> (8u * k)) & 255u)));
                                #pragma unroll
                                for (uint t = 0u; t < T; ++t) {
                                    g_sums[r][t] += gw * xv[t];
                                    u_sums[r][t] += uw * xv[t];
                                }
                            }
                        }
                    }
                }
            } else {
                const ulong t_off = ulong(blocks) * 2ul + ulong(full) * 512ul + ulong(lane) * cnt;
                for (uint k = 0u; k < cnt; ++k) {
                    const uint col_local = k * 32u + uint(lane);
                    float xv[T];
                    #pragma unroll
                    for (uint t = 0u; t < T; ++t) xv[t] = sx[t][col_local];
                    #pragma unroll
                    for (uint r = 0u; r < RowsPerSG; ++r) {
                        if (r < active_rows) {
                            device const uchar* g_data = gate_w + ulong(row0 + r) * a.row_bytes;
                            device const uchar* u_data = up_w + ulong(row0 + r) * a.row_bytes;
                            device const half* g_scales = reinterpret_cast<device const half*>(g_data);
                            device const half* u_scales = reinterpret_cast<device const half*>(u_data);
                            float g_sc = float(g_scales[b0 + k]), u_sc = float(u_scales[b0 + k]);
                            if (dflash2_detail::nonfinite_bit(g_sc)) { bad |= 1u; g_sc = 0.0f; }
                            if (dflash2_detail::nonfinite_bit(u_sc)) { bad |= 1u; u_sc = 0.0f; }
                            const float gw = g_sc * float(as_type<char>((g_data + t_off)[k]));
                            const float uw = u_sc * float(as_type<char>((u_data + t_off)[k]));
                            #pragma unroll
                            for (uint t = 0u; t < T; ++t) {
                                g_sums[r][t] += gw * xv[t];
                                u_sums[r][t] += uw * xv[t];
                            }
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row0 < a.rows) {
        #pragma unroll
        for (uint r = 0u; r < RowsPerSG; ++r) {
            if (row0 + r < a.rows) {
                #pragma unroll
                for (uint t = 0u; t < T; ++t) {
                    const float g0 = simd_sum(g_sums[r][t]);
                    const float u0 = simd_sum(u_sums[r][t]);
                    if (!lane) write_swiglu_pair_epilogue(g0, u0, output + ulong(t) * out_stride, row0 + r, bad);
                }
            }
        }
    }
    bad = simd_or(bad);
    if (!lane && bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

} // namespace dflash2_dense_detail

// Multi-row BF16 dense projection (1..2048 token rows, 8 or 32 output rows per 256-thread threadgroup with staged input).
kernel void dflash2_dense_bf16_m8(
    device const bfloat* matrix [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant dflash2_detail::DFlash2DenseBf16Args& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sx[8][256];
    dflash2_dense_detail::dense_bf16_staged<1u>(matrix, input, output, a, error, group, tid, sg, lane, sx);
}

kernel void dflash2_dense_bf16_r4_m8(
    device const bfloat* matrix [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant dflash2_detail::DFlash2DenseBf16Args& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sx[8][256];
    dflash2_dense_detail::dense_bf16_staged<4u>(matrix, input, output, a, error, group, tid, sg, lane, sx);
}

kernel void dflash2_gate_up_swiglu_bf16_m8(
    device const bfloat* gate_matrix [[buffer(0)]],
    device const bfloat* up_matrix [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant dflash2_detail::DFlash2DenseBf16Args& a [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (!a.columns || (a.columns % 32u) != 0u || !a.rows || !a.tokens || a.tokens > 8u ||
        a.row_bytes != a.columns * 2u || (a.input_stride_bytes % 4ul) != 0ul || (a.output_stride_bytes % 4ul) != 0ul) {
        if (!group.x && !group.y && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    constexpr uint RowsPerSG = 2u;
    const uint nt = a.tokens;
    const uint row0 = (group.x * 8u + uint(sg)) * RowsPerSG;
    const ulong in_stride = a.input_stride_bytes / sizeof(float);
    const ulong out_stride = a.output_stride_bytes / sizeof(float);
    threadgroup float sx[8][256];
    float g_sums[RowsPerSG][8];
    float u_sums[RowsPerSG][8];
    #pragma unroll
    for (uint r = 0u; r < RowsPerSG; ++r) {
        #pragma unroll
        for (uint t = 0u; t < 8u; ++t) { g_sums[r][t] = 0.0f; u_sums[r][t] = 0.0f; }
    }
    uint bad = 0u;
    for (uint col0 = 0u; col0 < a.columns; col0 += 256u) {
        const uint cnt = min(256u, a.columns - col0);
        for (uint t = 0u; t < nt; ++t) {
            if (tid < cnt) {
                float xv = input[ulong(t) * in_stride + col0 + tid];
                if (dflash2_detail::nonfinite_bit(xv)) { bad |= 1u; xv = 0.0f; }
                sx[t][tid] = xv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row0 < a.rows) {
            for (uint k = lane; k < cnt; k += 32u) {
                const uint col = col0 + k;
                #pragma unroll
                for (uint r = 0u; r < RowsPerSG; ++r) {
                    if (row0 + r < a.rows) {
                        float gw = float(gate_matrix[ulong(row0 + r) * a.columns + col]);
                        float uw = float(up_matrix[ulong(row0 + r) * a.columns + col]);
                        if (dflash2_detail::nonfinite_bit(gw)) { bad |= 1u; gw = 0.0f; }
                        if (dflash2_detail::nonfinite_bit(uw)) { bad |= 1u; uw = 0.0f; }
                        for (uint t = 0u; t < nt; ++t) {
                            const float xv = sx[t][k];
                            g_sums[r][t] += gw * xv;
                            u_sums[r][t] += uw * xv;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 < a.rows) {
        #pragma unroll
        for (uint r = 0u; r < RowsPerSG; ++r) {
            if (row0 + r < a.rows) {
                for (uint t = 0u; t < nt; ++t) {
                    const float g0 = simd_sum(g_sums[r][t]);
                    const float u0 = simd_sum(u_sums[r][t]);
                    if (!lane) dflash2_dense_detail::write_swiglu_pair_epilogue(g0, u0, output + ulong(t) * out_stride, row0 + r, bad);
                }
            }
        }
    }
    bad = simd_or(bad);
    if (!lane && bad) atomic_fetch_or_explicit(error, bad, memory_order_relaxed);
}

kernel void dflash2_dense_q8_m8(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant dflash2_detail::DFlash2DenseBf16Args& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (dflash2_q8_encoding != 8u || dflash2_q8_layout > 1u || !a.columns || (a.columns % 32u) != 0u ||
        !a.rows || !a.tokens || a.tokens > 8u || a.row_bytes != (a.columns / 32u) * 34u ||
        (a.input_stride_bytes % 4ul) != 0ul || (a.output_stride_bytes % 4ul) != 0ul) {
        if (!group.x && !group.y && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    threadgroup float sx[8][256];
    switch (a.tokens) {
        case 1u: dflash2_dense_detail::dense_q8_staged<1u, 1u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 2u: dflash2_dense_detail::dense_q8_staged<2u, 1u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 3u: dflash2_dense_detail::dense_q8_staged<3u, 1u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 4u: dflash2_dense_detail::dense_q8_staged<4u, 1u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 5u: dflash2_dense_detail::dense_q8_staged<5u, 1u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 6u: dflash2_dense_detail::dense_q8_staged<6u, 1u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 7u: dflash2_dense_detail::dense_q8_staged<7u, 1u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        default: dflash2_dense_detail::dense_q8_staged<8u, 1u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
    }
}

kernel void dflash2_dense_q8_r4_m8(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant dflash2_detail::DFlash2DenseBf16Args& a [[buffer(3)]],
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (dflash2_q8_encoding != 8u || dflash2_q8_layout > 1u || !a.columns || (a.columns % 32u) != 0u ||
        !a.rows || !a.tokens || a.tokens > 8u || a.row_bytes != (a.columns / 32u) * 34u ||
        (a.input_stride_bytes % 4ul) != 0ul || (a.output_stride_bytes % 4ul) != 0ul) {
        if (!group.x && !group.y && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    threadgroup float sx[8][256];
    switch (a.tokens) {
        case 1u: dflash2_dense_detail::dense_q8_staged<1u, 4u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 2u: dflash2_dense_detail::dense_q8_staged<2u, 4u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 3u: dflash2_dense_detail::dense_q8_staged<3u, 4u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 4u: dflash2_dense_detail::dense_q8_staged<4u, 4u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 5u: dflash2_dense_detail::dense_q8_staged<5u, 4u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 6u: dflash2_dense_detail::dense_q8_staged<6u, 4u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 7u: dflash2_dense_detail::dense_q8_staged<7u, 4u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
        default: dflash2_dense_detail::dense_q8_staged<8u, 4u>(weights, input, output, a, error, group.x, tid, sg, lane, sx); break;
    }
}

kernel void dflash2_gate_up_swiglu_q8_m8(
    device const uchar* gate_w [[buffer(0)]],
    device const uchar* up_w [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant dflash2_detail::DFlash2DenseBf16Args& a [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (dflash2_q8_encoding != 8u || dflash2_q8_layout > 1u || !a.columns || (a.columns % 32u) != 0u ||
        !a.rows || !a.tokens || a.tokens > 8u || a.row_bytes != (a.columns / 32u) * 34u ||
        (a.input_stride_bytes % 4ul) != 0ul || (a.output_stride_bytes % 4ul) != 0ul) {
        if (!group.x && !group.y && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    threadgroup float sx[8][512];
    switch (a.tokens) {
        case 1u: dflash2_dense_detail::gate_up_swiglu_q8_staged<1u>(gate_w, up_w, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 2u: dflash2_dense_detail::gate_up_swiglu_q8_staged<2u>(gate_w, up_w, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 3u: dflash2_dense_detail::gate_up_swiglu_q8_staged<3u>(gate_w, up_w, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 4u: dflash2_dense_detail::gate_up_swiglu_q8_staged<4u>(gate_w, up_w, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 5u: dflash2_dense_detail::gate_up_swiglu_q8_staged<5u>(gate_w, up_w, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 6u: dflash2_dense_detail::gate_up_swiglu_q8_staged<6u>(gate_w, up_w, input, output, a, error, group.x, tid, sg, lane, sx); break;
        case 7u: dflash2_dense_detail::gate_up_swiglu_q8_staged<7u>(gate_w, up_w, input, output, a, error, group.x, tid, sg, lane, sx); break;
        default: dflash2_dense_detail::gate_up_swiglu_q8_staged<8u>(gate_w, up_w, input, output, a, error, group.x, tid, sg, lane, sx); break;
    }
}

kernel void dflash2_conv_finish_rms_f32(
    device const float* input [[buffer(0)]],
    device const float* dynamic [[buffer(1)]],
    device const float* base [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    device const float* gains [[buffer(4)]],
    device float* residual_out [[buffer(5)]],
    device float* norm_out [[buffer(6)]],
    device atomic_uint* error [[buffer(7)]],
    constant dflash2_detail::DFlash2ConvRmsArgs& a [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (a.hidden != 2560u || !a.rows || a.rows > 8u || a.group_size != 16u || !(a.eps > 0.0f)) {
        if (!row && !tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (row >= a.rows) return;
    constexpr uint Hidden = 2560u, GroupSize = 16u, ConvGroups = Hidden / GroupSize, DynWidth = 4u * ConvGroups;
    threadgroup float row_buf[Hidden];
    threadgroup float s_inv;
    threadgroup uint s_bad;
    if (!tid) s_bad = 0u;
    uint bad = 0u;
    const ulong row_off = ulong(row) * Hidden;
    for (uint ch = tid; ch < Hidden; ch += 128u) {
        const uint cg = ch / GroupSize;
        const float x0 = input[row_off + ch];
        const float b0 = base[2u * Hidden + ch];
        const float d0 = dynamic[row * DynWidth + 2u * ConvGroups + cg];
        bad |= dflash2_detail::nonfinite_bit(x0) | dflash2_detail::nonfinite_bit(b0) | dflash2_detail::nonfinite_bit(d0);
        float val = x0 * (b0 + d0);
        if (row > 0u) {
            const float x1 = input[(row - 1u) * Hidden + ch];
            const float b1 = base[3u * Hidden + ch];
            const float d1 = dynamic[row * DynWidth + 3u * ConvGroups + cg];
            bad |= dflash2_detail::nonfinite_bit(x1) | dflash2_detail::nonfinite_bit(b1) | dflash2_detail::nonfinite_bit(d1);
            val += x1 * (b1 + d1);
        }
        const float res = residual[row_off + ch];
        bad |= dflash2_detail::nonfinite_bit(res);
        val += res;
        bad |= dflash2_detail::nonfinite_bit(val);
        residual_out[row_off + ch] = val;
        row_buf[ch] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float sum = 0.0f;
        for (uint j = lane; j < Hidden; j += 32u) {
            const float v = row_buf[j];
            sum += v * v;
        }
        sum = simd_sum(sum);
        const float denom = sum / float(Hidden) + a.eps;
        if (!isfinite(sum) || !isfinite(denom)) {
            if (!lane) { s_bad = 1u; s_inv = 0.0f; }
        } else if (!lane) {
            s_inv = rsqrt(denom);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s_bad) {
        bad |= 1u;
    } else {
        const float inv = s_inv;
        for (uint ch = tid; ch < Hidden; ch += 128u) {
            const float g = gains[ch];
            bad |= dflash2_detail::nonfinite_bit(g);
            const float out_v = (row_buf[ch] * inv) * g;
            bad |= dflash2_detail::nonfinite_bit(out_v);
            norm_out[row_off + ch] = out_v;
        }
    }
    if (bad) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
}
