#include <metal_stdlib>
using namespace metal;

// Original native primitives. Dense GQA follows the equations of the pinned
// qwen4exp nextn graph, without QSA block selection. BF16 cache and the existing
// owned attention lane/online-softmax arithmetic are retained explicitly.
void mtp_checked_store(device float* output, uint i, float value, device atomic_uint* error) {
    if (!isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    output[i] = value;
}
float mtp_bf16(ushort value) { return as_type<float>(uint(value) << 16); }

kernel void mtp_concat(device const float* embedding [[buffer(0)]], device const float* hidden [[buffer(1)]],
                       device float* output [[buffer(2)]], constant uint2& s [[buffer(3)]],
                       device atomic_uint* error [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= s.x * s.y * 2) return;
    const uint branch = i / (2 * s.y), channel = i % (2 * s.y);
    const float value = channel < s.y ? embedding[channel] : hidden[branch * s.y + channel - s.y];
    mtp_checked_store(output, i, value, error);
}

kernel void mtp_dense_attention(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
                                device const ushort* values [[buffer(2)]], device float* output [[buffer(3)]],
                                constant uint4& s [[buffer(4)]], device atomic_uint* error [[buffer(7)]],
                                uint group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group * 4 + subgroup;
    if (head >= s.x) return;
    const uint key_head = head / (s.x / s.y), row_width = s.y * s.z;
    float maximum = -INFINITY, denominator = 0, sum[8];
    for (uint c = 0; c < 8; ++c) sum[c] = 0;
    for (uint token = 0; token < s.w; ++token) {
        const uint base = token * row_width + key_head * s.z;
        float dot = 0;
        for (uint d = lane; d < s.z; d += 32) dot += query[head * s.z + d] * mtp_bf16(keys[base + d]);
        const float score = simd_sum(dot) * rsqrt(float(s.z));
        if (!isfinite(score)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        const float next_max = max(maximum, score), old_scale = exp(maximum - next_max), scale = exp(score - next_max);
        denominator = denominator * old_scale + scale;
        for (uint c = 0, d = lane; d < s.z; ++c, d += 32) sum[c] = sum[c] * old_scale + scale * mtp_bf16(values[base + d]);
        maximum = next_max;
    }
    for (uint c = 0, d = lane; d < s.z; ++c, d += 32) mtp_checked_store(output, head * s.z + d, sum[c] / denominator, error);
}

// Original draft-Q-only tiled online softmax. Each SIMD group owns one
// (query-head,key-tile), storing maximum, denominator and UNNORMALISED value
// numerators. The later merge rescales partials to one global maximum. No model
// engine code is copied; these are the existing owned attention equations with
// an explicit different FP32 reduction association for the auxiliary proposal.
kernel void mtp_dense_attention_tile(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
                                    device const ushort* values [[buffer(2)]], device float* partials [[buffer(3)]],
                                    constant uint4& s [[buffer(4)]], constant uint& tile_tokens [[buffer(5)]],
                                    device atomic_uint* error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]],
                                    ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group.x * 4 + subgroup, tile = group.y;
    if (head >= s.x) return;
    const uint tiles = (s.w + tile_tokens - 1) / tile_tokens;
    const uint start = tile * tile_tokens, end = min(start + tile_tokens, s.w);
    const uint key_head = head / (s.x / s.y), row_width = s.y * s.z;
    float maximum = -INFINITY, denominator = 0, sum[8];
    for (uint c = 0; c < 8; ++c) sum[c] = 0;
    for (uint token = start; token < end; ++token) {
        const uint base = token * row_width + key_head * s.z;
        float dot = 0;
        for (uint d = lane; d < s.z; d += 32) dot += query[head * s.z + d] * mtp_bf16(keys[base + d]);
        const float score = simd_sum(dot) * rsqrt(float(s.z));
        if (!isfinite(score)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        const float next_max = max(maximum, score), old_scale = exp(maximum - next_max), scale = exp(score - next_max);
        denominator = denominator * old_scale + scale;
        for (uint c = 0, d = lane; d < s.z; ++c, d += 32) sum[c] = sum[c] * old_scale + scale * mtp_bf16(values[base + d]);
        maximum = next_max;
    }
    const uint out = (head * tiles + tile) * (s.z + 2);
    if (lane == 0) {
        mtp_checked_store(partials, out, maximum, error);
        mtp_checked_store(partials, out + 1, denominator, error);
    }
    for (uint c = 0, d = lane; d < s.z; ++c, d += 32) mtp_checked_store(partials, out + 2 + d, sum[c], error);
}

kernel void mtp_dense_attention_merge(device const float* partials [[buffer(0)]], device float* output [[buffer(1)]],
                                     constant uint4& s [[buffer(4)]], constant uint& tile_tokens [[buffer(5)]],
                                     device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
                                     ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group * 4 + subgroup;
    if (head >= s.x) return;
    const uint tiles = (s.w + tile_tokens - 1) / tile_tokens, stride = s.z + 2;
    const uint base = head * tiles * stride;
    float maximum = -INFINITY;
    for (uint tile = lane; tile < tiles; tile += 32) maximum = max(maximum, partials[base + tile * stride]);
    maximum = simd_max(maximum);
    float denominator = 0, sum[8];
    for (uint c = 0; c < 8; ++c) sum[c] = 0;
    for (uint tile = 0; tile < tiles; ++tile) {
        const uint at = base + tile * stride;
        const float scale = exp(partials[at] - maximum);
        denominator += partials[at + 1] * scale;
        for (uint c = 0, d = lane; d < s.z; ++c, d += 32) sum[c] += partials[at + 2 + d] * scale;
    }
    if (!isfinite(denominator) || denominator <= 0) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    for (uint c = 0, d = lane; d < s.z; ++c, d += 32) mtp_checked_store(output, head * s.z + d, sum[c] / denominator, error);
}

kernel void mtp_embed_from_topk_q8(device const uchar* token_embd [[buffer(0)]],
                                   device const uint* topk_row [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   constant uint2& shape [[buffer(3)]],
                                   device atomic_uint* error [[buffer(7)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= shape.x) return;
    const uint count = topk_row[1];
    const uint nonfinite_col = topk_row[2];
    const uint token_id = topk_row[4];
    if (nonfinite_col != 0xffffffffu || count == 0u) {
        if (i == 0u) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[i] = 0.0f;
        return;
    }
    if (token_id >= shape.y) {
        if (i == 0u) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        output[i] = 0.0f;
        return;
    }
    const ulong row_bytes = ulong(shape.x / 32u) * 34ul;
    device const uchar* block = token_embd + ulong(token_id) * row_bytes + ulong(i / 32u) * 34ul;
    const ushort d_bits = ushort(block[0]) | (ushort(block[1]) << 8);
    if ((d_bits & 0x7c00u) == 0x7c00u) {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[i] = 0.0f;
        return;
    }
    const float d = float(as_type<half>(d_bits));
    const float q = float(as_type<char>(block[2u + (i & 31u)]));
    mtp_checked_store(output, i, d * q, error);
}

struct MtpSampleEmbedArgs {
    uint channels;
    uint embd_vocab;
    uint max_k;
    uint embed_flag;
    float inv_temp;
    float min_p;
    float top_p;
    float pad;
};

struct MtpSampleTicketsBuf {
    uint cursor;
    uint tickets24[3];
    uint reserved[4];
};

struct MtpSampledLawBuf {
    uint count;
    uint chosen_token;
    uint used_ticket;
    uint ticket_index;
    uint ids[20];
    uint units24[20];
    uint reserved[20];
};
static_assert(sizeof(MtpSampleTicketsBuf) == 32, "MtpSampleTicketsBuf ABI");
static_assert(sizeof(MtpSampledLawBuf) == 256, "MtpSampledLawBuf ABI");

kernel void mtp_sample_and_embed_from_topk_q8(device const uchar* token_embd [[buffer(0)]],
                                              device const uint* topk_row [[buffer(1)]],
                                              device MtpSampleTicketsBuf* tickets_buf [[buffer(2)]],
                                              device MtpSampledLawBuf* law_out [[buffer(3)]],
                                              device float* output [[buffer(4)]],
                                              constant MtpSampleEmbedArgs& a [[buffer(5)]],
                                              device atomic_uint* error [[buffer(7)]],
                                              ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint shared_meta[3];
    threadgroup uint tg_ids[20];
    threadgroup float tg_logits[20];
    threadgroup float tg_w[20];
    threadgroup uint tg_out_ids[20];
    threadgroup uint tg_out_u24[20];

    if (tid < 3u) {
        shared_meta[tid] = 0u;
    }
    if (tid < 20u) {
        tg_out_ids[tid] = 0u;
        tg_out_u24[tid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        const uint topk_vocab = topk_row[0];
        const uint count = topk_row[1];
        const uint nonfinite_col = topk_row[2];
        if (nonfinite_col != 0xffffffffu || count == 0u || topk_vocab == 0u ||
            !isfinite(a.inv_temp) || a.inv_temp <= 0.0f) {
            atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            law_out->count = 0u;
            law_out->chosen_token = 0u;
        } else {
            const uint k_cap = min(count, min(max(a.max_k, 1u), 20u));
            device const uint* row_ids = topk_row + 4u;
            device const float* row_logits = reinterpret_cast<device const float*>(topk_row + 68u);
            bool bad_row = false;
            for (uint i = 0u; i < k_cap; ++i) {
                const uint id = row_ids[i];
                const float lg = row_logits[i];
                if (id >= topk_vocab) {
                    atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
                    bad_row = true;
                }
                if (!isfinite(lg)) {
                    atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                    bad_row = true;
                }
                tg_ids[i] = id;
                tg_logits[i] = lg;
            }
            if (!bad_row) {
                const float l0 = tg_logits[0];
                uint kept = 0u;
                for (uint i = 0u; i < k_cap; ++i) {
                    float wi = (i == 0u) ? 1.0f : exp((tg_logits[i] - l0) * a.inv_temp);
                    if (!isfinite(wi)) {
                        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                        wi = (i == 0u) ? 1.0f : 0.0f;
                    }
                    if (i == 0u || (wi > 0.0f && wi >= a.min_p)) {
                        tg_ids[kept] = tg_ids[i];
                        tg_logits[kept] = tg_logits[i];
                        tg_w[kept] = wi;
                        ++kept;
                    }
                }
                if (a.top_p < 1.0f && kept > 1u) {
                    float sum_mp = 0.0f;
                    for (uint i = 0u; i < kept; ++i) sum_mp += tg_w[i];
                    const float thresh = a.top_p * sum_mp;
                    float cum_mp = 0.0f;
                    uint tp_kept = kept;
                    for (uint i = 0u; i < kept; ++i) {
                        cum_mp += tg_w[i];
                        if (cum_mp >= thresh) {
                            tp_kept = i + 1u;
                            while (tp_kept < kept && tg_logits[tp_kept] == tg_logits[i]) ++tp_kept;
                            break;
                        }
                    }
                    kept = tp_kept;
                }
                constexpr uint kGrid24 = 1u << 24u;
                constexpr float kGrid24F = 16777216.0f;
                float sum_w = 0.0f;
                for (uint i = 0u; i < kept; ++i) sum_w += tg_w[i];
                uint out_n = 0u;
                if (!(sum_w > 0.0f) || !isfinite(sum_w) || kept <= 1u) {
                    tg_out_ids[0] = tg_ids[0];
                    tg_out_u24[0] = kGrid24;
                    out_n = 1u;
                } else {
                    const float inv_sum = 1.0f / sum_w;
                    uint prev24 = 0u;
                    float cum_w = 0.0f;
                    for (uint i = 0u; i < kept; ++i) {
                        cum_w += tg_w[i];
                        uint end24;
                        if (i + 1u == kept) {
                            end24 = kGrid24;
                        } else {
                            const float scaled = clamp(cum_w * inv_sum, 0.0f, 1.0f) * kGrid24F;
                            end24 = scaled >= kGrid24F ? kGrid24 : uint(scaled);
                            if (end24 < prev24) end24 = prev24;
                        }
                        const uint u24 = end24 - prev24;
                        prev24 = end24;
                        if (u24 > 0u) {
                            tg_out_ids[out_n] = tg_ids[i];
                            tg_out_u24[out_n] = u24;
                            ++out_n;
                        }
                    }
                    if (out_n == 0u) {
                        tg_out_ids[0] = tg_ids[0];
                        tg_out_u24[0] = kGrid24;
                        out_n = 1u;
                    }
                }
                const bool det = (out_n == 1u);
                const uint t_idx = tickets_buf->cursor;
                const uint ticket24 = det ? 0u : tickets_buf->tickets24[min(t_idx, 2u)];
                if (!det) tickets_buf->cursor = t_idx + 1u;
                uint cum24 = 0u;
                uint chosen = tg_out_ids[out_n - 1u];
                bool picked = false;
                for (uint j = 0u; j < out_n; ++j) {
                    cum24 += tg_out_u24[j];
                    if (!picked && ticket24 < cum24) {
                        chosen = tg_out_ids[j];
                        picked = true;
                    }
                }
                law_out->count = out_n;
                law_out->chosen_token = chosen;
                law_out->used_ticket = det ? 0u : 1u;
                law_out->ticket_index = t_idx;
                shared_meta[0] = chosen;
                shared_meta[1] = 1u;
                shared_meta[2] = out_n;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (shared_meta[1] != 0u && tid < 20u) {
        const uint n_valid = shared_meta[2];
        law_out->ids[tid] = (tid < n_valid) ? tg_out_ids[tid] : 0u;
        law_out->units24[tid] = (tid < n_valid) ? tg_out_u24[tid] : 0u;
        law_out->reserved[tid] = 0u;
    }

    if (a.embed_flag == 0u) return;
    const uint token_id = shared_meta[0];
    const bool ok = (shared_meta[1] != 0u) && (token_id < a.embd_vocab);
    if (!ok) {
        if (tid == 0u && shared_meta[1] != 0u && token_id >= a.embd_vocab) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        }
        for (uint i = tid; i < a.channels; i += 128u) output[i] = 0.0f;
        return;
    }
    const ulong row_bytes = ulong(a.channels / 32u) * 34ul;
    for (uint i = tid; i < a.channels; i += 128u) {
        device const uchar* block = token_embd + ulong(token_id) * row_bytes + ulong(i / 32u) * 34ul;
        const ushort d_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        if ((d_bits & 0x7c00u) == 0x7c00u) {
            atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[i] = 0.0f;
        } else {
            const float d = float(as_type<half>(d_bits));
            const float q = float(as_type<char>(block[2u + (i & 31u)]));
            mtp_checked_store(output, i, d * q, error);
        }
    }
}
