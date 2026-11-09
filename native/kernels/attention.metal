#include <metal_stdlib>
using namespace metal;

void qsa_store(device float* out, uint i, float x, device atomic_uint* error) {
    if (!isfinite(x)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    out[i] = x;
}
float qsa_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }

kernel void qsa_split(device const float* packed [[buffer(0)]], device float* query [[buffer(1)]],
                      device float* gate [[buffer(2)]], constant uint4& s [[buffer(3)]],
                      device atomic_uint* error [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= s.x * s.y) return;
    const uint q = s.z ? i : (i / s.y) * 2 * s.y + i % s.y;
    const uint z = q + (s.z ? s.x * s.y : s.y);
    qsa_store(query, i, packed[q], error); qsa_store(gate, i, packed[z], error);
}

kernel void qsa_rotary(device const float* input [[buffer(0)]], device const float* frequency [[buffer(1)]],
                       device float* output [[buffer(2)]], constant uint4& s [[buffer(3)]], constant uint& mode [[buffer(4)]],
                       device atomic_uint* error [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= s.x * s.y) return;
    const uint d = i % s.y, base = i - d;
    if (d < s.z) {
        if (!isfinite(frequency[d]) || frequency[d] <= 0) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        const uint a = base + (mode ? 2 * d : d), b = a + (mode ? 1 : s.z);
        const float angle = float(s.w) * frequency[d], c = cos(angle), t = sin(angle);
        qsa_store(output, a, input[a] * c - input[b] * t, error);
        qsa_store(output, b, input[b] * c + input[a] * t, error);
    } else if (d >= 2 * s.z) qsa_store(output, i, input[i], error);
}

kernel void qsa_store_bf16(device const float* input [[buffer(0)]], device ushort* output [[buffer(1)]],
                           constant uint& count [[buffer(2)]], device atomic_uint* error [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint bits = as_type<uint>(input[i]);
    const ushort rounded = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
    if (!isfinite(input[i]) || (rounded & 0x7f80u) == 0x7f80u) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    output[i] = rounded;
}

kernel void qsa_index_accumulate(device const float* sum [[buffer(0)]], device const float* key [[buffer(1)]],
                                 device float* next [[buffer(2)]], device float* pooled [[buffer(3)]], constant uint4& s [[buffer(4)]],
                                 device atomic_uint* error [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= s.x) return;
    const float value = sum[i] + key[i];
    if (!isfinite(value)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    next[i] = s.y ? 0 : value; pooled[i] = s.y ? value / float(s.z) : 0;
}

kernel void qsa_index_scores(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
                             device float* output [[buffer(2)]], constant uint4& s [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
                             uint group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    const uint b = group * 4 + subgroup;
    if (b >= s.z) return;
    float result = 0;
    for (uint h = 0; h < s.x; ++h) {
        float dot = 0;
        for (uint d = lane; d < s.y; d += 32) dot += query[h * s.y + d] * qsa_bf16(keys[b * s.y + d]);
        dot = simd_sum(dot);
        if (!isfinite(dot)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        result += max(dot, 0.0f);
    }
    if (lane == 0) qsa_store(output, b, result, error);
}

bool qsa_before(float a, uint ai, float b, uint bi) { return a > b || (a == b && ai < bi); }

// Bounded local sorting followed by exact parallel merges; no B-by-K repeated
// scan and no full-sequence threadgroup allocation. Exact score ties prefer ID.
kernel void qsa_sort_chunks(device const float* input [[buffer(0)]], device float* scores [[buffer(1)]],
                            device uint* ids [[buffer(2)]], constant uint2& s [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
                            uint group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float local_scores[512]; threadgroup uint local_ids[512];
    for (uint i = tid; i < 512; i += 128) {
        const uint id = group * 512 + i;
        const float score = id < s.x ? input[id] : -INFINITY;
        if (id < s.x && (!isfinite(score) || score < 0)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        local_scores[i] = score; local_ids[i] = id < s.x ? id : 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= 512; k *= 2) for (uint stride = k / 2; stride; stride /= 2) {
        for (uint i = tid; i < 512; i += 128) {
            const uint j = i ^ stride;
            if (j > i) {
                const bool before = qsa_before(local_scores[i], local_ids[i], local_scores[j], local_ids[j]);
                if (((i & k) == 0 && !before) || ((i & k) != 0 && before)) {
                    const float a = local_scores[i]; const uint ai = local_ids[i];
                    local_scores[i] = local_scores[j]; local_ids[i] = local_ids[j]; local_scores[j] = a; local_ids[j] = ai;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint count = min(s.y, min(512u, s.x - group * 512));
    for (uint i = tid; i < count; i += 128) { scores[group * s.y + i] = local_scores[i]; ids[group * s.y + i] = local_ids[i]; }
}

kernel void qsa_merge(device const float* scores [[buffer(0)]], device const uint* ids [[buffer(1)]],
                       device float* out_scores [[buffer(2)]], device uint* out_ids [[buffer(3)]], constant uint4& s [[buffer(4)]],
                       uint group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint start = group * s.z * 2;
    const uint left = min(s.y, s.x > start ? s.x - start : 0u);
    const uint right = min(s.y, s.x > start + s.z ? s.x - start - s.z : 0u);
    const uint base = group * s.y * 2;
    for (uint item = tid; item < left + right; item += 128) {
        const bool r = item >= left;
        const uint own = r ? item - left : item;
        const uint at = base + (r ? s.y : 0) + own, other = base + (r ? 0 : s.y), count = r ? left : right;
        const float score = scores[at]; const uint id = ids[at];
        uint low = 0, high = count;
        while (low < high) {
            const uint mid = (low + high) / 2;
            if (qsa_before(scores[other + mid], ids[other + mid], score, id)) low = mid + 1; else high = mid;
        }
        const uint rank = own + low;
        if (rank < s.y) { out_scores[group * s.y + rank] = score; out_ids[group * s.y + rank] = id; }
    }
}

kernel void qsa_canonical_ids(device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]], constant uint2& s [[buffer(2)]],
                              device atomic_uint* error [[buffer(7)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint ids[512];
    for (uint i = tid; i < 512; i += 128) {
        ids[i] = i < s.x ? input[i] : 0xffffffffu;
        if (i < s.x && ids[i] >= s.y) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= 512; k *= 2) for (uint stride = k / 2; stride; stride /= 2) {
        for (uint i = tid; i < 512; i += 128) {
            const uint j = i ^ stride;
            if (j > i && (((i & k) == 0 && ids[i] > ids[j]) || ((i & k) != 0 && ids[i] < ids[j]))) {
                const uint a = ids[i]; ids[i] = ids[j]; ids[j] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < s.x; i += 128) output[i] = ids[i];
}

// One SIMD group per query head, online max-subtracted softmax, coalesced BF16
// K/V reads. Each lane owns up to eight output channels. Tail is always included.
kernel void qsa_attention(device const float* query [[buffer(0)]], device const ushort* keys [[buffer(1)]],
                          device const ushort* values [[buffer(2)]], device const uint* ids [[buffer(3)]],
                          device float* output [[buffer(4)]], constant uint4& s [[buffer(5)]], constant uint4& p [[buffer(6)]],
                          device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
                          ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group * 4 + subgroup;
    if (head >= s.x) return;
    const uint key_head = head / (s.x / s.y), row_width = s.y * s.z, blocks = p.x / s.w;
    const uint selected = min(blocks, p.y), tail = p.x % s.w, tokens = selected * s.w + tail;
    float maximum = -INFINITY, denominator = 0, sum[8];
    for (uint c = 0; c < 8; ++c) sum[c] = 0;
    for (uint item = 0; item < tokens; ++item) {
        uint token;
        if (item < selected * s.w) {
            const uint slot = item / s.w, id = blocks <= p.y ? slot : ids[slot];
            if (id >= blocks || (blocks > p.y && slot && id <= ids[slot - 1])) { atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); return; }
            token = id * s.w + item % s.w;
        } else token = blocks * s.w + item - selected * s.w;
        const uint base = token * row_width + key_head * s.z;
        float dot = 0;
        for (uint d = lane; d < s.z; d += 32) dot += query[head * s.z + d] * qsa_bf16(keys[base + d]);
        const float score = simd_sum(dot) * rsqrt(float(s.z));
        if (!isfinite(score)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        const float next_max = max(maximum, score), old_scale = exp(maximum - next_max), scale = exp(score - next_max);
        denominator = denominator * old_scale + scale;
        for (uint c = 0, d = lane; d < s.z; ++c, d += 32) sum[c] = sum[c] * old_scale + scale * qsa_bf16(values[base + d]);
        maximum = next_max;
    }
    for (uint c = 0, d = lane; d < s.z; ++c, d += 32) qsa_store(output, head * s.z + d, sum[c] / denominator, error);
}
