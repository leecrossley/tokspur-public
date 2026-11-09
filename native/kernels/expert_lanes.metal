#include <metal_stdlib>
using namespace metal;

// Private lossless expert-layout spike. Each kernel keeps the owned gemv.metal lane assignment (lane L owns columns L, L+32, ...), its decoded value expression, one scalar accumulator per input row in ascending column order and one simd_sum. Only the location of the bytes a lane reads changes.
//
// Buffers: 0 weights, 1 inputs [row][C], 2 outputs [row][R], 3 shape (columns, rows, row bytes, experts), 4 tasks, 5 limits (tasks, input rows, output rows, 0), 7 error flags. A task is {expert, n, in[T], out[T]} with 1 <= n <= T input rows that share the expert's weights.

template<uint T>
inline bool admit_task(device const uint* tasks, constant uint4& s, constant uint4& limits, uint task_id,
                       thread uint& expert, thread uint& n, thread uint (&in)[T], thread uint (&out)[T]) {
    device const uint* task = tasks + ulong(task_id) * (2 + 2 * T);
    expert = task[0]; n = task[1];
    if (expert >= s.w || n == 0 || n > T) return false;
    for (uint i = 0; i < T; ++i) {
        const uint k = min(i, n - 1);
        in[i] = task[2 + k]; out[i] = task[2 + T + k];
        if (in[i] >= limits.y || out[i] >= limits.z) return false;
    }
    return true;
}

template<uint T>
inline void publish(thread float (&sums)[T], uint n, thread uint (&out)[T], device float* output, uint rows, uint row,
                    ushort lane, device atomic_uint* error) {
    for (uint i = 0; i < T; ++i) {
        if (i >= n) continue;
        const float total = simd_sum(sums[i]);
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[ulong(out[i]) * rows + row] = total;
        }
    }
}

inline uint q4k_scale(uchar4 scales, uchar4 extra, uint group) {
    const uint slot = group % 4;
    return group < 4 ? (uint(scales[slot]) & 63) : ((uint(extra[slot]) & 15) | ((uint(scales[slot]) >> 6) << 4));
}
inline uint q4k_minimum(uchar4 minima, uchar4 extra, uint group) {
    const uint slot = group % 4;
    return group < 4 ? (uint(minima[slot]) & 63) : ((uint(extra[slot]) >> 4) | ((uint(minima[slot]) >> 6) << 4));
}

// Q4_K: Packed selects the lane-run layout, otherwise the original GGUF blocks.
template<uint C, uint T, bool Packed>
void q4k(device const uchar* weights, device const float* input, device float* output, constant uint4& s,
         device const uint* tasks, constant uint4& limits, device atomic_uint* error,
         uint2 group, ushort subgroup, ushort lane) {
    constexpr uint B = C / 256;
    const uint row = group.x * 4 + subgroup;
    if (row >= s.y || group.y >= limits.x) return;
    uint expert, n, in[T], out[T];
    if (s.x != C || s.z != B * 144 || !admit_task<T>(tasks, s, limits, group.y, expert, n, in, out)) {
        if (!lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    device const uchar* base = weights + (ulong(expert) * s.y + row) * s.z;
    float sums[T];
    for (uint i = 0; i < T; ++i) sums[i] = 0;
    #pragma unroll
    for (uint run = 0; run < (B + 3) / 4; ++run) {
        const uint count = min(4u, B - run * 4);
        uint words[4] = {0, 0, 0, 0};
        if (Packed) {
            // One 16-byte (or shorter final) load holds this lane's quant bytes for up to four consecutive blocks: block k, pair p at byte 4k+p.
            device const uchar* bytes = base + B * 16 + run * 512 + lane * count * 4;
            if (count == 4) {
                const uint4 v = *reinterpret_cast<device const uint4*>(bytes);
                words[0] = v.x; words[1] = v.y; words[2] = v.z; words[3] = v.w;
            } else if (count == 2) {
                const uint2 v = *reinterpret_cast<device const uint2*>(bytes);
                words[0] = v.x; words[1] = v.y;
            } else {
                for (uint k = 0; k < count; ++k) words[k] = *reinterpret_cast<device const uint*>(bytes + 4 * k);
            }
        }
        #pragma unroll
        for (uint k = 0; k < count; ++k) {
            const uint b = run * 4 + k;
            device const uchar* block = Packed ? base + b * 16 : base + b * 144;
            const uint4 h = *reinterpret_cast<device const uint4*>(block);
            const float d = float(as_type<half2>(h.x).x), dmin = float(as_type<half2>(h.x).y);
            const uchar4 scales = as_type<uchar4>(h.y), minima = as_type<uchar4>(h.z), extra = as_type<uchar4>(h.w);
            #pragma unroll
            for (uint pair = 0; pair < 4; ++pair) {
                const uint bits = Packed ? (words[k] >> (8 * pair)) & 255 : uint(block[16 + pair * 32 + lane]);
                #pragma unroll
                for (uint nibble = 0; nibble < 2; ++nibble) {
                    const uint group = pair * 2 + nibble;
                    const uint scale = q4k_scale(scales, extra, group);
                    const uint minimum = q4k_minimum(minima, extra, group);
                    const float q = float((bits >> (nibble * 4)) & 15);
                    const float w = (d * float(scale)) * q - dmin * float(minimum);
                    for (uint i = 0; i < T; ++i)
                        if (i < n) sums[i] += w * input[ulong(in[i]) * C + b * 256 + group * 32 + lane];
                }
            }
        }
    }
    publish<T>(sums, n, out, output, s.y, row, lane, error);
}

// Q5_1: lanes L and L+16 share one quant byte per block; nibble L/16.
template<uint C, uint T, bool Packed>
void q51(device const uchar* weights, device const float* input, device float* output, constant uint4& s,
         device const uint* tasks, constant uint4& limits, device atomic_uint* error,
         uint2 group, ushort subgroup, ushort lane) {
    constexpr uint B = C / 32;
    static_assert(B % 4 == 0, "Q5_1 lane runs use whole 32-bit words");
    const uint row = group.x * 4 + subgroup;
    if (row >= s.y || group.y >= limits.x) return;
    uint expert, n, in[T], out[T];
    if (s.x != C || s.z != B * 24 || !admit_task<T>(tasks, s, limits, group.y, expert, n, in, out)) {
        if (!lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    device const uchar* base = weights + (ulong(expert) * s.y + row) * s.z;
    const uint j = lane % 16, shift = 4 * (lane / 16);
    float sums[T];
    for (uint i = 0; i < T; ++i) sums[i] = 0;
    #pragma unroll
    for (uint run = 0; run < (B + 15) / 16; ++run) {
        const uint count = min(16u, B - run * 16);
        uint words[4] = {0, 0, 0, 0};
        if (Packed) {
            device const uchar* bytes = base + B * 8 + run * 256 + j * count;
            if (count == 16) {
                const uint4 v = *reinterpret_cast<device const uint4*>(bytes);
                words[0] = v.x; words[1] = v.y; words[2] = v.z; words[3] = v.w;
            } else {
                for (uint k = 0; k < count / 4; ++k) words[k] = *reinterpret_cast<device const uint*>(bytes + 4 * k);
            }
        }
        #pragma unroll
        for (uint k = 0; k < count; ++k) {
            const uint b = run * 16 + k;
            device const uchar* block = Packed ? base + b * 8 : base + b * 24;
            const uint2 h = *reinterpret_cast<device const uint2*>(block);
            const float d = float(as_type<half2>(h.x).x), m = float(as_type<half2>(h.x).y);
            const uint byte = Packed ? (words[k / 4] >> (8 * (k % 4))) & 255 : uint(block[8 + j]);
            const uint low = (byte >> shift) & 15;
            const uint q = low | (((h.y >> lane) & 1) << 4);
            const float w = d * float(q) + m;
            for (uint i = 0; i < T; ++i)
                if (i < n) sums[i] += w * input[ulong(in[i]) * C + b * 32 + lane];
        }
    }
    publish<T>(sums, n, out, output, s.y, row, lane, error);
}

#define LANES_KERNEL(NAME, FN, C, T, PACKED) \
kernel void NAME(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]], \
                 device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], \
                 device const uint* tasks [[buffer(4)]], constant uint4& limits [[buffer(5)]], \
                 device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]], \
                 ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) { \
    FN<C, T, PACKED>(weights, input, output, shape, tasks, limits, error, group, subgroup, lane); \
}
#define LANES_FAMILY(T) \
LANES_KERNEL(lanes_q4k_c2560_t##T, q4k, 2560, T, true) \
LANES_KERNEL(orig_q4k_c2560_t##T, q4k, 2560, T, false) \
LANES_KERNEL(lanes_q51_c640_t##T, q51, 640, T, true) \
LANES_KERNEL(orig_q51_c640_t##T, q51, 640, T, false)
LANES_FAMILY(1)
LANES_FAMILY(2)
LANES_FAMILY(4)
LANES_FAMILY(8)
#undef LANES_FAMILY
#undef LANES_KERNEL

// Minimal layout probes with the production routed interface (indices, routing), runtime column count and the production block loop. Compared with graph_block_routed_gemv / graph_routed_gemv, only byte locations change (and, for the Q5_1 original-layout control, block-level header reads).
inline void q4k_block(device const uchar* header, uint b, uint word, device const float* input, ushort lane, thread float& accumulator) {
    const float d = float(*reinterpret_cast<device const half*>(header));
    const float dmin = float(*reinterpret_cast<device const half*>(header + 2));
    const uchar4 scales = *reinterpret_cast<device const uchar4*>(header + 4);
    const uchar4 minima = *reinterpret_cast<device const uchar4*>(header + 8);
    const uchar4 extra = *reinterpret_cast<device const uchar4*>(header + 12);
    for (uint pair = 0; pair < 4; ++pair) {
        const uint bits = (word >> (8 * pair)) & 255;
        for (uint nibble = 0; nibble < 2; ++nibble) {
            const uint group = pair * 2 + nibble;
            const uint scale = q4k_scale(scales, extra, group);
            const uint minimum = q4k_minimum(minima, extra, group);
            const float q = float((bits >> (nibble * 4)) & 15);
            const float w = (d * float(scale)) * q - dmin * float(minimum);
            accumulator += w * input[b * 256 + group * 32 + lane];
        }
    }
}

float q4k_lanes_dot(device const uchar* row, device const float* input, uint columns, ushort lane) {
    const uint blocks = columns / 256, full = blocks / 4, rest = blocks % 4;
    device const uchar* quants = row + blocks * 16;
    float accumulator = 0;
    for (uint run = 0; run < full; ++run) {
        const uint4 w = *reinterpret_cast<device const uint4*>(quants + run * 512 + lane * 16);
        const uint b = run * 4;
        q4k_block(row + b * 16, b, w.x, input, lane, accumulator);
        q4k_block(row + (b + 1) * 16, b + 1, w.y, input, lane, accumulator);
        q4k_block(row + (b + 2) * 16, b + 2, w.z, input, lane, accumulator);
        q4k_block(row + (b + 3) * 16, b + 3, w.w, input, lane, accumulator);
    }
    device const uchar* tail = quants + full * 512 + lane * rest * 4;
    for (uint k = 0; k < rest; ++k) {
        const uint b = full * 4 + k;
        q4k_block(row + b * 16, b, *reinterpret_cast<device const uint*>(tail + 4 * k), input, lane, accumulator);
    }
    return accumulator;
}

inline void q51_block(uint b, uint2 h, uint byte, uint shift, device const float* input, ushort lane, thread float& accumulator) {
    const float d = float(as_type<half2>(h.x).x), m = float(as_type<half2>(h.x).y);
    const uint low = (byte >> shift) & 15;
    const uint q = low | (((h.y >> lane) & 1) << 4);
    const float w = d * float(q) + m;
    accumulator += w * input[b * 32 + lane];
}

float q51_orig_dot(device const uchar* row, device const float* input, uint columns, ushort lane) {
    const uint shift = 4 * (lane / 16), j = lane % 16;
    float accumulator = 0;
    for (uint b = 0; b < columns / 32; ++b) {
        device const uchar* block = row + b * 24;
        q51_block(b, *reinterpret_cast<device const uint2*>(block), uint(block[8 + j]), shift, input, lane, accumulator);
    }
    return accumulator;
}

float q51_lanes_dot(device const uchar* row, device const float* input, uint columns, ushort lane) {
    const uint blocks = columns / 32, full = blocks / 16, rest = blocks % 16, shift = 4 * (lane / 16), j = lane % 16;
    device const uint2* headers = reinterpret_cast<device const uint2*>(row);
    device const uchar* quants = row + blocks * 8;
    float accumulator = 0;
    for (uint run = 0; run < full; ++run) {
        const uint4 w = *reinterpret_cast<device const uint4*>(quants + run * 256 + j * 16);
        const uint b = run * 16;
        #pragma unroll
        for (uint k = 0; k < 4; ++k) q51_block(b + k, headers[b + k], (w.x >> (8 * k)) & 255, shift, input, lane, accumulator);
        #pragma unroll
        for (uint k = 0; k < 4; ++k) q51_block(b + 4 + k, headers[b + 4 + k], (w.y >> (8 * k)) & 255, shift, input, lane, accumulator);
        #pragma unroll
        for (uint k = 0; k < 4; ++k) q51_block(b + 8 + k, headers[b + 8 + k], (w.z >> (8 * k)) & 255, shift, input, lane, accumulator);
        #pragma unroll
        for (uint k = 0; k < 4; ++k) q51_block(b + 12 + k, headers[b + 12 + k], (w.w >> (8 * k)) & 255, shift, input, lane, accumulator);
    }
    // Remaining blocks: whole 32-bit words (row_layout admits rest % 4 == 0 here).
    device const uchar* tail = quants + full * 256 + j * rest;
    for (uint k = 0; k < rest; ++k) {
        const uint word = *reinterpret_cast<device const uint*>(tail + 4 * (k / 4));
        q51_block(full * 16 + k, headers[full * 16 + k], (word >> (8 * (k % 4))) & 255, shift, input, lane, accumulator);
    }
    return accumulator;
}

#define ROUTED_PROBE(NAME, DOT) \
kernel void NAME(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]], \
                 device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], \
                 device const uint* indices [[buffer(4)]], constant uint2& routing [[buffer(5)]], \
                 device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]], \
                 ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) { \
    const uint row = group.x * 4 + subgroup; \
    if (row >= shape.y || group.y >= routing.x) return; \
    const uint expert = indices[group.y]; \
    if (expert >= shape.w) { \
        if (!lane) { atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); output[group.y * shape.y + row] = 0; } \
        return; \
    } \
    device const uchar* data = weights + (ulong(expert) * shape.y + row) * shape.z; \
    device const float* x = input + (routing.y ? 0 : group.y * shape.x); \
    const float total = simd_sum(DOT(data, x, shape.x, lane)); \
    if (!lane) { \
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); \
        output[group.y * shape.y + row] = total; \
    } \
}
ROUTED_PROBE(route_lanes_q4k, q4k_lanes_dot)
ROUTED_PROBE(route_orig_q51, q51_orig_dot)
ROUTED_PROBE(route_lanes_q51, q51_lanes_dot)
#undef ROUTED_PROBE

// Kernel-structure control: R output rows per SIMD group share each input load. Every row keeps its own decode, expression and ascending accumulation, so outputs stay byte-identical; only the x loads are shared.
template<uint R, bool Packed>
void q4k_rows(device const uchar* weights, device const float* input, device float* output, constant uint4& shape,
              device const uint* indices, constant uint2& routing, device atomic_uint* error,
              uint2 group, ushort subgroup, ushort lane) {
    const uint row0 = (group.x * 4 + subgroup) * R;
    if (row0 >= shape.y || group.y >= routing.x) return;
    const uint expert = indices[group.y];
    if (expert >= shape.w) {
        if (!lane) { atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); for (uint r = 0; r < R && row0 + r < shape.y; ++r) output[group.y * shape.y + row0 + r] = 0; }
        return;
    }
    const uint rows = min(R, shape.y - row0), blocks = shape.x / 256;
    device const uchar* data = weights + (ulong(expert) * shape.y + row0) * shape.z;
    device const float* x = input + (routing.y ? 0 : group.y * shape.x);
    float sums[R];
    for (uint r = 0; r < R; ++r) sums[r] = 0;
    for (uint b = 0; b < blocks; ++b) {
        float d[R], dmin[R]; uchar4 scales[R], minima[R], extra[R]; uint bytes[R];
        for (uint r = 0; r < R; ++r) {
            if (r >= rows) continue;
            device const uchar* row = data + r * shape.z;
            device const uchar* header = Packed ? row + b * 16 : row + b * 144;
            d[r] = float(*reinterpret_cast<device const half*>(header));
            dmin[r] = float(*reinterpret_cast<device const half*>(header + 2));
            scales[r] = *reinterpret_cast<device const uchar4*>(header + 4);
            minima[r] = *reinterpret_cast<device const uchar4*>(header + 8);
            extra[r] = *reinterpret_cast<device const uchar4*>(header + 12);
            if (Packed) {
                const uint run = b / 4, count = min(4u, blocks - run * 4);
                bytes[r] = *reinterpret_cast<device const uint*>(row + blocks * 16 + run * 512 + lane * count * 4 + (b % 4) * 4);
            } else {
                bytes[r] = uint(header[16 + lane]) | (uint(header[48 + lane]) << 8) | (uint(header[80 + lane]) << 16) | (uint(header[112 + lane]) << 24);
            }
        }
        for (uint pair = 0; pair < 4; ++pair)
            for (uint nibble = 0; nibble < 2; ++nibble) {
                const uint group_index = pair * 2 + nibble;
                const float value = x[b * 256 + group_index * 32 + lane];
                for (uint r = 0; r < R; ++r) {
                    if (r >= rows) continue;
                    const uint scale = q4k_scale(scales[r], extra[r], group_index);
                    const uint minimum = q4k_minimum(minima[r], extra[r], group_index);
                    const float q = float((bytes[r] >> (8 * pair + nibble * 4)) & 15);
                    const float w = (d[r] * float(scale)) * q - dmin[r] * float(minimum);
                    sums[r] += w * value;
                }
            }
    }
    for (uint r = 0; r < R; ++r) {
        if (r >= rows) continue;
        const float total = simd_sum(sums[r]);
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[group.y * shape.y + row0 + r] = total;
        }
    }
}

template<uint R, bool Packed>
void q51_rows(device const uchar* weights, device const float* input, device float* output, constant uint4& shape,
              device const uint* indices, constant uint2& routing, device atomic_uint* error,
              uint2 group, ushort subgroup, ushort lane) {
    const uint row0 = (group.x * 4 + subgroup) * R;
    if (row0 >= shape.y || group.y >= routing.x) return;
    const uint expert = indices[group.y];
    if (expert >= shape.w) {
        if (!lane) { atomic_fetch_or_explicit(error, 4u, memory_order_relaxed); for (uint r = 0; r < R && row0 + r < shape.y; ++r) output[group.y * shape.y + row0 + r] = 0; }
        return;
    }
    const uint rows = min(R, shape.y - row0), blocks = shape.x / 32, shift = 4 * (lane / 16), j = lane % 16;
    device const uchar* data = weights + (ulong(expert) * shape.y + row0) * shape.z;
    device const float* x = input + (routing.y ? 0 : group.y * shape.x);
    float sums[R];
    for (uint r = 0; r < R; ++r) sums[r] = 0;
    for (uint run = 0; run * 16 < blocks; ++run) {
        const uint count = min(16u, blocks - run * 16);
        uint4 words[R];
        for (uint r = 0; r < R; ++r) {
            if (r >= rows || !Packed) continue;
            device const uchar* q = data + r * shape.z + blocks * 8 + run * 256 + j * count;
            if (count == 16) words[r] = *reinterpret_cast<device const uint4*>(q);
            else { words[r] = uint4(0); for (uint k = 0; k < count / 4; ++k) words[r][k] = *reinterpret_cast<device const uint*>(q + 4 * k); }
        }
        for (uint k = 0; k < count; ++k) {
            const uint b = run * 16 + k;
            const float value = x[b * 32 + lane];
            for (uint r = 0; r < R; ++r) {
                if (r >= rows) continue;
                device const uchar* row = data + r * shape.z;
                const uint2 h = Packed ? *reinterpret_cast<device const uint2*>(row + b * 8) : *reinterpret_cast<device const uint2*>(row + b * 24);
                const uint byte = Packed ? (words[r][k / 4] >> (8 * (k % 4))) & 255 : uint(row[b * 24 + 8 + j]);
                const float dd = float(as_type<half2>(h.x).x), m = float(as_type<half2>(h.x).y);
                const uint low = (byte >> shift) & 15;
                const uint q = low | (((h.y >> lane) & 1) << 4);
                const float w = dd * float(q) + m;
                sums[r] += w * value;
            }
        }
    }
    for (uint r = 0; r < R; ++r) {
        if (r >= rows) continue;
        const float total = simd_sum(sums[r]);
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[group.y * shape.y + row0 + r] = total;
        }
    }
}

#define ROWS_PROBE(NAME, FN, R, PACKED) \
kernel void NAME(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]], \
                 device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], \
                 device const uint* indices [[buffer(4)]], constant uint2& routing [[buffer(5)]], \
                 device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]], \
                 ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) { \
    FN<R, PACKED>(weights, input, output, shape, indices, routing, error, group, subgroup, lane); \
}
ROWS_PROBE(route_orig_q4k_r2, q4k_rows, 2, false)
ROWS_PROBE(route_orig_q4k_r4, q4k_rows, 4, false)
ROWS_PROBE(route_lanes_q4k_r4, q4k_rows, 4, true)
ROWS_PROBE(route_orig_q51_r4, q51_rows, 4, false)
ROWS_PROBE(route_lanes_q51_r2, q51_rows, 2, true)
ROWS_PROBE(route_lanes_q51_r4, q51_rows, 4, true)
#undef ROWS_PROBE

// Q8_0 dense rows in the lane layout: scales first, then sixteen-block lane runs of signed bytes. Same expression and order as block_dot's Q8 path.
inline void q8_block(uint b, float scale, uint byte, device const float* input, ushort lane, thread float& accumulator) {
    const float q = float(as_type<char>(uchar(byte)));
    accumulator += (scale * q) * input[b * 32 + lane];
}

float q8_lanes_dot(device const uchar* row, device const float* input, uint columns, ushort lane) {
    const uint blocks = columns / 32, full = blocks / 16, rest = blocks % 16;
    device const half* scales = reinterpret_cast<device const half*>(row);
    device const uchar* quants = row + blocks * 2;
    float accumulator = 0;
    for (uint run = 0; run < full; ++run) {
        const uint4 w = *reinterpret_cast<device const uint4*>(quants + run * 512 + lane * 16);
        const uint b = run * 16;
        #pragma unroll
        for (uint k = 0; k < 4; ++k) q8_block(b + k, float(scales[b + k]), (w.x >> (8 * k)) & 255, input, lane, accumulator);
        #pragma unroll
        for (uint k = 0; k < 4; ++k) q8_block(b + 4 + k, float(scales[b + 4 + k]), (w.y >> (8 * k)) & 255, input, lane, accumulator);
        #pragma unroll
        for (uint k = 0; k < 4; ++k) q8_block(b + 8 + k, float(scales[b + 8 + k]), (w.z >> (8 * k)) & 255, input, lane, accumulator);
        #pragma unroll
        for (uint k = 0; k < 4; ++k) q8_block(b + 12 + k, float(scales[b + 12 + k]), (w.w >> (8 * k)) & 255, input, lane, accumulator);
    }
    // Short final run: widest aligned loads the run width allows.
    device const uchar* tail = quants + full * 512 + lane * rest;
    const uint b = full * 16;
    if (rest % 4 == 0) {
        for (uint k = 0; k < rest; k += 4) {
            const uint w = *reinterpret_cast<device const uint*>(tail + k);
            for (uint i = 0; i < 4; ++i) q8_block(b + k + i, float(scales[b + k + i]), (w >> (8 * i)) & 255, input, lane, accumulator);
        }
    } else if (rest % 2 == 0) {
        for (uint k = 0; k < rest; k += 2) {
            const uint w = *reinterpret_cast<device const ushort*>(tail + k);
            for (uint i = 0; i < 2; ++i) q8_block(b + k + i, float(scales[b + k + i]), (w >> (8 * i)) & 255, input, lane, accumulator);
        }
    } else {
        for (uint k = 0; k < rest; ++k) q8_block(b + k, float(scales[b + k]), uint(tail[k]), input, lane, accumulator);
    }
    return accumulator;
}

kernel void dense_lanes_q8(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
                           device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]],
                           device atomic_uint* error [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
                           ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + subgroup;
    if (row >= shape.y) return;
    const float total = simd_sum(q8_lanes_dot(weights + ulong(row) * shape.z, input, shape.x, lane));
    if (!lane) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[row] = total;
    }
}

#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
// Each input keeps the scalar lane/block arithmetic. Only the unchanged weight bytes are shared across independent inputs.
template<uint T>
inline void lane_q8_many_block(uint block, float scale, uint byte, device const float* input,
                               uint columns, ushort lane, thread float (&sums)[T]) {
    const float value = scale * float(as_type<char>(uchar(byte)));
    for (uint i = 0; i < T; ++i) sums[i] += value * input[ulong(i) * columns + block * 32 + lane];
}
template<uint T>
void lane_q8_many(device const uchar* weights, device const float* input, device float* output,
                  constant uint4& shape, device atomic_uint* error, uint group, ushort subgroup, ushort lane) {
    const uint row = group * 4 + subgroup;
    if (row >= shape.y) return;
    if (!shape.x || shape.x > 65536 || shape.x % 32 || shape.z != shape.x / 32 * 34 || shape.w != T ||
        (shape.x >= 512 && shape.x % 256)) {
        if (!lane) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    if (T == 1) {
        const float total = simd_sum(q8_lanes_dot(weights + ulong(row) * shape.z, input, shape.x, lane));
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[row] = total;
        }
        return;
    }
    device const uchar* data = weights + ulong(row) * shape.z;
    const uint blocks = shape.x / 32, full = blocks / 16, rest = blocks % 16;
    device const half* scales = reinterpret_cast<device const half*>(data);
    device const uchar* quants = data + blocks * 2;
    float sums[T]; for (uint i = 0; i < T; ++i) sums[i] = 0;
    for (uint run = 0; run < full; ++run) {
        const uint4 words = *reinterpret_cast<device const uint4*>(quants + run * 512 + lane * 16);
        for (uint part = 0; part < 4; ++part) {
            for (uint k = 0; k < 4; ++k) {
                const uint block = run * 16 + part * 4 + k;
                lane_q8_many_block<T>(block, float(scales[block]), (words[part] >> (8 * k)) & 255, input, shape.x, lane, sums);
            }
        }
    }
    device const uchar* tail = quants + full * 512 + lane * rest;
    for (uint k = 0; k < rest; ++k) {
        const uint block = full * 16 + k;
        lane_q8_many_block<T>(block, float(scales[block]), uint(tail[k]), input, shape.x, lane, sums);
    }
    for (uint i = 0; i < T; ++i) {
        const float total = simd_sum(sums[i]);
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[ulong(i) * shape.y + row] = total;
        }
    }
}
#define LANE_Q8_MANY(T) \
kernel void graph_lanes_q8_many_##T(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]], \
    device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]], device atomic_uint* error [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) { \
    lane_q8_many<T>(weights, input, output, shape, error, group, subgroup, lane); \
}
LANE_Q8_MANY(1)
LANE_Q8_MANY(2)
LANE_Q8_MANY(4)
LANE_Q8_MANY(8)
#undef LANE_Q8_MANY
#endif
