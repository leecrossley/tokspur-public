#include <metal_stdlib>
using namespace metal;

// Each pipeline is specialized for one on-disk type. No external kernel is loaded.
constant uint weight_encoding [[function_constant(0)]];
constant short nonlinear_alphabet[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
float element(device const uchar* row, uint column);

#ifdef TOKSPUR_PRIVATE_NATIVE_PLAIN_LOADS
// Keep column visits, scalar products and simd_sum identical to the ordinary decoder. Graph bindings are four-byte aligned; BF16 row strides need only two.
kernel void graph_native_plain_gemv(device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]], device float* output [[buffer(2)]],
    constant uint4& shape [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + subgroup;
    if (row >= shape.y) return;
    device const uchar* data = weights + ulong(row) * shape.z;
    float sum = 0;
    for (uint column = lane; column < shape.x; column += 32) {
        const float weight = weight_encoding == 0 ? reinterpret_cast<device const float*>(data)[column]
            : as_type<float>(uint(reinterpret_cast<device const ushort*>(data)[column]) << 16);
        sum += weight * input[column];
    }
    const float total = simd_sum(sum);
    if (!lane) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[row] = total;
    }
}
#endif

uint word(device const uchar* data, uint offset) {
    return uint(data[offset]) | (uint(data[offset + 1]) << 8);
}

// Aligned FP32 loads, with exactly the ordinary lane assignment, per-lane
// accumulation order and SIMD reduction. Used by the small-output policy and
// by explicit expanded-layout probes; no new sum association in either mode.
kernel void graph_small_gemv_direct(device const float* weights [[buffer(0)]],
                                    device const float* input [[buffer(1)]],
                                    device float* output [[buffer(2)]],
                                    constant uint4& shape [[buffer(3)]],
                                    device atomic_uint* error [[buffer(7)]],
                                    uint group [[threadgroup_position_in_grid]],
                                    ushort subgroup [[simdgroup_index_in_threadgroup]],
                                    ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + subgroup;
    if (row >= shape.y) return;
    float sum = 0;
    for (uint column = lane; column < shape.x; column += 32)
        sum += weights[ulong(row) * shape.x + column] * input[column];
    const float total = simd_sum(sum);
    if (!lane) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[row] = total;
    }
}

// Small row counts otherwise launch only one or a few groups. Split columns
// over independent groups, then reduce partials in fixed split order. The
// represented FP32 weights are unchanged; sum association is explicitly different.
kernel void graph_small_gemv_partials(device const float* weights [[buffer(0)]],
                                      device const float* input [[buffer(1)]],
                                      device float* partials [[buffer(2)]],
                                      constant uint4& shape [[buffer(3)]],
                                      device atomic_uint* error [[buffer(7)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      ushort subgroup [[simdgroup_index_in_threadgroup]],
                                      ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x * 4 + subgroup;
    if (row >= shape.y) return;
    const uint width = ((shape.x + shape.z * 32 - 1) / (shape.z * 32)) * 32;
    const uint begin = group.y * width, end = min(begin + width, shape.x);
    float sum = 0;
    for (uint column = begin + lane; column < end; column += 32)
        sum += weights[ulong(row) * shape.x + column] * input[column];
    const float total = simd_sum(sum);
    if (!lane) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        partials[row * shape.z + group.y] = total;
    }
}

kernel void graph_small_gemv_reduce(device const float* partials [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    constant uint4& shape [[buffer(2)]],
                                    device atomic_uint* error [[buffer(7)]],
                                    uint row [[thread_position_in_grid]]) {
    if (row >= shape.y) return;
    float sum = 0;
    for (uint i = 0; i < shape.z; ++i) sum += partials[row * shape.z + i];
    if (!isfinite(sum)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    output[row] = sum;
}

// The persistent graph uses the same owned format decoder but checks numerical
// failure through its batch flag, rather than requiring a host read per layer.
kernel void graph_quantized_gemv(device const uchar* weights [[buffer(0)]],
                                 device const float* input [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant uint4& shape [[buffer(3)]],
                                 device atomic_uint* error [[buffer(7)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 ushort subgroup [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + subgroup;
    if (row >= shape.y) return;
    device const uchar* data = weights + ulong(row) * shape.z;
    float accumulator = 0;
    for (uint column = lane; column < shape.x; column += 32) accumulator += element(data, column) * input[column];
    const float total = simd_sum(accumulator);
    if (lane == 0) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[row] = total;
    }
}

float half_value(device const uchar* data, uint offset) {
    return float(as_type<half>(ushort(word(data, offset))));
}

uint dword(device const uchar* data, uint offset) {
    return word(data, offset) | (word(data, offset + 2) << 16);
}

float element(device const uchar* row, uint column) {
    if (weight_encoding == 0) return as_type<float>(dword(row, column * 4));
    if (weight_encoding == 1) return half_value(row, column * 2);
    if (weight_encoding == 30) return as_type<float>(word(row, column * 2) << 16);
    if (weight_encoding == 7) {
        device const uchar* block = row + (column / 32) * 24;
        const uint i = column % 32;
        const uint low = (uint(block[8 + i % 16]) >> (4 * (i / 16))) & 15;
        const uint q = low | (((dword(block, 4) >> i) & 1) << 4);
        return half_value(block, 0) * float(q) + half_value(block, 2);
    }
    if (weight_encoding == 8) {
        device const uchar* block = row + (column / 32) * 34;
        return half_value(block, 0) * float(as_type<char>(block[2 + column % 32]));
    }
    if (weight_encoding == 12 || weight_encoding == 13) {
        const uint block_bytes = weight_encoding == 12 ? 144 : 176;
        device const uchar* block = row + (column / 256) * block_bytes;
        const uint i = column % 256;
        const uint group = i / 32;
        uint scale;
        uint minimum;
        if (group < 4) {
            scale = uint(block[4 + group]) & 63;
            minimum = uint(block[8 + group]) & 63;
        } else {
            const uint slot = group - 4;
            scale = (uint(block[12 + slot]) & 15) | ((uint(block[4 + slot]) >> 6) << 4);
            minimum = (uint(block[12 + slot]) >> 4) | ((uint(block[8 + slot]) >> 6) << 4);
        }
        const uint payload = weight_encoding == 12 ? 16 : 48;
        uint q = (uint(block[payload + (i / 64) * 32 + i % 32]) >> (4 * ((i % 64) / 32))) & 15;
        if (weight_encoding == 13) q |= ((uint(block[16 + i % 32]) >> (i / 32)) & 1) << 4;
        return half_value(block, 0) * float(scale) * float(q) - half_value(block, 2) * float(minimum);
    }
    if (weight_encoding == 14) {
        device const uchar* block = row + (column / 256) * 210;
        const uint i = column % 256;
        const uint local = i % 128;
        const uint low = (uint(block[(i / 128) * 64 + local % 64]) >> (4 * (local / 64))) & 15;
        const uint high = (uint(block[128 + (i / 128) * 32 + local % 32]) >> (2 * (local / 32))) & 3;
        return half_value(block, 208) * float(as_type<char>(block[192 + i / 16])) * float(int(low | (high << 4)) - 32);
    }
    // IQ4_NL's format-defined nonlinear alphabet.
    device const uchar* block = row + (column / 32) * 18;
    const uint i = column % 32;
    const uint code = (uint(block[2 + i % 16]) >> (4 * (i / 16))) & 15;
    return half_value(block, 0) * float(nonlinear_alphabet[code]);
}

// Four independent rows per 128-thread group; accumulation never uses half precision.
kernel void quantized_gemv(device const uchar* weights [[buffer(0)]],
                           device const float* input [[buffer(1)]],
                           device float* output [[buffer(2)]],
                           constant uint4& shape [[buffer(3)]],
                           uint group [[threadgroup_position_in_grid]],
                           ushort simd_group [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + simd_group;
    if (row >= shape.y) return;
    device const uchar* data = weights + ulong(row) * shape.z;
    float accumulator = 0;
    for (uint column = lane; column < shape.x; column += 32) {
        accumulator += element(data, column) * input[column];
    }
    const float total = simd_sum(accumulator);
    if (lane == 0) output[row] = total;
}

// Expert IDs stay on the device. Matrices are [expert,row,column]; inputs are
// either shared across routes or [route,column]. No CPU-selected weight copies.
kernel void graph_routed_gemv(device const uchar* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint4& shape [[buffer(3)]],
                              device const uint* indices [[buffer(4)]],
                              constant uint2& routing [[buffer(5)]],
                              device atomic_uint* error [[buffer(7)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              ushort subgroup [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x * 4 + subgroup;
    if (row >= shape.y || group.y >= routing.x) return;
    const uint expert = indices[group.y];
    if (expert >= shape.w) {
        if (lane == 0) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            output[group.y * shape.y + row] = 0;
        }
        return;
    }
    device const uchar* data = weights + (ulong(expert) * shape.y + row) * shape.z;
    device const float* x = input + (routing.y ? 0 : group.y * shape.x);
    float accumulator = 0;
    for (uint column = lane; column < shape.x; column += 32) accumulator += element(data, column) * x[column];
    const float total = simd_sum(accumulator);
    if (lane == 0) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[group.y * shape.y + row] = total;
    }
}

// Independently authored block traversal of the same on-disk bytes. The input
// lane visits lane, lane+32, ... exactly as in element(); no partial sums or
// reassociation. All half loads are only two-byte aligned (Q8 rows can be 2 mod 4).
float block_dot(device const uchar* row, device const float* input, uint columns, ushort lane) {
    float accumulator = 0;
    if (weight_encoding == 8) {
        for (uint b = 0; b < columns / 32; ++b) {
            device const uchar* block = row + ulong(b) * 34;
            const float scale = float(*reinterpret_cast<device const half*>(block));
            const float q = float(*reinterpret_cast<device const char*>(block + 2 + lane));
            accumulator += (scale * q) * input[b * 32 + lane];
        }
    } else { // Q4_K only; host dispatch admits no other encoding.
        for (uint b = 0; b < columns / 256; ++b) {
            device const uchar* block = row + ulong(b) * 144;
            const float d = float(*reinterpret_cast<device const half*>(block));
            const float dmin = float(*reinterpret_cast<device const half*>(block + 2));
            const uchar4 scales = *reinterpret_cast<device const uchar4*>(block + 4);
            const uchar4 minima = *reinterpret_cast<device const uchar4*>(block + 8);
            const uchar4 extra = *reinterpret_cast<device const uchar4*>(block + 12);
            for (uint pair = 0; pair < 4; ++pair) {
                const uint bits = uint(block[16 + pair * 32 + lane]);
                for (uint nibble = 0; nibble < 2; ++nibble) {
                    const uint group = pair * 2 + nibble;
                    const uint slot = group % 4;
                    const uint scale = group < 4 ? (uint(scales[slot]) & 63)
                        : ((uint(extra[slot]) & 15) | ((uint(scales[slot]) >> 6) << 4));
                    const uint minimum = group < 4 ? (uint(minima[slot]) & 63)
                        : ((uint(extra[slot]) >> 4) | ((uint(minima[slot]) >> 6) << 4));
                    const float q = float((bits >> (nibble * 4)) & 15);
                    const float w = (d * float(scale)) * q - dmin * float(minimum);
                    accumulator += w * input[b * 256 + group * 32 + lane];
                }
            }
        }
    }
    return accumulator;
}

#ifdef TOKSPUR_PRIVATE_GROUPED_ORDERED
// Same decoder, column/lane order, scalar accumulation and SIMD reduction as
// graph_routed_gemv. Only independent input rows are interleaved. Stable route
// compaction and task construction are the existing grouped component's work.
template<uint Tile>
void grouped_ordered(device const uchar* weights, device const float* input,
    device float* output, constant uint4& s, device const uint* indices,
    constant uint4& route, device const uint* token_map, device const uint* tasks,
    device const uint* counts, device atomic_uint* error, uint2 group,
    ushort subgroup, ushort lane) {
    const uint bound = route.y*route.x/Tile+s.w, row = group.x*4+subgroup;
    if (tasks[0] > bound || tasks[1] != route.y*route.x) {
        atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
    }
    if (group.y >= tasks[0] || row >= s.y) return;
    const uint expert = tasks[2+2*group.y], start = tasks[3+2*group.y];
    if (expert >= s.w || start >= route.y || counts[expert] > route.y || start >= counts[expert]) {
        atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
    }
    if constexpr (Tile == 1) {
        // Pack the ordinary independent GEMVs, retaining the already tested
        // block traversal as well as its arithmetic. No input-row arrays.
        const uint rid = token_map[expert*route.y+start];
        if (rid >= route.y*route.x || indices[rid] != expert) {
            atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
        }
        device const uchar* data = weights+(ulong(expert)*s.y+row)*s.z;
        device const float* x = input+ulong(route.w ? rid/route.x : rid)*s.x;
        const float input_bound = 0x1.fffffep127f/(8.0f*float(s.x)*134217728.0f);
        for (uint column = lane; column < s.x; column += 32)
            if (!isfinite(x[column]) || abs(x[column]) > input_bound)
                atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        float sum = 0;
        if (weight_encoding == 8 || weight_encoding == 12) sum = block_dot(data,x,s.x,lane);
        else for (uint column = lane; column < s.x; column += 32) sum += element(data,column)*x[column];
        const float total = simd_sum(sum);
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            output[ulong(rid)*s.y+row] = total;
        }
        return;
    }
    const uint n = min(Tile,counts[expert]-start);
    uint routes[Tile]; float sums[Tile];
    // Fixed indices let the compiler keep these independent accumulators in
    // registers. A runtime array subscript can otherwise spill each update.
    #pragma unroll
    for (uint i = 0; i < Tile; ++i) {
        routes[i] = 0; sums[i] = 0;
        if (i >= n) continue;
        routes[i] = token_map[expert*route.y+start+i];
        if (routes[i] >= route.y*route.x || indices[routes[i]] != expert) {
            atomic_fetch_or_explicit(error,4u,memory_order_relaxed); return;
        }
    }
    device const uchar* data = weights+(ulong(expert)*s.y+row)*s.z;
    const float input_bound = 0x1.fffffep127f/(8.0f*float(s.x)*134217728.0f);
    for (uint column = lane; column < s.x; column += 32) {
        const float w = element(data,column);
        #pragma unroll
        for (uint i = 0; i < Tile; ++i) {
            if (i >= n) continue;
            const uint rid = routes[i];
            const float x = input[ulong(route.w ? rid/route.x : rid)*s.x+column];
            if (!isfinite(x) || abs(x) > input_bound) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            sums[i] += w*x;
        }
    }
    #pragma unroll
    for (uint i = 0; i < Tile; ++i) {
        if (i >= n) continue;
        const float total = simd_sum(sums[i]);
        if (!lane) {
            if (!isfinite(total)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            output[ulong(routes[i])*s.y+row] = total;
        }
    }
}
#define GROUPED_ORDERED(TILE) \
kernel void graph_grouped_ordered##TILE(device const uchar* weights [[buffer(0)]], \
    device const float* input [[buffer(1)]], device float* output [[buffer(2)]], \
    constant uint4& shape [[buffer(3)]], device const uint* indices [[buffer(4)]], \
    constant uint4& route [[buffer(5)]], device const uint* token_map [[buffer(6)]], \
    device atomic_uint* error [[buffer(7)]], device const uint* tasks [[buffer(8)]], \
    device const uint* counts [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]], \
    ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) { \
    grouped_ordered<TILE>(weights,input,output,shape,indices,route,token_map,tasks,counts,error,group,subgroup,lane); \
}
GROUPED_ORDERED(1)
GROUPED_ORDERED(4)
GROUPED_ORDERED(8)
GROUPED_ORDERED(16)
#undef GROUPED_ORDERED
#endif

// A value-preserving alternative layout: one unchanged half scale per32 signed
// bytes, in two separate buffers. No conversion or float reduction on the GPU.
kernel void graph_q8_soa_gemv(device const half* scales [[buffer(0)]],
                              device const char* values [[buffer(1)]],
                              device const float* input [[buffer(2)]],
                              device float* output [[buffer(3)]],
                              constant uint2& shape [[buffer(4)]],
                              device atomic_uint* error [[buffer(7)]],
                              uint group [[threadgroup_position_in_grid]],
                              ushort subgroup [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + subgroup;
    if (row >= shape.y) return;
    const uint blocks = shape.x / 32;
    float accumulator = 0;
    for (uint b = 0; b < blocks; ++b) {
        const float scale = float(scales[ulong(row) * blocks + b]);
        const float q = float(values[ulong(row) * shape.x + b * 32 + lane]);
        accumulator += (scale * q) * input[b * 32 + lane];
    }
    const float total = simd_sum(accumulator);
    if (!lane) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[row] = total;
    }
}

kernel void graph_block_gemv(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint4& shape [[buffer(3)]],
                             device atomic_uint* error [[buffer(7)]],
                             uint group [[threadgroup_position_in_grid]],
                             ushort subgroup [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + subgroup;
    if (row >= shape.y) return;
    const float total = simd_sum(block_dot(weights + ulong(row) * shape.z, input, shape.x, lane));
    if (!lane) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[row] = total;
    }
}

kernel void graph_block_routed_gemv(device const uchar* weights [[buffer(0)]],
                                    device const float* input [[buffer(1)]],
                                    device float* output [[buffer(2)]],
                                    constant uint4& shape [[buffer(3)]],
                                    device const uint* indices [[buffer(4)]],
                                    constant uint2& routing [[buffer(5)]],
                                    device atomic_uint* error [[buffer(7)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    ushort subgroup [[simdgroup_index_in_threadgroup]],
                                    ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x * 4 + subgroup;
    if (row >= shape.y || group.y >= routing.x) return;
    const uint expert = indices[group.y];
    if (expert >= shape.w) {
        if (!lane) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            output[group.y * shape.y + row] = 0;
        }
        return;
    }
    device const uchar* data = weights + (ulong(expert) * shape.y + row) * shape.z;
    device const float* x = input + (routing.y ? 0 : group.y * shape.x);
    const float total = simd_sum(block_dot(data, x, shape.x, lane));
    if (!lane) {
        if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[group.y * shape.y + row] = total;
    }
}
