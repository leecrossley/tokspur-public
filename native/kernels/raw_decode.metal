// Copyright (c) 2023-2026 The ggml authors
// Adapted from llama.cpp 6050b467 mul_mv.metal under the MIT licence in native/third_party/llama-cpp/LICENSE.
// Tokspur owns the buffers, route IDs, dispatches and error flag. Raw GGUF only.
#include <metal_stdlib>
using namespace metal;

struct DecodeArgs {
    uint columns, rows, experts, selected;
    uint row_bytes, broadcast, reserved0, reserved1;
    ulong input_stride, output_stride, expert_stride, matrix_bytes;
};
static_assert(sizeof(DecodeArgs) == 64, "decode argument ABI");

inline bool decode_valid(constant DecodeArgs& a, uint block_columns, uint block_bytes) {
    return a.columns && a.columns <= 65536 && !(a.columns % block_columns) && a.rows && a.rows <= 1000000 &&
        a.experts && a.experts <= 4096 && a.selected && a.selected <= 64 && a.broadcast <= 1 &&
        !a.reserved0 && !a.reserved1 && a.row_bytes == (a.columns / block_columns) * block_bytes &&
        a.expert_stride == ulong(a.row_bytes) * a.rows && a.matrix_bytes == a.expert_stride * a.experts &&
        a.matrix_bytes <= (2ul << 30) && a.input_stride >= ulong(a.columns) * 4 && !(a.input_stride % 4) &&
        a.output_stride >= ulong(a.rows) * 4 && !(a.output_stride % 4);
}

inline void decode_write(float sum, device float* output, device atomic_uint* error, ushort lane) {
    if (!lane) {
        if (!isfinite(sum)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        *output = sum;
    }
}

// The reference dispatch uses NSG=2 and NR0=2 for Q4_K at this pin.
kernel void raw_decode_q4_k(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant DecodeArgs& a [[buffer(3)]], device const uint* indices [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    if (!decode_valid(a, 256, 144)) {
        if (!lane) atomic_fetch_or_explicit(error, 8u, memory_order_relaxed);
        return;
    }
    if (group.y >= a.selected) return;
    const uint first_row = (group.x * 2 + subgroup) * 2;
    if (first_row >= a.rows) return;
    device float* out = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(group.y) * a.output_stride);
    const uint expert = indices[group.y];
    if (expert >= a.experts) {
        if (!lane) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            for (uint row = 0; row < 2 && first_row + row < a.rows; ++row) out[first_row + row] = 0;
        }
        return;
    }
    device const uchar* matrix = weights + ulong(expert) * a.expert_stride;
    device const float* y = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + (a.broadcast ? 0 : ulong(group.y) * a.input_stride));
    const ushort ix = lane / 8, iq = (lane % 8) / 4, ir = lane % 4;
    const uint nb = a.columns / 256;
    float sumf[2] = {0, 0};
    for (uint ib = ix; ib < nb; ib += 4) {
        device const float* y4 = y + ib * 256 + 64 * iq + 8 * ir;
        float yl[16], yh[16];
        float4 sumy = 0;
        for (ushort i = 0; i < 8; ++i) {
            yl[i] = y4[i]; sumy[0] += yl[i]; yl[i + 8] = y4[i + 32]; sumy[1] += yl[i + 8];
            yh[i] = y4[i + 128]; sumy[2] += yh[i]; yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        for (ushort row = 0; row < 2; ++row) {
            // The upstream padded-row assumption is replaced by an in-bounds inactive-row load.
            const uint physical_row = min(first_row + row, a.rows - 1);
            device const uchar* block = matrix + ulong(physical_row) * a.row_bytes + ib * 144;
            device const ushort* sc = reinterpret_cast<device const ushort*>(block + 4) + iq;
            device const ushort* q1 = reinterpret_cast<device const ushort*>(block + 16) + 16 * iq + 4 * ir;
            device const ushort* q2 = q1 + 32;
            const float d = float(*reinterpret_cast<device const half*>(block));
            const float minimum = float(*reinterpret_cast<device const half*>(block + 2));
            ushort sc16[4] = {ushort(sc[0] & 0x3f3f), ushort(sc[2] & 0x3f3f),
                ushort((sc[4] & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2)),
                ushort(((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2))};
            thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);
            float4 acc1 = 0, acc2 = 0;
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i] * (q1[i] & 0x000f); acc1[1] += yl[2*i+1] * (q1[i] & 0x0f00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00f0); acc1[3] += yl[2*i+9] * (q1[i] & 0xf000);
                acc2[0] += yh[2*i] * (q2[i] & 0x000f); acc2[1] += yh[2*i+1] * (q2[i] & 0x0f00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00f0); acc2[3] += yh[2*i+9] * (q2[i] & 0xf000);
            }
            sumf[row] += d * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                minimum * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
        }
    }
    for (ushort row = 0; row < 2 && first_row + row < a.rows; ++row)
        decode_write(simd_sum(sumf[row]), out + first_row + row, error, lane);
}

inline float decode_q5_1_dot(device const uchar* b, float sumy, thread const float* yl, ushort il) {
    const float d = float(*reinterpret_cast<device const half*>(b));
    const float m = float(*reinterpret_cast<device const half*>(b + 2));
    device const ushort* qs = reinterpret_cast<device const ushort*>(b + 8) + il / 2;
    const uint qh = *reinterpret_cast<device const uint*>(b + 4);
    float4 acc = 0;
    for (ushort i = 0; i < 8; i += 2) {
        acc[0] += yl[i] * ((qs[i/2] & 0x000f) | ((qh >> (i+il) << 4) & 0x10));
        acc[1] += yl[i+1] * ((qs[i/2] & 0x0f00) | ((qh >> (i+1+il) << 12) & 0x1000));
        acc[2] += yl[i+8] * ((qs[i/2] & 0x00f0) | ((qh >> (i+il+16) << 8) & 0x100));
        acc[3] += yl[i+9] * ((qs[i/2] & 0xf000) | ((qh >> (i+1+il+16) << 16) & 0x10000));
    }
    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// The reference dispatch uses NSG=2 and NR0=4 for Q5_1 at this pin.
kernel void raw_decode_q5_1(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant DecodeArgs& a [[buffer(3)]], device const uint* indices [[buffer(4)]],
    device atomic_uint* error [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    ushort subgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    if (!decode_valid(a, 32, 24)) {
        if (!lane) atomic_fetch_or_explicit(error, 8u, memory_order_relaxed);
        return;
    }
    if (group.y >= a.selected) return;
    const uint first_row = (group.x * 2 + subgroup) * 4;
    if (first_row >= a.rows) return;
    device float* out = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(output) + ulong(group.y) * a.output_stride);
    const uint expert = indices[group.y];
    if (expert >= a.experts) {
        if (!lane) {
            atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
            for (uint row = 0; row < 4 && first_row + row < a.rows; ++row) out[first_row + row] = 0;
        }
        return;
    }
    device const uchar* matrix = weights + ulong(expert) * a.expert_stride;
    device const float* y = reinterpret_cast<device const float*>(reinterpret_cast<device const uchar*>(input) + (a.broadcast ? 0 : ulong(group.y) * a.input_stride));
    const ushort ix = lane / 2, il = (lane % 2) * 8;
    float sumf[4] = {0, 0, 0, 0};
    for (uint ib = ix; ib < a.columns / 32; ib += 16) {
        device const float* yb = y + ib * 32 + il;
        float yl[16], sumy[2] = {0, 0};
        for (ushort i = 0; i < 8; i += 2) {
            sumy[0] += yb[i] + yb[i+1]; yl[i] = yb[i]; yl[i+1] = yb[i+1]/256.f;
            sumy[1] += yb[i+16] + yb[i+17]; yl[i+8] = yb[i+16]/16.f; yl[i+9] = yb[i+17]/4096.f;
        }
        for (ushort row = 0; row < 4; ++row) {
            const uint physical_row = min(first_row + row, a.rows - 1);
            device const uchar* b = matrix + ulong(physical_row) * a.row_bytes + ib * 24;
            sumf[row] += decode_q5_1_dot(b, sumy[0] + sumy[1], yl, il);
        }
    }
    for (ushort row = 0; row < 4 && first_row + row < a.rows; ++row)
        decode_write(simd_sum(sumf[row]), out + first_row + row, error, lane);
}
