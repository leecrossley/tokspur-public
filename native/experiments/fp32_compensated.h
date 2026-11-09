// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <metal_stdlib>

namespace tokspur_fp32 {
using namespace metal;
struct Sum { float high = 0, low = 0; };

// Recover the rounding residual of each addition. Callers disable implicit contraction, while product residuals use explicit fma.
inline void add(thread Sum& sum, float value) {
    const float next = sum.high + value;
    const float part = next - sum.high;
    const float residual = (sum.high - (next - part)) + (value - part);
    const float correction = sum.low + residual;
    sum.high = next + correction;
    const float tail = sum.high - next;
    sum.low = (next - (sum.high - tail)) + (correction - tail);
}
inline void product(thread Sum& sum, float weight, float input) {
    const float product = weight * input;
    const float residual = fma(weight, input, -product);
    add(sum, product);
    add(sum, residual);
}
inline float reduce(Sum sum, ushort lane) {
    for (ushort step = 1; step < 32; step *= 2) {
        const float high = simd_shuffle_down(sum.high, step);
        const float low = simd_shuffle_down(sum.low, step);
        if (lane % (2 * step) == 0) {
            add(sum, high);
            add(sum, low);
        }
    }
    return sum.high + sum.low;
}

}
