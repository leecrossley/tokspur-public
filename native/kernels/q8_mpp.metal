// Extends Tokspur's 24 October cooperative-FP32 probe to Q8_0 blocks and
// token-major batches. Apple TensorOps API; no external engine implementation.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

template<uint Groups, uint Tile, typename Execution>
void q8_mpp_tile(device const uchar* weights, device const float* input,
                 device float* output, constant uint4& s,
                 device atomic_uint* error, uint3 tile, ushort simdgroup) {
    constexpr uint K = 32;
    constexpr auto descriptor = matmul2d_descriptor(Tile, Tile, K, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    static_assert(!descriptor.relaxed_precision);
    matmul2d<descriptor, Execution> operation;
    auto left = operation.template get_left_input_cooperative_tensor<float, float, float>();
    auto right = operation.template get_right_input_cooperative_tensor<float, float, float>();
    static_assert(is_same_v<typename decltype(left)::value_type, float>);
    static_assert(is_same_v<typename decltype(right)::value_type, float>);
    auto result = operation.template get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
    for (auto it = result.begin(); it != result.end(); ++it) *it = 0;
    constexpr uint side = Tile == 32 || Groups == 4 ? 32 : 16;
    const uint row_base = tile.x * side + (Groups == 4 && Tile == 16 ? simdgroup / 2 * 16 : 0);
    const uint batch_base = tile.y * side + (Groups == 4 && Tile == 16 ? simdgroup % 2 * 16 : 0);
    // Conservative declared range guarantees sum(abs(products)) <= FLT_MAX/8
    // even with the largest finite binary16 scale and signed-int8 magnitude.
    const float input_bound = 0x1.fffffep127f / (8.0f * float(s.x) * 8388608.0f);
    for (uint at = 0; at < s.x; at += K) {
        for (auto it = left.begin(); it != left.end(); ++it) {
            *it = 0;
            if (left.is_valid_element(it)) {
                const auto c = left.get_multidimensional_index(it);
                const uint row = row_base + c[1];
                const uint column = at + c[0];
                if (c[0] < int(K) && c[1] < int(Tile) && row < s.y && column < s.x) {
                    device const uchar* block = weights + ulong(row) * s.z + ulong(column / 32) * 34;
                    const float scale = float(*reinterpret_cast<device const half*>(block));
                    const float value = float(*reinterpret_cast<device const char*>(block + 2 + column % 32));
                    if (!isfinite(scale)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                    *it = scale * value;
                }
            }
        }
        for (auto it = right.begin(); it != right.end(); ++it) {
            *it = 0;
            if (right.is_valid_element(it)) {
                const auto c = right.get_multidimensional_index(it);
                const uint batch = batch_base + c[0];
                if (c[0] < int(Tile) && c[1] < int(K) && batch < s.w && at + c[1] < s.x) {
                    const float value = input[ulong(batch) * s.x + at + c[1]];
                    if (!isfinite(value) || abs(value) > input_bound)
                        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                    *it = value;
                }
            }
        }
        operation.run(left, right, result);
    }
    for (auto it = result.begin(); it != result.end(); ++it) {
        if (!result.is_valid_element(it)) continue;
        const auto c = result.get_multidimensional_index(it);
        const uint batch = batch_base + c[0], row = row_base + c[1];
        if (c[0] < int(Tile) && c[1] < int(Tile) && row < s.y && batch < s.w) {
            const float value = *it;
            if (!isfinite(value)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            output[ulong(batch) * s.y + row] = value;
        }
    }
}

#define Q8_MPP_KERNEL(GROUPS, TILE, EXECUTION, NAME) \
kernel void NAME(device const uchar* weights [[buffer(0)]], \
                  device const float* input [[buffer(1)]], \
                  device float* output [[buffer(2)]], \
                  constant uint4& shape [[buffer(3)]], \
                  device atomic_uint* error [[buffer(7)]], \
                  uint3 tile [[threadgroup_position_in_grid]], \
                  ushort simdgroup [[simdgroup_index_in_threadgroup]]) { \
    q8_mpp_tile<GROUPS,TILE,EXECUTION>(weights, input, output, shape, error, tile, simdgroup); \
}
Q8_MPP_KERNEL(1, 16, execution_simdgroup, graph_q8_mpp)
Q8_MPP_KERNEL(4, 16, execution_simdgroup, graph_q8_mpp_quad)
Q8_MPP_KERNEL(1, 32, execution_simdgroup, graph_q8_mpp_wide)
