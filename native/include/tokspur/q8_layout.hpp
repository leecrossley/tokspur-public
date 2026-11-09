#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tokspur {

// Row-major Q8_0 blocks become row-major raw half-scale bits and row-major
// signed quantized values. No represented weight or scale bit is changed.
struct Q8SoA {
    uint32_t columns = 0;
    uint32_t rows = 0;
    std::vector<uint16_t> scales;
    std::vector<int8_t> values;
};

// Bounds cover the total returned payload bytes, not vector/allocator overhead
// or the caller-owned input. Geometry and all source scales are validated before
// allocating the output; malformed/nonfinite input cannot publish a partial
// result. Callers must retain/charge actual allocation overhead separately.
Q8SoA q8_soa(std::span<const std::byte> bytes, uint32_t columns, uint32_t rows,
             uint64_t max_output_bytes);
std::vector<std::byte> q8_reconstruct(const Q8SoA& matrix, uint64_t max_output_bytes);

// A finite binary16 scale times any int8 value is exactly representable in FP32.
// This is a value-preserving expansion, including signed zero, not a higher-
// precision re-quantization or recovery of information lost before Q8 encoding.
std::vector<float> q8_expand_f32(std::span<const std::byte> bytes, uint32_t columns,
                                uint32_t rows, uint64_t max_output_bytes);

} // namespace tokspur
