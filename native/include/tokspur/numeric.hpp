#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tokspur {

float binary16(uint16_t bits);
bool can_decode(uint32_t type);
void dequantize(uint32_t type, std::span<const std::byte> bytes, std::span<float> output);

// Deliberately scalar and double-accumulating: this is a numerical reference,
// not the production fast path or an inference-speed comparator.
std::vector<double> reference_gemv(uint32_t type, std::span<const std::byte> matrix,
                                  uint32_t columns, uint32_t rows, std::span<const float> input);
std::vector<double> reference_gemv_doubleinput(uint32_t type, std::span<const std::byte> matrix,
                                             uint32_t columns, uint32_t rows, std::span<const double> input);

} // namespace tokspur
