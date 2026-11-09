#include "tokspur/numeric.hpp"
#include "tokspur/gguf.hpp"

#include <array>
#include <bit>
#include <cmath>

namespace tokspur {
namespace {

uint8_t byte(std::span<const std::byte> data, size_t offset) {
    return std::to_integer<uint8_t>(data[offset]);
}

uint16_t word(std::span<const std::byte> data, size_t offset) {
    return uint16_t(byte(data, offset)) | (uint16_t(byte(data, offset + 1)) << 8);
}

uint32_t dword(std::span<const std::byte> data, size_t offset) {
    return uint32_t(word(data, offset)) | (uint32_t(word(data, offset + 2)) << 16);
}

// In Q4_K/Q5_K, groups A-D use the low six bits of bytes 0-7;
// groups E-H combine four bits in bytes 8-11 with the high two bits above.
uint32_t group_scale(std::span<const std::byte> block, size_t group, bool minimum) {
    const size_t lane = group % 4;
    if (group < 4) return byte(block, 4 + lane + (minimum ? 4 : 0)) & 63;
    const auto packed = byte(block, 12 + lane);
    const auto upper = byte(block, 4 + lane + (minimum ? 4 : 0)) >> 6;
    return ((packed >> (minimum ? 4 : 0)) & 15) | (upper << 4);
}

double value(uint32_t type, std::span<const std::byte> block, size_t index) {
    switch (type) {
        case 0: return std::bit_cast<float>(dword(block, 0));
        case 1: return binary16(word(block, 0));
        case 30: return std::bit_cast<float>(uint32_t(word(block, 0)) << 16);
        case 7: {
            const uint32_t low = (byte(block, 8 + index % 16) >> (4 * (index / 16))) & 15;
            const uint32_t high = (dword(block, 4) >> index) & 1;
            return double(binary16(word(block, 0))) * (low | (high << 4)) + binary16(word(block, 2));
        }
        case 8: return double(binary16(word(block, 0))) * std::bit_cast<int8_t>(byte(block, 2 + index));
        case 12: case 13: {
            const auto low_offset = type == 12 ? 16 : 48;
            uint32_t quant = (byte(block, low_offset + (index / 64) * 32 + index % 32) >> (4 * ((index % 64) / 32))) & 15;
            if (type == 13) quant |= ((byte(block, 16 + index % 32) >> (index / 32)) & 1) << 4;
            return double(binary16(word(block, 0))) * group_scale(block, index / 32, false) * quant -
                   double(binary16(word(block, 2))) * group_scale(block, index / 32, true);
        }
        case 14: {
            const auto within = index % 128;
            const auto section = index / 128;
            const auto low = (byte(block, section * 64 + within % 64) >> (4 * (within / 64))) & 15;
            const auto high = (byte(block, 128 + section * 32 + within % 32) >> (2 * (within / 32))) & 3;
            const auto scale = std::bit_cast<int8_t>(byte(block, 192 + index / 16));
            return double(binary16(word(block, 208))) * scale * (int(low | (high << 4)) - 32);
        }
        case 20: {
            // This is the encoded format's fixed nonlinear value alphabet, not learned weights.
            constexpr std::array<int, 16> alphabet{-127, -104, -83, -65, -49, -35, -22, -10,
                                                   1, 13, 25, 38, 53, 69, 89, 113};
            const auto code = (byte(block, 2 + index % 16) >> (4 * (index / 16))) & 15;
            return double(binary16(word(block, 0))) * alphabet[code];
        }
        default: throw FormatError("no owned numerical decoder for this encoding");
    }
}

void validate(uint32_t type, size_t bytes, size_t elements) {
    if (!can_decode(type)) throw FormatError("no owned numerical decoder for this encoding");
    const auto& layout = encoding(type);
    if (!elements || elements % layout.elements_per_block ||
        checked_mul(elements / layout.elements_per_block, layout.bytes_per_block) != bytes) {
        throw FormatError("quantized input/output size mismatch");
    }
}

} // namespace

float binary16(uint16_t bits) {
    const uint32_t sign = uint32_t(bits & 0x8000) << 16;
    uint32_t mantissa = bits & 1023;
    const uint32_t exponent = (bits >> 10) & 31;
    if (exponent == 31) return std::bit_cast<float>(sign | 0x7f800000 | (mantissa << 13));
    if (exponent) return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
    if (!mantissa) return std::bit_cast<float>(sign);
    uint32_t normalized_exponent = 113;
    while (!(mantissa & 1024)) { mantissa <<= 1; --normalized_exponent; }
    return std::bit_cast<float>(sign | (normalized_exponent << 23) | ((mantissa & 1023) << 13));
}

bool can_decode(uint32_t type) {
    return type == 0 || type == 1 || type == 7 || type == 8 || type == 12 || type == 13 || type == 14 || type == 20 || type == 30;
}

void dequantize(uint32_t type, std::span<const std::byte> bytes, std::span<float> output) {
    validate(type, bytes.size(), output.size());
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    if (type == 8) {
        const size_t blocks = output.size() / 32;
        const auto* src = reinterpret_cast<const uint8_t*>(bytes.data());
        float* dst = output.data();
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t* bp = src + b * 34;
            const uint16_t d_bits = uint16_t(bp[0]) | (uint16_t(bp[1]) << 8);
            const float scale = binary16(d_bits);
            if ((std::bit_cast<uint32_t>(scale) & 0x7f800000u) == 0x7f800000u) throw FormatError("nonfinite model weight");
            const auto* qs = reinterpret_cast<const int8_t*>(bp + 2);
            float* out_b = dst + b * 32;
            for (size_t i = 0; i < 32; ++i) out_b[i] = scale * static_cast<float>(qs[i]);
        }
        return;
    }
    if (type == 20) {
        constexpr std::array<float, 16> alphabet{-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
                                                 1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f};
        const size_t blocks = output.size() / 32;
        const auto* src = reinterpret_cast<const uint8_t*>(bytes.data());
        float* dst = output.data();
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t* bp = src + b * 18;
            const uint16_t d_bits = uint16_t(bp[0]) | (uint16_t(bp[1]) << 8);
            const float scale = binary16(d_bits);
            if ((std::bit_cast<uint32_t>(scale) & 0x7f800000u) == 0x7f800000u) throw FormatError("nonfinite model weight");
            const uint8_t* qs = bp + 2;
            float* out_b = dst + b * 32;
            for (size_t i = 0; i < 16; ++i) {
                const uint8_t packed = qs[i];
                out_b[i] = scale * alphabet[packed & 15u];
                out_b[i + 16] = scale * alphabet[packed >> 4u];
            }
        }
        return;
    }
#endif
    const auto& layout = encoding(type);
    for (size_t start = 0; start < output.size(); start += layout.elements_per_block) {
        const auto block = bytes.subspan((start / layout.elements_per_block) * layout.bytes_per_block, layout.bytes_per_block);
        for (uint32_t i = 0; i < layout.elements_per_block; ++i) {
            const auto decoded = value(type, block, i);
            if (!std::isfinite(decoded)) throw FormatError("nonfinite model weight");
            const float rounded = static_cast<float>(decoded);
            if (!std::isfinite(rounded)) throw FormatError("model weight exceeds FP32 range");
            output[start + i] = rounded;
        }
    }
}

template<class Scalar>
static std::vector<double> reference_gemv_impl(uint32_t type, std::span<const std::byte> matrix,
                                               uint32_t columns, uint32_t rows, std::span<const Scalar> input) {
    if (!rows || columns != input.size()) throw FormatError("GEMV shape mismatch");
    validate(type, matrix.size(), checked_mul(columns, rows));
    const auto& layout = encoding(type);
    if (columns % layout.elements_per_block) throw FormatError("GEMV row is not block-aligned");
    for (const auto x : input) if (!std::isfinite(x)) throw FormatError("nonfinite GEMV input");
    std::vector<double> output(rows, 0);
    const size_t blocks_per_row = columns / layout.elements_per_block;
    for (uint32_t row = 0; row < rows; ++row) {
        double sum = 0;
        for (size_t group = 0; group < blocks_per_row; ++group) {
            const auto block = matrix.subspan((row * blocks_per_row + group) * layout.bytes_per_block, layout.bytes_per_block);
            for (uint32_t i = 0; i < layout.elements_per_block; ++i) {
                const double weight = value(type, block, i);
                if (!std::isfinite(weight)) throw FormatError("nonfinite model weight");
                sum += weight * input[group * layout.elements_per_block + i];
            }
        }
        if (!std::isfinite(sum)) throw FormatError("nonfinite GEMV reference result");
        output[row] = sum;
    }
    return output;
}

std::vector<double> reference_gemv(uint32_t type, std::span<const std::byte> matrix,
                                  uint32_t columns, uint32_t rows, std::span<const float> input) {
    return reference_gemv_impl(type, matrix, columns, rows, input);
}

std::vector<double> reference_gemv_doubleinput(uint32_t type, std::span<const std::byte> matrix,
                                             uint32_t columns, uint32_t rows, std::span<const double> input) {
    return reference_gemv_impl(type, matrix, columns, rows, input);
}

} // namespace tokspur
