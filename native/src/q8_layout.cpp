#include "tokspur/q8_layout.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <bit>
#include <limits>

namespace tokspur {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw FormatError(message);
}

struct Geometry {
    uint64_t elements;
    uint64_t blocks;
    uint64_t encoded_bytes;
};

Geometry geometry(uint32_t columns, uint32_t rows) {
    const auto& layout = encoding(8);
    require(columns && rows && columns % layout.elements_per_block == 0,
            "invalid Q8 layout matrix geometry");
    const auto elements = checked_mul(columns, rows);
    const auto blocks = checked_mul(columns / layout.elements_per_block, rows);
    return {elements, blocks, checked_mul(blocks, layout.bytes_per_block)};
}

template<class T> size_t count(uint64_t elements) {
    require(elements <= std::numeric_limits<size_t>::max() &&
            elements <= std::vector<T>{}.max_size(), "Q8 layout vector size exceeds host limits");
    return static_cast<size_t>(elements);
}

void budget(uint64_t bytes, uint64_t maximum) {
    require(bytes <= maximum, "Q8 layout output exceeds explicit byte bound");
}

uint16_t scale_bits(std::span<const std::byte> bytes, size_t offset) {
    return uint16_t(std::to_integer<uint8_t>(bytes[offset])) |
           (uint16_t(std::to_integer<uint8_t>(bytes[offset + 1])) << 8);
}

void finite(uint16_t bits) {
    require((bits & 0x7c00) != 0x7c00, "Q8 layout source scale is nonfinite");
}

void validate(std::span<const std::byte> bytes, const Geometry& shape) {
    require(bytes.size() == shape.encoded_bytes, "Q8 layout source byte count differs from geometry");
    for (size_t offset = 0; offset < bytes.size(); offset += 34) finite(scale_bits(bytes, offset));
}
} // namespace

Q8SoA q8_soa(std::span<const std::byte> bytes, uint32_t columns, uint32_t rows,
             uint64_t max_output_bytes) {
    const auto shape = geometry(columns, rows);
    budget(shape.encoded_bytes, max_output_bytes);
    const auto blocks = count<uint16_t>(shape.blocks), elements = count<int8_t>(shape.elements);
    validate(bytes, shape);
    Q8SoA output{columns, rows, std::vector<uint16_t>(blocks), std::vector<int8_t>(elements)};
    for (size_t block = 0; block < blocks; ++block) {
        output.scales[block] = scale_bits(bytes, block * 34);
        for (size_t lane = 0; lane < 32; ++lane)
            output.values[block * 32 + lane] = std::bit_cast<int8_t>(std::to_integer<uint8_t>(bytes[block * 34 + 2 + lane]));
    }
    return output;
}

std::vector<std::byte> q8_reconstruct(const Q8SoA& matrix, uint64_t max_output_bytes) {
    const auto shape = geometry(matrix.columns, matrix.rows);
    budget(shape.encoded_bytes, max_output_bytes);
    const auto bytes = count<std::byte>(shape.encoded_bytes);
    require(matrix.scales.size() == shape.blocks && matrix.values.size() == shape.elements,
            "Q8 SoA payload differs from matrix geometry");
    for (const auto scale : matrix.scales) finite(scale);
    std::vector<std::byte> output(bytes);
    for (size_t block = 0; block < matrix.scales.size(); ++block) {
        const auto scale = matrix.scales[block];
        output[block * 34] = std::byte(scale & 255);
        output[block * 34 + 1] = std::byte(scale >> 8);
        for (size_t lane = 0; lane < 32; ++lane)
            output[block * 34 + 2 + lane] = std::byte(std::bit_cast<uint8_t>(matrix.values[block * 32 + lane]));
    }
    return output;
}

std::vector<float> q8_expand_f32(std::span<const std::byte> bytes, uint32_t columns,
                                uint32_t rows, uint64_t max_output_bytes) {
    const auto shape = geometry(columns, rows);
    budget(checked_mul(shape.elements, sizeof(float)), max_output_bytes);
    const auto elements = count<float>(shape.elements);
    validate(bytes, shape);
    std::vector<float> output(elements);
    for (size_t block = 0; block < elements / 32; ++block) {
        const float scale = binary16(scale_bits(bytes, block * 34));
        for (size_t lane = 0; lane < 32; ++lane) {
            const auto value = std::bit_cast<int8_t>(std::to_integer<uint8_t>(bytes[block * 34 + 2 + lane]));
            output[block * 32 + lane] = scale * float(value);
        }
    }
    return output;
}

} // namespace tokspur
