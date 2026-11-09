#include "tokspur/gguf.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

namespace {
void put(std::span<std::byte> block, size_t offset, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) block[offset + i] = std::byte((value >> (i * 8)) & 255);
}

void fill(uint32_t type, std::span<std::byte> bytes, std::mt19937& random) {
    const auto& layout = tokspur::encoding(type);
    for (size_t at = 0; at < bytes.size(); at += layout.bytes_per_block) {
        auto block = bytes.subspan(at, layout.bytes_per_block);
        for (auto& b : block) b = std::byte(random() & 255);
        if (type == 0) {
            const float weight = (int(random() % 1001) - 500) / 256.0f;
            put(block, 0, std::bit_cast<uint32_t>(weight), 4);
        } else if (type == 30) {
            const float weight = (int(random() % 1001) - 500) / 256.0f;
            put(block, 0, std::bit_cast<uint32_t>(weight) >> 16, 2);
        } else {
            // Finite normal, subnormal, signed and zero scales; arbitrary packed bits.
            const std::array<uint16_t, 5> scales{0, 0x0011, 0x3400, 0xb400, 0x2800};
            put(block, type == 14 ? 208 : 0, scales[random() % scales.size()], 2);
            if (type == 7 || type == 12 || type == 13) put(block, 2, scales[random() % scales.size()], 2);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned metallib path");
        tokspur::MetalGemv gpu(argv[1]);
        std::mt19937 random(0x6d657461);
        size_t cases = 0;
        double worst = 0;
        for (uint32_t type : {0, 1, 7, 8, 12, 13, 14, 20, 30}) {
            const auto& layout = tokspur::encoding(type);
            const auto columns = layout.elements_per_block * 3;
            for (uint32_t rows : {1, 3, 4, 5, 33}) {
                std::vector<std::byte> matrix(rows * 3 * layout.bytes_per_block);
                fill(type, matrix, random);
                for (bool zero : {false, true}) {
                    std::vector<float> input(columns);
                    for (auto& x : input) x = zero ? 0 : (int(random() % 1001) - 500) / 128.0f;
                    const auto expected = tokspur::reference_gemv(type, matrix, columns, rows, input);
                    const auto actual = gpu.run(type, matrix, columns, rows, input, 1);
                    if (actual.output.size() != rows) throw std::runtime_error("Metal output shape mismatch");
                    for (uint32_t row = 0; row < rows; ++row) {
                        const auto scaled = std::abs(actual.output[row] - expected[row]) / std::max(1.0, std::abs(expected[row]));
                        worst = std::max(worst, scaled);
                        if (scaled > 5e-5) throw std::runtime_error("Metal numerical tolerance exceeded");
                        if (zero && actual.output[row] != 0) throw std::runtime_error("nonzero output for zero vector");
                    }
                    ++cases;
                }
            }
        }
        const auto rejected = [&](uint32_t type, std::span<const std::byte> matrix, uint32_t columns,
                                  uint32_t rows, std::span<const float> input, uint32_t repetitions) {
            try { static_cast<void>(gpu.run(type, matrix, columns, rows, input, repetitions)); }
            catch (const tokspur::FormatError&) { ++cases; return; }
            throw std::runtime_error("invalid Metal request was accepted");
        };
        const std::array<float, 1> input{1};
        const std::array<std::byte, 4> data{};
        rejected(999, data, 1, 1, input, 1);
        rejected(0, data, 1, 1, input, 0);
        rejected(0, data, 1, 1, input, 1001);
        rejected(0, data, 1, 0, input, 1);
        rejected(0, data, 2, 1, input, 1);
        rejected(12, data, 1, 1, input, 1);
        const std::array<float, 1> infinity{std::numeric_limits<float>::infinity()};
        rejected(0, data, 1, 1, infinity, 1);
        std::cout << "Metal component checks passed: " << cases << ", max scaled error: " << worst
                  << ", device: " << gpu.device_name() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Metal component check failed: " << error.what() << '\n';
        return 1;
    }
}
