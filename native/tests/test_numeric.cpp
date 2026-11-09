#include "tokspur/numeric.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/graph.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

namespace {
size_t checks = 0;
using Bytes = std::vector<std::byte>;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid numerical request was accepted");
}
void put(Bytes& bytes, size_t offset, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) bytes.at(offset + i) = std::byte((value >> (i * 8)) & 255);
}

void profile_clocks() {
    const auto infinity = std::numeric_limits<double>::infinity();
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    for (const auto mismatch : {0.0, std::nextafter(0.03, 0.0), 0.03})
        require(tokspur::profile_calibration_valid(1, mismatch), "admissible profile calibration refused");
    for (const auto mismatch : {std::nextafter(0.03, infinity), 0.5, -0.001, infinity, -infinity, nan})
        require(!tokspur::profile_calibration_valid(1, mismatch), "invalid profile calibration accepted");
    for (const auto tick : {0.0, -1.0, infinity, -infinity, nan})
        require(!tokspur::profile_calibration_valid(tick, 0), "invalid profile tick scale accepted");
    require(!tokspur::GraphProfile{}.timings_valid, "empty profile advertises usable timing");
}

void halves() {
    // Independent mathematical oracle for every binary16 bit pattern.
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const auto actual = tokspur::binary16(static_cast<uint16_t>(bits));
        const int exponent = (bits >> 10) & 31;
        const int fraction = bits & 1023;
        const double sign = bits & 32768 ? -1 : 1;
        if (exponent == 31) {
            require(fraction ? std::isnan(actual) : (std::isinf(actual) && std::signbit(actual) == (sign < 0)), "binary16 nonfinite conversion");
        } else {
            const double expected = sign * (exponent ? std::ldexp(1 + fraction / 1024.0, exponent - 15) : std::ldexp(fraction, -24));
            require(actual == expected && std::signbit(actual) == (sign < 0), "binary16 finite conversion");
        }
    }
}

void formats() {
    // The fixture encoder uses logical values and bit ownership; it does not
    // call the production decoder or its scale-unpacking helper.
    for (const uint32_t type : {7, 8, 12, 13, 14, 20}) {
        const auto& layout = tokspur::encoding(type);
        Bytes block(layout.bytes_per_block);
        std::vector<float> expected(layout.elements_per_block);
        if (type == 7) {
            put(block, 0, 0x3800, 2); // 0.5
            put(block, 2, 0xc000, 2); // -2
            uint32_t high = 0;
            for (uint32_t i = 0; i < 32; ++i) {
                const uint32_t code = (i * 13 + 7) % 32;
                high |= (code / 16) << i;
                block[8 + i % 16] |= std::byte((code % 16) << ((i / 16) * 4));
                expected[i] = 0.5f * code - 2;
            }
            put(block, 4, high, 4);
        } else if (type == 8) {
            put(block, 0, 0x3400, 2); // 0.25
            for (int i = 0; i < 32; ++i) {
                const int code = i * 8 - 128;
                block[2 + i] = std::byte(code & 255);
                expected[i] = 0.25f * code;
            }
        } else if (type == 12 || type == 13) {
            put(block, 0, 0x3000, 2); // 0.125
            put(block, 2, 0x3400, 2); // 0.25
            for (uint32_t group = 0; group < 8; ++group) {
                const uint32_t scale = (group * 11 + 3) % 64;
                const uint32_t minimum = (group * 7 + 5) % 64;
                for (uint32_t bit = 0; bit < 6; ++bit) {
                    const uint32_t scale_byte = group < 4 ? 4 + group : (bit < 4 ? 8 + group : group);
                    const uint32_t min_byte = group < 4 ? 8 + group : (bit < 4 ? 8 + group : 4 + group);
                    const uint32_t scale_bit = group < 4 || bit < 4 ? bit : bit + 2;
                    const uint32_t min_bit = group < 4 ? bit : (bit < 4 ? bit + 4 : bit + 2);
                    block[scale_byte] |= std::byte(((scale >> bit) & 1) << scale_bit);
                    block[min_byte] |= std::byte(((minimum >> bit) & 1) << min_bit);
                }
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    const uint32_t index = group * 32 + lane;
                    const uint32_t code = (lane * 3 + group * 5) % (type == 12 ? 16 : 32);
                    const auto low_offset = type == 12 ? 16 : 48;
                    block[low_offset + (group / 2) * 32 + lane] |= std::byte((code % 16) << ((group % 2) * 4));
                    if (type == 13) block[16 + lane] |= std::byte((code / 16) << group);
                    expected[index] = 0.125f * scale * code - 0.25f * minimum;
                }
            }
        } else if (type == 14) {
            put(block, 208, 0x2800, 2); // 0.03125
            for (uint32_t i = 0; i < 256; ++i) {
                const uint32_t code = (i * 7 + 1) % 64;
                const uint32_t low_byte = (i / 128) * 64 + (i % 64);
                block[low_byte] |= std::byte((code % 16) << ((i % 128 / 64) * 4));
                block[128 + (i / 128) * 32 + i % 32] |= std::byte((code / 16) << ((i % 128 / 32) * 2));
                const int scale = int(i / 16) * 15 - 120;
                block[192 + i / 16] = std::byte(scale & 255);
                expected[i] = 0.03125f * scale * (int(code) - 32);
            }
        } else {
            put(block, 0, 0x3800, 2);
            constexpr std::array<int, 16> alphabet{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
            for (uint32_t i = 0; i < 32; ++i) {
                const auto code = (i * 7) % 16;
                block[2 + i % 16] |= std::byte(code << ((i / 16) * 4));
                expected[i] = 0.5f * alphabet[code];
            }
        }
        std::vector<float> decoded(expected.size());
        tokspur::dequantize(type, block, decoded);
        for (size_t i = 0; i < expected.size(); ++i) require(decoded[i] == expected[i], "quantized fixture value mismatch");
        std::vector<float> x(expected.size());
        double dot = 0;
        for (size_t i = 0; i < x.size(); ++i) { x[i] = (int(i % 13) - 6) * 0.25f; dot += double(expected[i]) * x[i]; }
        const auto actual = tokspur::reference_gemv(type, block, static_cast<uint32_t>(x.size()), 1, x);
        require(actual[0] == dot, "double GEMV fixture mismatch");
        std::vector<double> xd(x.begin(), x.end());
        require(tokspur::reference_gemv_doubleinput(type, block, static_cast<uint32_t>(xd.size()), 1, xd)[0] == dot, "double-input GEMV fixture mismatch");
        block.pop_back();
        rejected([&] { tokspur::dequantize(type, block, decoded); });
    }
    for (uint32_t type : {0, 1, 30}) {
        const auto width = type == 0 ? 4 : 2;
        Bytes bytes(width * 3);
        put(bytes, 0, type == 0 ? 0x3f800000 : (type == 1 ? 0x3c00 : 0x3f80), width);
        put(bytes, width, type == 0 ? 0xc0000000 : (type == 1 ? 0xc000 : 0xc000), width);
        put(bytes, width * 2, type == 0 ? 0x3e800000 : (type == 1 ? 0x3400 : 0x3e80), width);
        std::vector<float> actual(3);
        tokspur::dequantize(type, bytes, actual);
        require(actual == std::vector<float>({1, -2, 0.25}), "floating decoder");
        put(bytes, 0, type == 0 ? 0x7f800000 : (type == 1 ? 0x7c00 : 0x7f80), width);
        rejected([&] { tokspur::dequantize(type, bytes, actual); });
    }
    std::vector<float> empty;
    rejected([&] { tokspur::dequantize(0, {}, empty); });
    rejected([&] { tokspur::dequantize(999, {}, empty); });
    Bytes f32(4);
    rejected([&] { static_cast<void>(tokspur::reference_gemv(0, f32, 1, 0, empty)); });
    const std::array<float, 1> nan{std::numeric_limits<float>::quiet_NaN()};
    rejected([&] { static_cast<void>(tokspur::reference_gemv(0, f32, 1, 1, nan)); });
    const std::array<double, 1> dn{std::numeric_limits<double>::quiet_NaN()};
    rejected([&] { static_cast<void>(tokspur::reference_gemv_doubleinput(0, f32, 1, 1, dn)); });
    put(f32, 0, 0x40000000, 4);
    const std::array<double, 1> precise{1 + 1e-12}, overflowing{std::numeric_limits<double>::max()};
    require(tokspur::reference_gemv_doubleinput(0, f32, 1, 1, precise)[0] == precise[0] * 2, "double input rounded to float");
    rejected([&] { static_cast<void>(tokspur::reference_gemv_doubleinput(0, f32, 1, 1, overflowing)); });
}

void q8_iq4_multiblock() {
    constexpr std::array<uint16_t, 9> scales{0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x7bff, 0xfbff};
    constexpr std::array<int, 16> alphabet{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (const uint32_t type : {8u, 20u}) {
        const size_t block_bytes = type == 8 ? 34 : 18;
        Bytes encoded(scales.size() * block_bytes);
        std::vector<float> expected(scales.size() * 32), decoded(expected.size());
        for (size_t block = 0; block < scales.size(); ++block) {
            const auto bits = scales[block];
            put(encoded, block * block_bytes, bits, 2);
            const int exponent = (bits >> 10) & 31, fraction = bits & 1023;
            const double scale = (bits & 0x8000 ? -1.0 : 1.0) * (exponent ? std::ldexp(1 + fraction / 1024.0, exponent - 15) : std::ldexp(fraction, -24));
            for (size_t i = 0; i < 32; ++i) {
                const int code = int((block * 37 + i * 19) & 255);
                const int value = type == 8 ? (code < 128 ? code : code - 256) : alphabet[code & 15];
                if (type == 8) encoded[block * block_bytes + 2 + i] = std::byte(code);
                else encoded[block * block_bytes + 2 + i % 16] |= std::byte((code & 15) << (4 * (i / 16)));
                expected[block * 32 + i] = float(scale * value);
            }
        }
        tokspur::dequantize(type, encoded, decoded);
        for (size_t i = 0; i < expected.size(); ++i)
            require(std::bit_cast<uint32_t>(decoded[i]) == std::bit_cast<uint32_t>(expected[i]), "multiblock Q8/IQ4 scale or code differs");
        for (const uint16_t bad_scale : {uint16_t(0x7c00), uint16_t(0xfc00), uint16_t(0x7e01)}) {
            put(encoded, (scales.size() - 1) * block_bytes, bad_scale, 2);
            rejected([&] { tokspur::dequantize(type, encoded, decoded); });
        }
    }
}
} // namespace

int main() {
    try {
        halves();
        formats();
        q8_iq4_multiblock();
        profile_clocks();
        std::cout << "Numerical checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Numerical check failed: " << error.what() << '\n';
        return 1;
    }
}
