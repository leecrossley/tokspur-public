// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace raw_reference {
enum class Format : std::uint32_t { q4_k = 12, q5_1 = 7 };

inline unsigned bits(const std::uint8_t* data, unsigned first, unsigned count) {
    unsigned value = 0;
    for (unsigned bit = 0; bit < count; ++bit)
        value |= unsigned((data[(first + bit) / 8] >> ((first + bit) % 8)) & 1) << bit;
    return value;
}

inline double half(const std::uint8_t* data) {
    const unsigned sign = bits(data, 15, 1), exponent = bits(data, 10, 5), fraction = bits(data, 0, 10);
    const double magnitude = exponent == 31 ? (fraction ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity())
        : exponent ? std::ldexp(1.0 + fraction / 1024.0, int(exponent) - 15) : std::ldexp(double(fraction), -24);
    return sign ? -magnitude : magnitude;
}

// Read encoded fields one bit at a time, independently of the packed-product helpers.
inline double weight(Format format, const std::uint8_t* block, unsigned column) {
    if (format == Format::q5_1) {
        if (column >= 32) throw std::invalid_argument("reference Q5 column");
        const unsigned code = bits(block, 64 + (column % 16) * 8 + (column / 16) * 4, 4)
            + 16 * bits(block, 32 + column, 1);
        return half(block) * code + half(block + 2);
    }
    if (format != Format::q4_k || column >= 256) throw std::invalid_argument("reference Q4 column");
    const unsigned group = column / 32;
    unsigned scale = 0, minimum = 0;
    for (unsigned bit = 0; bit < 6; ++bit) {
        const unsigned scale_byte = group < 4 ? 4 + group : (bit < 4 ? 8 + group : group);
        const unsigned minimum_byte = group < 4 ? 8 + group : (bit < 4 ? 8 + group : 4 + group);
        const unsigned scale_bit = group < 4 || bit < 4 ? bit : bit + 2;
        const unsigned minimum_bit = group < 4 ? bit : (bit < 4 ? bit + 4 : bit + 2);
        scale |= bits(block, scale_byte * 8 + scale_bit, 1) << bit;
        minimum |= bits(block, minimum_byte * 8 + minimum_bit, 1) << bit;
    }
    const unsigned code = bits(block, 128 + (column / 64) * 256 + (column % 32) * 8 + ((column / 32) % 2) * 4, 4);
    return half(block) * scale * code - half(block + 2) * minimum;
}

inline double dot(Format format, const std::uint8_t* row, std::span<const float> input) {
    const unsigned columns = format == Format::q4_k ? 256 : 32, bytes = format == Format::q4_k ? 144 : 24;
    if (input.empty() || input.size() % columns) throw std::invalid_argument("reference partial block");
    double sum = 0;
    for (std::size_t column = 0; column < input.size(); ++column)
        sum += double(float(weight(format, row + (column / columns) * bytes, unsigned(column % columns)))) * input[column];
    return sum;
}
}
