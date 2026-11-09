// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "raw_decode_reference.hpp"
#include <algorithm>
#include <array>

namespace moe_projection_reference {
inline float represented(uint32_t type, const std::byte* row, uint32_t column) {
    const auto* data = reinterpret_cast<const uint8_t*>(row);
    if (type == 7) return float(raw_reference::weight(raw_reference::Format::q5_1, data + (column / 32) * 24, column % 32));
    if (type == 8) {
        data += (column / 32) * 34;
        const unsigned byte = raw_reference::bits(data, 16 + (column % 32) * 8, 8);
        return float(raw_reference::half(data) * (byte < 128 ? int(byte) : int(byte) - 256));
    }
    if (type == 12) return float(raw_reference::weight(raw_reference::Format::q4_k, data + (column / 256) * 144, column % 256));
    data += (column / 256) * 176;
    std::array<uint8_t, 144> q4{};
    std::copy_n(data, 16, q4.begin()); std::copy_n(data + 48, 128, q4.begin() + 16);
    const uint32_t group = (column % 256) / 32;
    const double low = raw_reference::weight(raw_reference::Format::q4_k, q4.data(), column % 256);
    unsigned scale = 0;
    for (unsigned bit = 0; bit < 6; ++bit) {
        const unsigned at = group < 4 ? 4 + group : bit < 4 ? 8 + group : group;
        scale |= raw_reference::bits(data, at * 8 + (group < 4 || bit < 4 ? bit : bit + 2), 1) << bit;
    }
    const unsigned high = raw_reference::bits(data, (16 + column % 32) * 8 + group, 1);
    return float(low + raw_reference::half(data) * scale * 16 * high);
}
}
