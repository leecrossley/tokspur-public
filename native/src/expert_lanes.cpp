#include "tokspur/expert_lanes.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tokspur::lanes {
namespace {
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }

// Q4_K: 16 header bytes (d, dmin, 12 packed 6-bit scale/minimum bytes) and 128 quant bytes per 256 elements. SIMD lane L reads quant bytes 32p+L, p = 0..3, for every block; four consecutive blocks form one 16-byte lane run.
constexpr uint32_t q4k_block = 144, q4k_header = 16, q4k_elements = 256, q4k_run = 4;
// Q5_1: 8 header bytes (d, m, 32 high bits) and 16 nibble bytes per 32 elements. Lanes L and L+16 share quant byte L%16; sixteen blocks form a run.
constexpr uint32_t q51_block = 24, q51_header = 8, q51_elements = 32, q51_run = 16;
// Q8_0: one half scale and 32 signed bytes per 32 elements; lane L reads byte L of every block. Scales form the metadata run; sixteen blocks form a lane run.
constexpr uint32_t q8_block = 34, q8_header = 2, q8_elements = 32, q8_run = 16;

uint16_t u16(const std::byte* p) { return uint16_t(uint8_t(p[0])) | uint16_t(uint8_t(p[1])) << 8; }
uint32_t u32(const std::byte* p) { return uint32_t(u16(p)) | uint32_t(u16(p + 2)) << 16; }

double half_value(const std::byte* p) {
    const uint16_t h = u16(p);
    const int sign = h >> 15, exponent = (h >> 10) & 31, mantissa = h & 1023;
    require(exponent != 31, "nonfinite half in quant block");
    const double magnitude = exponent ? std::ldexp(1024.0 + mantissa, exponent - 25) : std::ldexp(double(mantissa), -24);
    return sign ? -magnitude : magnitude;
}

// Represented value from header bytes and the element's quant bits. Both layouts feed identical fields to this one function, so equal results show equal represented values without relying on GPU arithmetic.
double q4k_value(const std::byte* header, uint32_t i, uint32_t quant) {
    const auto byte = [&](uint32_t k) { return uint32_t(uint8_t(header[k])); };
    const uint32_t group = i / 32;
    uint32_t scale, minimum;
    if (group < 4) { scale = byte(4 + group) & 63; minimum = byte(8 + group) & 63; }
    else {
        const uint32_t slot = group - 4;
        scale = (byte(12 + slot) & 15) | ((byte(4 + slot) >> 6) << 4);
        minimum = (byte(12 + slot) >> 4) | ((byte(8 + slot) >> 6) << 4);
    }
    return half_value(header) * double(scale) * double(quant) - half_value(header + 2) * double(minimum);
}

double q51_value(const std::byte* header, uint32_t i, uint32_t low) {
    const uint32_t q = low | (((u32(header + 4) >> i) & 1) << 4);
    return half_value(header) * double(q) + half_value(header + 2);
}

void check_rows(const RowLayout& layout, std::span<const std::byte> a, std::span<const std::byte> b) {
    const auto expected = row_layout(layout.type, layout.columns);
    require(layout.blocks == expected.blocks && layout.meta_bytes == expected.meta_bytes && layout.row_bytes == expected.row_bytes,
        "lane layout derived fields differ from canonical geometry");
    require(!a.empty() && a.size() == b.size() && a.size() % layout.row_bytes == 0, "lane layout spans must hold equal whole rows");
    const auto first = reinterpret_cast<uintptr_t>(a.data()), second = reinterpret_cast<uintptr_t>(b.data());
    require(first <= UINTPTR_MAX - a.size() && second <= UINTPTR_MAX - b.size(), "lane layout pointer range overflow");
    require(first + a.size() <= second || second + b.size() <= first, "lane layout input and output overlap");
}
} // namespace

RowLayout row_layout(uint32_t type, uint32_t columns) {
    RowLayout r{type, columns, 0, 0, 0};
    if (type == 12) {
        require(columns && columns % q4k_elements == 0 && columns <= 65536, "Q4_K lane layout needs whole 256-element blocks");
        r.blocks = columns / q4k_elements; r.meta_bytes = r.blocks * q4k_header; r.row_bytes = uint64_t(r.blocks) * q4k_block;
    } else if (type == 7) {
        require(columns && columns % q51_elements == 0 && columns <= 65536, "Q5_1 lane layout needs whole 32-element blocks");
        r.blocks = columns / q51_elements; r.meta_bytes = r.blocks * q51_header; r.row_bytes = uint64_t(r.blocks) * q51_block;
    } else if (type == 8) {
        require(columns && columns % q8_elements == 0 && columns <= 65536, "Q8_0 lane layout needs whole 32-element blocks");
        r.blocks = columns / q8_elements; r.meta_bytes = r.blocks * q8_header; r.row_bytes = uint64_t(r.blocks) * q8_block;
    } else throw FormatError("lane layout admits only Q4_K, Q5_1 and Q8_0");
    return r;
}

uint64_t packed_quant_offset(const RowLayout& layout, uint32_t block, uint32_t index) {
    require(block < layout.blocks, "lane layout block bound");
    if (layout.type == 12) {
        require(index < 128, "Q4_K quant byte bound");
        const uint32_t run = block / q4k_run, width = std::min(q4k_run, layout.blocks - run * q4k_run) * 4;
        const uint32_t pair = index / 32, lane = index % 32;
        return layout.meta_bytes + uint64_t(run) * 512 + lane * width + (block % q4k_run) * 4 + pair;
    }
    if (layout.type == 8) {
        require(index < 32, "Q8_0 quant byte bound");
        const uint32_t run = block / q8_run, width = std::min(q8_run, layout.blocks - run * q8_run);
        return layout.meta_bytes + uint64_t(run) * 512 + index * width + block % q8_run;
    }
    require(index < 16, "Q5_1 quant byte bound");
    const uint32_t run = block / q51_run, width = std::min(q51_run, layout.blocks - run * q51_run);
    return layout.meta_bytes + uint64_t(run) * 256 + index * width + block % q51_run;
}

void pack_rows(const RowLayout& layout, std::span<const std::byte> original, std::span<std::byte> packed) {
    check_rows(layout, original, packed);
    const uint32_t block_bytes = layout.type == 12 ? q4k_block : layout.type == 8 ? q8_block : q51_block;
    const uint32_t header = layout.type == 12 ? q4k_header : layout.type == 8 ? q8_header : q51_header;
    for (size_t row = 0; row < original.size(); row += layout.row_bytes) {
        const std::byte* in = original.data() + row; std::byte* out = packed.data() + row;
        for (uint32_t b = 0; b < layout.blocks; ++b) {
            const std::byte* block = in + uint64_t(b) * block_bytes;
            std::memcpy(out + uint64_t(b) * header, block, header);
            for (uint32_t i = 0; i < block_bytes - header; ++i) out[packed_quant_offset(layout, b, i)] = block[header + i];
        }
    }
}

void unpack_rows(const RowLayout& layout, std::span<const std::byte> packed, std::span<std::byte> original) {
    check_rows(layout, packed, original);
    const uint32_t block_bytes = layout.type == 12 ? q4k_block : layout.type == 8 ? q8_block : q51_block;
    const uint32_t header = layout.type == 12 ? q4k_header : layout.type == 8 ? q8_header : q51_header;
    for (size_t row = 0; row < packed.size(); row += layout.row_bytes) {
        const std::byte* in = packed.data() + row; std::byte* out = original.data() + row;
        for (uint32_t b = 0; b < layout.blocks; ++b) {
            std::byte* block = out + uint64_t(b) * block_bytes;
            std::memcpy(block, in + uint64_t(b) * header, header);
            for (uint32_t i = 0; i < block_bytes - header; ++i) block[header + i] = in[packed_quant_offset(layout, b, i)];
        }
    }
}

float decode_original(const RowLayout& layout, const std::byte* row, uint32_t column) {
    require(column < layout.columns, "lane decode column bound");
    if (layout.type == 8) {
        const std::byte* block = row + uint64_t(column / q8_elements) * q8_block;
        return float(half_value(block) * double(int8_t(uint8_t(block[q8_header + column % q8_elements]))));
    }
    if (layout.type == 12) {
        const std::byte* block = row + uint64_t(column / q4k_elements) * q4k_block; const uint32_t i = column % q4k_elements;
        const uint32_t q = (uint32_t(uint8_t(block[q4k_header + (i / 64) * 32 + i % 32])) >> (4 * ((i % 64) / 32))) & 15;
        return float(q4k_value(block, i, q));
    }
    const std::byte* block = row + uint64_t(column / q51_elements) * q51_block; const uint32_t i = column % q51_elements;
    const uint32_t low = (uint32_t(uint8_t(block[q51_header + i % 16])) >> (4 * (i / 16))) & 15;
    return float(q51_value(block, i, low));
}

float decode_packed(const RowLayout& layout, const std::byte* row, uint32_t column) {
    require(column < layout.columns, "lane decode column bound");
    if (layout.type == 8) {
        const uint32_t b = column / q8_elements;
        return float(half_value(row + uint64_t(b) * q8_header) * double(int8_t(uint8_t(row[packed_quant_offset(layout, b, column % q8_elements)]))));
    }
    if (layout.type == 12) {
        const uint32_t b = column / q4k_elements, i = column % q4k_elements;
        const uint32_t byte = uint8_t(row[packed_quant_offset(layout, b, (i / 64) * 32 + i % 32)]);
        return float(q4k_value(row + uint64_t(b) * q4k_header, i, (byte >> (4 * ((i % 64) / 32))) & 15));
    }
    const uint32_t b = column / q51_elements, i = column % q51_elements;
    const uint32_t byte = uint8_t(row[packed_quant_offset(layout, b, i % 16)]);
    return float(q51_value(row + uint64_t(b) * q51_header, i, (byte >> (4 * (i / 16))) & 15));
}

} // namespace tokspur::lanes
