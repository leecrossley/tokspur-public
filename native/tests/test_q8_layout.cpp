#include "tokspur/q8_layout.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
size_t checks = 0;
uint64_t products = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void reject(const std::function<void()>& operation) {
    ++checks;
    try { operation(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid Q8 layout input accepted");
}
void put16(std::span<std::byte> bytes, size_t offset, uint16_t value) {
    bytes[offset] = std::byte(value & 255); bytes[offset + 1] = std::byte(value >> 8);
}
uint16_t word(std::span<const std::byte> bytes, size_t offset) {
    return uint16_t(std::to_integer<uint8_t>(bytes[offset])) |
           (uint16_t(std::to_integer<uint8_t>(bytes[offset + 1])) << 8);
}
bool same(const tokspur::Q8SoA& a, const tokspur::Q8SoA& b) {
    return a.columns == b.columns && a.rows == b.rows && a.scales == b.scales && a.values == b.values;
}

void conversion(std::span<const std::byte> input, uint32_t columns, uint32_t rows) {
    const std::vector<std::byte> original(input.begin(), input.end());
    const auto elements = uint64_t(columns) * rows;
    const auto soa = tokspur::q8_soa(input, columns, rows, input.size());
    require(soa.columns == columns && soa.rows == rows && soa.values.size() == elements &&
            soa.scales.size() == elements / 32, "SoA geometry differs");
    require(soa.scales.size() * 2 + soa.values.size() == input.size(), "SoA payload budget differs");
    const auto before = soa;
    require(tokspur::q8_reconstruct(soa, input.size()) == original, "Q8 inverse lost encoded bytes");
    require(same(soa, before), "inverse changed SoA input");
    const auto expanded = tokspur::q8_expand_f32(input, columns, rows, elements * 4);
    std::vector<float> ordinary(elements);
    tokspur::dequantize(8, input, ordinary);
    require(expanded.size() == elements, "FP32 expansion geometry differs");
    for (size_t i = 0; i < expanded.size(); ++i) {
        const auto bits = word(input, (i / 32) * 34);
        const auto q = std::bit_cast<int8_t>(std::to_integer<uint8_t>(input[(i / 32) * 34 + 2 + i % 32]));
        const double exact = double(tokspur::binary16(bits)) * double(q);
        require(std::isfinite(expanded[i]) && double(expanded[i]) == exact &&
                std::signbit(expanded[i]) == std::signbit(exact), "FP32 expansion changed represented value or signed zero");
        require(std::bit_cast<uint32_t>(expanded[i]) == std::bit_cast<uint32_t>(ordinary[i]), "FP32 expansion differs from owned numerical decoder");
        require(soa.scales[i / 32] == bits && soa.values[i] == q, "SoA changed raw scale or quantized value");
        ++products;
    }
    require(std::equal(input.begin(), input.end(), original.begin()), "transformation modified source bytes");
}

void exhaustive_products() {
    constexpr uint32_t columns = 256, chunk = 512;
    uint32_t finite_scales = 0;
    // Every finite half bit pattern times every int8 bit pattern, in bounded
    // chunks: maximum FP32 output is 512 KiB, not a full-model expansion.
    for (uint32_t base = 0; base < 65536; base += chunk) {
        std::vector<uint16_t> scales;
        for (uint32_t bits = base; bits < base + chunk; ++bits)
            if ((bits & 0x7c00) != 0x7c00) scales.push_back(uint16_t(bits));
        if (scales.empty()) continue;
        const uint32_t rows = uint32_t(scales.size()); finite_scales += rows;
        // Deliberately unaligned CPU byte span. No input reinterpret_cast is
        // needed, even when Q8 blocks/rows start two bytes past word alignment.
        std::vector<std::byte> storage(size_t(rows) * 8 * 34 + 2, std::byte{0x6d});
        const auto output = std::span(storage).subspan(1, size_t(rows) * 8 * 34);
        for (uint32_t row = 0; row < rows; ++row) for (uint32_t block = 0; block < 8; ++block) {
            const size_t offset = (size_t(row) * 8 + block) * 34;
            put16(output, offset, scales[row]);
            for (uint32_t lane = 0; lane < 32; ++lane) output[offset + 2 + lane] = std::byte(block * 32 + lane);
        }
        conversion(output, columns, rows);
        require(storage.front() == std::byte{0x6d} && storage.back() == std::byte{0x6d}, "unaligned source guard changed");
    }
    require(finite_scales == 63488 && products == uint64_t(63488) * 256, "exhaustive product population differs");
}

void shapes() {
    std::mt19937 random(0x5138534f);
    const std::array<uint16_t, 8> scales{0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff};
    for (const auto shape : std::array<std::array<uint32_t, 2>, 7>{{{32, 1}, {96, 3}, {320, 5}, {256, 7}, {10240, 2}, {10240, 320}, {320, 10240}}}) {
        const auto columns = shape[0], rows = shape[1];
        std::vector<std::byte> input(size_t(columns / 32) * rows * 34);
        for (auto& byte : input) byte = std::byte(random() & 255);
        for (size_t offset = 0, block = 0; offset < input.size(); offset += 34, ++block)
            put16(input, offset, scales[block % scales.size()]);
        conversion(input, columns, rows);
    }
}

void malformed() {
    std::vector<std::byte> bytes(34, std::byte{0x7f}); put16(bytes, 0, 0x3c00);
    const auto soa = tokspur::q8_soa(bytes, 32, 1, 34);
    for (const auto shape : std::array<std::array<uint32_t, 2>, 5>{{{0, 1}, {32, 0}, {31, 1}, {33, 1}, {64, 1}}}) {
        reject([&] { static_cast<void>(tokspur::q8_soa(bytes, shape[0], shape[1], 1000)); });
        reject([&] { static_cast<void>(tokspur::q8_expand_f32(bytes, shape[0], shape[1], 1000)); });
    }
    reject([&] { static_cast<void>(tokspur::q8_soa(bytes, 32, 1, 33)); });
    reject([&] { static_cast<void>(tokspur::q8_reconstruct(soa, 33)); });
    reject([&] { static_cast<void>(tokspur::q8_expand_f32(bytes, 32, 1, 127)); });
    for (const auto count : {0u, 33u, 35u}) {
        const std::vector<std::byte> malformed(count);
        reject([&] { static_cast<void>(tokspur::q8_soa(malformed, 32, 1, 1000)); });
        reject([&] { static_cast<void>(tokspur::q8_expand_f32(malformed, 32, 1, 1000)); });
    }
    for (uint32_t field = 0; field < 4; ++field) {
        auto bad = soa;
        if (field == 0) bad.columns = 0;
        else if (field == 1) bad.rows = 2;
        else if (field == 2) bad.scales.pop_back();
        else bad.values.pop_back();
        const auto before = bad;
        reject([&] { static_cast<void>(tokspur::q8_reconstruct(bad, 1000)); });
        require(same(bad, before), "invalid reconstruction changed input");
    }
    constexpr auto maximum = std::numeric_limits<uint32_t>::max();
    constexpr auto unlimited = std::numeric_limits<uint64_t>::max();
    reject([&] { static_cast<void>(tokspur::q8_soa({}, maximum - 31, maximum, unlimited)); });
    reject([&] { static_cast<void>(tokspur::q8_expand_f32({}, uint32_t(1) << 31, uint32_t(1) << 31, unlimited)); });
    auto huge = soa; huge.columns = maximum - 31; huge.rows = maximum;
    reject([&] { static_cast<void>(tokspur::q8_reconstruct(huge, unlimited)); });

    // Exhaust all positive/negative Inf and NaN encodings at the final block,
    // ensuring late validation cannot replace an existing published result.
    std::vector<std::byte> late(3 * 34, std::byte{0x7f});
    for (size_t offset = 0; offset < late.size(); offset += 34) put16(late, offset, 0x3c00);
    auto retained = soa;
    std::vector<float> retained_f32{123, -456};
    const auto initial_f32 = retained_f32;
    for (uint32_t bits = 0; bits < 65536; ++bits) if ((bits & 0x7c00) == 0x7c00) {
        put16(late, 2 * 34, uint16_t(bits)); const auto original = late;
        reject([&] { retained = tokspur::q8_soa(late, 96, 1, late.size()); });
        reject([&] { retained_f32 = tokspur::q8_expand_f32(late, 96, 1, 96 * 4); });
        auto bad = soa; bad.scales[0] = uint16_t(bits); const auto before = bad;
        reject([&] { static_cast<void>(tokspur::q8_reconstruct(bad, 34)); });
        require(late == original && same(bad, before) && same(retained, soa) && retained_f32 == initial_f32,
                "nonfinite refusal changed input or published output");
    }
    require(tokspur::q8_reconstruct(tokspur::q8_soa(bytes, 32, 1, 34), 34) == bytes,
            "valid conversion failed after malformed inputs");
}
} // namespace

int main() {
    try {
        exhaustive_products(); shapes(); malformed();
        std::cout << "Owned Q8 layout checks passed: " << checks << ", exact products: " << products
                  << ", exhaustive finite half scales: 63488, exhaustive int8 patterns: 256\n";
    } catch (const std::exception& error) {
        std::cerr << "Owned Q8 layout check failed: " << error.what() << '\n'; return 1;
    }
}
