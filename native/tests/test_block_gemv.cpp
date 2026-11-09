#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>

namespace {
size_t checks = 0, batches = 0;
double worst = 0;
void require(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const tokspur::FormatError&) { return; } throw std::runtime_error("invalid block GEMV accepted"); }
void reject_alias(const std::function<void()>& f) {
    ++checks;
    try { f(); } catch (const tokspur::FormatError& error) {
        require(std::string_view(error.what()) == "unsupported GPU buffer alias", "alias fixture failed before the overlap guard");
        return;
    }
    throw std::runtime_error("shape-valid partial alias accepted");
}
void put16(std::span<std::byte> data, size_t offset, uint16_t value) {
    data[offset] = std::byte(value & 255); data[offset + 1] = std::byte(value >> 8);
}
std::vector<float> read(tokspur::MetalGraph& gpu, tokspur::GpuView output) {
    const auto bytes = gpu.read(output); std::vector<float> result(bytes.size() / 4);
    require(bytes.size() % 4 == 0, "non-float result"); std::memcpy(result.data(), bytes.data(), bytes.size()); return result;
}
void close(std::span<const float> actual, std::span<const double> expected) {
    require(actual.size() == expected.size(), "block result shape");
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i]));
        worst = std::max(worst, error); require(std::isfinite(actual[i]) && error <= 5e-5, "block GEMV independent reference differs");
    }
}
struct Arena {
    tokspur::MetalGraph& gpu;
    std::vector<tokspur::GpuView> all;
    tokspur::GpuView bytes(std::span<const std::byte> data, bool immutable = true) {
        // Exercise views starting at four bytes, rather than accidentally relying
        // on page/16-byte alignment in typed loads.
        std::vector<std::byte> padded(data.size() + 8, std::byte{0x57});
        std::copy(data.begin(), data.end(), padded.begin() + 4);
        const auto full = gpu.upload(padded, "block fixture", immutable); all.push_back(full);
        return gpu.slice(full, 4, data.size());
    }
    tokspur::GpuView floats(std::span<const float> data, bool immutable = true) { return bytes(std::as_bytes(data), immutable); }
    tokspur::GpuView zeros(size_t count) { return floats(std::vector<float>(count), false); }
    ~Arena() { try { gpu.abort(); } catch (const tokspur::FormatError&) {} for (const auto view : all) gpu.release(view); }
};
void submit(tokspur::MetalGraph& gpu, uint64_t allocations) {
    const auto result = gpu.submit(); ++batches;
    require(result.dispatches == 1 && gpu.memory().allocations == allocations, "block path changed physical dispatch/allocation count");
}
std::vector<std::byte> matrix(uint32_t type, uint32_t columns, uint32_t rows, std::mt19937& rng) {
    const auto layout = tokspur::encoding(type);
    std::vector<std::byte> result(uint64_t(columns / layout.elements_per_block) * layout.bytes_per_block * rows);
    for (auto& v : result) v = std::byte(rng() & 255);
    const std::array<uint16_t, 12> scales{0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x2000, 0x2400, 0xa400, 0x3c00, 0x7bff, 0xfbff};
    for (size_t offset = 0, block = 0; offset < result.size(); offset += layout.bytes_per_block, ++block) {
        // Moderate scales for mixed cancellation tests; exhaustive extremes use
        // a separate one-block fixture with a well-conditioned reference.
        put16(result, offset, scales[block % 10]);
        if (type == 12) put16(result, offset + 2, scales[(block * 3 + 1) % 10]);
        if (type == 8) { result[offset + 2] = std::byte{128}; result[offset + 3] = std::byte{127}; }
        else {
            // All four header low/high-bit packing cases, including 0 and 63.
            result[offset + 4] = std::byte{0}; result[offset + 5] = std::byte{255};
            result[offset + 8] = std::byte{255}; result[offset + 9] = std::byte{0};
            result[offset + 12] = std::byte{0}; result[offset + 13] = std::byte{255};
            result[offset + 16] = std::byte{0}; result[offset + 17] = std::byte{255};
        }
    }
    return result;
}
void plain(tokspur::MetalGraph& gpu, uint32_t type, uint32_t columns, uint32_t rows, std::mt19937& rng) {
    Arena arena{gpu, {}}; const auto weights = matrix(type, columns, rows, rng);
    std::vector<float> x(columns); for (auto& v : x) v = (int(rng() % 2049) - 1024) / 4096.0f;
    const auto expected = tokspur::reference_gemv(type, weights, columns, rows, x);
    const auto w = arena.bytes(weights), input = arena.floats(x, false), out = arena.zeros(rows);
    std::vector<std::byte> ordinary;
    for (const uint32_t mode : {0, 1, 2, 3}) {
        gpu.configure_block_gemv(mode); const auto allocations = gpu.memory().allocations;
        for (uint32_t repeat = 0; repeat < 2; ++repeat) {
            gpu.begin(); reject([&] { gpu.configure_block_gemv(0); }); gpu.gemv(type, w, input, out, columns, rows); submit(gpu, allocations);
            close(read(gpu, out), expected);
            if (!mode) ordinary = gpu.read(out); else require(gpu.read(out) == ordinary, "block decoding changed ordinary result bits");
        }
    }
    gpu.configure_block_gemv(3);
    const auto allocation = gpu.memory().allocations;
    gpu.begin(); reject([&] { gpu.gemv(type, w, input, input, columns, rows); }); gpu.abort();
    gpu.begin(); reject([&] { gpu.gemv(type, w, input, w, columns, rows); }); gpu.abort();
    gpu.begin(); reject([&] { gpu.gemv(type, w, input, out, columns - 1, rows); }); gpu.abort();
    std::fill(x.begin(), x.end(), std::numeric_limits<float>::infinity()); gpu.write(input, std::as_bytes(std::span(x)));
    gpu.begin(); gpu.gemv(type, w, input, out, columns, rows); reject([&] { static_cast<void>(gpu.submit()); });
    std::fill(x.begin(), x.end(), 0); gpu.write(input, std::as_bytes(std::span(x)));
    gpu.begin(); gpu.gemv(type, w, input, out, columns, rows); submit(gpu, allocation); close(read(gpu, out), std::vector<double>(rows, 0));
    require(gpu.read(w) == weights, "block path changed immutable matrix");
}
void scales(tokspur::MetalGraph& gpu) {
    Arena arena{gpu, {}};
    constexpr uint32_t rows = 63488; // All finite positive/negative binary16 bit patterns.
    std::vector<std::byte> weights(uint64_t(rows) * 34);
    std::vector<double> expected(rows); uint32_t row = 0;
    std::array<float, 32> x{}; x.fill(0.03125f);
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        if ((bits & 0x7c00) == 0x7c00) continue;
        put16(weights, size_t(row) * 34, uint16_t(bits));
        // Fixed positive sum, with signed extremes in every row. The operations
        // remain exactly representable until the final FP32 reduction.
        for (uint32_t i = 0; i < 32; ++i) weights[size_t(row) * 34 + 2 + i] = std::byte(i == 0 ? 128 : 127);
        expected[row] = double(tokspur::binary16(uint16_t(bits))) * (31 * 127 - 128) / 32.0;
        ++row;
    }
    require(row == rows, "finite scale fixture population");
    const auto w = arena.bytes(weights), input = arena.floats(x), out = arena.zeros(rows);
    std::vector<std::byte> ordinary;
    for (const uint32_t mode : {0, 1}) {
        gpu.configure_block_gemv(mode); const auto allocations = gpu.memory().allocations;
        gpu.begin(); gpu.gemv(8, w, input, out, 32, rows); submit(gpu, allocations); close(read(gpu, out), expected);
        if (!mode) ordinary = gpu.read(out); else require(gpu.read(out) == ordinary, "finite half scale typed load changed bits");
    }
    // Nonfinite scale rejection and reuse, including a signed NaN payload.
    for (const uint16_t bits : {uint16_t(0x7c00), uint16_t(0xfc00), uint16_t(0x7e01), uint16_t(0xfe01)}) {
        std::vector<std::byte> bad(34, std::byte{1}); put16(bad, 0, bits);
        const auto bad_w = arena.bytes(bad), bad_out = arena.zeros(1);
        gpu.begin(); gpu.gemv(8, bad_w, input, bad_out, 32, 1); reject([&] { static_cast<void>(gpu.submit()); });
        gpu.begin(); gpu.gemv(8, w, input, out, 32, rows); submit(gpu, gpu.memory().allocations);
        require(gpu.read(out) == ordinary, "nonfinite scale damaged subsequent work");
    }
}
void q4_scales(tokspur::MetalGraph& gpu) {
    constexpr uint32_t rows = 63488, columns = 256;
    // Sequential arenas retain only one 9,142,272-byte exhaustive matrix at a
    // time. Uniform coefficients make the independent result exactly d/-dmin.
    for (const bool minimum_axis : {false, true}) {
        Arena arena{gpu, {}};
        std::vector<std::byte> weights(uint64_t(rows) * 144);
        std::vector<double> expected(rows); uint32_t row = 0;
        for (uint32_t bits = 0; bits < 65536; ++bits) {
            if ((bits & 0x7c00) == 0x7c00) continue;
            const size_t at = size_t(row) * 144;
            put16(weights, at + (minimum_axis ? 2 : 0), uint16_t(bits));
            for (uint32_t slot = 0; slot < 4; ++slot) {
                weights[at + 4 + slot] = std::byte(minimum_axis ? 0 : 1);
                weights[at + 8 + slot] = std::byte(minimum_axis ? 1 : 0);
                weights[at + 12 + slot] = std::byte(minimum_axis ? 0x10 : 1);
            }
            std::fill(weights.begin() + at + 16, weights.begin() + at + 144, std::byte{0x11});
            expected[row] = (minimum_axis ? -1.0 : 1.0) * double(tokspur::binary16(uint16_t(bits)));
            ++row;
        }
        require(row == rows, "Q4 finite scale fixture population differs");
        std::array<float, columns> x{}; x.fill(1.0f / columns);
        const auto w = arena.bytes(weights), input = arena.floats(x), out = arena.zeros(rows);
        std::vector<std::byte> ordinary;
        for (const uint32_t mode : {0, 2, 3}) {
            gpu.configure_block_gemv(mode); const auto allocations = gpu.memory().allocations;
            gpu.begin(); gpu.gemv(12, w, input, out, columns, rows); submit(gpu, allocations);
            const auto actual = read(gpu, out); close(actual, expected);
            for (size_t i = 0; i < actual.size(); ++i)
                require(double(actual[i]) == expected[i], "Q4 uniform coefficient fixture lost an exactly representable value");
            if (!mode) ordinary = gpu.read(out);
            else require(gpu.read(out) == ordinary, "Q4 finite half typed loads changed ordinary bits");
        }
        require(gpu.read(w) == weights, "Q4 exhaustive scale test changed source weights");
    }
    Arena arena{gpu, {}};
    // Nonzero d and dmin terms: 1*1*2 - 0.5*1 = 1.5 in every channel.
    std::vector<std::byte> good(144, std::byte{0x22});
    put16(good, 0, 0x3c00); put16(good, 2, 0x3800);
    for (uint32_t slot = 0; slot < 4; ++slot) {
        good[4 + slot] = std::byte{1}; good[8 + slot] = std::byte{1}; good[12 + slot] = std::byte{0x11};
    }
    std::array<float, columns> x{}; x.fill(1.0f / columns);
    const auto input = arena.floats(x), valid_w = arena.bytes(good), out = arena.zeros(1);
    for (const uint32_t mode : {0, 2, 3}) {
        gpu.configure_block_gemv(mode);
        gpu.begin(); gpu.gemv(12, valid_w, input, out, columns, 1); submit(gpu, gpu.memory().allocations);
        close(read(gpu, out), std::array<double, 1>{1.5}); const auto ordinary = gpu.read(out);
        for (const uint32_t offset : {0u, 2u}) for (const uint16_t bits : {
                 uint16_t(0x7c00), uint16_t(0xfc00), uint16_t(0x7c01), uint16_t(0xfc01),
                 uint16_t(0x7e00), uint16_t(0xfe00), uint16_t(0x7fff), uint16_t(0xffff)}) {
            auto bad = good; put16(bad, offset, bits); const auto bad_w = arena.bytes(bad);
            gpu.begin(); gpu.gemv(12, bad_w, input, out, columns, 1); reject([&] { static_cast<void>(gpu.submit()); });
            gpu.begin(); gpu.gemv(12, valid_w, input, out, columns, 1); submit(gpu, gpu.memory().allocations);
            require(gpu.read(out) == ordinary && gpu.read(bad_w) == bad, "Q4 nonfinite scale damaged reuse or source bytes");
        }
    }
}

void partial_aliases(tokspur::MetalGraph& gpu, std::mt19937& rng) {
    for (const uint32_t type : {8u, 12u}) {
        Arena arena{gpu, {}};
        const uint32_t columns = tokspur::encoding(type).elements_per_block, rows = 4, experts = 3, selected = 2;
        const auto weights = matrix(type, columns, rows * experts, rng);
        const auto w = arena.bytes(weights, false);
        const auto plain_w = gpu.slice(w, 0, weights.size() / experts);
        const auto input = arena.floats(std::vector<float>(columns, 0.125f), false);
        const auto plain_out = arena.zeros(rows), routed_out = arena.zeros(rows * selected);
        const std::array<uint32_t, 10> id_data{2, 0};
        const auto id_storage = arena.bytes(std::as_bytes(std::span(id_data)), false);
        const auto indices = gpu.slice(id_storage, 0, selected * 4);
        const auto refusal = [&](tokspur::GpuView overlap, const std::function<void()>& operation) {
            const auto before = gpu.read(overlap); const auto allocations = gpu.memory().allocations;
            gpu.begin(); reject_alias(operation); gpu.abort();
            require(gpu.read(overlap) == before && gpu.memory().allocations == allocations, "alias refusal changed bytes or allocated scratch");
        };
        for (const uint32_t mode : {0, 3}) {
            gpu.configure_block_gemv(mode);
            const auto input_overlap = gpu.slice(input, 4, rows * 4);
            const auto matrix_overlap = gpu.slice(plain_w, 4, rows * 4);
            refusal(input_overlap, [&] { gpu.gemv(type, plain_w, input, input_overlap, columns, rows); });
            refusal(matrix_overlap, [&] { gpu.gemv(type, plain_w, input, matrix_overlap, columns, rows); });
            gpu.begin(); gpu.gemv(type, plain_w, input, plain_out, columns, rows); submit(gpu, gpu.memory().allocations);
            const auto expected = tokspur::reference_gemv(type, std::span(weights).first(weights.size() / experts), columns, rows, std::vector<float>(columns, 0.125f));
            close(read(gpu, plain_out), expected);
            for (const bool broadcast : {false, true}) {
                const auto routed_input = arena.floats(std::vector<float>(columns * (broadcast ? 1 : selected), 0.125f), false);
                const auto x_overlap = gpu.slice(routed_input, 4, rows * selected * 4);
                const auto w_overlap = gpu.slice(w, 4, rows * selected * 4);
                const auto id_overlap = gpu.slice(id_storage, 4, rows * selected * 4);
                refusal(x_overlap, [&] { gpu.routed_gemv(type, w, indices, routed_input, x_overlap, columns, rows, experts, broadcast); });
                refusal(w_overlap, [&] { gpu.routed_gemv(type, w, indices, routed_input, w_overlap, columns, rows, experts, broadcast); });
                refusal(id_overlap, [&] { gpu.routed_gemv(type, w, indices, routed_input, id_overlap, columns, rows, experts, broadcast); });
                gpu.begin(); gpu.routed_gemv(type, w, indices, routed_input, routed_out, columns, rows, experts, broadcast); submit(gpu, gpu.memory().allocations);
                std::vector<double> routed_expected;
                for (uint32_t route = 0; route < selected; ++route) {
                    const auto value = tokspur::reference_gemv(type,
                        std::span(weights).subspan(id_data[route] * weights.size() / experts, weights.size() / experts),
                        columns, rows, std::vector<float>(columns, 0.125f));
                    routed_expected.insert(routed_expected.end(), value.begin(), value.end());
                }
                close(read(gpu, routed_out), routed_expected);
            }
        }
    }
}

void fallback_formats(tokspur::MetalGraph& gpu, std::mt19937& rng) {
    constexpr uint32_t rows = 5, experts = 3, selected = 2;
    const std::array<uint32_t, selected> ids{2, 0};
    for (const uint32_t type : {0u, 1u, 7u, 13u, 14u, 20u, 30u}) {
        Arena arena{gpu, {}}; const auto layout = tokspur::encoding(type);
        const auto columns = layout.elements_per_block * 3, row_bytes = layout.bytes_per_block * 3;
        std::vector<std::byte> weights(size_t(row_bytes) * rows * experts);
        for (auto& byte : weights) byte = std::byte(rng() & 255);
        for (size_t at = 0, block = 0; at < weights.size(); at += layout.bytes_per_block, ++block) {
            if (type == 0) {
                const auto bits = std::bit_cast<uint32_t>(float(int(block % 17) - 8) / 512.0f);
                for (uint32_t i = 0; i < 4; ++i) weights[at + i] = std::byte((bits >> (i * 8)) & 255);
            } else if (type == 30) put16(weights, at, block % 2 ? 0xbc00 : 0x3c00);
            else {
                put16(weights, at + (type == 14 ? 208 : 0), 0x1400);
                if (type == 7 || type == 13) put16(weights, at + 2, 0x1000);
            }
        }
        const auto w = arena.bytes(weights), plain_w = gpu.slice(w, 0, rows * row_bytes);
        const auto indices = arena.bytes(std::as_bytes(std::span(ids))), out = arena.zeros(rows * selected), plain_out = arena.zeros(rows);
        for (const bool broadcast : {false, true}) {
            std::vector<float> x(columns * (broadcast ? 1 : selected));
            for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 19) - 9) / 512.0f;
            const auto input = arena.floats(x), plain_input = gpu.slice(input, 0, columns * 4);
            const auto plain_expected = tokspur::reference_gemv(type, std::span(weights).first(rows * row_bytes), columns, rows, std::span(x).first(columns));
            std::vector<double> expected;
            for (uint32_t route = 0; route < selected; ++route) {
                const auto result = tokspur::reference_gemv(type, std::span(weights).subspan(ids[route] * rows * row_bytes, rows * row_bytes),
                    columns, rows, std::span(x).subspan(broadcast ? 0 : route * columns, columns));
                expected.insert(expected.end(), result.begin(), result.end());
            }
            std::vector<std::byte> ordinary, plain_ordinary;
            for (const uint32_t mode : {0, 3}) {
                gpu.configure_block_gemv(mode); const auto allocations = gpu.memory().allocations;
                gpu.begin(); gpu.gemv(type, plain_w, plain_input, plain_out, columns, rows); submit(gpu, allocations);
                close(read(gpu, plain_out), plain_expected);
                gpu.begin(); gpu.routed_gemv(type, w, indices, input, out, columns, rows, experts, broadcast); submit(gpu, allocations);
                close(read(gpu, out), expected);
                if (!mode) { ordinary = gpu.read(out); plain_ordinary = gpu.read(plain_out); }
                else require(gpu.read(out) == ordinary && gpu.read(plain_out) == plain_ordinary, "block mask changed non-Q8/Q4 fallback result bits");
            }
        }
        require(gpu.read(w) == weights, "fallback format test changed source weights");
    }
}
void routed(tokspur::MetalGraph& gpu, uint32_t type, uint32_t columns, std::mt19937& rng) {
    constexpr uint32_t rows = 5, experts = 7, routes = 4;
    Arena arena{gpu, {}}; const auto weights = matrix(type, columns, rows * experts, rng);
    const auto row_bytes = columns / tokspur::encoding(type).elements_per_block * tokspur::encoding(type).bytes_per_block;
    std::array<uint32_t, routes> ids{6, 0, 6, 2}; const auto indices = arena.bytes(std::as_bytes(std::span(ids)), false);
    const auto w = arena.bytes(weights), out = arena.zeros(rows * routes);
    for (const bool broadcast : {false, true}) {
        std::vector<float> x(columns * (broadcast ? 1 : routes)); for (auto& v : x) v = (int(rng() % 2049) - 1024) / 4096.0f;
        std::vector<double> expected;
        for (uint32_t route = 0; route < routes; ++route) {
            const auto value = tokspur::reference_gemv(type,
                std::span(weights).subspan(size_t(ids[route]) * rows * row_bytes, rows * row_bytes), columns, rows,
                std::span(x).subspan(broadcast ? 0 : route * columns, columns));
            expected.insert(expected.end(), value.begin(), value.end());
        }
        const auto input = arena.floats(x); std::vector<std::byte> ordinary;
        for (const uint32_t mode : {0, 1, 2, 3}) {
            gpu.configure_block_gemv(mode); const auto allocations = gpu.memory().allocations;
            gpu.begin(); gpu.routed_gemv(type, w, indices, input, out, columns, rows, experts, broadcast); submit(gpu, allocations);
            close(read(gpu, out), expected);
            if (!mode) ordinary = gpu.read(out); else require(gpu.read(out) == ordinary, "routed block changed ordinary bits");
        }
        ids[0] = experts; gpu.write(indices, std::as_bytes(std::span(ids)));
        gpu.begin(); gpu.routed_gemv(type, w, indices, input, out, columns, rows, experts, broadcast); reject([&] { static_cast<void>(gpu.submit()); });
        ids[0] = 6; gpu.write(indices, std::as_bytes(std::span(ids)));
        const auto allocations = gpu.memory().allocations;
        gpu.begin(); gpu.routed_gemv(type, w, indices, input, out, columns, rows, experts, broadcast); submit(gpu, allocations);
        require(gpu.read(out) == ordinary, "invalid expert damaged subsequent work");
    }
}
}
int main(int argc, char** argv) {
    try {
        using namespace tokspur;
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph gpu(argv[1], 64 * 1024 * 1024); std::mt19937 rng(0x424c4f43);
        const auto initial = gpu.memory().live_bytes; reject([&] { gpu.configure_block_gemv(4); });
        for (const auto shape : std::array<std::array<uint32_t, 2>, 8>{{{32, 1}, {64, 3}, {96, 4}, {256, 5}, {2560, 7}, {10240, 320}, {2560, 48}, {320, 10240}}})
            plain(gpu, 8, shape[0], shape[1], rng);
        for (const auto shape : std::array<std::array<uint32_t, 2>, 5>{{{256, 1}, {512, 3}, {768, 4}, {2560, 7}, {2560, 640}}})
            plain(gpu, 12, shape[0], shape[1], rng);
        scales(gpu); q4_scales(gpu);
        std::mt19937 coverage_rng(0x51434f56);
        partial_aliases(gpu, coverage_rng); fallback_formats(gpu, coverage_rng);
        routed(gpu, 8, 96, rng); routed(gpu, 8, 2560, rng);
        routed(gpu, 12, 256, rng); routed(gpu, 12, 2560, rng);
        gpu.configure_block_gemv(0); require(gpu.memory().live_bytes == initial, "block tests leaked GPU backing");
        std::cout << "Owned block GEMV checks passed: " << checks << ", batches: " << batches << ", max scaled error: " << worst << '\n';
    } catch (const std::exception& e) { std::cerr << "Owned block GEMV failed: " << e.what() << '\n'; return 1; }
}
