#include "tokspur/gguf.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/moe.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/operators.hpp"
#include "tokspur/routing.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
size_t checks = 0, batches = 0;
double worst = 0;
void require(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
void rejected(const std::function<void()>& action) {
    ++checks; try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid MoE operation was accepted");
}
std::vector<double> real(std::span<const float> values) { return {values.begin(), values.end()}; }
std::vector<float> values(size_t count, std::mt19937& rng) {
    std::vector<float> output(count);
    for (auto& value : output) value = (int(rng() % 2001) - 1000) / 1000.0f;
    return output;
}
struct Arena {
    tokspur::MetalGraph& gpu;
    std::vector<tokspur::GpuView> owned;
    tokspur::GpuView data(std::span<const std::byte> bytes, bool immutable = true) {
        const auto view = gpu.upload(bytes, "generated MoE fixture", immutable); owned.push_back(view); return view;
    }
    tokspur::GpuView floats(std::span<const float> input, bool immutable = true) { return data(std::as_bytes(input), immutable); }
    tokspur::GpuView zero(size_t count) { const auto view = gpu.zeros(count * 4, "MoE test output"); owned.push_back(view); return view; }
    ~Arena() {
        try { gpu.abort(); } catch (const tokspur::FormatError&) {}
        for (const auto view : owned) gpu.release(view);
    }
};
template<class T> std::vector<T> read(tokspur::MetalGraph& gpu, tokspur::GpuView view) {
    const auto data = gpu.read(view);
    require(data.size() % sizeof(T) == 0, "readback alignment");
    std::vector<T> output(data.size() / sizeof(T));
    std::memcpy(output.data(), data.data(), data.size());
    return output;
}
void close(std::span<const float> actual, std::span<const double> expected, double tolerance = 5e-5) {
    require(actual.size() == expected.size(), "MoE result shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        const auto error = std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i]));
        worst = std::max(worst, error);
        require(std::isfinite(actual[i]) && error <= tolerance, "MoE numerical tolerance exceeded");
    }
}
void submit(tokspur::MetalGraph& gpu, size_t dispatches, uint64_t allocations) {
    const auto batch = gpu.submit();
    require(batch.dispatches == dispatches && gpu.memory().allocations == allocations, "MoE dispatch/allocation invariant");
    ++batches;
}
std::vector<double> dense(std::span<const float> matrix, std::span<const double> input) {
    require(matrix.size() % input.size() == 0, "dense oracle shape");
    std::vector<double> output(matrix.size() / input.size(), 0);
    for (size_t row = 0; row < output.size(); ++row) for (size_t col = 0; col < input.size(); ++col) output[row] += matrix[row * input.size() + col] * input[col];
    return output;
}

void routes(tokspur::MetalGraph& gpu, std::mt19937& rng) {
    for (const auto count : {1u, 5u, 127u, 512u, 4096u}) for (size_t seed = 0; seed < 5; ++seed) {
        Arena arena{gpu, {}};
        auto logits = values(count, rng);
        if (seed == 0) std::fill(logits.begin(), logits.end(), 0);
        if (seed == 1) for (auto& x : logits) x *= 1000;
        const uint32_t selected = std::min(count, seed % 2 ? 64u : 10u);
        const auto input = arena.floats(logits), ids = arena.zero(selected), weights = arena.zero(selected);
        const auto allocations = gpu.memory().allocations;
        for (const auto mode : {tokspur::RouterMode::softmax, tokspur::RouterMode::sigmoid}) for (const bool normalize : {false, true}) {
            const auto expected = tokspur::route_reference(real(logits), selected, mode, normalize);
            try {
                gpu.begin(); gpu.route(input, ids, weights, selected, mode, normalize); submit(gpu, 1, allocations);
            } catch (...) {
                std::cerr << "router fixture count=" << count << " seed=" << seed << " selected=" << selected
                          << " mode=" << uint32_t(mode) << " normalize=" << normalize << " max=" << *std::max_element(logits.begin(), logits.end()) << '\n';
                throw;
            }
            require(read<uint32_t>(gpu, ids) == expected.indices, "GPU route IDs differ");
            close(read<float>(gpu, weights), expected.weights);
        }
    }
}

void dispatched_formats(tokspur::MetalGraph& gpu, std::mt19937& rng) {
    constexpr uint32_t experts = 5, selected = 3, rows = 7;
    const std::array<uint32_t, selected> indices{4, 0, 2};
    for (const uint32_t type : {0, 1, 7, 8, 12, 13, 14, 20, 30}) {
        Arena arena{gpu, {}};
        const auto& layout = tokspur::encoding(type);
        const auto columns = layout.elements_per_block * 2;
        const auto row_bytes = layout.bytes_per_block * 2;
        std::vector<std::byte> weights(experts * rows * row_bytes);
        for (auto& byte : weights) byte = std::byte(rng() & 255);
        const auto put = [&](size_t offset, uint64_t value, size_t bytes) {
            for (size_t i = 0; i < bytes; ++i) weights[offset + i] = std::byte((value >> (i * 8)) & 255);
        };
        for (size_t at = 0; at < weights.size(); at += layout.bytes_per_block) {
            if (type == 0) put(at, std::bit_cast<uint32_t>(0.125f), 4);
            else if (type == 30) put(at, 0x3e00, 2);
            else { put(at + (type == 14 ? 208 : 0), 0x2400, 2); if (type == 7 || type == 12 || type == 13) put(at + 2, 0x2000, 2); }
        }
        const auto matrix = arena.data(weights), selectors = arena.data(std::as_bytes(std::span(indices))), output = arena.zero(selected * rows);
        for (const bool broadcast : {false, true}) {
            const auto input_values = values(columns * (broadcast ? 1 : selected), rng);
            const auto input = arena.floats(input_values);
            gpu.begin(); gpu.routed_gemv(type, matrix, selectors, input, output, columns, rows, experts, broadcast);
            submit(gpu, 1, gpu.memory().allocations);
            std::vector<double> expected;
            for (uint32_t slot = 0; slot < selected; ++slot) {
                const auto part = tokspur::reference_gemv(type, std::span(weights).subspan(indices[slot] * rows * row_bytes, rows * row_bytes), columns, rows,
                    std::span(input_values).subspan(broadcast ? 0 : slot * columns, columns));
                expected.insert(expected.end(), part.begin(), part.end());
            }
            close(read<float>(gpu, output), expected);
        }
    }
}

void composed(tokspur::MetalGraph& gpu, std::mt19937& rng) {
    for (const auto shape : std::array<tokspur::MoeShape, 4>{{{1, 1, 1, 1}, {7, 9, 5, 3}, {32, 16, 17, 4}, {64, 32, 512, 10}}}) {
        Arena arena{gpu, {}};
        const auto d = shape.channels, h = shape.hidden, e = shape.experts;
        const auto x = values(d, rng), router = values(d * e, rng), gate = values(d * h * e, rng), up = values(d * h * e, rng);
        const auto down = values(h * d * e, rng), sg = values(d * h, rng), su = values(d * h, rng), sd = values(h * d, rng), sr = values(d, rng);
        const auto matrix = [&](const auto& input) { return tokspur::GpuMatrix{arena.floats(input), 0}; };
        const tokspur::MoeWeights w{matrix(router), matrix(gate), matrix(up), matrix(down), matrix(sg), matrix(su), matrix(sd), matrix(sr)};
        const auto input = arena.floats(x), output = arena.zero(d);
        const auto scratch = tokspur::allocate_moe_scratch(gpu, shape);
        const auto allocations = gpu.memory().allocations;
        for (const auto mode : {tokspur::RouterMode::softmax, tokspur::RouterMode::sigmoid}) for (const bool normalize : {false, true}) {
            const auto logits = dense(router, real(x));
            const auto routing = tokspur::route_reference(logits, shape.selected, mode, normalize);
            std::vector<double> routed;
            for (const auto expert : routing.indices) {
                auto g = dense(std::span(gate).subspan(expert * d * h, d * h), real(x));
                const auto u = dense(std::span(up).subspan(expert * d * h, d * h), real(x));
                for (size_t i = 0; i < h; ++i) g[i] = tokspur::stable_silu(g[i]) * u[i];
                const auto result = dense(std::span(down).subspan(expert * h * d, h * d), g);
                routed.insert(routed.end(), result.begin(), result.end());
            }
            auto shared_g = dense(sg, real(x));
            const auto shared_u = dense(su, real(x));
            for (size_t i = 0; i < h; ++i) shared_g[i] = tokspur::stable_silu(shared_g[i]) * shared_u[i];
            const auto expected = tokspur::mixture_reference(routed, routing.weights, dense(sd, shared_g), dense(sr, real(x))[0]);
            gpu.begin(); tokspur::encode_moe(gpu, w, scratch, input, output, shape, mode, normalize); submit(gpu, 14, allocations);
            require(read<uint32_t>(gpu, scratch.indices) == routing.indices, "composed router IDs differ");
            close(read<float>(gpu, output), expected);
        }
        tokspur::release_moe_scratch(gpu, scratch);
    }
}

void invalid(tokspur::MetalGraph& gpu, const std::filesystem::path& library) {
    Arena arena{gpu, {}};
    const std::array<float, 4> logits{1, 2, 3, 4};
    const auto x = arena.floats(logits), ids = arena.zero(2), weights = arena.zero(2);
    gpu.begin();
    rejected([&] { gpu.route(x, ids, weights, 0, tokspur::RouterMode::softmax, true); });
    rejected([&] { gpu.route(x, ids, weights, 5, tokspur::RouterMode::softmax, true); });
    rejected([&] { gpu.route(x, ids, weights, 2, static_cast<tokspur::RouterMode>(3), true); });
    rejected([&] { gpu.route(x, ids, ids, 2, tokspur::RouterMode::softmax, true); });
    gpu.route(x, ids, weights, 2, tokspur::RouterMode::softmax, true); submit(gpu, 1, gpu.memory().allocations);
    const std::array<uint32_t, 2> bad_indices{0, 0xffffffff};
    const auto bad = arena.data(std::as_bytes(std::span(bad_indices))), output = arena.zero(2);
    const auto scalar = gpu.slice(x, 0, 4);
    gpu.begin(); gpu.routed_gemv(0, x, bad, scalar, output, 1, 1, 4, true);
    rejected([&] { static_cast<void>(gpu.submit()); });
    const std::array<float, 4> bad_logits{0, 1, std::numeric_limits<float>::infinity(), 2};
    const auto invalid_logits = arena.floats(bad_logits);
    gpu.begin(); gpu.route(invalid_logits, ids, weights, 2, tokspur::RouterMode::softmax, true);
    rejected([&] { static_cast<void>(gpu.submit()); });
    gpu.begin(); gpu.route(x, ids, weights, 2, tokspur::RouterMode::sigmoid, true); submit(gpu, 1, gpu.memory().allocations);
    require(read<uint32_t>(gpu, ids) == std::vector<uint32_t>({3, 2}), "router did not recover after invalid input");
    tokspur::MetalGraph limited(library, 65536);
    const auto base = limited.memory().live_bytes;
    rejected([&] { static_cast<void>(tokspur::allocate_moe_scratch(limited, {2560, 640, 512, 10})); });
    require(limited.memory().live_bytes == base, "failed scratch allocation did not roll back its buffers");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        tokspur::MetalGraph gpu(argv[1], 256 * 1024 * 1024);
        const auto initial = gpu.memory().live_bytes;
        std::mt19937 rng(0x6d6f6531);
        routes(gpu, rng); dispatched_formats(gpu, rng); composed(gpu, rng); invalid(gpu, argv[1]);
        require(gpu.memory().live_bytes == initial, "MoE test allocation leak");
        std::cout << "Owned MoE graph checks passed: " << checks << ", batches: " << batches << ", max scaled error: " << worst << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Owned MoE graph check failed: " << error.what() << '\n'; return 1;
    }
}
