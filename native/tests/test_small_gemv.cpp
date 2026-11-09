#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
size_t checks = 0;
double worst = 0;
void require(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const tokspur::FormatError&) { return; } throw std::runtime_error("invalid split GEMV accepted"); }
double median(std::vector<double> x) { std::sort(x.begin(), x.end()); return x[x.size() / 2]; }
void check(tokspur::MetalGraph& graph, tokspur::GpuView output, std::span<const double> expected) {
    const auto bytes = graph.read(output); std::vector<float> actual(expected.size());
    require(bytes.size() == actual.size() * sizeof(float), "split GEMV result shape");
    std::memcpy(actual.data(), bytes.data(), bytes.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const auto error = std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i])); worst = std::max(worst, error);
        require(std::isfinite(actual[i]) && error < 5e-5, "split GEMV numerical error");
    }
}
}
int main(int argc, char** argv) {
    try {
        using namespace tokspur;
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph graph(argv[1], 64 * 1024 * 1024); std::mt19937 rng(0x53504c54);
        const auto initial = graph.memory().live_bytes;
        reject([&] { graph.configure_small_gemv(2); }); reject([&] { graph.configure_small_gemv(64); });
        for (const auto& dimensions : std::array<std::array<uint32_t, 2>, 8>{{
                 {10240, 4}, {2560, 48}, {2560, 1}, {65536, 64}, {1025, 3}, {2049, 63}, {1023, 4}, {1024, 65}}}) {
            const auto columns = dimensions[0], rows = dimensions[1];
            std::vector<float> w(uint64_t(columns) * rows), x(columns);
            for (auto& a : w) a = (int(rng() % 2001) - 1000) / 16384.0f;
            for (auto& a : x) a = (int(rng() % 2001) - 1000) / 1024.0f;
            const auto expected = reference_gemv(0, std::as_bytes(std::span(w)), columns, rows, x);
            const auto weights = graph.upload(std::as_bytes(std::span(w)), "split fixture matrix");
            const auto input = graph.upload(std::as_bytes(std::span(x)), "split fixture input", false);
            const auto output = graph.zeros(rows * sizeof(float), "split fixture output");
            std::vector<std::byte> ordinary;
            for (const uint32_t splits : {0, 1, 4, 8, 16, 32}) {
                graph.configure_small_gemv(splits); const auto allocations = graph.memory().allocations;
                std::vector<double> timings;
                for (uint32_t repeat = 0; repeat < 7; ++repeat) {
                    graph.begin(); reject([&] { graph.configure_small_gemv(0); });
                    for (uint32_t operation = 0; operation < 16; ++operation) graph.gemv(0, weights, input, output, columns, rows);
                    const auto batch = graph.submit();
                    const bool split = splits > 1 && rows <= 64 && columns >= 1024;
                    require(batch.dispatches == 16 * (split ? 2 : 1) && graph.memory().allocations == allocations, "split GEMV dispatch or allocation differs");
                    check(graph, output, expected);
                    if (!splits) ordinary = graph.read(output);
                    if (splits == 1) require(graph.read(output) == ordinary, "aligned FP32 read changed ordinary result bits");
                    if (repeat) timings.push_back(batch.gpu_ms / 16);
                }
                std::cout << "{\"component\":\"small_gemv\",\"columns\":" << columns << ",\"rows\":" << rows
                          << ",\"splits\":" << splits << ",\"diagnostic_median_gpu_ms\":" << median(timings) << "}\n";
            }
            graph.configure_small_gemv(16);
            std::fill(x.begin(), x.end(), 0); graph.write(input, std::as_bytes(std::span(x)));
            graph.begin(); graph.gemv(0, weights, input, output, columns, rows); static_cast<void>(graph.submit());
            check(graph, output, std::vector<double>(rows, 0));
            std::fill(x.begin(), x.end(), std::numeric_limits<float>::infinity()); graph.write(input, std::as_bytes(std::span(x)));
            graph.begin(); graph.gemv(0, weights, input, output, columns, rows); reject([&] { static_cast<void>(graph.submit()); });
            std::fill(x.begin(), x.end(), 0); graph.write(input, std::as_bytes(std::span(x)));
            graph.begin(); graph.gemv(0, weights, input, output, columns, rows); static_cast<void>(graph.submit());
            check(graph, output, std::vector<double>(rows, 0));
            graph.release(weights); graph.release(input); graph.release(output); graph.configure_small_gemv(0);
            require(graph.memory().live_bytes == initial, "split GEMV scratch release leaked backing");
        }
        graph.configure_small_gemv(16); static_cast<void>(graph.configure_profiling(1));
        const std::vector<float> ones(1024, 1);
        const auto pair_weights = graph.upload(std::as_bytes(std::span(ones)), "profile split weights");
        const auto pair_input = graph.upload(std::as_bytes(std::span(ones)), "profile split input");
        const auto pair_output = graph.zeros(4, "profile split output");
        graph.begin(); reject([&] { graph.gemv(0, pair_weights, pair_input, pair_output, 1024, 1); }); graph.abort();
        check(graph, pair_output, std::array<double, 1>{0});
        require(graph.last_profile().kernels.empty(), "refused split profile exposed partial samples");
        static_cast<void>(graph.configure_profiling(8)); graph.begin(); graph.gemv(0, pair_weights, pair_input, pair_output, 1024, 1);
        require(graph.submit().dispatches == 2 && graph.last_profile().kernels.size() == 2, "split profiler missed a physical pass");
        check(graph, pair_output, std::array<double, 1>{1024});
        graph.release(pair_weights); graph.release(pair_input); graph.release(pair_output);
        static_cast<void>(graph.configure_profiling(0)); graph.configure_small_gemv(0);
        require(graph.memory().live_bytes == initial, "combined experiment resources leaked");
        MetalGraph limited(argv[1], 16384); const auto low = limited.memory().live_bytes;
        reject([&] { limited.configure_small_gemv(16); });
        require(limited.memory().live_bytes == low, "failed split scratch allocation changed ledger");
        std::cout << "Owned split GEMV checks passed: " << checks << ", max scaled error: " << worst << '\n';
    } catch (const std::exception& e) { std::cerr << "Owned split GEMV failed: " << e.what() << '\n'; return 1; }
}
