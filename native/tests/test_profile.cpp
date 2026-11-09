#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>

namespace {
size_t checks = 0;
void require(bool v, const char* why) { ++checks; if (!v) throw std::runtime_error(why); }
void reject(const std::function<void()>& f) { ++checks; try { f(); } catch (const tokspur::FormatError&) { return; } throw std::runtime_error("invalid profiler operation accepted"); }
double checked_timings(const tokspur::GraphProfile& profile) {
    const auto mismatch = profile.cpu_clock_mismatch_fraction;
    const bool calibrated = std::isfinite(mismatch) && mismatch >= 0 && mismatch <= 0.03;
    require(profile.timings_valid == calibrated, "profile calibration status differs from independent clock bound");
    double maximum = 0, sum = 0;
    for (const auto& timing : profile.kernels) {
        require(timing.start && timing.end >= timing.start, "invalid resolved dispatch timestamp");
        if (profile.timings_valid) {
            require(std::isfinite(timing.milliseconds) && timing.milliseconds >= 0, "valid profile has unusable dispatch milliseconds");
            maximum = std::max(maximum, timing.milliseconds); sum += timing.milliseconds;
        } else require(std::isnan(timing.milliseconds), "uncalibrated profile exposed dispatch milliseconds");
    }
    if (profile.timings_valid) {
        require(std::isfinite(profile.gpu_tick_nanoseconds) && profile.gpu_tick_nanoseconds > 0 &&
            std::isfinite(profile.sampled_milliseconds) && profile.sampled_milliseconds >= 0 &&
            profile.sampled_milliseconds == sum, "valid profile clock scale/sum is invalid");
    } else require(std::isnan(profile.gpu_tick_nanoseconds) && std::isnan(profile.sampled_milliseconds), "uncalibrated profile exposed scale/sum");
    return maximum;
}
}
int main(int argc, char** argv) {
    try {
        using namespace tokspur;
        if (argc != 2) throw std::runtime_error("expected owned Metal library");
        MetalGraph graph(argv[1], 128 * 1024 * 1024);
        constexpr uint32_t columns = 4096, rows = 2048;
        std::vector<float> matrix(columns * rows), x(columns), ones(rows, 1);
        for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = float(int((i * 13 + 5) % 251) - 125) / 32768;
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(int((i * 17 + 3) % 257) - 128) / 128;
        const auto weights = graph.upload(std::as_bytes(std::span(matrix)), "profile fixture weights");
        const auto input = graph.upload(std::as_bytes(std::span(x)), "profile fixture input"), gain = graph.upload(std::as_bytes(std::span(ones)), "profile fixture gain");
        const auto output = graph.zeros(rows * 4, "profile fixture output"), norm = graph.zeros(rows * 4, "profile fixture norm");
        const auto base = graph.memory().live_bytes;
        const auto encode = [&] {
            graph.gemv(0, weights, input, output, columns, rows);
            graph.activate(output, output, Activation::silu);
            graph.multiply(output, gain, output);
            graph.rms(output, gain, norm, 64, 1e-6f);
        };
        graph.begin(); encode(); static_cast<void>(graph.submit()); const auto expected = graph.read(norm);
        require(graph.last_profile().kernels.empty(), "disabled profiler recorded data");
        // Exercise the complete target's chunked capacity, not just a tiny fixture.
        const auto mode = graph.configure_profiling(4096);
        require(mode != ProfileMode::disabled && graph.memory().profile_counter_payload_bytes > 0, "timestamp profiling did not enable");
        double maximum = 0;
        for (uint32_t repeat = 0; repeat < 8; ++repeat) {
            const auto allocations = graph.memory().allocations;
            graph.begin(); reject([&] { static_cast<void>(graph.last_profile()); }); reject([&] { static_cast<void>(graph.configure_profiling(0)); });
            encode(); const auto batch = graph.submit(); const auto& profile = graph.last_profile();
            require(batch.dispatches == 4 && profile.kernels.size() == 4 && profile.mode == mode, "profile lost/mislabelled a dispatch");
            require(graph.memory().allocations == allocations && graph.read(norm) == expected, "profiling allocated during encoding or changed outputs");
            require(profile.kernels[0].kernel == "graph_quantized_gemv" && profile.kernels[0].shape[0] == columns && profile.kernels[0].shape[1] == rows, "profile matrix shape metadata differs");
            // A tiny batch need not calibrate within 3%; that must never turn
            // successful GPU work into a failed run or usable bad measurements.
            maximum = std::max(maximum, checked_timings(profile));
        }
        graph.begin(); encode();
        for (uint32_t i = 0; i < 2050; ++i) graph.activate(norm, norm, Activation::identity);
        const auto crossing = graph.submit();
        require(crossing.dispatches == 2054 && graph.last_profile().kernels.size() == 2054 && graph.read(norm) == expected,
                "counter-buffer boundary lost samples or changed output");
        require(graph.last_profile().kernels[2048].kernel == "graph_activate", "second counter chunk lost metadata");
        maximum = std::max(maximum, checked_timings(graph.last_profile()));
        reject([&] { static_cast<void>(graph.configure_profiling(16385)); });
        const auto before = graph.read(output); static_cast<void>(graph.configure_profiling(1));
        graph.begin(); graph.gemv(0, weights, input, output, columns, rows); reject([&] { graph.activate(output, output, Activation::silu); }); graph.abort();
        require(graph.read(output) == before && graph.last_profile().kernels.empty(), "capacity refusal/abort executed or retained stale profile");
        static_cast<void>(graph.configure_profiling(8));
        const std::vector<float> infinity(rows, std::numeric_limits<float>::infinity());
        const auto invalid = graph.upload(std::as_bytes(std::span(infinity)), "profile invalid input");
        graph.begin(); graph.activate(invalid, output, Activation::sigmoid); reject([&] { static_cast<void>(graph.submit()); });
        require(graph.last_profile().kernels.empty(), "device error exposed a stale successful profile");
        graph.begin(); encode(); static_cast<void>(graph.submit()); require(graph.read(norm) == expected, "profiled reuse after device error differs");
        graph.release(invalid); static_cast<void>(graph.configure_profiling(0));
        require(graph.memory().live_bytes == base && graph.memory().profile_counter_payload_bytes == 0, "profiler payload was not released");
        graph.begin(); encode(); static_cast<void>(graph.submit()); require(graph.last_profile().kernels.empty() && graph.read(norm) == expected, "disabled reuse differs");
        MetalGraph limited(argv[1], 16384); const auto low = limited.memory().live_bytes;
        reject([&] { static_cast<void>(limited.configure_profiling(16384)); });
        require(limited.memory().live_bytes == low && limited.last_profile().kernels.empty(), "refused counter allocation changed graph resources");
        std::cout << "Owned GPU profiler checks passed: " << checks << ", mode: " << (mode == ProfileMode::dispatch_boundary ? "dispatch" : "encoder") << ", maximum sampled kernel ms: " << maximum << '\n';
    } catch (const std::exception& e) { std::cerr << "Owned GPU profiler check failed: " << e.what() << '\n'; return 1; }
}
