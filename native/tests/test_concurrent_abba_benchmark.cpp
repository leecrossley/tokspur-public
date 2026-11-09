#include "tokspur/graph.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/json.hpp"
#include "tokspur/numeric.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#ifndef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
#error This benchmark requires the private concurrent-dispatch build.
#endif

namespace {
using namespace tokspur;

std::string sha256_digest(std::span<const std::byte> bytes) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    if (!CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data())) {
        throw std::runtime_error("CC_SHA256 failed");
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (auto b : digest) {
        out += hex[b >> 4];
        out += hex[b & 15];
    }
    return out;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct TrialResult {
    bool concurrent;
    uint64_t dispatches;
    uint64_t barriers;
    double gpu_ms;
    double wall_ms;
    uint32_t thermal_before;
    uint32_t thermal_after;
    std::string sha256;
};

TrialResult run_trial(const char* lib_path, bool concurrent, uint32_t repeats) {
    const uint32_t thermal_before = host_thermal_state();
    require(thermal_before == 0, "Nominal thermal state == 0 required before trial");

    MetalGraph graph(lib_path, 256ull << 20);
    graph.configure_concurrent_dispatch(concurrent);
    graph.configure_small_gemv(16);
    graph.configure_block_gemv(3);

    // Setup input/output buffers
    constexpr uint32_t channels = 2560;
    constexpr uint32_t branches = 4;
    constexpr uint32_t width = channels * branches; // 10240

    std::vector<float> input_data(width);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = float(int((i * 31) % 127) - 63) / 128.0f;
    }
    const auto input = graph.upload(std::as_bytes(std::span(input_data)), "bench input");
    const auto a = graph.zeros(width * sizeof(float), "bench a");
    const auto b = graph.zeros(width * sizeof(float), "bench b");
    const auto c = graph.zeros(width * sizeof(float), "bench c");
    const auto d = graph.zeros(width * sizeof(float), "bench d");
    const auto out = graph.zeros(width * sizeof(float), "bench out");

    graph.begin();
    for (uint32_t r = 0; r < repeats; ++r) {
        graph.activate(input, a, Activation::silu);
        graph.activate(input, b, Activation::silu);
        graph.multiply(a, b, c);
        graph.activate(c, d, Activation::silu);
        graph.multiply(c, d, out);
    }
    const auto res = graph.submit();

    const uint32_t thermal_after = host_thermal_state();
    const auto out_bytes = graph.read(out);
    const auto hash_val = sha256_digest(out_bytes);

    return {
        concurrent,
        res.dispatches,
        graph.concurrent_barriers(),
        res.gpu_ms,
        res.wall_ms,
        thermal_before,
        thermal_after,
        hash_val
    };
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2, "usage: test-concurrent_abba_benchmark METALLIB [CYCLES>=5] [REPEATS>=10]");
        const char* lib_path = argv[1];
        const uint32_t cycles = (argc >= 3) ? std::max(5u, uint32_t(std::atoi(argv[2]))) : 6u;
        const uint32_t repeats = (argc >= 4) ? std::max(1u, uint32_t(std::atoi(argv[3]))) : 100u;

        std::cout << std::setprecision(6) << std::fixed;
        std::cout << "{\n  \"benchmark\": \"concurrent_dispatch_abba\",\n";
        std::cout << "  \"hardware\": \"Apple M5 Max\",\n";
        std::cout << "  \"nominal_thermal_required\": true,\n";
        std::cout << "  \"initial_thermal_state\": " << host_thermal_state() << ",\n";
        std::cout << "  \"cycles\": " << cycles << ",\n";
        std::cout << "  \"repeats_per_trial\": " << repeats << ",\n";

        require(host_thermal_state() == 0, "Initial host thermal state must be 0 (nominal)");

        std::vector<TrialResult> trials_serial;
        std::vector<TrialResult> trials_concurrent;
        trials_serial.reserve(cycles * 2);
        trials_concurrent.reserve(cycles * 2);

        std::cout << "  \"trials\": [\n";
        bool first_trial = true;

        for (uint32_t cycle = 0; cycle < cycles; ++cycle) {
            // Balanced ABBA pattern: even cycles are ABBA, odd cycles are BAAB
            const std::array<bool, 4> arm_order = (cycle % 2 == 0)
                ? std::array<bool, 4>{false, true, true, false}
                : std::array<bool, 4>{true, false, false, true};

            for (uint32_t step = 0; step < 4; ++step) {
                const bool concurrent = arm_order[step];
                const auto trial = run_trial(lib_path, concurrent, repeats);

                if (concurrent) {
                    trials_concurrent.push_back(trial);
                } else {
                    trials_serial.push_back(trial);
                }

                if (!first_trial) std::cout << ",\n";
                first_trial = false;

                std::cout << "    {\n"
                          << "      \"cycle\": " << cycle << ",\n"
                          << "      \"step\": " << step << ",\n"
                          << "      \"arm\": \"" << (concurrent ? "on" : "off") << "\",\n"
                          << "      \"concurrent\": " << (concurrent ? "true" : "false") << ",\n"
                          << "      \"dispatches\": " << trial.dispatches << ",\n"
                          << "      \"barriers\": " << trial.barriers << ",\n"
                          << "      \"gpu_ms\": " << trial.gpu_ms << ",\n"
                          << "      \"wall_ms\": " << trial.wall_ms << ",\n"
                          << "      \"thermal_before\": " << trial.thermal_before << ",\n"
                          << "      \"thermal_after\": " << trial.thermal_after << ",\n"
                          << "      \"output_sha256\": \"" << trial.sha256 << "\"\n"
                          << "    }";
            }
        }
        std::cout << "\n  ],\n";

        // Verification of bit-exactness and barrier reduction
        require(!trials_serial.empty() && !trials_concurrent.empty(), "Empty trial vectors");
        const std::string& canonical_hash = trials_serial[0].sha256;
        for (const auto& t : trials_serial) {
            require(t.sha256 == canonical_hash, "Serial output hash mismatch across trials");
        }
        for (const auto& t : trials_concurrent) {
            require(t.sha256 == canonical_hash, "Concurrent output hash differs from Serial! NMSE > 0");
        }

        // Compute averages
        double total_gpu_serial = 0, total_wall_serial = 0;
        uint64_t barriers_serial = trials_serial[0].barriers;
        for (const auto& t : trials_serial) {
            total_gpu_serial += t.gpu_ms;
            total_wall_serial += t.wall_ms;
        }
        const double avg_gpu_serial = total_gpu_serial / trials_serial.size();
        const double avg_wall_serial = total_wall_serial / trials_serial.size();

        double total_gpu_concurrent = 0, total_wall_concurrent = 0;
        uint64_t barriers_concurrent = trials_concurrent[0].barriers;
        for (const auto& t : trials_concurrent) {
            total_gpu_concurrent += t.gpu_ms;
            total_wall_concurrent += t.wall_ms;
        }
        const double avg_gpu_concurrent = total_gpu_concurrent / trials_concurrent.size();
        const double avg_wall_concurrent = total_wall_concurrent / trials_concurrent.size();

        const double gpu_ratio = avg_gpu_concurrent / avg_gpu_serial;
        const double wall_ratio = avg_wall_concurrent / avg_wall_serial;
        const double barrier_reduction_pct = 100.0 * (1.0 - double(barriers_concurrent) / double(barriers_serial));

        std::cout << "  \"summary\": {\n"
                  << "    \"serial_dispatches\": " << trials_serial[0].dispatches << ",\n"
                  << "    \"serial_barriers\": " << barriers_serial << ",\n"
                  << "    \"serial_avg_gpu_ms\": " << avg_gpu_serial << ",\n"
                  << "    \"serial_avg_wall_ms\": " << avg_wall_serial << ",\n"
                  << "    \"concurrent_dispatches\": " << trials_concurrent[0].dispatches << ",\n"
                  << "    \"concurrent_barriers\": " << barriers_concurrent << ",\n"
                  << "    \"concurrent_avg_gpu_ms\": " << avg_gpu_concurrent << ",\n"
                  << "    \"concurrent_avg_wall_ms\": " << avg_wall_concurrent << ",\n"
                  << "    \"barrier_reduction_percent\": " << barrier_reduction_pct << ",\n"
                  << "    \"gpu_time_ratio\": " << gpu_ratio << ",\n"
                  << "    \"wall_time_ratio\": " << wall_ratio << ",\n"
                  << "    \"bit_exact_equivalence\": true,\n"
                  << "    \"nmse\": 0.0,\n"
                  << "    \"qualification_passed\": " << (barriers_concurrent < barriers_serial && gpu_ratio < 1.0 ? "true" : "false") << "\n"
                  << "  }\n"
                  << "}\n";

        require(barriers_concurrent < barriers_serial, "Barriers were not reduced by concurrent dispatch");
        require(gpu_ratio <= 1.0, "GPU execution did not show speedup");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ABBA benchmark failure: " << error.what() << '\n';
        return 1;
    }
}
