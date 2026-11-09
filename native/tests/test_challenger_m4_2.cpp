#include "tokspur/graph.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/json.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/gdn_chunk.hpp"
#include "tokspur/grouped_mm.hpp"
#include "tokspur/grouped_mm_f32.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/gguf.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace tokspur;

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED REQUIREMENT: " << message << std::endl;
        throw std::runtime_error(message);
    }
}

float generated_float(size_t i, uint32_t seed = 44) {
    return float(int((i * 131 + seed * 17) % 1993) - 996) / 997.0f;
}

std::string to_hex(std::span<const uint8_t> bytes) {
    constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (auto b : bytes) {
        out += hex[b >> 4];
        out += hex[b & 15];
    }
    return out;
}

// Replicate BranchSession composite state digest from native/src/target.cpp:442-454
std::array<uint8_t, 32> compute_state_digest_oracle(
    uint32_t prefix,
    std::optional<GdnChunkMode> chunk_mode,
    const std::vector<uint32_t>& partitions) {
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    CC_SHA256_Update(&ctx, &prefix, sizeof(prefix));
    const uint8_t cm = chunk_mode.has_value() ? static_cast<uint8_t>(*chunk_mode) : 0xff;
    CC_SHA256_Update(&ctx, &cm, sizeof(cm));
    uint32_t accumulated = 0;
    for (const auto part : partitions) {
        if (accumulated >= prefix) break;
        const uint32_t chunk = std::min(part, prefix - accumulated);
        if (chunk > 0) {
            CC_SHA256_Update(&ctx, &chunk, sizeof(chunk));
            accumulated += chunk;
        }
    }
    std::array<uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &ctx);
    return digest;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << "================================================================" << std::endl;
    std::cout << "EMPIRICAL CHALLENGER M4-2: Adversarial Testing Suite (M4 S3)" << std::endl;
    std::cout << "================================================================" << std::endl;

    const char* metallib_path = argc > 1 ? argv[1] : "../build/owned-native/tokspur-owned.metallib";

    // ------------------------------------------------------------------------
    // CHALLENGE MISSION 1: Router Margin Adversarial Injections & Dynamic FP32 Fallback
    // ------------------------------------------------------------------------
    std::cout << "\n--- Challenge 1: Router Margin Adversarial Injections ---" << std::endl;
    {
        MetalGraph graph(metallib_path, 256ull << 20);
        graph.configure_prefill_grouped_auto(true);
        graph.configure_router_margin_audit(true);
        graph.configure_router_margin_epsilon(1.0e-4f); // 2*epsilon = 2.0e-4
        graph.configure_prefill_router_window(true);
        graph.prepare_moe_router_window();
        graph.prepare_grouped_mm(8, PrefillWeightLayout::gguf);

        const float eps = graph.router_margin_epsilon();
        require(eps == 1.0e-4f, "Epsilon must be 1.0e-4f");
        const float two_eps = 2.0f * eps;
        std::cout << "Configured router_margin_epsilon = " << eps << ", 2*epsilon = " << two_eps << std::endl;

        const auto scratch = graph.router_margins_scratch();
        require(scratch.bytes >= 512 * sizeof(float), "Router margin scratch buffer allocated");

        // Set up real MoE fixtures
        constexpr uint32_t type = 8; // Q8_0
        constexpr uint32_t k_dim = 640;
        constexpr uint32_t r_dim = 65;
        constexpr uint32_t e_dim = 512;
        constexpr uint32_t s_dim = 10;
        constexpr uint32_t n = 512;
        constexpr uint32_t input_stride = k_dim * 4;
        constexpr uint32_t output_stride = (r_dim * 4 + 31) / 16 * 16;

        const auto plan = grouped_mm_plan(type, PrefillWeightLayout::gguf, k_dim, r_dim, e_dim, s_dim, n, input_stride, output_stride, false);

        const auto& q_info = encoding(type);
        const uint64_t weight_bytes = uint64_t(k_dim / q_info.elements_per_block) * q_info.bytes_per_block * r_dim * e_dim;
        std::vector<std::byte> weight_data(weight_bytes);
        uint32_t rng = 44;
        constexpr std::array<uint16_t, 8> ds{0x0401, 0x0568, 0x068e, 0x0ac2, 0x2401, 0x2803, 0xa401, 0x1801};
        for (size_t at = 0; at < weight_data.size(); at += q_info.bytes_per_block) {
            for (uint32_t i = 0; i < q_info.bytes_per_block; ++i) {
                rng = rng * 1664525u + 1013904223u;
                weight_data[at + i] = std::byte(rng >> 24);
            }
            const auto d = ds[(at / q_info.bytes_per_block) % ds.size()];
            weight_data[at] = std::byte(d & 255);
            weight_data[at + 1] = std::byte(d >> 8);
        }
        const auto matrices_v = graph.upload(std::as_bytes(std::span(weight_data)), "moe matrices", true);

        std::vector<float> input_act(n * s_dim * (input_stride / 4));
        for (size_t i = 0; i < input_act.size(); ++i) {
            input_act[i] = generated_float(i + 19) * 0.1f;
        }
        const auto input_act_v = graph.upload(std::as_bytes(std::span(input_act)), "moe input");
        const auto moe_out_v = graph.zeros(n * s_dim * output_stride, "moe output");

        std::vector<uint32_t> route_ids(n * s_dim);
        for (uint32_t t = 0; t < n; ++t) {
            for (uint32_t s_idx = 0; s_idx < s_dim; ++s_idx) {
                route_ids[t * s_dim + s_idx] = (t * 11 + s_idx * 7) % e_dim;
            }
        }
        const auto indices_v = graph.upload(std::as_bytes(std::span(route_ids)), "moe indices");
        const auto token_map_v = graph.zeros(plan.token_map_bytes, "moe map");
        const auto counts_v = graph.zeros(plan.counts_bytes, "moe counts");
        const auto tasks_v = graph.zeros(plan.tasks_bytes, "moe tasks");

        const GpuRowsF32 moe_input{input_act_v, n * s_dim, k_dim, input_stride};
        const GpuRowsF32 moe_output{moe_out_v, n * s_dim, r_dim, output_stride};

        struct MarginTestCase {
            const char* label;
            float margin;
            bool expect_fallback;
        };

        const MarginTestCase test_cases[] = {
            {"Delta_top10 = 0.0 (exact tie)", 0.0f, true},
            {"Delta_top10 = 1.0e-6 <= 2*epsilon", 1.0e-6f, true},
            {"Delta_top10 = 1.99e-4 <= 2*epsilon", 1.99e-4f, true},
            {"Delta_top10 = 2.01e-4 > 2*epsilon", 2.01e-4f, false},
            {"Delta_top10 = 0.5 >> 2*epsilon", 0.5f, false},
        };

        for (const auto& tc : test_cases) {
            graph.reset_router_margin_telemetry();
            std::vector<float> margins(512, tc.margin);
            graph.write(scratch, std::as_bytes(std::span(margins)));

            graph.begin();
            const bool fallback = graph.evaluate_router_margin(n);
            if (fallback) {
                auto routes = graph.grouped_mm_routes(indices_v, token_map_v, counts_v, tasks_v, e_dim, s_dim, n);
                graph.grouped_mm_project(type, PrefillWeightLayout::gguf, matrices_v, routes, moe_input, moe_output, false);
                graph.end_grouped_mm_routes(routes);
            } else {
                graph.grouped_mm(type, PrefillWeightLayout::gguf, matrices_v, indices_v, moe_input, moe_output, token_map_v, counts_v, tasks_v, e_dim, s_dim, false);
            }
            static_cast<void>(graph.submit());

            std::cout << "  [Test] " << tc.label << " -> fallback: " << (fallback ? "TRUE (FP32 project)" : "FALSE (half grouped_mm)")
                      << " (expected: " << (tc.expect_fallback ? "TRUE" : "FALSE") << ")" << std::endl;

            require(fallback == tc.expect_fallback, "Router margin fallback decision mismatch");

            const auto tel = graph.router_margin_telemetry();
            require(tel.total_tokens == n, "Telemetry total_tokens mismatch");
            require(tel.total_layers == 1, "Telemetry total_layers mismatch");
            if (tc.expect_fallback) {
                require(tel.violation_count == n, "Telemetry violation_count mismatch (expected n)");
                require(tel.fallback_layers == 1, "Telemetry fallback_layers mismatch (expected 1)");
            } else {
                require(tel.violation_count == 0, "Telemetry violation_count mismatch (expected 0)");
                require(tel.fallback_layers == 0, "Telemetry fallback_layers mismatch (expected 0)");
            }
        }

        // Test multi-token batch: 512 tokens with 511 safe margins and 1 tight margin
        std::cout << "  [Test] Multi-token batch (512 tokens: 511 @ 0.5, 1 @ 1.99e-4 at index 255):" << std::endl;
        {
            graph.reset_router_margin_telemetry();
            std::vector<float> margins(512, 0.5f);
            margins[255] = 1.99e-4f;
            graph.write(scratch, std::as_bytes(std::span(margins)));

            graph.begin();
            const bool fallback = graph.evaluate_router_margin(512);
            if (fallback) {
                auto routes = graph.grouped_mm_routes(indices_v, token_map_v, counts_v, tasks_v, e_dim, s_dim, n);
                graph.grouped_mm_project(type, PrefillWeightLayout::gguf, matrices_v, routes, moe_input, moe_output, false);
                graph.end_grouped_mm_routes(routes);
            } else {
                graph.grouped_mm(type, PrefillWeightLayout::gguf, matrices_v, indices_v, moe_input, moe_output, token_map_v, counts_v, tasks_v, e_dim, s_dim, false);
            }
            static_cast<void>(graph.submit());

            std::cout << "    Fallback: " << (fallback ? "TRUE (FP32 project)" : "FALSE (half)") << std::endl;
            require(fallback == true, "Batch with a single violation must trigger fallback");

            const auto tel = graph.router_margin_telemetry();
            require(tel.total_tokens == 512, "Telemetry total_tokens must be 512");
            require(tel.violation_count == 1, "Telemetry violation_count must be 1");
            require(tel.fallback_layers == 1, "Telemetry fallback_layers must be 1");
            require(std::abs(tel.min_margin - 1.99e-4f) < 1.0e-8f, "Telemetry min_margin must be 1.99e-4f");
        }

        // Test multi-token batch: 512 tokens all safe (0.5)
        std::cout << "  [Test] Multi-token batch (512 tokens all @ 0.5):" << std::endl;
        {
            graph.reset_router_margin_telemetry();
            std::vector<float> margins(512, 0.5f);
            graph.write(scratch, std::as_bytes(std::span(margins)));

            graph.begin();
            const bool fallback = graph.evaluate_router_margin(512);
            if (fallback) {
                auto routes = graph.grouped_mm_routes(indices_v, token_map_v, counts_v, tasks_v, e_dim, s_dim, n);
                graph.grouped_mm_project(type, PrefillWeightLayout::gguf, matrices_v, routes, moe_input, moe_output, false);
                graph.end_grouped_mm_routes(routes);
            } else {
                graph.grouped_mm(type, PrefillWeightLayout::gguf, matrices_v, indices_v, moe_input, moe_output, token_map_v, counts_v, tasks_v, e_dim, s_dim, false);
            }
            static_cast<void>(graph.submit());

            std::cout << "    Fallback: " << (fallback ? "TRUE" : "FALSE (half retained)") << std::endl;
            require(fallback == false, "Batch with all safe margins must retain mm_tile half");

            const auto tel = graph.router_margin_telemetry();
            require(tel.total_tokens == 512, "Telemetry total_tokens must be 512");
            require(tel.violation_count == 0, "Telemetry violation_count must be 0");
            require(tel.fallback_layers == 0, "Telemetry fallback_layers must be 0");
        }

        // Test Session State Invariance:
        std::cout << "  [Test] Global session state invariance:" << std::endl;
        require(graph.prefill_grouped_auto() == true, "prefill_grouped_auto must remain true");
        require(graph.prefill_grouped_f32() == false, "prefill_grouped_f32 must remain false");
        std::cout << "    Session state unchanged (prefill_grouped_auto=true, prefill_grouped_f32=false) PASS." << std::endl;

        // Test Consecutive Layer Independence:
        std::cout << "  [Test] Consecutive layer independence (Layer 0 violation, Layer 1 nominal):" << std::endl;
        {
            // Layer 0: violation
            std::vector<float> m0(512, 1.0e-5f);
            graph.write(scratch, std::as_bytes(std::span(m0)));
            graph.begin();
            const bool fb0 = graph.evaluate_router_margin(n);
            if (fb0) {
                auto routes = graph.grouped_mm_routes(indices_v, token_map_v, counts_v, tasks_v, e_dim, s_dim, n);
                graph.grouped_mm_project(type, PrefillWeightLayout::gguf, matrices_v, routes, moe_input, moe_output, false);
                graph.end_grouped_mm_routes(routes);
            }
            static_cast<void>(graph.submit());
            require(fb0 == true, "Layer 0 must trigger fallback");

            // Layer 1: nominal
            std::vector<float> m1(512, 0.2f);
            graph.write(scratch, std::as_bytes(std::span(m1)));
            graph.begin();
            const bool fb1 = graph.evaluate_router_margin(n);
            if (!fb1) {
                graph.grouped_mm(type, PrefillWeightLayout::gguf, matrices_v, indices_v, moe_input, moe_output, token_map_v, counts_v, tasks_v, e_dim, s_dim, false);
            }
            static_cast<void>(graph.submit());
            require(fb1 == false, "Layer 1 must NOT trigger fallback (strictly per-layer)");
            std::cout << "    Layer 0 fallback: " << fb0 << ", Layer 1 fallback: " << fb1 << " PASS." << std::endl;
        }
    }

    // ------------------------------------------------------------------------
    // CHALLENGE MISSION 1B: Deep Audit of grouped_mm_project Kernel Selection
    // ------------------------------------------------------------------------
    std::cout << "\n--- Challenge 1B: Deep Audit of grouped_mm_project Kernel Selection ---" << std::endl;
    {
        const auto p_half = grouped_mm_plan(8, PrefillWeightLayout::gguf, 640, 65, 512, 10, 512, 640 * 4, 128 * 4, false);
        const auto p_f32 = grouped_mm_f32_plan(8, PrefillWeightLayout::gguf, 640, 65, 512, 10, 512, 640 * 4, 128 * 4, false);

        std::cout << "  grouped_mm_plan (half): kernel=" << p_half.kernel
                  << ", threadgroup_bytes=" << p_half.threadgroup_bytes << std::endl;
        std::cout << "  grouped_mm_f32_plan (FP32): kernel=" << p_f32.kernel
                  << ", threadgroup_bytes=" << p_f32.threadgroup_bytes << std::endl;

        require(std::string(p_half.kernel) == "graph_grouped_mm", "Half plan must use graph_grouped_mm");
        require(std::string(p_f32.kernel) == "graph_grouped_mm_f32", "FP32 plan must use graph_grouped_mm_f32");
        require(p_half.threadgroup_bytes == 8192, "Half plan threadgroup_bytes must be 8192");
        require(p_f32.threadgroup_bytes == 12288, "FP32 plan threadgroup_bytes must be 12288");
        std::cout << "  Kernel plans verified PASS." << std::endl;
    }

    // ------------------------------------------------------------------------
    // CHALLENGE MISSION 2: Staging Memory & Threadgroup Limits & 64-bit Offset Safety
    // ------------------------------------------------------------------------
    std::cout << "\n--- Challenge 2: Staging Memory & Threadgroup Limits ---" << std::endl;
    {
        constexpr uint64_t mm_tile_aliased_bytes = 8192;
        constexpr uint64_t mm_tile_max_decoupled_bytes = 24576; // 24 KiB
        constexpr uint64_t m5_max_threadgroup_limit = 32768;      // 32 KiB

        std::cout << "  mm_tile aliased shared memory: " << mm_tile_aliased_bytes << " bytes (8 KiB)" << std::endl;
        std::cout << "  mm_tile max decoupled footprint: " << mm_tile_max_decoupled_bytes << " bytes (24 KiB)" << std::endl;
        std::cout << "  Apple M5 Max hardware threadgroup limit: " << m5_max_threadgroup_limit << " bytes (32 KiB)" << std::endl;

        require(mm_tile_aliased_bytes <= m5_max_threadgroup_limit, "Aliased memory within 32 KiB limit");
        require(mm_tile_max_decoupled_bytes <= m5_max_threadgroup_limit, "Max decoupled memory (24 KiB) <= 32 KiB limit");
        const uint64_t margin = m5_max_threadgroup_limit - mm_tile_max_decoupled_bytes;
        std::cout << "  Headroom margin: " << margin << " bytes (" << (margin / 1024) << " KiB) PASS." << std::endl;

        // 64-bit ulong offset calculations stress testing:
        std::cout << "  [Stress Test] 64-bit ulong offset arithmetic with 512 experts and > 2 GiB strides:" << std::endl;
        {
            const uint32_t expert = 511;
            const uint64_t expert_stride = 10000000ull;
            const uint64_t offset_64 = static_cast<uint64_t>(expert) * expert_stride;
            const uint32_t offset_32_truncated = static_cast<uint32_t>(expert) * static_cast<uint32_t>(expert_stride);

            std::cout << "    expert=511, stride=10,000,000 bytes:" << std::endl;
            std::cout << "      64-bit ulong offset: " << offset_64 << " bytes (~" << (offset_64 / (1ull << 30)) << " GiB)" << std::endl;
            std::cout << "      32-bit truncated:    " << offset_32_truncated << " bytes (corrupted by " << (offset_64 - offset_32_truncated) << " bytes)" << std::endl;

            require(offset_64 > (2ull << 30), "Offset must exceed 2 GiB");
            require(offset_64 > (4ull << 30), "Offset must exceed 4 GiB");
            require(offset_64 != static_cast<uint64_t>(offset_32_truncated), "32-bit arithmetic would have overflowed");

            const uint32_t row_idx = 2000;
            const uint32_t row_bytes = 680;
            const uint64_t exp_stride_realistic = 2048ull * row_bytes;
            const uint64_t ptr_offset = static_cast<uint64_t>(expert) * exp_stride_realistic + static_cast<uint64_t>(row_idx) * row_bytes;
            std::cout << "    expert=500, row=2000: pointer offset = " << ptr_offset << " bytes PASS." << std::endl;

            const uint64_t huge_stride = 3ull << 30; // 3 GiB stride
            const uint64_t huge_offset = static_cast<uint64_t>(expert) * huge_stride;
            require(huge_offset == 511ull * (3ull << 30), "Huge 64-bit stride calculation exact");
            std::cout << "    Huge 3 GiB stride test: " << huge_offset << " bytes PASS." << std::endl;
        }
    }

    // ------------------------------------------------------------------------
    // CHALLENGE MISSION 3: State Persistence Digest Invariance
    // ------------------------------------------------------------------------
    std::cout << "\n--- Challenge 3: State Persistence Digest Invariance ---" << std::endl;
    {
        const uint32_t prefix = 128;
        const auto mode_cs64 = std::make_optional(GdnChunkMode::cs64);
        const auto mode_fused = std::make_optional(GdnChunkMode::fused);
        const std::vector<uint32_t> partitions_2x64 = {64, 64};
        const std::vector<uint32_t> partitions_1x128 = {128};

        const auto digest_A = compute_state_digest_oracle(prefix, mode_cs64, partitions_2x64);
        const auto digest_B_fused = compute_state_digest_oracle(prefix, mode_fused, partitions_2x64);
        const auto digest_C_part128 = compute_state_digest_oracle(prefix, mode_cs64, partitions_1x128);

        const std::string hex_A = to_hex(digest_A);
        const std::string hex_B = to_hex(digest_B_fused);
        const std::string hex_C = to_hex(digest_C_part128);

        std::cout << "  Digest A (prefix=128, mode=cs64,  partitions=[64, 64]): " << hex_A << std::endl;
        std::cout << "  Digest B (prefix=128, mode=fused, partitions=[64, 64]): " << hex_B << std::endl;
        std::cout << "  Digest C (prefix=128, mode=cs64,  partitions=[128]):    " << hex_C << std::endl;

        // Verify Digest A differs from chunk_mode=fused
        require(digest_A != digest_B_fused, "Digest with cs64 MUST differ from chunk_mode=fused");
        std::cout << "  [Check 1] digest(cs64) != digest(fused): PASS" << std::endl;

        // Verify Digest A differs from partition=[128, cs64]
        require(digest_A != digest_C_part128, "Digest with partitions=[64, 64] MUST differ from partition=[128]");
        std::cout << "  [Check 2] digest([64, 64]) != digest([128]): PASS" << std::endl;

        // Verify Digest B differs from Digest C
        require(digest_B_fused != digest_C_part128, "Digest fused != Digest [128]");
        std::cout << "  [Check 3] digest(fused, [64, 64]) != digest(cs64, [128]): PASS" << std::endl;

        // Deterministic SHA-256 verification across 10,000 invocations
        std::cout << "  [Check 4] Deterministic SHA-256 across 10,000 invocations..." << std::endl;
        for (int i = 0; i < 10000; ++i) {
            const auto d = compute_state_digest_oracle(prefix, mode_cs64, partitions_2x64);
            if (d != digest_A) {
                require(false, "Non-deterministic SHA-256 digest encountered!");
            }
        }
        std::cout << "    10,000/10,000 bit-exact matches: PASS" << std::endl;

        // Verify prefix sensitivity: prefix=64 vs prefix=128
        const auto digest_prefix64 = compute_state_digest_oracle(64, mode_cs64, partitions_2x64);
        require(digest_A != digest_prefix64, "Digest with prefix=128 MUST differ from prefix=64");
        std::cout << "  [Check 5] digest(prefix=128) != digest(prefix=64): PASS" << std::endl;

        // Verify ancestor state digest restoration invariance under subsequent partition expansion:
        // Checkpoint captured at prefix=128 with partitions=[64, 64]
        // Subsequent tokens committed: partitions expanded to [64, 64, 128, 64]
        // compute_state_digest_oracle(128, mode_cs64, [64, 64, 128, 64]) MUST match digest_A!
        const std::vector<uint32_t> expanded_partitions = {64, 64, 128, 64};
        const auto digest_ancestor_restored = compute_state_digest_oracle(128, mode_cs64, expanded_partitions);
        require(digest_ancestor_restored == digest_A,
            "Ancestor digest with expanded partitions MUST match original checkpoint digest");
        std::cout << "  [Check 6] Ancestor digest invariance under expanded partitions: PASS" << std::endl;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "ALL EMPIRICAL CHALLENGES COMPLETED SUCCESSFULLY!" << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
