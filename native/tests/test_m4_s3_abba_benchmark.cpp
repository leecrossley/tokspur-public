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

float generated_float(size_t i, uint32_t seed = 44) {
    return float(int((i * 131 + seed * 17) % 1993) - 996) / 997.0f;
}

struct TrialResult {
    bool candidate;
    uint64_t dispatches;
    double gpu_ms;
    double wall_ms;
    double gdn_gpu_ms;
    double moe_gpu_ms;
    uint32_t thermal_before;
    uint32_t thermal_after;
    std::string sha256;
    std::string moe_sha256;
    double nmse;
    double gdn_nmse;
    double moe_nmse;
    RouterMarginTelemetry telemetry;
    std::vector<float> output_sample;
    std::vector<float> moe_output_sample;
};

void emit_gdn(MetalGraph& g, GdnChunkOp op, GdnChunkParams p,
              std::initializer_list<std::pair<uint32_t, GpuView>> views) {
    GdnChunkDispatch d{};
    d.op = op;
    d.params = p;
    for (const auto& [idx, v] : views) d.buffers.at(idx) = v;
    validate_gdn_chunk_dispatch(d);
    g.gdn_chunk(d);
}

TrialResult run_trial(const char* lib_path, bool candidate, uint32_t repeats,
                      const std::vector<float>& baseline_reference,
                      const std::vector<float>& baseline_moe_reference) {
    const uint32_t thermal_before = host_thermal_state();
    require(thermal_before == 0, "Nominal thermal state == 0 required before trial");

    MetalGraph graph(lib_path, 512ull << 20);

    if (candidate) {
        graph.configure_prefill_grouped_auto(true);
        graph.configure_router_margin_audit(true);
        graph.configure_router_margin_epsilon(1e-4f);
    } else {
        graph.configure_prefill_grouped_f32(true);
        graph.configure_prefill_grouped_auto(false);
    }

    graph.prepare_gdn_chunk();
    graph.prepare_grouped_mm(8, PrefillWeightLayout::gguf);
    graph.configure_prefill_router_window(true);
    graph.prepare_moe_router_window();

    // Setup GDN Fixtures (512 tokens, 48 heads, 128 key width, 128 value width)
    constexpr uint32_t n = 512;
    constexpr uint32_t heads = 48;
    constexpr uint32_t dk = 128;
    constexpr uint32_t dv = 128;
    constexpr size_t state_count = size_t(heads) * dk * dv;

    std::vector<float> init_state(state_count);
    for (size_t i = 0; i < state_count; ++i) init_state[i] = generated_float(i) * 0.01f;

    std::vector<float> q_data(n * heads * dk);
    std::vector<float> k_data(n * heads * dk);
    std::vector<float> v_data(n * heads * dv);
    std::vector<float> logg_data(n * heads);
    std::vector<float> beta_data(n * heads);

    for (uint32_t t = 0; t < n; ++t) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t j = 0; j < dk; ++j) {
                q_data[(t * heads + h) * dk + j] = generated_float(((size_t(t) * heads + h) * dk + j)) * 0.01f;
                k_data[(t * heads + h) * dk + j] = generated_float(((size_t(t) * heads + h) * dk + j + 31)) * 0.01f;
            }
            for (uint32_t j = 0; j < dv; ++j) {
                v_data[(t * heads + h) * dv + j] = generated_float(((size_t(t) * heads + h) * dv + j + 17)) * 0.03f;
            }
            logg_data[t * heads + h] = -float((t + h) % 7 + 1) / 97.0f;
            beta_data[t * heads + h] = float((t + h) % 13) / 12.0f;
        }
    }

    const auto state_v = graph.upload(std::as_bytes(std::span(init_state)), "gdn init state");
    const auto q_v = graph.upload(std::as_bytes(std::span(q_data)), "gdn q");
    const auto k_v = graph.upload(std::as_bytes(std::span(k_data)), "gdn k");
    const auto v_v = graph.upload(std::as_bytes(std::span(v_data)), "gdn v");
    const auto logg_v = graph.upload(std::as_bytes(std::span(logg_data)), "gdn logg");
    const auto beta_v = graph.upload(std::as_bytes(std::span(beta_data)), "gdn beta");

    const auto next_state_v = graph.zeros(state_count * sizeof(float), "gdn next state");
    const auto gdn_out_v = graph.zeros(n * heads * dv * sizeof(float), "gdn output");

    // Setup CS64 extra scratch buffers if candidate
    GpuView prefix_v{}, lower_v{}, att_v{}, res_v{}, sa_v{}, sb_v{};
    if (candidate) {
        prefix_v = graph.zeros(heads * 64 * sizeof(float), "gdn prefix64");
        lower_v = graph.zeros(heads * 64 * 64 * sizeof(float), "gdn lower64");
        att_v = graph.zeros(heads * 64 * 64 * sizeof(float), "gdn att64");
        res_v = graph.zeros(64 * heads * dv * sizeof(float), "gdn res64");
        sa_v = graph.zeros(state_count * sizeof(float), "gdn sa");
        sb_v = graph.zeros(state_count * sizeof(float), "gdn sb");
    }

    // Setup Grouped mm Fixtures (512 tokens, 512 experts, 10 selected, k=640, r=65)
    constexpr uint32_t type = 8; // Q8_0
    constexpr uint32_t k_dim = 640;
    constexpr uint32_t r_dim = 65;
    constexpr uint32_t e_dim = 512;
    constexpr uint32_t s_dim = 10;
    constexpr uint32_t input_stride = k_dim * 4;
    constexpr uint32_t output_stride = (r_dim * 4 + 31) / 16 * 16;

    const auto plan = candidate
        ? grouped_mm_plan(type, PrefillWeightLayout::gguf, k_dim, r_dim, e_dim, s_dim, n, input_stride, output_stride, false)
        : grouped_mm_f32_plan(type, PrefillWeightLayout::gguf, k_dim, r_dim, e_dim, s_dim, n, input_stride, output_stride, false);

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

    // Execution
    GdnChunkParams gdn_p{};
    gdn_p.rows = n;
    gdn_p.heads = heads;
    gdn_p.key_width = dk;
    gdn_p.value_width = dv;
    gdn_p.input_stride = dk * heads * 4;
    gdn_p.output_stride = gdn_p.auxiliary_stride = dv * heads * 4;
    gdn_p.gate_stride = heads * 4;

    const GpuRowsF32 moe_input{input_act_v, n * s_dim, k_dim, input_stride};
    const GpuRowsF32 moe_output{moe_out_v, n * s_dim, r_dim, output_stride};

    // 1. Benchmark GDN Chunking
    graph.begin();
    for (uint32_t r = 0; r < repeats; ++r) {
        if (!candidate) {
            emit_gdn(graph, GdnChunkOp::fused, gdn_p, {
                {0, state_v}, {1, q_v}, {2, k_v}, {3, v_v}, {4, logg_v}, {5, beta_v}, {6, next_state_v}, {8, gdn_out_v}
            });
        } else {
            auto current = state_v;
            uint32_t chunk = 0;
            for (uint32_t offset = 0; offset < n; offset += 64, ++chunk) {
                gdn_p.offset = offset;
                gdn_p.count = std::min(64u, n - offset);
                auto next_dst = (offset + gdn_p.count == n) ? next_state_v : (chunk % 2 ? sb_v : sa_v);
                emit_gdn(graph, GdnChunkOp::prefix64, gdn_p, {{0, logg_v}, {1, prefix_v}});
                emit_gdn(graph, GdnChunkOp::coefficients64, gdn_p, {{0, q_v}, {1, k_v}, {2, prefix_v}, {3, beta_v}, {4, lower_v}, {5, att_v}});
                emit_gdn(graph, GdnChunkOp::solve64, gdn_p, {{0, current}, {1, k_v}, {2, v_v}, {3, prefix_v}, {4, beta_v}, {5, lower_v}, {6, res_v}});
                emit_gdn(graph, GdnChunkOp::emit64, gdn_p, {{0, current}, {1, q_v}, {2, k_v}, {3, prefix_v}, {4, att_v}, {5, res_v}, {6, next_dst}, {8, gdn_out_v}});
                current = next_dst;
            }
        }
    }
    const auto gdn_res = graph.submit();

    // 2. Benchmark MoE Grouped mm with Router Margin Auditing
    // In candidate mode, populate router margins scratch and audit margin evaluation
    if (candidate) {
        std::vector<float> margins(n, 0.05f); // nominal margin > 2*epsilon
        // On repeat 0, inject a margin violation to verify dynamic fallback
        margins[17] = 1.0e-5f; // <= 2*epsilon (2e-4) -> violation
        graph.write(graph.router_margins_scratch(), std::as_bytes(std::span(margins)));
    }

    graph.begin();
    for (uint32_t r = 0; r < repeats; ++r) {
        bool fallback = false;
        if (candidate && r == 0) {
            fallback = graph.evaluate_router_margin(n);
        }
        if (fallback) {
            // Dynamically fall back to FP32 grouped_mm
            auto routes = graph.grouped_mm_routes(indices_v, token_map_v, counts_v, tasks_v, e_dim, s_dim, n);
            graph.grouped_mm_project(type, PrefillWeightLayout::gguf, matrices_v, routes,
                                        moe_input, moe_output, false);
            graph.end_grouped_mm_routes(routes);
        } else {
            // Primary path: mm_tile 4-simdgroup cooperative GEMM
            graph.grouped_mm(type, PrefillWeightLayout::gguf, matrices_v, indices_v,
                                moe_input, moe_output, token_map_v, counts_v, tasks_v,
                                e_dim, s_dim, false);
        }
    }
    const auto moe_res = graph.submit();

    const double total_gpu_ms = gdn_res.gpu_ms + moe_res.gpu_ms;
    const double total_wall_ms = gdn_res.wall_ms + moe_res.wall_ms;
    const uint64_t total_dispatches = gdn_res.dispatches + moe_res.dispatches;

    const uint32_t thermal_after = host_thermal_state();
    require(thermal_after == 0, "Nominal thermal state must remain 0 after trial");

    // Readback and verification
    const auto out_bytes = graph.read(gdn_out_v);
    const auto hash_val = sha256_digest(out_bytes);

    std::vector<float> float_out(out_bytes.size() / sizeof(float));
    std::memcpy(float_out.data(), out_bytes.data(), out_bytes.size());

    const auto moe_bytes = graph.read(moe_out_v);
    const auto moe_hash_val = sha256_digest(moe_bytes);

    std::vector<float> moe_float_out(moe_bytes.size() / sizeof(float));
    std::memcpy(moe_float_out.data(), moe_bytes.data(), moe_bytes.size());

    double gdn_nmse = 0.0;
    if (!baseline_reference.empty()) {
        require(baseline_reference.size() == float_out.size(), "GDN output tensor size mismatch");
        double diff_sq = 0.0;
        double base_sq = 0.0;
        for (size_t i = 0; i < float_out.size(); ++i) {
            const double diff = double(float_out[i]) - double(baseline_reference[i]);
            diff_sq += diff * diff;
            base_sq += double(baseline_reference[i]) * double(baseline_reference[i]);
        }
        gdn_nmse = base_sq > 0.0 ? (diff_sq / base_sq) : 0.0;
    }

    double moe_nmse = 0.0;
    if (!baseline_moe_reference.empty()) {
        require(baseline_moe_reference.size() == moe_float_out.size(), "MoE output tensor size mismatch");
        double diff_sq = 0.0;
        double base_sq = 0.0;
        for (size_t i = 0; i < moe_float_out.size(); ++i) {
            const double diff = double(moe_float_out[i]) - double(baseline_moe_reference[i]);
            diff_sq += diff * diff;
            base_sq += double(baseline_moe_reference[i]) * double(baseline_moe_reference[i]);
        }
        moe_nmse = base_sq > 0.0 ? (diff_sq / base_sq) : 0.0;
    }

    const double nmse = std::max(gdn_nmse, moe_nmse);

    RouterMarginTelemetry telemetry = graph.router_margin_telemetry();

    return {
        candidate,
        total_dispatches,
        total_gpu_ms,
        total_wall_ms,
        gdn_res.gpu_ms,
        moe_res.gpu_ms,
        thermal_before,
        thermal_after,
        hash_val,
        moe_hash_val,
        nmse,
        gdn_nmse,
        moe_nmse,
        telemetry,
        float_out,
        moe_float_out
    };
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2, "usage: test-m4_s3_abba_benchmark METALLIB [CYCLES>=5] [REPEATS>=1]");
        const char* lib_path = argv[1];
        const uint32_t cycles = (argc >= 3) ? std::max(5u, uint32_t(std::atoi(argv[2]))) : 6u;
        const uint32_t repeats = (argc >= 4) ? std::max(1u, uint32_t(std::atoi(argv[3]))) : 5u;

        require(host_thermal_state() == 0, "Initial host thermal state must be 0 (nominal)");

        std::cout << "{\n  \"benchmark\": \"m4_s3_reassociating_kernels_abba\",\n";
        std::cout << "  \"hardware\": \"Apple M5 Max\",\n";
        std::cout << "  \"configuration\": \"40-core GPU, 128 GB unified memory\",\n";
        std::cout << "  \"nominal_thermal_required\": true,\n";
        std::cout << "  \"initial_thermal_state\": " << host_thermal_state() << ",\n";
        std::cout << "  \"cycles\": " << cycles << ",\n";
        std::cout << "  \"repeats_per_trial\": " << repeats << ",\n";
        std::cout << "  \"numerical_tier\": \"Tier A-Assoc\",\n";
        std::cout << "  \"declared_seed44_break\": true,\n";

        // First, run a baseline trial to establish golden reference tensor for NMSE calculation
        const auto ref_trial = run_trial(lib_path, false, repeats, {}, {});
        const std::vector<float> baseline_ref = ref_trial.output_sample;
        const std::vector<float> baseline_moe_ref = ref_trial.moe_output_sample;

        std::vector<TrialResult> trials_baseline;
        std::vector<TrialResult> trials_candidate;
        trials_baseline.reserve(cycles * 2);
        trials_candidate.reserve(cycles * 2);

        std::cout << "  \"trials\": [\n";
        bool first_trial = true;

        for (uint32_t cycle = 0; cycle < cycles; ++cycle) {
            // Balanced ABBA pattern: even cycles are ABBA, odd cycles are BAAB
            const std::array<bool, 4> arm_order = (cycle % 2 == 0)
                ? std::array<bool, 4>{false, true, true, false}
                : std::array<bool, 4>{true, false, false, true};

            for (uint32_t step = 0; step < 4; ++step) {
                const bool candidate = arm_order[step];
                const auto trial = run_trial(lib_path, candidate, repeats, baseline_ref, baseline_moe_ref);

                if (candidate) {
                    trials_candidate.push_back(trial);
                } else {
                    trials_baseline.push_back(trial);
                }

                if (!first_trial) std::cout << ",\n";
                first_trial = false;

                std::cout << "    {\n"
                          << "      \"cycle\": " << cycle << ",\n"
                          << "      \"step\": " << step << ",\n"
                          << "      \"arm\": \"" << (candidate ? "candidate" : "baseline") << "\",\n"
                          << "      \"candidate\": " << (candidate ? "true" : "false") << ",\n"
                          << "      \"dispatches\": " << trial.dispatches << ",\n"
                          << std::setprecision(6) << std::fixed
                          << "      \"gpu_ms\": " << trial.gpu_ms << ",\n"
                          << "      \"wall_ms\": " << trial.wall_ms << ",\n"
                          << "      \"gdn_gpu_ms\": " << trial.gdn_gpu_ms << ",\n"
                          << "      \"moe_gpu_ms\": " << trial.moe_gpu_ms << ",\n"
                          << "      \"thermal_before\": " << trial.thermal_before << ",\n"
                          << "      \"thermal_after\": " << trial.thermal_after << ",\n"
                          << std::scientific << std::setprecision(8)
                          << "      \"nmse\": " << trial.nmse << ",\n"
                          << "      \"gdn_nmse\": " << trial.gdn_nmse << ",\n"
                          << "      \"moe_nmse\": " << trial.moe_nmse << ",\n"
                          << "      \"output_sha256\": \"" << trial.sha256 << "\",\n"
                          << "      \"moe_sha256\": \"" << trial.moe_sha256 << "\"\n"
                          << "    }";
            }
        }
        std::cout << "\n  ],\n";

        // Summary Statistics
        double total_gpu_baseline = 0.0, total_wall_baseline = 0.0;
        double total_gdn_baseline = 0.0, total_moe_baseline = 0.0;
        for (const auto& t : trials_baseline) {
            total_gpu_baseline += t.gpu_ms;
            total_wall_baseline += t.wall_ms;
            total_gdn_baseline += t.gdn_gpu_ms;
            total_moe_baseline += t.moe_gpu_ms;
        }
        const double avg_gpu_baseline = total_gpu_baseline / trials_baseline.size();
        const double avg_wall_baseline = total_wall_baseline / trials_baseline.size();
        const double avg_gdn_baseline = total_gdn_baseline / trials_baseline.size();
        const double avg_moe_baseline = total_moe_baseline / trials_baseline.size();

        double total_gpu_candidate = 0.0, total_wall_candidate = 0.0;
        double total_gdn_candidate = 0.0, total_moe_candidate = 0.0;
        double max_candidate_nmse = 0.0;
        double max_candidate_gdn_nmse = 0.0;
        double max_candidate_moe_nmse = 0.0;
        for (const auto& t : trials_candidate) {
            total_gpu_candidate += t.gpu_ms;
            total_wall_candidate += t.wall_ms;
            total_gdn_candidate += t.gdn_gpu_ms;
            total_moe_candidate += t.moe_gpu_ms;
            if (t.nmse > max_candidate_nmse) max_candidate_nmse = t.nmse;
            if (t.gdn_nmse > max_candidate_gdn_nmse) max_candidate_gdn_nmse = t.gdn_nmse;
            if (t.moe_nmse > max_candidate_moe_nmse) max_candidate_moe_nmse = t.moe_nmse;
        }
        const double avg_gpu_candidate = total_gpu_candidate / trials_candidate.size();
        const double avg_wall_candidate = total_wall_candidate / trials_candidate.size();
        const double avg_gdn_candidate = total_gdn_candidate / trials_candidate.size();
        const double avg_moe_candidate = total_moe_candidate / trials_candidate.size();

        const double moe_speedup = avg_moe_baseline / avg_moe_candidate;
        const auto final_telemetry = trials_candidate.back().telemetry;

        std::cout << "  \"summary\": {\n"
                  << "    \"baseline_dispatches\": " << trials_baseline[0].dispatches << ",\n"
                  << std::setprecision(6) << std::fixed
                  << "    \"baseline_avg_gpu_ms\": " << avg_gpu_baseline << ",\n"
                  << "    \"baseline_avg_wall_ms\": " << avg_wall_baseline << ",\n"
                  << "    \"baseline_gdn_avg_gpu_ms\": " << avg_gdn_baseline << ",\n"
                  << "    \"baseline_moe_avg_gpu_ms\": " << avg_moe_baseline << ",\n"
                  << "    \"candidate_dispatches\": " << trials_candidate[0].dispatches << ",\n"
                  << "    \"candidate_avg_gpu_ms\": " << avg_gpu_candidate << ",\n"
                  << "    \"candidate_avg_wall_ms\": " << avg_wall_candidate << ",\n"
                  << "    \"candidate_gdn_avg_gpu_ms\": " << avg_gdn_candidate << ",\n"
                  << "    \"candidate_moe_avg_gpu_ms\": " << avg_moe_candidate << ",\n"
                  << "    \"moe_speedup_ratio\": " << moe_speedup << ",\n"
                  << std::scientific << std::setprecision(8)
                  << "    \"max_nmse\": " << max_candidate_nmse << ",\n"
                  << "    \"max_gdn_nmse\": " << max_candidate_gdn_nmse << ",\n"
                  << "    \"max_moe_nmse\": " << max_candidate_moe_nmse << ",\n"
                  << "    \"tier_a_assoc_passed\": " << (max_candidate_nmse <= 1e-7 && max_candidate_gdn_nmse <= 1e-7 && max_candidate_moe_nmse <= 1e-7 ? "true" : "false") << ",\n"
                  << "    \"router_margin_violations_handled\": " << final_telemetry.violation_count << ",\n"
                  << "    \"router_margin_fallback_layers\": " << final_telemetry.fallback_layers << ",\n"
                  << std::setprecision(6) << std::fixed
                  << "    \"router_margin_min\": " << final_telemetry.min_margin << ",\n"
                  << "    \"router_margin_mean\": " << final_telemetry.mean_margin() << ",\n"
                  << "    \"declared_seed44_break\": true,\n"
                  << "    \"qualification_passed\": " << (max_candidate_nmse <= 1e-7 && max_candidate_gdn_nmse <= 1e-7 && max_candidate_moe_nmse <= 1e-7 && moe_speedup >= 1.0 ? "true" : "false") << "\n"
                  << "  }\n"
                  << "}\n";

        require(max_candidate_nmse <= 1e-7, "Tier A-Assoc NMSE exceeded 1e-7 limit");
        require(max_candidate_gdn_nmse <= 1e-7, "GDN NMSE exceeded 1e-7 limit");
        require(max_candidate_moe_nmse <= 1e-7, "MoE NMSE exceeded 1e-7 limit");
        require(moe_speedup >= 1.0, "mm_tile MoE GEMM did not show speedup over FP32");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M4 S3 ABBA benchmark failure: " << error.what() << '\n';
        return 1;
    }
}
