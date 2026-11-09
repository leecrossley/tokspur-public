#include "tokspur/gdn_chunk.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/operators.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
using namespace tokspur;

size_t tests_run = 0;
size_t assertions_passed = 0;
double global_worst_nmse = 0.0;
double global_worst_state_nmse = 0.0;
constexpr float sentinel_guard = -98765.4375f;
constexpr double tier_a_assoc_threshold = 1.0e-7;

void require(bool condition, const char* message) {
    ++assertions_passed;
    if (!condition) {
        std::cerr << "ASSERTION FAILED: " << message << std::endl;
        throw std::runtime_error(message);
    }
}

float pseudo_random_float(size_t index, uint32_t seed = 44) {
    return float(int((index * 131 + seed * 17) % 1993) - 996) / 997.0f;
}

struct MetalArena {
    MetalGraph& g;
    std::vector<GpuView> allocations;

    GpuView upload_floats(std::span<const float> data, bool immutable = true) {
        std::vector<std::byte> buf(data.size_bytes() + 16, std::byte{0x7e});
        std::memcpy(buf.data() + 8, data.data(), data.size_bytes());
        auto view = g.upload(buf, "challenger arena buffer", immutable);
        allocations.push_back(view);
        return g.slice(view, 8, data.size_bytes());
    }

    GpuView allocate_zeros(size_t count) {
        return upload_floats(std::vector<float>(count, 0.0f), false);
    }

    GpuView allocate_sentinel(size_t count) {
        return upload_floats(std::vector<float>(count, sentinel_guard), false);
    }

    void verify_guards() {
        for (auto v : allocations) {
            const auto b = g.read(v);
            for (size_t i = 0; i < 8; ++i) {
                require(b[i] == std::byte{0x7e} && b[b.size() - 1 - i] == std::byte{0x7e},
                        "Arena guard byte corrupted");
            }
        }
    }

    ~MetalArena() {
        try { g.abort(); } catch (const FormatError&) {}
        for (auto v : allocations) g.release(v);
    }
};

std::vector<float> read_floats(MetalGraph& g, GpuView v) {
    const auto b = g.read(v);
    std::vector<float> res(b.size() / sizeof(float));
    std::memcpy(res.data(), b.data(), b.size());
    return res;
}

void emit_gdn_op(MetalGraph& g, GdnChunkOp op, GdnChunkParams p,
                 std::initializer_list<std::pair<uint32_t, GpuView>> bindings) {
    GdnChunkDispatch d{};
    d.op = op;
    d.params = p;
    for (const auto& [slot, view] : bindings) d.buffers.at(slot) = view;
    validate_gdn_chunk_dispatch(d);
    g.gdn_chunk(d);
}

double compute_nmse(std::span<const float> actual, std::span<const double> expected) {
    require(actual.size() == expected.size(), "NMSE size mismatch");
    double diff_sq = 0.0;
    double ref_sq = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "Non-finite actual value detected");
        require(std::isfinite(expected[i]), "Non-finite expected value detected");
        const double diff = double(actual[i]) - expected[i];
        diff_sq += diff * diff;
        ref_sq += expected[i] * expected[i];
    }
    return ref_sq > 0.0 ? (diff_sq / ref_sq) : 0.0;
}

// ---------------------------------------------------------------------------
// TEST SUITE 1: Mathematical Equivalence Across Boundary Sizes
// ---------------------------------------------------------------------------
void test_boundary_row_counts(MetalGraph& g) {
    std::cout << "\n=== Running Test Suite 1: Boundary Row Counts & Forward Substitution ===\n";
    constexpr std::array<uint32_t, 14> boundary_sizes{
        1, 2, 7, 31, 63, 64, 65, 127, 128, 129, 255, 256, 511, 512
    };

    // Use full model head count (48) and dimension (128x128)
    constexpr uint32_t heads = 48;
    constexpr uint32_t dk = 128;
    constexpr uint32_t dv = 128;
    constexpr size_t state_count = size_t(heads) * dk * dv;

    for (uint32_t n : boundary_sizes) {
        ++tests_run;
        MetalArena arena{g, {}};

        const uint32_t q_stride = heads * dk;
        const uint32_t v_stride = heads * dv;
        const uint32_t g_stride = heads;

        std::vector<float> init_state(state_count);
        for (size_t i = 0; i < state_count; ++i) {
            init_state[i] = pseudo_random_float(i) * 0.01f;
        }

        std::vector<float> q(n * q_stride);
        std::vector<float> k(n * q_stride);
        std::vector<float> v(n * v_stride);
        std::vector<float> log_decay(n * g_stride);
        std::vector<float> beta(n * g_stride);

        std::vector<double> ref_s(init_state.begin(), init_state.end());
        std::vector<double> ref_q(q.size()), ref_k(k.size()), ref_v(v.size());
        std::vector<double> ref_logg(log_decay.size()), ref_beta(beta.size());

        for (uint32_t t = 0; t < n; ++t) {
            for (uint32_t h = 0; h < heads; ++h) {
                for (uint32_t a = 0; a < dk; ++a) {
                    const size_t idx = (size_t(t) * heads + h) * dk + a;
                    q[idx] = pseudo_random_float(idx) * 0.01f;
                    k[idx] = pseudo_random_float(idx + 31) * 0.01f;
                    ref_q[idx] = double(q[idx]);
                    ref_k[idx] = double(k[idx]);
                }
                for (uint32_t b = 0; b < dv; ++b) {
                    const size_t idx = (size_t(t) * heads + h) * dv + b;
                    v[idx] = pseudo_random_float(idx + 17) * 0.03f;
                    ref_v[idx] = double(v[idx]);
                }
                const size_t g_idx = size_t(t) * heads + h;
                log_decay[g_idx] = -float((t + h) % 7 + 1) / 97.0f;
                beta[g_idx] = float((t + h) % 13) / 12.0f;
                ref_logg[g_idx] = double(log_decay[g_idx]);
                ref_beta[g_idx] = double(beta[g_idx]);
            }
        }

        // Exact double-precision reference
        const auto expected = gdn_chunk_reference(
            ref_s, ref_q, ref_k, ref_v, ref_logg, ref_beta,
            heads, dk, dv, n, 64
        );

        // Upload to GPU
        const auto accepted_v = arena.upload_floats(init_state);
        const auto q_v = arena.upload_floats(q);
        const auto k_v = arena.upload_floats(k);
        const auto v_v = arena.upload_floats(v);
        const auto logg_v = arena.upload_floats(log_decay);
        const auto beta_v = arena.upload_floats(beta);

        const auto next_v = arena.allocate_sentinel(state_count);
        const auto out_v = arena.allocate_sentinel(n * v_stride);

        const auto prefix_v = arena.allocate_zeros(heads * 64);
        const auto lower_v = arena.allocate_zeros(heads * 64 * 64);
        const auto att_v = arena.allocate_zeros(heads * 64 * 64);
        const auto res_v = arena.allocate_zeros(64 * heads * dv);
        const auto sa_v = arena.allocate_zeros(state_count);
        const auto sb_v = arena.allocate_zeros(state_count);

        GdnChunkParams p{};
        p.rows = n;
        p.heads = heads;
        p.key_width = dk;
        p.value_width = dv;
        p.input_stride = uint64_t(q_stride) * 4;
        p.output_stride = p.auxiliary_stride = uint64_t(v_stride) * 4;
        p.gate_stride = uint64_t(g_stride) * 4;

        g.begin();
        auto current = accepted_v;
        uint32_t chunk = 0;
        for (uint32_t offset = 0; offset < n; offset += 64, ++chunk) {
            p.offset = offset;
            p.count = std::min(64u, n - offset);
            auto next_dst = (offset + p.count == n) ? next_v : (chunk % 2 ? sb_v : sa_v);

            emit_gdn_op(g, GdnChunkOp::prefix64, p, {{0, logg_v}, {1, prefix_v}});
            emit_gdn_op(g, GdnChunkOp::coefficients64, p,
                        {{0, q_v}, {1, k_v}, {2, prefix_v}, {3, beta_v}, {4, lower_v}, {5, att_v}});
            emit_gdn_op(g, GdnChunkOp::solve64, p,
                        {{0, current}, {1, k_v}, {2, v_v}, {3, prefix_v}, {4, beta_v}, {5, lower_v}, {6, res_v}});
            emit_gdn_op(g, GdnChunkOp::emit64, p,
                        {{0, current}, {1, q_v}, {2, k_v}, {3, prefix_v}, {4, att_v}, {5, res_v}, {6, next_dst}, {8, out_v}});
            current = next_dst;
        }
        static_cast<void>(g.submit());

        const auto actual_out = read_floats(g, out_v);
        const auto actual_state = read_floats(g, next_v);

        const double output_nmse = compute_nmse(actual_out, expected.output);
        const double state_nmse = compute_nmse(actual_state, expected.state);

        global_worst_nmse = std::max(global_worst_nmse, output_nmse);
        global_worst_state_nmse = std::max(global_worst_state_nmse, state_nmse);

        std::cout << "  n = " << std::setw(3) << n
                  << " | chunks: " << std::setw(2) << ((n + 63) / 64)
                  << " | output NMSE: " << std::scientific << std::setprecision(4) << output_nmse
                  << " | state NMSE: " << std::scientific << std::setprecision(4) << state_nmse
                  << " | gate: <= 1.0e-7 " << (output_nmse <= tier_a_assoc_threshold && state_nmse <= tier_a_assoc_threshold ? "PASS" : "FAIL")
                  << std::endl;

        require(output_nmse <= tier_a_assoc_threshold, "Output NMSE exceeded Tier A-Assoc limit");
        require(state_nmse <= tier_a_assoc_threshold, "State NMSE exceeded Tier A-Assoc limit");

        // Verify accepted state was not modified
        require(read_floats(g, accepted_v) == init_state, "Accepted state was unexpectedly modified");
        arena.verify_guards();
    }
}

// ---------------------------------------------------------------------------
// TEST SUITE 2: Numerical Extremes & Saturation
// ---------------------------------------------------------------------------
void test_numerical_extremes(MetalGraph& g) {
    std::cout << "\n=== Running Test Suite 2: Numerical Extremes & Saturation ===\n";

    constexpr uint32_t heads = 48;
    constexpr uint32_t dk = 128;
    constexpr uint32_t dv = 128;
    constexpr size_t state_count = size_t(heads) * dk * dv;

    struct ExtremeCase {
        const char* name;
        float log_decay_val;
        float beta_val;
        bool expect_zero_residual;
    };

    const std::vector<ExtremeCase> cases{
        {"Extreme Decay (-1000.0f) + Normal Beta", -1000.0f, 0.5f, false},
        {"Near-Zero Decay (-1.0e-7f) + Normal Beta", -1.0e-7f, 0.5f, false},
        {"Boundary Beta 0.0f (No Input Innovation)", -0.1f, 0.0f, true},
        {"Boundary Beta 1.0f (Max Input Innovation)", -0.1f, 1.0f, false},
        {"Extreme Decay (-1000.0f) + Beta 0.0f", -1000.0f, 0.0f, true},
        {"Extreme Decay (-1000.0f) + Beta 1.0f", -1000.0f, 1.0f, false},
        {"Near-Zero Decay (-1.0e-7f) + Beta 0.0f", -1.0e-7f, 0.0f, true},
        {"Near-Zero Decay (-1.0e-7f) + Beta 1.0f", -1.0e-7f, 1.0f, false},
    };

    const std::vector<uint32_t> test_rows{1, 63, 64, 65, 129, 512};

    for (const auto& tc : cases) {
        std::cout << " Testing: " << tc.name << "\n";
        for (uint32_t n : test_rows) {
            ++tests_run;
            MetalArena arena{g, {}};

            const uint32_t q_stride = heads * dk;
            const uint32_t v_stride = heads * dv;
            const uint32_t g_stride = heads;

            std::vector<float> init_state(state_count);
            for (size_t i = 0; i < state_count; ++i) {
                init_state[i] = pseudo_random_float(i) * 0.01f;
            }

            std::vector<float> q(n * q_stride);
            std::vector<float> k(n * q_stride);
            std::vector<float> v(n * v_stride);
            std::vector<float> log_decay(n * g_stride, tc.log_decay_val);
            std::vector<float> beta(n * g_stride, tc.beta_val);

            std::vector<double> ref_s(init_state.begin(), init_state.end());
            std::vector<double> ref_q(q.size()), ref_k(k.size()), ref_v(v.size());
            std::vector<double> ref_logg(log_decay.begin(), log_decay.end());
            std::vector<double> ref_beta(beta.begin(), beta.end());

            for (uint32_t t = 0; t < n; ++t) {
                for (uint32_t h = 0; h < heads; ++h) {
                    for (uint32_t a = 0; a < dk; ++a) {
                        const size_t idx = (size_t(t) * heads + h) * dk + a;
                        q[idx] = pseudo_random_float(idx) * 0.01f;
                        k[idx] = pseudo_random_float(idx + 31) * 0.01f;
                        ref_q[idx] = double(q[idx]);
                        ref_k[idx] = double(k[idx]);
                    }
                    for (uint32_t b = 0; b < dv; ++b) {
                        const size_t idx = (size_t(t) * heads + h) * dv + b;
                        v[idx] = pseudo_random_float(idx + 17) * 0.03f;
                        ref_v[idx] = double(v[idx]);
                    }
                }
            }

            // Reference
            const auto expected = gdn_chunk_reference(
                ref_s, ref_q, ref_k, ref_v, ref_logg, ref_beta,
                heads, dk, dv, n, 64
            );

            // GPU execution
            const auto accepted_v = arena.upload_floats(init_state);
            const auto q_v = arena.upload_floats(q);
            const auto k_v = arena.upload_floats(k);
            const auto v_v = arena.upload_floats(v);
            const auto logg_v = arena.upload_floats(log_decay);
            const auto beta_v = arena.upload_floats(beta);

            const auto next_v = arena.allocate_sentinel(state_count);
            const auto out_v = arena.allocate_sentinel(n * v_stride);

            const auto prefix_v = arena.allocate_zeros(heads * 64);
            const auto lower_v = arena.allocate_zeros(heads * 64 * 64);
            const auto att_v = arena.allocate_zeros(heads * 64 * 64);
            const auto res_v = arena.allocate_zeros(64 * heads * dv);
            const auto sa_v = arena.allocate_zeros(state_count);
            const auto sb_v = arena.allocate_zeros(state_count);

            GdnChunkParams p{};
            p.rows = n;
            p.heads = heads;
            p.key_width = dk;
            p.value_width = dv;
            p.input_stride = uint64_t(q_stride) * 4;
            p.output_stride = p.auxiliary_stride = uint64_t(v_stride) * 4;
            p.gate_stride = uint64_t(g_stride) * 4;

            g.begin();
            auto current = accepted_v;
            uint32_t chunk = 0;
            for (uint32_t offset = 0; offset < n; offset += 64, ++chunk) {
                p.offset = offset;
                p.count = std::min(64u, n - offset);
                auto next_dst = (offset + p.count == n) ? next_v : (chunk % 2 ? sb_v : sa_v);

                emit_gdn_op(g, GdnChunkOp::prefix64, p, {{0, logg_v}, {1, prefix_v}});
                emit_gdn_op(g, GdnChunkOp::coefficients64, p,
                            {{0, q_v}, {1, k_v}, {2, prefix_v}, {3, beta_v}, {4, lower_v}, {5, att_v}});
                emit_gdn_op(g, GdnChunkOp::solve64, p,
                            {{0, current}, {1, k_v}, {2, v_v}, {3, prefix_v}, {4, beta_v}, {5, lower_v}, {6, res_v}});
                emit_gdn_op(g, GdnChunkOp::emit64, p,
                            {{0, current}, {1, q_v}, {2, k_v}, {3, prefix_v}, {4, att_v}, {5, res_v}, {6, next_dst}, {8, out_v}});
                current = next_dst;
            }
            static_cast<void>(g.submit());

            const auto actual_out = read_floats(g, out_v);
            const auto actual_state = read_floats(g, next_v);

            // Verify absence of NaN / Inf
            for (float val : actual_out) {
                require(std::isfinite(val), "Non-finite float in output activations");
            }
            for (float val : actual_state) {
                require(std::isfinite(val), "Non-finite float in recurrent state");
            }

            const double output_nmse = compute_nmse(actual_out, expected.output);
            const double state_nmse = compute_nmse(actual_state, expected.state);

            global_worst_nmse = std::max(global_worst_nmse, output_nmse);
            global_worst_state_nmse = std::max(global_worst_state_nmse, state_nmse);

            std::cout << "    n=" << std::setw(3) << n
                      << " | out NMSE: " << std::scientific << std::setprecision(3) << output_nmse
                      << " | state NMSE: " << std::scientific << std::setprecision(3) << state_nmse
                      << " | finite: YES"
                      << std::endl;

            require(output_nmse <= tier_a_assoc_threshold, "Extreme condition output NMSE exceeded Tier A-Assoc threshold");
            require(state_nmse <= tier_a_assoc_threshold, "Extreme condition state NMSE exceeded Tier A-Assoc threshold");

            arena.verify_guards();
        }
    }

    // Adversarial invalid inputs: ensure bad_gdn triggers atomic error and graph aborts
    std::cout << " Testing: Adversarial invalid domain detection\n";
    {
        ++tests_run;
        MetalArena arena{g, {}};
        std::vector<float> log_decay(64 * heads, 0.5f); // INVALID: log_decay > 0

        const auto logg_v = arena.upload_floats(log_decay);
        const auto prefix_v = arena.allocate_zeros(heads * 64);

        GdnChunkParams p{};
        p.rows = 64;
        p.heads = heads;
        p.key_width = dk;
        p.value_width = dv;
        p.count = 64;
        p.offset = 0;
        p.gate_stride = heads * 4;

        g.begin();
        emit_gdn_op(g, GdnChunkOp::prefix64, p, {{0, logg_v}, {1, prefix_v}});
        bool threw = false;
        try {
            static_cast<void>(g.submit());
        } catch (const FormatError&) {
            threw = true;
        }
        require(threw, "Positive log_decay must cause graph submission to fail via bad_gdn atomic error");
        try { g.abort(); } catch (...) {}
        std::cout << "    Positive log_decay rejected by atomic error flag: YES\n";
    }

    {
        ++tests_run;
        MetalArena arena{g, {}};
        std::vector<float> q(64 * heads * dk, 0.01f);
        std::vector<float> k(64 * heads * dk, 0.01f);
        std::vector<float> beta(64 * heads, 1.5f); // INVALID: beta > 1

        const auto q_v = arena.upload_floats(q);
        const auto k_v = arena.upload_floats(k);
        const auto beta_v = arena.upload_floats(beta);
        const auto prefix_v = arena.allocate_zeros(heads * 64);
        const auto lower_v = arena.allocate_zeros(heads * 64 * 64);
        const auto att_v = arena.allocate_zeros(heads * 64 * 64);

        GdnChunkParams p{};
        p.rows = 64;
        p.heads = heads;
        p.key_width = dk;
        p.value_width = dv;
        p.count = 64;
        p.offset = 0;
        p.input_stride = heads * dk * 4;
        p.gate_stride = heads * 4;

        g.begin();
        emit_gdn_op(g, GdnChunkOp::coefficients64, p,
                    {{0, q_v}, {1, k_v}, {2, prefix_v}, {3, beta_v}, {4, lower_v}, {5, att_v}});
        bool threw = false;
        try {
            static_cast<void>(g.submit());
        } catch (const FormatError&) {
            threw = true;
        }
        require(threw, "Beta > 1.0 must cause graph submission to fail via bad_gdn atomic error");
        try { g.abort(); } catch (...) {}
        std::cout << "    Beta > 1.0 rejected by atomic error flag: YES\n";
    }

    {
        ++tests_run;
        MetalArena arena{g, {}};
        std::vector<float> q(64 * heads * dk, 0.01f);
        q[13] = std::numeric_limits<float>::quiet_NaN(); // INVALID: NaN
        std::vector<float> k(64 * heads * dk, 0.01f);
        std::vector<float> beta(64 * heads, 0.5f);

        const auto q_v = arena.upload_floats(q);
        const auto k_v = arena.upload_floats(k);
        const auto beta_v = arena.upload_floats(beta);
        const auto prefix_v = arena.allocate_zeros(heads * 64);
        const auto lower_v = arena.allocate_zeros(heads * 64 * 64);
        const auto att_v = arena.allocate_zeros(heads * 64 * 64);

        GdnChunkParams p{};
        p.rows = 64;
        p.heads = heads;
        p.key_width = dk;
        p.value_width = dv;
        p.count = 64;
        p.offset = 0;
        p.input_stride = heads * dk * 4;
        p.gate_stride = heads * 4;

        g.begin();
        emit_gdn_op(g, GdnChunkOp::coefficients64, p,
                    {{0, q_v}, {1, k_v}, {2, prefix_v}, {3, beta_v}, {4, lower_v}, {5, att_v}});
        bool threw = false;
        try {
            static_cast<void>(g.submit());
        } catch (const FormatError&) {
            threw = true;
        }
        require(threw, "NaN in query must cause graph submission to fail via checked_gdn atomic error");
        try { g.abort(); } catch (...) {}
        std::cout << "    NaN in query detected and rejected: YES\n";
    }
}

// ---------------------------------------------------------------------------
// TEST SUITE 3: Memory & Bounds Invariants for Capacity 512 Tokens
// ---------------------------------------------------------------------------
void test_memory_and_bounds(MetalGraph& g) {
    std::cout << "\n=== Running Test Suite 3: Memory & Bounds Invariants (18 Suballocated GpuViews) ===\n";

    GdnShape s{2560, 16, 48, 128, 128, 4};
    const auto bytes_fused = gdn_chunk_bytes(s, 512, GdnChunkMode::fused);
    const auto bytes_cs64 = gdn_chunk_bytes(s, 512, GdnChunkMode::cs64);

    std::cout << " Accounting check for capacity 512 tokens:\n";
    std::cout << "   Fused rows: " << bytes_fused.rows << " bytes\n";
    std::cout << "   CS64 rows: " << bytes_cs64.rows << " bytes (matches fused: "
              << (bytes_cs64.rows == bytes_fused.rows ? "YES" : "NO") << ")\n";
    std::cout << "   CS64 recurrence: " << bytes_cs64.recurrence << " bytes (expected 9449472)\n";
    std::cout << "   CS64 total: " << bytes_cs64.total << " bytes (expected 127283200)\n";

    require(bytes_fused.rows == 117833728, "Fused rows byte mismatch");
    require(bytes_cs64.recurrence == 9449472, "CS64 recurrence byte mismatch");
    require(bytes_cs64.total == 127283200, "CS64 total byte mismatch");

    // Bounds checking on capacity
    ++tests_run;
    bool rejected_0 = false;
    try { static_cast<void>(gdn_chunk_bytes(s, 0, GdnChunkMode::cs64)); } catch (const FormatError&) { rejected_0 = true; }
    require(rejected_0, "capacity 0 must be rejected");

    bool rejected_513 = false;
    try { static_cast<void>(gdn_chunk_bytes(s, 513, GdnChunkMode::cs64)); } catch (const FormatError&) { rejected_513 = true; }
    require(rejected_513, "capacity 513 must be rejected");

    std::cout << " Bounds checking on capacity {0, 513}: REJECTED AS EXPECTED\n";

    // Suballocation Canary Isolation Test:
    // Suballocate 18 GpuViews inside a physical buffer with 64-byte canary zones around each view!
    std::cout << " Canary guard stress test across all 18 GpuViews in arena...\n";
    ++tests_run;

    constexpr size_t num_views = 18;
    constexpr size_t canary_bytes = 64;
    const std::array<uint64_t, 18> expected_counts{
        10240 * 512, // 0: projected
        10240 * 512, // 1: convolved
        6144 * 512,  // 2: query
        6144 * 512,  // 3: key
        6144 * 512,  // 4: value
        6144 * 512,  // 5: z
        48 * 512,    // 6: alpha
        48 * 512,    // 7: beta_logits
        48 * 512,    // 8: log_decay
        48 * 512,    // 9: beta
        6144 * 512,  // 10: raw
        6144 * 512,  // 11: gated
        48 * 64,     // 12: prefix
        48 * 64 * 64,// 13: lower
        48 * 64 * 64,// 14: attention
        64 * 6144,   // 15: residual
        786432,      // 16: state_a
        786432       // 17: state_b
    };

    uint64_t total_view_bytes = 0;
    for (size_t i = 0; i < num_views; ++i) {
        total_view_bytes += expected_counts[i] * 4;
    }
    require(total_view_bytes == 127283200, "View sum must equal 127283200");

    const size_t total_buffer_bytes = total_view_bytes + (num_views + 1) * canary_bytes;
    std::vector<std::byte> arena_storage(total_buffer_bytes, std::byte{0x00});

    // Plant canaries
    for (size_t c = 0; c <= num_views; ++c) {
        size_t canary_offset = c * canary_bytes;
        for (size_t j = 0; j < c; ++j) {
            canary_offset += expected_counts[j] * 4;
        }
        for (size_t b = 0; b < canary_bytes; ++b) {
            arena_storage[canary_offset + b] = std::byte(0xC0 + (c % 16));
        }
    }

    auto master_view = g.upload(arena_storage, "canary arena buffer", false);

    // Slice 18 views between canaries
    std::array<GpuView, num_views> sliced_views{};
    size_t curr_offset = canary_bytes;
    for (size_t i = 0; i < num_views; ++i) {
        const uint64_t v_bytes = expected_counts[i] * 4;
        sliced_views[i] = g.slice(master_view, curr_offset, v_bytes);
        curr_offset += v_bytes + canary_bytes;
    }
    require(curr_offset == total_buffer_bytes, "Canary arena alignment mismatch");

    GdnChunkScratch scratch{
        512,
        sliced_views[0], sliced_views[1], sliced_views[2], sliced_views[3],
        sliced_views[4], sliced_views[5], sliced_views[6], sliced_views[7],
        sliced_views[8], sliced_views[9], sliced_views[10], sliced_views[11],
        sliced_views[12], sliced_views[13], sliced_views[14], sliced_views[15],
        sliced_views[16], sliced_views[17],
        true
    };

    // Now execute CS64 microchunk operations for capacity 512 tokens using scratch views!
    {
        MetalArena temp_arena{g, {}};
        constexpr uint32_t heads = 48;
        constexpr uint32_t dk = 128;
        constexpr uint32_t dv = 128;
        constexpr uint32_t n = 512;
        constexpr size_t state_count = size_t(heads) * dk * dv;

        std::vector<float> init_state(state_count, 0.01f);
        std::vector<float> q(n * heads * dk, 0.01f);
        std::vector<float> k(n * heads * dk, 0.01f);
        std::vector<float> v(n * heads * dv, 0.01f);
        std::vector<float> log_decay(n * heads, -0.05f);
        std::vector<float> beta(n * heads, 0.75f);

        const auto accepted_v = temp_arena.upload_floats(init_state);
        const auto next_v = temp_arena.allocate_sentinel(state_count);
        const auto out_v = temp_arena.allocate_sentinel(n * heads * dv);

        // Copy inputs into scratch views 2, 3, 4, 8, 9
        g.write(scratch.query, std::as_bytes(std::span(q)));
        g.write(scratch.key, std::as_bytes(std::span(k)));
        g.write(scratch.value, std::as_bytes(std::span(v)));
        g.write(scratch.log_decay, std::as_bytes(std::span(log_decay)));
        g.write(scratch.beta, std::as_bytes(std::span(beta)));

        GdnChunkParams p{};
        p.rows = n;
        p.heads = heads;
        p.key_width = dk;
        p.value_width = dv;
        p.input_stride = uint64_t(heads) * dk * 4;
        p.output_stride = p.auxiliary_stride = uint64_t(heads) * dv * 4;
        p.gate_stride = uint64_t(heads) * 4;

        g.begin();
        auto current = accepted_v;
        uint32_t chunk = 0;
        for (uint32_t offset = 0; offset < n; offset += 64, ++chunk) {
            p.offset = offset;
            p.count = std::min(64u, n - offset);
            auto next_dst = (offset + p.count == n) ? next_v : (chunk % 2 ? scratch.state_b : scratch.state_a);

            emit_gdn_op(g, GdnChunkOp::prefix64, p, {{0, scratch.log_decay}, {1, scratch.prefix}});
            emit_gdn_op(g, GdnChunkOp::coefficients64, p,
                        {{0, scratch.query}, {1, scratch.key}, {2, scratch.prefix}, {3, scratch.beta}, {4, scratch.lower}, {5, scratch.attention}});
            emit_gdn_op(g, GdnChunkOp::solve64, p,
                        {{0, current}, {1, scratch.key}, {2, scratch.value}, {3, scratch.prefix}, {4, scratch.beta}, {5, scratch.lower}, {6, scratch.residual}});
            emit_gdn_op(g, GdnChunkOp::emit64, p,
                        {{0, current}, {1, scratch.query}, {2, scratch.key}, {3, scratch.prefix}, {4, scratch.attention}, {5, scratch.residual}, {6, next_dst}, {8, out_v}});
            current = next_dst;
        }
        static_cast<void>(g.submit());

        // Read back entire arena storage and verify all 19 canary zones
        const auto readback_arena = g.read(master_view);
        require(readback_arena.size() == total_buffer_bytes, "Arena readback size mismatch");

        for (size_t c = 0; c <= num_views; ++c) {
            size_t canary_offset = c * canary_bytes;
            for (size_t j = 0; j < c; ++j) {
                canary_offset += expected_counts[j] * 4;
            }
            for (size_t b = 0; b < canary_bytes; ++b) {
                const std::byte expected_byte = std::byte(0xC0 + (c % 16));
                const std::byte actual_byte = readback_arena[canary_offset + b];
                if (actual_byte != expected_byte) {
                    std::cerr << "CANARY CORRUPTION DETECTED at zone " << c
                              << " byte " << b << "! Expected " << int(expected_byte)
                              << " got " << int(actual_byte) << std::endl;
                    require(false, "Canary zone corrupted: out-of-bounds write occurred into suballocated arena");
                }
            }
        }
        std::cout << "   All 19 canary zones (1,216 bytes total) 100% UNTOUCHED! No out-of-bounds writes.\n";
    }

    g.release(master_view);

    // Check dispatch validation on undersized buffers
    std::cout << " Checking dispatch validation on undersized views...\n";
    {
        GdnChunkDispatch d{};
        d.op = GdnChunkOp::prefix64;
        d.params.rows = 64;
        d.params.heads = 48;
        d.params.key_width = 128;
        d.params.value_width = 128;
        d.params.count = 64;
        d.params.gate_stride = 48 * 4;
        d.buffers[0] = {1, 0, 0, 64 * 48 * 4};
        d.buffers[1] = {1, 1, 0, 48 * 64 * 4};
        validate_gdn_chunk_dispatch(d); // Valid

        // Corrupt prefix buffer (under by 1 byte)
        d.buffers[1].bytes--;
        bool rejected = false;
        try { validate_gdn_chunk_dispatch(d); } catch (const FormatError&) { rejected = true; }
        require(rejected, "Undersized prefix buffer must be rejected");

        // Solve64 undersized residual buffer
        d.op = GdnChunkOp::solve64;
        d.params.rows = 64;
        d.params.heads = 48;
        d.params.count = 64;
        d.params.offset = 0;
        d.params.key_width = 128;
        d.params.value_width = 128;
        d.params.input_stride = 48 * 128 * 4;
        d.params.auxiliary_stride = 48 * 128 * 4;
        d.params.gate_stride = 48 * 4;
        d.buffers[0] = {1, 0, 0, 48 * 128 * 128 * 4};
        d.buffers[1] = {1, 1, 0, 64 * 48 * 128 * 4};
        d.buffers[2] = {1, 2, 0, 64 * 48 * 128 * 4};
        d.buffers[3] = {1, 3, 0, 48 * 64 * 4};
        d.buffers[4] = {1, 4, 0, 64 * 48 * 4};
        d.buffers[5] = {1, 5, 0, 48 * 64 * 64 * 4};
        d.buffers[6] = {1, 6, 0, 64 * 48 * 128 * 4};
        validate_gdn_chunk_dispatch(d); // Valid

        d.buffers[6].bytes -= 4; // Under by 1 float
        rejected = false;
        try { validate_gdn_chunk_dispatch(d); } catch (const FormatError&) { rejected = true; }
        require(rejected, "Undersized solve64 residual buffer must be rejected");

        std::cout << "   Dispatch validation correctly rejects undersized buffers: YES\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <tokspur-owned.metallib>\n";
            return 1;
        }

        const char* metallib_path = argv[1];
        std::cout << "=================================================================\n";
        std::cout << " EMPIRICAL CHALLENGER: CS64 WY GDN Chunking Stress Test Suite\n";
        std::cout << " Target Metallib: " << metallib_path << "\n";
        std::cout << "=================================================================\n";

        MetalGraph graph(metallib_path, 1024ull * 1024 * 1024);
        graph.prepare_gdn_chunk();

        const auto mem_start = graph.memory().live_bytes;

        test_boundary_row_counts(graph);
        test_numerical_extremes(graph);
        test_memory_and_bounds(graph);

        const auto mem_end = graph.memory().live_bytes;
        require(mem_start == mem_end, "Memory leak detected in MetalGraph");

        std::cout << "\n=================================================================\n";
        std::cout << " CHALLENGE SUMMARY & VERDICT:\n";
        std::cout << " Tests run: " << tests_run << " suites/scenarios\n";
        std::cout << " Assertions passed: " << assertions_passed << "\n";
        std::cout << " Global worst output NMSE: " << std::scientific << std::setprecision(6) << global_worst_nmse << "\n";
        std::cout << " Global worst state NMSE:  " << std::scientific << std::setprecision(6) << global_worst_state_nmse << "\n";
        std::cout << " Tier A-Assoc limit:       " << std::scientific << std::setprecision(1) << tier_a_assoc_threshold << "\n";
        std::cout << " VERDICT: CONFIRMED (All mathematical, numerical & memory invariants hold)\n";
        std::cout << "=================================================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CHALLENGER FATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
