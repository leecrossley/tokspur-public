#include "tokspur/expert_lanes.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/grouped_mm.hpp"
#include "tokspur/verify_nax.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <vector>

using namespace tokspur;

namespace {

void check(bool ok, const char* why) {
    if (!ok) throw FormatError(why);
}

template <class F>
void rejects(F&& action, const char* why) {
    try {
        action();
    } catch (const FormatError&) {
        return;
    }
    throw FormatError(why);
}

uint16_t f32_to_f16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = int32_t((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return uint16_t(sign);
    if (exp >= 31) return uint16_t(sign | 0x7c00u);
    mant = (mant + 0x00001000u) >> 13;
    if (mant == 0x0400u) return uint16_t(sign | uint32_t(exp + 1) << 10);
    return uint16_t(sign | (uint32_t(exp) << 10) | (mant & 0x03ffu));
}

std::vector<std::byte> make_q8_gguf_matrix(uint32_t rows, uint32_t cols, uint32_t seed) {
    check(cols % 32u == 0u, "Q8_0 cols must be multiple of 32");
    const uint32_t blocks_per_row = cols / 32u;
    const size_t row_bytes = size_t(blocks_per_row) * 34u;
    std::vector<std::byte> out(size_t(rows) * row_bytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qdist(-95, 95);
    std::uniform_int_distribution<int> sdist(1, 12);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            std::byte* blk = out.data() + size_t(r) * row_bytes + size_t(b) * 34u;
            const uint16_t d = f32_to_f16_bits(float(sdist(rng)) * (1.0f / 512.0f));
            std::memcpy(blk, &d, sizeof(d));
            for (uint32_t q = 0; q < 32u; ++q) {
                const int8_t v = int8_t(qdist(rng));
                std::memcpy(blk + 2 + q, &v, 1);
            }
        }
    }
    return out;
}

std::vector<float> make_f32_vector(size_t count, uint32_t seed, float scale = 0.25f) {
    std::vector<float> out(count);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    for (float& v : out) {
        v = dist(rng);
        if (!std::isfinite(v) || std::fpclassify(v) == FP_SUBNORMAL) v = 0.125f;
    }
    return out;
}

uint32_t float_bits(float v) {
    uint32_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

void assert_exact_f32(std::span<const float> a, std::span<const float> b, const char* what) {
    check(a.size() == b.size(), what);
    for (size_t i = 0; i < a.size(); ++i) {
        if (float_bits(a[i]) != float_bits(b[i])) {
            std::cerr << what << " mismatch at " << i << ": baseline=" << a[i] << " fused=" << b[i]
                      << " diff=" << std::fabs(a[i] - b[i]) << '\n';
            throw FormatError(what);
        }
    }
}

void run_parity_suite(const char* metallib, PrefillWeightLayout layout, uint32_t small_splits) {
    constexpr ResidualShape rshape{4u, 2560u, 320u};
    constexpr GdnShape gshape{2560u, 16u, 16u, 128u, 128u, 4u};
    constexpr GdnPolicy gpol{0.0883883476f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f};
    constexpr uint32_t stream_width = rshape.branches * rshape.channels; // 10240
    constexpr uint32_t max_steps = 8u;
    constexpr uint32_t test_vocab = 4096u;
    constexpr float eps = 1e-6f;

    const auto down_gguf = make_q8_gguf_matrix(rshape.rank, stream_width, 0x101u);
    const auto up_gguf = make_q8_gguf_matrix(stream_width, rshape.rank, 0x102u);
    const auto out_gguf = make_q8_gguf_matrix(test_vocab, rshape.channels, 0x103u);

    std::vector<std::byte> down_bytes = down_gguf;
    std::vector<std::byte> up_bytes = up_gguf;
    std::vector<std::byte> out_bytes = out_gguf;
    if (layout == PrefillWeightLayout::q8_lanes_v1) {
        const auto dl = lanes::row_layout(8, stream_width);
        const auto ul = lanes::row_layout(8, rshape.rank);
        const auto ol = lanes::row_layout(8, rshape.channels);
        down_bytes.assign(down_gguf.size(), std::byte{0});
        up_bytes.assign(up_gguf.size(), std::byte{0});
        out_bytes.assign(out_gguf.size(), std::byte{0});
        lanes::pack_rows(dl, down_gguf, down_bytes);
        lanes::pack_rows(ul, up_gguf, up_bytes);
        lanes::pack_rows(ol, out_gguf, out_bytes);
    }

    const auto gains_f32 = make_f32_vector(stream_width, 0x201u, 0.5f);
    const auto inject_f32 = make_f32_vector(size_t(rshape.branches) * stream_width, 0x202u, 0.05f);
    const auto streams_f32 = make_f32_vector(size_t(max_steps) * stream_width, 0x203u, 0.75f);
    const auto branch_out_f32 = make_f32_vector(size_t(max_steps) * rshape.channels, 0x204u, 0.5f);
    const auto scales_in_f32 = make_f32_vector(size_t(max_steps) * rshape.branches, 0x205u, 0.4f);

    const auto alpha_w_f32 = make_f32_vector(size_t(gshape.value_heads) * gshape.channels, 0x301u, 0.05f);
    const auto beta_w_f32 = make_f32_vector(size_t(gshape.value_heads) * gshape.channels, 0x302u, 0.05f);
    auto neg_rates_f32 = make_f32_vector(gshape.value_heads, 0x303u, 0.5f);
    for (float& v : neg_rates_f32) v = -std::fabs(v) - 0.05f;
    const auto dt_bias_f32 = make_f32_vector(gshape.value_heads, 0x304u, 0.2f);
    const auto gdn_in_f32 = make_f32_vector(size_t(max_steps) * gshape.channels, 0x305u, 0.6f);

    auto build_and_run = [&](bool fused_enabled, uint32_t steps,
                             std::vector<float>& read_input_out, std::vector<float>& read_scales_out,
                             std::vector<float>& write_dest_out, std::vector<float>& write_rms_dest_out,
                             std::vector<float>& write_rms_norm_out, std::vector<float>& gdn_decay_out,
                             std::vector<float>& gdn_wgate_out, std::vector<float>& final_mixed_out,
                             std::vector<float>& final_logits_out, uint64_t& total_dispatches, double& gpu_ms) {
        MetalGraph g(metallib, 256ull << 20);
        g.configure_block_gemv(1u);
        g.configure_small_gemv(small_splits);
        g.configure_verify_nax(true);
        g.configure_verify_hc_gdn_fused(fused_enabled);
        g.prepare_q8_matvec_many_all();
        g.prepare_gdn_chunk();
        if (small_splits == 16u) g.prepare_hc_rows();
        const std::array<std::pair<uint32_t, PrefillWeightLayout>, 0> no_moe{};
        g.prepare_verify_nax(no_moe);
        if (fused_enabled) g.prepare_verify_hc_gdn_fused();

        const auto w_down = g.upload(down_bytes, "hc.down", true);
        const auto w_up = g.upload(up_bytes, "hc.up", true);
        const auto w_out = g.upload(out_bytes, "final.out", true);
        if (layout == PrefillWeightLayout::q8_lanes_v1) {
            g.register_lane_matrix(w_down, {8u, stream_width, rshape.rank, 1u});
            g.register_lane_matrix(w_up, {8u, rshape.rank, stream_width, 1u});
            g.register_lane_matrix(w_out, {8u, rshape.channels, test_vocab, 1u});
        }
        const auto w_gains = g.upload(std::as_bytes(std::span(gains_f32)), "hc.gains", true);
        const auto w_inject = g.upload(std::as_bytes(std::span(inject_f32)), "hc.inject", true);

        const ResidualWeights rweights{w_gains, GpuMatrix{w_down, 8u}, GpuMatrix{w_up, 8u}, GpuMatrix{w_inject, 0u}};
        const MixerWeights mweights{w_gains, GpuMatrix{w_down, 8u}, GpuMatrix{w_up, 8u}};
        ResidualScratch rscratch{
            g.zeros(uint64_t(max_steps) * stream_width * 4u, "r.norm"),
            g.zeros(uint64_t(max_steps) * rshape.rank * 4u, "r.bot"),
            g.zeros(uint64_t(max_steps) * stream_width * 4u, "r.scores"),
            g.zeros(uint64_t(max_steps) * 16u * rshape.branches * 4u, "r.inject"),
            g.zeros(uint64_t(max_steps) * rshape.channels * 4u, "r.input"),
            g.zeros(uint64_t(max_steps) * rshape.branches * 4u, "r.scales"),
        };

        const auto streams_gpu = g.upload(std::as_bytes(std::span(streams_f32)), "streams", false);
        const auto branch_out_gpu = g.upload(std::as_bytes(std::span(branch_out_f32)), "branch_out", false);
        const auto scales_in_gpu = g.upload(std::as_bytes(std::span(scales_in_f32)), "scales_in", false);
        const auto write_dest_gpu = g.zeros(uint64_t(max_steps) * stream_width * 4u, "write_dest");
        const auto write_rms_streams_gpu = g.upload(std::as_bytes(std::span(streams_f32)), "write_rms_streams", false);
        const auto write_rms_norm_gpu = g.zeros(uint64_t(max_steps) * stream_width * 4u, "write_rms_norm");
        const auto final_mixed_gpu = g.zeros(uint64_t(max_steps) * rshape.channels * 4u, "final_mixed");
        const auto final_logits_gpu = g.zeros(uint64_t(max_steps) * test_vocab * 4u, "final_logits");

        const auto w_alpha = g.upload(std::as_bytes(std::span(alpha_w_f32)), "gdn.alpha", true);
        const auto w_beta = g.upload(std::as_bytes(std::span(beta_w_f32)), "gdn.beta", true);
        const auto w_neg = g.upload(std::as_bytes(std::span(neg_rates_f32)), "gdn.neg", true);
        const auto w_dtb = g.upload(std::as_bytes(std::span(dt_bias_f32)), "gdn.dtb", true);
        const auto gdn_in_gpu = g.upload(std::as_bytes(std::span(gdn_in_f32)), "gdn.in");

        GdnWeights gweights{};
        gweights.alpha = GpuMatrix{w_alpha, 0u};
        gweights.beta = GpuMatrix{w_beta, 0u};
        gweights.negative_rates = w_neg;
        gweights.dt_bias = w_dtb;

        GdnScratch gscratch{};
        gscratch.alpha = g.zeros(uint64_t(max_steps) * gshape.value_heads * 4u, "gdn.s_alpha");
        gscratch.beta = g.zeros(uint64_t(max_steps) * gshape.value_heads * 4u, "gdn.s_beta");
        gscratch.decay = g.zeros(uint64_t(max_steps) * gshape.value_heads * 4u, "gdn.s_decay");
        gscratch.write_gate = g.zeros(uint64_t(max_steps) * gshape.value_heads * 4u, "gdn.s_wgate");

        const uint64_t exp_read = verify_nax_residual_read_dispatches(g, rweights, rshape, steps);
        const uint64_t exp_write = verify_nax_residual_write_dispatches(g, rshape, steps);
        const uint64_t exp_final = verify_nax_final_dispatches(g, mweights, GpuMatrix{w_out, 8u}, rscratch, rshape, test_vocab, steps);
        const uint64_t exp_gates = fused_enabled ? 1ull : 2ull;

        const auto encode_suite = [&]() {
            g.begin();
            {
                VerifyNaxScopeGuard scope(g, true);
                encode_verify_nax_residual_read(g, rweights, rscratch, streams_gpu, rshape, steps, eps);
                encode_verify_nax_residual_write(g, streams_gpu, branch_out_gpu, scales_in_gpu, write_dest_gpu, rshape, steps);
                encode_verify_nax_gdn_f32_gates(g, gweights, gscratch, gdn_in_gpu, gshape, gpol, steps);
                encode_verify_nax_final_logits(g, mweights, GpuMatrix{w_out, 8u}, rscratch, streams_gpu, final_mixed_gpu,
                                               final_logits_gpu, rshape, test_vocab, steps, eps);
            }
            return g.submit();
        };
        const BatchResult single_batch = encode_suite();
        check(single_batch.dispatches == exp_read + exp_write + exp_gates + exp_final,
              "verify_nax dispatch prediction differed from actual submitted dispatches");
        total_dispatches = single_batch.dispatches;

        // Time a realistic 30-sublayer verify command buffer so GPU clocks reach steady state.
        const auto encode_timed_stack = [&]() {
            g.begin();
            {
                VerifyNaxScopeGuard scope(g, true);
                for (uint32_t l = 0; l < 30u; ++l) {
                    encode_verify_nax_residual_read(g, rweights, rscratch, streams_gpu, rshape, steps, eps);
                    encode_verify_nax_gdn_f32_gates(g, gweights, gscratch, gdn_in_gpu, gshape, gpol, steps);
                    encode_verify_nax_residual_write(g, streams_gpu, branch_out_gpu, scales_in_gpu, write_dest_gpu, rshape, steps);
                }
                encode_verify_nax_final_logits(g, mweights, GpuMatrix{w_out, 8u}, rscratch, streams_gpu, final_mixed_gpu,
                                               final_logits_gpu, rshape, test_vocab, steps, eps);
            }
            return g.submit();
        };
        for (uint32_t w = 0; w < 6u; ++w) static_cast<void>(encode_timed_stack());
        double sum_ms = 0.0;
        constexpr uint32_t timed_iters = 16u;
        for (uint32_t it = 0; it < timed_iters; ++it) {
            sum_ms += encode_timed_stack().gpu_ms;
        }
        gpu_ms = sum_ms / timed_iters;

        // Re-run single_batch once before readback so output buffers reflect the single-pass values.
        static_cast<void>(encode_suite());

        // Also test in-place write + RMS fusion against separate per-row residual_write + rms_rows.
        g.begin();
        const auto active_wrms = g.slice(write_rms_streams_gpu, 0, uint64_t(steps) * stream_width * 4u);
        const auto active_wnorm = g.slice(write_rms_norm_gpu, 0, uint64_t(steps) * stream_width * 4u);
        if (fused_enabled) {
            g.verify_fused_hc_write_rms_rows(active_wrms,
                                             g.slice(branch_out_gpu, 0, uint64_t(steps) * rshape.channels * 4u),
                                             g.slice(scales_in_gpu, 0, uint64_t(steps) * rshape.branches * 4u),
                                             w_gains, active_wrms, active_wnorm,
                                             rshape.branches, rshape.channels, steps, eps, false);
        } else {
            for (uint32_t t = 0; t < steps; ++t) {
                const auto s_t = g.slice(write_rms_streams_gpu, uint64_t(t) * stream_width * 4u, stream_width * 4u);
                const auto v_t = g.slice(branch_out_gpu, uint64_t(t) * rshape.channels * 4u, rshape.channels * 4u);
                const auto c_t = g.slice(scales_in_gpu, uint64_t(t) * rshape.branches * 4u, rshape.branches * 4u);
                g.residual_write(s_t, v_t, c_t, s_t);
            }
            g.rms_rows(GpuRowsF32{active_wrms, steps, stream_width, uint64_t(stream_width) * 4u}, w_gains,
                       GpuRowsF32{active_wnorm, steps, stream_width, uint64_t(stream_width) * 4u}, rshape.channels, eps, false);
        }
        static_cast<void>(g.submit());

        read_input_out.resize(size_t(steps) * rshape.channels);
        read_scales_out.resize(size_t(steps) * rshape.branches);
        write_dest_out.resize(size_t(steps) * stream_width);
        write_rms_dest_out.resize(size_t(steps) * stream_width);
        write_rms_norm_out.resize(size_t(steps) * stream_width);
        gdn_decay_out.resize(size_t(steps) * gshape.value_heads);
        gdn_wgate_out.resize(size_t(steps) * gshape.value_heads);
        final_mixed_out.resize(size_t(steps) * rshape.channels);
        final_logits_out.resize(size_t(steps) * test_vocab);

        g.read(g.slice(rscratch.input, 0, read_input_out.size() * 4u), std::as_writable_bytes(std::span(read_input_out)));
        g.read(g.slice(rscratch.scales, 0, read_scales_out.size() * 4u), std::as_writable_bytes(std::span(read_scales_out)));
        g.read(g.slice(write_dest_gpu, 0, write_dest_out.size() * 4u), std::as_writable_bytes(std::span(write_dest_out)));
        g.read(g.slice(write_rms_streams_gpu, 0, write_rms_dest_out.size() * 4u), std::as_writable_bytes(std::span(write_rms_dest_out)));
        g.read(g.slice(write_rms_norm_gpu, 0, write_rms_norm_out.size() * 4u), std::as_writable_bytes(std::span(write_rms_norm_out)));
        g.read(g.slice(gscratch.alpha, 0, gdn_decay_out.size() * 4u), std::as_writable_bytes(std::span(gdn_decay_out)));
        g.read(g.slice(gscratch.beta, 0, gdn_wgate_out.size() * 4u), std::as_writable_bytes(std::span(gdn_wgate_out)));
        g.read(g.slice(final_mixed_gpu, 0, final_mixed_out.size() * 4u), std::as_writable_bytes(std::span(final_mixed_out)));
        g.read(g.slice(final_logits_gpu, 0, final_logits_out.size() * 4u), std::as_writable_bytes(std::span(final_logits_out)));
    };

    for (uint32_t steps : {2u, 3u, 4u, 7u, 8u}) {
        std::vector<float> ref_in, ref_sc, ref_wr, ref_wrd, ref_wrn, ref_dec, ref_wg, ref_mix, ref_log;
        std::vector<float> fus_in, fus_sc, fus_wr, fus_wrd, fus_wrn, fus_dec, fus_wg, fus_mix, fus_log;
        uint64_t ref_disp = 0, fus_disp = 0;
        double ref_ms = 0.0, fus_ms = 0.0;

        build_and_run(false, steps, ref_in, ref_sc, ref_wr, ref_wrd, ref_wrn, ref_dec, ref_wg, ref_mix, ref_log, ref_disp, ref_ms);
        build_and_run(true, steps, fus_in, fus_sc, fus_wr, fus_wrd, fus_wrn, fus_dec, fus_wg, fus_mix, fus_log, fus_disp, fus_ms);

        assert_exact_f32(ref_in, fus_in, "HC residual_read input 0-ULP parity failed");
        assert_exact_f32(ref_sc, fus_sc, "HC residual_read scales 0-ULP parity failed");
        assert_exact_f32(ref_wr, fus_wr, "HC residual_write 0-ULP parity failed");
        assert_exact_f32(ref_wrd, fus_wrd, "HC write_rms_rows destination 0-ULP parity failed");
        assert_exact_f32(ref_wrn, fus_wrn, "HC write_rms_rows normalized 0-ULP parity failed");
        assert_exact_f32(ref_dec, fus_dec, "GDN alpha/decay 0-ULP parity failed");
        assert_exact_f32(ref_wg, fus_wg, "GDN beta/write_gate 0-ULP parity failed");
        assert_exact_f32(ref_mix, fus_mix, "Final mixer output 0-ULP parity failed");
        assert_exact_f32(ref_log, fus_log, "Final logits 0-ULP parity failed");
        check(fus_disp < ref_disp, "fused verify_hc_gdn did not reduce dispatches");

        if (steps == 3u || steps == 4u || steps == 8u) {
            std::cout << "verify_hc_gdn_fused [" << (layout == PrefillWeightLayout::q8_lanes_v1 ? "q8_lanes_v1" : "gguf")
                      << ", split=" << small_splits << ", m=" << steps << "]: 0-ULP exact | 1-layer dispatches "
                      << ref_disp << " -> " << fus_disp << " | 30-sublayer gpu_ms " << ref_ms << " -> " << fus_ms << '\n';
        }
    }
}

void test_sticky_nonfinite_guards(const char* metallib) {
    MetalGraph g(metallib, 64ull << 20);
    g.configure_verify_hc_gdn_fused(true);
    g.prepare_verify_hc_gdn_fused();
    const auto initial_bytes = g.memory().live_bytes;

    // 1. Non-finite input in verify_fused_hc_q8 must trip sticky error_flag.
    auto w_bytes = make_q8_gguf_matrix(320u, 10240u, 0x901u);
    const auto w_gpu = g.upload(w_bytes, "bad_test.w", true);
    auto x_f32 = make_f32_vector(2u * 10240u, 0x902u);
    x_f32[10240u + 77u] = std::numeric_limits<float>::quiet_NaN();
    const auto x_gpu = g.upload(std::as_bytes(std::span(x_f32)), "bad_test.x", false);
    const auto y_gpu = g.zeros(2u * 320u * 4u, "bad_test.y");
    rejects([&] {
        g.begin();
        g.verify_fused_hc_q8(w_gpu, GpuRowsF32{x_gpu, 2u, 10240u, 10240ull * 4u},
                             GpuRowsF32{y_gpu, 2u, 320u, 320ull * 4u}, true, 0.5f);
        static_cast<void>(g.submit());
    }, "verify_fused_hc_q8 accepted NaN input");

    // 2. Non-finite FP32 weight in verify_fused_gdn_alpha_beta_f32 must trip sticky error_flag.
    x_f32[10240u + 77u] = 0.25f;
    g.write(x_gpu, std::as_bytes(std::span(x_f32)));
    auto wa = make_f32_vector(16u * 2560u, 0x903u);
    auto wb = make_f32_vector(16u * 2560u, 0x904u);
    wb[3u * 2560u + 19u] = std::numeric_limits<float>::infinity();
    const auto wa_gpu = g.upload(std::as_bytes(std::span(wa)), "bad_test.wa", true);
    const auto wb_gpu = g.upload(std::as_bytes(std::span(wb)), "bad_test.wb", true);
    const auto oa_gpu = g.zeros(2u * 16u * 4u, "bad_test.oa");
    const auto ob_gpu = g.zeros(2u * 16u * 4u, "bad_test.ob");
    const auto xin_slice = g.slice(x_gpu, 0, 2u * 2560u * 4u);
    rejects([&] {
        g.begin();
        g.verify_fused_gdn_alpha_beta_f32(wa_gpu, wb_gpu, GpuRowsF32{xin_slice, 2u, 2560u, 2560ull * 4u},
                                          GpuRowsF32{oa_gpu, 2u, 16u, 16ull * 4u},
                                          GpuRowsF32{ob_gpu, 2u, 16u, 16ull * 4u});
        static_cast<void>(g.submit());
    }, "verify_fused_gdn_alpha_beta_f32 accepted Inf weight");

    // 3. 0-ULP parity and sticky NaN guards for verify_fused_expert_gate_f32 & verify_fused_shared_logit_f32.
    for (uint32_t sp : {1u, 16u}) {
        g.configure_small_gemv(sp);
        auto w_exp = make_f32_vector(512u * 2560u, 0x905u, 0.05f);
        auto w_sh = make_f32_vector(2560u, 0x906u, 0.05f);
        auto x8 = make_f32_vector(8u * 2560u, 0x907u, 0.5f);
        const auto w_exp_gpu = g.upload(std::as_bytes(std::span(w_exp)), "moe.router", true);
        const auto w_sh_gpu = g.upload(std::as_bytes(std::span(w_sh)), "moe.srouter", true);
        const auto x8_gpu = g.upload(std::as_bytes(std::span(x8)), "moe.x8", false);
        const auto ref_exp_gpu = g.zeros(8u * 512u * 4u, "moe.ref_exp");
        const auto fus_exp_gpu = g.zeros(8u * 512u * 4u, "moe.fus_exp");
        const auto ref_sh_gpu = g.zeros(8u * 4u, "moe.ref_sh");
        const auto fus_sh_gpu = g.zeros(8u * 4u, "moe.fus_sh");
        for (uint32_t m : {2u, 3u, 5u, 8u}) {
            const GpuRowsF32 in_m{g.slice(x8_gpu, 0, uint64_t(m) * 2560u * 4u), m, 2560u, 2560ull * 4u};
            const GpuRowsF32 re_m{g.slice(ref_exp_gpu, 0, uint64_t(m) * 512u * 4u), m, 512u, 512ull * 4u};
            const GpuRowsF32 fe_m{g.slice(fus_exp_gpu, 0, uint64_t(m) * 512u * 4u), m, 512u, 512ull * 4u};
            const GpuRowsF32 rs_m{g.slice(ref_sh_gpu, 0, uint64_t(m) * 4u), m, 1u, 4ull};
            const GpuRowsF32 fs_m{g.slice(fus_sh_gpu, 0, uint64_t(m) * 4u), m, 1u, 4ull};
            g.begin();
            g.gemv_rows(0u, w_exp_gpu, in_m, re_m);
            g.gemv_rows(0u, w_sh_gpu, in_m, rs_m);
            g.verify_fused_expert_gate_f32(w_exp_gpu, in_m, fe_m);
            g.verify_fused_shared_logit_f32(w_sh_gpu, in_m, fs_m);
            static_cast<void>(g.submit());
            std::vector<float> re(size_t(m) * 512u), fe(size_t(m) * 512u), rs(m), fs(m);
            g.read(re_m.view, std::as_writable_bytes(std::span(re)));
            g.read(fe_m.view, std::as_writable_bytes(std::span(fe)));
            g.read(rs_m.view, std::as_writable_bytes(std::span(rs)));
            g.read(fs_m.view, std::as_writable_bytes(std::span(fs)));
            assert_exact_f32(re, fe, "verify_fused_expert_gate_f32 0-ULP parity failed");
            assert_exact_f32(rs, fs, "verify_fused_shared_logit_f32 0-ULP parity failed");
        }
        if (sp == 16u) {
            x8[3u * 2560u + 111u] = std::numeric_limits<float>::quiet_NaN();
            g.write(x8_gpu, std::as_bytes(std::span(x8)));
            rejects([&] {
                g.begin();
                g.verify_fused_expert_gate_f32(w_exp_gpu, GpuRowsF32{x8_gpu, 8u, 2560u, 2560ull * 4u},
                                               GpuRowsF32{fus_exp_gpu, 8u, 512u, 512ull * 4u});
                static_cast<void>(g.submit());
            }, "verify_fused_expert_gate_f32 accepted NaN input");
            rejects([&] {
                g.begin();
                g.verify_fused_shared_logit_f32(w_sh_gpu, GpuRowsF32{x8_gpu, 8u, 2560u, 2560ull * 4u},
                                                GpuRowsF32{fus_sh_gpu, 8u, 1u, 4ull});
                static_cast<void>(g.submit());
            }, "verify_fused_shared_logit_f32 accepted NaN input");
        }
        g.release(fus_sh_gpu);
        g.release(ref_sh_gpu);
        g.release(fus_exp_gpu);
        g.release(ref_exp_gpu);
        g.release(x8_gpu);
        g.release(w_sh_gpu);
        g.release(w_exp_gpu);
    }
    g.configure_small_gemv(0u);

    g.release(ob_gpu);
    g.release(oa_gpu);
    g.release(wb_gpu);
    g.release(wa_gpu);
    g.release(y_gpu);
    g.release(x_gpu);
    g.release(w_gpu);
    check(g.memory().live_bytes == initial_bytes, "test_sticky_nonfinite_guards leaked GPU buffers");
}

} // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 2, "usage: test-verify_hc_gdn_fused METALLIB");
        run_parity_suite(argv[1], PrefillWeightLayout::gguf, 16u);
        run_parity_suite(argv[1], PrefillWeightLayout::q8_lanes_v1, 16u);
        run_parity_suite(argv[1], PrefillWeightLayout::q8_lanes_v1, 1u);
        test_sticky_nonfinite_guards(argv[1]);
        std::cout << "verify_hc_gdn_fused: all layouts, split policies, step widths and non-finite guards passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
