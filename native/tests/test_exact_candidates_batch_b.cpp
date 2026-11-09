#include "tokspur/expert_lanes.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/memory.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/moe_mm_many.hpp"
#include "tokspur/prefill_attention.hpp"
#include "tokspur/prefill_gdn.hpp"
#include "tokspur/prefill_residual.hpp"
#include "tokspur/qsa_prefill.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace tokspur;
uint64_t checks = 0, cases = 0;

void need(bool ok, const char* why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}

void refuses(const std::function<void()>& fn) {
    bool bad = false;
    try { fn(); } catch (const FormatError&) { bad = true; }
    need(bad, "invalid Batch B Graph or memory call admitted");
}

struct Arena {
    MetalGraph& g;
    std::vector<GpuView> owned;
    GpuView bytes(std::span<const std::byte> x, bool immutable = true) {
        std::vector<std::byte> padded(x.size() + 32, std::byte{0x5a});
        std::copy(x.begin(), x.end(), padded.begin() + 16);
        auto v = g.upload(padded, "Batch B exact candidate fixture", immutable);
        owned.push_back(v);
        return g.slice(v, 16, x.size());
    }
    GpuView floats(std::span<const float> x, bool immutable = true) {
        return bytes(std::as_bytes(x), immutable);
    }
    GpuView zero(uint64_t nbytes) {
        std::vector<std::byte> z(nbytes, std::byte{0});
        return bytes(z, false);
    }
    void guards() {
        for (auto v : owned) {
            auto p = g.read(v);
            for (size_t i = 0; i < 16; ++i) {
                need(p[i] == std::byte{0x5a} && p[p.size() - 1 - i] == std::byte{0x5a}, "Batch B canary guard modified");
            }
        }
    }
    ~Arena() {
        try { g.abort(); } catch (const FormatError&) {}
        for (auto v : owned) g.release(v);
    }
};

std::vector<float> make_floats(size_t n, uint32_t seed, float scale = 1.0f / 499.0f) {
    std::vector<float> out(n);
    uint32_t rng = seed;
    for (auto& v : out) {
        rng = rng * 1664525u + 1013904223u;
        v = float(int(rng % 3999) - 1999) * scale;
    }
    return out;
}

std::vector<std::byte> make_bf16(size_t n, uint32_t seed) {
    std::vector<std::byte> out(n * 2);
    uint32_t rng = seed;
    for (size_t i = 0; i < n; ++i) {
        rng = rng * 1664525u + 1013904223u;
        float f = float(int(rng % 2001) - 1000) / 512.0f;
        uint32_t bits = 0;
        std::memcpy(&bits, &f, 4);
        uint16_t bf = uint16_t(bits >> 16);
        std::memcpy(out.data() + i * 2, &bf, 2);
    }
    return out;
}

std::vector<std::byte> make_q8_weights(uint32_t columns, uint32_t rows, uint32_t seed) {
    const uint64_t blocks = uint64_t(columns / 32) * rows;
    std::vector<std::byte> w(blocks * 34);
    uint32_t rng = seed;
    constexpr std::array<uint16_t, 7> scales{0x1801, 0x3555, 0xb555, 0x0001, 0x8001, 0x3c01, 0x8000};
    for (uint64_t b = 0; b < blocks; ++b) {
        const auto d = scales[b % scales.size()];
        std::memcpy(w.data() + b * 34, &d, 2);
        for (size_t j = 0; j < 32; ++j) {
            rng = rng * 1664525u + 1013904223u;
            w[b * 34 + 2 + j] = std::byte(rng >> 24);
        }
    }
    return w;
}

void setup_staged_prefill(MetalGraph& g, bool concurrent = false) {
    g.configure_small_gemv(16);
    g.configure_block_gemv(3);
    g.configure_hc_exact(HcExactMode::combined);
    g.configure_concurrent_dispatch(concurrent);
    g.configure_prefill_dense_f32(true);
    g.configure_prefill_grouped_f32(true);
    g.configure_prefill_dense_staged_f32(true);
    g.configure_prefill_hc_rows(true);
    g.configure_prefill_router_window(true);
    g.configure_prefill_hc_down_window(true);
}

void test_b1_dense_m64_swizzle_multiseg(const char* lib) {
    {
        MetalGraph g(lib, 32ull << 20);
        need(g.prefill_m64_mode() == PrefillM64Mode::off && g.prefill_m64_calls() == 0, "B1 m64 enabled by default");
        need(!g.prefill_multiseg_enabled() && g.prefill_multiseg_calls() == 0, "B1 multiseg enabled by default");
        need(!g.prefill_swizzle_enabled(), "B1 swizzle enabled by default");
    }
    for (bool concurrent : {false, true}) {
        for (bool packed : {false, true}) {
            for (PrefillM64Mode m64 : {PrefillM64Mode::on, PrefillM64Mode::short_only}) {
                MetalGraph g_ref(lib, 256ull << 20);
                MetalGraph g_opt(lib, 256ull << 20);
                setup_staged_prefill(g_ref, concurrent);
                setup_staged_prefill(g_opt, concurrent);
                g_opt.configure_prefill_m64(m64);
                g_opt.configure_prefill_swizzle(true);
                g_opt.configure_prefill_multiseg(true);

                Arena ar{g_ref, {}}, ao{g_opt, {}};
                const uint32_t cols = 256, r0 = 192, r1 = 64, r2 = 128;
                auto w0_raw = make_q8_weights(cols, r0, 701);
                auto w1_raw = make_q8_weights(cols, r1, 702);
                auto w2_raw = make_q8_weights(cols, r2, 703);
                auto w0 = w0_raw, w1 = w1_raw, w2 = w2_raw;
                const auto layout = packed ? PrefillWeightLayout::q8_lanes_v1 : PrefillWeightLayout::gguf;
                if (packed) {
                    lanes::pack_rows(lanes::row_layout(8, cols), w0_raw, w0);
                    lanes::pack_rows(lanes::row_layout(8, cols), w1_raw, w1);
                    lanes::pack_rows(lanes::row_layout(8, cols), w2_raw, w2);
                }
                GpuMatrix mr0{ar.bytes(w0), 8}, mr1{ar.bytes(w1), 8}, mr2{ar.bytes(w2), 8};
                GpuMatrix mo0{ao.bytes(w0), 8}, mo1{ao.bytes(w1), 8}, mo2{ao.bytes(w2), 8};
                if (packed) {
                    g_ref.register_lane_matrix(mr0.view, {8, cols, r0, 1});
                    g_ref.register_lane_matrix(mr1.view, {8, cols, r1, 1});
                    g_ref.register_lane_matrix(mr2.view, {8, cols, r2, 1});
                    g_opt.register_lane_matrix(mo0.view, {8, cols, r0, 1});
                    g_opt.register_lane_matrix(mo1.view, {8, cols, r1, 1});
                    g_opt.register_lane_matrix(mo2.view, {8, cols, r2, 1});
                }
                g_ref.prepare_dense_mm_q8(layout);
                g_opt.prepare_dense_mm_q8(layout);

                for (uint32_t tokens : {2u, 17u, 32u, 33u, 48u, 64u, 96u, 128u}) {
                    const auto x_vals = make_floats(uint64_t(tokens) * cols, 800 + tokens);
                    const std::vector<float> p0(uint64_t(tokens) * r0, -123.5f);
                    const std::vector<float> p1(uint64_t(tokens) * r1, -234.5f);
                    const std::vector<float> p2(uint64_t(tokens) * r2, -345.5f);

                    GpuRowsF32 xr{ar.floats(x_vals, false), tokens, cols, uint64_t(cols) * 4};
                    GpuRowsF32 yr0{ar.floats(p0, false), tokens, r0, uint64_t(r0) * 4};
                    GpuRowsF32 yr1{ar.floats(p1, false), tokens, r1, uint64_t(r1) * 4};
                    GpuRowsF32 yr2{ar.floats(p2, false), tokens, r2, uint64_t(r2) * 4};

                    GpuRowsF32 xo{ao.floats(x_vals, false), tokens, cols, uint64_t(cols) * 4};
                    GpuRowsF32 yo0{ao.floats(p0, false), tokens, r0, uint64_t(r0) * 4};
                    GpuRowsF32 yo1{ao.floats(p1, false), tokens, r1, uint64_t(r1) * 4};
                    GpuRowsF32 yo2{ao.floats(p2, false), tokens, r2, uint64_t(r2) * 4};

                    // Single-matrix m64 + swizzle parity
                    const uint64_t m64_before = g_opt.prefill_m64_calls();
                    g_ref.begin();
                    g_ref.dense_mm_q8(layout, mr0.view, xr, yr0);
                    need(g_ref.submit().dispatches == 1, "baseline dense_mm dispatch count");

                    g_opt.begin();
                    g_opt.dense_mm_q8(layout, mo0.view, xo, yo0);
                    need(g_opt.submit().dispatches == 1, "B1 dense_mm dispatch count");
                    need(g_opt.read(yo0.view) == g_ref.read(yr0.view), "B1 dense_mm m64/swizzle bit mismatch (NMSE != 0)");
                    const bool expect_m64 = m64 == PrefillM64Mode::on ? (tokens >= 33) : (tokens >= 33 && tokens <= 64);
                    need(g_opt.prefill_m64_calls() == m64_before + (expect_m64 ? 1 : 0), "B1 prefill_m64_calls mismatch");

                    // Multiseg-3 parity (3 matrices in 1 dispatch vs 3 dispatches)
                    g_ref.write(yr0.view, std::as_bytes(std::span(p0)));
                    g_opt.write(yo0.view, std::as_bytes(std::span(p0)));

                    g_ref.begin();
                    g_ref.dense_mm_q8(layout, mr0.view, xr, yr0);
                    g_ref.dense_mm_q8(layout, mr1.view, xr, yr1);
                    g_ref.dense_mm_q8(layout, mr2.view, xr, yr2);
                    need(g_ref.submit().dispatches == 3, "baseline 3x dense_mm dispatch count");

                    const uint64_t ms_before = g_opt.prefill_multiseg_calls();
                    g_opt.begin();
                    g_opt.dense_mm_q8_multiseg3(layout, mo0.view, layout, mo1.view, layout, mo2.view, xo, yo0, yo1, yo2);
                    need(g_opt.submit().dispatches == 1, "B1 multiseg3 did not fuse into 1 dispatch");
                    need(g_opt.prefill_multiseg_calls() == ms_before + 1, "B1 multiseg3 call counter mismatch");
                    need(g_opt.read(yo0.view) == g_ref.read(yr0.view), "B1 multiseg3 seg0 bit mismatch");
                    need(g_opt.read(yo1.view) == g_ref.read(yr1.view), "B1 multiseg3 seg1 bit mismatch");
                    need(g_opt.read(yo2.view) == g_ref.read(yr2.view), "B1 multiseg3 seg2 bit mismatch");

                    // Multiseg-2 parity + abort/NaN refusal checks
                    g_opt.write(yo0.view, std::as_bytes(std::span(p0)));
                    g_opt.write(yo1.view, std::as_bytes(std::span(p1)));
                    g_opt.begin();
                    g_opt.dense_mm_q8_multiseg2(layout, mo0.view, layout, mo1.view, xo, yo0, yo1);
                    need(g_opt.submit().dispatches == 1, "B1 multiseg2 did not fuse into 1 dispatch");
                    need(g_opt.read(yo0.view) == g_ref.read(yr0.view), "B1 multiseg2 seg0 bit mismatch");
                    need(g_opt.read(yo1.view) == g_ref.read(yr1.view), "B1 multiseg2 seg1 bit mismatch");

                    if (tokens == 48 && !concurrent && !packed && m64 == PrefillM64Mode::on) {
                        const uint64_t m64_now = g_opt.prefill_m64_calls();
                        const uint64_t ms_now = g_opt.prefill_multiseg_calls();
                        g_opt.begin();
                        g_opt.dense_mm_q8_multiseg2(layout, mo0.view, layout, mo1.view, xo, yo0, yo1);
                        g_opt.abort();
                        need(g_opt.prefill_m64_calls() == m64_now && g_opt.prefill_multiseg_calls() == ms_now,
                            "aborted B1 multiseg published counters");

                        auto bad_x = x_vals;
                        bad_x.back() = std::numeric_limits<float>::quiet_NaN();
                        g_opt.write(xo.view, std::as_bytes(std::span(bad_x)));
                        g_opt.begin();
                        g_opt.dense_mm_q8_multiseg2(layout, mo0.view, layout, mo1.view, xo, yo0, yo1);
                        refuses([&] { static_cast<void>(g_opt.submit()); });
                        need(g_opt.prefill_m64_calls() == m64_now && g_opt.prefill_multiseg_calls() == ms_now,
                            "poisoned B1 multiseg published counters");
                    }
                    ++cases;
                }
                ar.guards();
                ao.guards();
            }
        }
    }
    std::cout << "B1 dense M64 + 2D CTA swizzle + multiseg2/3 bit-exact parity checks passed\n";
}

void test_b2_hc_consumer_and_layers(const char* lib) {
    // 1. Direct encode_prefill_residual_write_and_read bit-exactness
    for (uint32_t tokens : {2u, 19u, 33u, 64u, 128u}) {
        MetalGraph g_ref(lib, 256ull << 20);
        MetalGraph g_opt(lib, 256ull << 20);
        setup_staged_prefill(g_ref);
        setup_staged_prefill(g_opt);
        g_opt.configure_hc_consumer_fused(true);
        need(g_opt.hc_consumer_fused_enabled(), "B2 hc_consumer_fused did not stick");

        Arena ar{g_ref, {}}, ao{g_opt, {}};
        const ResidualShape shape{4, 2560, 320};
        const uint32_t width = shape.branches * shape.channels;
        const auto gains1 = make_floats(width, 901, 1.0f / 2048.0f);
        const auto gains2 = make_floats(width, 902, 1.0f / 2048.0f);
        const auto inj1 = make_floats(uint64_t(width) * 4, 903, 1.0f / 4096.0f);
        const auto inj2 = make_floats(uint64_t(width) * 4, 904, 1.0f / 4096.0f);
        const auto down1 = make_q8_weights(width, shape.rank, 905);
        const auto up1 = make_q8_weights(shape.rank, width, 906);
        const auto down2 = make_q8_weights(width, shape.rank, 907);
        const auto up2 = make_q8_weights(shape.rank, width, 908);

        const ResidualWeights rw1{ar.floats(gains1), {ar.bytes(down1), 8}, {ar.bytes(up1), 8}, {ar.floats(inj1), 0}};
        const ResidualWeights rw2{ar.floats(gains2), {ar.bytes(down2), 8}, {ar.bytes(up2), 8}, {ar.floats(inj2), 0}};
        const ResidualWeights ow1{ao.floats(gains1), {ao.bytes(down1), 8}, {ao.bytes(up1), 8}, {ao.floats(inj1), 0}};
        const ResidualWeights ow2{ao.floats(gains2), {ao.bytes(down2), 8}, {ao.bytes(up2), 8}, {ao.floats(inj2), 0}};

        prepare_prefill_residual(g_ref, rw1, shape);
        prepare_prefill_residual(g_ref, rw2, shape);
        prepare_prefill_residual(g_opt, ow1, shape);
        prepare_prefill_residual(g_opt, ow2, shape);

        auto s_ref = allocate_prefill_residual_scratch(g_ref, shape, 128);
        auto s_opt = allocate_prefill_residual_scratch(g_opt, shape, 128);

        const auto x_vals = make_floats(uint64_t(tokens) * width, 910 + tokens);
        const auto v_vals = make_floats(uint64_t(tokens) * shape.channels, 920 + tokens);
        const std::vector<float> poison(uint64_t(tokens) * width, -999.0f);

        GpuRowsF32 xr{ar.floats(x_vals, false), tokens, width, uint64_t(width) * 4};
        GpuRowsF32 vr{ar.floats(v_vals, false), tokens, shape.channels, uint64_t(shape.channels) * 4};
        GpuRowsF32 mr{ar.floats(poison, false), tokens, width, uint64_t(width) * 4};

        GpuRowsF32 xo{ao.floats(x_vals, false), tokens, width, uint64_t(width) * 4};
        GpuRowsF32 vo{ao.floats(v_vals, false), tokens, shape.channels, uint64_t(shape.channels) * 4};
        GpuRowsF32 mo{ao.floats(poison, false), tokens, width, uint64_t(width) * 4};

        g_ref.begin();
        encode_prefill_residual(g_ref, rw1, s_ref, xr, shape, 1e-6f);
        encode_prefill_residual_write(g_ref, s_ref, xr, vr, mr, shape);
        encode_prefill_residual(g_ref, rw2, s_ref, mr, shape, 1e-6f);
        const auto d_ref = g_ref.submit().dispatches;

        g_opt.begin();
        encode_prefill_residual(g_opt, ow1, s_opt, xo, shape, 1e-6f);
        encode_prefill_residual_write_and_read(g_opt, ow2, s_opt, xo, vo, mo, shape, 1e-6f);
        const auto d_opt = g_opt.submit().dispatches;

        need(d_opt + 1 == d_ref, "B2 fused HC consumer did not save 1 dispatch");
        need(g_opt.hc_consumer_fused_calls() == 1, "B2 hc_consumer_fused_calls mismatch");
        need(g_opt.read(mo.view) == g_ref.read(mr.view), "B2 fused HC consumer mid stream bit mismatch");
        need(g_opt.read(s_opt.rows.input) == g_ref.read(s_ref.rows.input), "B2 fused HC consumer input bit mismatch");
        need(g_opt.read(s_opt.rows.scales) == g_ref.read(s_ref.rows.scales), "B2 fused HC consumer scales bit mismatch");
        need(g_opt.read(s_opt.rows.bottleneck) == g_ref.read(s_ref.rows.bottleneck), "B2 fused HC consumer bottleneck bit mismatch");

        release_prefill_residual_scratch(g_ref, s_ref);
        release_prefill_residual_scratch(g_opt, s_opt);
        ar.guards();
        ao.guards();
        ++cases;
    }

    // 2. Full Attention prefill layer (multiseg3 + qsa_prefill_fused + swizzle + m64 + hc_consumer_fused + moe multiseg2)
    for (uint32_t base : {0u, 3u, 8u}) {
        for (uint32_t rows : {5u, 36u}) {
            MetalGraph g_ref(lib, 512ull << 20);
            MetalGraph g_opt(lib, 512ull << 20);
            setup_staged_prefill(g_ref);
            setup_staged_prefill(g_opt);
            g_opt.configure_prefill_m64(PrefillM64Mode::on);
            g_opt.configure_prefill_multiseg(true);
            g_opt.configure_prefill_swizzle(true);
            g_opt.configure_qsa_prefill_fused(true);
            g_opt.configure_hc_consumer_fused(true);

            Arena ar{g_ref, {}}, ao{g_opt, {}};
            const ResidualShape rshape{4, 2560, 320};
            const QsaShape qshape{2560, 24, 2, 256, 4, 128, 64, 4, 32, 128};
            const MoeShape mshape{2560, 128, 8, 2};
            const AttentionLayerShape lshape{rshape, qshape, mshape};
            const QsaPolicy qpol{QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f};
            const uint32_t rwidth = 10240;

            auto make_res = [&](Arena& a, uint32_t s) {
                return ResidualWeights{
                    a.floats(make_floats(rwidth, s)),
                    {a.bytes(make_q8_weights(rwidth, 320, s + 1)), 8},
                    {a.bytes(make_q8_weights(320, rwidth, s + 2)), 8},
                    {a.floats(make_floats(uint64_t(rwidth) * 4, s + 3, 1.0f / 4096.0f)), 0}
                };
            };
            const auto rq_raw = make_q8_weights(2560, 12288, 1001);
            const auto rk_raw = make_q8_weights(2560, 512, 1002);
            const auto rv_raw = make_q8_weights(2560, 512, 1003);
            const auto ro_raw = make_q8_weights(6144, 2560, 1004);
            const auto iq_raw = make_bf16(2560ull * 512, 1005);
            const auto ik_raw = make_bf16(2560ull * 128, 1006);
            const auto qn = make_floats(256, 1007), kn = make_floats(256, 1008);
            const auto iqn = make_floats(128, 1009), ikn = make_floats(128, 1010);
            const auto freqs = rotary_frequencies(qshape.rotary_width, 10000.0);

            const auto mg_inp = make_floats(2560ull * 8, 1020);
            const auto mge = make_q8_weights(2560, 128 * 8, 1021);
            const auto mue = make_q8_weights(2560, 128 * 8, 1022);
            const auto mde = make_q8_weights(128, 2560 * 8, 1023);
            const auto mgs = make_q8_weights(2560, 128, 1024);
            const auto mus = make_q8_weights(2560, 128, 1025);
            const auto mds = make_q8_weights(128, 2560, 1026);
            const auto mginps = make_floats(2560, 1027);

            auto build_layer = [&](Arena& a) {
                QsaWeights qw{
                    {a.bytes(rq_raw), 8}, {a.bytes(rk_raw), 8}, {a.bytes(rv_raw), 8}, {a.bytes(ro_raw), 8},
                    {a.bytes(iq_raw), 30}, {a.bytes(ik_raw), 30},
                    a.floats(qn), a.floats(kn), a.floats(iqn), a.floats(ikn), a.floats(freqs)
                };
                MoeWeights mw{
                    {a.floats(mg_inp), 0}, {a.bytes(mge), 8}, {a.bytes(mue), 8}, {a.bytes(mde), 8},
                    {a.bytes(mgs), 8}, {a.bytes(mus), 8}, {a.bytes(mds), 8}, {a.floats(mginps), 0}
                };
                return AttentionLayerWeights{make_res(a, 1100), make_res(a, 1200), qw, mw};
            };

            const auto lw_ref = build_layer(ar);
            const auto lw_opt = build_layer(ao);
            const AttentionPrefillLayouts layouts{};
            prepare_attention_prefill(g_ref, lw_ref, lshape, layouts);
            prepare_attention_prefill(g_opt, lw_opt, lshape, layouts);

            auto ls_ref = allocate_prefill_layer_scratch(g_ref, rshape, 64);
            auto ls_opt = allocate_prefill_layer_scratch(g_opt, rshape, 64);
            auto qs_ref = allocate_qsa_prefill_scratch(g_ref, qshape, 64);
            auto qs_opt = allocate_qsa_prefill_scratch(g_opt, qshape, 64);
            auto ms_ref = allocate_moe_mm_many_scratch(g_ref, mshape, 64);
            auto ms_opt = allocate_moe_mm_many_scratch(g_opt, mshape, 64);

            const auto kv_init = make_bf16(uint64_t(qshape.capacity) * qshape.kv_heads * qshape.head_width, 1300 + base);
            const auto v_init = make_bf16(uint64_t(qshape.capacity) * qshape.kv_heads * qshape.head_width, 1301 + base);
            const auto ik_init = make_bf16(uint64_t(qshape.capacity / qshape.compression) * qshape.index_width, 1302 + base);
            const auto sum_init = make_floats(qshape.index_width, 1303 + base);
            const auto streams_init = make_floats(uint64_t(rows) * rwidth, 1304 + rows);
            const std::vector<float> out_poison(uint64_t(rows) * rwidth, -4321.0f);

            QsaCache cref{ar.bytes(kv_init, false), ar.bytes(v_init, false), ar.bytes(ik_init, false)};
            QsaCache copt{ao.bytes(kv_init, false), ao.bytes(v_init, false), ao.bytes(ik_init, false)};
            auto sum_acc_r = ar.floats(sum_init), sum_cand_r = ar.zero(qshape.index_width * 4);
            auto sum_acc_o = ao.floats(sum_init), sum_cand_o = ao.zero(qshape.index_width * 4);
            GpuRowsF32 in_r{ar.floats(streams_init, false), rows, rwidth, uint64_t(rwidth) * 4};
            GpuRowsF32 out_r{ar.floats(out_poison, false), rows, rwidth, uint64_t(rwidth) * 4};
            GpuRowsF32 in_o{ao.floats(streams_init, false), rows, rwidth, uint64_t(rwidth) * 4};
            GpuRowsF32 out_o{ao.floats(out_poison, false), rows, rwidth, uint64_t(rwidth) * 4};

            g_ref.begin();
            encode_attention_prefill(g_ref, lw_ref, ls_ref, qs_ref, ms_ref, cref, sum_acc_r, sum_cand_r,
                in_r, out_r, base, lshape, qpol, RouterMode::softmax, true, layouts);
            const auto d_ref = g_ref.submit().dispatches;
            need(d_ref == attention_prefill_dispatches(g_ref, lw_ref, lshape, base, rows), "baseline attention_prefill_dispatches mismatch");

            g_opt.begin();
            encode_attention_prefill(g_opt, lw_opt, ls_opt, qs_opt, ms_opt, copt, sum_acc_o, sum_cand_o,
                in_o, out_o, base, lshape, qpol, RouterMode::softmax, true, layouts);
            const auto d_opt = g_opt.submit().dispatches;
            need(d_opt == attention_prefill_dispatches(g_opt, lw_opt, lshape, base, rows), "B2 attention_prefill_dispatches mismatch");
            need(d_opt < d_ref, "B2 attention prefill did not reduce dispatch count");

            need(g_opt.read(out_o.view) == g_ref.read(out_r.view), "B2 full attention prefill output bit mismatch (NMSE != 0)");
            need(g_opt.read(copt.keys) == g_ref.read(cref.keys), "B2 full attention prefill KV keys bit mismatch");
            need(g_opt.read(copt.values) == g_ref.read(cref.values), "B2 full attention prefill KV values bit mismatch");
            need(g_opt.read(copt.index_keys) == g_ref.read(cref.index_keys), "B2 full attention prefill index_keys bit mismatch");
            need(g_opt.read(sum_cand_o) == g_ref.read(sum_cand_r), "B2 full attention prefill candidate_sum bit mismatch");

            release_moe_mm_many_scratch(g_ref, ms_ref);
            release_moe_mm_many_scratch(g_opt, ms_opt);
            release_qsa_prefill_scratch(g_ref, qs_ref);
            release_qsa_prefill_scratch(g_opt, qs_opt);
            release_prefill_layer_scratch(g_ref, ls_ref);
            release_prefill_layer_scratch(g_opt, ls_opt);
            ar.guards();
            ao.guards();
            ++cases;
        }
    }

    // 3. Full GDN prefill layer (multiseg2 + swizzle + m64 + hc_consumer_fused + moe multiseg2)
    for (uint32_t rows : {6u, 40u}) {
        MetalGraph g_ref(lib, 512ull << 20);
        MetalGraph g_opt(lib, 512ull << 20);
        setup_staged_prefill(g_ref);
        setup_staged_prefill(g_opt);
        g_opt.configure_prefill_m64(PrefillM64Mode::on);
        g_opt.configure_prefill_multiseg(true);
        g_opt.configure_prefill_swizzle(true);
        g_opt.configure_hc_consumer_fused(true);

        Arena ar{g_ref, {}}, ao{g_opt, {}};
        const ResidualShape rshape{4, 2560, 320};
        const GdnShape gshape{2560, 16, 48, 128, 128, 4};
        const MoeShape mshape{2560, 128, 8, 2};
        const RecurrentShape lshape{rshape, gshape, mshape};
        const GdnPolicy gpol{0.25f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f};
        const uint32_t rwidth = 10240;

        auto make_res = [&](Arena& a, uint32_t s) {
            return ResidualWeights{
                a.floats(make_floats(rwidth, s)),
                {a.bytes(make_q8_weights(rwidth, 320, s + 1)), 8},
                {a.bytes(make_q8_weights(320, rwidth, s + 2)), 8},
                {a.floats(make_floats(uint64_t(rwidth) * 4, s + 3, 1.0f / 4096.0f)), 0}
            };
        };
        const auto qkv_raw = make_q8_weights(2560, 10240, 2001);
        const auto gate_raw = make_q8_weights(2560, 6144, 2002);
        const auto alpha_raw = make_floats(2560ull * 48, 2003, 1.0f / 2048.0f);
        const auto beta_raw = make_floats(2560ull * 48, 2004, 1.0f / 2048.0f);
        const auto out_raw = make_q8_weights(6144, 2560, 2005);
        const auto conv_w = make_floats(10240ull * 4, 2006, 1.0f / 2048.0f);
        auto ssm_a = make_floats(48, 2007, 1.0f / 2048.0f);
        for (auto& v : ssm_a) v = -std::abs(v) - 0.001f;
        const auto ssm_dt = make_floats(48, 2008, 1.0f / 2048.0f);
        const auto ssm_norm = make_floats(128, 2009);

        const auto mg_inp = make_floats(2560ull * 8, 2020);
        const auto mge = make_q8_weights(2560, 128 * 8, 2021);
        const auto mue = make_q8_weights(2560, 128 * 8, 2022);
        const auto mde = make_q8_weights(128, 2560 * 8, 2023);
        const auto mgs = make_q8_weights(2560, 128, 2024);
        const auto mus = make_q8_weights(2560, 128, 2025);
        const auto mds = make_q8_weights(128, 2560, 2026);
        const auto mginps = make_floats(2560, 2027);

        auto build_gdn = [&](Arena& a) {
            GdnWeights gw{
                {a.bytes(qkv_raw), 8}, {a.bytes(gate_raw), 8}, {a.floats(alpha_raw), 0}, {a.floats(beta_raw), 0},
                {a.bytes(out_raw), 8}, a.floats(conv_w), a.floats(ssm_a), a.floats(ssm_dt), a.floats(ssm_norm)
            };
            MoeWeights mw{
                {a.floats(mg_inp), 0}, {a.bytes(mge), 8}, {a.bytes(mue), 8}, {a.bytes(mde), 8},
                {a.bytes(mgs), 8}, {a.bytes(mus), 8}, {a.bytes(mds), 8}, {a.floats(mginps), 0}
            };
            return RecurrentWeights{make_res(a, 2100), make_res(a, 2200), gw, mw};
        };

        const auto lw_ref = build_gdn(ar);
        const auto lw_opt = build_gdn(ao);
        const GdnPrefillLayouts layouts{};
        prepare_gdn_prefill_layer(g_ref, lw_ref, lshape, layouts, false);
        prepare_gdn_prefill_layer(g_opt, lw_opt, lshape, layouts, false);

        auto ls_ref = allocate_prefill_layer_scratch(g_ref, rshape, 64);
        auto ls_opt = allocate_prefill_layer_scratch(g_opt, rshape, 64);
        auto gs_ref = allocate_gdn_chunk_scratch(g_ref, gshape, 64, GdnChunkMode::fused);
        auto gs_opt = allocate_gdn_chunk_scratch(g_opt, gshape, 64, GdnChunkMode::fused);
        auto ms_ref = allocate_moe_mm_many_scratch(g_ref, mshape, 64);
        auto ms_opt = allocate_moe_mm_many_scratch(g_opt, mshape, 64);

        const auto mat_init = make_floats(48ull * 128 * 128, 2300, 1.0f / 4096.0f);
        const auto conv_init = make_floats(10240ull * 3, 2301, 1.0f / 2048.0f);
        const auto streams_init = make_floats(uint64_t(rows) * rwidth, 2302 + rows);
        const std::vector<float> out_poison(uint64_t(rows) * rwidth, -8765.0f);

        GdnState acc_r{ar.floats(mat_init), ar.floats(conv_init)}, cand_r{ar.zero(mat_init.size() * 4), ar.zero(conv_init.size() * 4)};
        GdnState acc_o{ao.floats(mat_init), ao.floats(conv_init)}, cand_o{ao.zero(mat_init.size() * 4), ao.zero(conv_init.size() * 4)};
        GpuRowsF32 in_r{ar.floats(streams_init, false), rows, rwidth, uint64_t(rwidth) * 4};
        GpuRowsF32 out_r{ar.floats(out_poison, false), rows, rwidth, uint64_t(rwidth) * 4};
        GpuRowsF32 in_o{ao.floats(streams_init, false), rows, rwidth, uint64_t(rwidth) * 4};
        GpuRowsF32 out_o{ao.floats(out_poison, false), rows, rwidth, uint64_t(rwidth) * 4};

        g_ref.begin();
        encode_gdn_prefill_layer(g_ref, lw_ref, ls_ref, gs_ref, ms_ref, acc_r, cand_r, in_r, out_r,
            lshape, gpol, RouterMode::softmax, true, GdnChunkMode::fused, layouts, false);
        const auto d_ref = g_ref.submit().dispatches;
        need(d_ref == gdn_prefill_layer_dispatches(g_ref, lw_ref, lshape, rows, GdnChunkMode::fused), "baseline gdn_prefill_layer_dispatches mismatch");

        g_opt.begin();
        encode_gdn_prefill_layer(g_opt, lw_opt, ls_opt, gs_opt, ms_opt, acc_o, cand_o, in_o, out_o,
            lshape, gpol, RouterMode::softmax, true, GdnChunkMode::fused, layouts, false);
        const auto d_opt = g_opt.submit().dispatches;
        need(d_opt == gdn_prefill_layer_dispatches(g_opt, lw_opt, lshape, rows, GdnChunkMode::fused), "B2 gdn_prefill_layer_dispatches mismatch");
        need(d_opt + 3 == d_ref, "B2 GDN prefill layer did not save 3 dispatches (qkv|gate + hc_consumer + shared_gate|up)");

        need(g_opt.read(out_o.view) == g_ref.read(out_r.view), "B2 full GDN prefill output bit mismatch (NMSE != 0)");
        need(g_opt.read(cand_o.matrix) == g_ref.read(cand_r.matrix), "B2 full GDN prefill state matrix bit mismatch");
        need(g_opt.read(cand_o.convolution) == g_ref.read(cand_r.convolution), "B2 full GDN prefill conv state bit mismatch");

        release_moe_mm_many_scratch(g_ref, ms_ref);
        release_moe_mm_many_scratch(g_opt, ms_opt);
        release_gdn_chunk_scratch(g_ref, gs_ref);
        release_gdn_chunk_scratch(g_opt, gs_opt);
        release_prefill_layer_scratch(g_ref, ls_ref);
        release_prefill_layer_scratch(g_opt, ls_opt);
        ar.guards();
        ao.guards();
        ++cases;
    }
    std::cout << "B2 QSA fused prep + HC consumer fusion + full Attention/GDN layer parity checks passed\n";
}

void test_b3_ple_gather_dedup() {
    const uint32_t width = 160, stored = 520, logical = 500;
    const auto table = make_q8_weights(width, stored, 3001);
    EmbeddingRows ref_cache(table, 8, width, stored, logical, 64 * 1024);
    EmbeddingRows opt_cache(table, 8, width, stored, logical, 64 * 1024);
    PleGatherScratch scratch;

    std::vector<uint64_t> ids(8192);
    uint32_t rng = 4242;
    for (size_t i = 0; i < ids.size(); ++i) {
        rng = rng * 1664525u + 1013904223u;
        // Heavy realistic repetition + scattered misses
        ids[i] = (i % 3 == 0) ? uint64_t(rng % 24) : uint64_t(rng % logical);
    }

    std::vector<float> ref_out(ids.size() * width, -11.0f);
    std::vector<float> dedup_cache_out(ids.size() * width, -22.0f);
    std::vector<float> stateless_out(ids.size() * width, -33.0f);

    for (size_t off = 0; off < ids.size(); off += 16) {
        ref_cache.gather(std::span<const uint64_t>(ids).subspan(off, 16),
            std::span<float>(ref_out).subspan(off * width, 16 * width));
    }
    opt_cache.gather_chunk_dedup(ids, dedup_cache_out, scratch);
    dequantize_rows_sorted_dedup(table, 8, width, stored, logical, ids, stateless_out, scratch);

    need(dedup_cache_out == ref_out, "B3 EmbeddingRows::gather_chunk_dedup bit mismatch (NMSE != 0)");
    need(stateless_out == ref_out, "B3 dequantize_rows_sorted_dedup bit mismatch (NMSE != 0)");
    const auto stats = opt_cache.stats();
    need(stats.hits + stats.misses == ids.size(), "B3 gather_chunk_dedup hit+miss accounting mismatch");
    need(stats.hits > 7000, "B3 gather_chunk_dedup did not deduplicate repeated chunk rows");

    // Out-of-range refusal must leave output and cache stats untouched
    auto bad_ids = ids;
    bad_ids[4095] = logical; // padding row is unaddressable
    const auto saved_out = stateless_out;
    const auto saved_stats = opt_cache.stats();
    refuses([&] { dequantize_rows_sorted_dedup(table, 8, width, stored, logical, bad_ids, stateless_out, scratch); });
    refuses([&] { opt_cache.gather_chunk_dedup(bad_ids, dedup_cache_out, scratch); });
    need(stateless_out == saved_out, "B3 stateless out-of-range refusal mutated output");
    need(opt_cache.stats().hits == saved_stats.hits && opt_cache.stats().misses == saved_stats.misses,
        "B3 gather_chunk_dedup out-of-range refusal mutated LRU stats");
    ++cases;
    std::cout << "B3 chunk-wide sorted deduplicated PLE gather bit-exact parity checks passed\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test-exact_candidates_batch_b <path-to-metallib>\n";
        return 2;
    }
    test_b1_dense_m64_swizzle_multiseg(argv[1]);
    test_b2_hc_consumer_and_layers(argv[1]);
    test_b3_ple_gather_dedup();
    std::cout << "exact_candidates_batch_b: cases=" << cases << " checks=" << checks << " nmse=0.0\n";
    return 0;
}
