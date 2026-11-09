#include "tokspur/expert_lanes.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/moe_q8_three_rows.hpp"
#include "tokspur/q8_many.hpp"
#include "grouped_mm_fixture.hpp"

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
    need(bad, "invalid Batch A Graph call admitted");
}

struct Arena {
    MetalGraph& g;
    std::vector<GpuView> owned;
    GpuView bytes(std::span<const std::byte> x, bool immutable = true) {
        std::vector<std::byte> padded(x.size() + 32, std::byte{0x6d});
        std::copy(x.begin(), x.end(), padded.begin() + 16);
        auto v = g.upload(padded, "Batch A exact candidate fixture", immutable);
        owned.push_back(v);
        return g.slice(v, 16, x.size());
    }
    GpuView floats(std::span<const float> x, bool immutable = true) {
        return bytes(std::as_bytes(x), immutable);
    }
    void guards() {
        for (auto v : owned) {
            auto p = g.read(v);
            for (size_t i = 0; i < 16; ++i) {
                need(p[i] == std::byte{0x6d} && p[p.size() - 1 - i] == std::byte{0x6d}, "Batch A canary guard modified");
            }
        }
    }
    ~Arena() {
        try { g.abort(); } catch (const FormatError&) {}
        for (auto v : owned) g.release(v);
    }
};

std::vector<float> make_floats(size_t n, uint32_t seed) {
    std::vector<float> out(n);
    uint32_t rng = seed;
    for (auto& v : out) {
        rng = rng * 1664525u + 1013904223u;
        v = float(int(rng % 3999) - 1999) / 499.0f;
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

void test_a1_swiglu(const char* lib) {
    {
        MetalGraph g(lib, 16ull << 20);
        need(!g.swiglu_fused_enabled() && g.swiglu_fused_calls() == 0, "A1 enabled by default");
    }
    for (bool concurrent : {false, true}) {
        MetalGraph g_ref(lib, 32ull << 20);
        MetalGraph g_opt(lib, 32ull << 20);
        g_ref.configure_concurrent_dispatch(concurrent);
        g_opt.configure_concurrent_dispatch(concurrent);
        g_opt.configure_swiglu_fused(true);
        need(g_opt.swiglu_fused_enabled(), "A1 configure_swiglu_fused did not stick");

        for (uint32_t count : {1u, 7u, 32u, 257u, 640u, 2560u, 18432u}) {
            for (float scale : {1.0f, 0.5f, 1.75f}) {
                Arena ar{g_ref, {}}, ao{g_opt, {}};
                const auto gate_vals = make_floats(count, 101 + count);
                const auto up_vals = make_floats(count, 202 + count);
                const std::vector<float> poison(count, -7654.25f);

                const auto r_gate = ar.floats(gate_vals, false), r_up = ar.floats(up_vals, false), r_out = ar.floats(poison, false);
                const auto o_gate = ao.floats(gate_vals, false), o_up = ao.floats(up_vals, false), o_out = ao.floats(poison, false);

                g_ref.begin();
                g_ref.swiglu(r_gate, r_up, r_out, scale);
                need(g_ref.submit().dispatches == 2, "unfused SwiGLU did not emit 2 dispatches");

                const uint64_t calls_before = g_opt.swiglu_fused_calls();
                g_opt.begin();
                g_opt.swiglu(o_gate, o_up, o_out, scale);
                need(g_opt.submit().dispatches == 1, "fused SwiGLU did not emit 1 dispatch");
                need(g_opt.swiglu_fused_calls() == calls_before + 1, "fused SwiGLU call count mismatch");
                need(g_opt.read(o_out) == g_ref.read(r_out), "A1 out-of-place SwiGLU bit mismatch (NMSE != 0)");

                // In-place gate == output parity.
                g_ref.begin();
                g_ref.swiglu(r_gate, r_up, r_gate, scale);
                need(g_ref.submit().dispatches == 2, "unfused in-place SwiGLU dispatch count");
                g_opt.begin();
                g_opt.swiglu(o_gate, o_up, o_gate, scale);
                need(g_opt.submit().dispatches == 1, "fused in-place SwiGLU dispatch count");
                need(g_opt.read(o_gate) == g_ref.read(r_gate), "A1 in-place SwiGLU bit mismatch (NMSE != 0)");

                ar.guards();
                ao.guards();
                ++cases;
            }
        }

        // Abort and NaN/Inf refusal must roll back pending counter and refuse Reconfiguration after lock.
        refuses([&] { g_opt.configure_swiglu_fused(false); });
        Arena ao{g_opt, {}};
        auto gate_vals = make_floats(64, 777);
        const auto up_vals = make_floats(64, 888);
        const auto g_view = ao.floats(gate_vals, false), u_view = ao.floats(up_vals, false), out_view = ao.floats(gate_vals, false);
        const uint64_t saved = g_opt.swiglu_fused_calls();
        g_opt.begin();
        g_opt.swiglu(g_view, u_view, out_view);
        g_opt.abort();
        need(g_opt.swiglu_fused_calls() == saved, "aborted A1 published call count");

        // Aliasing output == up is refused.
        g_opt.begin();
        refuses([&] { g_opt.swiglu(g_view, u_view, u_view); });
        g_opt.abort();
        need(g_opt.swiglu_fused_calls() == saved, "refused A1 published call count");

        // Poison input triggers GPU error flag refusal and preserves counter.
        gate_vals[19] = std::numeric_limits<float>::infinity();
        g_opt.write(g_view, std::as_bytes(std::span(gate_vals)));
        g_opt.begin();
        g_opt.swiglu(g_view, u_view, out_view);
        refuses([&] { static_cast<void>(g_opt.submit()); });
        need(g_opt.swiglu_fused_calls() == saved, "failed A1 published call count");
        ao.guards();
    }
    std::cout << "A1 graph_swiglu bit-exact parity + poison/abort checks passed\n";
}

void test_a2_q8_gemv_rg4(const char* lib) {
    {
        MetalGraph g(lib, 16ull << 20);
        need(g.q8_gemv_rg4_mode() == Q8GemvRg4Mode::off && g.q8_gemv_rg4_routes() == MetalGraph::Q8GemvRg4Routes{0, 0}, "A2 enabled by default");
    }
    struct ShapeCase { Q8GemvRg4Mode mode; uint32_t cols, rows; bool packed, expects_rg4; };
    const std::array<ShapeCase, 10> shapes{{
        {Q8GemvRg4Mode::on, 512, 4096, false, false},
        {Q8GemvRg4Mode::on, 256, 16384, false, false},
        {Q8GemvRg4Mode::on, 512, 16384, false, true},
        {Q8GemvRg4Mode::on, 544, 16397, false, true},
        {Q8GemvRg4Mode::on, 512, 16384, true, true},
        {Q8GemvRg4Mode::on, 512, 16397, true, true},
        {Q8GemvRg4Mode::wide, 512, 2048, false, false},
        {Q8GemvRg4Mode::wide, 512, 4096, false, true},
        {Q8GemvRg4Mode::wide, 576, 4103, false, true},
        {Q8GemvRg4Mode::wide, 512, 6144, true, true},
    }};
    for (bool concurrent : {false, true}) {
        for (const auto& sc : shapes) {
            MetalGraph g_ref(lib, 128ull << 20);
            MetalGraph g_opt(lib, 128ull << 20);
            for (MetalGraph* gp : {&g_ref, &g_opt}) {
                gp->configure_block_gemv(3);
                gp->configure_small_gemv(16);
                gp->configure_concurrent_dispatch(concurrent);
            }
            g_opt.configure_q8_gemv_rg4(sc.mode);

            Arena ar{g_ref, {}}, ao{g_opt, {}};
            const auto raw = make_q8_weights(sc.cols, sc.rows, 303 + sc.rows);
            auto payload = raw;
            if (sc.packed) lanes::pack_rows(lanes::row_layout(8, sc.cols), raw, payload);

            const auto w_ref = ar.bytes(payload), w_opt = ao.bytes(payload);
            if (sc.packed) {
                g_ref.register_lane_matrix(w_ref, {8, sc.cols, sc.rows, 1});
                g_opt.register_lane_matrix(w_opt, {8, sc.cols, sc.rows, 1});
            }
            const auto x = make_floats(sc.cols, 404 + sc.cols);
            const std::vector<float> poison(sc.rows, -7654.25f);
            const auto x_ref = ar.floats(x, false), y_ref = ar.floats(poison, false);
            const auto x_opt = ao.floats(x, false), y_opt = ao.floats(poison, false);

            g_ref.begin();
            g_ref.gemv(8, w_ref, x_ref, y_ref, sc.cols, sc.rows);
            need(g_ref.submit().dispatches == 1, "baseline Q8 GEMV dispatch count");

            g_opt.begin();
            g_opt.gemv(8, w_opt, x_opt, y_opt, sc.cols, sc.rows);
            need(g_opt.submit().dispatches == 1, "A2 Q8 GEMV dispatch count");
            need(g_opt.read(y_opt) == g_ref.read(y_ref), "A2 Q8 GEMV rg4 bit mismatch (NMSE != 0)");

            MetalGraph::Q8GemvRg4Routes expected_routes{0, 0};
            if (sc.expects_rg4) expected_routes[sc.packed ? 1 : 0] = 1;
            need(g_opt.q8_gemv_rg4_routes() == expected_routes, "A2 Q8 GEMV rg4 route bucket mismatch");

            if (sc.expects_rg4 && !concurrent) {
                g_opt.begin();
                g_opt.gemv(8, w_opt, x_opt, y_opt, sc.cols, sc.rows);
                g_opt.abort();
                need(g_opt.q8_gemv_rg4_routes() == expected_routes, "aborted A2 published route");

                auto bad_x = x;
                bad_x.back() = std::numeric_limits<float>::quiet_NaN();
                g_opt.write(x_opt, std::as_bytes(std::span(bad_x)));
                g_opt.begin();
                g_opt.gemv(8, w_opt, x_opt, y_opt, sc.cols, sc.rows);
                refuses([&] { static_cast<void>(g_opt.submit()); });
                need(g_opt.q8_gemv_rg4_routes() == expected_routes, "poisoned A2 published route");
            }
            ar.guards();
            ao.guards();
            ++cases;
        }
    }
    std::cout << "A2 graph_q8_gemv_rg4 + dense_lanes_q8_rg4 bit-exact parity checks passed\n";
}

void test_a3_q8_three_msg(const char* lib) {
    {
        MetalGraph g(lib, 16ull << 20);
        need(g.q8_three_msg_mode() == Q8ThreeMsgMode::off && g.q8_three_msg_routes() == MetalGraph::Q8ThreeMsgRoutes{0, 0}, "A3 enabled by default");
        refuses([&] { g.configure_q8_three_msg(Q8ThreeMsgMode::on); });
    }
    struct MsgCase { Q8ThreeMsgMode mode; uint32_t cols, rows; bool packed, wide_many, expects_msg; };
    const std::array<MsgCase, 10> shapes{{
        {Q8ThreeMsgMode::on, 512, 4096, false, false, false},
        {Q8ThreeMsgMode::on, 256, 16384, false, false, false},
        {Q8ThreeMsgMode::on, 512, 16384, false, false, true},
        {Q8ThreeMsgMode::on, 544, 16397, false, false, true},
        {Q8ThreeMsgMode::on, 512, 16384, true, false, true},
        {Q8ThreeMsgMode::on, 512, 16397, true, false, true},
        {Q8ThreeMsgMode::wide, 512, 2048, false, false, false},
        {Q8ThreeMsgMode::wide, 576, 4119, false, false, true},
        {Q8ThreeMsgMode::wide, 512, 6144, true, false, true},
        {Q8ThreeMsgMode::wide, 2560, 10240, false, true, false}, // Wide dedicated kernel stays on its own route!
    }};
    for (bool concurrent : {false, true}) {
        for (const auto& sc : shapes) {
            MetalGraph g_ref(lib, 128ull << 20);
            MetalGraph g_opt(lib, 128ull << 20);
            for (MetalGraph* gp : {&g_ref, &g_opt}) {
                gp->configure_q8_three_rows(true);
                gp->configure_block_gemv(3);
                gp->configure_small_gemv(16);
                gp->configure_concurrent_dispatch(concurrent);
                gp->configure_hc_exact(HcExactMode::combined);
                gp->configure_wide_q8_many(sc.wide_many);
                gp->prepare_q8_matvec_many_all();
            }
            g_opt.configure_q8_three_msg(sc.mode);

            Arena ar{g_ref, {}}, ao{g_opt, {}};
            const auto raw = make_q8_weights(sc.cols, sc.rows, 505 + sc.rows);
            auto payload = raw;
            if (sc.packed) lanes::pack_rows(lanes::row_layout(8, sc.cols), raw, payload);

            const auto w_ref = ar.bytes(payload), w_opt = ao.bytes(payload);
            if (sc.packed) {
                g_ref.register_lane_matrix(w_ref, {8, sc.cols, sc.rows, 1});
                g_opt.register_lane_matrix(w_opt, {8, sc.cols, sc.rows, 1});
            }
            const auto x = make_floats(uint64_t(sc.cols) * 3, 606 + sc.cols);
            const std::vector<float> poison(uint64_t(sc.rows) * 3, -7654.25f);
            const auto x_ref = ar.floats(x, false), y_ref = ar.floats(poison, false);
            const auto x_opt = ao.floats(x, false), y_opt = ao.floats(poison, false);

            g_ref.begin();
            g_ref.q8_matvec_many(w_ref, x_ref, y_ref, sc.cols, sc.rows, 3);
            need(g_ref.submit().dispatches == 1, "baseline N3 dispatch count");

            g_opt.begin();
            g_opt.q8_matvec_many(w_opt, x_opt, y_opt, sc.cols, sc.rows, 3);
            need(g_opt.submit().dispatches == 1, "A3 N3 msg dispatch count");
            need(g_opt.read(y_opt) == g_ref.read(y_ref), "A3 N3 msg bit mismatch (NMSE != 0)");
            need(g_opt.q8_three_rows_routes() == g_ref.q8_three_rows_routes(), "A3 altered canonical q8_three_rows_routes bucket");

            MetalGraph::Q8ThreeMsgRoutes expected_msg{0, 0};
            if (sc.expects_msg) expected_msg[sc.packed ? 1 : 0] = 1;
            need(g_opt.q8_three_msg_routes() == expected_msg, "A3 q8_three_msg_routes bucket mismatch");

            if (sc.expects_msg && !concurrent) {
                g_opt.begin();
                g_opt.q8_matvec_many(w_opt, x_opt, y_opt, sc.cols, sc.rows, 3);
                g_opt.abort();
                need(g_opt.q8_three_msg_routes() == expected_msg, "aborted A3 published msg route");

                auto bad_x = x;
                bad_x.back() = -std::numeric_limits<float>::infinity();
                g_opt.write(x_opt, std::as_bytes(std::span(bad_x)));
                g_opt.begin();
                g_opt.q8_matvec_many(w_opt, x_opt, y_opt, sc.cols, sc.rows, 3);
                refuses([&] { static_cast<void>(g_opt.submit()); });
                need(g_opt.q8_three_msg_routes() == expected_msg, "poisoned A3 published msg route");
            }
            ar.guards();
            ao.guards();
            ++cases;
        }
    }
    std::cout << "A3 graph_q8_three_rows_msg + graph_lanes_q8_three_rows_msg bit-exact parity checks passed\n";
}

void test_a4_moe_three_union_case(const char* lib, uint32_t c, uint32_t h, uint32_t e, uint32_t s_top,
    uint32_t gate_up_type, uint32_t down_type, bool down_packed, bool shared_packed,
    bool disjoint_experts, uint32_t small_splits, bool concurrent) {
    MetalGraph g_ref(lib, 128ull << 20);
    MetalGraph g_opt(lib, 128ull << 20);
    for (MetalGraph* gp : {&g_ref, &g_opt}) {
        gp->configure_q8_three_rows(true);
        gp->configure_concurrent_dispatch(concurrent);
        gp->configure_small_gemv(small_splits);
        gp->configure_block_gemv(3);
        gp->prepare_q8_matvec_many_all();
    }
    g_opt.configure_moe_three_union(true);

    Arena ar{g_ref, {}}, ao{g_opt, {}};
    const MoeShape shape{c, h, e, s_top};

    // Build deterministic weights shared between ref and opt graphs.
    std::vector<float> router_f(uint64_t(c) * e, 0.0f);
    if (disjoint_experts) {
        // Partition expert affinity across 3 orthogonal channel bands so token 0, 1, 2 select completely disjoint top-S experts (U == 3*S)!
        for (uint32_t ex = 0; ex < e; ++ex) {
            const uint32_t band = ex % 3;
            for (uint32_t j = 0; j < c; ++j) {
                router_f[uint64_t(ex) * c + j] = (j % 3 == band ? 0.25f : -0.25f) + float(int((ex * 73 + j * 19) % 101) - 50) * 1e-4f;
            }
        }
    } else {
        router_f = make_floats(uint64_t(c) * e, 707 + e);
        for (auto& v : router_f) v *= 0.02f;
    }
    std::vector<float> sr_f = make_floats(c, 808 + c);
    for (auto& v : sr_f) v *= 0.02f;

    const auto make_quant_pair = [&](uint32_t type, uint32_t k, uint32_t r, uint32_t experts, bool lane) {
        auto raw = grouped_fixture::weights({type, k, r, experts, s_top, 3, true});
        auto bytes = raw;
        if (lane) lanes::pack_rows(lanes::row_layout(type, k), raw, bytes);
        auto vr = ar.bytes(bytes), vo = ao.bytes(bytes);
        if (lane) {
            g_ref.register_lane_matrix(vr, {type, k, r, experts});
            g_opt.register_lane_matrix(vo, {type, k, r, experts});
        }
        return std::pair{GpuMatrix{vr, type}, GpuMatrix{vo, type}};
    };

    const GpuMatrix r_router{ar.floats(router_f), 0}, o_router{ao.floats(router_f), 0};
    const GpuMatrix r_sr{ar.floats(sr_f), 0}, o_sr{ao.floats(sr_f), 0};
    const auto [r_gate, o_gate] = make_quant_pair(gate_up_type, c, h, e, false);
    const auto [r_up, o_up] = make_quant_pair(gate_up_type, c, h, e, false);
    const auto [r_down, o_down] = make_quant_pair(down_type, h, c, e, down_packed);
    const auto [r_sg, o_sg] = make_quant_pair(8, c, h, 1, shared_packed);
    const auto [r_su, o_su] = make_quant_pair(8, c, h, 1, shared_packed);
    const auto [r_sd, o_sd] = make_quant_pair(8, h, c, 1, false);

    const MoeWeights w_ref{r_router, r_gate, r_up, r_down, r_sg, r_su, r_sd, r_sr};
    const MoeWeights w_opt{o_router, o_gate, o_up, o_down, o_sg, o_su, o_sd, o_sr};
    need(moe_three_union_layer_supported(g_opt, w_opt, shape), "A4 layer unexpectedly reported unsupported");

    const std::array<uint64_t, 10> count{e, s_top, s_top, uint64_t(h) * s_top, uint64_t(h) * s_top, uint64_t(c) * s_top, h, h, c, 1};
    const auto make_scratch = [&](Arena& a) {
        std::array<GpuView, 10> v{};
        for (size_t i = 0; i < v.size(); ++i) v[i] = a.floats(std::vector<float>(count[i] * 8, -7654.25f), false);
        return MoeScratch{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]};
    };
    const auto s_ref = make_scratch(ar);
    const auto s_opt = make_scratch(ao);
    const auto up_poison = g_opt.read(s_opt.up);

    std::vector<float> in_vals(uint64_t(c) * 8, -9987.25f);
    for (uint32_t t = 0; t < 3; ++t) {
        for (uint32_t j = 0; j < c; ++j) {
            if (disjoint_experts) {
                in_vals[uint64_t(t) * c + j] = (j % 3 == t) ? 1.25f : -0.5f;
            } else {
                // Token 0 and token 1 identical (guaranteed duplicate experts!), token 2 slightly perturbed (partial overlap!).
                const uint32_t src_t = (t == 1) ? 0u : t;
                in_vals[uint64_t(t) * c + j] = float(int(((src_t * 31 + j * 97) % 1993)) - 996) / 997.0f;
            }
        }
    }

    const auto x_ref = ar.floats(in_vals, false), y_ref = ar.floats(std::vector<float>(uint64_t(c) * 8, -7654.25f), false);
    const auto x_opt = ao.floats(in_vals, false), y_opt = ao.floats(std::vector<float>(uint64_t(c) * 8, -7654.25f), false);

    const auto token_scratch = [&](MetalGraph& g, const MoeScratch& ss, uint32_t t) {
        auto v = ss.views();
        for (size_t i = 0; i < v.size(); ++i) v[i] = g.slice(v[i], uint64_t(t) * count[i] * 4, count[i] * 4);
        return MoeScratch{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]};
    };

    for (RouterMode mode : {RouterMode::softmax, RouterMode::sigmoid}) {
        for (bool normalize : {true, false}) {
            g_ref.begin();
            for (uint32_t t = 0; t < 3; ++t) {
                encode_moe(g_ref, w_ref, token_scratch(g_ref, s_ref, t),
                    g_ref.slice(x_ref, uint64_t(t) * c * 4, uint64_t(c) * 4),
                    g_ref.slice(y_ref, uint64_t(t) * c * 4, uint64_t(c) * 4),
                    shape, mode, normalize);
            }
            const auto ordinary = g_ref.submit();

            const uint64_t calls_before = g_opt.moe_three_union_calls();
            const auto ledger = g_opt.memory();
            g_opt.begin();
            encode_moe_q8_three_rows(g_opt, w_opt, s_opt, x_opt, y_opt, shape, mode, normalize, 8);
            const auto batch = g_opt.submit();

            need(batch.dispatches == 11, "A4 moe_three_union did not execute in 11 physical dispatches");
            need(batch.dispatches + moe_q8_three_rows_saved_dispatches(g_opt, w_opt, shape) == ordinary.dispatches,
                "A4 saved_dispatches accounting mismatch");
            need(g_opt.moe_three_union_calls() == calls_before + 1, "A4 moe_three_union_calls did not increment");
            need(g_opt.memory().allocations == ledger.allocations, "A4 allocated transient GPU buffers during encode");
            need(g_opt.read(y_opt) == g_ref.read(y_ref), "A4 moe_three_union output bit mismatch vs 3x encode_moe (NMSE != 0)");

            // Verify expert union size behaved as intended (duplicate vs disjoint).
            const auto idx_bytes = g_opt.read(g_opt.slice(s_opt.indices, 0, uint64_t(3) * s_top * 4));
            std::vector<uint32_t> idx(3 * s_top);
            std::memcpy(idx.data(), idx_bytes.data(), idx_bytes.size());
            std::sort(idx.begin(), idx.end());
            const size_t unique_experts = size_t(std::unique(idx.begin(), idx.end()) - idx.begin());
            if (disjoint_experts) {
                need(unique_experts == 3 * s_top, "disjoint A4 fixture did not produce U == 3*S unique experts");
            } else {
                need(unique_experts < 3 * s_top, "duplicate A4 fixture did not produce U < 3*S shared experts");
            }

            // Every scratch tensor except s.up (index 4, intentionally bypassed by fused gate+up+SwiGLU) must match 3x encode_moe bit-for-bit!
            const auto opt_views = s_opt.views(), ref_views = s_ref.views();
            for (size_t i = 0; i < opt_views.size(); ++i) {
                if (i == 4) {
                    need(g_opt.read(opt_views[i]) == up_poison, "A4 unexpectedly wrote to bypassed s.up scratch");
                } else {
                    need(g_opt.read(opt_views[i]) == g_ref.read(ref_views[i]), "A4 intermediate scratch bit mismatch vs 3x encode_moe");
                }
            }
            ++cases;
        }
    }

    if (!concurrent) {
        const uint64_t saved_calls = g_opt.moe_three_union_calls();
        g_opt.begin();
        encode_moe_q8_three_rows(g_opt, w_opt, s_opt, x_opt, y_opt, shape, RouterMode::softmax, true, 8);
        g_opt.abort();
        need(g_opt.moe_three_union_calls() == saved_calls, "aborted A4 published moe_three_union_calls");

        auto bad_in = in_vals;
        bad_in[uint64_t(2) * c + 3] = std::numeric_limits<float>::infinity();
        g_opt.write(x_opt, std::as_bytes(std::span(bad_in)));
        g_opt.begin();
        encode_moe_q8_three_rows(g_opt, w_opt, s_opt, x_opt, y_opt, shape, RouterMode::softmax, true, 8);
        refuses([&] { static_cast<void>(g_opt.submit()); });
        need(g_opt.moe_three_union_calls() == saved_calls, "poisoned A4 published moe_three_union_calls");
    }

    ar.guards();
    ao.guards();
}

void test_a4_moe_three_union(const char* lib) {
    {
        MetalGraph g(lib, 16ull << 20);
        need(!g.moe_three_union_enabled() && g.moe_three_union_calls() == 0, "A4 enabled by default");
        refuses([&] { g.configure_moe_three_union(true); });
    }
    for (bool concurrent : {false, true}) {
        // Production Qwen3.8-Flash-Next quant types: Q4_K (12) gate/up + Q5_1 (7) down (raw + q51_lanes_v1 packed at H=640).
        test_a4_moe_three_union_case(lib, 256, 640, 16, 10, 12, 7, false, false, false, 16, concurrent);
        test_a4_moe_three_union_case(lib, 256, 640, 32, 10, 12, 7, true, true, false, 16, concurrent);
        test_a4_moe_three_union_case(lib, 256, 640, 32, 10, 12, 7, true, true, true, 0, concurrent);
        // Additional supported quant types: Q5_K (13) gate/up + Q8_0 (8) down, and Q8_0 (8) gate/up + Q4_K (12) down.
        test_a4_moe_three_union_case(lib, 256, 256, 16, 6, 13, 8, false, false, false, 16, concurrent);
        test_a4_moe_three_union_case(lib, 256, 256, 24, 6, 8, 12, false, false, true, 16, concurrent);
    }
    std::cout << "A4 moe_three_union bit-exact parity (Q4_K/Q5_K/Q8_0/Q5_1 + q51_lanes_v1, duplicate & disjoint U=3*S) checks passed\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        need(argc == 2, "usage: test-exact_candidates_batch_a OWNED_METALLIB");
        const auto thermal_before = host_thermal_state();
        test_a1_swiglu(argv[1]);
        test_a2_q8_gemv_rg4(argv[1]);
        test_a3_q8_three_msg(argv[1]);
        test_a4_moe_three_union(argv[1]);
        std::cout << "PASS Batch A exact GPU candidates cases=" << cases << " checks=" << checks
                  << " nmse=0.0 thermal_before=" << thermal_before << " thermal_after=" << host_thermal_state() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL Batch A exact GPU candidates: " << e.what() << '\n';
        return 1;
    }
}
