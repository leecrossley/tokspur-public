#include "grouped_mm_fixture.hpp"
#include "tokspur/attention_math.hpp"
#include "tokspur/dflash2_drafter.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/gdn_chunk.hpp"
#include "tokspur/gdn_math.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/grouped_mm_f32.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/mtp_head.hpp"
#include "tokspur/ple.hpp"
#include "tokspur/qsa_prefill.hpp"
#include "tokspur/qsa_prefill_ops.hpp"
#include "tokspur/stack.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace tokspur;

namespace {

size_t total_checks = 0;

void need(bool ok, const char* why) {
    ++total_checks;
    if (!ok) throw FormatError(why);
}

// These fixtures have enough budget for every registered source. No dispatch
// trace is used: tracing deliberately suppresses Tile64 repacking.
void need_tile64_state(const MetalGraph& g, uint64_t candidates) {
    const auto s = g.q8_tile64_cache_stats();
    need(s.enabled == q8_tile64_cache_enabled && s.registered_candidates == candidates,
         "Q8 cache policy or registered source count differs");
    need(s.aliases == 0, "ordinary matrix fixture acquired a draft alias");
    if (!candidates) {
        need(!s.registered_source_bytes && !s.candidate_repack_logical_bytes && !s.covered_candidates &&
             !s.omitted_candidates && !s.omitted_repack_logical_bytes && !s.owning_allocations &&
             !s.owned_logical_bytes && !s.owned_charged_bytes, "released Q8 source retained cache accounting");
    } else {
        need(s.registered_source_bytes && s.candidate_repack_logical_bytes, "registered Q8 source identity is missing");
        if (q8_tile64_cache_enabled) {
            need(s.covered_candidates == candidates && !s.omitted_candidates && !s.omitted_repack_logical_bytes &&
                 s.owning_allocations == candidates && s.owned_logical_bytes == s.candidate_repack_logical_bytes &&
                 s.owned_charged_bytes >= s.owned_logical_bytes, "enabled Q8 fixture did not retain all distinct copies");
        } else {
            need(!s.covered_candidates && s.omitted_candidates == candidates &&
                 s.omitted_repack_logical_bytes == s.candidate_repack_logical_bytes && !s.owning_allocations &&
                 !s.owned_logical_bytes && !s.owned_charged_bytes, "disabled Q8 cache allocated or covered a source");
        }
    }
}

void need_tile64_registration(const MetalGraph& g, uint64_t before) {
    need(q8_tile64_cache_enabled ? g.memory().live_bytes > before : g.memory().live_bytes == before,
         "Q8 registration allocation differs from compile policy");
    need_tile64_state(g, 1);
}

void need_kernel(const MetalGraph& g, size_t index, const char* name, size_t count) {
    const auto& kernels = g.last_profile().kernels;
    need(kernels.size() == count && index < count && kernels[index].kernel == name,
         "Q8 policy selected an unexpected kernel");
}

struct Arena {
    MetalGraph& g;
    std::vector<GpuView> views;
    ~Arena() {
        try { g.abort(); } catch (...) {}
        for (auto v : views) g.release(v);
    }
    GpuView upload(std::span<const std::byte> data, bool immutable = true) {
        auto v = g.upload(data, "NAX prefill Metal test fixture", immutable);
        views.push_back(v);
        return v;
    }
    GpuView floats(std::span<const float> data, bool immutable = true) {
        return upload(std::as_bytes(data), immutable);
    }
    GpuView zeros(uint64_t bytes) {
        auto v = g.zeros(bytes, "NAX prefill Metal test scratch");
        views.push_back(v);
        return v;
    }
};

double nmse_f32(std::span<const float> got, std::span<const float> ref) {
    need(got.size() == ref.size() && !got.empty(), "NMSE span mismatch");
    double err = 0.0, sig = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        need(std::isfinite(got[i]) && std::isfinite(ref[i]), "nonfinite output in NAX test");
        const double d = double(got[i]) - double(ref[i]);
        err += d * d;
        sig += double(ref[i]) * double(ref[i]);
    }
    need(sig > 0.0, "zero reference signal in NAX test");
    return err / sig;
}

void test_dense_nax(const char* metallib, bool bounded_policy = false) {
    for (const auto layout : {PrefillWeightLayout::gguf, PrefillWeightLayout::q8_lanes_v1}) {
        for (const auto [k, r] : std::array<std::pair<uint32_t, uint32_t>, 4>{{{64, 71}, {320, 128}, {512, 193}, {2560, 131}}}) {
            MetalGraph g(metallib, 512ull << 20);
            g.configure_prefill_dense_f32(true);
            g.configure_prefill_dense_nax(true);
            need(g.configure_profiling(2) != ProfileMode::disabled, "dense route proof requires profiling support");
            Arena a{g, {}};
            const grouped_fixture::Shape f{8, k, r, 1, 1, 1, true, layout};
            const auto raw = grouped_fixture::weights({8, k, r, 1, 1, 1, true, PrefillWeightLayout::gguf});
            const auto truth = grouped_fixture::represented(f, raw);
            std::vector<std::byte> payload = raw;
            if (layout == PrefillWeightLayout::q8_lanes_v1) {
                const auto rl = lanes::row_layout(8, k);
                lanes::pack_rows(rl, raw, payload);
            }
            const auto w = a.upload(payload);
            if (layout == PrefillWeightLayout::q8_lanes_v1) {
                g.register_lane_matrix(w, {8, k, r, 1});
            } else g.register_q8_matrix(w, k, r);
            g.prepare_dense_mm_q8(layout);
            need_tile64_state(g, 1);
            need(g.prefill_weight_layout(w) == layout, "NAX dense registered layout query");
            for (const uint32_t n : {2u, 7u, 32u, 63u, 64u, 127u, 128u, 129u, 513u, 1024u, 2048u}) {
                if (bounded_policy && n != 7u && n != 127u && n != 128u && n != 129u) continue;
                std::vector<float> x(uint64_t(n) * k);
                uint32_t rng = 1337u + n;
                for (auto& v : x) {
                    rng = rng * 1664525u + 1013904223u;
                    v = float(int(rng % 1993) - 996) / 997.0f;
                }
                const uint64_t ys = ((uint64_t(r) * 4 + 15) / 16) * 16;
                const auto xv = a.floats(x);
                const auto yv = a.zeros(uint64_t(n) * ys);
                g.begin();
                g.dense_mm_q8(layout, w, {xv, n, k, uint64_t(k) * 4}, {yv, n, r, ys});
                const auto batch = g.submit();
                need(batch.dispatches == 2, "NAX dense_mm_q8 dispatch count");
                need_kernel(g, 1, q8_tile64_cache_enabled
                    ? (n >= 128 ? "nax_prefill_dense_q8_t64_m32" : "nax_prefill_dense_q8_t64_m8")
                    : (n >= 128 ? "nax_prefill_dense_q8_m32" : "nax_prefill_dense_q8_m8"), 2);
                need_tile64_state(g, 1);
                const auto bytes = g.read(yv);
                std::vector<float> raw_y(bytes.size() / 4);
                std::memcpy(raw_y.data(), bytes.data(), bytes.size());
                std::vector<float> got(uint64_t(n) * r), ref(uint64_t(n) * r);
                for (uint32_t row = 0; row < n; ++row) {
                    for (uint32_t col = 0; col < r; ++col) {
                        got[uint64_t(row) * r + col] = raw_y[uint64_t(row) * (ys / 4) + col];
                        double dot = 0.0;
                        for (uint32_t c = 0; c < k; ++c) {
                            dot += double(truth[uint64_t(col) * k + c]) * double(x[uint64_t(row) * k + c]);
                        }
                        ref[uint64_t(row) * r + col] = float(dot);
                    }
                }
                const double e = nmse_f32(got, ref);
                need(e <= 1e-4, "NAX dense_mm_q8 NMSE exceeded 1e-4");
            }
        }
    }
    std::cout << (bounded_policy ? "PASS NAX dense_mm_q8 policy subset (M7/127/128/129)\n"
                                : "PASS NAX dense_mm_q8 (m8 + m32 up to 2048 rows)\n");
}

void test_grouped_nax(const char* metallib) {
    struct GroupedCase {
        uint32_t type, k, r;
        PrefillWeightLayout layout;
    };
    const std::array<GroupedCase, 6> cases{{
        {7u, 128u, 64u, PrefillWeightLayout::gguf},
        {7u, 640u, 64u, PrefillWeightLayout::q51_lanes_v1},
        {8u, 128u, 64u, PrefillWeightLayout::gguf},
        {12u, 256u, 64u, PrefillWeightLayout::gguf},
        {12u, 256u, 640u, PrefillWeightLayout::gguf},
        {13u, 256u, 64u, PrefillWeightLayout::gguf},
    }};
    for (const auto gc : cases) {
        MetalGraph g(metallib, 1ull << 30);
        g.configure_prefill_dense_f32(true);
        g.configure_prefill_grouped_f32(true);
        g.configure_prefill_dense_nax(true);
        g.configure_prefill_grouped_nax(true);
        const uint32_t type = gc.type, k = gc.k, r = gc.r;
        const auto layout = gc.layout;
        const uint32_t e = 8u, s = 4u;
        Arena a{g, {}};
        const grouped_fixture::Shape f{type, k, r, e, s, 1, true, layout};
        const auto raw = grouped_fixture::weights({type, k, r, e, s, 1, true, PrefillWeightLayout::gguf});
        const auto truth = grouped_fixture::represented(f, raw);
        auto payload = raw;
        if (layout != PrefillWeightLayout::gguf) {
            lanes::pack_rows(lanes::row_layout(type, k), raw, payload);
        }
        const auto w = a.upload(payload);
        if (layout != PrefillWeightLayout::gguf) {
            g.register_lane_matrix(w, {type, k, r, e});
        }
        g.prepare_grouped_mm(type, layout);
        need(g.prefill_weight_layout(w) == layout, "NAX grouped registered layout query");
        for (const bool broadcast : {true, false}) {
            for (const uint32_t n : {2u, 37u, 513u, 1024u, 2048u}) {
                const auto plan = grouped_mm_f32_plan(type, layout, k, r, e, s, n, uint64_t(k) * 4, uint64_t(r) * 4, broadcast, true);
                const uint32_t in_rows = broadcast ? n : n * s;
                const uint32_t out_rows = n * s;
                std::vector<float> x(uint64_t(in_rows) * k);
                uint32_t rng = 4242u + type * 17u + n;
                for (auto& v : x) {
                    rng = rng * 1664525u + 1013904223u;
                    v = float(int(rng % 1993) - 996) / 997.0f;
                }
                std::vector<uint32_t> ids(out_rows);
                for (uint32_t t = 0; t < n; ++t)
                    for (uint32_t slot = 0; slot < s; ++slot)
                        ids[t * s + slot] = (t * 5 + slot * 3) % e;
                const auto xv = a.floats(x);
                const auto yv = a.zeros(uint64_t(out_rows) * r * 4);
                const auto ix = a.upload(std::as_bytes(std::span(ids)));
                const auto map = a.zeros(plan.token_map_bytes);
                const auto counts = a.zeros(plan.counts_bytes);
                const auto tasks = a.zeros(plan.tasks_bytes);
                g.begin();
                const auto routes = g.grouped_mm_routes(ix, map, counts, tasks, e, s, n);
                g.grouped_mm_project(type, layout, w, routes, {xv, in_rows, k, uint64_t(k) * 4}, {yv, out_rows, r, uint64_t(r) * 4}, broadcast);
                g.end_grouped_mm_routes(routes);
                const auto batch = g.submit();
                need(batch.dispatches == 4, "NAX grouped_mm_routes + project dispatch count");
                const auto bytes = g.read(yv);
                std::vector<float> got(out_rows * r), ref(out_rows * r);
                std::memcpy(got.data(), bytes.data(), bytes.size());
                for (uint32_t rid = 0; rid < out_rows; ++rid) {
                    const uint32_t expert = ids[rid];
                    const uint32_t in_r = broadcast ? (rid / s) : rid;
                    for (uint32_t col = 0; col < r; ++col) {
                        double dot = 0.0;
                        for (uint32_t c = 0; c < k; ++c) {
                            dot += double(truth[(uint64_t(expert) * r + col) * k + c]) * double(x[uint64_t(in_r) * k + c]);
                        }
                        ref[uint64_t(rid) * r + col] = float(dot);
                    }
                }
                const double err = nmse_f32(got, ref);
                need(err <= 1e-4, "NAX grouped_mm NMSE exceeded 1e-4");
                if (broadcast && r == 640u && n == 37u) {
                    const auto y_gu = a.zeros(uint64_t(out_rows) * r * 4);
                    g.begin();
                    const auto gu_routes = g.grouped_mm_routes(ix, map, counts, tasks, e, s, n);
                    g.grouped_mm_gate_up_swiglu(type, layout, w, w, gu_routes, {xv, in_rows, k, uint64_t(k) * 4}, {y_gu, out_rows, r, uint64_t(r) * 4}, true);
                    g.end_grouped_mm_routes(gu_routes);
                    const auto gu_batch = g.submit();
                    need(gu_batch.dispatches == 4, "NAX grouped_mm_gate_up_swiglu r=640 dispatch count");
                    std::vector<float> got_gu(out_rows * r), ref_gu(out_rows * r);
                    g.read(y_gu, std::as_writable_bytes(std::span(got_gu)));
                    for (size_t idx = 0; idx < ref.size(); ++idx) {
                        const float gv = ref[idx];
                        const float sig = gv >= 0.0f ? 1.0f / (1.0f + std::exp(-gv)) : std::exp(gv) / (1.0f + std::exp(gv));
                        ref_gu[idx] = (gv * sig) * gv;
                    }
                    need(nmse_f32(got_gu, ref_gu) <= 2e-4, "NAX grouped_mm_gate_up_swiglu r=640 NMSE exceeded 2e-4");
                }
                if (broadcast && r == 640u && (n == 37u || n == 513u)) {
                    need(g.can_grouped_mm_handoff_bf16(e, s, n, k, r), "can_grouped_mm_handoff_bf16 must admit 640-col hidden");
                    const grouped_fixture::Shape f_dn{7u, r, k, e, s, 1, false, PrefillWeightLayout::q51_lanes_v1};
                    const auto raw_dn = grouped_fixture::weights({7u, r, k, e, s, 1, false, PrefillWeightLayout::gguf});
                    const auto truth_dn = grouped_fixture::represented(f_dn, raw_dn);
                    auto pay_dn = raw_dn;
                    lanes::pack_rows(lanes::row_layout(7u, r), raw_dn, pay_dn);
                    MetalGraph g_ho(metallib, 1ull << 30);
                    g_ho.configure_prefill_dense_f32(true);
                    g_ho.configure_prefill_grouped_f32(true);
                    g_ho.configure_prefill_dense_nax(true);
                    g_ho.configure_prefill_grouped_nax(true);
                    Arena a_ho{g_ho, {}};
                    const auto w_gu = a_ho.upload(payload);
                    const auto w_dn = a_ho.upload(pay_dn);
                    g_ho.register_lane_matrix(w_dn, {7u, r, k, e});
                    g_ho.prepare_grouped_mm(type, layout);
                    g_ho.prepare_grouped_mm(7u, PrefillWeightLayout::q51_lanes_v1);
                    const auto xv_ho = a_ho.floats(x);
                    const auto y_gu_ho = a_ho.zeros(uint64_t(out_rows) * r * 4);
                    const auto y_dn_ho = a_ho.zeros(uint64_t(out_rows) * k * 4);
                    const auto ix_ho = a_ho.upload(std::as_bytes(std::span(ids)));
                    const auto map_ho = a_ho.zeros(plan.token_map_bytes);
                    const auto cnt_ho = a_ho.zeros(plan.counts_bytes);
                    const auto tsk_ho = a_ho.zeros(plan.tasks_bytes);
                    g_ho.begin();
                    const auto ho_routes = g_ho.grouped_mm_routes(ix_ho, map_ho, cnt_ho, tsk_ho, e, s, n);
                    g_ho.grouped_mm_gate_up_swiglu(type, layout, w_gu, w_gu, ho_routes,
                        {xv_ho, in_rows, k, uint64_t(k) * 4}, {y_gu_ho, out_rows, r, uint64_t(r) * 4}, true, true);
                    g_ho.grouped_mm_project(7u, PrefillWeightLayout::q51_lanes_v1, w_dn, ho_routes,
                        {y_gu_ho, out_rows, r, uint64_t(r) * 4}, {y_dn_ho, out_rows, k, uint64_t(k) * 4}, false);
                    g_ho.end_grouped_mm_routes(ho_routes);
                    const auto ho_batch = g_ho.submit();
                    need(ho_batch.dispatches == 5, "NAX grouped BF16 handoff must execute in 5 dispatches");
                    std::vector<float> got_dn(out_rows * k), ref_dn(out_rows * k);
                    g_ho.read(y_dn_ho, std::as_writable_bytes(std::span(got_dn)));
                    for (uint32_t rid = 0; rid < out_rows; ++rid) {
                        const uint32_t ex = ids[rid];
                        std::vector<float> hid_row(r);
                        for (uint32_t col = 0; col < r; ++col) {
                            const float gv = ref[uint64_t(rid) * r + col];
                            const float sig = gv >= 0.0f ? 1.0f / (1.0f + std::exp(-gv)) : std::exp(gv) / (1.0f + std::exp(gv));
                            hid_row[col] = (gv * sig) * gv;
                        }
                        for (uint32_t col = 0; col < k; ++col) {
                            double dd = 0.0;
                            for (uint32_t c = 0; c < r; ++c) {
                                dd += double(truth_dn[(uint64_t(ex) * k + col) * r + c]) * double(hid_row[c]);
                            }
                            ref_dn[uint64_t(rid) * k + col] = float(dd);
                        }
                    }
                    need(nmse_f32(got_dn, ref_dn) <= 2e-4, "NAX grouped BF16 handoff NMSE exceeded 2e-4");
                }
            }
        }
    }
    std::cout << "PASS NAX grouped_mm (Q5_1, Q8_0, Q4_K, Q5_K, broadcast=true/false, up to 2048 tokens)\n";
}

GpuMatrix make_q8(MetalGraph& g, std::vector<GpuView>& owned, uint32_t c, uint32_t r, uint32_t experts = 1, uint32_t seed = 1) {
    std::vector<std::byte> data(uint64_t(c / 32) * r * experts * 34);
    for (uint64_t b = 0; b < data.size() / 34; ++b) {
        const uint16_t d = 0x1400;
        std::memcpy(data.data() + b * 34, &d, 2);
        for (uint32_t j = 0; j < 32; ++j) {
            const int8_t q = int8_t(int(((b + seed) * 17 + j * 11) % 15) - 7);
            std::memcpy(data.data() + b * 34 + 2 + j, &q, 1);
        }
    }
    const auto v = g.upload(data, "NAX test Q8 weight", true);
    owned.push_back(v);
    return {v, 8};
}

GpuView make_vec(MetalGraph& g, std::vector<GpuView>& owned, size_t n, float bias, std::mt19937& rng) {
    std::vector<float> vals(n);
    for (auto& x : vals) x = bias + float(int(rng() % 2001) - 1000) * 0.00015f;
    const auto v = g.upload(std::as_bytes(std::span(vals)), "NAX test vector", true);
    owned.push_back(v);
    return v;
}

GpuMatrix make_f32_mat(MetalGraph& g, std::vector<GpuView>& owned, size_t n, std::mt19937& rng) {
    return {make_vec(g, owned, n, 0.0f, rng), 0};
}

GpuMatrix make_bf16_mat(MetalGraph& g, std::vector<GpuView>& owned, size_t n, std::mt19937& rng) {
    std::vector<uint16_t> vals(n);
    for (auto& x : vals) {
        const float f = float(int(rng() % 2001) - 1000) * 0.00015f;
        x = finite_bf16(f);
    }
    const auto v = g.upload(std::as_bytes(std::span(vals)), "NAX test BF16 weight", true);
    owned.push_back(v);
    return {v, 30};
}

void test_stack_2048_and_ple_nax(const char* metallib, uint32_t capacity) {
    MetalGraph g(metallib, 1ull << 30);
    g.configure_small_gemv(16);
    g.configure_block_gemv(3);
    const auto configure_prefill = [&] {
        g.configure_prefill_dense_f32(true);
        g.configure_prefill_grouped_f32(true);
        g.configure_prefill_dense_nax(true);
        g.configure_prefill_grouped_nax(true);
        g.configure_prefill_hc_rows(true);
        g.configure_prefill_router_window(true);
        g.configure_prefill_hc_down_window(true);
    };
    if (capacity != 2048) configure_prefill();

    std::vector<GpuView> owned;
    std::mt19937 rng(44);
    const StackShape shape{
        {4, 128, 64},
        {128, 1, 2, 128, 32, 4},
        {128, 4, 2, 32, 2, 32, 32, 4, 16, 4096},
        {128, 64, 16, 4},
        {4, 128, 64, 4, 3},
        64,
        1
    };
    const NgramSpec spec{3, 2, 16, 64, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
    const StackPolicy policy{
        {0.08838835f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f},
        {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f},
        RouterMode::softmax,
        true,
        {HashOrder::newest_first, NgramBoundary(0), true}
    };

    const uint32_t c = shape.residual.channels, h = shape.residual.branches * c, rank = shape.residual.rank;
    auto mk_res = [&](uint32_t s) -> ResidualWeights {
        return {make_vec(g, owned, h, 1.0f, rng), make_q8(g, owned, h, rank, 1, s), make_q8(g, owned, rank, h, 1, s + 1), make_f32_mat(g, owned, uint64_t(h) * shape.residual.branches, rng)};
    };
    auto mk_moe = [&](uint32_t s) -> MoeWeights {
        return {
            make_f32_mat(g, owned, uint64_t(c) * shape.experts.experts, rng),
            make_q8(g, owned, c, shape.experts.hidden, shape.experts.experts, s),
            make_q8(g, owned, c, shape.experts.hidden, shape.experts.experts, s + 1),
            make_q8(g, owned, shape.experts.hidden, c, shape.experts.experts, s + 2),
            make_q8(g, owned, c, shape.experts.hidden, 1, s + 3),
            make_q8(g, owned, c, shape.experts.hidden, 1, s + 4),
            make_q8(g, owned, shape.experts.hidden, c, 1, s + 5),
            make_f32_mat(g, owned, c, rng)
        };
    };

    const uint32_t qkv_rows = uint32_t(gdn_qkv_count(shape.recurrent));
    RecurrentWeights l0{
        mk_res(10),
        mk_res(20),
        {
            make_q8(g, owned, c, qkv_rows, 1, 12),
            make_q8(g, owned, c, 64, 1, 13),
            make_f32_mat(g, owned, uint64_t(c) * 2, rng),
            make_f32_mat(g, owned, uint64_t(c) * 2, rng),
            make_q8(g, owned, 64, c, 1, 14),
            make_vec(g, owned, qkv_rows * 4, 0.0f, rng),
            make_vec(g, owned, 2, -1.0f, rng),
            make_vec(g, owned, 2, 0.0f, rng),
            make_vec(g, owned, 32, 1.0f, rng)
        },
        mk_moe(30)
    };
    const auto freqs = rotary_frequencies(shape.attention.rotary_width, 10000000);
    const auto freq_view = g.upload(std::as_bytes(std::span(freqs)), "NAX test freqs", true);
    owned.push_back(freq_view);
    AttentionLayerWeights l1{
        mk_res(40),
        mk_res(50),
        {
            make_q8(g, owned, c, 256, 1, 41),
            make_q8(g, owned, c, 64, 1, 42),
            make_q8(g, owned, c, 64, 1, 43),
            make_q8(g, owned, 128, c, 1, 44),
            make_bf16_mat(g, owned, uint64_t(c) * shape.attention.index_heads * shape.attention.index_width, rng),
            make_bf16_mat(g, owned, uint64_t(c) * shape.attention.index_width, rng),
            make_vec(g, owned, 32, 1.0f, rng),
            make_vec(g, owned, 32, 1.0f, rng),
            make_vec(g, owned, 32, 1.0f, rng),
            make_vec(g, owned, 32, 1.0f, rng),
            freq_view
        },
        mk_moe(60)
    };

    StackWeights weights{
        {l0, l1},
        {
            make_q8(g, owned, 64, h, 1, 70),
            make_q8(g, owned, 64, c, 1, 71),
            make_vec(g, owned, h, 1.0f, rng),
            make_vec(g, owned, h, 1.0f, rng),
            make_vec(g, owned, h, 1.0f, rng),
            make_vec(g, owned, uint64_t(h) * 4, 0.1f, rng)
        },
        {make_vec(g, owned, h, 1.0f, rng), make_q8(g, owned, h, rank, 1, 80), make_q8(g, owned, rank, h, 1, 81)},
        make_q8(g, owned, c, shape.vocabulary, 1, 90)
    };

    {
        TextStack serial(g, weights, shape, policy, spec);
        TextStack candidate(g, weights, shape, policy, spec, StackExecution::direct, capacity, true);
        // Serving selects NAX after stack construction, so this case exercises deferred Q8 registration.
        if (capacity == 2048) configure_prefill();
        candidate.prepare_chunked_prefill(GdnChunkMode::fused, 512ull << 20);
        candidate.prepare_final_hc_tap(128ull << 20);

        const uint32_t count = capacity + 38;
        const uint32_t mem_w = shape.memory.embedding;
        std::vector<uint32_t> tokens(count);
        std::vector<float> rows(uint64_t(count) * c), memory(uint64_t(count) * mem_w);
        auto history = empty_ngram_history(spec);
        for (uint32_t t = 0; t < count; ++t) {
            tokens[t] = (t * 7) % shape.vocabulary;
            const auto ids = ngram_step(spec, policy.ngram, history, tokens[t]);
            history = ids.next;
            for (uint32_t j = 0; j < c; ++j) rows[uint64_t(t) * c + j] = float(int((t * 13 + j * 7) % 131) - 65) / 997.0f;
            for (uint32_t j = 0; j < mem_w; ++j) memory[uint64_t(t) * mem_w + j] = float(int((ids.rows[j % 4] + j) % 17) - 8) / 32.0f;
        }

        // Configured prefill packet, then a 37-token tail and an ordinary continuation.
        const auto r2048 = candidate.step_known(
            std::span(tokens).first(capacity),
            std::span(rows).first(uint64_t(capacity) * c),
            std::span(memory).first(uint64_t(capacity) * mem_w));
        StackResult s2048;
        for (uint32_t t = 0; t < capacity; ++t)
            s2048 = serial.step(tokens[t], std::span(rows).subspan(uint64_t(t) * c, c), std::span(memory).subspan(uint64_t(t) * mem_w, mem_w));
        const double nmse_2048 = nmse_f32(r2048.logits, s2048.logits);
        need(nmse_2048 <= 1e-3, "NAX configured-packet TextStack step_known NMSE exceeded 1e-3");

        const auto r37 = candidate.step_known(
            std::span(tokens).subspan(capacity, 37),
            std::span(rows).subspan(uint64_t(capacity) * c, 37 * c),
            std::span(memory).subspan(uint64_t(capacity) * mem_w, 37 * mem_w));
        StackResult s37;
        for (uint32_t t = capacity; t < capacity + 37; ++t)
            s37 = serial.step(tokens[t], std::span(rows).subspan(uint64_t(t) * c, c), std::span(memory).subspan(uint64_t(t) * mem_w, mem_w));
        const double nmse_37 = nmse_f32(r37.logits, s37.logits);
        need(nmse_37 <= 1e-3, "NAX 37-token tail step_known NMSE exceeded 1e-3");

        const auto r1 = candidate.step(tokens[capacity + 37], std::span(rows).subspan(uint64_t(capacity + 37) * c, c), std::span(memory).subspan(uint64_t(capacity + 37) * mem_w, mem_w));
        const auto s1 = serial.step(tokens[capacity + 37], std::span(rows).subspan(uint64_t(capacity + 37) * c, c), std::span(memory).subspan(uint64_t(capacity + 37) * mem_w, mem_w));
        const double nmse_1 = nmse_f32(r1.logits, s1.logits);
        need(nmse_1 <= 1e-3, "NAX post-tail ordinary decode NMSE exceeded 1e-3");

        std::cout << "PASS NAX " << capacity << "-token TextStack + batched Layer-1 PLE (nmse_chunk=" << nmse_2048
                  << ", nmse_37=" << nmse_37 << ", nmse_1=" << nmse_1 << ")\n";
    }
    for (auto v : owned) g.release(v);
}

void test_mtp_priming_nax(const char* metallib) {
    MetalGraph g_nax(metallib, 512ull << 20);
    g_nax.configure_small_gemv(16);
    g_nax.configure_block_gemv(3);
    g_nax.configure_prefill_dense_f32(true);
    g_nax.configure_prefill_grouped_f32(true);
    g_nax.configure_prefill_dense_nax(true);
    g_nax.configure_prefill_grouped_nax(true);
    g_nax.configure_prefill_hc_rows(true);

    std::vector<GpuView> owned;
    std::mt19937 rng(99);
    const MtpHeadShape shape{{4, 64, 64}, {64, 64, 8, 2}, 4, 2, 32, 16, 64, 2048};
    const uint32_t c = shape.residual.channels, h = shape.residual.branches * c, rank = shape.residual.rank;
    const uint32_t q = shape.query_heads * shape.head_width, k = shape.kv_heads * shape.head_width;
    const auto freqs = rotary_frequencies(shape.rotary_width, 10000000);
    const auto freq_view = g_nax.upload(std::as_bytes(std::span(freqs)), "MTP NAX freqs", true);
    owned.push_back(freq_view);

    auto mk_res = [&](uint32_t s) -> ResidualWeights {
        return {make_vec(g_nax, owned, h, 1.0f, rng), make_q8(g_nax, owned, h, rank, 1, s), make_q8(g_nax, owned, rank, h, 1, s + 1), make_f32_mat(g_nax, owned, uint64_t(h) * shape.residual.branches, rng)};
    };
    MtpHeadWeights w{
        make_vec(g_nax, owned, h, 1.0f, rng),
        make_vec(g_nax, owned, c, 1.0f, rng),
        make_q8(g_nax, owned, 2 * c, c, 1, 1),
        mk_res(2),
        mk_res(4),
        {
            make_q8(g_nax, owned, c, q * 2, 1, 6),
            make_q8(g_nax, owned, c, k, 1, 7),
            make_q8(g_nax, owned, c, k, 1, 8),
            make_q8(g_nax, owned, q, c, 1, 9),
            make_vec(g_nax, owned, shape.head_width, 1.0f, rng),
            make_vec(g_nax, owned, shape.head_width, 1.0f, rng),
            freq_view
        },
        {
            make_f32_mat(g_nax, owned, uint64_t(c) * shape.experts.experts, rng),
            make_q8(g_nax, owned, c, shape.experts.hidden, shape.experts.experts, 10),
            make_q8(g_nax, owned, c, shape.experts.hidden, shape.experts.experts, 11),
            make_q8(g_nax, owned, shape.experts.hidden, c, shape.experts.experts, 12),
            make_q8(g_nax, owned, c, shape.experts.hidden, 1, 13),
            make_q8(g_nax, owned, c, shape.experts.hidden, 1, 14),
            make_q8(g_nax, owned, shape.experts.hidden, c, 1, 15),
            make_f32_mat(g_nax, owned, c, rng)
        },
        {make_vec(g_nax, owned, h, 1.0f, rng), make_q8(g_nax, owned, h, rank, 1, 16), make_q8(g_nax, owned, rank, h, 1, 17)},
        make_q8(g_nax, owned, c, shape.vocabulary, 1, 18)
    };

    {
        MtpHead batched(g_nax, w, shape), serial(g_nax, w, shape);
        batched.prepare_priming_batch(2048, 160ull << 20);
        need(batched.priming_batch_capacity() == 2048, "MTP NAX priming capacity 2048");
        batched.reset(1);
        serial.reset(1);

        constexpr uint32_t prime_rows = 131; // Crosses 128-row slice boundary (128 + 3)
        std::vector<float> hc(uint64_t(prime_rows + 1) * h), emb(uint64_t(prime_rows + 1) * c);
        for (auto& x : hc) x = float(int(rng() % 2001) - 1000) * 0.001f;
        for (auto& x : emb) x = float(int(rng() % 2001) - 1000) * 0.001f;

        const auto hc_before = g_nax.hc_rows_encoded();
        const auto batch = batched.prime_batch(
            std::span(hc).first(uint64_t(prime_rows) * h),
            std::span(emb).first(uint64_t(prime_rows) * c),
            1);
        // Verify priming scope prevented target hc_rows wire counter pollution, and dispatches drastically dropped vs 131 * 19 = 2489!
        need(g_nax.hc_rows_encoded() == hc_before, "MTP NAX priming polluted target hc_rows_encoded");
        need(batch.dispatches < 500, "MTP NAX priming did not batch rows");

        for (uint32_t i = 0; i < prime_rows; ++i) {
            static_cast<void>(serial.prime_pair(
                std::span(hc).subspan(uint64_t(i) * h, h),
                std::span(emb).subspan(uint64_t(i) * c, c),
                1 + i));
        }

        const auto b_adv = batched.advance(
            std::span(hc).subspan(uint64_t(prime_rows) * h, h),
            std::span(emb).subspan(uint64_t(prime_rows) * c, c),
            1 + prime_rows);
        const auto s_adv = serial.advance(
            std::span(hc).subspan(uint64_t(prime_rows) * h, h),
            std::span(emb).subspan(uint64_t(prime_rows) * c, c),
            1 + prime_rows);
        const double adv_nmse = nmse_f32(b_adv.logits, s_adv.logits);
        need(adv_nmse <= 1e-4, "MTP NAX priming advance logits NMSE exceeded 1e-4");
        std::cout << "PASS NAX MtpHead prime_batch(131 rows in " << batch.dispatches << " dispatches vs 2489 serial, advance NMSE=" << adv_nmse << ")\n";
    }
    for (auto v : owned) g_nax.release(v);
}

void test_nax_qsa_attention_and_bf16_reuse(const char* metallib) {
    // 1. Test dense_mm_q8 BF16 reuse + write invalidation
    {
        MetalGraph g(metallib, 256ull << 20);
        g.configure_prefill_dense_f32(true);
        g.configure_prefill_dense_nax(true);
        g.prepare_dense_mm_q8(PrefillWeightLayout::gguf);
        std::vector<GpuView> owned;
        std::mt19937 rng(777);
        const uint32_t m = 73, k = 128, n1 = 64, n2 = 96;
        const auto w1 = make_q8(g, owned, k, n1, 1, 1);
        const auto w2 = make_q8(g, owned, k, n2, 1, 2);
        std::vector<float> h_x(uint64_t(m) * k);
        for (auto& val : h_x) val = float(int(rng() % 2001) - 1000) * 0.001f;
        const auto x = g.zeros(uint64_t(m) * k * 4, "x_mutable");
        const auto x_orig = g.upload(std::as_bytes(std::span(h_x)), "x_orig", true);
        g.write(x, std::as_bytes(std::span(h_x)));
        const auto y1 = g.zeros(uint64_t(m) * n1 * 4, "y1");
        const auto y2 = g.zeros(uint64_t(m) * n2 * 4, "y2");
        const auto y2_ref = g.zeros(uint64_t(m) * n2 * 4, "y2_ref");
        owned.push_back(x); owned.push_back(x_orig); owned.push_back(y1); owned.push_back(y2); owned.push_back(y2_ref);
        const GpuRowsF32 x_rows{x, m, k, uint64_t(k) * 4};

        g.begin();
        need(!g.can_reuse_nax_dense_bf16(x_rows), "fresh batch must not claim valid NAX BF16");
        g.dense_mm_q8(PrefillWeightLayout::gguf, w1.view, x_rows, {y1, m, n1, uint64_t(n1) * 4}, false);
        need(g.can_reuse_nax_dense_bf16(x_rows), "post-dense_mm_q8 must mark NAX BF16 valid");
        g.dense_mm_q8(PrefillWeightLayout::gguf, w2.view, x_rows, {y2, m, n2, uint64_t(n2) * 4}, true);
        // Mutating x in-place must immediately invalidate BF16 reuse!
        g.activate(x, x, Activation::silu);
        need(!g.can_reuse_nax_dense_bf16(x_rows), "in-place mutation of input must invalidate NAX BF16 reuse");
        const auto batch = g.submit();
        need(batch.dispatches == 4, "NAX BF16 reuse batch expected 2 + 1 + 1 = 4 dispatches");

        g.begin();
        g.dense_mm_q8(PrefillWeightLayout::gguf, w2.view, {x_orig, m, k, uint64_t(k) * 4}, {y2_ref, m, n2, uint64_t(n2) * 4}, false);
        static_cast<void>(g.submit());

        std::vector<float> h_y2(uint64_t(m) * n2), h_y2_ref(uint64_t(m) * n2);
        g.read(y2, std::as_writable_bytes(std::span(h_y2)));
        g.read(y2_ref, std::as_writable_bytes(std::span(h_y2_ref)));
        for (size_t i = 0; i < h_y2.size(); ++i) {
            need(h_y2[i] == h_y2_ref[i], "NAX BF16 reuse must produce bit-identical output");
            ++total_checks;
        }
        for (auto v : owned) g.release(v);
        std::cout << "PASS NAX dense_mm_q8 BF16 reuse + write invalidation (" << h_y2.size() << " exact values)\n";
    }

    // 2. Test nax_qsa_attention at exact target head geometry (32 query heads, 2 KV heads, D=256)
    // across both dense prefix (base=0, n=67) and sparse selected-block boundary (base=2040, n=23 -> 2063 tokens)
    {
        MetalGraph g_nax(metallib, 512ull << 20), g_ref(metallib, 512ull << 20);
        g_nax.configure_prefill_dense_f32(true);
        g_nax.configure_prefill_dense_nax(true);
        g_nax.prepare_gdn_chunk(); g_nax.prepare_qsa_prefill();
        g_ref.prepare_gdn_chunk(); g_ref.prepare_qsa_prefill();

        const QsaShape shape{256, 24, 2, 256, 4, 128, 64, 4, 2048, 2304};
        need(g_nax.nax_qsa_attention_supported(shape), "g_nax must support nax_qsa_attention on 24x2x256");
        need(!g_ref.nax_qsa_attention_supported(shape), "g_ref must keep legacy QSA attention when prefill_dense_nax is off");

        const uint32_t k_dim = shape.kv_heads * shape.head_width; // 512
        const uint32_t q_dim = shape.query_heads * shape.head_width; // 8192
        const uint32_t block_cap = shape.capacity / 4; // 576
        const uint32_t block_budget = shape.token_budget / 4; // 512

        std::mt19937 rng(2026);
        std::vector<uint16_t> h_keys(uint64_t(shape.capacity) * k_dim), h_vals(uint64_t(shape.capacity) * k_dim);
        for (auto& x : h_keys) x = finite_bf16(float(int(rng() % 2001) - 1000) * 0.0008f);
        for (auto& x : h_vals) x = finite_bf16(float(int(rng() % 2001) - 1000) * 0.0008f);

        auto run_case = [&](uint32_t base, uint32_t n) {
            std::vector<float> h_q(uint64_t(n) * q_dim);
            for (auto& x : h_q) x = float(int(rng() % 2001) - 1000) * 0.0008f;
            std::vector<uint32_t> h_ids(uint64_t(n) * block_budget, 0u);
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t blocks = (base + r + 1) / 4;
                if (blocks <= block_budget) {
                    for (uint32_t b = 0; b < blocks; ++b) h_ids[uint64_t(r) * block_budget + b] = b;
                } else {
                    // Deterministic strided ascending subset of block_budget blocks < blocks
                    const uint32_t skip = blocks - block_budget;
                    for (uint32_t b = 0; b < block_budget; ++b) {
                        h_ids[uint64_t(r) * block_budget + b] = b + ((b >= block_budget / 2) ? skip : 0u);
                    }
                }
            }

            const auto q_nax = g_nax.upload(std::as_bytes(std::span(h_q)), "q_nax");
            const auto k_nax = g_nax.upload(std::as_bytes(std::span(h_keys)), "k_nax");
            const auto v_nax = g_nax.upload(std::as_bytes(std::span(h_vals)), "v_nax");
            const auto ids_nax = g_nax.upload(std::as_bytes(std::span(h_ids)), "ids_nax");
            const auto out_nax = g_nax.zeros(uint64_t(n) * q_dim * 4, "out_nax");

            g_nax.begin();
            g_nax.nax_qsa_attention(q_nax, k_nax, v_nax, ids_nax, out_nax, base, n, shape, block_budget);
            static_cast<void>(g_nax.submit());

            const uint32_t tiles = (shape.token_budget + 3 + 127) / 128;
            const auto q_ref = g_ref.upload(std::as_bytes(std::span(h_q)), "q_ref");
            const auto k_ref = g_ref.upload(std::as_bytes(std::span(h_keys)), "k_ref");
            const auto v_ref = g_ref.upload(std::as_bytes(std::span(h_vals)), "v_ref");
            const auto ids_ref = g_ref.upload(std::as_bytes(std::span(h_ids)), "ids_ref");
            const auto partials = g_ref.zeros(uint64_t(32) * shape.query_heads * tiles * (shape.head_width + 2) * 4, "partials");
            const auto out_ref = g_ref.zeros(uint64_t(n) * q_dim * 4, "out_ref");

            g_ref.begin();
            for (uint32_t first = 0; first < n; first += 32) {
                const uint32_t sn = std::min(32u, n - first), sb = base + first;
                const auto qs = g_ref.slice(q_ref, uint64_t(first) * q_dim * 4, uint64_t(sn) * q_dim * 4);
                const auto is = g_ref.slice(ids_ref, uint64_t(first) * block_budget * 4, uint64_t(sn) * block_budget * 4);
                const auto os = g_ref.slice(out_ref, uint64_t(first) * q_dim * 4, uint64_t(sn) * q_dim * 4);
                const std::array<GpuView, 5> b1{qs, k_ref, v_ref, is, partials};
                g_ref.qsa_prefill_dispatch(QsaPrefillOp::attention_partial128, b1,
                    {sn, sb, shape.query_heads, shape.kv_heads, shape.head_width, 4, block_budget, q_dim, block_budget, tiles});
                const std::array<GpuView, 2> b2{partials, os};
                g_ref.qsa_prefill_dispatch(QsaPrefillOp::attention_reduce, b2,
                    {sn, sb, shape.query_heads, shape.head_width, 4, block_budget, tiles, q_dim});
            }
            static_cast<void>(g_ref.submit());

            std::vector<float> got(uint64_t(n) * q_dim), want(uint64_t(n) * q_dim);
            g_nax.read(out_nax, std::as_writable_bytes(std::span(got)));
            g_ref.read(out_ref, std::as_writable_bytes(std::span(want)));
            const double nmse = nmse_f32(got, want);
            float max_abs = 0.0f;
            for (size_t i = 0; i < got.size(); ++i) {
                max_abs = std::max(max_abs, std::fabs(got[i] - want[i]));
                ++total_checks;
            }
            need(nmse <= 1e-4 && max_abs <= 5e-3f, "nax_qsa_attention NMSE or max_abs exceeded BF16 tolerance");
            std::cout << "PASS nax_qsa_attention(base=" << base << ", n=" << n << ") nmse=" << nmse << " max_abs=" << max_abs << '\n';

            for (auto v : {q_nax, k_nax, v_nax, ids_nax, out_nax}) g_nax.release(v);
            for (auto v : {q_ref, k_ref, v_ref, ids_ref, partials, out_ref}) g_ref.release(v);
            static_cast<void>(block_cap);
        };
        run_case(0, 67);
        run_case(520, 8);
        run_case(2060, 8);
        run_case(2040, 23);
        run_case(2040, 140);

        // Verify-only scope (without prefill_dense_nax) + sticky error check on invalid block IDs
        MetalGraph g_vnax(metallib, 512ull << 20);
        g_vnax.configure_verify_nax(true);
        g_vnax.prepare_verify_nax({});
        need(g_vnax.verify_nax_qsa_attention_supported(shape), "g_vnax must support verify_nax_qsa_attention on 24x2x256");
        {
            const uint32_t base = 2060, n = 8;
            std::vector<float> h_q(uint64_t(n) * q_dim);
            for (auto& x : h_q) x = float(int(rng() % 2001) - 1000) * 0.0008f;
            std::vector<uint32_t> h_ids(uint64_t(n) * block_budget, 0u);
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t blocks = (base + r + 1) / 4;
                const uint32_t skip = blocks - block_budget;
                for (uint32_t b = 0; b < block_budget; ++b) {
                    h_ids[uint64_t(r) * block_budget + b] = b + ((b >= block_budget / 2) ? skip : 0u);
                }
            }
            const auto q_v = g_vnax.upload(std::as_bytes(std::span(h_q)), "q_v");
            const auto k_v = g_vnax.upload(std::as_bytes(std::span(h_keys)), "k_v");
            const auto v_v = g_vnax.upload(std::as_bytes(std::span(h_vals)), "v_v");
            const auto ids_v = g_vnax.upload(std::as_bytes(std::span(h_ids)), "ids_v");
            const auto out_v = g_vnax.zeros(uint64_t(n) * q_dim * 4, "out_v");
            g_vnax.set_verify_nax_scope(true);
            g_vnax.begin();
            g_vnax.nax_qsa_attention(q_v, k_v, v_v, ids_v, out_v, base, n, shape, block_budget);
            const auto b_ok = g_vnax.submit();
            g_vnax.set_verify_nax_scope(false);
            need(b_ok.dispatches == qsa_nax_attention_dispatches(base, n, shape), "verify_nax G12 QSA dispatch count mismatch");

            const auto q_n = g_nax.upload(std::as_bytes(std::span(h_q)), "q_n");
            const auto k_n = g_nax.upload(std::as_bytes(std::span(h_keys)), "k_n");
            const auto v_n = g_nax.upload(std::as_bytes(std::span(h_vals)), "v_n");
            const auto ids_n = g_nax.upload(std::as_bytes(std::span(h_ids)), "ids_n");
            const auto out_n = g_nax.zeros(uint64_t(n) * q_dim * 4, "out_n");
            g_nax.begin();
            g_nax.nax_qsa_attention(q_n, k_n, v_n, ids_n, out_n, base, n, shape, block_budget);
            static_cast<void>(g_nax.submit());

            std::vector<float> got_v(uint64_t(n) * q_dim), got_n(uint64_t(n) * q_dim);
            g_vnax.read(out_v, std::as_writable_bytes(std::span(got_v)));
            g_nax.read(out_n, std::as_writable_bytes(std::span(got_n)));
            for (size_t i = 0; i < got_v.size(); ++i) {
                need(got_v[i] == got_n[i], "verify_nax G12 QSA output differed from prefill G12 QSA");
                ++total_checks;
            }

            // Duplicate / non-ascending block ID must raise sticky FormatError
            h_ids[10] = h_ids[9];
            const auto bad_ids = g_vnax.upload(std::as_bytes(std::span(h_ids)), "bad_ids");
            g_vnax.set_verify_nax_scope(true);
            g_vnax.begin();
            g_vnax.nax_qsa_attention(q_v, k_v, v_v, bad_ids, out_v, base, n, shape, block_budget);
            bool caught_bad = false;
            try { static_cast<void>(g_vnax.submit()); } catch (const FormatError&) { caught_bad = true; }
            g_vnax.set_verify_nax_scope(false);
            need(caught_bad, "duplicate block ID must fail nax_qsa_build_tile_lists with sticky FormatError");

            for (auto v : {q_v, k_v, v_v, ids_v, out_v, bad_ids}) g_vnax.release(v);
            for (auto v : {q_n, k_n, v_n, ids_n, out_n}) g_nax.release(v);
            std::cout << "PASS verify_nax G12 QSA attention + invalid ID sticky rejection\n";
        }

        // Batched nax_qsa_select_blocks parity vs 32-row legacy stripes across 73 rows (single batch vs 3 stripes)
        {
            const uint32_t base = 2040, n = 73;
            const uint32_t iq_dim = shape.index_heads * shape.index_width; // 512
            std::vector<float> h_iq(uint64_t(n) * iq_dim);
            for (auto& x : h_iq) x = float(int(rng() % 2001) - 1000) * 0.001f;
            std::vector<uint16_t> h_ik(uint64_t(block_cap) * shape.index_width);
            for (auto& x : h_ik) x = finite_bf16(float(int(rng() % 2001) - 1000) * 0.001f);

            const auto iq_n = g_nax.upload(std::as_bytes(std::span(h_iq)), "iq_n");
            const auto ik_n = g_nax.upload(std::as_bytes(std::span(h_ik)), "ik_n");
            const auto out_ids_n = g_nax.zeros(uint64_t(n) * block_budget * 4, "out_ids_n");
            g_nax.begin();
            g_nax.nax_qsa_select_blocks(iq_n, ik_n, out_ids_n, base, n, shape, block_budget);
            const auto b_sel = g_nax.submit();
            need(b_sel.dispatches == 4, "nax_qsa_select_blocks(base=2040, n=73) must execute in 1 batch of 4 dispatches");

            const auto iq_r = g_ref.upload(std::as_bytes(std::span(h_iq)), "iq_r");
            const auto ik_r = g_ref.upload(std::as_bytes(std::span(h_ik)), "ik_r");
            const auto out_ids_r = g_ref.zeros(uint64_t(n) * block_budget * 4, "out_ids_r");
            const uint32_t sel_stride = ((block_cap + 511u) / 512u) * block_budget;
            const auto sc_r = g_ref.zeros(uint64_t(32) * block_cap * 4, "sc_r");
            const auto sa_r = g_ref.zeros(uint64_t(32) * sel_stride * 4, "sa_r");
            const auto ia_r = g_ref.zeros(uint64_t(32) * sel_stride * 4, "ia_r");
            const auto sb_r = g_ref.zeros(uint64_t(32) * sel_stride * 4, "sb_r");
            const auto ib_r = g_ref.zeros(uint64_t(32) * sel_stride * 4, "ib_r");
            g_ref.begin();
            for (uint32_t first = 0; first < n; first += 32) {
                const uint32_t sn = std::min(32u, n - first), sb = base + first, mb = (sb + sn) / 4u;
                if (mb <= block_budget) continue;
                const auto iqs = g_ref.slice(iq_r, uint64_t(first) * iq_dim * 4, uint64_t(sn) * iq_dim * 4);
                const auto ids = g_ref.slice(out_ids_r, uint64_t(first) * block_budget * 4, uint64_t(sn) * block_budget * 4);
                g_ref.qsa_prefill_dispatch(QsaPrefillOp::index_scores, std::array<GpuView, 3>{iqs, ik_r, sc_r},
                    {sn, sb, shape.index_heads, shape.index_width, block_cap, iq_dim, block_budget});
                g_ref.qsa_prefill_dispatch(QsaPrefillOp::select_sort, std::array<GpuView, 3>{sc_r, sa_r, ia_r},
                    {sn, sb, block_cap, sel_stride, block_budget});
                GpuView in_s = sa_r, in_i = ia_r, out_s = sb_r, out_i = ib_r;
                for (uint32_t span = 512u; span < mb; span *= 2u) {
                    g_ref.qsa_prefill_dispatch(QsaPrefillOp::select_merge, std::array<GpuView, 4>{in_s, in_i, out_s, out_i},
                        {sn, sb, sel_stride, block_budget, span});
                    std::swap(in_s, out_s);
                    std::swap(in_i, out_i);
                }
                g_ref.qsa_prefill_dispatch(QsaPrefillOp::select_ids, std::array<GpuView, 2>{in_i, ids},
                    {sn, sb, sel_stride, block_budget, block_budget});
            }
            static_cast<void>(g_ref.submit());

            std::vector<uint32_t> got_ids(uint64_t(n) * block_budget), ref_ids(uint64_t(n) * block_budget);
            g_nax.read(out_ids_n, std::as_writable_bytes(std::span(got_ids)));
            g_ref.read(out_ids_r, std::as_writable_bytes(std::span(ref_ids)));
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t blocks = (base + r + 1) / 4u;
                if (blocks <= block_budget) continue;
                for (uint32_t b = 0; b < block_budget; ++b) {
                    need(got_ids[uint64_t(r) * block_budget + b] == ref_ids[uint64_t(r) * block_budget + b],
                        "nax_qsa_select_blocks selected ID mismatch");
                    ++total_checks;
                }
            }
            for (auto v : {iq_n, ik_n, out_ids_n}) g_nax.release(v);
            for (auto v : {iq_r, ik_r, out_ids_r, sc_r, sa_r, ia_r, sb_r, ib_r}) g_ref.release(v);
            std::cout << "PASS nax_qsa_select_blocks(base=2040, n=73) exact selected IDs in 4 dispatches\n";
        }
    }
}

void test_verify_nax_m2_m3_m8(const char* metallib) {
    auto configure_graph = [](MetalGraph& g, bool nax) {
        g.configure_small_gemv(16);
        g.configure_block_gemv(3);
        g.configure_hc_exact(HcExactMode::combined);
        g.configure_qsa_stage_v8(true);
        g.configure_qsa_channel_chronological(true);
        g.configure_qsa_sparse(true);
        g.configure_gdn_ported(true);
        if (nax) g.configure_verify_nax(true);
    };

    MetalGraph g_nax(metallib, 1ull << 30), g_ref(metallib, 1ull << 30);
    configure_graph(g_nax, true);
    configure_graph(g_ref, false);

    std::vector<GpuView> owned_nax, owned_ref;
    std::mt19937 rng_nax(20261102), rng_ref(20261102);

    const StackShape shape{
        {4, 128, 64},
        {128, 1, 2, 128, 32, 4},
        {128, 4, 2, 32, 2, 32, 32, 4, 16, 512},
        {128, 64, 8, 4},
        {4, 128, 64, 4, 3},
        128,
        1
    };
    const NgramSpec spec{3, 2, 16, 128, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
    const StackPolicy policy{
        {0.08838835f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f},
        {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f},
        RouterMode::softmax,
        true,
        {HashOrder::newest_first, NgramBoundary(0), true}
    };

    auto build_weights = [&](MetalGraph& g, std::vector<GpuView>& owned, std::mt19937& rng) -> StackWeights {
        const uint32_t c = shape.residual.channels, h = shape.residual.branches * c, rank = shape.residual.rank;
        auto mk_res = [&](uint32_t s) -> ResidualWeights {
            return {make_vec(g, owned, h, 1.0f, rng), make_q8(g, owned, h, rank, 1, s), make_q8(g, owned, rank, h, 1, s + 1), make_f32_mat(g, owned, uint64_t(h) * shape.residual.branches, rng)};
        };
        auto mk_moe = [&](uint32_t s) -> MoeWeights {
            return {
                make_f32_mat(g, owned, uint64_t(c) * shape.experts.experts, rng),
                make_q8(g, owned, c, shape.experts.hidden, shape.experts.experts, s),
                make_q8(g, owned, c, shape.experts.hidden, shape.experts.experts, s + 1),
                make_q8(g, owned, shape.experts.hidden, c, shape.experts.experts, s + 2),
                make_q8(g, owned, c, shape.experts.hidden, 1, s + 3),
                make_q8(g, owned, c, shape.experts.hidden, 1, s + 4),
                make_q8(g, owned, shape.experts.hidden, c, 1, s + 5),
                make_f32_mat(g, owned, c, rng)
            };
        };
        const uint32_t qkv_rows = uint32_t(gdn_qkv_count(shape.recurrent));
        RecurrentWeights l0{
            mk_res(10),
            mk_res(20),
            {
                make_q8(g, owned, c, qkv_rows, 1, 12),
                make_q8(g, owned, c, 64, 1, 13),
                make_f32_mat(g, owned, uint64_t(c) * 2, rng),
                make_f32_mat(g, owned, uint64_t(c) * 2, rng),
                make_q8(g, owned, 64, c, 1, 14),
                make_vec(g, owned, qkv_rows * 4, 0.0f, rng),
                make_vec(g, owned, 2, -1.0f, rng),
                make_vec(g, owned, 2, 0.0f, rng),
                make_vec(g, owned, 32, 1.0f, rng)
            },
            mk_moe(30)
        };
        const auto freqs = rotary_frequencies(shape.attention.rotary_width, 10000000);
        const auto freq_view = g.upload(std::as_bytes(std::span(freqs)), "verify NAX test freqs", true);
        owned.push_back(freq_view);
        AttentionLayerWeights l1{
            mk_res(40),
            mk_res(50),
            {
                make_q8(g, owned, c, 256, 1, 41),
                make_q8(g, owned, c, 64, 1, 42),
                make_q8(g, owned, c, 64, 1, 43),
                make_q8(g, owned, 128, c, 1, 44),
                make_bf16_mat(g, owned, uint64_t(c) * shape.attention.index_heads * shape.attention.index_width, rng),
                make_bf16_mat(g, owned, uint64_t(c) * shape.attention.index_width, rng),
                make_vec(g, owned, 32, 1.0f, rng),
                make_vec(g, owned, 32, 1.0f, rng),
                make_vec(g, owned, 32, 1.0f, rng),
                make_vec(g, owned, 32, 1.0f, rng),
                freq_view
            },
            mk_moe(60)
        };
        return {
            {l0, l1},
            {
                make_q8(g, owned, 64, h, 1, 70),
                make_q8(g, owned, 64, c, 1, 71),
                make_vec(g, owned, h, 1.0f, rng),
                make_vec(g, owned, h, 1.0f, rng),
                make_vec(g, owned, h, 1.0f, rng),
                make_vec(g, owned, uint64_t(h) * 4, 0.1f, rng)
            },
            {make_vec(g, owned, h, 1.0f, rng), make_q8(g, owned, h, rank, 1, 80), make_q8(g, owned, rank, h, 1, 81)},
            make_q8(g, owned, c, shape.vocabulary, 1, 90)
        };
    };

    const auto w_nax = build_weights(g_nax, owned_nax, rng_nax);
    const auto w_ref = build_weights(g_ref, owned_ref, rng_ref);

    const uint32_t c = shape.residual.channels, mem_w = shape.memory.embedding;
    auto tok_row = [&](uint32_t tok) {
        std::vector<float> r(c);
        for (uint32_t j = 0; j < c; ++j) r[j] = float(int((tok * 13 + j * 7 + 3) % 131) - 65) / 251.0f;
        return r;
    };
    auto mem_row = [&](std::span<const uint64_t> ids) {
        std::vector<float> m(mem_w);
        for (uint32_t j = 0; j < mem_w; ++j) m[j] = float(int((ids[j / spec.row_width] + j * 3) % 29) - 14) / 64.0f;
        return m;
    };

    {
    // Prepare both sampled and greedy8 verifiers before the first MetalGraph batch submission (immutable PSO lock).
    TextStack s_nax(g_nax, w_nax, shape, policy, spec, StackExecution::direct, 8, true);
    TextStack s_ref(g_ref, w_ref, shape, policy, spec, StackExecution::direct, 8, true);
    s_nax.prepare_final_hc_tap(16ull << 20);
    s_ref.prepare_final_hc_tap(16ull << 20);
    s_nax.prepare_sampled_verification2(128ull << 20, {true, QsaVerifyRowsMode::core_and_q8});
    s_ref.prepare_sampled_verification2(128ull << 20, {true, QsaVerifyRowsMode::core_and_q8});

    TextStack gs_nax(g_nax, w_nax, shape, policy, spec, StackExecution::direct, 8, true);
    TextStack gs_ref(g_ref, w_ref, shape, policy, spec, StackExecution::direct, 8, true);
    TextStack s_oracle(g_ref, w_ref, shape, policy, spec);
    gs_nax.prepare_transient_verification8(128ull << 20, {true, QsaVerifyRowsMode::core_and_q8});
    gs_ref.prepare_transient_verification8(128ull << 20, {true, QsaVerifyRowsMode::core_and_q8});

    // 1. Sampled verification: m=1 fallback, m=2 (verify_sampled2), m=3 (verify_sampled_depth2), m=5 & m=8 (verify_sampled8) with partial commits!
    {

        auto advance_one = [&](TextStack& st, uint32_t tok) {
            const auto lk = st.lookup(tok);
            return st.step(tok, tok_row(tok), mem_row(lk.rows));
        };
        for (uint32_t i = 0; i < 19; ++i) {
            const uint32_t tok = (i * 11 + 5) % shape.vocabulary;
            static_cast<void>(advance_one(s_nax, tok));
            static_cast<void>(advance_one(s_ref, tok));
        }

        struct FixedChoice {
            uint32_t commit;
            static uint32_t choose(void* ctx, const StackSampledVerificationRows&) {
                return static_cast<FixedChoice*>(ctx)->commit;
            }
        };

        auto build_verify_inputs = [&](const TextStack& st, std::span<const uint32_t> ids, std::vector<float>& rows, std::vector<float>& mems) {
            rows.clear(); mems.clear();
            const auto lks = st.sampled_verification8_lookups(ids);
            for (size_t i = 0; i < ids.size(); ++i) {
                const auto r = tok_row(ids[i]);
                const auto m = mem_row(lks[i].rows);
                rows.insert(rows.end(), r.begin(), r.end());
                mems.insert(mems.end(), m.begin(), m.end());
            }
        };

        struct Case { uint32_t m, commit; };
        const std::array<Case, 5> cases{{{1, 1}, {2, 2}, {3, 2}, {5, 2}, {8, 3}}};
        for (const auto tc : cases) {
            std::vector<uint32_t> ids(tc.m);
            for (uint32_t i = 0; i < tc.m; ++i) ids[i] = (s_nax.position() * 7 + i * 17 + 3) % shape.vocabulary;
            std::vector<float> rows, mems;
            build_verify_inputs(s_nax, ids, rows, mems);
            const uint32_t prefix = s_nax.position();
            need(s_ref.position() == prefix, "verify NAX test prefix sync");
            FixedChoice choice{tc.commit};

            const auto run_one = [&](TextStack& st) {
                if (tc.m <= 2) return st.verify_sampled2(ids, rows, mems, prefix, {FixedChoice::choose, &choice});
                if (tc.m == 3) return st.verify_sampled_depth2(ids, rows, mems, prefix, {FixedChoice::choose, &choice});
                return st.verify_sampled8(ids, rows, mems, prefix, {FixedChoice::choose, &choice});
            };
            const auto out_nax = run_one(s_nax);
            const auto out_ref = run_one(s_ref);

            need(out_nax.committed == tc.commit && out_ref.committed == tc.commit, "verify NAX committed count");
            need(out_nax.evaluated.count == tc.m && out_ref.evaluated.count == tc.m, "verify NAX evaluated count");
            if (tc.m >= 2) {
                need(out_nax.verification.dispatches < out_ref.verification.dispatches, "verify NAX must cut verification dispatches vs non-NAX");
            } else {
                need(out_nax.verification.dispatches == out_ref.verification.dispatches, "verify NAX m=1 must preserve exact fallback dispatches");
            }
            for (uint32_t i = 0; i < tc.m; ++i) {
                need(nmse_f32(out_nax.evaluated.logits[i], out_ref.evaluated.logits[i]) <= 1e-4, "verify NAX evaluated logits NMSE > 1e-4");
            }
            need(nmse_f32(out_nax.logits, out_ref.logits) <= 1e-4, "verify NAX committed logits NMSE > 1e-4");
            need(nmse_f32(out_nax.evaluated.final_hc_rows, out_ref.evaluated.final_hc_rows) <= 1e-4, "verify NAX final_hc_rows NMSE > 1e-4");
            need(nmse_f32(s_nax.last_final_hc_residual(), s_ref.last_final_hc_residual()) <= 1e-4, "verify NAX committed HC residual NMSE > 1e-4");

            // Advance one ordinary continuation token after partial/full commit to prove GDN tape + PLE/QSA state recovery parity!
            const uint32_t cont_tok = (s_nax.position() * 13 + 9) % shape.vocabulary;
            const auto c_nax = advance_one(s_nax, cont_tok);
            const auto c_ref = advance_one(s_ref, cont_tok);
            need(nmse_f32(c_nax.logits, c_ref.logits) <= 1e-4, "verify NAX post-recovery continuation logits NMSE > 1e-4");
            std::cout << "PASS verify_nax sampled(m=" << tc.m << ", commit=" << tc.commit
                      << ") dispatches=" << out_nax.verification.dispatches << " vs ref=" << out_ref.verification.dispatches << '\n';
        }
    }

    // 2. Greedy verification8 (verify_greedy8) with m=6: partial acceptance + recovery!
    {
        auto advance_one = [&](TextStack& st, uint32_t tok) {
            const auto lk = st.lookup(tok);
            return st.step(tok, tok_row(tok), mem_row(lk.rows));
        };
        StackResult base_nax{}, base_ref{}, cur_oracle{};
        for (uint32_t i = 0; i < 9; ++i) {
            const uint32_t tok = (i * 5 + 2) % shape.vocabulary;
            base_nax = advance_one(gs_nax, tok);
            base_ref = advance_one(gs_ref, tok);
            cur_oracle = advance_one(s_oracle, tok);
        }

        auto argmax_tok = [](std::span<const float> lg) {
            return uint32_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
        };
        std::array<uint32_t, 6> greedy_ids{};
        auto oracle_logits = cur_oracle.logits;
        for (size_t i = 0; i < greedy_ids.size(); ++i) {
            greedy_ids[i] = argmax_tok(oracle_logits);
            oracle_logits = advance_one(s_oracle, greedy_ids[i]).logits;
        }
        // Corrupt token index 3 so exactly 3 tokens are accepted out of 6!
        greedy_ids[3] = (greedy_ids[3] + 1) % shape.vocabulary;

        std::vector<float> rows, mems;
        const auto lks = gs_nax.verification_lookups8(greedy_ids, 64);
        need(lks.count == 6, "verify_greedy8 staged count");
        for (size_t i = 0; i < greedy_ids.size(); ++i) {
            const auto r = tok_row(greedy_ids[i]);
            const auto m = mem_row(lks.rows[i].rows);
            rows.insert(rows.end(), r.begin(), r.end());
            mems.insert(mems.end(), m.begin(), m.end());
        }

        const auto g_out_nax = gs_nax.verify_greedy8(greedy_ids, rows, mems, 9, base_nax.logits, 64);
        const auto g_out_ref = gs_ref.verify_greedy8(greedy_ids, rows, mems, 9, base_ref.logits, 64);
        need(g_out_nax.evaluated == 6 && g_out_nax.head_dispatches == 1u, "verify_greedy8 NAX head_dispatches must be 1");
        need(g_out_nax.accepted == 3 && g_out_ref.accepted == 3, "verify_greedy8 NAX partial acceptance count must match reference");
        need(g_out_nax.verification.dispatches < g_out_ref.verification.dispatches, "verify_greedy8 NAX must cut verification dispatches");
        need(nmse_f32(g_out_nax.logits, g_out_ref.logits) <= 1e-4, "verify_greedy8 NAX accepted logits NMSE > 1e-4");
        need(nmse_f32(g_out_nax.evaluated_logits, g_out_ref.evaluated_logits) <= 1e-4, "verify_greedy8 NAX evaluated_logits NMSE > 1e-4");

        const auto c_nax = advance_one(gs_nax, 7);
        const auto c_ref = advance_one(gs_ref, 7);
        need(nmse_f32(c_nax.logits, c_ref.logits) <= 1e-4, "verify_greedy8 NAX post-recovery logits NMSE > 1e-4");
        std::cout << "PASS verify_nax greedy8(m=6, accepted=" << g_out_nax.accepted
                  << ") dispatches=" << g_out_nax.verification.dispatches << " vs ref=" << g_out_ref.verification.dispatches << '\n';
    }
    }

    for (auto v : owned_nax) g_nax.release(v);
    for (auto v : owned_ref) g_ref.release(v);
}

void test_verify_nax_routed_moe_all_encodings(const char* metallib) {
    struct RoutedCase {
        uint32_t type, k, r;
        PrefillWeightLayout layout;
    };
    const std::array<RoutedCase, 6> cases{{
        {7u, 128u, 67u, PrefillWeightLayout::gguf},
        {7u, 640u, 71u, PrefillWeightLayout::q51_lanes_v1},
        {8u, 128u, 69u, PrefillWeightLayout::gguf},
        {8u, 640u, 65u, PrefillWeightLayout::gguf},
        {12u, 256u, 65u, PrefillWeightLayout::gguf},
        {13u, 256u, 67u, PrefillWeightLayout::gguf},
    }};
    for (const auto rc : cases) {
        MetalGraph g(metallib, 256ull << 20);
        g.configure_verify_nax(true);
        Arena a{g, {}};
        const uint32_t e = 8u, s = 3u;
        const grouped_fixture::Shape f{rc.type, rc.k, rc.r, e, s, 1, true, rc.layout};
        const auto raw = grouped_fixture::weights({rc.type, rc.k, rc.r, e, s, 1, true, PrefillWeightLayout::gguf});
        const auto truth = grouped_fixture::represented(f, raw);
        auto payload = raw;
        if (rc.layout != PrefillWeightLayout::gguf) {
            lanes::pack_rows(lanes::row_layout(rc.type, rc.k), raw, payload);
        }
        const auto w = a.upload(payload);
        if (rc.layout != PrefillWeightLayout::gguf) {
            g.register_lane_matrix(w, {rc.type, rc.k, rc.r, e});
        }
        const std::array<std::pair<uint32_t, PrefillWeightLayout>, 1> moe_layouts{{{rc.type, rc.layout}}};
        g.prepare_verify_nax(moe_layouts);
        for (const bool broadcast : {true, false}) {
            for (const bool swiglu : {false, true}) {
                for (const uint32_t m : {1u, 3u, 8u}) {
                    const uint32_t in_rows = broadcast ? m : m * s;
                    const uint32_t out_rows = m * s;
                    std::vector<float> x(uint64_t(in_rows) * rc.k);
                    std::vector<float> init_gate(uint64_t(out_rows) * rc.r);
                    uint32_t rng = 9001u + rc.type * 31u + m * 7u + uint32_t(swiglu);
                    for (auto& v : x) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 997.0f; }
                    for (auto& v : init_gate) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 499.0f; }
                    std::vector<uint32_t> ids(out_rows);
                    for (uint32_t t = 0; t < out_rows; ++t) ids[t] = (t * 5u + 2u) % e;
                    const auto xv = a.floats(x);
                    const auto yv = a.floats(init_gate, false);
                    const auto ix = a.upload(std::as_bytes(std::span(ids)));
                    g.set_verify_nax_scope(true);
                    g.begin();
                    g.verify_nax_moe_routed(rc.type, rc.layout, w, ix, {xv, in_rows, rc.k, uint64_t(rc.k) * 4}, {yv, out_rows, rc.r, uint64_t(rc.r) * 4}, e, s, broadcast, swiglu);
                    const auto batch = g.submit();
                    g.set_verify_nax_scope(false);
                    need(batch.dispatches == 1, "verify_nax_moe_routed must execute in 1 dispatch");
                    std::vector<float> got(out_rows * rc.r), ref(out_rows * rc.r);
                    g.read(yv, std::as_writable_bytes(std::span(got)));
                    for (uint32_t rid = 0; rid < out_rows; ++rid) {
                        const uint32_t ex = ids[rid], in_r = broadcast ? (rid / s) : rid;
                        for (uint32_t col = 0; col < rc.r; ++col) {
                            double dot = 0.0;
                            for (uint32_t c = 0; c < rc.k; ++c) {
                                dot += double(truth[(uint64_t(ex) * rc.r + col) * rc.k + c]) * double(x[uint64_t(in_r) * rc.k + c]);
                            }
                            float val = float(dot);
                            if (swiglu) {
                                const float gv = init_gate[uint64_t(rid) * rc.r + col];
                                const float sig = gv >= 0.0f ? 1.0f / (1.0f + std::exp(-gv)) : std::exp(gv) / (1.0f + std::exp(gv));
                                val = (gv * sig) * val;
                            }
                            ref[uint64_t(rid) * rc.r + col] = val;
                        }
                    }
                    need(nmse_f32(got, ref) <= 1e-10, "verify_nax_moe_routed FP32 NMSE exceeded 1e-10");
                }
            }
            // Heavy duplicate-expert collision check (m=8, s=4, out_rows=32, 16 routes sharing expert 2 across two 8-route batches):
            // coalesced verify_nax_moe_routed must be bit-for-bit identical (0 ULP) to uncoalesced unique-expert routing.
            {
                const uint32_t m_dup = 8u, s_dup = 4u, out_rows = m_dup * s_dup, in_rows = broadcast ? m_dup : out_rows;
                std::vector<uint32_t> ids_dup(out_rows), ids_uniq(out_rows);
                for (uint32_t rid = 0; rid < out_rows; ++rid) {
                    ids_dup[rid] = (rid % 2u == 0u) ? 2u : ((rid % 5u == 0u) ? 5u : (rid % 3u));
                    ids_uniq[rid] = rid;
                }
                const size_t expert_bytes = raw.size() / e;
                std::vector<std::byte> raw_uniq(expert_bytes * out_rows);
                for (uint32_t rid = 0; rid < out_rows; ++rid) {
                    std::memcpy(raw_uniq.data() + rid * expert_bytes, raw.data() + ids_dup[rid] * expert_bytes, expert_bytes);
                }
                auto payload_uniq = raw_uniq;
                if (rc.layout != PrefillWeightLayout::gguf) {
                    lanes::pack_rows(lanes::row_layout(rc.type, rc.k), raw_uniq, payload_uniq);
                }
                MetalGraph g_uniq(metallib, 256ull << 20);
                g_uniq.configure_verify_nax(true);
                Arena a_uniq{g_uniq, {}};
                const auto w_uniq = a_uniq.upload(payload_uniq);
                if (rc.layout != PrefillWeightLayout::gguf) {
                    g_uniq.register_lane_matrix(w_uniq, {rc.type, rc.k, rc.r, out_rows});
                }
                g_uniq.prepare_verify_nax(moe_layouts);

                std::vector<float> x(uint64_t(in_rows) * rc.k), init_gate(uint64_t(out_rows) * rc.r);
                uint32_t rng = 424242u + rc.type * 13u + uint32_t(broadcast);
                for (auto& v : x) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 997.0f; }
                for (auto& v : init_gate) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 499.0f; }

                const auto xv_d = a.floats(x), yv_d = a.floats(init_gate, false);
                const auto ix_d = a.upload(std::as_bytes(std::span(ids_dup)));
                g.set_verify_nax_scope(true);
                g.begin();
                g.verify_nax_moe_routed(rc.type, rc.layout, w, ix_d, {xv_d, in_rows, rc.k, uint64_t(rc.k) * 4},
                                        {yv_d, out_rows, rc.r, uint64_t(rc.r) * 4}, e, s_dup, broadcast, true);
                static_cast<void>(g.submit());
                g.set_verify_nax_scope(false);

                const auto xv_u = a_uniq.floats(x), yv_u = a_uniq.floats(init_gate, false);
                const auto ix_u = a_uniq.upload(std::as_bytes(std::span(ids_uniq)));
                g_uniq.set_verify_nax_scope(true);
                g_uniq.begin();
                g_uniq.verify_nax_moe_routed(rc.type, rc.layout, w_uniq, ix_u, {xv_u, in_rows, rc.k, uint64_t(rc.k) * 4},
                                             {yv_u, out_rows, rc.r, uint64_t(rc.r) * 4}, out_rows, s_dup, broadcast, true);
                static_cast<void>(g_uniq.submit());
                g_uniq.set_verify_nax_scope(false);

                std::vector<float> got_dup(out_rows * rc.r), got_uniq(out_rows * rc.r);
                g.read(yv_d, std::as_writable_bytes(std::span(got_dup)));
                g_uniq.read(yv_u, std::as_writable_bytes(std::span(got_uniq)));
                for (size_t i = 0; i < got_dup.size(); ++i) {
                    need(got_dup[i] == got_uniq[i], "coalesced verify_nax_moe_routed differed from unique-expert reference");
                }
            }
        }
    }
    for (const auto layout : {PrefillWeightLayout::gguf, PrefillWeightLayout::q8_lanes_v1}) {
        for (const uint32_t k : {96u, 128u, 512u, 768u}) {
            for (const uint32_t r : {131u, 1543u, 4103u}) {
                if (r > 2560u && k != 96u && k != 768u) continue;
                MetalGraph g(metallib, 256ull << 20);
                g.configure_verify_nax(true);
                Arena a{g, {}};
                const grouped_fixture::Shape f{8u, k, r, 1, 1, 1, true, layout};
                const auto raw = grouped_fixture::weights({8u, k, r, 1, 1, 1, true, PrefillWeightLayout::gguf});
                const auto truth = grouped_fixture::represented(f, raw);
                auto payload = raw;
                if (layout == PrefillWeightLayout::q8_lanes_v1) {
                    lanes::pack_rows(lanes::row_layout(8u, k), raw, payload);
                }
                const auto w = a.upload(payload);
                if (layout == PrefillWeightLayout::q8_lanes_v1) {
                    g.register_lane_matrix(w, {8u, k, r, 1});
                }
                auto raw_up = raw;
                for (size_t i = 0; i < raw_up.size(); ++i) {
                    if (i % 34u >= 2u) raw_up[i] ^= std::byte((i * 13u + 7u) & 0x0fu);
                }
                const auto truth_up = grouped_fixture::represented(f, raw_up);
                auto payload_up = raw_up;
                if (layout == PrefillWeightLayout::q8_lanes_v1) {
                    lanes::pack_rows(lanes::row_layout(8u, k), raw_up, payload_up);
                }
                const auto w_up = a.upload(payload_up);
                if (layout == PrefillWeightLayout::q8_lanes_v1) {
                    g.register_lane_matrix(w_up, {8u, k, r, 1});
                }
                g.prepare_verify_nax({});
                for (const bool swiglu : {false, true}) {
                    for (const uint32_t m : {1u, 3u, 4u, 5u, 8u}) {
                        if (r > 2560u && m != 4u && m != 8u) continue;
                        std::vector<float> x(uint64_t(m) * k), init_gate(uint64_t(m) * r);
                        uint32_t rng = 7717u + k * 5u + r * 7u + m * 19u + uint32_t(layout) * 3u + uint32_t(swiglu);
                        for (auto& v : x) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 997.0f; }
                        for (auto& v : init_gate) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 499.0f; }
                        const auto xv = a.floats(x);
                        const auto yv = a.floats(init_gate, false);
                        g.set_verify_nax_scope(true);
                        g.begin();
                        g.verify_nax_dense_q8(w, {xv, m, k, uint64_t(k) * 4}, {yv, m, r, uint64_t(r) * 4}, swiglu);
                        const auto batch = g.submit();
                        g.set_verify_nax_scope(false);
                        need(batch.dispatches == 1, "verify_nax_dense_q8 must execute in 1 dispatch");
                        std::vector<float> got(m * r), ref(m * r);
                        g.read(yv, std::as_writable_bytes(std::span(got)));
                        for (uint32_t row = 0; row < m; ++row) {
                            for (uint32_t col = 0; col < r; ++col) {
                                double dot = 0.0;
                                for (uint32_t c = 0; c < k; ++c) dot += double(truth[uint64_t(col) * k + c]) * double(x[uint64_t(row) * k + c]);
                                float val = float(dot);
                                if (swiglu) {
                                    const float gv = init_gate[uint64_t(row) * r + col];
                                    const float sig = gv >= 0.0f ? 1.0f / (1.0f + std::exp(-gv)) : std::exp(gv) / (1.0f + std::exp(gv));
                                    val = (gv * sig) * val;
                                }
                                ref[uint64_t(row) * r + col] = val;
                            }
                        }
                        need(nmse_f32(got, ref) <= 1e-10, "verify_nax_dense_q8 FP32 NMSE exceeded 1e-10");
                        if (r == 131u && !swiglu) {
                            const auto y_two = a.floats(init_gate, false);
                            const auto y_one = a.floats(init_gate, false);
                            g.set_verify_nax_scope(true);
                            g.begin();
                            g.verify_nax_dense_q8(w, {xv, m, k, uint64_t(k) * 4}, {y_two, m, r, uint64_t(r) * 4}, false);
                            g.verify_nax_dense_q8(w_up, {xv, m, k, uint64_t(k) * 4}, {y_two, m, r, uint64_t(r) * 4}, true);
                            const auto b_two = g.submit();
                            g.begin();
                            g.verify_nax_dense_q8_swiglu(w, w_up, {xv, m, k, uint64_t(k) * 4}, {y_one, m, r, uint64_t(r) * 4});
                            const auto b_one = g.submit();
                            g.set_verify_nax_scope(false);
                            need(b_two.dispatches == 2 && b_one.dispatches == 1, "verify_nax_dense_q8_swiglu must execute in 1 dispatch");
                            std::vector<float> got_two(m * r), got_one(m * r);
                            g.read(y_two, std::as_writable_bytes(std::span(got_two)));
                            g.read(y_one, std::as_writable_bytes(std::span(got_one)));
                            need(got_one == got_two, "verify_nax_dense_q8_swiglu differed bit-for-bit from two-pass SwiGLU");
                            std::vector<float> ref_one(m * r);
                            for (uint32_t row = 0; row < m; ++row) {
                                for (uint32_t col = 0; col < r; ++col) {
                                    double dg = 0.0, du = 0.0;
                                    for (uint32_t c = 0; c < k; ++c) {
                                        dg += double(truth[uint64_t(col) * k + c]) * double(x[uint64_t(row) * k + c]);
                                        du += double(truth_up[uint64_t(col) * k + c]) * double(x[uint64_t(row) * k + c]);
                                    }
                                    const float gv = float(dg), uv = float(du);
                                    const float sig = gv >= 0.0f ? 1.0f / (1.0f + std::exp(-gv)) : std::exp(gv) / (1.0f + std::exp(gv));
                                    ref_one[uint64_t(row) * r + col] = (gv * sig) * uv;
                                }
                            }
                            need(nmse_f32(got_one, ref_one) <= 1e-10, "verify_nax_dense_q8_swiglu FP32 NMSE exceeded 1e-10");
                        }
                    }
                }
            }
        }
        for (const uint32_t k : {96u, 256u, 768u}) {
            MetalGraph g(metallib, 256ull << 20);
            g.configure_verify_nax(true);
            Arena a{g, {}};
            const uint32_t r = 131u;
            const auto raw0 = grouped_fixture::weights({8u, k, r, 1, 1, 1, true, PrefillWeightLayout::gguf});
            auto raw1 = raw0;
            for (size_t b = 0; b + 34 <= raw1.size(); b += 34) {
                for (size_t q = 2; q < 34; ++q) {
                    const int v = int(static_cast<int8_t>(raw1[b + q]));
                    raw1[b + q] = static_cast<std::byte>(int8_t(((v + 3 + int(q)) % 15) - 7));
                }
            }
            auto pay0 = raw0, pay1 = raw1;
            if (layout == PrefillWeightLayout::q8_lanes_v1) {
                lanes::pack_rows(lanes::row_layout(8u, k), raw0, pay0);
                lanes::pack_rows(lanes::row_layout(8u, k), raw1, pay1);
            }
            const auto wg = a.upload(pay0), wu = a.upload(pay1);
            if (layout == PrefillWeightLayout::q8_lanes_v1) {
                g.register_lane_matrix(wg, {8u, k, r, 1});
                g.register_lane_matrix(wu, {8u, k, r, 1});
            }
            g.prepare_verify_nax({});
            for (const uint32_t m : {1u, 3u, 4u, 5u, 8u}) {
                std::vector<float> x(uint64_t(m) * k);
                uint32_t rng = 9923u + k * 7u + m * 17u + uint32_t(layout);
                for (auto& v : x) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 997.0f; }
                const auto xv = a.floats(x);
                const auto y_ref = a.zeros(uint64_t(m) * r * 4ull);
                const auto y_fus = a.zeros(uint64_t(m) * r * 4ull);
                g.set_verify_nax_scope(true);
                g.begin();
                g.verify_nax_dense_q8(wg, {xv, m, k, uint64_t(k) * 4}, {y_ref, m, r, uint64_t(r) * 4}, false);
                g.verify_nax_dense_q8(wu, {xv, m, k, uint64_t(k) * 4}, {y_ref, m, r, uint64_t(r) * 4}, true);
                const auto b_ref = g.submit();
                g.begin();
                g.verify_nax_gate_up_swiglu_q8(wg, wu, {xv, m, k, uint64_t(k) * 4}, {y_fus, m, r, uint64_t(r) * 4});
                const auto b_fus = g.submit();
                g.set_verify_nax_scope(false);
                need(b_ref.dispatches == 2 && b_fus.dispatches == 1, "verify_nax_gate_up_swiglu_q8 must replace 2 dispatches with 1");
                need(g.read(y_fus) == g.read(y_ref), "verify_nax_gate_up_swiglu_q8 differed bit-for-bit from unfused gate+up+SwiGLU");
            }
            std::vector<float> bad_x(4ull * k, 0.25f);
            bad_x[k * 2ull + k - 1u] = std::numeric_limits<float>::infinity();
            const auto xv_bad = a.floats(bad_x);
            const auto y_bad = a.zeros(4ull * r * 4ull);
            bool sticky = false;
            try {
                g.set_verify_nax_scope(true);
                g.begin();
                g.verify_nax_gate_up_swiglu_q8(wg, wu, {xv_bad, 4u, k, uint64_t(k) * 4}, {y_bad, 4u, r, uint64_t(r) * 4});
                static_cast<void>(g.submit());
            } catch (const FormatError&) {
                sticky = true;
            }
            g.set_verify_nax_scope(false);
            need(sticky, "verify_nax_gate_up_swiglu_q8 must report sticky nonfinite input error");
        }
    }
    for (const auto layout : {PrefillWeightLayout::gguf, PrefillWeightLayout::q8_lanes_v1}) {
        const uint32_t second_k = layout == PrefillWeightLayout::gguf ? 640u : 512u;
        for (const uint32_t k : {320u, second_k}) {
            for (const uint32_t r : {131u, 2563u}) {
                MetalGraph g(metallib, 256ull << 20);
                g.configure_verify_nax(true);
                g.configure_verify_nax_m8(true);
                Arena a{g, {}};
                const grouped_fixture::Shape f{8u, k, r, 1, 1, 1, true, layout};
                const auto raw = grouped_fixture::weights({8u, k, r, 1, 1, 1, true, PrefillWeightLayout::gguf});
                const auto truth = grouped_fixture::represented(f, raw);
                auto payload = raw;
                if (layout == PrefillWeightLayout::q8_lanes_v1) {
                    lanes::pack_rows(lanes::row_layout(8u, k), raw, payload);
                }
                const auto w = a.upload(payload);
                if (layout == PrefillWeightLayout::q8_lanes_v1) {
                    g.register_lane_matrix(w, {8u, k, r, 1});
                }
                g.prepare_verify_nax({});
                for (const int epilogue_mode : {0, 1, 2}) {
                    const bool swiglu = epilogue_mode == 1;
                    const float silu_scale = epilogue_mode == 2 ? 0.25f : 0.0f;
                    for (const uint32_t m : {1u, 3u, 5u, 8u}) {
                        std::vector<float> x(uint64_t(m) * k), init_gate(uint64_t(m) * r);
                        uint32_t rng = 9919u + m * 23u + k + r + uint32_t(layout) * 5u + uint32_t(epilogue_mode);
                        for (auto& v : x) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 997.0f; }
                        for (auto& v : init_gate) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 499.0f; }
                        const auto xv = a.floats(x);
                        const auto yv = a.floats(init_gate, false);
                        g.set_verify_nax_scope(true);
                        g.begin();
                        g.verify_nax_dense_q8(w, {xv, m, k, uint64_t(k) * 4}, {yv, m, r, uint64_t(r) * 4}, swiglu, silu_scale);
                        const auto batch = g.submit();
                        g.set_verify_nax_scope(false);
                        need(batch.dispatches == 1, "nax_m8_dense_q8 must execute in 1 dispatch");
                        std::vector<float> got(m * r), ref(m * r);
                        g.read(yv, std::as_writable_bytes(std::span(got)));
                        for (uint32_t row = 0; row < m; ++row) {
                            for (uint32_t col = 0; col < r; ++col) {
                                double dot = 0.0;
                                for (uint32_t c = 0; c < k; ++c) dot += double(truth[uint64_t(col) * k + c]) * double(x[uint64_t(row) * k + c]);
                                float val = float(dot);
                                if (swiglu) {
                                    const float gv = init_gate[uint64_t(row) * r + col];
                                    const float sig = gv >= 0.0f ? 1.0f / (1.0f + std::exp(-gv)) : std::exp(gv) / (1.0f + std::exp(gv));
                                    val = (gv * sig) * val;
                                } else if (silu_scale != 0.0f) {
                                    const float sv = val * silu_scale;
                                    const float sig = sv >= 0.0f ? 1.0f / (1.0f + std::exp(-sv)) : std::exp(sv) / (1.0f + std::exp(sv));
                                    val = sv * sig;
                                }
                                ref[uint64_t(row) * r + col] = val;
                            }
                        }
                        need(nmse_f32(got, ref) <= 2e-5, "nax_m8_dense_q8 NMSE exceeded 2e-5");
                    }
                }
            }
        }
    }
    for (const auto rc : cases) {
        MetalGraph g(metallib, 256ull << 20);
        g.configure_verify_nax(true);
        g.configure_verify_nax_m8(true);
        Arena a{g, {}};
        const uint32_t e = 8u;
        const grouped_fixture::Shape f{rc.type, rc.k, rc.r, e, 4u, 1, true, rc.layout};
        const auto raw = grouped_fixture::weights({rc.type, rc.k, rc.r, e, 4u, 1, true, PrefillWeightLayout::gguf});
        const auto truth = grouped_fixture::represented(f, raw);
        auto payload = raw;
        if (rc.layout != PrefillWeightLayout::gguf) {
            lanes::pack_rows(lanes::row_layout(rc.type, rc.k), raw, payload);
        }
        const auto w = a.upload(payload);
        if (rc.layout != PrefillWeightLayout::gguf) {
            g.register_lane_matrix(w, {rc.type, rc.k, rc.r, e});
        }
        const std::array<std::pair<uint32_t, PrefillWeightLayout>, 1> moe_layouts{{{rc.type, rc.layout}}};
        g.prepare_verify_nax(moe_layouts);
        for (const bool broadcast : {true, false}) {
            for (const bool swiglu : {false, true}) {
                // 1. Singleton path (match_count == 1): distinct experts across all 8 routes -> exact FP32 SIMT!
                {
                    const uint32_t m_s = 2u, s_s = 4u, out_rows = m_s * s_s, in_rows = broadcast ? m_s : out_rows;
                    std::vector<uint32_t> ids(out_rows);
                    for (uint32_t rid = 0; rid < out_rows; ++rid) ids[rid] = rid % e;
                    std::vector<float> x(uint64_t(in_rows) * rc.k), init_gate(uint64_t(out_rows) * rc.r);
                    uint32_t rng = 8811u + rc.type * 17u + uint32_t(broadcast) * 3u + uint32_t(swiglu);
                    for (auto& v : x) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 997.0f; }
                    for (auto& v : init_gate) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 499.0f; }
                    const auto xv = a.floats(x), yv = a.floats(init_gate, false);
                    const auto ix = a.upload(std::as_bytes(std::span(ids)));
                    g.set_verify_nax_scope(true);
                    g.begin();
                    g.verify_nax_moe_routed(rc.type, rc.layout, w, ix, {xv, in_rows, rc.k, uint64_t(rc.k) * 4},
                                            {yv, out_rows, rc.r, uint64_t(rc.r) * 4}, e, s_s, broadcast, swiglu);
                    static_cast<void>(g.submit());
                    g.set_verify_nax_scope(false);
                    std::vector<float> got(out_rows * rc.r), ref(out_rows * rc.r);
                    g.read(yv, std::as_writable_bytes(std::span(got)));
                    for (uint32_t rid = 0; rid < out_rows; ++rid) {
                        const uint32_t ex = ids[rid], in_r = broadcast ? (rid / s_s) : rid;
                        for (uint32_t col = 0; col < rc.r; ++col) {
                            double dot = 0.0;
                            for (uint32_t c = 0; c < rc.k; ++c) dot += double(truth[(uint64_t(ex) * rc.r + col) * rc.k + c]) * double(x[uint64_t(in_r) * rc.k + c]);
                            float val = float(dot);
                            if (swiglu) {
                                const float gv = init_gate[uint64_t(rid) * rc.r + col];
                                const float sig = gv >= 0.0f ? 1.0f / (1.0f + std::exp(-gv)) : std::exp(gv) / (1.0f + std::exp(gv));
                                val = (gv * sig) * val;
                            }
                            ref[uint64_t(rid) * rc.r + col] = val;
                        }
                    }
                    need(nmse_f32(got, ref) <= 1e-10, "nax_m8_moe singleton FP32 NMSE exceeded 1e-10");
                }
                // 2. Multi-route collision path (match_count >= 2): matmul2d(8, 32, 32) across 32 routes!
                {
                    const uint32_t m_d = 8u, s_d = 4u, out_rows = m_d * s_d, in_rows = broadcast ? m_d : out_rows;
                    std::vector<uint32_t> ids(out_rows);
                    for (uint32_t rid = 0; rid < out_rows; ++rid) {
                        ids[rid] = (rid % 2u == 0u) ? 2u : ((rid % 5u == 0u) ? 5u : (rid % 3u));
                    }
                    std::vector<float> x(uint64_t(in_rows) * rc.k), init_gate(uint64_t(out_rows) * rc.r);
                    uint32_t rng = 9922u + rc.type * 19u + uint32_t(broadcast) * 7u + uint32_t(swiglu);
                    for (auto& v : x) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 997.0f; }
                    for (auto& v : init_gate) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 499.0f; }
                    const auto xv = a.floats(x), yv = a.floats(init_gate, false);
                    const auto ix = a.upload(std::as_bytes(std::span(ids)));
                    g.set_verify_nax_scope(true);
                    g.begin();
                    g.verify_nax_moe_routed(rc.type, rc.layout, w, ix, {xv, in_rows, rc.k, uint64_t(rc.k) * 4},
                                            {yv, out_rows, rc.r, uint64_t(rc.r) * 4}, e, s_d, broadcast, swiglu);
                    static_cast<void>(g.submit());
                    g.set_verify_nax_scope(false);
                    std::vector<float> got(out_rows * rc.r), ref(out_rows * rc.r);
                    g.read(yv, std::as_writable_bytes(std::span(got)));
                    for (uint32_t rid = 0; rid < out_rows; ++rid) {
                        const uint32_t ex = ids[rid], in_r = broadcast ? (rid / s_d) : rid;
                        for (uint32_t col = 0; col < rc.r; ++col) {
                            double dot = 0.0;
                            for (uint32_t c = 0; c < rc.k; ++c) dot += double(truth[(uint64_t(ex) * rc.r + col) * rc.k + c]) * double(x[uint64_t(in_r) * rc.k + c]);
                            float val = float(dot);
                            if (swiglu) {
                                const float gv = init_gate[uint64_t(rid) * rc.r + col];
                                const float sig = gv >= 0.0f ? 1.0f / (1.0f + std::exp(-gv)) : std::exp(gv) / (1.0f + std::exp(gv));
                                val = (gv * sig) * val;
                            }
                            ref[uint64_t(rid) * rc.r + col] = val;
                        }
                    }
                    need(nmse_f32(got, ref) <= 5e-5, "nax_m8_moe multi-route NMSE exceeded 5e-5");
                }
            }
        }
    }
    for (const bool use_m8 : {false, true}) {
        for (const auto rc : cases) {
            MetalGraph g(metallib, 256ull << 20);
            g.configure_verify_nax(true);
            if (use_m8) g.configure_verify_nax_m8(true);
            Arena a{g, {}};
            const uint32_t e = 8u, h = (rc.type == 12u || rc.type == 13u) ? 256u : 640u;
            const grouped_fixture::Shape f_up{rc.type, rc.k, h, e, 4u, 1, true, rc.layout};
            const grouped_fixture::Shape f_dn{rc.type, h, rc.k, e, 4u, 1, true, rc.layout};
            const auto raw_gate = grouped_fixture::weights({rc.type, rc.k, h, e, 4u, 1, true, PrefillWeightLayout::gguf});
            auto raw_up = raw_gate;
            std::rotate(raw_up.begin(), raw_up.begin() + (raw_up.size() / e), raw_up.end());
            const auto raw_dn = grouped_fixture::weights({rc.type, h, rc.k, e, 4u, 1, true, PrefillWeightLayout::gguf});
            const auto truth_gate = grouped_fixture::represented(f_up, raw_gate);
            const auto truth_up = grouped_fixture::represented(f_up, raw_up);
            const auto truth_dn = grouped_fixture::represented(f_dn, raw_dn);
            auto pay_gate = raw_gate, pay_up = raw_up, pay_dn = raw_dn;
            if (rc.layout != PrefillWeightLayout::gguf) {
                lanes::pack_rows(lanes::row_layout(rc.type, rc.k), raw_gate, pay_gate);
                lanes::pack_rows(lanes::row_layout(rc.type, rc.k), raw_up, pay_up);
                lanes::pack_rows(lanes::row_layout(rc.type, h), raw_dn, pay_dn);
            }
            const auto wg = a.upload(pay_gate), wu = a.upload(pay_up), wd = a.upload(pay_dn);
            if (rc.layout != PrefillWeightLayout::gguf) {
                g.register_lane_matrix(wg, {rc.type, rc.k, h, e});
                g.register_lane_matrix(wu, {rc.type, rc.k, h, e});
                g.register_lane_matrix(wd, {rc.type, h, rc.k, e});
            }
            const std::array<std::pair<uint32_t, PrefillWeightLayout>, 1> moe_layouts{{{rc.type, rc.layout}}};
            g.prepare_verify_nax(moe_layouts);
            for (const uint32_t m_u : {2u, 8u}) {
                const uint32_t s_u = 4u, out_rows = m_u * s_u;
                std::vector<uint32_t> ids(out_rows);
                for (uint32_t rid = 0; rid < out_rows; ++rid) {
                    ids[rid] = (m_u == 2u) ? (rid % e) : ((rid % 2u == 0u) ? 2u : ((rid % 5u == 0u) ? 5u : (rid % 3u)));
                }
                std::vector<float> x(uint64_t(m_u) * rc.k);
                uint32_t rng = 5511u + rc.type * 13u + m_u * 7u + uint32_t(use_m8);
                for (auto& v : x) { rng = rng * 1664525u + 1013904223u; v = float(int(rng % 1993) - 996) / 997.0f; }
                const auto xv = a.floats(x);
                const auto gv = a.zeros(uint64_t(out_rows) * h * 4);
                const auto yv = a.zeros(uint64_t(out_rows) * rc.k * 4);
                const auto ix = a.upload(std::as_bytes(std::span(ids)));
                g.set_verify_nax_scope(true);
                g.begin();
                g.verify_nax_moe_union(rc.type, rc.layout, wg, wu, rc.type, rc.layout, wd, ix,
                    {xv, m_u, rc.k, uint64_t(rc.k) * 4}, {gv, out_rows, h, uint64_t(h) * 4},
                    {yv, out_rows, rc.k, uint64_t(rc.k) * 4}, e, s_u);
                const auto batch = g.submit();
                g.set_verify_nax_scope(false);
                need(batch.dispatches == 3, "verify_nax_moe_union must execute in 3 dispatches");
                std::vector<float> got(out_rows * rc.k), ref(out_rows * rc.k);
                g.read(yv, std::as_writable_bytes(std::span(got)));
                for (uint32_t rid = 0; rid < out_rows; ++rid) {
                    const uint32_t ex = ids[rid], in_r = rid / s_u;
                    std::vector<float> hid(h);
                    for (uint32_t col = 0; col < h; ++col) {
                        double dg = 0.0, du = 0.0;
                        for (uint32_t c = 0; c < rc.k; ++c) {
                            const double xv_c = double(x[uint64_t(in_r) * rc.k + c]);
                            dg += double(truth_gate[(uint64_t(ex) * h + col) * rc.k + c]) * xv_c;
                            du += double(truth_up[(uint64_t(ex) * h + col) * rc.k + c]) * xv_c;
                        }
                        const float g0 = float(dg), u0 = float(du);
                        const float sig = g0 >= 0.0f ? 1.0f / (1.0f + std::exp(-g0)) : std::exp(g0) / (1.0f + std::exp(g0));
                        hid[col] = (g0 * sig) * u0;
                    }
                    for (uint32_t col = 0; col < rc.k; ++col) {
                        double dd = 0.0;
                        for (uint32_t c = 0; c < h; ++c) dd += double(truth_dn[(uint64_t(ex) * rc.k + col) * h + c]) * double(hid[c]);
                        ref[uint64_t(rid) * rc.k + col] = float(dd);
                    }
                }
                const double tol = (use_m8 && m_u > 2u) ? 1e-4 : 1e-9;
                need(nmse_f32(got, ref) <= tol, "verify_nax_moe_union NMSE exceeded tolerance");
            }
        }
    }
    std::cout << "PASS verify_nax_moe_routed + verify_nax_moe_union + verify_nax_dense_q8 + verify_nax_gate_up_swiglu_q8 + nax_m8 (Q5_1/Q8_0/Q4_K/Q5_K, gguf + lanes_v1, swiglu=false/true, m=1..8)\n";
}

void test_dflash2_kernels(const char* metallib) {
    MetalGraph g(metallib, 256ull << 20);
    g.prepare_dflash2();
    Arena a{g, {}};
    std::mt19937 rng(20261102);

    // 1. dflash2_stream_mean: average 4 streams across M=5 rows into slot 2 of tap_slots=5
    {
        const uint32_t m = 5, c = 256, branches = 4, tap_slots = 5, slot = 2;
        std::vector<float> streams(uint64_t(m) * branches * c);
        for (auto& v : streams) v = float(int(rng() % 2001) - 1000) / 503.0f;
        const auto sv = a.floats(streams);
        const auto tv = a.zeros(uint64_t(m) * tap_slots * c * 4);
        g.begin();
        g.dflash2_stream_mean(sv, tv, m, c, slot, tap_slots, uint64_t(branches) * c * 4, uint64_t(tap_slots) * c * 4);
        const auto b = g.submit();
        need(b.dispatches == 1, "dflash2_stream_mean must dispatch once");
        std::vector<float> got(uint64_t(m) * tap_slots * c);
        g.read(tv, std::as_writable_bytes(std::span(got)));
        for (uint32_t row = 0; row < m; ++row) {
            for (uint32_t s = 0; s < tap_slots; ++s) {
                for (uint32_t ch = 0; ch < c; ++ch) {
                    const float actual = got[(uint64_t(row) * tap_slots + s) * c + ch];
                    if (s != slot) {
                        need(actual == 0.0f, "dflash2_stream_mean touched non-target slot");
                    } else {
                        float expected = 0.0f;
                        for (uint32_t br = 0; br < branches; ++br)
                            expected += streams[(uint64_t(row) * branches + br) * c + ch];
                        expected *= 0.25f;
                        need(std::abs(actual - expected) <= 1e-6f, "dflash2_stream_mean mismatch");
                    }
                }
            }
        }
    }

    // 2. dflash2_conv_f32: dynamic grouped 2-tap conv (rows=8, channels=2560, group_size=16) for finish=false and finish=true
    for (bool finish : {false, true}) {
        const uint32_t rows = 8, c = 2560, gs = 16, cg_count = c / gs, dyn_w = 4 * cg_count;
        std::vector<float> x(rows * c), dyn(rows * dyn_w), base(4 * c), res(rows * c);
        for (auto& v : x) v = float(int(rng() % 1999) - 999) / 701.0f;
        for (auto& v : dyn) v = float(int(rng() % 997) - 498) / 997.0f;
        for (auto& v : base) v = float(int(rng() % 991) - 495) / 991.0f;
        for (auto& v : res) v = float(int(rng() % 1501) - 750) / 601.0f;
        const auto xv = a.floats(x), dynv = a.floats(dyn), basev = a.floats(base), resv = a.floats(res), yv = a.zeros(rows * c * 4);
        g.set_dflash2_scope(true);
        g.begin();
        g.dflash2_conv_f32(xv, dynv, basev, resv, yv, c, finish, rows, gs);
        const auto b = g.submit();
        g.set_dflash2_scope(false);
        need(b.dispatches == 1, "dflash2_conv_f32 must dispatch once");
        std::vector<float> got(rows * c), ref(rows * c);
        g.read(yv, std::as_writable_bytes(std::span(got)));
        const uint32_t kind = finish ? 1u : 0u;
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t ch = 0; ch < c; ++ch) {
                const uint32_t cg = ch / gs;
                float acc = x[r * c + ch] * (base[(kind * 2u) * c + ch] + dyn[r * dyn_w + (kind * 2u) * cg_count + cg]);
                if (r > 0u) {
                    acc += x[(r - 1u) * c + ch] * (base[(kind * 2u + 1u) * c + ch] + dyn[r * dyn_w + (kind * 2u + 1u) * cg_count + cg]);
                }
                if (finish) acc += res[r * c + ch];
                ref[r * c + ch] = acc;
            }
        }
        need(nmse_f32(got, ref) <= 1e-11, "dflash2_conv_f32 NMSE exceeded 1e-11");
    }

    // 3. dflash2_select: sharded top-16 + edge score table across M=7 rows, V=4096
    {
        const uint32_t m = 7, v = 4096, rank = 256, anchor = 42;
        std::vector<float> logits(m * v), sel_hid(m * rank);
        for (uint32_t r = 0; r < m; ++r) {
            for (uint32_t col = 0; col < v; ++col) {
                logits[r * v + col] = float(int((r * 131 + col * 17) % 1993) - 996) / 251.0f;
            }
            logits[r * v + (300 + r * 37)] = 25.0f + float(r);
            logits[r * v + (900 + r * 19)] = 19.5f + float(r);
        }
        for (auto& h : sel_hid) h = float(int(rng() % 501) - 250) / 251.0f;
        const auto to_bf16 = [](float x) -> uint16_t {
            uint32_t bits = 0;
            std::memcpy(&bits, &x, 4);
            return uint16_t((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
        };
        const auto from_bf16 = [](uint16_t b) -> float {
            uint32_t bits = uint32_t(b) << 16u;
            float out = 0;
            std::memcpy(&out, &bits, 4);
            return out;
        };
        std::vector<uint16_t> pred_cb(v * rank), succ_cb(v * rank);
        for (auto& b : pred_cb) b = to_bf16(float(int(rng() % 401) - 200) / 256.0f);
        for (auto& b : succ_cb) b = to_bf16(float(int(rng() % 401) - 200) / 256.0f);

        const auto lv = a.floats(logits);
        const auto hid_v = a.floats(sel_hid);
        const auto pred_v = a.upload(std::as_bytes(std::span(pred_cb)));
        const auto succ_v = a.upload(std::as_bytes(std::span(succ_cb)));
        const auto pid_v = a.zeros(7ull * 8ull * 16ull * sizeof(uint32_t));
        const auto pval_v = a.zeros(7ull * 8ull * 16ull * sizeof(float));
        const uint64_t out_bytes = 7ull * 16ull * (sizeof(uint32_t) + sizeof(float)) + 7ull * 256ull * sizeof(float);
        const auto sel_out_v = a.zeros(out_bytes);

        g.set_dflash2_scope(true);
        g.begin();
        g.dflash2_select(lv, pid_v, pval_v, sel_out_v, hid_v, pred_v, succ_v, m, v, anchor);
        const auto b = g.submit();
        g.set_dflash2_scope(false);
        need(b.dispatches == 2, "dflash2_select must execute in 2 dispatches");

        std::vector<std::byte> out_buf(out_bytes);
        g.read(sel_out_v, out_buf);
        const auto* cands = reinterpret_cast<const uint32_t*>(out_buf.data());
        const auto* unary = reinterpret_cast<const float*>(out_buf.data() + 7ull * 16ull * sizeof(uint32_t));
        const auto* edges = reinterpret_cast<const float*>(out_buf.data() + 7ull * 16ull * (sizeof(uint32_t) + sizeof(float)));

        for (uint32_t r = 0; r < m; ++r) {
            need(cands[r * 16] == 300 + r * 37, "dflash2_select rank-0 id mismatch");
            need(cands[r * 16 + 1] == 900 + r * 19, "dflash2_select rank-1 id mismatch");
            need(std::abs(unary[r * 16] - (25.0f + float(r))) <= 1e-6f, "dflash2_select rank-0 unary mismatch");
            const uint32_t pred_tok = (r == 0) ? anchor : cands[(r - 1) * 16];
            const uint32_t succ_tok = cands[r * 16];
            double ref_edge = 0.0;
            for (uint32_t d = 0; d < rank; ++d) {
                ref_edge += double(from_bf16(pred_cb[pred_tok * rank + d])) * double(sel_hid[r * rank + d]) *
                            double(from_bf16(succ_cb[succ_tok * rank + d]));
            }
            need(std::abs(edges[r * 256] - float(ref_edge)) <= 1e-4f, "dflash2_select edge table score mismatch");
        }
    }

    // 4. dflash2_context_kv_write + dflash2_qkv_prepare + dflash2_attention across 2048-token ring wraparound (cache_len = 2052)
    {
        const uint32_t window = 2048u, c1 = 2040u, c2 = 12u, cache_len = c1 + c2;
        const uint32_t q_heads = 32u, kv_heads = 8u, head_dim = 128u, prop_rows = 8u;
        const float eps = 1e-6f;
        const auto to_bf16 = [](float x) -> uint16_t {
            uint32_t bits = 0;
            std::memcpy(&bits, &x, 4);
            return uint16_t((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
        };
        const auto from_bf16 = [](uint16_t b) -> float {
            uint32_t bits = uint32_t(b) << 16u;
            float out = 0;
            std::memcpy(&out, &bits, 4);
            return out;
        };
        const auto bf16_round = [&](float x) -> float { return from_bf16(to_bf16(x)); };

        std::vector<float> qk_norm_freq(320u);
        for (uint32_t i = 0; i < 256u; ++i) qk_norm_freq[i] = 0.85f + float(i % 17u) * 0.02f;
        const auto inv_f = rotary_frequencies(head_dim, 10000);
        std::copy(inv_f.begin(), inv_f.end(), qk_norm_freq.begin() + 256u);

        std::vector<float> ctx_kv(uint64_t(cache_len) * 2048ull);
        for (auto& v : ctx_kv) v = float(int(rng() % 1997) - 998) / 503.0f;
        std::vector<float> prop_qkv(uint64_t(prop_rows) * 6144ull);
        for (auto& v : prop_qkv) v = float(int(rng() % 1993) - 996) / 499.0f;

        const auto qknf_v = a.floats(qk_norm_freq);
        const auto ctx1_v = a.floats(std::span(ctx_kv).first(uint64_t(c1) * 2048ull));
        const auto ctx2_v = a.floats(std::span(ctx_kv).subspan(uint64_t(c1) * 2048ull, uint64_t(c2) * 2048ull));
        const auto pqkv_v = a.floats(prop_qkv);
        const auto ring_k_v = a.zeros(uint64_t(kv_heads) * window * head_dim * sizeof(uint16_t));
        const auto ring_v_v = a.zeros(uint64_t(kv_heads) * head_dim * window * sizeof(uint16_t));
        const auto queries_v = a.zeros(uint64_t(q_heads) * prop_rows * head_dim * sizeof(uint16_t));
        const auto qkeys_v = a.zeros(uint64_t(kv_heads) * prop_rows * head_dim * sizeof(uint16_t));
        const auto qvals_v = a.zeros(uint64_t(kv_heads) * head_dim * prop_rows * sizeof(uint16_t));
        const auto partials_v = a.zeros(8ull * 16ull * (32ull * 128ull + 64ull) * sizeof(float));
        const auto out_v = a.zeros(uint64_t(prop_rows) * q_heads * head_dim * sizeof(float));

        g.set_dflash2_scope(true);
        g.begin();
        g.dflash2_context_kv_write(ctx1_v, qknf_v, ring_k_v, ring_v_v, c1, 0u, window, eps);
        g.dflash2_context_kv_write(ctx2_v, qknf_v, ring_k_v, ring_v_v, c2, c1, window, eps);
        g.dflash2_qkv_prepare(pqkv_v, qknf_v, queries_v, qkeys_v, qvals_v, cache_len, prop_rows, eps);
        g.dflash2_attention(queries_v, ring_k_v, ring_v_v, qkeys_v, qvals_v, partials_v, out_v, cache_len, window);
        const auto b = g.submit();
        g.set_dflash2_scope(false);
        need(b.dispatches == 5, "dflash2 wraparound attention batch expected 5 dispatches");

        auto norm_rope_bf16 = [&](const float* src, const float* nw, uint32_t pos, float* dst) {
            double sq = 0.0;
            for (uint32_t d = 0; d < head_dim; ++d) sq += double(src[d]) * double(src[d]);
            const float inv_rms = 1.0f / std::sqrt(float(sq / double(head_dim)) + eps);
            std::array<float, 128> normed{};
            for (uint32_t d = 0; d < head_dim; ++d) normed[d] = bf16_round(src[d] * inv_rms * nw[d]);
            for (uint32_t d = 0; d < head_dim / 2u; ++d) {
                const float ang = float(pos) * qk_norm_freq[256u + d];
                const float cs = std::cos(ang), sn = std::sin(ang);
                dst[d] = bf16_round(normed[d] * cs - normed[d + 64u] * sn);
                dst[d + 64u] = bf16_round(normed[d + 64u] * cs + normed[d] * sn);
            }
        };

        std::vector<float> ref_ctx_k(uint64_t(cache_len) * kv_heads * head_dim);
        std::vector<float> ref_ctx_v(uint64_t(cache_len) * kv_heads * head_dim);
        for (uint32_t p = 0; p < cache_len; ++p) {
            for (uint32_t kh = 0; kh < kv_heads; ++kh) {
                norm_rope_bf16(&ctx_kv[uint64_t(p) * 2048ull + kh * head_dim], &qk_norm_freq[128u], p,
                               &ref_ctx_k[(uint64_t(p) * kv_heads + kh) * head_dim]);
                for (uint32_t d = 0; d < head_dim; ++d) {
                    ref_ctx_v[(uint64_t(p) * kv_heads + kh) * head_dim + d] =
                        bf16_round(ctx_kv[uint64_t(p) * 2048ull + 1024ull + kh * head_dim + d]);
                }
            }
        }

        std::vector<float> ref_q(uint64_t(prop_rows) * q_heads * head_dim);
        std::vector<float> ref_qk(uint64_t(prop_rows) * kv_heads * head_dim);
        std::vector<float> ref_qv(uint64_t(prop_rows) * kv_heads * head_dim);
        for (uint32_t r = 0; r < prop_rows; ++r) {
            const float* row = &prop_qkv[uint64_t(r) * 6144ull];
            for (uint32_t qh = 0; qh < q_heads; ++qh) {
                norm_rope_bf16(row + qh * head_dim, &qk_norm_freq[0], cache_len + r,
                               &ref_q[(uint64_t(r) * q_heads + qh) * head_dim]);
            }
            for (uint32_t kh = 0; kh < kv_heads; ++kh) {
                norm_rope_bf16(row + 4096u + kh * head_dim, &qk_norm_freq[128u], cache_len + r,
                               &ref_qk[(uint64_t(r) * kv_heads + kh) * head_dim]);
                for (uint32_t d = 0; d < head_dim; ++d) {
                    ref_qv[(uint64_t(r) * kv_heads + kh) * head_dim + d] = bf16_round(row[5120u + kh * head_dim + d]);
                }
            }
        }

        std::vector<float> got_attn(uint64_t(prop_rows) * q_heads * head_dim), ref_attn(uint64_t(prop_rows) * q_heads * head_dim, 0.0f);
        g.read(out_v, std::as_writable_bytes(std::span(got_attn)));
        for (uint32_t r = 0; r < prop_rows; ++r) {
            const uint32_t qpos = cache_len + r;
            const uint32_t row_start = qpos >= (window - 1u) ? (qpos - (window - 1u)) : 0u;
            const uint32_t ctx_n = cache_len - row_start;
            for (uint32_t qh = 0; qh < q_heads; ++qh) {
                const uint32_t kh = qh / 4u;
                const float* qvec = &ref_q[(uint64_t(r) * q_heads + qh) * head_dim];
                std::vector<double> scores(ctx_n + prop_rows);
                double max_s = -1e30;
                for (uint32_t i = 0; i < ctx_n; ++i) {
                    const float* kvec = &ref_ctx_k[(uint64_t(row_start + i) * kv_heads + kh) * head_dim];
                    double dot = 0.0;
                    for (uint32_t d = 0; d < head_dim; ++d) dot += double(qvec[d]) * double(kvec[d]);
                    scores[i] = dot * 0.08838834765;
                    max_s = std::max(max_s, scores[i]);
                }
                for (uint32_t t = 0; t < prop_rows; ++t) {
                    const float* kvec = &ref_qk[(uint64_t(t) * kv_heads + kh) * head_dim];
                    double dot = 0.0;
                    for (uint32_t d = 0; d < head_dim; ++d) dot += double(qvec[d]) * double(kvec[d]);
                    scores[ctx_n + t] = dot * 0.08838834765;
                    max_s = std::max(max_s, scores[ctx_n + t]);
                }
                double sum_w = 0.0;
                for (auto& sc : scores) { sc = std::exp(sc - max_s); sum_w += sc; }
                for (uint32_t d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (uint32_t i = 0; i < ctx_n; ++i) {
                        acc += scores[i] * double(ref_ctx_v[(uint64_t(row_start + i) * kv_heads + kh) * head_dim + d]);
                    }
                    for (uint32_t t = 0; t < prop_rows; ++t) {
                        acc += scores[ctx_n + t] * double(ref_qv[(uint64_t(t) * kv_heads + kh) * head_dim + d]);
                    }
                    ref_attn[(uint64_t(r) * q_heads + qh) * head_dim + d] = float(acc / sum_w);
                }
            }
        }
        const double attn_nmse = nmse_f32(got_attn, ref_attn);
        need(attn_nmse <= 2e-4, "dflash2 ring-wraparound attention NMSE exceeded 2e-4");
        std::cout << "PASS dflash2 ring-wraparound attention(cache_len=" << cache_len << ", nmse=" << attn_nmse << ")\n";
    }
    std::cout << "PASS dflash2 Metal kernels (stream_mean, conv_f32, top16_sharded + select_edges, context_kv_write + qkv_prepare + attention)\n";
}

void test_dflash2_verify_budget() {
    DFlash2VerifyBudget budget;
    const auto seed_costs = budget.costs();
    need(std::abs(seed_costs[1] - 0.60f) <= 1e-6f && std::abs(seed_costs[8] - 1.0f) <= 1e-6f,
         "DFlash2VerifyBudget seed cost endpoints mismatch");
    for (uint32_t r = 1; r < 8u; ++r) {
        need(seed_costs[r] <= seed_costs[r + 1u], "DFlash2VerifyBudget seed costs must be monotone");
    }

    const std::array<float, 7> low{0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f};
    const std::array<float, 7> mid{0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f};
    const std::array<float, 7> high{0.98f, 0.98f, 0.98f, 0.98f, 0.98f, 0.98f, 0.98f};
    need(budget.choose_verify_rows(low, 8u) == 1u, "DFlash2VerifyBudget must shrink low-confidence block to 1 row");
    need(budget.choose_verify_rows(high, 8u) == 8u, "DFlash2VerifyBudget must retain 8 rows on high-confidence block");

    // Unbiased observation labelling:
    // 1) verify_rows == 1 -> 0 labels recorded
    budget.observe(low, 1u, 0u, 35.0, true);
    need(budget.cycles() == 1u && budget.label_count() == 0u, "verify_rows=1 must not record proposal labels");

    // 2) verify_rows == 4, accepted == 3 -> all 3 verified proposals accepted; unverified pos=3 not labelled
    budget.observe(high, 4u, 3u, 55.0, true);
    need(budget.cycles() == 2u && budget.label_count() == 3u, "verify_rows=4, accepted=3 must record 3 labels");
    {
        const auto lbs = budget.labels();
        for (uint32_t i = 0; i < 3u; ++i) {
            need(lbs[i].position == i && lbs[i].accepted, "accepted prefix labels mismatch");
        }
    }

    // 3) verify_rows == 4, accepted == 1 -> pos=0 accepted, pos=1 rejected, pos=2 unlabelled (prefix broke)
    budget.observe(low, 4u, 1u, 55.0, true);
    need(budget.cycles() == 3u && budget.label_count() == 5u, "verify_rows=4, accepted=1 must record 2 labels");
    {
        const auto lbs = budget.labels();
        need(lbs[3].position == 0u && lbs[3].accepted, "pos=0 label must be accepted");
        need(lbs[4].position == 1u && !lbs[4].accepted, "pos=1 label must be rejected");
    }

    // Advance to cycle 31 (exploration round): explores() must be true and choose_verify_rows must return max_rows
    while (budget.cycles() < 31u) {
        budget.observe(low, 1u, 0u, 35.0, true);
    }
    need(budget.explores(), "cycle 31 must trigger exploration");
    need(budget.choose_verify_rows(low, 8u) == 8u && budget.choose_verify_rows(low, 5u) == 5u,
         "exploration cycle must probe full max_rows");

    // IRLS calibration direction over 64 cycles: high-acceptance regime vs zero-acceptance regime
    DFlash2VerifyBudget b_hi, b_lo;
    for (uint32_t c = 0; c < 64u; ++c) {
        b_hi.observe(mid, 8u, 7u, 60.0, true);
        b_lo.observe(mid, 8u, 0u, 60.0, true);
    }
    need(b_hi.acceptance().intercepts[0] > b_lo.acceptance().intercepts[0],
         "IRLS refit must raise acceptance intercept under high acceptance vs zero acceptance");
    need(b_hi.choose_verify_rows(mid, 8u) > b_lo.choose_verify_rows(mid, 8u),
         "calibrated high-acceptance budget must choose strictly more verify rows than zero-acceptance budget");
    std::cout << "PASS DFlash2VerifyBudget (seed costs, unbiased prefix labels, 1-in-32 exploration, 64-cycle IRLS calibration)\n";
}

void test_nax_dense_q8_swiglu_and_fast_router(const char* metallib, bool bounded_policy = false) {
    if (bounded_policy) {
        // Production keeps multiseg off: its provenance guard deliberately
        // makes fused prefill unreachable. Verify that guard, not the retired
        // implementation. Live M8 verify/DFlash fusion is tested separately.
        MetalGraph g(metallib, 256ull << 20);
        g.configure_prefill_dense_f32(true);
        g.configure_prefill_dense_nax(true);
        const auto before = g.memory();
        const auto dispatches = g.lifetime_dispatches();
        bool refused = false;
        try { g.configure_prefill_multiseg(true); }
        catch (const FormatError& error) {
            refused = std::string(error.what()) == "this optimisation was removed because its source could not be verified";
        }
        need(refused, "retired prefill multiseg must retain its provenance refusal");
        for (const auto layout : {PrefillWeightLayout::gguf, PrefillWeightLayout::q8_lanes_v1})
            need(!g.can_nax_dense_q8_swiglu(layout, layout, 256u, 96u, 64u),
                 "retired multiseg refusal enabled fused prefill");
        need(g.memory().live_bytes == before.live_bytes && g.memory().allocations == before.allocations &&
             g.lifetime_dispatches() == dispatches, "retired multiseg refusal allocated or dispatched work");
        std::cout << "PASS retired prefill multiseg provenance refusal; fused prefill and router execution are not selected\n";
        return;
    }
    for (const auto layout : {PrefillWeightLayout::gguf, PrefillWeightLayout::q8_lanes_v1}) {
        for (const auto [k, r] : std::array<std::pair<uint32_t, uint32_t>, 3>{{{64u, 64u}, {256u, 96u}, {2560u, 640u}}}) {
            MetalGraph g_ref(metallib, 512ull << 20);
            g_ref.configure_prefill_dense_f32(true);
            g_ref.configure_prefill_dense_nax(true);
            MetalGraph g_fus(metallib, 512ull << 20);
            g_fus.configure_prefill_dense_f32(true);
            g_fus.configure_prefill_dense_nax(true);
            g_fus.configure_prefill_multiseg(true);
            need(g_fus.configure_profiling(2) != ProfileMode::disabled, "fused prefill route proof requires profiling support");
            need(!g_ref.can_nax_dense_q8_swiglu(layout, layout, k, r, 16u), "NAX dense SwiGLU must stay opt-in without prefill_multiseg");
            need(g_fus.can_nax_dense_q8_swiglu(layout, layout, k, r, 16u), "NAX dense SwiGLU must admit valid multiseg NAX shape");
            need(!g_fus.can_nax_dense_q8_swiglu(layout, layout, k, 71u, 16u), "NAX dense SwiGLU must reject hidden not divisible by 32");

            Arena a_ref{g_ref, {}}, a_fus{g_fus, {}};
            const grouped_fixture::Shape f{8, k, r, 1, 1, 1, true, layout};
            const auto raw0 = grouped_fixture::weights({8, k, r, 1, 1, 1, true, PrefillWeightLayout::gguf});
            auto raw1 = raw0;
            for (size_t b = 0; b + 34 <= raw1.size(); b += 34) {
                for (size_t q = 2; q < 34; ++q) {
                    const int v = int(static_cast<int8_t>(raw1[b + q]));
                    raw1[b + q] = static_cast<std::byte>(int8_t(((v + 5 + int(q)) % 15) - 7));
                }
            }
            const auto truth0 = grouped_fixture::represented(f, raw0);
            const auto truth1 = grouped_fixture::represented(f, raw1);
            std::vector<std::byte> pay0 = raw0, pay1 = raw1;
            if (layout == PrefillWeightLayout::q8_lanes_v1) {
                const auto rl = lanes::row_layout(8, k);
                lanes::pack_rows(rl, raw0, pay0);
                lanes::pack_rows(rl, raw1, pay1);
            }
            const auto w0_ref = a_ref.upload(pay0), w1_ref = a_ref.upload(pay1);
            const auto w0_fus = a_fus.upload(pay0), w1_fus = a_fus.upload(pay1);
            if (layout == PrefillWeightLayout::q8_lanes_v1) {
                g_ref.register_lane_matrix(w0_ref, {8, k, r, 1});
                g_ref.register_lane_matrix(w1_ref, {8, k, r, 1});
                g_fus.register_lane_matrix(w0_fus, {8, k, r, 1});
                g_fus.register_lane_matrix(w1_fus, {8, k, r, 1});
            } else {
                g_ref.register_q8_matrix(w0_ref, k, r);
                g_ref.register_q8_matrix(w1_ref, k, r);
                g_fus.register_q8_matrix(w0_fus, k, r);
                g_fus.register_q8_matrix(w1_fus, k, r);
            }
            g_ref.prepare_dense_mm_q8(layout);
            g_fus.prepare_dense_mm_q8(layout);
            need_tile64_state(g_ref, 2);
            need_tile64_state(g_fus, 2);

            for (const uint32_t n : {2u, 7u, 32u, 63u, 64u, 129u, 513u}) {
                std::vector<float> x(uint64_t(n) * k);
                uint32_t rng = 9001u + n * 13u + k;
                for (auto& v : x) {
                    rng = rng * 1664525u + 1013904223u;
                    v = float(int(rng % 1993) - 996) / 997.0f;
                }
                const uint64_t ys = uint64_t(r) * 4 + 16;
                const auto xv_ref = a_ref.floats(x), xv_fus = a_fus.floats(x);
                const auto g_out = a_ref.zeros(uint64_t(n) * r * 4);
                const auto u_out = a_ref.zeros(uint64_t(n) * r * 4);
                const auto f_out = a_fus.zeros(uint64_t(n) * ys);

                g_ref.begin();
                g_ref.dense_mm_q8(layout, w0_ref, {xv_ref, n, k, uint64_t(k) * 4}, {g_out, n, r, uint64_t(r) * 4});
                g_ref.dense_mm_q8(layout, w1_ref, {xv_ref, n, k, uint64_t(k) * 4}, {u_out, n, r, uint64_t(r) * 4}, true);
                g_ref.swiglu(g_out, u_out, g_out);
                const auto b_ref = g_ref.submit();

                const uint64_t ms_before = g_fus.prefill_multiseg_calls();
                g_fus.begin();
                g_fus.nax_dense_q8_swiglu(layout, w0_fus, layout, w1_fus, {xv_fus, n, k, uint64_t(k) * 4}, {f_out, n, r, ys});
                const auto b_fus = g_fus.submit();
                need(b_fus.dispatches == 2 && b_ref.dispatches == 5, "NAX dense SwiGLU must replace 5 dispatches with 2");
                need_kernel(g_fus, 1, q8_tile64_cache_enabled
                    ? (n >= 64 ? "nax_prefill_dense_q8_swiglu_t64_m32" : "nax_prefill_dense_q8_swiglu_t64_m8")
                    : (n >= 64 ? "nax_prefill_dense_q8_swiglu_m32" : "nax_prefill_dense_q8_swiglu_m8"), 2);
                need_tile64_state(g_fus, 2);
                need(g_fus.prefill_multiseg_calls() == ms_before + 1, "NAX dense SwiGLU must publish prefill_multiseg_calls");

                const auto ref_bytes = g_ref.read(g_out);
                const auto fus_bytes = g_fus.read(f_out);
                std::vector<float> unfused(uint64_t(n) * r), raw_fus(fus_bytes.size() / 4), got(uint64_t(n) * r), truth_y(uint64_t(n) * r);
                std::memcpy(unfused.data(), ref_bytes.data(), ref_bytes.size());
                std::memcpy(raw_fus.data(), fus_bytes.data(), fus_bytes.size());
                for (uint32_t row = 0; row < n; ++row) {
                    for (uint32_t col = 0; col < r; ++col) {
                        got[uint64_t(row) * r + col] = raw_fus[uint64_t(row) * (ys / 4) + col];
                        double dg = 0.0, du = 0.0;
                        for (uint32_t c = 0; c < k; ++c) {
                            const double xv = double(x[uint64_t(row) * k + c]);
                            dg += double(truth0[uint64_t(col) * k + c]) * xv;
                            du += double(truth1[uint64_t(col) * k + c]) * xv;
                        }
                        const double sig = 1.0 / (1.0 + std::exp(-dg));
                        truth_y[uint64_t(row) * r + col] = float((dg * sig) * du);
                    }
                }
                need(nmse_f32(got, truth_y) <= 2e-4, "NAX dense SwiGLU NMSE vs FP64 truth exceeded 2e-4");
                need(nmse_f32(got, unfused) <= 1e-11, "NAX dense SwiGLU drifted from unfused NAX dense + SwiGLU");
            }
        }
    }

    // Verify 1-pass BF16 fast NAX router parity, kernel identity, refusal and sticky nonfinite error.
    {
        MetalGraph g_bad(metallib, 256ull << 20);
        g_bad.configure_prefill_router_window(true);
        g_bad.configure_prefill_router_fast(true);
        bool refused = false;
        try { g_bad.prepare_moe_router_window(); } catch (const FormatError&) { refused = true; }
        need(refused, "fast BF16 router must refuse non-NAX router window preparation");
    }
    {
        MetalGraph g_exact(metallib, 512ull << 20);
        g_exact.configure_prefill_dense_f32(true);
        g_exact.configure_prefill_dense_nax(true);
        g_exact.configure_prefill_router_window(true);
        need(std::string(g_exact.moe_router_window_projection_kernel()) == "nax_router_f32", "default NAX router kernel identity");
        g_exact.prepare_moe_router_window();

        MetalGraph g_fast(metallib, 512ull << 20);
        g_fast.configure_prefill_dense_f32(true);
        g_fast.configure_prefill_dense_nax(true);
        g_fast.configure_prefill_router_window(true);
        g_fast.configure_prefill_router_fast(true);
        need(g_fast.prefill_router_fast() && std::string(g_fast.moe_router_window_projection_kernel()) == "nax_router_bf16_fast",
             "fast BF16 NAX router kernel identity");
        g_fast.prepare_moe_router_window();

        Arena a_ex{g_exact, {}}, a_fa{g_fast, {}};
        std::mt19937 rng(20261103u);
        std::vector<float> rw(2560ull * 512ull);
        for (auto& v : rw) v = float(int(rng() % 2001) - 1000) * 0.0005f;
        const auto w_ex = a_ex.floats(rw), w_fa = a_fa.floats(rw);

        for (const uint32_t n : {2u, 17u, 65u, 513u}) {
            std::vector<float> x(uint64_t(n) * 2560ull);
            for (auto& v : x) v = float(int(rng() % 2001) - 1000) * 0.001f;
            const auto x_ex = a_ex.floats(x), x_fa = a_fa.floats(x);
            const auto y_ex = a_ex.zeros(uint64_t(n) * 512ull * 4ull);
            const auto y_fa = a_fa.zeros(uint64_t(n) * 512ull * 4ull);

            g_exact.begin();
            g_exact.moe_router_f32_window(w_ex, {x_ex, n, 2560u, 2560ull * 4ull}, {y_ex, n, 512u, 512ull * 4ull});
            const auto b_ex = g_exact.submit();
            g_fast.begin();
            g_fast.moe_router_f32_window(w_fa, {x_fa, n, 2560u, 2560ull * 4ull}, {y_fa, n, 512u, 512ull * 4ull});
            const auto b_fa = g_fast.submit();
            need(b_ex.dispatches == 2 && b_fa.dispatches == 2, "NAX router dispatch count");

            const auto ex_b = g_exact.read(y_ex), fa_b = g_fast.read(y_fa);
            std::vector<float> got_ex(uint64_t(n) * 512ull), got_fa(uint64_t(n) * 512ull);
            std::memcpy(got_ex.data(), ex_b.data(), ex_b.size());
            std::memcpy(got_fa.data(), fa_b.data(), fa_b.size());
            need(nmse_f32(got_fa, got_ex) <= 1e-4, "fast BF16 NAX router NMSE exceeded 1e-4 vs Sterbenz router");
        }

        // Sticky nonfinite router weight check.
        auto bad_rw = rw;
        bad_rw[2560u * 7u + 13u] = std::numeric_limits<float>::infinity();
        const auto w_nan = a_fa.floats(bad_rw);
        std::vector<float> x2(2ull * 2560ull, 0.25f);
        const auto x_nan = a_fa.floats(x2);
        const auto y_nan = a_fa.zeros(2ull * 512ull * 4ull);
        bool sticky_caught = false;
        try {
            g_fast.begin();
            g_fast.moe_router_f32_window(w_nan, {x_nan, 2u, 2560u, 2560ull * 4ull}, {y_nan, 2u, 512u, 512ull * 4ull});
            static_cast<void>(g_fast.submit());
        } catch (const FormatError&) {
            sticky_caught = true;
        }
        need(sticky_caught, "fast BF16 NAX router must report sticky nonfinite weight error");
    }
    std::cout << "PASS NAX fused dense Q8 SwiGLU (m8 + m32, GGUF + q8_lanes_v1) and fast BF16 NAX router\n";
}

void test_nax_dense_bf16_and_verify_bf16(const char* metallib) {
    const auto to_bf16 = [](float x) -> uint16_t {
        uint32_t bits = 0;
        std::memcpy(&bits, &x, 4);
        return uint16_t((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
    };
    const auto from_bf16 = [](uint16_t b) -> float {
        uint32_t bits = uint32_t(b) << 16u;
        float out = 0;
        std::memcpy(&out, &bits, 4);
        return out;
    };

    // 1. NAX prefill dense BF16 (m8 + m32, reuse_bf16=false and reuse_bf16=true after dense_mm_q8, strided output, sticky nonfinite input & BF16 weight).
    for (const auto [k, r] : std::array<std::pair<uint32_t, uint32_t>, 3>{{{128u, 64u}, {256u, 128u}, {2560u, 512u}}}) {
        MetalGraph g(metallib, 512ull << 20);
        g.configure_prefill_dense_f32(true);
        g.configure_prefill_dense_nax(true);
        g.prepare_dense_mm_q8(PrefillWeightLayout::gguf);
        Arena a{g, {}};
        std::mt19937 rng(20261104u + k + r);
        std::vector<uint16_t> w_bf16(uint64_t(r) * k);
        for (auto& b : w_bf16) b = to_bf16(float(int(rng() % 2001) - 1000) / 1009.0f);
        const auto wv = a.upload(std::as_bytes(std::span(w_bf16)));
        const auto q8_raw = grouped_fixture::weights({8u, k, 64u, 1, 1, 1, true, PrefillWeightLayout::gguf});
        const auto q8v = a.upload(q8_raw);

        for (const uint32_t n : {2u, 17u, 32u, 65u, 129u}) {
            std::vector<float> x(uint64_t(n) * k);
            for (auto& v : x) v = float(int(rng() % 1993) - 996) / 997.0f;
            const uint64_t ys = uint64_t(r) * 4ull + 16ull;
            const auto xv = a.floats(x);
            const auto y_fresh = a.zeros(uint64_t(n) * ys);
            const auto y_reuse = a.zeros(uint64_t(n) * ys);
            const auto y_q8 = a.zeros(uint64_t(n) * 64ull * 4ull);

            g.begin();
            g.nax_dense_bf16(wv, {xv, n, k, uint64_t(k) * 4ull}, {y_fresh, n, r, ys}, false);
            const auto b_fresh = g.submit();
            need(b_fresh.dispatches == 2, "nax_dense_bf16(reuse=false) must execute in 2 dispatches");

            g.begin();
            g.dense_mm_q8(PrefillWeightLayout::gguf, q8v, {xv, n, k, uint64_t(k) * 4ull}, {y_q8, n, 64u, 64ull * 4ull}, false);
            need(g.can_reuse_nax_dense_bf16({xv, n, k, uint64_t(k) * 4ull}), "dense_mm_q8 must leave x_bf16 reusable");
            g.nax_dense_bf16(wv, {xv, n, k, uint64_t(k) * 4ull}, {y_reuse, n, r, ys}, true);
            const auto b_reuse = g.submit();
            need(b_reuse.dispatches == 3, "dense_mm_q8 + nax_dense_bf16(reuse=true) must execute in 2 + 1 = 3 dispatches");

            const auto fresh_b = g.read(y_fresh), reuse_b = g.read(y_reuse);
            need(fresh_b == reuse_b, "nax_dense_bf16(reuse=true) must match reuse=false bit-for-bit");
            std::vector<float> raw_out(fresh_b.size() / 4ull), got(uint64_t(n) * r), ref_bf16(uint64_t(n) * r);
            std::memcpy(raw_out.data(), fresh_b.data(), fresh_b.size());
            for (uint32_t row = 0; row < n; ++row) {
                for (uint32_t col = 0; col < r; ++col) {
                    got[uint64_t(row) * r + col] = raw_out[uint64_t(row) * (ys / 4ull) + col];
                    double acc = 0.0;
                    for (uint32_t c = 0; c < k; ++c) {
                        acc += double(from_bf16(to_bf16(x[uint64_t(row) * k + c]))) * double(from_bf16(w_bf16[uint64_t(col) * k + c]));
                    }
                    ref_bf16[uint64_t(row) * r + col] = float(acc);
                    ++total_checks;
                }
            }
            need(nmse_f32(got, ref_bf16) <= 1e-10, "nax_dense_bf16 drifted from exact BF16xBF16->FP32 reference");
        }

        // Sticky nonfinite BF16 weight check (0x7f80 = +inf).
        auto bad_w = w_bf16;
        bad_w[uint64_t(r / 2u) * k + 19u] = 0x7f80u;
        const auto w_bad = a.upload(std::as_bytes(std::span(bad_w)));
        std::vector<float> x_ok(17ull * k, 0.25f);
        const auto xv_ok = a.floats(x_ok);
        const auto y_bad = a.zeros(17ull * r * 4ull);
        bool caught = false;
        try {
            g.begin();
            g.nax_dense_bf16(w_bad, {xv_ok, 17u, k, uint64_t(k) * 4ull}, {y_bad, 17u, r, uint64_t(r) * 4ull}, false);
            static_cast<void>(g.submit());
        } catch (const FormatError&) {
            caught = true;
        }
        need(caught, "nax_dense_bf16 must report sticky nonfinite BF16 weight");
    }

    // 2. Verify batched BF16 projection (r1_m4, r1_m8 for k <= 2560, and m4, m8 when r > 2560; m=1..8; sticky input & BF16 weight).
    for (const auto [k, r] : std::array<std::pair<uint32_t, uint32_t>, 3>{{{128u, 128u}, {2560u, 512u}, {256u, 2600u}}}) {
        MetalGraph g(metallib, 256ull << 20);
        g.configure_verify_nax(true);
        g.prepare_verify_nax({});
        Arena a{g, {}};
        std::mt19937 rng(70701u + k + r);
        std::vector<uint16_t> w_bf16(uint64_t(r) * k);
        for (auto& b : w_bf16) b = to_bf16(float(int(rng() % 2001) - 1000) / 1009.0f);
        const auto wv = a.upload(std::as_bytes(std::span(w_bf16)));

        for (const uint32_t m : {1u, 3u, 4u, 5u, 8u}) {
            if (r > 2560u && m != 4u && m != 8u) continue;
            std::vector<float> x(uint64_t(m) * k);
            for (auto& v : x) v = float(int(rng() % 1993) - 996) / 997.0f;
            const uint64_t ys = uint64_t(r) * 4ull + 16ull;
            const auto xv = a.floats(x);
            const auto yv = a.zeros(uint64_t(m) * ys);

            g.set_verify_nax_scope(true);
            g.begin();
            g.verify_nax_dense_bf16(wv, {xv, m, k, uint64_t(k) * 4ull}, {yv, m, r, ys});
            const auto batch = g.submit();
            g.set_verify_nax_scope(false);
            need(batch.dispatches == 1, "verify_nax_dense_bf16 must execute in 1 dispatch");

            const auto out_b = g.read(yv);
            std::vector<float> raw_out(out_b.size() / 4ull), got(uint64_t(m) * r), ref(uint64_t(m) * r);
            std::memcpy(raw_out.data(), out_b.data(), out_b.size());
            for (uint32_t row = 0; row < m; ++row) {
                for (uint32_t col = 0; col < r; ++col) {
                    got[uint64_t(row) * r + col] = raw_out[uint64_t(row) * (ys / 4ull) + col];
                    double acc = 0.0;
                    for (uint32_t c = 0; c < k; ++c) {
                        acc += double(x[uint64_t(row) * k + c]) * double(from_bf16(w_bf16[uint64_t(col) * k + c]));
                    }
                    ref[uint64_t(row) * r + col] = float(acc);
                    ++total_checks;
                }
            }
            need(nmse_f32(got, ref) <= 1e-11, "verify_nax_dense_bf16 FP32xBF16->FP32 NMSE exceeded 1e-11");
        }

        if (k == 128u) {
            std::vector<float> bad_x(4ull * k, 0.25f);
            bad_x[2ull * k + 7u] = std::numeric_limits<float>::quiet_NaN();
            const auto xv_bad = a.floats(bad_x);
            const auto y_bad = a.zeros(4ull * r * 4ull);
            bool caught_in = false;
            try {
                g.set_verify_nax_scope(true);
                g.begin();
                g.verify_nax_dense_bf16(wv, {xv_bad, 4u, k, uint64_t(k) * 4ull}, {y_bad, 4u, r, uint64_t(r) * 4ull});
                static_cast<void>(g.submit());
            } catch (const FormatError&) {
                caught_in = true;
            }
            g.set_verify_nax_scope(false);
            need(caught_in, "verify_nax_dense_bf16 must report sticky nonfinite input");

            auto bad_w = w_bf16;
            bad_w[7ull * k + 31u] = 0x7fc0u;
            const auto w_bad = a.upload(std::as_bytes(std::span(bad_w)));
            std::vector<float> ok_x(4ull * k, 0.25f);
            const auto xv_ok = a.floats(ok_x);
            bool caught_w = false;
            try {
                g.set_verify_nax_scope(true);
                g.begin();
                g.verify_nax_dense_bf16(w_bad, {xv_ok, 4u, k, uint64_t(k) * 4ull}, {y_bad, 4u, r, uint64_t(r) * 4ull});
                static_cast<void>(g.submit());
            } catch (const FormatError&) {
                caught_w = true;
            }
            g.set_verify_nax_scope(false);
            need(caught_w, "verify_nax_dense_bf16 must report sticky nonfinite BF16 weight");
        }
    }

    // 3. End-to-end encode_qsa_prefill_indexers parity with 64-aligned indexer geometry (index_heads=4, index_width=128 -> iq=512).
    {
        MetalGraph g_nax(metallib, 512ull << 20), g_ref(metallib, 512ull << 20);
        g_nax.configure_small_gemv(16); g_nax.configure_block_gemv(3);
        g_ref.configure_small_gemv(16); g_ref.configure_block_gemv(3);
        g_nax.configure_prefill_dense_f32(true);
        g_nax.configure_prefill_dense_nax(true);
        g_ref.configure_prefill_dense_f32(true);
        g_nax.prepare_qsa_prefill();
        g_nax.prepare_dense_mm_q8(PrefillWeightLayout::gguf);
        g_ref.prepare_qsa_prefill();

        std::vector<GpuView> owned_nax, owned_ref;
        std::mt19937 rng_n(8881u), rng_r(8881u);
        const QsaShape qs{128u, 4u, 2u, 64u, 4u, 128u, 64u, 4u, 256u, 512u};
        const auto freqs = rotary_frequencies(qs.rotary_width, 10000000);
        auto mk_qw = [&](MetalGraph& g, std::vector<GpuView>& owned, std::mt19937& rng) -> QsaWeights {
            const auto fv = g.upload(std::as_bytes(std::span(freqs)), "qsa bf16 idx freqs", true);
            owned.push_back(fv);
            return {
                make_q8(g, owned, qs.channels, qs.query_heads * qs.head_width * 2u, 1, 1),
                make_q8(g, owned, qs.channels, qs.kv_heads * qs.head_width, 1, 2),
                make_q8(g, owned, qs.channels, qs.kv_heads * qs.head_width, 1, 3),
                make_q8(g, owned, qs.query_heads * qs.head_width, qs.channels, 1, 4),
                make_bf16_mat(g, owned, uint64_t(qs.channels) * qs.index_heads * qs.index_width, rng),
                make_bf16_mat(g, owned, uint64_t(qs.channels) * qs.index_width, rng),
                make_vec(g, owned, qs.head_width, 1.0f, rng),
                make_vec(g, owned, qs.head_width, 1.0f, rng),
                make_vec(g, owned, qs.index_width, 1.0f, rng),
                make_vec(g, owned, qs.index_width, 1.0f, rng),
                fv
            };
        };
        const auto qw_nax = mk_qw(g_nax, owned_nax, rng_n);
        const auto qw_ref = mk_qw(g_ref, owned_ref, rng_r);
        auto sc_nax = allocate_qsa_prefill_scratch(g_nax, qs, 96u);
        auto sc_ref = allocate_qsa_prefill_scratch(g_ref, qs, 96u);

        const uint32_t n = 67u;
        std::vector<float> x(uint64_t(n) * qs.channels);
        for (auto& v : x) v = float(int(rng_n() % 1993) - 996) / 997.0f;
        const auto x_nax = g_nax.upload(std::as_bytes(std::span(x)), "qsa x nax", true);
        const auto x_ref = g_ref.upload(std::as_bytes(std::span(x)), "qsa x ref", true);
        owned_nax.push_back(x_nax);
        owned_ref.push_back(x_ref);

        const auto proj_nax = qsa_prefill_projection_rows(sc_nax, n);
        const auto proj_ref = qsa_prefill_projection_rows(sc_ref, n);
        const GpuRowsF32 in_nax{x_nax, n, qs.channels, uint64_t(qs.channels) * 4ull};
        const GpuRowsF32 in_ref{x_ref, n, qs.channels, uint64_t(qs.channels) * 4ull};

        g_nax.begin();
        g_nax.dense_mm_q8(PrefillWeightLayout::gguf, qw_nax.key.view, in_nax, proj_nax.key, false);
        encode_qsa_prefill_indexers(g_nax, qw_nax, sc_nax, in_nax, qs);
        const auto b_nax = g_nax.submit();
        need(b_nax.dispatches == 4, "dense_mm_q8 + encode_qsa_prefill_indexers(NAX reuse) must execute in 2 + 1 + 1 = 4 dispatches");

        g_ref.begin();
        encode_qsa_prefill_indexers(g_ref, qw_ref, sc_ref, in_ref, qs);
        const auto b_ref = g_ref.submit();
        need(b_ref.dispatches == 2, "legacy encode_qsa_prefill_indexers must execute in 2 dispatches");

        std::vector<float> iq_nax(uint64_t(n) * 512u), iq_ref(uint64_t(n) * 512u);
        std::vector<float> ik_nax(uint64_t(n) * 128u), ik_ref(uint64_t(n) * 128u);
        g_nax.read(g_nax.slice(proj_nax.index_query.view, 0, uint64_t(n) * 512u * 4u), std::as_writable_bytes(std::span(iq_nax)));
        g_ref.read(g_ref.slice(proj_ref.index_query.view, 0, uint64_t(n) * 512u * 4u), std::as_writable_bytes(std::span(iq_ref)));
        g_nax.read(g_nax.slice(proj_nax.index_key.view, 0, uint64_t(n) * 128u * 4u), std::as_writable_bytes(std::span(ik_nax)));
        g_ref.read(g_ref.slice(proj_ref.index_key.view, 0, uint64_t(n) * 128u * 4u), std::as_writable_bytes(std::span(ik_ref)));
        need(nmse_f32(iq_nax, iq_ref) <= 1e-4 && nmse_f32(ik_nax, ik_ref) <= 1e-4,
             "encode_qsa_prefill_indexers NAX BF16 drifted from FP32 reference");

        release_qsa_prefill_scratch(g_nax, sc_nax);
        release_qsa_prefill_scratch(g_ref, sc_ref);
        for (auto v : owned_nax) g_nax.release(v);
        for (auto v : owned_ref) g_ref.release(v);
    }
    std::cout << "PASS nax_dense_bf16 (reuse=false/true + encode_qsa_prefill_indexers) and verify_nax_dense_bf16 (r1/r4, m=1..8)\n";
}

// This generated fixture repeats 61 independently varied weight rows to keep the FP64 oracle bounded while crossing every 64-column tile boundary.
std::vector<float> test_registered_q8_t64_m8_single(const char* metallib, const char* profile_arm = nullptr, uint32_t leg = 0u,
    std::vector<uint64_t>* resources = nullptr, bool concurrent_heads = false) {
    constexpr uint32_t patterns = 61u, capacity = 8u;
    constexpr float sentinel = -12345.25f;
    constexpr std::array<uint16_t, 12> scales{0x0000u, 0x0001u, 0x03ffu, 0x0801u, 0x1803u, 0x2401u, 0x2c03u, 0x3007u, 0x3401u, 0x3803u, 0xa40bu, 0xb003u};
    std::vector<std::pair<uint32_t, uint32_t>> shapes{{256u, 31u}, {256u, 65u}, {10240u, 320u}, {320u, 10240u}, {2560u, 10240u}, {2560u, 6144u}, {6144u, 2560u}, {2560u, 12288u}, {2560u, 512u}, {640u, 2560u}};
    if (concurrent_heads) {
        need(q8_tile64_cache_enabled, "concurrent M8 head control requires actual Tile64 copies");
        shapes.emplace_back(2560u, 32768u);
        shapes.emplace_back(2560u, 248320u);
    }
    std::vector<float> captured;
    for (const auto [columns, outputs] : shapes) {
        const bool head = concurrent_heads && outputs >= 32768u;
        const uint32_t input_stride = columns + 4u, output_stride = ((outputs + 3u) / 4u) * 4u + 4u;
        const uint32_t row_bytes = columns / 32u * 34u;
        const grouped_fixture::Shape pattern_shape{8u, columns, patterns, 1u, 1u, 1u, true, PrefillWeightLayout::gguf};
        auto pattern_weights = grouped_fixture::weights(pattern_shape);
        for (uint32_t col = 0; col < patterns; ++col) {
            for (uint32_t block = 0; block < columns / 32u; ++block) {
                const auto scale = scales[(col * 7u + block * 5u) % scales.size()];
                const size_t at = size_t(col) * row_bytes + block * 34u;
                pattern_weights[at] = std::byte(scale & 255u);
                pattern_weights[at + 1u] = std::byte(scale >> 8u);
            }
        }
        auto represented = grouped_fixture::represented(pattern_shape, pattern_weights);
        if (q8_tile64_cache_enabled) for (auto& value : represented) value = float(_Float16(value));
        std::vector<std::byte> weights(uint64_t(outputs) * row_bytes);
        for (uint32_t col = 0; col < outputs; ++col)
            std::memcpy(weights.data() + uint64_t(col) * row_bytes, pattern_weights.data() + uint64_t(col % patterns) * row_bytes, row_bytes);
        std::vector<float> input(uint64_t(capacity + 1u) * input_stride, sentinel), rounded(uint64_t(capacity) * columns);
        uint32_t rng = 0x18d4u + columns + outputs;
        for (uint32_t row = 0; row < capacity; ++row) {
            for (uint32_t k = 0; k < columns; ++k) {
                rng = rng * 1664525u + 1013904223u;
                const float value = float(int(rng % 2003u) - 1001) / 1001.0f;
                input[uint64_t(row) * input_stride + k] = value;
                const uint32_t bits = uint32_t(finite_bf16(value)) << 16u;
                std::memcpy(&rounded[uint64_t(row) * columns + k], &bits, sizeof(bits));
                if (!q8_tile64_cache_enabled) rounded[uint64_t(row) * columns + k] = value;
            }
        }
        std::vector<float> reference(uint64_t(capacity) * patterns);
        for (uint32_t row = 0; row < capacity; ++row) {
            for (uint32_t col = 0; col < patterns; ++col) {
                double sum = 0.0;
                for (uint32_t k = 0; k < columns; ++k)
                    sum += double(rounded[uint64_t(row) * columns + k]) * double(represented[uint64_t(col) * columns + k]);
                reference[uint64_t(row) * patterns + col] = float(sum);
            }
        }
        const std::vector<PrefillWeightLayout> layouts = (columns < 512u || columns % 256u == 0)
            ? std::vector<PrefillWeightLayout>{PrefillWeightLayout::gguf, PrefillWeightLayout::q8_lanes_v1}
            : std::vector<PrefillWeightLayout>{PrefillWeightLayout::gguf};
        for (const auto layout : layouts) {
            MetalGraph g(metallib, head ? (2ull << 30) : (256ull << 20));
            if (concurrent_heads) {
                g.configure_concurrent_dispatch(true);
                need(g.configure_profiling(0) == ProfileMode::disabled, "concurrent M8 head control must be unprofiled");
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
                g.configure_batch_phase_clock(false);
#endif
            }
            g.configure_verify_nax(true);
            g.configure_verify_nax_m8(true);
            g.prepare_verify_nax({});
            if (concurrent_heads) g.prepare_dflash2();
            const auto prepared_bytes = g.memory().live_bytes;
            {
                Arena a{g, {}};
                std::vector<std::byte> payload;
                if (layout == PrefillWeightLayout::q8_lanes_v1) {
                    payload.resize(weights.size());
                    lanes::pack_rows(lanes::row_layout(8u, columns), weights, payload);
                }
                const auto w = a.upload(layout == PrefillWeightLayout::q8_lanes_v1
                    ? std::span<const std::byte>(payload) : std::span<const std::byte>(weights));
                const auto before_repack = g.memory().live_bytes;
                if (layout == PrefillWeightLayout::q8_lanes_v1) g.register_lane_matrix(w, {8u, columns, outputs, 1u});
                else g.register_q8_matrix(w, columns, outputs);
                need_tile64_registration(g, before_repack);
                if (concurrent_heads)
                    need(g.prefill_weight_layout(w) == layout && g.q8_tile64_cache_stats().registered_source_bytes == weights.size(),
                         "concurrent M8 head control lost registered source identity");
                const auto x = a.floats(input, false);
                const std::vector<float> poison(uint64_t(capacity + 1u) * output_stride, sentinel);
                const auto y = a.floats(poison, false);
                const auto live = g.memory().live_bytes, allocations = g.memory().allocations;
                const std::vector<uint32_t> row_cases = head ? std::vector<uint32_t>{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}
                    : std::vector<uint32_t>{4u, 8u};
                for (uint32_t role = 0u; role < (concurrent_heads ? 2u : 1u); ++role) {
                const bool draft = role == 1u;
                g.set_verify_nax_scope(!draft);
                if (concurrent_heads) {
                    g.set_target_verification_role(!draft);
                    g.set_dflash2_scope(draft);
                }
                for (const uint32_t rows : row_cases) {
                    const auto execute = [&](uint32_t iterations, uint32_t epilogue) {
                        g.begin();
                        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
                            const GpuRowsF32 in{x, rows, columns, uint64_t(input_stride) * sizeof(float)};
                            const GpuRowsF32 out{y, rows, outputs, uint64_t(output_stride) * sizeof(float)};
                            if (draft) g.dflash2_dense_q8(w, in, out, epilogue == 1u);
                            else g.verify_nax_dense_q8(w, in, out, epilogue == 1u, epilogue == 2u ? 0.25f : 0.0f);
                        }
                        const auto batch = g.submit();
                        need(batch.dispatches == iterations, "registered M8 dense must execute one dispatch per iteration");
                        need(g.memory().live_bytes == live && g.memory().allocations == allocations, "registered M8 dense changed live resources");
                        if (concurrent_heads) {
                            need(g.concurrent_dispatch() && g.last_profile().mode == ProfileMode::disabled,
                                 "concurrent M8 head control changed encoder or profile mode");
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
                            need(!g.batch_phase_clock_state().enabled, "concurrent M8 head control enabled phase clocks");
#endif
                        }
                        return batch;
                    };
                    g.write(y, std::as_bytes(std::span(poison)));
                    if (profile_arm) {
                        const uint32_t iterations = concurrent_heads ? 1u : 32u;
                        const uint32_t warm_ms = concurrent_heads ? 100u : 200u;
                        const uint32_t trials = concurrent_heads ? 4u : 8u;
                        need(host_thermal_state() == 0u, "M8 single-stage timing requires nominal thermal state");
                        const auto warm_begin = std::chrono::steady_clock::now();
                        const auto warm_end = warm_begin + std::chrono::milliseconds(warm_ms);
                        uint32_t warm_batches = 0u;
                        do {
                            static_cast<void>(execute(iterations, 0u));
                            ++warm_batches;
                            need(host_thermal_state() == 0u, "M8 single-stage warm-up must remain nominal");
                        } while (std::chrono::steady_clock::now() < warm_end);
                        const double actual_warm_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warm_begin).count();
                        for (uint32_t trial = 0; trial < trials; ++trial) {
                            const auto before = host_thermal_state();
                            need(before == 0u, "M8 single-stage timing must start nominal");
                            const auto begin = std::chrono::steady_clock::now();
                            const auto batch = execute(iterations, 0u);
                            const double host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                            const auto after = host_thermal_state();
                            need(after == 0u, "M8 single-stage timing must finish nominal");
                            need(std::isfinite(batch.gpu_ms) && batch.gpu_ms > 0.0 && std::isfinite(host_ms) && host_ms > 0.0, "M8 single-stage timing must be positive and finite");
                            std::cout << (concurrent_heads ? "Q8_T64_M8_CONCURRENT_PROFILE {\"arm\":\"" : "Q8_T64_M8_PROFILE {\"arm\":\"") << profile_arm << "\",\"leg\":" << leg
                                      << ",\"columns\":" << columns << ",\"outputs\":" << outputs << ",\"rows\":" << rows
                                      << ",\"layout\":" << uint32_t(layout) << ",\"trial\":" << trial << ",\"iterations\":" << iterations
                                      << ",\"warm_batches\":" << warm_batches << ",\"batch_gpu_ms\":" << batch.gpu_ms
                                      << ",\"iteration_gpu_ms\":" << batch.gpu_ms / iterations << ",\"batch_host_ms\":" << host_ms
                                      << ",\"iteration_host_ms\":" << host_ms / iterations << ",\"dispatches\":" << batch.dispatches
                                      << ",\"live_metal_bytes\":" << live << ",\"allocations\":" << allocations
                                      << ",\"thermal_before\":" << before << ",\"thermal_after\":" << after;
                            if (concurrent_heads)
                                std::cout << ",\"role\":\"" << (draft ? "dflash2" : "target") << "\",\"concurrent\":true,\"profile\":false,\"phase_clock\":false"
                                          << ",\"warm_target_ms\":" << warm_ms << ",\"warm_actual_ms\":" << actual_warm_ms
                                          << ",\"trials\":" << trials << ",\"head\":" << (head ? "true" : "false")
                                          << ",\"owned_tile64_bytes\":" << g.q8_tile64_cache_stats().owned_charged_bytes;
                            std::cout << "}\n";
                        }
                    } else {
                        static_cast<void>(execute(1u, 0u));
                    }
                    std::vector<float> clean(poison.size());
                    g.read(y, std::as_writable_bytes(std::span(clean)));
                    const auto validate = [&](const std::vector<float>& actual, uint32_t epilogue, const std::vector<float>& gate) {
                        std::vector<float> got, ref;
                        for (uint32_t row = 0; row <= capacity; ++row) {
                            for (uint32_t col = 0; col < output_stride; ++col) {
                                const auto at = uint64_t(row) * output_stride + col;
                                if (row >= rows || col >= outputs) {
                                    need(actual[at] == sentinel, "M8 single-stage wrote inactive output rows or padding");
                                    continue;
                                }
                                float value = reference[uint64_t(row) * patterns + col % patterns];
                                if (concurrent_heads) need(std::isfinite(actual[at]), "concurrent M8 head control produced a nonfinite value");
                                if (epilogue == 1u) {
                                    const float v = gate[at];
                                    const float sigmoid = v >= 0.0f ? 1.0f / (1.0f + std::exp(-v)) : std::exp(v) / (1.0f + std::exp(v));
                                    value = (v * sigmoid) * value;
                                } else if (epilogue == 2u) {
                                    const float v = value * 0.25f;
                                    const float sigmoid = v >= 0.0f ? 1.0f / (1.0f + std::exp(-v)) : std::exp(v) / (1.0f + std::exp(v));
                                    value = v * sigmoid;
                                }
                                got.push_back(actual[at]); ref.push_back(value);
                            }
                        }
                        need(nmse_f32(got, ref) <= 1e-4, "registered M8 exceeds selected-operand FP64 oracle NMSE1e-4");
                    };
                    validate(clean, 0u, poison);
                    captured.insert(captured.end(), clean.begin(), clean.end());
                    if (resources) { resources->push_back(live); resources->push_back(allocations); }
                    if (columns == 256u) {
                        for (uint32_t epilogue : {1u, 2u}) {
                            if (draft && epilogue == 2u) continue; // DFlash dense exposes only the multiply-SwiGLU epilogue.
                            auto gate = poison;
                            for (uint32_t row = 0; row < rows; ++row)
                                for (uint32_t col = 0; col < outputs; ++col)
                                    gate[uint64_t(row) * output_stride + col] = float(int((row * 17u + col * 7u) % 71u) - 35) / 19.0f;
                            g.write(y, std::as_bytes(std::span(gate)));
                            static_cast<void>(execute(1u, epilogue));
                            std::vector<float> actual(poison.size());
                            g.read(y, std::as_writable_bytes(std::span(actual)));
                            validate(actual, epilogue, gate);
                            captured.insert(captured.end(), actual.begin(), actual.end());
                        }
                        for (float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
                            auto bad = input;
                            bad[uint64_t(rows - 1u) * input_stride + columns - 1u] = invalid;
                            g.write(x, std::as_bytes(std::span(bad)));
                            bool refused = false;
                            try { static_cast<void>(execute(1u, 0u)); } catch (const FormatError&) { refused = true; }
                            try { g.abort(); } catch (const FormatError&) {}
                            need(refused, "M8 single-stage must refuse active nonfinite input");
                            g.write(x, std::as_bytes(std::span(input)));
                            g.write(y, std::as_bytes(std::span(poison)));
                            static_cast<void>(execute(1u, 0u));
                            std::vector<float> retried(poison.size());
                            g.read(y, std::as_writable_bytes(std::span(retried)));
                            need(std::memcmp(retried.data(), clean.data(), clean.size() * sizeof(float)) == 0, "M8 single-stage retry differs from own clean padded result");
                            captured.insert(captured.end(), retried.begin(), retried.end());
                        }
                    }
                }
                g.set_verify_nax_scope(false);
                if (concurrent_heads) { g.set_target_verification_role(false); g.set_dflash2_scope(false); }
                }
            }
            need(g.memory().live_bytes == prepared_bytes, "M8 single-stage fixture leaked registered tile64 storage");
            need_tile64_state(g, 0);
        }
    }
    std::cout << (concurrent_heads
        ? "PASS concurrent Q8 M8 target/DFlash: original shapes M4/M8 plus K2560 heads N32768/248320 M1..M8, selected-operand FP64 oracle NMSE1e-4, guards and finite retry\n"
        : "PASS registered Q8 M8 selected-operand FP64 oracle, partial column tiles, strides, epilogues and finite retry\n");
    return captured;
}

std::vector<float> test_registered_q8_t64_m32(const char* metallib, const char* profile_arm = nullptr, uint32_t leg = 0u) {
    constexpr float sentinel = -12345.25f;
    constexpr double bf16_nmse_limit = 1e-4;
    constexpr std::array<uint16_t, 10> scales{0x0000u, 0x0801u, 0x1803u, 0x2401u, 0x2c03u, 0x3007u, 0x3401u, 0x3803u, 0xa40bu, 0xb003u};
    std::vector<float> captured;
    for (const auto [columns, outputs] : std::array<std::pair<uint32_t, uint32_t>, 2>{{{256u, 131u}, {2560u, 320u}}}) {
        const uint32_t capacity = columns == 256u ? 257u : 129u;
        const uint32_t input_stride = columns + 4u;
        const uint32_t output_stride = ((outputs + 3u) / 4u) * 4u + 4u;
        const grouped_fixture::Shape shape{8u, columns, outputs, 1u, 1u, 1u, true, PrefillWeightLayout::gguf};
        auto weights = grouped_fixture::weights(shape);
        for (uint32_t col = 0; col < outputs; ++col) {
            for (uint32_t block = 0; block < columns / 32u; ++block) {
                const uint16_t scale = scales[(col * 3u + block * 5u + col / 64u * 7u) % scales.size()];
                const size_t at = (size_t(col) * (columns / 32u) + block) * 34u;
                weights[at] = std::byte(scale & 255u);
                weights[at + 1u] = std::byte(scale >> 8u);
            }
        }
        auto represented = grouped_fixture::represented(shape, weights);
        for (auto& value : represented) value = float(_Float16(value));
        std::vector<float> input(uint64_t(capacity + 1u) * input_stride, sentinel);
        std::vector<float> rounded(uint64_t(capacity) * columns);
        uint32_t rng = 0x124864u;
        for (uint32_t row = 0; row < capacity; ++row) {
            for (uint32_t k = 0; k < columns; ++k) {
                rng = rng * 1664525u + 1013904223u;
                const float value = float(int(rng % 2003u) - 1001) / 1001.0f;
                input[uint64_t(row) * input_stride + k] = value;
                const uint32_t bits = uint32_t(finite_bf16(value)) << 16u;
                std::memcpy(&rounded[uint64_t(row) * columns + k], &bits, sizeof(bits));
            }
        }
        std::vector<float> reference(uint64_t(capacity) * outputs);
        for (uint32_t row = 0; row < capacity; ++row) {
            for (uint32_t col = 0; col < outputs; ++col) {
                double sum = 0.0;
                for (uint32_t k = 0; k < columns; ++k)
                    sum += double(rounded[uint64_t(row) * columns + k]) * double(represented[uint64_t(col) * columns + k]);
                reference[uint64_t(row) * outputs + col] = float(sum);
            }
        }

        MetalGraph g(metallib, 256ull << 20);
        g.configure_prefill_dense_f32(true);
        g.configure_prefill_dense_nax(true);
        g.prepare_dense_mm_q8(PrefillWeightLayout::gguf);
        const auto prepared_bytes = g.memory().live_bytes;
        {
            Arena a{g, {}};
            const auto w = a.upload(weights);
            const auto before_repack = g.memory().live_bytes;
            g.register_q8_matrix(w, columns, outputs);
            need_tile64_registration(g, before_repack);
            const auto x = a.floats(input);
            const std::vector<float> poison(uint64_t(capacity + 1u) * output_stride, sentinel);
            const auto y = a.floats(poison, false);
            std::vector<uint32_t> row_cases{128u, 129u};
            if (capacity > 129u) row_cases.push_back(capacity);
            for (const uint32_t rows : row_cases) {
                g.write(y, std::as_bytes(std::span(poison)));
                const auto execute = [&](uint32_t iterations = 1u) {
                    g.begin();
                    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
                        g.dense_mm_q8(PrefillWeightLayout::gguf, w,
                            {x, rows, columns, uint64_t(input_stride) * sizeof(float)},
                            {y, rows, outputs, uint64_t(output_stride) * sizeof(float)}, false);
                    }
                    const auto batch = g.submit();
                    need(batch.dispatches == 2u * iterations, "registered Q8 M32 fixture dispatch count");
                    return batch;
                };
                if (profile_arm) {
                    need(host_thermal_state() == 0u, "Q8 M32 component timing requires nominal thermal state");
                    constexpr uint32_t iterations = 32u;
                    const auto warm_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
                    do {
                        static_cast<void>(execute(iterations));
                        need(host_thermal_state() == 0u, "Q8 M32 component warm-up must remain nominal");
                    } while (std::chrono::steady_clock::now() < warm_end);
                    for (uint32_t trial = 0; trial < 8u; ++trial) {
                        const auto before = host_thermal_state();
                        need(before == 0u, "Q8 M32 component timing must start nominal");
                        const auto batch = execute(iterations);
                        const auto after = host_thermal_state();
                        std::cout << "Q8_T64_M32_PROFILE {\"arm\":\"" << profile_arm << "\",\"leg\":" << leg
                                  << ",\"columns\":" << columns << ",\"outputs\":" << outputs << ",\"rows\":" << rows
                                  << ",\"trial\":" << trial << ",\"iterations\":" << iterations << ",\"batch_gpu_ms\":" << batch.gpu_ms
                                  << ",\"thermal_before\":" << before << ",\"thermal_after\":" << after
                                  << ",\"scope\":\"GPU interval for 32 warm BF16 conversion plus dense Q8 M32 iterations\"}\n";
                        need(after == 0u, "Q8 M32 component timing must finish nominal");
                        need(std::isfinite(batch.gpu_ms) && batch.gpu_ms > 0.0, "Q8 M32 component GPU interval must be positive and finite");
                    }
                } else {
                    static_cast<void>(execute());
                }
                std::vector<float> actual(poison.size());
                g.read(y, std::as_writable_bytes(std::span(actual)));
                std::vector<float> got(uint64_t(rows) * outputs);
                for (uint32_t row = 0; row <= capacity; ++row) {
                    for (uint32_t col = 0; col < output_stride; ++col) {
                        const float value = actual[uint64_t(row) * output_stride + col];
                        if (row < rows && col < outputs) got[uint64_t(row) * outputs + col] = value;
                        else need(value == sentinel, "registered Q8 M32 wrote inactive rows or output padding");
                    }
                }
                need(nmse_f32(got, std::span(reference).first(got.size())) <= bf16_nmse_limit,
                    "registered Q8 M32 exceeds the BF16-input and FP16-weight NMSE bound against FP64 reference");
                captured.insert(captured.end(), got.begin(), got.end());
            }
        }
        need(g.memory().live_bytes == prepared_bytes, "registered Q8 M32 fixture leaked tile64 buffers");
    }
    const auto bytes = std::as_bytes(std::span(captured));
    need(bytes.size() <= std::numeric_limits<CC_LONG>::max(), "registered Q8 M32 digest input exceeds bound");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    need(CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data()) != nullptr, "registered Q8 M32 digest failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string hash;
    for (const auto byte : digest) { hash += hex[byte >> 4u]; hash += hex[byte & 15u]; }
    std::cout << "PASS registered Q8 tile64 M32 varied scales/codes, column tiles and row tails: output_sha256=" << hash << '\n';
    return captured;
}

std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> test_registered_q8_t64_m32_large(const char* metallib, const char* profile_arm = nullptr, uint32_t leg = 0u) {
    constexpr uint32_t columns = 2560u, input_stride = columns + 4u;
    constexpr float sentinel = -12345.25f;
    CC_SHA256_CTX digest_context;
    need(CC_SHA256_Init(&digest_context) != 0, "large Q8 M32 digest initialisation failed");
    // Keep each generated GPU fixture, including prepared shared scratch and tile64 copies, below 100 MiB.
    for (const auto [outputs, capacity] : std::array<std::pair<uint32_t, uint32_t>, 3>{{{2560u, 2048u}, {5120u, 129u}, {8192u, 129u}}}) {
        const uint32_t output_stride = outputs + 4u;
        const grouped_fixture::Shape shape{8u, columns, outputs, 1u, 1u, 1u, true, PrefillWeightLayout::gguf};
        const auto weights = grouped_fixture::weights(shape);
        std::vector<float> input(uint64_t(capacity + 1u) * input_stride, sentinel);
        uint32_t rng = 0x644832u;
        for (uint32_t row = 0; row < capacity; ++row) {
            for (uint32_t k = 0; k < columns; ++k) {
                rng = rng * 1664525u + 1013904223u;
                const uint32_t bits = uint32_t(finite_bf16(float(int(rng % 2003u) - 1001) / 1001.0f)) << 16u;
                std::memcpy(&input[uint64_t(row) * input_stride + k], &bits, sizeof(bits));
            }
        }
        MetalGraph g(metallib, 256ull << 20);
        g.configure_prefill_dense_f32(true);
        g.configure_prefill_dense_nax(true);
        g.prepare_dense_mm_q8(PrefillWeightLayout::gguf);
        const auto prepared_bytes = g.memory().live_bytes;
        {
            Arena a{g, {}};
            const auto w = a.upload(weights);
            const auto before_repack = g.memory().live_bytes;
            g.register_q8_matrix(w, columns, outputs);
            need_tile64_registration(g, before_repack);
            const auto x = a.floats(input, false);
            std::vector<float> actual(uint64_t(capacity + 1u) * output_stride, sentinel);
            const auto y = a.floats(actual, false);
            need(g.memory().live_bytes <= (100ull << 20), "large Q8 M32 fixture exceeds its 100 MiB allocation bound");
            const std::array row_cases{capacity == 2048u ? 2047u : 128u, capacity};
            for (const uint32_t rows : row_cases) {
                std::fill(actual.begin(), actual.end(), sentinel);
                g.write(y, std::as_bytes(std::span(actual)));
                const auto execute = [&](uint32_t iterations = 1u) {
                    g.begin();
                    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
                        g.dense_mm_q8(PrefillWeightLayout::gguf, w,
                            {x, rows, columns, uint64_t(input_stride) * sizeof(float)},
                            {y, rows, outputs, uint64_t(output_stride) * sizeof(float)}, false);
                    }
                    const auto batch = g.submit();
                    need(batch.dispatches == 2u * iterations, "large Q8 M32 fixture dispatch count");
                    return batch;
                };
                static_cast<void>(execute());
                if (profile_arm) {
                    constexpr uint32_t iterations = 32u;
                    const auto warm_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
                    do {
                        need(host_thermal_state() == 0u, "large Q8 M32 component warm-up must remain nominal");
                        static_cast<void>(execute(iterations));
                    } while (std::chrono::steady_clock::now() < warm_end);
                    for (uint32_t trial = 0; trial < 8u; ++trial) {
                        const auto before = host_thermal_state();
                        need(before == 0u, "large Q8 M32 component timing must start nominal");
                        const auto batch = execute(iterations);
                        const auto after = host_thermal_state();
                        std::cout << "Q8_T64_M32_PROFILE {\"arm\":\"" << profile_arm << "\",\"leg\":" << leg
                                  << ",\"columns\":" << columns << ",\"outputs\":" << outputs << ",\"rows\":" << rows
                                  << ",\"trial\":" << trial << ",\"iterations\":" << iterations << ",\"batch_gpu_ms\":" << batch.gpu_ms
                                  << ",\"thermal_before\":" << before << ",\"thermal_after\":" << after
                                  << ",\"scope\":\"GPU interval for 32 warm BF16 conversion plus dense Q8 M32 iterations\"}\n";
                        need(after == 0u && std::isfinite(batch.gpu_ms) && batch.gpu_ms > 0.0, "large Q8 M32 timing requires nominal thermal state and a finite positive interval");
                    }
                }
                g.read(y, std::as_writable_bytes(std::span(actual)));
                for (uint32_t row = 0; row <= capacity; ++row) {
                    for (uint32_t col = 0; col < output_stride; ++col) {
                        const float value = actual[uint64_t(row) * output_stride + col];
                        need(row < rows && col < outputs ? std::isfinite(value) : value == sentinel,
                            "large Q8 M32 output must be finite and preserve inactive rows and stride padding");
                    }
                    if (row < rows) need(CC_SHA256_Update(&digest_context, actual.data() + uint64_t(row) * output_stride, outputs * sizeof(float)) != 0, "large Q8 M32 digest update failed");
                }
                std::vector<uint32_t> sample_rows{0u, 1u, 31u, 32u, 63u, 64u, 127u, rows - 1u};
                if (rows > 128u) sample_rows.push_back(128u);
                if (rows > 1024u) { sample_rows.push_back(1023u); sample_rows.push_back(1024u); }
                const std::array sample_cols{0u, 1u, 31u, 32u, 63u, 64u, 127u, 128u, outputs / 2u - 1u, outputs / 2u, outputs - 2u, outputs - 1u};
                std::vector<float> represented(columns), got, reference;
                constexpr uint32_t weight_row_bytes = columns / 32u * 34u;
                for (const uint32_t col : sample_cols) {
                    dequantize(8u, std::span(weights).subspan(uint64_t(col) * weight_row_bytes, weight_row_bytes), represented);
                    for (auto& value : represented) value = float(_Float16(value));
                    for (const uint32_t row : sample_rows) {
                        double sum = 0.0;
                        for (uint32_t k = 0; k < columns; ++k) sum += double(input[uint64_t(row) * input_stride + k]) * double(represented[k]);
                        got.push_back(actual[uint64_t(row) * output_stride + col]);
                        reference.push_back(float(sum));
                    }
                }
                need(nmse_f32(got, reference) <= 1e-4, "large Q8 M32 boundary samples exceed the represented-operand FP64 NMSE bound");
                const size_t bad_index = uint64_t(rows - 1u) * input_stride + columns - 1u;
                const float clean = input[bad_index];
                for (const float invalid : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
                    input[bad_index] = invalid;
                    g.write(x, std::as_bytes(std::span(input)));
                    bool refused = false;
                    try { static_cast<void>(execute()); } catch (const FormatError&) { refused = true; }
                    try { g.abort(); } catch (const FormatError&) {}
                    need(refused, "large Q8 M32 must refuse nonfinite active input");
                }
                input[bad_index] = clean;
                g.write(x, std::as_bytes(std::span(input)));
                static_cast<void>(execute());
            }
        }
        need(g.memory().live_bytes == prepared_bytes, "large Q8 M32 fixture leaked tile64 buffers");
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    need(CC_SHA256_Final(digest.data(), &digest_context) != 0, "large Q8 M32 digest finalisation failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string hash;
    for (const auto byte : digest) { hash += hex[byte >> 4u]; hash += hex[byte & 15u]; }
    std::cout << "PASS registered Q8 tile64 M32 large shapes, FP64 boundary samples, finite refusal and output guards: output_sha256=" << hash << '\n';
    return digest;
}

void test_registered_q8_t64_m32_scale_edges(const char* metallib) {
    constexpr uint32_t columns = 64u, outputs = 65u, rows = 129u, weight_row_bytes = columns / 32u * 34u;
    constexpr uint32_t output_stride = 68u;
    constexpr float sentinel = -12345.25f;
    constexpr std::array<uint16_t, 13> scales{0x0000u, 0x8000u, 0x0001u, 0x03ffu, 0x0400u, 0x5fffu, 0x6000u, 0x8400u, 0xe000u, 0x7c00u, 0x7e00u, 0xfc00u, 0x6400u};
    MetalGraph g(metallib, 256ull << 20);
    g.configure_prefill_dense_f32(true);
    g.configure_prefill_dense_nax(true);
    g.prepare_dense_mm_q8(PrefillWeightLayout::gguf);
    const auto prepared_bytes = g.memory().live_bytes;
    for (const uint16_t scale : scales) {
        {
            Arena a{g, {}};
            std::vector<std::byte> weights(uint64_t(outputs) * weight_row_bytes, std::byte{127});
            for (size_t offset = 0; offset < weights.size(); offset += 34u) {
                weights[offset] = std::byte{0}; weights[offset + 1u] = std::byte{0x38};
            }
            const size_t at = uint64_t(outputs - 1u) * weight_row_bytes + 34u;
            weights[at] = std::byte(scale & 255u); weights[at + 1u] = std::byte(scale >> 8u);
            const auto w = a.upload(weights);
            const auto before_repack = g.memory().live_bytes;
            g.register_q8_matrix(w, columns, outputs);
            need_tile64_registration(g, before_repack);
            const std::vector<float> input(uint64_t(rows) * columns, 0.125f);
            const std::vector<float> poison(uint64_t(rows) * output_stride, sentinel);
            const auto x = a.floats(input), y = a.floats(poison, false);
            bool refused = false;
            std::string refusal;
            try {
                g.begin();
                g.dense_mm_q8(PrefillWeightLayout::gguf, w, {x, rows, columns, columns * sizeof(float)}, {y, rows, outputs, output_stride * sizeof(float)}, false);
                static_cast<void>(g.submit());
            } catch (const FormatError& error) { refused = true; refusal = error.what(); }
            try { g.abort(); } catch (const FormatError&) {}
            const bool invalid = (scale & 0x7fffu) >= 0x7c00u || scale == 0x6400u;
            if (refused != invalid) std::cerr << "Q8 M32 scale boundary: scale=" << scale << " expected_refusal=" << invalid << " refused=" << refused << " reason=" << refusal << '\n';
            need(refused == invalid, "Q8 M32 scale boundary must retain finite and half-overflow refusal");
            if (!invalid) {
                std::vector<float> actual(poison.size()), represented(columns);
                g.read(y, std::as_writable_bytes(std::span(actual)));
                dequantize(8u, std::span(weights).subspan(uint64_t(outputs - 1u) * weight_row_bytes, weight_row_bytes), represented);
                double reference = 0.0;
                for (const auto value : represented) reference += double(float(_Float16(value))) * 0.125;
                for (uint32_t row = 0; row < rows; ++row) {
                    for (uint32_t col = 0; col < output_stride; ++col) {
                        const float value = actual[uint64_t(row) * output_stride + col];
                        if (col >= outputs) { need(value == sentinel, "Q8 M32 scale boundary wrote output padding"); continue; }
                        const double expected = col + 1u == outputs ? reference : 508.0;
                        need(std::isfinite(value) && std::abs(double(value) - expected) <= 1e-5 * std::max(1.0, std::abs(expected)), "Q8 M32 finite scale boundary differs from represented FP64 reference");
                    }
                }
            }
        }
        need(g.memory().live_bytes == prepared_bytes, "Q8 M32 scale-edge fixture leaked tile64 buffers");
    }
    std::cout << "PASS registered Q8 tile64 M32 signed zero, subnormal and fast-path scale boundaries, nonfinite scales and half overflow\n";
}

void test_nax_gdn_projection_gates(const char* metallib) {
    constexpr float sentinel = -12345.25f;
    constexpr double bf16_nmse_limit = 1e-4;
    for (const uint32_t columns : {64u, 2560u}) {
        MetalGraph g(metallib, 256ull << 20);
        g.configure_prefill_dense_f32(true);
        g.configure_prefill_dense_nax(true);
        g.prepare_dense_mm_q8(PrefillWeightLayout::gguf);
        g.prepare_gdn_chunk();
        const auto prepared_bytes = g.memory().live_bytes;
        {
            Arena a{g, {}};
            const uint32_t heads = columns == 64u ? 17u : 48u;
            const uint32_t output_stride = heads + 3u;
            const uint32_t input_stride = columns + 4u;
            const uint32_t capacity = columns == 64u ? 2048u : 65u;
            std::vector<float> wa(heads * columns), wb(wa.size()), rate(heads), bias(heads);
            std::vector<float> x(uint64_t(capacity + 1u) * input_stride, sentinel);
            uint32_t rng = 0x1248u;
            const auto next = [&] { rng = rng * 1664525u + 1013904223u; return float(int(rng % 2003u) - 1001) / 1001.0f; };
            for (auto& value : wa) value = next() / 64.0f;
            for (auto& value : wb) value = next() / 32.0f;
            for (uint32_t h = 0; h < heads; ++h) {
                rate[h] = -float(h) / 31.0f;
                bias[h] = h == 1u ? -80.0f : (h == 2u ? 80.0f : next());
            }
            for (uint32_t t = 0; t < capacity; ++t)
                for (uint32_t k = 0; k < columns; ++k) x[uint64_t(t) * input_stride + k] = next();
            const auto wav = a.floats(wa, false), wbv = a.floats(wb, false);
            const auto rv = a.floats(rate, false), bv = a.floats(bias, false), xv = a.floats(x, false);
            const std::vector<float> poison(uint64_t(capacity + 1u) * output_stride, sentinel);
            const auto logg = a.floats(poison, false), beta = a.floats(poison, false);
            std::vector<std::byte> q8(uint64_t(columns / 32u) * 34u * 64u, std::byte{1});
            for (size_t off = 0; off < q8.size(); off += 34u) {
                q8[off] = std::byte{0}; q8[off + 1u] = std::byte{0x38};
            }
            const auto q8v = a.upload(q8);
            const auto before_repack = g.memory().live_bytes;
            g.register_q8_matrix(q8v, columns, 64u);
            need_tile64_registration(g, before_repack);
            const auto repacked_allocations = g.memory().allocations;
            g.prepare_dense_mm_q8(PrefillWeightLayout::gguf);
            need(g.memory().allocations == repacked_allocations, "repeated Q8 preparation must reuse completed planes");
            const auto dense_output = a.zeros(uint64_t(capacity) * 64u * sizeof(float));
            const auto execute = [&](uint32_t n) {
                GdnChunkDispatch d{};
                d.op = GdnChunkOp::projection_gates_f32;
                d.params.rows = n; d.params.columns = columns; d.params.outputs = heads; d.params.heads = heads;
                d.params.input_stride = uint64_t(input_stride) * sizeof(float);
                d.params.gate_stride = uint64_t(output_stride) * sizeof(float);
                d.buffers[0] = wav; d.buffers[1] = wbv; d.buffers[2] = xv;
                d.buffers[3] = rv; d.buffers[4] = bv; d.buffers[5] = logg; d.buffers[6] = beta;
                const GpuRowsF32 input{xv, n, columns, d.params.input_stride};
                g.begin();
                g.dense_mm_q8(PrefillWeightLayout::gguf, q8v, input, {dense_output, n, 64u, 64u * sizeof(float)}, false);
                need(g.can_reuse_nax_dense_bf16(input), "GDN fixture must select the cached BF16 activation route");
                g.gdn_chunk(d);
                const auto result = g.submit();
                need(result.dispatches == 3u, "NAX GDN projection fixture dispatch count");
            };
            const auto rounded_input = [](float value) {
                const uint32_t bits = uint32_t(finite_bf16(value)) << 16u;
                float rounded;
                std::memcpy(&rounded, &bits, sizeof(rounded));
                return rounded;
            };
            std::vector<uint32_t> row_cases{16u, 17u, 31u, 32u, 63u, 64u, 65u};
            if (capacity > 65u) row_cases.push_back(capacity);
            for (const auto n : row_cases) {
                g.write(logg, std::as_bytes(std::span(poison))); g.write(beta, std::as_bytes(std::span(poison)));
                execute(n);
                std::vector<float> actual_logg(poison.size()), actual_beta(poison.size()), dense(uint64_t(n) * 64u);
                g.read(logg, std::as_writable_bytes(std::span(actual_logg)));
                g.read(beta, std::as_writable_bytes(std::span(actual_beta)));
                g.read(g.slice(dense_output, 0, dense.size() * sizeof(float)), std::as_writable_bytes(std::span(dense)));
                std::vector<float> got_logg, got_beta, ref_logg, ref_beta;
                for (uint32_t t = 0; t < n; ++t) {
                    double dense_reference = 0.0;
                    for (uint32_t k = 0; k < columns; ++k) dense_reference += double(rounded_input(x[uint64_t(t) * input_stride + k])) * 0.5;
                    for (uint32_t j = 0; j < 64u; ++j)
                        need(std::isfinite(dense[uint64_t(t) * 64u + j]) && std::abs(double(dense[uint64_t(t) * 64u + j]) - dense_reference) <= 1e-4 * std::max(1.0, std::abs(dense_reference)), "Q8 tile64 repack consumer differs from represented weights");
                    for (uint32_t h = 0; h < heads; ++h) {
                        double alpha = 0.0, beta_logit = 0.0;
                        for (uint32_t k = 0; k < columns; ++k) {
                            const double operand = rounded_input(x[uint64_t(t) * input_stride + k]);
                            alpha += double(wa[uint64_t(h) * columns + k]) * operand;
                            beta_logit += double(wb[uint64_t(h) * columns + k]) * operand;
                        }
                        const double z = alpha + double(bias[h]);
                        const double softplus = std::max(z, 0.0) + std::log1p(std::exp(-std::abs(z)));
                        const double sigmoid = beta_logit >= 0.0 ? 1.0 / (1.0 + std::exp(-beta_logit)) : std::exp(beta_logit) / (1.0 + std::exp(beta_logit));
                        got_logg.push_back(actual_logg[uint64_t(t) * output_stride + h]);
                        got_beta.push_back(actual_beta[uint64_t(t) * output_stride + h]);
                        ref_logg.push_back(float(double(rate[h]) * softplus)); ref_beta.push_back(float(sigmoid));
                    }
                }
                need(nmse_f32(got_logg, ref_logg) <= bf16_nmse_limit && nmse_f32(got_beta, ref_beta) <= bf16_nmse_limit, "NAX GDN projections exceed the BF16 operand NMSE bound against independent FP64 equations");
                for (uint32_t t = 0; t <= capacity; ++t)
                    for (uint32_t h = t < n ? heads : 0u; h < output_stride; ++h)
                        need(actual_logg[uint64_t(t) * output_stride + h] == sentinel && actual_beta[uint64_t(t) * output_stride + h] == sentinel, "NAX GDN projection wrote inactive output rows or stride padding");
            }
            const auto reject_value = [&](GpuView view, const std::vector<float>& clean, size_t index, float invalid) {
                auto bad = clean; bad[index] = invalid;
                g.write(view, std::as_bytes(std::span(bad)));
                bool refused = false;
                try { execute(17u); } catch (const FormatError&) { refused = true; }
                try { g.abort(); } catch (const FormatError&) {}
                need(refused, "NAX GDN projection must refuse invalid active operands");
                g.write(view, std::as_bytes(std::span(clean)));
                execute(17u);
            };
            reject_value(wav, wa, columns + 7u, std::numeric_limits<float>::quiet_NaN());
            reject_value(wbv, wb, columns * 16u + 11u, std::numeric_limits<float>::infinity());
            reject_value(rv, rate, 16u, std::numeric_limits<float>::quiet_NaN());
            reject_value(rv, rate, 16u, 0.1f);
            reject_value(bv, bias, 16u, std::numeric_limits<float>::infinity());
            reject_value(xv, x, input_stride * 16u + columns - 1u, std::numeric_limits<float>::infinity());
        }
        need(g.memory().live_bytes == prepared_bytes, "Q8 tile64 source release or GDN fixture leaked buffers");
    }
    std::cout << "PASS NAX GDN projection gates against BF16-input/FP32-weight FP64 equations, row tails, invalid operands and Q8 repack lifecycle\n";
}

void test_q8_cache_preparation_entries(const char* metallib) {
    for (const auto layout : {PrefillWeightLayout::gguf, PrefillWeightLayout::q8_lanes_v1}) {
        for (const uint32_t entry : {0u, 1u, 2u, 3u}) for (const bool late : {false, true}) {
            MetalGraph g(metallib, 256ull << 20);
            g.configure_prefill_dense_f32(true); g.configure_prefill_dense_nax(true);
            g.configure_verify_nax(true); g.configure_verify_nax_m8(true);
            g.configure_prefill_hc_down_window(true);
            const auto prepare = [&] {
                if (entry == 0u) g.prepare_dense_mm_q8(layout);
                else if (entry == 1u) g.prepare_verify_nax({});
                else if (entry == 2u) g.prepare_hc_down_window();
                else g.prepare_dflash2();
            };
            if (late) prepare();
            const auto raw = grouped_fixture::weights({8u, 256u, 65u, 1, 1, 1, true, PrefillWeightLayout::gguf});
            auto payload = raw;
            if (layout == PrefillWeightLayout::q8_lanes_v1) lanes::pack_rows(lanes::row_layout(8u, 256u), raw, payload);
            const auto before_upload = g.memory().live_bytes;
            const auto w = g.upload(payload, "Q8 preparation entry fixture", true);
            const auto source_charge = g.memory().live_bytes - before_upload;
            if (layout == PrefillWeightLayout::q8_lanes_v1) g.register_lane_matrix(w, {8u, 256u, 65u, 1});
            else g.register_q8_matrix(w, 256u, 65u);
            if (!late) prepare();
            need_tile64_state(g, 1);
            need(g.prefill_weight_layout(w) == layout && g.q8_tile64_cache_stats().registered_source_bytes == payload.size(),
                 "preparation entry changed registered source identity");
            const auto stable = g.memory();
            const auto copy_charge = g.q8_tile64_cache_stats().owned_charged_bytes;
            prepare(); g.register_q8_matrix(w, 256u, 65u);
            need(g.memory().live_bytes == stable.live_bytes && g.memory().allocations == stable.allocations,
                 "repeated preparation entry allocated another Q8 copy");
            g.release(w); need_tile64_state(g, 0);
            need(g.memory().live_bytes == stable.live_bytes - source_charge - copy_charge,
                 "source release did not retire exactly its original allocation and Q8 copies");
        }
    }
    std::cout << "PASS Q8 preparation entry policies: prefill, verify, HC down and DFlash, registration before/after preparation\n";
}

void test_registered_q8_m8_cache_policy(const char* metallib) {
    constexpr float sentinel = -12345.25f;
    for (const auto layout : {PrefillWeightLayout::gguf, PrefillWeightLayout::q8_lanes_v1}) {
        for (const bool late : {false, true}) {
            // K64 cannot split; K2560 and K6144 exercise the existing split
            // helpers with small output tiles, in both original layouts.
            for (const uint32_t k : {64u, 2560u, 6144u}) {
                constexpr uint32_t r = 131u, stride = 136u;
                MetalGraph g(metallib, 256ull << 20);
                g.configure_verify_nax(true);
                g.configure_verify_nax_m8(true);
                need(g.configure_profiling(3) != ProfileMode::disabled, "M8 route proof requires profiling support");
                if (late) { g.prepare_verify_nax({}); g.prepare_dflash2(); }
                {
                    Arena a{g, {}};
                    const grouped_fixture::Shape shape{8u, k, r, 1, 1, 1, true, PrefillWeightLayout::gguf};
                    const auto raw_gate = grouped_fixture::weights(shape);
                    auto raw_up = raw_gate;
                    for (size_t i = 0; i < raw_up.size(); ++i)
                        if (i % 34u >= 2u) raw_up[i] ^= std::byte((i * 13u + 7u) & 0x0fu);
                    auto gate = grouped_fixture::represented(shape, raw_gate);
                    auto up = grouped_fixture::represented(shape, raw_up);
                    if (q8_tile64_cache_enabled) {
                        for (auto& v : gate) v = float(_Float16(v));
                        for (auto& v : up) v = float(_Float16(v));
                    }
                    auto pay_gate = raw_gate, pay_up = raw_up;
                    if (layout == PrefillWeightLayout::q8_lanes_v1) {
                        lanes::pack_rows(lanes::row_layout(8u, k), raw_gate, pay_gate);
                        lanes::pack_rows(lanes::row_layout(8u, k), raw_up, pay_up);
                    }
                    const auto wg = a.upload(pay_gate), wu = a.upload(pay_up);
                    if (layout == PrefillWeightLayout::q8_lanes_v1) {
                        g.register_lane_matrix(wg, {8u, k, r, 1});
                        g.register_lane_matrix(wu, {8u, k, r, 1});
                    } else {
                        g.register_q8_matrix(wg, k, r); g.register_q8_matrix(wu, k, r);
                    }
                    if (!late) {
                        const auto pending = g.q8_tile64_cache_stats();
                        need(pending.registered_candidates == 2 && !pending.owning_allocations && !pending.aliases,
                             "registration before preparation created Q8 copies");
                        g.prepare_verify_nax({}); g.prepare_dflash2();
                    }
                    need_tile64_state(g, 2);
                    need(g.q8_tile64_cache_stats().registered_source_bytes == pay_gate.size() + pay_up.size(),
                         "registered GGUF/lane source byte accounting differs");
                    need(g.prefill_weight_layout(wg) == layout && g.prefill_weight_layout(wu) == layout,
                         "M8 registration lost original source layout");
                    const auto prepared = g.memory();
                    g.register_q8_matrix(wg, k, r); g.register_q8_matrix(wu, k, r);
                    g.prepare_verify_nax({}); g.prepare_dflash2();
                    need(g.memory().live_bytes == prepared.live_bytes && g.memory().allocations == prepared.allocations,
                         "repeated M8/DFlash preparation or registration allocated again");
                    for (const uint32_t m : {1u, 3u, 4u, 5u, 8u}) {
                        std::vector<float> x(uint64_t(m + 1u) * (k + 4u), sentinel), operands(uint64_t(m) * k);
                        uint32_t rng = 1901u + k + m;
                        for (uint32_t row = 0; row < m; ++row) for (uint32_t c = 0; c < k; ++c) {
                            rng = rng * 1664525u + 1013904223u;
                            const float v = float(int(rng % 1993u) - 996) / 997.0f;
                            x[uint64_t(row) * (k + 4u) + c] = v;
                            const uint32_t bits = uint32_t(finite_bf16(v)) << 16u;
                            float rounded; std::memcpy(&rounded, &bits, sizeof(rounded));
                            operands[uint64_t(row) * k + c] = q8_tile64_cache_enabled ? rounded : v;
                        }
                        const std::vector<float> poison(uint64_t(m + 1u) * stride, sentinel);
                        const auto xv = a.floats(x, false), yg = a.floats(poison, false);
                        const auto yu = a.floats(poison, false), yf = a.floats(poison, false);
                        std::vector<float> ref_gate(uint64_t(m) * r), ref_up(ref_gate.size()), ref_fused(ref_gate.size());
                        for (uint32_t row = 0; row < m; ++row) for (uint32_t col = 0; col < r; ++col) {
                            double dg = 0, du = 0;
                            for (uint32_t c = 0; c < k; ++c) {
                                dg += double(gate[uint64_t(col) * k + c]) * operands[uint64_t(row) * k + c];
                                du += double(up[uint64_t(col) * k + c]) * operands[uint64_t(row) * k + c];
                            }
                            const size_t at = uint64_t(row) * r + col;
                            ref_gate[at] = float(dg); ref_up[at] = float(du);
                            ref_fused[at] = float((dg / (1.0 + std::exp(-dg))) * du);
                        }
                        for (const bool draft : {false, true}) {
                            const auto live = g.memory();
                            const auto scope = [&](bool active) {
                                if (draft) g.set_dflash2_scope(active); else g.set_verify_nax_scope(active);
                            };
                            const auto execute = [&](bool fused_only = false) {
                                scope(true); g.begin();
                                const GpuRowsF32 in{xv, m, k, uint64_t(k + 4u) * 4};
                                const GpuRowsF32 gout{yg, m, r, stride * 4u}, uout{yu, m, r, stride * 4u}, fout{yf, m, r, stride * 4u};
                                if (draft) {
                                    if (!fused_only) { g.dflash2_dense_q8(wg, in, gout); g.dflash2_dense_q8(wu, in, uout); }
                                    g.dflash2_gate_up_swiglu(wg, wu, in, fout);
                                } else {
                                    if (!fused_only) { g.verify_nax_dense_q8(wg, in, gout); g.verify_nax_dense_q8(wu, in, uout); }
                                    g.verify_nax_dense_q8_swiglu(wg, wu, in, fout);
                                }
                                const auto result = g.submit(); scope(false); return result;
                            };
                            need(execute().dispatches == 3, "registered M8 dense/fused dispatch count differs");
                            need_kernel(g, 0, q8_tile64_cache_enabled ? "nax_m8_dense_q8_t64" : "nax_m8_qmv_q8", 3);
                            need_kernel(g, 1, q8_tile64_cache_enabled ? "nax_m8_dense_q8_t64" : "nax_m8_qmv_q8", 3);
                            need_kernel(g, 2, q8_tile64_cache_enabled ? "nax_m8_dense_q8_swiglu_t64" : "nax_m8_dense_q8_swiglu", 3);
                            const auto validate = [&](GpuView output, std::span<const float> ref, double limit) {
                                std::vector<float> values(poison.size()), active;
                                g.read(output, std::as_writable_bytes(std::span(values)));
                                for (uint32_t row = 0; row <= m; ++row) for (uint32_t col = 0; col < stride; ++col) {
                                    const float value = values[uint64_t(row) * stride + col];
                                    if (row == m || col >= r) need(value == sentinel, "M8 fallback wrote inactive rows or padding");
                                    else active.push_back(value);
                                }
                                need(nmse_f32(active, ref) <= limit, "registered M8 selected-operand FP64 oracle failed");
                            };
                            validate(yg, ref_gate, 2e-5); validate(yu, ref_up, 2e-5); validate(yf, ref_fused, 2e-4);
                            if (m == 3u) {
                                const auto clean = g.read(yf);
                                auto bad = x; bad[k + 4u + k - 1u] = std::numeric_limits<float>::infinity();
                                g.write(xv, std::as_bytes(std::span(bad)));
                                bool refused = false;
                                try { static_cast<void>(execute(true)); } catch (const FormatError&) { refused = true; }
                                scope(false); try { g.abort(); } catch (const FormatError&) {}
                                need(refused, "registered M8 fused-only dispatch accepted active nonfinite input");
                                g.write(xv, std::as_bytes(std::span(x))); static_cast<void>(execute());
                                need(g.read(yf) == clean, "registered M8 retry differs from its own clean result");
                            }
                            need(g.memory().live_bytes == live.live_bytes && g.memory().allocations == live.allocations,
                                 "registered M8 dispatch allocated GPU resources");
                            need_tile64_state(g, 2);
                        }
                    }
                    need(g.read(wg) == pay_gate && g.read(wu) == pay_up, "Q8 policy changed original source bytes");
                }
                need_tile64_state(g, 0);
            }
        }
    }
    std::cout << "PASS registered Q8 policy=" << q8_tile64_cache_enabled
              << " M8 verify/DFlash dense+fused original layouts, early/late registration, selected-operand oracles and route proof; profiling is diagnostic only\n";
}


// Target-only split controls use the ordinary metallib. Carrier batch records
// prove the encoded GroupedMmArgs outside timing; no counter profiling/trace.
#if defined(TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC) && defined(TOKSPUR_PRIVATE_CONCURRENT_DISPATCH)
template <class F> Json shared_split_record(F&& execute) {
    FILE* file = std::tmpfile();
    need(file != nullptr, "cannot create split-proof capture");
    const int saved = ::dup(STDERR_FILENO);
    if (saved < 0) { std::fclose(file); throw FormatError("cannot save split-proof stderr"); }
    try {
        need(std::fflush(stderr) == 0 && ::dup2(::fileno(file), STDERR_FILENO) >= 0, "cannot redirect split-proof stderr");
        execute();
        need(std::fflush(stderr) == 0 && ::dup2(saved, STDERR_FILENO) >= 0, "cannot restore split-proof stderr");
        need(std::fseek(file, 0, SEEK_END) == 0, "cannot size split-proof capture");
        const long bytes = std::ftell(file);
        need(bytes > 0 && bytes <= 65536 && std::fseek(file, 0, SEEK_SET) == 0, "split-proof capture bound");
        std::string raw(size_t(bytes), '\0');
        need(std::fread(raw.data(), 1, raw.size(), file) == raw.size(), "cannot read split-proof capture");
        // Preserve the exact graph-emitted evidence, including action/clock rows.
        need(std::fwrite(raw.data(), 1, raw.size(), stderr) == raw.size() && std::fflush(stderr) == 0, "cannot retain split-proof capture");
        Json found; uint32_t count = 0;
        for (size_t begin = 0; begin < raw.size();) {
            const auto end = raw.find('\n', begin);
            need(end != std::string::npos, "unterminated split-proof record");
            const auto row = parse_json(std::string_view(raw).substr(begin, end - begin));
            if (const auto* event = row.find("event"); event && event->string() == "carrier_diagnostic_batch") { found = row; ++count; }
            begin = end + 1;
        }
        need(count == 1, "split proof must contain exactly one native batch");
        ::close(saved); std::fclose(file); return found;
    } catch (...) {
        std::fflush(stderr); ::dup2(saved, STDERR_FILENO); ::close(saved); std::fclose(file); throw;
    }
}

void test_target_shared_splits(const char* metallib, bool dual, uint32_t candidate) {
    need(q8_tile64_cache_enabled, "shared split control requires actual Tile64 copies");
    need(dual ? (candidate == 8u || candidate == 4u) : (candidate == 4u || candidate == 2u), "unsupported shared split candidate");
    const uint32_t columns = dual ? 2560u : 640u, outputs = dual ? 640u : 2560u;
    const uint32_t automatic = dual ? 16u : 5u, tile = dual ? 32u : 64u;
    constexpr uint32_t rows = 8u, iterations = 16u, trials = 4u, cycles = 3u;
    constexpr float sentinel = -12345.25f;
    const uint32_t xs = columns + 4u, ys = outputs + 4u;
    const char* kernel = dual ? "nax_m8_dense_q8_swiglu_t64" : "nax_m8_dense_q8_t64";
    // Existing registered-M8 oracle limits; reduction-order changes are reported,
    // never silently converted to a bit-identity claim or a new tolerance.
    const double limit = dual ? 2e-4 : 2e-5;
    const grouped_fixture::Shape shape{8u, columns, outputs, 1u, 1u, 1u, true, PrefillWeightLayout::gguf};
    const auto raw_gate = grouped_fixture::weights(shape);
    auto raw_up = raw_gate;
    for (size_t i = 0; i < raw_up.size(); ++i) if (i % 34u >= 2u) raw_up[i] ^= std::byte((i * 13u + 7u) & 0x0fu);
    auto gate = grouped_fixture::represented(shape, raw_gate), up = grouped_fixture::represented(shape, raw_up);
    for (auto& v : gate) v = float(_Float16(v));
    for (auto& v : up) v = float(_Float16(v));
    std::vector<float> input(uint64_t(rows + 1u) * xs, sentinel), operands(uint64_t(rows) * columns);
    uint32_t rng = 1901u + columns + outputs;
    for (uint32_t row = 0; row < rows; ++row) for (uint32_t k = 0; k < columns; ++k) {
        rng = rng * 1664525u + 1013904223u;
        const float v = float(int(rng % 1993u) - 996) / 997.0f;
        input[uint64_t(row) * xs + k] = v;
        const uint32_t bits = uint32_t(finite_bf16(v)) << 16u;
        std::memcpy(&operands[uint64_t(row) * columns + k], &bits, sizeof(bits));
    }
    std::vector<float> reference(uint64_t(rows) * outputs);
    for (uint32_t row = 0; row < rows; ++row) for (uint32_t col = 0; col < outputs; ++col) {
        double dg = 0, du = 0;
        for (uint32_t k = 0; k < columns; ++k) {
            dg += double(gate[uint64_t(col) * columns + k]) * operands[uint64_t(row) * columns + k];
            if (dual) du += double(up[uint64_t(col) * columns + k]) * operands[uint64_t(row) * columns + k];
        }
        reference[uint64_t(row) * outputs + col] = dual ? float((dg / (1.0 + std::exp(-dg))) * du) : float(dg);
    }
    const std::vector<float> poison(uint64_t(rows + 1u) * ys, sentinel);
    const auto entry = Json(Json::Object{{"schema", Json::integer(1)}, {"entries", Json(Json::Array{Json(Json::Object{
        {"family", Json("q8-t64")}, {"columns", Json::integer(columns)}, {"outputs", Json::integer(outputs)},
        {"tokens", Json::integer(rows)}, {"tile", Json::integer(tile)}, {"dual", Json(dual)}, {"splits", Json::integer(candidate)}})})}});
    const auto overrides = NaxSplitOverrides::parse(entry);
    struct ArmResult { std::vector<float> values; uint64_t live, allocations; };
    const auto arm = [&](bool overridden, bool target, bool proof, uint32_t leg) -> ArmResult {
        MetalGraph g(metallib, 256ull << 20);
        g.configure_concurrent_dispatch(true);
        g.configure_small_gemv(16); g.configure_block_gemv(3);
        // Required by the existing carrier clock admission, in every arm.
        g.configure_qsa_stage_v8(true);
        g.configure_verify_nax(true); g.configure_verify_nax_m8(true);
        if (overridden) g.configure_target_m8_splits(overrides);
        need(g.target_m8_split_digest() == (overridden ? overrides.digest() : std::string{}), "split policy digest mismatch");
        g.prepare_verify_nax({});
        const auto prepared_bytes = g.memory().live_bytes;
        ArmResult result;
        {
            Arena a{g, {}};
            const auto wg = a.upload(raw_gate);
            const auto wu = dual ? a.upload(raw_up) : GpuView{};
            g.register_q8_matrix(wg, columns, outputs);
            if (dual) g.register_q8_matrix(wu, columns, outputs);
            need_tile64_state(g, dual ? 2u : 1u);
            need(g.q8_tile64_cache_stats().registered_source_bytes == raw_gate.size() * (dual ? 2u : 1u), "shared split source identity mismatch");
            const auto xv = a.floats(input, false), yv = a.floats(poison, false);
            const auto stable = g.memory();
            const auto execute = [&](uint32_t count) {
                g.set_verify_nax_scope(true); g.set_target_verification_role(target);
                try {
                    g.begin();
                    for (uint32_t i = 0; i < count; ++i) {
                        const GpuRowsF32 x{xv, rows, columns, uint64_t(xs) * 4}, y{yv, rows, outputs, uint64_t(ys) * 4};
                        if (dual) g.verify_nax_dense_q8_swiglu(wg, wu, x, y);
                        else g.verify_nax_dense_q8(wg, x, y);
                    }
                    const auto batch = g.submit();
                    g.set_target_verification_role(false); g.set_verify_nax_scope(false);
                    need(batch.dispatches == count && !batch.capture_executes, "shared split dispatch count or capture mismatch");
                    need(g.memory().live_bytes == stable.live_bytes && g.memory().allocations == stable.allocations, "shared split dispatch changed resources");
                    return batch;
                } catch (...) { g.set_target_verification_role(false); g.set_verify_nax_scope(false); g.abort(); throw; }
            };
            if (proof) {
                const auto record = shared_split_record([&] {
                    g.prepare_carrier_diagnostic(CarrierDiagnosticMode::batch, 1);
                    need(g.begin_carrier_diagnostic(false, true), "shared split proof did not activate");
                    static_cast<void>(execute(1)); g.end_carrier_diagnostic(); g.finish_carrier_diagnostic();
                });
                need(!record.find("serial_counter_schedule")->boolean() && record.find("dispatches")->unsigned_integer() == 1 &&
                    record.find("dispatch_records")->unsigned_integer() == 1, "shared split proof is not one unprofiled concurrent dispatch");
                const auto& actual = record.find("kernels")->array();
                need(actual.size() == 1 && actual[0].find("kernel")->string() == kernel &&
                    actual[0].find("selected_or_splits")->unsigned_integer() == (overridden && target ? candidate : automatic), "shared split encode selection differs");
                const auto& dims = actual[0].find("shape")->array();
                need(dims.size() == 4 && dims[0].unsigned_integer() == columns && dims[1].unsigned_integer() == outputs &&
                    dims[2].unsigned_integer() == rows && dims[3].unsigned_integer() == uint32_t(PrefillWeightLayout::gguf), "shared split encoded shape differs");
                std::cout << "SHARED_SPLIT_PROOF " << Json(Json::Object{{"dual", Json(dual)}, {"override", Json(overridden)},
                    {"target_role", Json(target)}, {"expected_splits", Json::integer(overridden && target ? candidate : automatic)}, {"record", record}}).dump() << '\n';
            } else static_cast<void>(execute(1));
            result.values.resize(poison.size());
            g.read(yv, std::as_writable_bytes(std::span(result.values)));
            std::vector<float> active;
            for (uint32_t row = 0; row <= rows; ++row) for (uint32_t col = 0; col < ys; ++col) {
                const float v = result.values[uint64_t(row) * ys + col];
                if (row == rows || col >= outputs) need(v == sentinel, "shared split overwrote row or stride guard");
                else active.push_back(v);
            }
            const double nmse = nmse_f32(active, reference);
            need(nmse <= limit, "shared split selected-operand FP64 oracle failed");
            std::cout << "SHARED_SPLIT_ORACLE {\"dual\":" << dual << ",\"override\":" << overridden << ",\"target_role\":" << target
                      << ",\"proof\":" << proof << ",\"leg\":" << leg << ",\"nmse\":" << nmse << ",\"limit\":" << limit << "}\n";
            if (!proof) {
                need(g.concurrent_dispatch() && g.last_profile().mode == ProfileMode::disabled && !g.batch_phase_clock_state().enabled,
                     "shared split timing must remain concurrent, unprofiled and without phase-clock telemetry");
                for (const float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
                    auto bad = input; bad[uint64_t(rows - 1u) * xs + columns - 1u] = invalid;
                    g.write(xv, std::as_bytes(std::span(bad)));
                    bool refused = false;
                    try { static_cast<void>(execute(1)); } catch (const FormatError&) { refused = true; }
                    need(refused, "shared split accepted nonfinite active input");
                    g.write(xv, std::as_bytes(std::span(input))); g.write(yv, std::as_bytes(std::span(poison)));
                    static_cast<void>(execute(1));
                    need(g.read(yv) == std::vector<std::byte>(std::as_bytes(std::span(result.values)).begin(), std::as_bytes(std::span(result.values)).end()), "shared split retry differs from own clean result");
                }
                const auto warm_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
                do { static_cast<void>(execute(iterations)); } while (std::chrono::steady_clock::now() < warm_end);
                for (uint32_t trial = 0; trial < trials; ++trial) {
                    const auto thermal_before = host_thermal_state();
                    const auto start = std::chrono::steady_clock::now();
                    const auto batch = execute(iterations);
                    const double host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                    const auto thermal_after = host_thermal_state();
                    need(std::isfinite(batch.gpu_ms) && batch.gpu_ms > 0 && std::isfinite(host_ms) && host_ms > 0, "shared split invalid component clocks");
                    std::cout << "SHARED_SPLIT_TIMING {\"dual\":" << dual << ",\"columns\":" << columns << ",\"outputs\":" << outputs
                              << ",\"rows\":8,\"tile\":" << tile << ",\"splits\":" << (overridden ? candidate : automatic)
                              << ",\"target_role\":true,\"concurrent\":true,\"profile\":false,\"phase_clock\":false,\"cycle\":" << leg / 4u << ",\"leg\":" << leg
                              << ",\"trial\":" << trial << ",\"iterations\":" << iterations << ",\"batch_gpu_ms\":" << batch.gpu_ms
                              << ",\"iteration_gpu_ms\":" << batch.gpu_ms / iterations << ",\"batch_host_ms\":" << host_ms
                              << ",\"iteration_host_ms\":" << host_ms / iterations << ",\"thermal_before\":" << thermal_before
                              << ",\"thermal_after\":" << thermal_after << ",\"live_bytes\":" << stable.live_bytes << ",\"allocations\":" << stable.allocations << "}\n";
                }
                need(g.read(yv) == std::vector<std::byte>(std::as_bytes(std::span(result.values)).begin(), std::as_bytes(std::span(result.values)).end()), "shared split repeated timing changed output bits");
            }
            need(g.read(xv) == std::vector<std::byte>(std::as_bytes(std::span(input)).begin(), std::as_bytes(std::span(input)).end()), "shared split changed input/guards");
            need(g.read(wg) == raw_gate && (!dual || g.read(wu) == raw_up), "shared split changed immutable source bytes");
            result.live = stable.live_bytes; result.allocations = stable.allocations;
        }
        need_tile64_state(g, 0);
        need(g.memory().live_bytes == prepared_bytes, "shared split leaked registered copies or fixture buffers");
        return result;
    };
    const auto base = arm(false, true, true, 0);
    const auto role_off = arm(true, false, true, 0);
    need(role_off.values == base.values && std::memcmp(role_off.values.data(), base.values.data(), base.values.size() * sizeof(float)) == 0, "target override changed automatic role-off output bits");
    const auto changed = arm(true, true, true, 0);
    need(base.live == role_off.live && base.live == changed.live && base.allocations == role_off.allocations && base.allocations == changed.allocations,
         "shared split override changed GPU resources");
    uint64_t mismatches = 0; double max_abs = 0;
    for (size_t i = 0; i < base.values.size(); ++i) {
        if (std::memcmp(&base.values[i], &changed.values[i], sizeof(float))) ++mismatches;
        max_abs = std::max(max_abs, std::abs(double(base.values[i]) - changed.values[i]));
    }
    std::cout << "SHARED_SPLIT_PARITY {\"dual\":" << dual << ",\"automatic\":" << automatic << ",\"candidate\":" << candidate
              << ",\"role_off_byte_equal\":true,\"cross_split_byte_equal\":" << (mismatches ? "false" : "true")
              << ",\"padded_float_bit_mismatches\":" << mismatches << ",\"max_absolute_error\":" << max_abs << "}\n";
    for (uint32_t leg = 0; leg < cycles * 4u; ++leg) {
        const bool selected = leg % 4u == 1u || leg % 4u == 2u;
        const auto measured = arm(selected, true, false, leg);
        const auto& expected = selected ? changed : base;
        need(measured.live == expected.live && measured.allocations == expected.allocations, "shared split timing/proof resource inventory differs");
        need(std::memcmp(measured.values.data(), expected.values.data(), expected.values.size() * sizeof(float)) == 0, "shared split timing/proof output bits differ");
    }
    std::cout << "PASS target-only shared " << (dual ? "GU" : "down") << " split " << automatic << '/' << candidate
              << " encode proof, independent oracle, role-off exact parity, finite/guards and three concurrent unprofiled ABBA cycles; component clocks only\n";
}
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << "Q8_TILE64_CACHE=" << q8_tile64_cache_enabled << '\n';
        if (argc == 4 && (std::strcmp(argv[2], "--target-shared-gu-splits") == 0 || std::strcmp(argv[2], "--target-shared-down-splits") == 0)) {
#if defined(TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC) && defined(TOKSPUR_PRIVATE_CONCURRENT_DISPATCH)
            const bool dual = std::strcmp(argv[2], "--target-shared-gu-splits") == 0;
            const uint32_t candidate = std::strcmp(argv[3], "8") == 0 ? 8u : std::strcmp(argv[3], "4") == 0 ? 4u : std::strcmp(argv[3], "2") == 0 ? 2u : 0u;
            test_target_shared_splits(argv[1], dual, candidate);
            return 0;
#else
            throw FormatError("target shared split controls require the existing carrier-diagnostic concurrent build");
#endif
        }
        if (argc == 3 && std::strcmp(argv[2], "--q8-cache-policy-only") == 0) {
            test_dense_nax(argv[1], true);
            test_nax_dense_q8_swiglu_and_fast_router(argv[1], true);
            test_q8_cache_preparation_entries(argv[1]);
            test_registered_q8_m8_cache_policy(argv[1]);
            test_nax_gdn_projection_gates(argv[1]);
            return 0;
        }
        if (argc == 3 && std::strcmp(argv[2], "--q8-t64-m8-single-only") == 0) {
            static_cast<void>(test_registered_q8_t64_m8_single(argv[1]));
            return 0;
        }
        if (argc == 4 && std::strcmp(argv[2], "--q8-t64-m8-single-control") == 0) {
            std::vector<uint64_t> control_resources;
            const auto control = test_registered_q8_t64_m8_single(argv[3], "double_stage", 0u, &control_resources);
            for (uint32_t leg = 1u; leg < 4u; ++leg) {
                const bool candidate = leg < 3u;
                std::vector<uint64_t> resources;
                const auto result = test_registered_q8_t64_m8_single(argv[candidate ? 1 : 3], candidate ? "single_stage" : "double_stage", leg, &resources);
                need(resources == control_resources, "registered M8 single-stage changed live GPU allocations or bytes");
                need(result.size() == control.size() && std::memcmp(result.data(), control.data(), result.size() * sizeof(float)) == 0,
                    "registered M8 single-stage candidate/control padded output bytes differ");
            }
            std::cout << "PASS registered Q8 M8 exact padded output parity, unchanged GPU resources and nominal component ABBA\n";
            return 0;
        }
        if (argc == 4 && std::strcmp(argv[2], "--q8-t64-m8-concurrent-control") == 0) {
            std::vector<uint64_t> control_resources;
            const auto control = test_registered_q8_t64_m8_single(argv[3], "double_stage", 0u, &control_resources, true);
            for (uint32_t leg = 1u; leg < 4u; ++leg) {
                const bool candidate = leg < 3u;
                std::vector<uint64_t> resources;
                const auto result = test_registered_q8_t64_m8_single(argv[candidate ? 1 : 3], candidate ? "single_stage" : "double_stage", leg, &resources, true);
                need(resources == control_resources, "concurrent M8 single-stage changed live GPU allocations or bytes");
                need(result.size() == control.size() && std::memcmp(result.data(), control.data(), result.size() * sizeof(float)) == 0,
                    "concurrent M8 single-stage candidate/control padded output bytes differ");
            }
            std::cout << "PASS concurrent Q8 M8 original/head shapes, both roles, exact padded output parity, unchanged GPU resources and one ABBA screen\n";
            return 0;
        }
        if (argc == 3 && std::strcmp(argv[2], "--q8-t64-m32-only") == 0) {
            static_cast<void>(test_registered_q8_t64_m32(argv[1]));
            static_cast<void>(test_registered_q8_t64_m32_large(argv[1]));
            test_registered_q8_t64_m32_scale_edges(argv[1]);
            return 0;
        }
        if (argc == 4 && std::strcmp(argv[2], "--q8-t64-m32-control") == 0) {
            const auto control = test_registered_q8_t64_m32(argv[3], "control", 0u);
            const auto large_control = test_registered_q8_t64_m32_large(argv[3], "control", 0u);
            test_registered_q8_t64_m32_scale_edges(argv[3]);
            for (uint32_t leg = 1u; leg < 4u; ++leg) {
                const bool candidate = leg < 3u;
                const auto result = test_registered_q8_t64_m32(argv[candidate ? 1 : 3], candidate ? "candidate" : "control", leg);
                need(result.size() == control.size() && std::memcmp(result.data(), control.data(), result.size() * sizeof(float)) == 0,
                    "registered Q8 M32 candidate/control output bytes differ");
                const auto large_result = test_registered_q8_t64_m32_large(argv[candidate ? 1 : 3], candidate ? "candidate" : "control", leg);
                need(large_result == large_control, "large registered Q8 M32 candidate/control output digests differ");
                test_registered_q8_t64_m32_scale_edges(argv[candidate ? 1 : 3]);
            }
            std::cout << "PASS registered Q8 tile64 M32 exact small-output and large-output digest parity with nominal component ABBA\n";
            return 0;
        }
        if (argc != 2) throw FormatError("usage: test-nax_prefill_metal OWNED_METALLIB [--target-shared-gu-splits 8|4 | --target-shared-down-splits 4|2 | --q8-cache-policy-only | --q8-t64-m32-only | --q8-t64-m32-control CONTROL_METALLIB | --q8-t64-m8-single-only | --q8-t64-m8-single-control CONTROL_METALLIB | --q8-t64-m8-concurrent-control CONTROL_METALLIB]");
        test_dense_nax(argv[1]);
        test_q8_cache_preparation_entries(argv[1]);
        test_registered_q8_m8_cache_policy(argv[1]);
        static_cast<void>(test_registered_q8_t64_m8_single(argv[1]));
        static_cast<void>(test_registered_q8_t64_m32(argv[1]));
        static_cast<void>(test_registered_q8_t64_m32_large(argv[1]));
        test_registered_q8_t64_m32_scale_edges(argv[1]);
        test_grouped_nax(argv[1]);
        test_nax_qsa_attention_and_bf16_reuse(argv[1]);
        for (const uint32_t capacity : {1024u, 2048u}) test_stack_2048_and_ple_nax(argv[1], capacity);
        test_mtp_priming_nax(argv[1]);
        test_verify_nax_routed_moe_all_encodings(argv[1]);
        test_verify_nax_m2_m3_m8(argv[1]);
        test_dflash2_kernels(argv[1]);
        test_dflash2_verify_budget();
        test_nax_dense_q8_swiglu_and_fast_router(argv[1]);
        test_nax_dense_bf16_and_verify_bf16(argv[1]);
        test_nax_gdn_projection_gates(argv[1]);
        std::cout << "ALL NAX PREFILL METAL CHECKS PASSED (" << total_checks << " checks)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
