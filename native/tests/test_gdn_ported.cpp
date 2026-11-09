#include "tokspur/gdn_chunk.hpp"
#include "tokspur/gdn_core_compare.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/mtp_wire.hpp"
#include "tokspur/operators.hpp"
#include "tokspur/q8_many.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace tokspur;

constexpr float sentinel = -321.25f;
constexpr double nmse_bound = 1e-7;

size_t checks = 0, batches = 0;
double worst_prefill_nmse = 0.0, worst_prepare_nmse = 0.0, worst_verify_nmse = 0.0;
double worst_replay_nmse = 0.0, worst_norm_gate_nmse = 0.0;

void require(bool ok, const char* why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}

void reject(const std::function<void()>& action) {
    ++checks;
    try {
        action();
    } catch (const FormatError&) {
        return;
    }
    throw std::runtime_error("invalid ported GDN operation admitted");
}

bool same_bits(float a, float b) {
    return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b);
}

void same_memory(GraphMemory a, GraphMemory b) {
    require(a.live_bytes == b.live_bytes && a.peak_bytes == b.peak_bytes && a.allocations == b.allocations &&
        a.imported_backing_bytes == b.imported_backing_bytes &&
        a.profile_counter_payload_bytes == b.profile_counter_payload_bytes &&
        a.capture_resource_bytes == b.capture_resource_bytes && a.live_captures == b.live_captures,
        "ported GDN changed the prepared allocation ledger");
}

template <class T>
std::vector<T> read_vec(MetalGraph& graph, GpuView view) {
    const auto bytes = graph.read(view);
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

double compute_nmse(std::span<const float> got, std::span<const double> ref) {
    require(got.size() == ref.size() && !got.empty(), "GDN NMSE span mismatch");
    double err = 0.0, power = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]) && std::isfinite(ref[i]), "non-finite GDN output or reference");
        const double d = double(got[i]) - ref[i];
        err += d * d;
        power += ref[i] * ref[i];
    }
    if (power == 0.0) return err == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    return err / power;
}

float generated(size_t i, uint32_t seed = 44) {
    return float(int((i * 131ull + uint64_t(seed) * 17ull) % 1993ull) - 996) / 997.0f;
}

struct Arena {
    MetalGraph& g;
    std::vector<GpuView> full;
    GpuView bytes(std::span<const std::byte> data, bool immutable = false) {
        std::vector<std::byte> padded(data.size() + 8, std::byte{0x69});
        std::copy(data.begin(), data.end(), padded.begin() + 4);
        const auto allocation = g.upload(padded, "generated guarded ported GDN fixture", immutable);
        full.push_back(allocation);
        return g.slice(allocation, 4, data.size());
    }
    template <class T>
    GpuView array(std::span<const T> data, bool immutable = false) {
        return bytes(std::as_bytes(data), immutable);
    }
    GpuView floats(size_t count, float value = 0.0f) {
        return array<float>(std::vector<float>(count, value));
    }
    void guards() {
        for (const auto view : full) {
            for (const auto at : {uint64_t(0), view.bytes - 4}) {
                require(g.read(g.slice(view, at, 4)) == std::vector<std::byte>(4, std::byte{0x69}),
                    "ported GDN outer canary changed");
            }
        }
    }
    ~Arena() {
        try {
            g.abort();
        } catch (const FormatError&) {
        }
        for (const auto view : full) g.release(view);
    }
};

void prepare_graph(MetalGraph& g, bool ported, bool concurrent = false) {
    g.configure_small_gemv(16);
    g.configure_block_gemv(3);
    g.configure_hc_exact(HcExactMode::combined);
    g.configure_concurrent_dispatch(concurrent);
    g.prepare_gdn_chunk();
    g.prepare_gdn_tiles(128);
    for (uint32_t w : {1u, 2u, 4u, 8u}) g.prepare_q8_matvec_many(w);
    g.configure_gdn_ported(ported);
    g.prepare_gdn_ported_variants();
}

void run_prefill_case(MetalGraph& ported_g, MetalGraph& conc_g, MetalGraph& base_g,
    uint32_t rows, uint32_t heads, uint32_t dv, bool padded_strides, bool strong_decay = false) {
    Arena ap{ported_g, {}}, ac{conc_g, {}}, ab{base_g, {}};
    constexpr uint32_t dk = 128;
    const uint32_t qs = heads * dk + (padded_strides ? 3u : 0u);
    const uint32_t vs = heads * dv + (padded_strides ? 5u : 0u);
    const uint32_t gs = heads + (padded_strides ? 2u : 0u);
    const size_t state_n = size_t(heads) * dk * dv;

    std::vector<float> s(state_n);
    std::vector<float> q(size_t(rows + 1) * qs, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> k(q.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> v(size_t(rows + 1) * vs, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> logg(size_t(rows + 1) * gs, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> beta(logg.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<double> qr, kr, vr, gr, br;
    qr.reserve(size_t(rows) * heads * dk);
    kr.reserve(size_t(rows) * heads * dk);
    vr.reserve(size_t(rows) * heads * dv);
    gr.reserve(size_t(rows) * heads);
    br.reserve(size_t(rows) * heads);

    for (size_t i = 0; i < s.size(); ++i) s[i] = generated(i) * 0.01f;
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t j = 0; j < dk; ++j) {
                const float qv = generated((size_t(t) * heads + h) * dk + j) * 0.02f;
                const float kv = generated((size_t(t) * heads + h) * dk + j + 31) * 0.02f;
                q[size_t(t) * qs + h * dk + j] = qv;
                k[size_t(t) * qs + h * dk + j] = kv;
                qr.push_back(qv);
                kr.push_back(kv);
            }
            for (uint32_t j = 0; j < dv; ++j) {
                const float vv = generated((size_t(t) * heads + h) * dv + j + 17) * 0.03f;
                v[size_t(t) * vs + h * dv + j] = vv;
                vr.push_back(vv);
            }
            const float lg = strong_decay ? -1000.0f : -float((t + h) % 7 + 1) / 89.0f;
            const float bt = float((t + h) % 13) / 12.0f;
            logg[size_t(t) * gs + h] = lg;
            beta[size_t(t) * gs + h] = bt;
            gr.push_back(lg);
            br.push_back(bt);
        }
    }

    const auto expected = gdn_chunk_reference(
        std::vector<double>(s.begin(), s.end()), qr, kr, vr, gr, br, heads, dk, dv, rows, 1);

    const auto dispatch_one = [&](MetalGraph& g, Arena& a, GpuView& next_out, GpuView& y_out) {
        const auto accepted = a.array<float>(s, true);
        const auto qv = a.array<float>(q, true), kv = a.array<float>(k, true), vv = a.array<float>(v, true);
        const auto gv = a.array<float>(logg, true), bv = a.array<float>(beta, true);
        next_out = a.floats(state_n, sentinel);
        y_out = a.floats(size_t(rows + 1) * vs, sentinel);
        GdnChunkDispatch d{};
        d.op = GdnChunkOp::fused;
        d.params.rows = rows;
        d.params.heads = heads;
        d.params.key_width = dk;
        d.params.value_width = dv;
        d.params.input_stride = uint64_t(qs) * 4;
        d.params.auxiliary_stride = d.params.output_stride = uint64_t(vs) * 4;
        d.params.gate_stride = uint64_t(gs) * 4;
        d.buffers[0] = accepted;
        d.buffers[1] = qv;
        d.buffers[2] = kv;
        d.buffers[3] = vv;
        d.buffers[4] = gv;
        d.buffers[5] = bv;
        d.buffers[6] = next_out;
        d.buffers[8] = y_out;
        const auto mem_before = g.memory();
        g.begin();
        g.gdn_chunk(d);
        const auto res = g.submit();
        ++batches;
        require(res.dispatches == 1 && !res.capture_executes, "ported GDN prefill dispatch count differs");
        same_memory(mem_before, g.memory());
        require(read_vec<float>(g, accepted) == s, "ported GDN prefill mutated accepted state");
    };

    const auto routes_before = ported_g.gdn_ported_routes();
    GpuView p_state{}, p_out{}, p_rep_state{}, p_rep_out{}, c_state{}, c_out{}, b_state{}, b_out{};
    dispatch_one(ported_g, ap, p_state, p_out);
    require(ported_g.gdn_ported_routes() == GdnPortedRoutes{routes_before[0] + 1, routes_before[1], routes_before[2], routes_before[3], routes_before[4]},
        "ported GDN prefill route counter delta differs");
    dispatch_one(ported_g, ap, p_rep_state, p_rep_out);
    require(ported_g.read(p_state) == ported_g.read(p_rep_state) && ported_g.read(p_out) == ported_g.read(p_rep_out),
        "ported GDN prefill own-repeat bytes differ");

    dispatch_one(conc_g, ac, c_state, c_out);
    require(conc_g.read(c_state) == ported_g.read(p_state) && conc_g.read(c_out) == ported_g.read(p_out),
        "ported GDN prefill concurrent encoder bytes differ");

    dispatch_one(base_g, ab, b_state, b_out);
    require(base_g.gdn_ported_routes() == GdnPortedRoutes{}, "disabled ported GDN incremented prefill counter");

    const auto got_out = read_vec<float>(ported_g, p_out);
    const auto got_state = read_vec<float>(ported_g, p_state);
    std::vector<float> compact_out;
    compact_out.reserve(size_t(rows) * heads * dv);
    for (uint32_t t = 0; t <= rows; ++t) {
        for (uint32_t j = 0; j < vs; ++j) {
            if (t < rows && j < heads * dv) {
                compact_out.push_back(got_out[size_t(t) * vs + j]);
            } else {
                require(same_bits(got_out[size_t(t) * vs + j], sentinel), "ported GDN prefill wrote tail or padding");
            }
        }
    }
    const double out_nmse = compute_nmse(compact_out, expected.output);
    const double state_nmse = compute_nmse(got_state, expected.state);
    worst_prefill_nmse = std::max({worst_prefill_nmse, out_nmse, state_nmse});
    require(out_nmse <= nmse_bound && state_nmse <= nmse_bound,
        "ported GDN prefill exceeded 1e-7 FP64 reference NMSE bound");

    ap.guards();
    ac.guards();
    ab.guards();
}

void run_prepare_case(MetalGraph& g, GdnShape shape, GdnPolicy policy, StateTile tile) {
    Arena a{g, {}};
    const uint64_t raw = gdn_qkv_count(shape);
    const uint64_t qk = uint64_t(shape.value_heads) * 128;
    const uint64_t val = uint64_t(shape.value_heads) * shape.value_width;

    std::vector<float> x(tile.capacity * raw, std::numeric_limits<float>::quiet_NaN());
    for (uint32_t t = 0; t < tile.steps; ++t) {
        for (uint64_t c = 0; c < raw; ++c) {
            x[uint64_t(t) * raw + c] = (c % 257ull == 0ull) ? 0.0f : generated(uint64_t(t) * raw + c, 19) * 0.05f;
        }
    }

    const auto xv = a.array<float>(x, true);
    const auto qv = a.floats(tile.capacity * qk, sentinel);
    const auto kv = a.floats(tile.capacity * qk, sentinel);
    const auto vv = a.floats(tile.capacity * val, sentinel);
    const auto q_rep = a.floats(tile.capacity * qk, sentinel);
    const auto k_rep = a.floats(tile.capacity * qk, sentinel);
    const auto v_rep = a.floats(tile.capacity * val, sentinel);

    const auto mem_before = g.memory();
    g.begin();
    g.gdn_prepare_rows(xv, qv, kv, vv, shape, policy, tile);
    const auto res = g.submit();
    ++batches;
    require(res.dispatches == 1, "gdn_prepare_rows dispatch count differs");
    same_memory(mem_before, g.memory());

    g.begin();
    g.gdn_prepare_rows(xv, q_rep, k_rep, v_rep, shape, policy, tile);
    static_cast<void>(g.submit());
    ++batches;
    require(g.read(qv) == g.read(q_rep) && g.read(kv) == g.read(k_rep) && g.read(vv) == g.read(v_rep),
        "gdn_prepare_rows own-repeat bytes differ");

    const auto got_q = read_vec<float>(g, qv);
    const auto got_k = read_vec<float>(g, kv);
    const auto got_v = read_vec<float>(g, vv);
    for (uint32_t t = 0; t < tile.capacity; ++t) {
        const auto qs = std::span<const float>(got_q).subspan(uint64_t(t) * qk, qk);
        const auto ks = std::span<const float>(got_k).subspan(uint64_t(t) * qk, qk);
        const auto vs = std::span<const float>(got_v).subspan(uint64_t(t) * val, val);
        if (t < tile.steps) {
            std::vector<double> in_row(x.begin() + uint64_t(t) * raw, x.begin() + uint64_t(t + 1) * raw);
            const auto ref = prepare_gdn_reference(in_row, shape, policy);
            const double qn = compute_nmse(qs, ref.query);
            const double kn = compute_nmse(ks, ref.key);
            const double vn = compute_nmse(vs, ref.value);
            worst_prepare_nmse = std::max({worst_prepare_nmse, qn, kn, vn});
            require(qn <= nmse_bound && kn <= nmse_bound && vn <= nmse_bound,
                "gdn_prepare_rows exceeded 1e-7 FP64 reference NMSE bound");
        } else {
            for (float f : qs) require(same_bits(f, sentinel), "gdn_prepare_rows wrote inactive query row");
            for (float f : ks) require(same_bits(f, sentinel), "gdn_prepare_rows wrote inactive key row");
            for (float f : vs) require(same_bits(f, sentinel), "gdn_prepare_rows wrote inactive value row");
        }
    }
    a.guards();
}

struct VerifyReference {
    std::vector<double> state, output, decay, beta;
};

VerifyReference verify_reference(
    std::span<const float> state_in, std::span<const float> q, std::span<const float> k, std::span<const float> v,
    std::span<const float> alpha, std::span<const float> beta_logits, std::span<const float> rates, std::span<const float> bias,
    uint32_t heads, uint32_t dv, uint32_t steps) {
    VerifyReference ref;
    ref.state.assign(state_in.begin(), state_in.end());
    ref.output.resize(size_t(steps) * heads * dv, 0.0);
    ref.decay.resize(size_t(steps) * heads, 0.0);
    ref.beta.resize(size_t(steps) * heads, 0.0);
    for (uint32_t t = 0; t < steps; ++t) {
        for (uint32_t h = 0; h < heads; ++h) {
            const size_t gi = size_t(t) * heads + h;
            const double log_g = double(rates[h]) * stable_softplus(double(alpha[gi]) + double(bias[h]));
            const double gt = std::exp(log_g);
            const double bt = stable_sigmoid(double(beta_logits[gi]));
            ref.decay[gi] = gt;
            ref.beta[gi] = bt;
            const size_t qk_base = (size_t(t) * heads + h) * 128;
            const size_t v_base = (size_t(t) * heads + h) * dv;
            const size_t st_head = size_t(h) * 128 * dv;
            for (uint32_t c = 0; c < dv; ++c) {
                double kv = 0.0;
                for (uint32_t j = 0; j < 128; ++j) {
                    const size_t idx = st_head + size_t(j) * dv + c;
                    ref.state[idx] *= gt;
                    kv += ref.state[idx] * double(k[qk_base + j]);
                }
                const double delta = (double(v[v_base + c]) - kv) * bt;
                double out = 0.0;
                for (uint32_t j = 0; j < 128; ++j) {
                    const size_t idx = st_head + size_t(j) * dv + c;
                    ref.state[idx] += double(k[qk_base + j]) * delta;
                    out += ref.state[idx] * double(q[qk_base + j]);
                }
                ref.output[v_base + c] = out;
            }
        }
    }
    return ref;
}

void run_verify_and_replay_case(MetalGraph& g, uint32_t heads, uint32_t dv, StateTile tile, bool extreme_gates = false) {
    Arena a{g, {}};
    const size_t state_n = size_t(heads) * 128 * dv;
    const size_t qk_n = size_t(tile.capacity) * heads * 128;
    const size_t val_n = size_t(tile.capacity) * heads * dv;
    const size_t gate_n = size_t(tile.capacity) * heads;

    std::vector<float> s0(state_n);
    std::vector<float> q(qk_n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> k(qk_n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> v(val_n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> alpha(gate_n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> beta_logits(gate_n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> rates(heads), bias(heads);

    for (size_t i = 0; i < state_n; ++i) s0[i] = generated(i, 7) * 0.02f;
    for (uint32_t h = 0; h < heads; ++h) {
        rates[h] = -float(h + 1) / 97.0f;
        bias[h] = generated(h, 11) * 0.1f;
    }
    for (uint32_t t = 0; t < tile.steps; ++t) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t j = 0; j < 128; ++j) {
                q[(size_t(t) * heads + h) * 128 + j] = generated((size_t(t) * heads + h) * 128 + j, 23) * 0.03f;
                k[(size_t(t) * heads + h) * 128 + j] = generated((size_t(t) * heads + h) * 128 + j, 29) * 0.03f;
            }
            for (uint32_t j = 0; j < dv; ++j) {
                v[(size_t(t) * heads + h) * dv + j] = generated((size_t(t) * heads + h) * dv + j, 31) * 0.04f;
            }
            const size_t gi = size_t(t) * heads + h;
            if (extreme_gates && h == 0) {
                alpha[gi] = (t % 2 == 0) ? 700.0f : -700.0f;
                beta_logits[gi] = (t % 2 == 0) ? 80.0f : -80.0f;
            } else {
                alpha[gi] = generated(gi, 37) * 3.0f;
                beta_logits[gi] = generated(gi, 41) * 5.0f;
            }
        }
    }

    const auto ref = verify_reference(s0, q, k, v, alpha, beta_logits, rates, bias, heads, dv, tile.steps);

    const auto s0_v = a.array<float>(s0, true);
    const auto q_v = a.array<float>(q, true), k_v = a.array<float>(k, true), v_v = a.array<float>(v, true);
    const auto a_v = a.array<float>(alpha, true), b_v = a.array<float>(beta_logits, true);
    const auto r_v = a.array<float>(rates, true), dt_v = a.array<float>(bias, true);

    for (bool write_tape : {false, true}) {
        const auto s_out = a.floats(state_n, sentinel);
        const auto y_out = a.floats(val_n, sentinel);
        const auto td_out = a.floats(gate_n, sentinel);
        const auto tb_out = a.floats(gate_n, sentinel);
        const auto routes_before = g.gdn_ported_routes();
        g.begin();
        g.gdn_verify_main(s0_v, q_v, k_v, v_v, a_v, b_v, r_v, dt_v, s_out, y_out, td_out, tb_out, heads, dv, tile, write_tape);
        const auto res = g.submit();
        ++batches;
        require(res.dispatches == 1, "gdn_verify_main dispatch count differs");
        const size_t bucket = tile.capacity == 1 ? 1 : 2;
        auto expected_routes = routes_before;
        ++expected_routes[bucket];
        require(g.gdn_ported_routes() == expected_routes, "gdn_verify_main route counter delta differs");

        const auto got_s = read_vec<float>(g, s_out);
        const auto got_y = read_vec<float>(g, y_out);
        const auto got_td = read_vec<float>(g, td_out);
        const auto got_tb = read_vec<float>(g, tb_out);

        const double sn = compute_nmse(got_s, ref.state);
        const double yn = compute_nmse(std::span<const float>(got_y).first(size_t(tile.steps) * heads * dv), ref.output);
        worst_verify_nmse = std::max({worst_verify_nmse, sn, yn});
        require(sn <= nmse_bound && yn <= nmse_bound, "gdn_verify_main exceeded 1e-7 FP64 reference NMSE bound");

        for (size_t i = size_t(tile.steps) * heads * dv; i < val_n; ++i) {
            require(same_bits(got_y[i], sentinel), "gdn_verify_main wrote inactive output row");
        }
        if (write_tape) {
            const double tdn = compute_nmse(std::span<const float>(got_td).first(size_t(tile.steps) * heads), ref.decay);
            const double tbn = compute_nmse(std::span<const float>(got_tb).first(size_t(tile.steps) * heads), ref.beta);
            require(tdn <= nmse_bound && tbn <= nmse_bound, "gdn_verify_main tape exceeded 1e-7 NMSE bound");
            for (size_t i = size_t(tile.steps) * heads; i < gate_n; ++i) {
                require(same_bits(got_td[i], sentinel) && same_bits(got_tb[i], sentinel),
                    "gdn_verify_main wrote inactive tape row");
            }
        } else {
            for (float f : got_td) require(same_bits(f, sentinel), "gdn_verify_main wrote decay tape when disabled");
            for (float f : got_tb) require(same_bits(f, sentinel), "gdn_verify_main wrote beta tape when disabled");
        }

        for (uint32_t db : {8u, 16u}) {
            const auto s_db = a.floats(state_n, sentinel);
            const auto y_db = a.floats(val_n, sentinel);
            const auto td_db = a.floats(gate_n, sentinel);
            const auto tb_db = a.floats(gate_n, sentinel);
            g.begin();
            g.gdn_verify_main_db(s0_v, q_v, k_v, v_v, a_v, b_v, r_v, dt_v, s_db, y_db, td_db, tb_db, heads, dv, tile, write_tape, db);
            static_cast<void>(g.submit());
            ++batches;
            const auto got_s_db = read_vec<float>(g, s_db);
            const auto got_y_db = read_vec<float>(g, y_db);
            const double sn_db = compute_nmse(got_s_db, ref.state);
            const double yn_db = compute_nmse(std::span<const float>(got_y_db).first(size_t(tile.steps) * heads * dv), ref.output);
            worst_verify_nmse = std::max({worst_verify_nmse, sn_db, yn_db});
            require(sn_db <= nmse_bound && yn_db <= nmse_bound, "gdn_verify_main_db exceeded 1e-7 NMSE bound");
            if (db == TOKSPUR_GDN_VERIFY_DB) {
                require(g.read(s_db) == g.read(s_out) && g.read(y_db) == g.read(y_out),
                    "gdn_verify_main differs from selected DB variant");
            }
        }

        if (write_tape && tile.capacity > 1 && tile.steps >= 2) {
            for (uint32_t keep = 1; keep < tile.steps; ++keep) {
                const auto prefix_ref = verify_reference(s0, q, k, v, alpha, beta_logits, rates, bias, heads, dv, keep);
                const auto s_rep = a.floats(state_n, sentinel);
                const auto s_direct = a.floats(state_n, sentinel);
                const auto y_direct = a.floats(val_n, sentinel);
                const auto td_direct = a.floats(gate_n, sentinel);
                const auto tb_direct = a.floats(gate_n, sentinel);

                const auto r_before = g.gdn_ported_routes();
                g.begin();
                g.gdn_verify_replay(s0_v, k_v, v_v, td_out, tb_out, s_rep, heads, dv, {tile.capacity, keep});
                const auto rep_res = g.submit();
                ++batches;
                require(rep_res.dispatches == 1, "gdn_verify_replay dispatch count differs");
                auto exp_r = r_before;
                ++exp_r[3];
                require(g.gdn_ported_routes() == exp_r, "gdn_verify_replay route counter delta differs");

                g.begin();
                g.gdn_verify_main(s0_v, q_v, k_v, v_v, a_v, b_v, r_v, dt_v,
                    s_direct, y_direct, td_direct, tb_direct, heads, dv, {tile.capacity, keep}, false);
                static_cast<void>(g.submit());
                ++batches;

                const auto got_rep = read_vec<float>(g, s_rep);
                const double rn = compute_nmse(got_rep, prefix_ref.state);
                worst_replay_nmse = std::max(worst_replay_nmse, rn);
                require(rn <= nmse_bound, "gdn_verify_replay exceeded 1e-7 FP64 reference NMSE bound");
                require(g.read(s_rep) == g.read(s_direct),
                    "gdn_verify_replay state differs from direct gdn_verify_main prefix state");

                for (uint32_t db : {8u, 16u}) {
                    const auto s_rep_db = a.floats(state_n, sentinel);
                    g.begin();
                    g.gdn_verify_replay_db(s0_v, k_v, v_v, td_out, tb_out, s_rep_db, heads, dv, {tile.capacity, keep}, db);
                    static_cast<void>(g.submit());
                    ++batches;
                    const double rn_db = compute_nmse(read_vec<float>(g, s_rep_db), prefix_ref.state);
                    worst_replay_nmse = std::max(worst_replay_nmse, rn_db);
                    require(rn_db <= nmse_bound, "gdn_verify_replay_db exceeded 1e-7 NMSE bound");
                    if (db == TOKSPUR_GDN_VERIFY_DB) {
                        require(g.read(s_rep_db) == g.read(s_rep), "gdn_verify_replay differs from selected DB variant");
                    }
                }

                // Fused deferred main + replay check: replay keep steps, then run tile.steps main steps.
                const auto s_fused = a.floats(state_n, sentinel);
                const auto y_fused = a.floats(val_n, sentinel);
                const auto s_two = a.floats(state_n, sentinel);
                const auto y_two = a.floats(val_n, sentinel);
                const auto td_two = a.floats(gate_n, sentinel);
                const auto tb_two = a.floats(gate_n, sentinel);
                g.begin();
                g.gdn_verify_main_replay(s0_v, k_v, v_v, td_out, tb_out, q_v, k_v, v_v, a_v, b_v, r_v, dt_v,
                    s_fused, y_fused, heads, dv, {tile.capacity, keep}, tile);
                const auto fused_res = g.submit();
                ++batches;
                require(fused_res.dispatches == 1, "gdn_verify_main_replay dispatch count differs");

                g.begin();
                g.gdn_verify_main(s_rep, q_v, k_v, v_v, a_v, b_v, r_v, dt_v,
                    s_two, y_two, td_two, tb_two, heads, dv, tile, false);
                static_cast<void>(g.submit());
                ++batches;

                require(g.read(s_fused) == g.read(s_two) && g.read(y_fused) == g.read(y_two),
                    "gdn_verify_main_replay differs from separate gdn_verify_replay + gdn_verify_main");
            }
        }
    }
    a.guards();
}

void run_norm_gate_case(MetalGraph& g, uint32_t heads, uint32_t dv, StateTile tile) {
    Arena a{g, {}};
    const size_t val_n = size_t(tile.capacity) * heads * dv;
    std::vector<float> raw(val_n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> gate(val_n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> gains(dv);
    for (uint32_t j = 0; j < dv; ++j) gains[j] = 1.0f + generated(j, 53) * 0.15f;
    for (uint32_t t = 0; t < tile.steps; ++t) {
        for (size_t i = 0; i < size_t(heads) * dv; ++i) {
            raw[size_t(t) * heads * dv + i] = generated(size_t(t) * heads * dv + i, 59) * 0.25f;
            gate[size_t(t) * heads * dv + i] = generated(size_t(t) * heads * dv + i, 61) * 5.0f;
        }
    }
    std::vector<double> ref(size_t(tile.steps) * heads * dv);
    constexpr float eps = 1e-6f;
    for (uint32_t t = 0; t < tile.steps; ++t) {
        for (uint32_t h = 0; h < heads; ++h) {
            const size_t base = (size_t(t) * heads + h) * dv;
            double sq = 0.0;
            for (uint32_t j = 0; j < dv; ++j) {
                const double x = double(raw[base + j]);
                sq += x * x;
            }
            const double inv = 1.0 / std::sqrt(sq / double(dv) + double(eps));
            for (uint32_t j = 0; j < dv; ++j) {
                ref[base + j] = double(raw[base + j]) * inv * double(gains[j]) * stable_sigmoid(double(gate[base + j]));
            }
        }
    }

    const auto raw_v = a.array<float>(raw, true);
    const auto gains_v = a.array<float>(gains, true);
    const auto gate_v = a.array<float>(gate, true);
    const auto out_v = a.floats(val_n, sentinel);
    const auto rep_v = a.floats(val_n, sentinel);

    const auto routes_before = g.gdn_ported_routes();
    g.begin();
    g.gdn_norm_gate(raw_v, gains_v, gate_v, out_v, heads, dv, eps, tile);
    const auto res = g.submit();
    ++batches;
    require(res.dispatches == 1, "gdn_norm_gate dispatch count differs");
    auto exp_routes = routes_before;
    ++exp_routes[4];
    require(g.gdn_ported_routes() == exp_routes, "gdn_norm_gate route counter delta differs");

    g.begin();
    g.gdn_norm_gate(raw_v, gains_v, gate_v, rep_v, heads, dv, eps, tile);
    static_cast<void>(g.submit());
    ++batches;
    require(g.read(out_v) == g.read(rep_v), "gdn_norm_gate own-repeat bytes differ");

    const auto got = read_vec<float>(g, out_v);
    const double nmse = compute_nmse(std::span<const float>(got).first(ref.size()), ref);
    worst_norm_gate_nmse = std::max(worst_norm_gate_nmse, nmse);
    require(nmse <= nmse_bound, "gdn_norm_gate exceeded 1e-7 FP64 reference NMSE bound");
    for (size_t i = ref.size(); i < val_n; ++i) {
        require(same_bits(got[i], sentinel), "gdn_norm_gate wrote inactive tail row");
    }
    a.guards();
}

void test_composite_core_and_recurrent(const char* library) {
    MetalGraph g_ported(library, 256ull << 20);
    MetalGraph g_base(library, 256ull << 20);
    prepare_graph(g_ported, true, false);
    prepare_graph(g_base, false, false);

    const GdnShape shape{32, 16, 48, 128, 128, 4};
    const GdnPolicy policy{1.0f / std::sqrt(128.0f), 1e-6f, L2Mode::add_squared_epsilon, 1e-6f};
    std::deque<std::vector<std::byte>> encoded_p, encoded_b;
    std::vector<GpuView> owned_p, owned_b;
    struct Cleanup {
        MetalGraph& g;
        std::vector<GpuView>& v;
        ~Cleanup() {
            try { g.abort(); } catch (const FormatError&) {}
            for (auto x : v) g.release(x);
        }
    } cp{g_ported, owned_p}, cb{g_base, owned_b};

    const auto make_weights = [&](MetalGraph& g, std::deque<std::vector<std::byte>>& enc, std::vector<GpuView>& own) {
        const auto mat = [&](uint32_t cols, uint32_t rows, bool q8) -> GpuMatrix {
            if (!q8) {
                std::vector<float> v(uint64_t(cols) * rows);
                for (size_t i = 0; i < v.size(); ++i) v[i] = generated(i, 71) * 0.01f;
                const auto raw = std::as_bytes(std::span(v));
                enc.emplace_back(raw.begin(), raw.end());
            } else {
                enc.emplace_back(q8_many_bytes({cols, rows, 1}).matrix);
                for (size_t at = 0; at < enc.back().size(); at += 34) {
                    enc.back()[at] = std::byte{0};
                    enc.back()[at + 1] = std::byte{0x18};
                    for (size_t l = 0; l < 32; ++l) enc.back()[at + 2 + l] = std::byte(uint8_t(int((at / 34 * 13 + l * 7) % 31) - 15));
                }
            }
            const auto view = g.upload(enc.back(), "composite GDN weight");
            own.push_back(view);
            return {view, q8 ? 8u : 0u};
        };
        const auto vec = [&](uint64_t n, float scale, float offset = 0.0f) -> GpuView {
            std::vector<float> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = generated(i, 73) * scale + offset;
            const auto view = g.upload(std::as_bytes(std::span(v)), "composite GDN vec");
            own.push_back(view);
            return view;
        };
        const uint32_t raw = uint32_t(gdn_qkv_count(shape)), val = shape.value_heads * shape.value_width;
        return GdnWeights{
            mat(shape.channels, raw, true), mat(shape.channels, val, true),
            mat(shape.channels, shape.value_heads, false), mat(shape.channels, shape.value_heads, false),
            mat(val, shape.channels, true), vec(uint64_t(raw) * 4, 0.1f),
            vec(shape.value_heads, 0.0f, -0.25f), vec(shape.value_heads, 0.02f), vec(shape.value_width, 0.1f, 1.0f)};
    };

    const auto wp = make_weights(g_ported, encoded_p, owned_p);
    const auto wb = make_weights(g_base, encoded_b, owned_b);

    // 1. Single-token encode_gdn (11 dispatches vs 12 legacy dispatches).
    {
        std::vector<float> x1(shape.channels), s_init(gdn_state_count(shape)), h_init(gdn_qkv_count(shape) * 3);
        for (size_t i = 0; i < x1.size(); ++i) x1[i] = generated(i, 79) * 0.05f;
        for (size_t i = 0; i < s_init.size(); ++i) s_init[i] = generated(i, 83) * 0.01f;
        for (size_t i = 0; i < h_init.size(); ++i) h_init[i] = generated(i, 89) * 0.01f;

        const auto sp_cur = allocate_gdn_state(g_ported, shape), sp_nxt = allocate_gdn_state(g_ported, shape);
        const auto sb_cur = allocate_gdn_state(g_base, shape), sb_nxt = allocate_gdn_state(g_base, shape);
        const auto scp = allocate_gdn_scratch(g_ported, shape), scb = allocate_gdn_scratch(g_base, shape);
        for (auto v : sp_cur.views()) owned_p.push_back(v);
        for (auto v : sp_nxt.views()) owned_p.push_back(v);
        for (auto v : scp.views()) owned_p.push_back(v);
        for (auto v : sb_cur.views()) owned_b.push_back(v);
        for (auto v : sb_nxt.views()) owned_b.push_back(v);
        for (auto v : scb.views()) owned_b.push_back(v);

        g_ported.write(sp_cur.matrix, std::as_bytes(std::span(s_init)));
        g_ported.write(sp_cur.convolution, std::as_bytes(std::span(h_init)));
        g_base.write(sb_cur.matrix, std::as_bytes(std::span(s_init)));
        g_base.write(sb_cur.convolution, std::as_bytes(std::span(h_init)));

        const auto xp = g_ported.upload(std::as_bytes(std::span(x1)), "x1p");
        const auto yp = g_ported.zeros(shape.channels * 4, "y1p");
        const auto xb = g_base.upload(std::as_bytes(std::span(x1)), "x1b");
        const auto yb = g_base.zeros(shape.channels * 4, "y1b");
        owned_p.push_back(xp); owned_p.push_back(yp);
        owned_b.push_back(xb); owned_b.push_back(yb);

        g_ported.begin();
        encode_gdn(g_ported, wp, scp, sp_cur, sp_nxt, xp, yp, shape, policy);
        const auto rp = g_ported.submit();
        ++batches;

        g_base.begin();
        encode_gdn(g_base, wb, scb, sb_cur, sb_nxt, xb, yb, shape, policy);
        const auto rb = g_base.submit();
        ++batches;

        require(rp.dispatches + 2 == rb.dispatches,
            "single-token encode_gdn dispatch reduction differs from exact -2 formula");
        const auto got_yp = read_vec<float>(g_ported, yp);
        const auto got_yb = read_vec<float>(g_base, yb);
        std::vector<double> ref_yb(got_yb.begin(), got_yb.end());
        require(compute_nmse(got_yp, ref_yb) <= nmse_bound, "single-token encode_gdn output NMSE exceeded 1e-7");
        const auto got_sp = read_vec<float>(g_ported, sp_nxt.matrix);
        const auto got_sb = read_vec<float>(g_base, sb_nxt.matrix);
        std::vector<double> ref_sb(got_sb.begin(), got_sb.end());
        require(compute_nmse(got_sp, ref_sb) <= nmse_bound, "single-token encode_gdn state NMSE exceeded 1e-7");
    }

    // 2. Multi-row encode_gdn_core_tile + encode_gdn_core_tape_prefix (steps = 3 and 7 in capacity 8).
    for (uint32_t steps : {3u, 7u}) {
        constexpr uint32_t cap = 8;
        std::vector<float> x(cap * shape.channels, std::numeric_limits<float>::quiet_NaN());
        for (size_t i = 0; i < size_t(steps) * shape.channels; ++i) x[i] = generated(i, 97) * 0.04f;
        std::vector<float> s_init(gdn_state_count(shape)), h_init(gdn_qkv_count(shape) * 3);
        for (size_t i = 0; i < s_init.size(); ++i) s_init[i] = generated(i, 101) * 0.01f;
        for (size_t i = 0; i < h_init.size(); ++i) h_init[i] = generated(i, 103) * 0.01f;

        const auto sp_cur = allocate_gdn_state(g_ported, shape), sp_nxt = allocate_gdn_state(g_ported, shape);
        const auto sb_cur = allocate_gdn_state(g_base, shape), sb_nxt = allocate_gdn_state(g_base, shape);
        const auto scp = allocate_gdn_core_tile_scratch(g_ported, shape, cap);
        const auto scb = allocate_gdn_core_tile_scratch(g_base, shape, cap);
        const auto rpp = allocate_gdn_core_replay_scratch(g_ported, shape, cap);
        const auto rpb = allocate_gdn_core_replay_scratch(g_base, shape, cap);
        for (auto v : sp_cur.views()) owned_p.push_back(v);
        for (auto v : sp_nxt.views()) owned_p.push_back(v);
        for (auto v : scp.views()) owned_p.push_back(v);
        owned_p.push_back(rpp.convolution_output); owned_p.push_back(rpp.delta_output);
        for (auto v : sb_cur.views()) owned_b.push_back(v);
        for (auto v : sb_nxt.views()) owned_b.push_back(v);
        for (auto v : scb.views()) owned_b.push_back(v);
        owned_b.push_back(rpb.convolution_output); owned_b.push_back(rpb.delta_output);

        g_ported.write(sp_cur.matrix, std::as_bytes(std::span(s_init)));
        g_ported.write(sp_cur.convolution, std::as_bytes(std::span(h_init)));
        g_base.write(sb_cur.matrix, std::as_bytes(std::span(s_init)));
        g_base.write(sb_cur.convolution, std::as_bytes(std::span(h_init)));

        const auto xp = g_ported.upload(std::as_bytes(std::span(x)), "xtp");
        const auto yp = g_ported.zeros(cap * shape.channels * 4, "ytp");
        const auto xb = g_base.upload(std::as_bytes(std::span(x)), "xtb");
        const auto yb = g_base.zeros(cap * shape.channels * 4, "ytb");
        owned_p.push_back(xp); owned_p.push_back(yp);
        owned_b.push_back(xb); owned_b.push_back(yb);

        g_ported.begin();
        encode_gdn_core_tile(g_ported, wp, scp, sp_cur, sp_nxt, xp, yp, shape, policy, {cap, steps});
        const auto rp = g_ported.submit();
        ++batches;

        g_base.begin();
        encode_gdn_core_tile(g_base, wb, scb, sb_cur, sb_nxt, xb, yb, shape, policy, {cap, steps});
        const auto rb = g_base.submit();
        ++batches;

        require(rp.dispatches + (steps * 2) == rb.dispatches,
            "encode_gdn_core_tile dispatch reduction differs from exact 2*steps formula");
        const auto got_yp = read_vec<float>(g_ported, yp);
        const auto got_yb = read_vec<float>(g_base, yb);
        std::vector<double> ref_yb(got_yb.begin(), got_yb.begin() + size_t(steps) * shape.channels);
        require(compute_nmse(std::span<const float>(got_yp).first(size_t(steps) * shape.channels), ref_yb) <= nmse_bound,
            "encode_gdn_core_tile output NMSE exceeded 1e-7");

        const GdnCoreTape tp{scp.projected, scp.query, scp.key, scp.value, scp.decay, scp.write_gate};
        const GdnCoreTape tb{scb.projected, scb.query, scb.key, scb.value, scb.decay, scb.write_gate};
        for (uint32_t keep = 1; keep < steps; ++keep) {
            g_ported.begin();
            encode_gdn_core_tape_prefix(g_ported, wp.convolution, tp, sp_cur, sp_nxt, rpp, shape, {cap, keep});
            const auto rrp = g_ported.submit();
            ++batches;

            g_base.begin();
            encode_gdn_core_tape_prefix(g_base, wb.convolution, tb, sb_cur, sb_nxt, rpb, shape, {cap, keep});
            const auto rrb = g_base.submit();
            ++batches;

            require(rrp.dispatches == 2 && rrb.dispatches == 2, "encode_gdn_core_tape_prefix dispatch count differs");
            require(g_ported.read(sp_nxt.convolution) == g_base.read(sb_nxt.convolution),
                "encode_gdn_core_tape_prefix convolution history differs");
            const auto got_sp = read_vec<float>(g_ported, sp_nxt.matrix);
            const auto got_sb = read_vec<float>(g_base, sb_nxt.matrix);
            std::vector<double> ref_sb(got_sb.begin(), got_sb.end());
            require(compute_nmse(got_sp, ref_sb) <= nmse_bound, "encode_gdn_core_tape_prefix state NMSE exceeded 1e-7");
        }
    }
}

void test_refusals_and_rollback(const char* library) {
    {
        MetalGraph g(library, 64ull << 20);
        require(!g.gdn_ported_enabled() && !g.gdn_ported_rollback_enabled() &&
            g.gdn_ported_routes() == GdnPortedRoutes{}, "default ported GDN state changed");
        g.configure_gdn_ported(false);
        require(!g.gdn_ported_enabled() && g.gdn_ported_rollback_enabled(), "explicit off enabled ported GDN");
        reject([&] { g.configure_gdn_ported(true); });
    }
    {
        MetalGraph g(library, 128ull << 20);
        prepare_graph(g, true, false);
        require(g.gdn_ported_enabled() && g.gdn_ported_rollback_enabled(), "ported GDN failed to enable");
        reject([&] { g.configure_gdn_ported(false); });
        reject([&] { g.begin_indirect_compatible(); });
        reject([&] { g.begin_capture(4); });

        Arena a{g, {}};
        constexpr uint32_t heads = 2, dv = 16, cap = 4, steps = 2;
        const auto s0 = a.floats(heads * 128 * dv, 0.01f);
        const auto q = a.floats(cap * heads * 128, 0.01f);
        const auto k = a.floats(cap * heads * 128, 0.01f);
        const auto v = a.floats(cap * heads * dv, 0.01f);
        const auto alpha = a.floats(cap * heads, 0.1f);
        const auto beta = a.floats(cap * heads, 0.1f);
        const auto rates = a.floats(heads, -0.2f);
        const auto bias = a.floats(heads, 0.0f);
        const auto s1 = a.floats(heads * 128 * dv, sentinel);
        const auto y = a.floats(cap * heads * dv, sentinel);
        const auto td = a.floats(cap * heads, sentinel);
        const auto tb = a.floats(cap * heads, sentinel);
        const auto gains = a.floats(dv, 1.0f);

        // Host validation refusals (aliasing, tile bounds, invalid db).
        g.begin();
        reject([&] { g.gdn_verify_main(s0, q, k, v, alpha, beta, rates, bias, s0, y, td, tb, heads, dv, {cap, steps}, true); });
        try { g.abort(); } catch (const FormatError&) {}

        g.begin();
        reject([&] { g.gdn_verify_main(s0, q, k, v, alpha, beta, rates, bias, s1, y, td, tb, heads, dv, {1, 2}, false); });
        try { g.abort(); } catch (const FormatError&) {}

        g.begin();
        reject([&] { g.gdn_verify_main_db(s0, q, k, v, alpha, beta, rates, bias, s1, y, td, tb, heads, dv, {cap, steps}, false, 4); });
        try { g.abort(); } catch (const FormatError&) {}

        g.begin();
        reject([&] { g.gdn_norm_gate(v, gains, v, v, heads, dv, 1e-6f, {cap, steps}); });
        try { g.abort(); } catch (const FormatError&) {}

        require(g.gdn_ported_routes() == GdnPortedRoutes{}, "host refusal published ported GDN routes");

        // GPU non-finite / invalid rate error flag refusal + rollback + clean recovery.
        const auto s0_bytes = g.read(s0);
        auto bad_alpha = std::vector<float>(cap * heads, 0.1f);
        bad_alpha[0] = std::numeric_limits<float>::quiet_NaN();
        g.write(alpha, std::as_bytes(std::span(bad_alpha)));
        g.begin();
        g.gdn_verify_main(s0, q, k, v, alpha, beta, rates, bias, s1, y, td, tb, heads, dv, {cap, steps}, true);
        reject([&] { static_cast<void>(g.submit()); });
        try { g.abort(); } catch (const FormatError&) {}
        require(g.read(s0) == s0_bytes && g.gdn_ported_routes() == GdnPortedRoutes{},
            "GPU NaN failure mutated accepted state or published ported GDN routes");

        // Positive decay rate (> 0) must also fail via device error flag.
        g.write(alpha, std::as_bytes(std::span<const float>(std::vector<float>(cap * heads, 0.1f))));
        g.write(rates, std::as_bytes(std::span<const float>(std::vector<float>(heads, 0.25f))));
        g.begin();
        g.gdn_verify_main(s0, q, k, v, alpha, beta, rates, bias, s1, y, td, tb, heads, dv, {cap, steps}, true);
        reject([&] { static_cast<void>(g.submit()); });
        try { g.abort(); } catch (const FormatError&) {}
        require(g.gdn_ported_routes() == GdnPortedRoutes{}, "positive decay rate published ported GDN routes");

        // Clean recovery after GPU error.
        g.write(rates, std::as_bytes(std::span<const float>(std::vector<float>(heads, -0.2f))));
        g.begin();
        g.gdn_verify_main(s0, q, k, v, alpha, beta, rates, bias, s1, y, td, tb, heads, dv, {cap, steps}, true);
        require(g.submit().dispatches == 1 && g.gdn_ported_routes() == GdnPortedRoutes{0, 0, 1, 0, 0},
            "clean recovery after GPU error failed");
        ++batches;

        // Replay NaN failure + clean recovery.
        auto bad_k = std::vector<float>(cap * heads * 128, 0.01f);
        bad_k[7] = std::numeric_limits<float>::infinity();
        g.write(k, std::as_bytes(std::span(bad_k)));
        g.begin();
        g.gdn_verify_replay(s0, k, v, td, tb, s1, heads, dv, {cap, 1});
        reject([&] { static_cast<void>(g.submit()); });
        try { g.abort(); } catch (const FormatError&) {}
        require(g.gdn_ported_routes() == GdnPortedRoutes{0, 0, 1, 0, 0}, "failed replay published route counter");

        g.write(k, std::as_bytes(std::span<const float>(std::vector<float>(cap * heads * 128, 0.01f))));
        g.begin();
        g.gdn_verify_replay(s0, k, v, td, tb, s1, heads, dv, {cap, 1});
        require(g.submit().dispatches == 1 && g.gdn_ported_routes() == GdnPortedRoutes{0, 0, 1, 1, 0},
            "replay recovery failed");
        ++batches;
        a.guards();
    }
}

template <class Fn>
std::pair<double, double> median_wall_gpu_ms(Fn&& fn, int warmup = 4, int iters = 15) {
    for (int i = 0; i < warmup; ++i) static_cast<void>(fn());
    std::vector<double> wall, gpu;
    wall.reserve(size_t(iters));
    gpu.reserve(size_t(iters));
    for (int i = 0; i < iters; ++i) {
        const BatchResult r = fn();
        wall.push_back(r.wall_ms);
        gpu.push_back(r.gpu_ms);
    }
    std::sort(wall.begin(), wall.end());
    std::sort(gpu.begin(), gpu.end());
    return {wall[size_t(iters / 2)], gpu[size_t(iters / 2)]};
}

int screen_db(const char* library) {
    constexpr uint32_t heads = 48, dk = 128, dv = 128;
    constexpr int layers = 36;
    MetalGraph ported_g(library, 1536ull << 20);
    MetalGraph base_g(library, 1536ull << 20);
    prepare_graph(ported_g, true, false);
    prepare_graph(base_g, false, false);

    Arena ap{ported_g, {}}, ab{base_g, {}};
    const size_t state_n = size_t(heads) * dk * dv;
    const auto s0_p = ap.floats(state_n, 0.01f), s1_p = ap.floats(state_n, 0.0f);
    const auto s0_b = ab.floats(state_n, 0.01f), s1_b = ab.floats(state_n, 0.0f);

    // 1. Prefill T=512 x36 layers: gdn_pipelined_tb12 vs legacy graph_gdn_chunk_fused.
    {
        constexpr uint32_t rows = 512;
        const auto q_p = ap.floats(size_t(rows) * heads * dk, 0.01f), k_p = ap.floats(size_t(rows) * heads * dk, 0.01f);
        const auto v_p = ap.floats(size_t(rows) * heads * dv, 0.01f), y_p = ap.floats(size_t(rows) * heads * dv, 0.0f);
        const auto g_p = ap.floats(size_t(rows) * heads, -0.05f), b_p = ap.floats(size_t(rows) * heads, 0.5f);

        const auto q_b = ab.floats(size_t(rows) * heads * dk, 0.01f), k_b = ab.floats(size_t(rows) * heads * dk, 0.01f);
        const auto v_b = ab.floats(size_t(rows) * heads * dv, 0.01f), y_b = ab.floats(size_t(rows) * heads * dv, 0.0f);
        const auto g_b = ab.floats(size_t(rows) * heads, -0.05f), b_b = ab.floats(size_t(rows) * heads, 0.5f);

        const auto make_d = [&](GpuView s_in, GpuView q, GpuView k, GpuView v, GpuView g, GpuView b, GpuView s_out, GpuView y) {
            GdnChunkDispatch d{};
            d.op = GdnChunkOp::fused;
            d.params.rows = rows;
            d.params.heads = heads;
            d.params.key_width = dk;
            d.params.value_width = dv;
            d.params.input_stride = uint64_t(heads) * dk * 4;
            d.params.auxiliary_stride = d.params.output_stride = uint64_t(heads) * dv * 4;
            d.params.gate_stride = uint64_t(heads) * 4;
            d.buffers = {s_in, q, k, v, g, b, s_out, {}, y, {}};
            return d;
        };
        const auto dp = make_d(s0_p, q_p, k_p, v_p, g_p, b_p, s1_p, y_p);
        const auto db = make_d(s0_b, q_b, k_b, v_b, g_b, b_b, s1_b, y_b);

        const auto [base_wall, base_gpu] = median_wall_gpu_ms([&] {
            base_g.begin();
            for (int l = 0; l < layers; ++l) base_g.gdn_chunk(db);
            return base_g.submit();
        });
        const auto [port_wall, port_gpu] = median_wall_gpu_ms([&] {
            ported_g.begin();
            for (int l = 0; l < layers; ++l) ported_g.gdn_chunk(dp);
            return ported_g.submit();
        });
        std::cout << std::fixed << std::setprecision(4)
                  << "screen_gdn mode=prefill_t512_x36 legacy_gpu_ms=" << base_gpu
                  << " pipelined_tb12_gpu_ms=" << port_gpu << " speedup=" << (base_gpu / port_gpu)
                  << " legacy_wall_ms=" << base_wall << " pipelined_tb12_wall_ms=" << port_wall << '\n';
    }

    // 2. Decode (T=1), Verify (T=3, T=8) and Replay (keep=1, keep=2) x36 layers: db8 vs db16 vs legacy.
    const auto q8_p = ap.floats(8ull * heads * dk, 0.01f), k8_p = ap.floats(8ull * heads * dk, 0.01f);
    const auto v8_p = ap.floats(8ull * heads * dv, 0.01f), y8_p = ap.floats(8ull * heads * dv, 0.0f);
    const auto a8_p = ap.floats(8ull * heads, 0.1f), b8_p = ap.floats(8ull * heads, 0.1f);
    const auto r_p = ap.floats(heads, -0.2f), dt_p = ap.floats(heads, 0.01f);
    const auto td8_p = ap.floats(8ull * heads, 0.95f), tb8_p = ap.floats(8ull * heads, 0.5f);

    const auto q8_b = ab.floats(8ull * heads * dk, 0.01f), k8_b = ab.floats(8ull * heads * dk, 0.01f);
    const auto v8_b = ab.floats(8ull * heads * dv, 0.01f), y8_b = ab.floats(8ull * heads * dv, 0.0f);
    const auto a8_b = ab.floats(8ull * heads, 0.1f), b8_b = ab.floats(8ull * heads, 0.1f);
    const auto r_b = ab.floats(heads, -0.2f), dt_b = ab.floats(heads, 0.01f);
    const auto td8_b = ab.floats(8ull * heads, 0.95f), tb8_b = ab.floats(8ull * heads, 0.5f);

    // T=1 decode: gdn_gates + delta vs gdn_verify_main_db(8/16).
    {
        const auto q1_p = ported_g.slice(q8_p, 0, heads * dk * 4), k1_p = ported_g.slice(k8_p, 0, heads * dk * 4);
        const auto v1_p = ported_g.slice(v8_p, 0, heads * dv * 4), y1_p = ported_g.slice(y8_p, 0, heads * dv * 4);
        const auto a1_p = ported_g.slice(a8_p, 0, heads * 4), b1_p = ported_g.slice(b8_p, 0, heads * 4);
        const auto td1_p = ported_g.slice(td8_p, 0, heads * 4), tb1_p = ported_g.slice(tb8_p, 0, heads * 4);

        const auto q1_b = base_g.slice(q8_b, 0, heads * dk * 4), k1_b = base_g.slice(k8_b, 0, heads * dk * 4);
        const auto v1_b = base_g.slice(v8_b, 0, heads * dv * 4), y1_b = base_g.slice(y8_b, 0, heads * dv * 4);
        const auto a1_b = base_g.slice(a8_b, 0, heads * 4), b1_b = base_g.slice(b8_b, 0, heads * 4);
        const auto td1_b = base_g.slice(td8_b, 0, heads * 4), tb1_b = base_g.slice(tb8_b, 0, heads * 4);

        const auto [leg_w, leg_g] = median_wall_gpu_ms([&] {
            base_g.begin();
            for (int l = 0; l < layers; ++l) {
                base_g.gdn_gates(a1_b, b1_b, r_b, dt_b, td1_b, tb1_b);
                base_g.delta(s0_b, q1_b, k1_b, v1_b, td1_b, tb1_b, s1_b, y1_b, heads, dk, dv);
            }
            return base_g.submit();
        });
        const auto [db8_w, db8_g] = median_wall_gpu_ms([&] {
            ported_g.begin();
            for (int l = 0; l < layers; ++l) {
                ported_g.gdn_verify_main_db(s0_p, q1_p, k1_p, v1_p, a1_p, b1_p, r_p, dt_p, s1_p, y1_p, td1_p, tb1_p, heads, dv, {1, 1}, false, 8);
            }
            return ported_g.submit();
        });
        const auto [db16_w, db16_g] = median_wall_gpu_ms([&] {
            ported_g.begin();
            for (int l = 0; l < layers; ++l) {
                ported_g.gdn_verify_main_db(s0_p, q1_p, k1_p, v1_p, a1_p, b1_p, r_p, dt_p, s1_p, y1_p, td1_p, tb1_p, heads, dv, {1, 1}, false, 16);
            }
            return ported_g.submit();
        });
        static_cast<void>(leg_w); static_cast<void>(db8_w); static_cast<void>(db16_w);
        std::cout << "screen_gdn mode=decode_t1_x36 legacy_gpu_ms=" << leg_g
                  << " db8_gpu_ms=" << db8_g << " db8_speedup=" << (leg_g / db8_g)
                  << " db16_gpu_ms=" << db16_g << " db16_speedup=" << (leg_g / db16_g) << '\n';
    }

    for (uint32_t steps : {3u, 8u}) {
        const auto [leg_w, leg_g] = median_wall_gpu_ms([&] {
            base_g.begin();
            for (int l = 0; l < layers; ++l) {
                for (uint32_t t = 0; t < steps; ++t) {
                    base_g.gdn_gates(base_g.slice(a8_b, uint64_t(t) * heads * 4, heads * 4),
                        base_g.slice(b8_b, uint64_t(t) * heads * 4, heads * 4), r_b, dt_b,
                        base_g.slice(td8_b, uint64_t(t) * heads * 4, heads * 4),
                        base_g.slice(tb8_b, uint64_t(t) * heads * 4, heads * 4));
                }
                base_g.gdn_delta_tile(s0_b, q8_b, k8_b, v8_b, td8_b, tb8_b, s1_b, y8_b, heads, dv, {8, steps});
            }
            return base_g.submit();
        });
        const auto [db8_w, db8_g] = median_wall_gpu_ms([&] {
            ported_g.begin();
            for (int l = 0; l < layers; ++l) {
                ported_g.gdn_verify_main_db(s0_p, q8_p, k8_p, v8_p, a8_p, b8_p, r_p, dt_p, s1_p, y8_p, td8_p, tb8_p, heads, dv, {8, steps}, true, 8);
            }
            return ported_g.submit();
        });
        const auto [db16_w, db16_g] = median_wall_gpu_ms([&] {
            ported_g.begin();
            for (int l = 0; l < layers; ++l) {
                ported_g.gdn_verify_main_db(s0_p, q8_p, k8_p, v8_p, a8_p, b8_p, r_p, dt_p, s1_p, y8_p, td8_p, tb8_p, heads, dv, {8, steps}, true, 16);
            }
            return ported_g.submit();
        });
        static_cast<void>(leg_w); static_cast<void>(db8_w); static_cast<void>(db16_w);
        std::cout << "screen_gdn mode=verify_t" << steps << "_x36 legacy_gpu_ms=" << leg_g
                  << " db8_gpu_ms=" << db8_g << " db8_speedup=" << (leg_g / db8_g)
                  << " db16_gpu_ms=" << db16_g << " db16_speedup=" << (leg_g / db16_g) << '\n';
    }

    for (uint32_t keep : {1u, 2u}) {
        const auto [leg_w, leg_g] = median_wall_gpu_ms([&] {
            base_g.begin();
            for (int l = 0; l < layers; ++l) {
                base_g.gdn_delta_tile(s0_b, q8_b, k8_b, v8_b, td8_b, tb8_b, s1_b, y8_b, heads, dv, {8, keep});
            }
            return base_g.submit();
        });
        const auto [db8_w, db8_g] = median_wall_gpu_ms([&] {
            ported_g.begin();
            for (int l = 0; l < layers; ++l) {
                ported_g.gdn_verify_replay_db(s0_p, k8_p, v8_p, td8_p, tb8_p, s1_p, heads, dv, {8, keep}, 8);
            }
            return ported_g.submit();
        });
        const auto [db16_w, db16_g] = median_wall_gpu_ms([&] {
            ported_g.begin();
            for (int l = 0; l < layers; ++l) {
                ported_g.gdn_verify_replay_db(s0_p, k8_p, v8_p, td8_p, tb8_p, s1_p, heads, dv, {8, keep}, 16);
            }
            return ported_g.submit();
        });
        static_cast<void>(leg_w); static_cast<void>(db8_w); static_cast<void>(db16_w);
        std::cout << "screen_gdn mode=replay_keep" << keep << "_x36 legacy_gpu_ms=" << leg_g
                  << " db8_gpu_ms=" << db8_g << " db8_speedup=" << (leg_g / db8_g)
                  << " db16_gpu_ms=" << db16_g << " db16_speedup=" << (leg_g / db16_g) << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            throw std::runtime_error("usage: test-gdn_ported OWNED_METALLIB [screen-db]");
        }
        if (argc == 3) {
            require(std::string(argv[2]) == "screen-db", "unsupported test-gdn_ported mode");
            return screen_db(argv[1]);
        }

        test_refusals_and_rollback(argv[1]);

        MetalGraph ported_g(argv[1], 1536ull << 20);
        MetalGraph conc_g(argv[1], 1536ull << 20);
        MetalGraph base_g(argv[1], 1536ull << 20);
        prepare_graph(ported_g, true, false);
        prepare_graph(conc_g, true, true);
        prepare_graph(base_g, false, false);
        same_memory(ported_g.memory(), base_g.memory());

        // Prefill chunk recurrence: gdn_pipelined_tb12 (dv % 16 == 0) and gdn_blocked_seq_tb16 (dv % 16 != 0).
        for (uint32_t rows : {1u, 3u, 11u, 12u, 13u, 16u, 17u, 24u, 25u, 64u, 65u, 500u, 512u}) {
            run_prefill_case(ported_g, conc_g, base_g, rows, 3, 32, true);
            run_prefill_case(ported_g, conc_g, base_g, rows, 3, 20, true);
        }
        for (uint32_t rows : {1u, 12u, 13u, 64u, 512u}) {
            run_prefill_case(ported_g, conc_g, base_g, rows, 48, 128, false);
        }
        run_prefill_case(ported_g, conc_g, base_g, 65, 2, 16, true, true);
        run_prefill_case(ported_g, conc_g, base_g, 65, 2, 12, true, true);

        // Batched gdn_prepare_rows across policies and geometries.
        for (auto l2_mode : {L2Mode::add_squared_epsilon, L2Mode::floor_squared_epsilon}) {
            for (float scale : {1.0f, 1.0f / std::sqrt(128.0f)}) {
                const GdnPolicy pol{scale, 1e-6f, l2_mode, 1e-6f};
                run_prepare_case(ported_g, {32, 2, 4, 128, 20, 4}, pol, {1, 1});
                for (uint32_t steps = 1; steps <= 8; ++steps) {
                    run_prepare_case(ported_g, {32, 2, 4, 128, 20, 4}, pol, {8, steps});
                }
            }
        }
        for (uint32_t steps : {1u, 3u, 8u}) {
            run_prepare_case(ported_g, {2560, 16, 48, 128, 128, 4},
                {1.0f / std::sqrt(128.0f), 1e-6f, L2Mode::add_squared_epsilon, 1e-6f}, {8, steps});
        }

        // gdn_verify_main, gdn_verify_replay and gdn_verify_main_replay across DB=8/16, column tails, and steps 1..8.
        for (uint32_t dv : {12u, 20u, 32u, 128u}) {
            run_verify_and_replay_case(ported_g, 4, dv, {1, 1});
            for (uint32_t steps : {1u, 2u, 3u, 5u, 8u}) {
                run_verify_and_replay_case(ported_g, 4, dv, {8, steps});
            }
        }
        run_verify_and_replay_case(ported_g, 48, 128, {1, 1});
        run_verify_and_replay_case(ported_g, 48, 128, {8, 3});
        run_verify_and_replay_case(ported_g, 48, 128, {8, 8});
        run_verify_and_replay_case(ported_g, 4, 32, {8, 4}, true);

        // gdn_norm_gate across geometries and steps.
        for (uint32_t dv : {12u, 32u, 128u}) {
            run_norm_gate_case(ported_g, 4, dv, {1, 1});
            for (uint32_t steps = 1; steps <= 8; ++steps) {
                run_norm_gate_case(ported_g, 4, dv, {8, steps});
            }
        }
        run_norm_gate_case(ported_g, 48, 128, {1, 1});
        run_norm_gate_case(ported_g, 48, 128, {8, 3});
        run_norm_gate_case(ported_g, 48, 128, {8, 8});

        test_composite_core_and_recurrent(argv[1]);

        std::cout << "PASS ported GDN component Step 1 checks=" << checks << " batches=" << batches
                  << " verify_db=" << TOKSPUR_GDN_VERIFY_DB
                  << " worst_prefill_nmse=" << std::scientific << worst_prefill_nmse
                  << " worst_prepare_nmse=" << worst_prepare_nmse
                  << " worst_verify_nmse=" << worst_verify_nmse
                  << " worst_replay_nmse=" << worst_replay_nmse
                  << " worst_norm_gate_nmse=" << worst_norm_gate_nmse
                  << " bound=1e-7\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
