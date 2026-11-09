#include "tokspur/graph.hpp"
#include "tokspur/mtp_wire.hpp"
#include "tokspur/qsa_verify_rows.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace tokspur;

constexpr QsaShape actual{2560, 24, 2, 256, 4, 128, 64, 4, 2048, 65536};
constexpr QsaShape small{32, 3, 1, 34, 2, 32, 16, 4, 16, 8192};
constexpr float sentinel = 321.25f;
constexpr double nmse_bound = 1e-7;

size_t checks = 0, batches = 0;
double worst_decode_nmse = 0.0, worst_verify_nmse = 0.0;

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
    throw std::runtime_error("invalid sparse QSA operation admitted");
}

bool same_bits(float a, float b) {
    return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b);
}

void same_memory(GraphMemory a, GraphMemory b) {
    require(a.live_bytes == b.live_bytes && a.peak_bytes == b.peak_bytes && a.allocations == b.allocations &&
        a.imported_backing_bytes == b.imported_backing_bytes &&
        a.profile_counter_payload_bytes == b.profile_counter_payload_bytes &&
        a.capture_resource_bytes == b.capture_resource_bytes && a.live_captures == b.live_captures,
        "sparse QSA changed the prepared allocation ledger");
}

template <class T>
std::vector<T> read_vec(MetalGraph& graph, GpuView view) {
    const auto bytes = graph.read(view);
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

double compute_nmse(std::span<const float> got, std::span<const double> ref) {
    require(got.size() == ref.size() && !got.empty(), "NMSE span mismatch");
    double err = 0.0, power = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]) && std::isfinite(ref[i]), "non-finite QSA output or reference");
        const double d = double(got[i]) - ref[i];
        err += d * d;
        power += ref[i] * ref[i];
    }
    if (power == 0.0) return err == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    return err / power;
}

void prepare_base_policy(MetalGraph& g, bool sparse_enabled, bool concurrent = false) {
    g.configure_small_gemv(16);
    g.configure_block_gemv(3);
    g.configure_hc_exact(HcExactMode::combined);
    g.configure_concurrent_dispatch(concurrent);
    g.configure_qsa_stage_v8(true);
    g.configure_qsa_channel_chronological(true);
    g.configure_qsa_sparse(sparse_enabled);
    g.prepare_qsa_verify_rows();
    g.prepare_qsa_sparse_variants();
}

struct Arena {
    MetalGraph& g;
    std::vector<GpuView> full;
    GpuView bytes(std::span<const std::byte> data, bool immutable = false) {
        std::vector<std::byte> padded(data.size() + 8, std::byte{0x69});
        std::copy(data.begin(), data.end(), padded.begin() + 4);
        const auto allocation = g.upload(padded, "generated guarded sparse QSA fixture", immutable);
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
                    "sparse QSA outer canary changed");
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

std::vector<uint32_t> canonical_selected_ids(uint32_t prefix, QsaShape s, uint32_t row_seed) {
    const uint32_t blocks = prefix / s.compression;
    const uint32_t budget = qsa_block_budget(s);
    std::vector<double> scores(blocks);
    for (uint32_t b = 0; b < blocks; ++b) {
        scores[b] = double((uint64_t(b) * 97ull + uint64_t(row_seed) * 31ull + 13ull) % 53ull);
    }
    return select_blocks_reference(scores, budget);
}

void fill_pattern(std::vector<float>& queries, std::vector<uint16_t>& keys, std::vector<uint16_t>& values,
    QsaShape s, uint32_t rows, uint32_t valid_tokens, uint32_t pattern) {
    const uint32_t q_width = s.query_heads * s.head_width;
    const uint32_t kv_width = s.kv_heads * s.head_width;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t d = 0; d < q_width; ++d) {
            float x = float(int((d * 19u + r * 23u + 7u) % 101u) - 50) / 512.0f;
            if (pattern == 1) x = 0.0f;
            if (pattern == 2 || pattern == 3) x = 1.0f;
            if (pattern == 4) x = ((d / 32u) % 2u) ? -1.0f : 1.0f;
            if (pattern == 5) x = (d % 2u) ? -0.0f : std::numeric_limits<float>::denorm_min();
            queries[uint64_t(r) * q_width + d] = x;
        }
    }
    for (uint32_t token = 0; token < valid_tokens; ++token) {
        for (uint32_t d = 0; d < kv_width; ++d) {
            const uint64_t at = uint64_t(token) * kv_width + d;
            float k = float(int((at * 13ull + 11ull) % 127ull) - 63) / 256.0f;
            if (pattern == 1) k = (d % 2u) ? -0.0f : 0.0f;
            if (pattern == 2) k = float(token % 8192u) / 4096.0f;
            if (pattern == 3) k = -float(token % 8192u) / 4096.0f;
            if (pattern == 4) k = (token % 2u) ? 128.0f : -128.0f;
            keys[at] = finite_bf16(k);
            values[at] = finite_bf16((at % 37ull) ? float(int((at * 23ull + 3ull) % 257ull) - 128) / 512.0f : -0.0f);
        }
    }
}

void run_decode_case(MetalGraph& sparse_g, MetalGraph& base_g, QsaShape s, uint32_t prefix, uint32_t pattern = 0) {
    Arena a_sparse{sparse_g, {}}, a_base{base_g, {}};
    const uint32_t q_width = s.query_heads * s.head_width;
    const uint32_t kv_width = s.kv_heads * s.head_width;
    const uint32_t budget = qsa_block_budget(s);
    const uint32_t blocks = prefix / s.compression;
    const uint32_t cache_rows = std::min(s.capacity, prefix + 3u);

    std::vector<float> q_host(q_width);
    std::vector<uint16_t> k_host(uint64_t(cache_rows) * kv_width, uint16_t(0x7fc1));
    std::vector<uint16_t> v_host = k_host;
    fill_pattern(q_host, k_host, v_host, s, 1, prefix, pattern);

    const auto sel = canonical_selected_ids(prefix, s, 0);
    std::vector<uint32_t> id_host(budget, 0xffffffffu);
    if (blocks > budget) {
        std::copy(sel.begin(), sel.end(), id_host.begin());
    }

    std::vector<double> q_f64(q_host.begin(), q_host.end());
    const auto ref = sparse_attention_reference(
        q_f64,
        std::span<const uint16_t>(k_host).first(uint64_t(prefix) * kv_width),
        std::span<const uint16_t>(v_host).first(uint64_t(prefix) * kv_width),
        sel,
        prefix,
        s);

    const auto score_bytes = qsa_stage_v8_bytes(prefix, s).scores;
    const auto q_s = a_sparse.array<float>(q_host, true);
    const auto k_s = a_sparse.array<uint16_t>(k_host, true);
    const auto v_s = a_sparse.array<uint16_t>(v_host, true);
    const auto id_s = a_sparse.array<uint32_t>(id_host, true);
    const auto out_s = a_sparse.floats(q_width, sentinel);
    const auto rep_s = a_sparse.floats(q_width, sentinel);
    const auto scratch_s = a_sparse.floats(score_bytes / 4, sentinel);

    const auto fixed_s = sparse_g.memory();
    const uint64_t before_calls = sparse_g.qsa_sparse_decode_calls();
    sparse_g.begin();
    sparse_g.sparse_attention_stage_v8(q_s, k_s, v_s, id_s, out_s, scratch_s, prefix, s);
    const auto reply = sparse_g.submit();
    ++batches;
    require(reply.dispatches == 1 && !reply.capture_executes, "sparse QSA decode physical dispatch count differs");
    require(sparse_g.qsa_sparse_decode_calls() == before_calls + 1 &&
        sparse_g.qsa_stage_v8_calls() == 0 && sparse_g.qsa_channel_calls() == 0,
        "sparse QSA decode route counters differ");
    same_memory(fixed_s, sparse_g.memory());

    const auto scratch_vals = read_vec<float>(sparse_g, scratch_s);
    for (float v : scratch_vals) require(same_bits(v, sentinel), "sparse QSA decode touched score scratch");

    sparse_g.begin();
    sparse_g.sparse_attention(q_s, k_s, v_s, id_s, rep_s, prefix, s);
    const auto rep_reply = sparse_g.submit();
    ++batches;
    require(rep_reply.dispatches == 1 && sparse_g.qsa_sparse_decode_calls() == before_calls + 2,
        "sparse_attention routed decode dispatch count differs");
    const auto got = read_vec<float>(sparse_g, out_s);
    const auto rep = read_vec<float>(sparse_g, rep_s);
    require(sparse_g.read(out_s) == sparse_g.read(rep_s), "sparse QSA decode own-repeat bytes differ");

    const double nmse = compute_nmse(got, ref);
    worst_decode_nmse = std::max(worst_decode_nmse, nmse);
    require(nmse <= nmse_bound, "sparse QSA decode exceeded 1e-7 FP64 reference NMSE bound");

    for (uint32_t bn : {8u, 16u, 32u}) {
        const auto out_bn = a_sparse.floats(q_width, sentinel);
        sparse_g.begin();
        sparse_g.sparse_attention_bn(q_s, k_s, v_s, id_s, out_bn, prefix, s, bn);
        const auto bn_reply = sparse_g.submit();
        ++batches;
        require(bn_reply.dispatches == 1, "sparse QSA BN variant dispatch count differs");
        const auto got_bn = read_vec<float>(sparse_g, out_bn);
        const double bn_nmse = compute_nmse(got_bn, ref);
        worst_decode_nmse = std::max(worst_decode_nmse, bn_nmse);
        require(bn_nmse <= nmse_bound, "sparse QSA BN variant exceeded 1e-7 FP64 reference NMSE bound");
        if (bn == TOKSPUR_QSA_SPARSE_DECODE_BN) {
            require(sparse_g.read(out_bn) == sparse_g.read(out_s),
                "routed sparse QSA decode differs from selected BN variant");
        }
    }

    const auto q_b = a_base.array<float>(q_host, true);
    const auto k_b = a_base.array<uint16_t>(k_host, true);
    const auto v_b = a_base.array<uint16_t>(v_host, true);
    const auto id_b = a_base.array<uint32_t>(id_host, true);
    const auto out_b = a_base.floats(q_width, sentinel);
    base_g.begin();
    base_g.sparse_attention(q_b, k_b, v_b, id_b, out_b, prefix, s);
    const auto base_reply = base_g.submit();
    ++batches;
    require(base_reply.dispatches == 3, "legacy chronological QSA decode dispatch count differs");
    const auto base_got = read_vec<float>(base_g, out_b);
    std::vector<double> base_f64(base_got.begin(), base_got.end());
    require(compute_nmse(got, base_f64) <= nmse_bound, "sparse QSA decode exceeded 1e-7 NMSE vs 3-pass GPU baseline");

    a_sparse.guards();
    a_base.guards();
}

void run_verify_case(MetalGraph& sparse_g, MetalGraph& conc_g, MetalGraph& base_g,
    QsaShape s, uint32_t base, uint32_t rows, uint32_t pattern = 0) {
    Arena a_sparse{sparse_g, {}}, a_conc{conc_g, {}}, a_base{base_g, {}};
    const uint32_t q_width = s.query_heads * s.head_width;
    const uint32_t kv_width = s.kv_heads * s.head_width;
    const uint32_t budget = qsa_block_budget(s);
    const uint32_t cache_rows = std::min(s.capacity, base + rows + 4u);

    std::vector<float> q_host(8ull * q_width, std::numeric_limits<float>::quiet_NaN());
    std::vector<uint16_t> k_host(uint64_t(cache_rows) * kv_width, uint16_t(0x7fc1));
    std::vector<uint16_t> v_host = k_host;
    fill_pattern(q_host, k_host, v_host, s, rows, base + rows, pattern);

    std::vector<uint32_t> id_host(8ull * budget, 0xffffffffu);
    std::vector<std::vector<double>> refs(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t causal = base + r + 1u;
        const auto sel = canonical_selected_ids(causal, s, r);
        if (causal / s.compression > budget) {
            std::copy(sel.begin(), sel.end(), id_host.begin() + uint64_t(r) * budget);
        }
        std::vector<double> q_f64(q_host.begin() + uint64_t(r) * q_width, q_host.begin() + uint64_t(r + 1) * q_width);
        refs[r] = sparse_attention_reference(
            q_f64,
            std::span<const uint16_t>(k_host).first(uint64_t(causal) * kv_width),
            std::span<const uint16_t>(v_host).first(uint64_t(causal) * kv_width),
            sel,
            causal,
            s);
    }

    auto make_factor_view = [](Arena& arena, MetalGraph& g) {
        const auto storage = arena.floats(maximum_qsa_verify_rows_factors / 4 + 2, sentinel);
        return std::pair{storage, g.slice(storage, 4, maximum_qsa_verify_rows_factors)};
    };

    const auto q_s = a_sparse.array<float>(q_host, true);
    const auto k_s = a_sparse.array<uint16_t>(k_host, true);
    const auto v_s = a_sparse.array<uint16_t>(v_host, true);
    const auto id_s = a_sparse.array<uint32_t>(id_host, true);
    const auto out_s = a_sparse.floats(8ull * q_width, sentinel);
    const auto rep_s = a_sparse.floats(8ull * q_width, sentinel);
    const auto scores_s = a_sparse.floats(maximum_qsa_verify_rows_scores / 4, sentinel);
    const auto [fstore_s, factors_s] = make_factor_view(a_sparse, sparse_g);

    const auto fixed_s = sparse_g.memory();
    const auto routes_before = sparse_g.qsa_verify_rows_routes();
    const uint64_t verify_before = sparse_g.qsa_sparse_verify_calls();
    sparse_g.begin();
    sparse_g.sparse_attention_rows_chronological(q_s, k_s, v_s, id_s, out_s, scores_s, factors_s, base, rows, s);
    const auto reply = sparse_g.submit();
    ++batches;
    require(reply.dispatches == 1 && !reply.capture_executes, "sparse QSA verify physical dispatch count differs");
    const auto routes_after = sparse_g.qsa_verify_rows_routes();
    require(routes_after == QsaVerifyRowsRoutes{routes_before[0] + 1, routes_before[1] + rows, 0, 0, 0} &&
        sparse_g.qsa_sparse_verify_calls() == verify_before + 1 &&
        sparse_g.qsa_stage_v8_calls() == 0 && sparse_g.qsa_channel_calls() == 0,
        "sparse QSA verify route counters differ");
    same_memory(fixed_s, sparse_g.memory());

    for (float v : read_vec<float>(sparse_g, scores_s)) require(same_bits(v, sentinel), "sparse QSA verify touched score scratch");
    for (float v : read_vec<float>(sparse_g, fstore_s)) require(same_bits(v, sentinel), "sparse QSA verify touched factor scratch");

    sparse_g.begin();
    sparse_g.sparse_attention_rows_chronological(q_s, k_s, v_s, id_s, rep_s, scores_s, factors_s, base, rows, s);
    static_cast<void>(sparse_g.submit());
    ++batches;
    require(sparse_g.read(out_s) == sparse_g.read(rep_s), "sparse QSA verify own-repeat bytes differ");

    const auto q_c = a_conc.array<float>(q_host, true);
    const auto k_c = a_conc.array<uint16_t>(k_host, true);
    const auto v_c = a_conc.array<uint16_t>(v_host, true);
    const auto id_c = a_conc.array<uint32_t>(id_host, true);
    const auto out_c = a_conc.floats(8ull * q_width, sentinel);
    const auto scores_c = a_conc.floats(maximum_qsa_verify_rows_scores / 4, sentinel);
    const auto [fstore_c, factors_c] = make_factor_view(a_conc, conc_g);
    static_cast<void>(fstore_c);
    conc_g.begin();
    conc_g.sparse_attention_rows_chronological(q_c, k_c, v_c, id_c, out_c, scores_c, factors_c, base, rows, s);
    static_cast<void>(conc_g.submit());
    ++batches;
    require(conc_g.read(out_c) == sparse_g.read(out_s), "sparse QSA verify concurrent encoder bytes differ");

    const auto got = read_vec<float>(sparse_g, out_s);
    for (uint32_t r = 0; r < 8; ++r) {
        const auto span = std::span<const float>(got).subspan(uint64_t(r) * q_width, q_width);
        if (r < rows) {
            const double nmse = compute_nmse(span, refs[r]);
            worst_verify_nmse = std::max(worst_verify_nmse, nmse);
            require(nmse <= nmse_bound, "sparse QSA verify row exceeded 1e-7 FP64 reference NMSE bound");
        } else {
            for (float v : span) require(same_bits(v, sentinel), "sparse QSA verify wrote inactive row");
        }
    }

    for (uint32_t keep = 1; keep <= rows; ++keep) {
        std::vector<double> cat_ref;
        cat_ref.reserve(uint64_t(keep) * q_width);
        for (uint32_t r = 0; r < keep; ++r) cat_ref.insert(cat_ref.end(), refs[r].begin(), refs[r].end());
        const double frontier_nmse = compute_nmse(std::span<const float>(got).first(uint64_t(keep) * q_width), cat_ref);
        require(frontier_nmse <= nmse_bound, "sparse QSA partial-commit frontier exceeded 1e-7 NMSE bound");
    }

    for (uint32_t bn : {8u, 16u, 32u}) {
        const auto out_bn = a_sparse.floats(8ull * q_width, sentinel);
        sparse_g.begin();
        sparse_g.sparse_attention_rows_bn(q_s, k_s, v_s, id_s, out_bn, base, rows, s, bn);
        const auto bn_reply = sparse_g.submit();
        ++batches;
        require(bn_reply.dispatches == 1, "sparse QSA verify BN variant dispatch count differs");
        const auto got_bn = read_vec<float>(sparse_g, out_bn);
        for (uint32_t r = 0; r < 8; ++r) {
            const auto span = std::span<const float>(got_bn).subspan(uint64_t(r) * q_width, q_width);
            if (r < rows) {
                const double nmse = compute_nmse(span, refs[r]);
                worst_verify_nmse = std::max(worst_verify_nmse, nmse);
                require(nmse <= nmse_bound, "sparse QSA verify BN variant exceeded 1e-7 NMSE bound");
            } else {
                for (float v : span) require(same_bits(v, sentinel), "sparse QSA verify BN variant wrote inactive row");
            }
        }
        if (bn == TOKSPUR_QSA_SPARSE_VERIFY_BN) {
            require(sparse_g.read(out_bn) == sparse_g.read(out_s),
                "routed sparse QSA verify differs from selected BN variant");
        }
    }

    const auto q_b = a_base.array<float>(q_host, true);
    const auto k_b = a_base.array<uint16_t>(k_host, true);
    const auto v_b = a_base.array<uint16_t>(v_host, true);
    const auto id_b = a_base.array<uint32_t>(id_host, true);
    const auto out_b = a_base.floats(8ull * q_width, sentinel);
    const auto scores_b = a_base.floats(maximum_qsa_verify_rows_scores / 4, sentinel);
    const auto [fstore_b, factors_b] = make_factor_view(a_base, base_g);
    static_cast<void>(fstore_b);
    base_g.begin();
    base_g.sparse_attention_rows_chronological(q_b, k_b, v_b, id_b, out_b, scores_b, factors_b, base, rows, s);
    const auto base_reply = base_g.submit();
    ++batches;
    require(base_reply.dispatches == 3, "legacy 3-pass QSA verify dispatch count differs");
    const auto base_got = read_vec<float>(base_g, out_b);
    for (uint32_t r = 0; r < rows; ++r) {
        std::vector<double> base_f64(base_got.begin() + uint64_t(r) * q_width, base_got.begin() + uint64_t(r + 1) * q_width);
        require(compute_nmse(std::span<const float>(got).subspan(uint64_t(r) * q_width, q_width), base_f64) <= nmse_bound,
            "sparse QSA verify row exceeded 1e-7 NMSE vs 3-pass GPU baseline");
    }

    a_sparse.guards();
    a_conc.guards();
    a_base.guards();
}

void test_refusals_and_rollback(const char* library) {
    {
        MetalGraph g(library, 64ull << 20);
        require(!g.qsa_sparse_enabled() && !g.qsa_sparse_rollback_enabled() &&
            g.qsa_sparse_decode_calls() == 0 && g.qsa_sparse_verify_calls() == 0,
            "default sparse QSA state changed");
        reject([&] { g.configure_qsa_sparse(true); });
        g.configure_small_gemv(16);
        g.configure_block_gemv(3);
        g.configure_hc_exact(HcExactMode::combined);
        g.configure_qsa_stage_v8(true);
        reject([&] { g.configure_qsa_sparse(true); });
        g.configure_qsa_channel_chronological(true);
        g.configure_qsa_sparse(false);
        require(!g.qsa_sparse_enabled() && g.qsa_sparse_rollback_enabled(), "explicit off enabled sparse QSA or missed rollback");
        reject([&] { g.configure_qsa_sparse(true); });
        reject([&] { g.begin_indirect_compatible(); });
        reject([&] { g.begin_capture(4); });
    }
    {
        MetalGraph g(library, 64ull << 20);
        g.configure_small_gemv(16);
        g.configure_block_gemv(3);
        g.configure_hc_exact(HcExactMode::combined);
        g.configure_qsa_stage_v8(true);
        g.configure_qsa_channel_chronological(true);
        g.begin();
        g.abort();
        reject([&] { g.configure_qsa_sparse(false); });
        reject([&] { g.configure_qsa_sparse(true); });
    }

    MetalGraph g(library, 512ull << 20);
    prepare_base_policy(g, true);
    Arena arena{g, {}};
    const uint32_t prefix = 25u;
    const uint32_t q_width = small.query_heads * small.head_width;
    const uint32_t kv_width = small.kv_heads * small.head_width;
    const uint32_t budget = qsa_block_budget(small);
    const uint32_t blocks = prefix / small.compression;
    require(blocks > budget, "device refusal fixture must exceed block budget");

    std::vector<float> q_host(8ull * q_width, 0.125f);
    std::vector<uint16_t> k_host(uint64_t(prefix + 8u) * kv_width, finite_bf16(0.25f));
    std::vector<uint16_t> v_host(uint64_t(prefix + 8u) * kv_width, finite_bf16(-0.125f));
    std::vector<uint32_t> id_host(8ull * budget, 0u);
    for (uint32_t r = 0; r < 8; ++r) {
        for (uint32_t b = 0; b < budget; ++b) id_host[uint64_t(r) * budget + b] = b;
    }

    const auto q = arena.array<float>(std::span<const float>(q_host).first(q_width));
    const auto q8 = arena.array<float>(q_host);
    const auto k = arena.array<uint16_t>(k_host);
    const auto v = arena.array<uint16_t>(v_host);
    const auto id = arena.array<uint32_t>(std::span<const uint32_t>(id_host).first(budget));
    const auto id8 = arena.array<uint32_t>(id_host);
    const auto out = arena.floats(q_width, sentinel);
    const auto out8 = arena.floats(8ull * q_width, sentinel);
    const auto scores8 = arena.floats(maximum_qsa_verify_rows_scores / 4, sentinel);
    const auto fstore8 = arena.floats(maximum_qsa_verify_rows_factors / 4 + 2, sentinel);
    const auto factors8 = g.slice(fstore8, 4, maximum_qsa_verify_rows_factors);

    // Aborted batch must not publish decode or verify route counters.
    g.begin();
    g.sparse_attention(q, k, v, id, out, prefix, small);
    g.sparse_attention_rows_chronological(q8, k, v, id8, out8, scores8, factors8, prefix, 3, small);
    g.abort();
    require(g.qsa_sparse_decode_calls() == 0 && g.qsa_sparse_verify_calls() == 0 &&
        g.qsa_verify_rows_routes() == QsaVerifyRowsRoutes{},
        "aborted sparse QSA batch published route counters");

    // Host BN variant validation.
    for (uint32_t bad_bn : {0u, 4u, 12u, 64u}) {
        g.begin();
        reject([&] { g.sparse_attention_bn(q, k, v, id, out, prefix, small, bad_bn); });
        reject([&] { g.sparse_attention_rows_bn(q8, k, v, id8, out8, prefix, 3, small, bad_bn); });
        g.abort();
    }

    const auto expect_span_sentinel = [&](GpuView view, size_t offset_floats, size_t count_floats) {
        const auto all = read_vec<float>(g, view);
        for (size_t i = 0; i < count_floats; ++i) {
            require(same_bits(all[offset_floats + i], sentinel), "refused sparse QSA wrote poisoned or inactive slice");
        }
        const std::vector<float> reset(all.size(), sentinel);
        g.write(view, std::as_bytes(std::span(reset)));
    };

    // Device refusal: non-finite query on last head (1u).
    {
        auto bad_q_host = std::vector<float>(q_width, 0.125f);
        bad_q_host[q_width - 1] = std::numeric_limits<float>::quiet_NaN();
        const auto bad_q = arena.array<float>(bad_q_host);
        g.begin();
        g.sparse_attention(bad_q, k, v, id, out, prefix, small);
        reject([&] { static_cast<void>(g.submit()); });
        expect_span_sentinel(out, size_t(small.query_heads - 1) * small.head_width, small.head_width);
        require(g.qsa_sparse_decode_calls() == 0, "failed sparse QSA query NaN published counter");
    }

    // Device refusal: non-finite BF16 key and value across single KV head (1u).
    for (bool corrupt_key : {true, false}) {
        auto bad_kv = corrupt_key ? k_host : v_host;
        bad_kv[kv_width - 1] = 0x7fc0u;
        const auto bad_view = arena.array<uint16_t>(bad_kv);
        g.begin();
        g.sparse_attention(q, corrupt_key ? bad_view : k, corrupt_key ? v : bad_view, id, out, prefix, small);
        reject([&] { static_cast<void>(g.submit()); });
        expect_span_sentinel(out, 0, q_width);
        require(g.qsa_sparse_decode_calls() == 0, "failed sparse QSA BF16 NaN published counter");
    }

    // Device refusal: malformed selected_ids (exact boundary OOB, duplicate, non-increasing) on decode and verify rows (4u).
    for (size_t case_idx = 0; case_idx < 3; ++case_idx) {
        const std::array<uint32_t, 4> bad_decode_ids =
            case_idx == 0 ? std::array<uint32_t, 4>{0u, 1u, 2u, blocks} :
            case_idx == 1 ? std::array<uint32_t, 4>{0u, 2u, 2u, 3u} :
                            std::array<uint32_t, 4>{0u, 3u, 2u, 4u};
        const uint32_t row2_blocks = (prefix + 2u + 1u) / small.compression;
        const std::array<uint32_t, 4> bad_row2_ids =
            case_idx == 0 ? std::array<uint32_t, 4>{0u, 1u, 2u, row2_blocks} : bad_decode_ids;
        const auto bad_id = arena.array<uint32_t>(bad_decode_ids);
        g.begin();
        g.sparse_attention(q, k, v, bad_id, out, prefix, small);
        reject([&] { static_cast<void>(g.submit()); });
        expect_span_sentinel(out, 0, q_width);

        auto bad_id8_host = id_host;
        std::copy(bad_row2_ids.begin(), bad_row2_ids.end(), bad_id8_host.begin() + 2ull * budget);
        const auto bad_id8 = arena.array<uint32_t>(bad_id8_host);
        g.begin();
        g.sparse_attention_rows_chronological(q8, k, v, bad_id8, out8, scores8, factors8, prefix, 3, small);
        reject([&] { static_cast<void>(g.submit()); });
        expect_span_sentinel(out8, 2ull * q_width, 6ull * q_width);
        require(g.qsa_sparse_decode_calls() == 0 && g.qsa_sparse_verify_calls() == 0 &&
            g.qsa_verify_rows_routes() == QsaVerifyRowsRoutes{},
            "failed sparse QSA ID check published route counters");
    }
    arena.guards();
}

int screen_bn(const char* library) {
    MetalGraph base_g(library, 1536ull << 20);
    MetalGraph sparse_g(library, 1536ull << 20);
    prepare_base_policy(base_g, false);
    prepare_base_policy(sparse_g, true);

    Arena a_base{base_g, {}}, a_sparse{sparse_g, {}};
    const uint32_t q_width = actual.query_heads * actual.head_width;
    const uint32_t kv_width = actual.kv_heads * actual.head_width;
    const uint32_t budget = qsa_block_budget(actual);
    const uint32_t max_rows = 8208u;

    std::vector<float> q_host(8ull * q_width);
    std::vector<uint16_t> k_host(uint64_t(max_rows) * kv_width);
    std::vector<uint16_t> v_host(uint64_t(max_rows) * kv_width);
    fill_pattern(q_host, k_host, v_host, actual, 8, max_rows, 0);

    const auto q1_b = a_base.array<float>(std::span<const float>(q_host).first(q_width), true);
    const auto q8_b = a_base.array<float>(q_host, true);
    const auto k_b = a_base.array<uint16_t>(k_host, true);
    const auto v_b = a_base.array<uint16_t>(v_host, true);
    const auto out1_b = a_base.floats(q_width, 0.0f);
    const auto out8_b = a_base.floats(8ull * q_width, 0.0f);
    const auto scores8_b = a_base.floats(maximum_qsa_verify_rows_scores / 4, 0.0f);
    const auto fstore8_b = a_base.floats(maximum_qsa_verify_rows_factors / 4 + 2, 0.0f);
    const auto factors8_b = base_g.slice(fstore8_b, 4, maximum_qsa_verify_rows_factors);

    const auto q1_s = a_sparse.array<float>(std::span<const float>(q_host).first(q_width), true);
    const auto q8_s = a_sparse.array<float>(q_host, true);
    const auto k_s = a_sparse.array<uint16_t>(k_host, true);
    const auto v_s = a_sparse.array<uint16_t>(v_host, true);
    const auto out1_s = a_sparse.floats(q_width, 0.0f);
    const auto out8_s = a_sparse.floats(8ull * q_width, 0.0f);

    constexpr int warmup = 16, iters = 64, layers = 10;
    auto median_ms = [](const std::function<double()>& step) {
        for (int i = 0; i < warmup; ++i) static_cast<void>(step());
        std::vector<double> samples(iters);
        for (int i = 0; i < iters; ++i) samples[size_t(i)] = step();
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    };

    std::cout << std::fixed << std::setprecision(4);
    for (uint32_t prefix : {512u, 2051u, 4096u, 8192u}) {
        std::vector<uint32_t> id1_host(budget, 0u);
        const auto sel1 = canonical_selected_ids(prefix, actual, 0);
        std::copy(sel1.begin(), sel1.end(), id1_host.begin());
        const auto id1_b = a_base.array<uint32_t>(id1_host, true);
        const auto id1_s = a_sparse.array<uint32_t>(id1_host, true);

        const double base_decode = median_ms([&] {
            base_g.begin();
            for (int l = 0; l < layers; ++l) base_g.sparse_attention(q1_b, k_b, v_b, id1_b, out1_b, prefix, actual);
            return base_g.submit().wall_ms;
        });
        std::array<double, 3> bn_decode{};
        const std::array<uint32_t, 3> bns{8u, 16u, 32u};
        for (size_t i = 0; i < bns.size(); ++i) {
            const uint32_t bn = bns[i];
            bn_decode[i] = median_ms([&] {
                sparse_g.begin();
                for (int l = 0; l < layers; ++l) sparse_g.sparse_attention_bn(q1_s, k_s, v_s, id1_s, out1_s, prefix, actual, bn);
                return sparse_g.submit().wall_ms;
            });
        }
        std::cout << "screen_bn mode=decode_t1_x10 prefix=" << prefix
                  << " legacy_3pass_ms=" << base_decode
                  << " bn8_ms=" << bn_decode[0] << " bn8_speedup=" << (base_decode / bn_decode[0])
                  << " bn16_ms=" << bn_decode[1] << " bn16_speedup=" << (base_decode / bn_decode[1])
                  << " bn32_ms=" << bn_decode[2] << " bn32_speedup=" << (base_decode / bn_decode[2]) << '\n';

        for (uint32_t rows : {3u, 8u}) {
            std::vector<uint32_t> id8_host(8ull * budget, 0u);
            for (uint32_t r = 0; r < rows; ++r) {
                const auto sel = canonical_selected_ids(prefix + r + 1u, actual, r);
                std::copy(sel.begin(), sel.end(), id8_host.begin() + uint64_t(r) * budget);
            }
            const auto id8_b = a_base.array<uint32_t>(id8_host, true);
            const auto id8_s = a_sparse.array<uint32_t>(id8_host, true);
            const double base_verify = median_ms([&] {
                base_g.begin();
                for (int l = 0; l < layers; ++l) {
                    base_g.sparse_attention_rows_chronological(q8_b, k_b, v_b, id8_b, out8_b, scores8_b, factors8_b, prefix, rows, actual);
                }
                return base_g.submit().wall_ms;
            });
            std::array<double, 3> bn_verify{};
            for (size_t i = 0; i < bns.size(); ++i) {
                const uint32_t bn = bns[i];
                bn_verify[i] = median_ms([&] {
                    sparse_g.begin();
                    for (int l = 0; l < layers; ++l) {
                        sparse_g.sparse_attention_rows_bn(q8_s, k_s, v_s, id8_s, out8_s, prefix, rows, actual, bn);
                    }
                    return sparse_g.submit().wall_ms;
                });
            }
            std::cout << "screen_bn mode=verify_t" << rows << "_x10 prefix=" << prefix
                      << " legacy_3pass_ms=" << base_verify
                      << " bn8_ms=" << bn_verify[0] << " bn8_speedup=" << (base_verify / bn_verify[0])
                      << " bn16_ms=" << bn_verify[1] << " bn16_speedup=" << (base_verify / bn_verify[1])
                      << " bn32_ms=" << bn_verify[2] << " bn32_speedup=" << (base_verify / bn_verify[2]) << '\n';
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            throw std::runtime_error("usage: test-qsa_sparse_attention OWNED_METALLIB [screen-bn]");
        }
        if (argc == 3) {
            require(std::string(argv[2]) == "screen-bn", "unsupported test-qsa_sparse_attention mode");
            return screen_bn(argv[1]);
        }

        test_refusals_and_rollback(argv[1]);

        MetalGraph sparse_g(argv[1], 1536ull << 20);
        MetalGraph conc_g(argv[1], 1536ull << 20);
        MetalGraph base_g(argv[1], 1536ull << 20);
        prepare_base_policy(sparse_g, true, false);
        prepare_base_policy(conc_g, true, true);
        prepare_base_policy(base_g, false, false);
        same_memory(sparse_g.memory(), base_g.memory());

        for (uint32_t prefix : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 15u, 16u, 17u, 19u, 20u, 21u, 31u, 32u, 33u, 64u, 513u, 2048u, 4096u, 8192u}) {
            run_decode_case(sparse_g, base_g, small, prefix);
        }
        for (uint32_t prefix : {1u, 3u, 4u, 5u, 15u, 16u, 17u, 31u, 32u, 33u, 64u, 512u, 2047u, 2048u, 2049u, 2050u, 2051u, 2052u, 2053u, 4096u, 8192u, 65536u}) {
            run_decode_case(sparse_g, base_g, actual, prefix);
        }
        for (uint32_t pattern : {1u, 2u, 3u, 4u, 5u}) {
            run_decode_case(sparse_g, base_g, actual, 2051u, pattern);
            run_decode_case(sparse_g, base_g, actual, 4096u, pattern);
            run_decode_case(sparse_g, base_g, small, 21u, pattern);
        }

        for (uint32_t base : {0u, 1u, 3u, 12u, 13u, 15u, 16u, 17u, 2045u, 2048u, 4093u, 8184u}) {
            for (uint32_t rows : {1u, 3u, 8u}) {
                if (rows <= small.capacity - base) run_verify_case(sparse_g, conc_g, base_g, small, base, rows);
            }
        }
        for (uint32_t base : {0u, 3u, 15u, 512u, 2045u, 2046u, 2047u, 2048u, 2049u, 4093u, 8189u, 65528u}) {
            for (uint32_t rows : {1u, 3u, 8u}) {
                if (rows <= actual.capacity - base) run_verify_case(sparse_g, conc_g, base_g, actual, base, rows);
            }
        }
        for (uint32_t pattern : {1u, 2u, 3u, 4u, 5u}) {
            run_verify_case(sparse_g, conc_g, base_g, actual, 2046u, 8u, pattern);
            run_verify_case(sparse_g, conc_g, base_g, actual, 4093u, 3u, pattern);
            run_verify_case(sparse_g, conc_g, base_g, small, 14u, 5u, pattern);
        }

        std::cout << "PASS sparse QSA component Step 1 checks=" << checks << " batches=" << batches
                  << " decode_bn=" << TOKSPUR_QSA_SPARSE_DECODE_BN << " verify_bn=" << TOKSPUR_QSA_SPARSE_VERIFY_BN
                  << " worst_decode_nmse=" << std::scientific << worst_decode_nmse
                  << " worst_verify_nmse=" << worst_verify_nmse << " bound=1e-7\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
