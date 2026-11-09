#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
#include "tokspur/moe_q8_three_rows.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_QSA_PREFILL
#include "tokspur/qsa_prefill_ops.hpp"
#endif
#include "tokspur/attention_tile.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/operator_checkpoints.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>

#ifndef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
#error "attention tile is a private COST row component, not an ordinary caller"
#endif

namespace tokspur {
namespace {
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
void geometry(AttentionLayerShape s) {
    validate_residual_shape(s.residual); static_cast<void>(qsa_verify_rows_plan(0, 1, s.attention));
    require(s.residual.channels == s.attention.channels && s.moe.channels == s.attention.channels &&
        s.moe.hidden && s.moe.hidden <= 65536 && s.moe.experts && s.moe.experts <= 4096 &&
        s.moe.selected && s.moe.selected <= 64 && s.moe.selected <= s.moe.experts,
        "inconsistent bounded attention tile shape");
}
void graph_policy(MetalGraph& g, AttentionTileHcMode hc) {
    require(uint32_t(hc) <= 1, "invalid attention tile HC mode");
    require(g.kernel_policy() == std::array<uint32_t, 2>{16, 3} && !g.hc_q8_policy().splits &&
        (!g.memory().profile_counter_payload_bytes || std::getenv("TOKSPUR_ALLOW_KNOWN_PROFILE")
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                || g.carrier_diagnostic_active()
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
                || g.mtp_profile_reserved()
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                || g.task4_profile_reserved()
#endif
            ) && !g.memory().live_captures,
        "attention tile requires split16/block3/HCsplit0 without profiling or captures");
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    if (hc == AttentionTileHcMode::off)
        require(g.hc_exact_mode() == HcExactMode::off, "attention tile refuses exact-HC");
    else require(g.hc_exact_mode() == HcExactMode::combined, "attention tile requires explicit matching combined-HC");
#else
    require(hc == AttentionTileHcMode::off, "attention tile combined-HC support is not compiled");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(!g.wide_q8_exact_enabled(), "attention tile refuses scalar wide-Q8");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!g.wide_q8_many_enabled(), "attention tile refuses many wide-Q8");
#endif
}
std::array<uint64_t, 43> counts(AttentionLayerShape s) {
    geometry(s); const auto b = s.residual.branches, c = s.residual.channels, r = s.residual.rank;
    const auto bc = uint64_t(b) * c, q = uint64_t(s.attention.query_heads) * s.attention.head_width;
    const auto k = uint64_t(s.attention.kv_heads) * s.attention.head_width;
    const auto iq = uint64_t(s.attention.index_heads) * s.attention.index_width, ik = uint64_t(s.attention.index_width);
    const auto blocks = uint64_t(qsa_block_capacity(s.attention)), budget = uint64_t(qsa_block_budget(s.attention));
    const auto slots = (blocks + 511) / 512 * budget, h = uint64_t(s.moe.hidden), selected = uint64_t(s.moe.selected);
    return {bc, r, bc, b, c, b,
        q * 2, k, k, q, q, q, k, q, k, iq, ik, iq, iq, ik, ik, ik, blocks, slots, slots, slots, slots, budget, q, q,
        s.moe.experts, selected, selected, h * selected, h * selected, uint64_t(c) * selected, h, h, c, 1,
        bc, c, c};
}
AttentionLayerScratch assemble(const std::array<GpuView, 43>& v) {
    return {{v[0], v[1], v[2], v[3], v[4], v[5]},
        {v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16], v[17], v[18], v[19], v[20], v[21], v[22],
            {v[23], v[24], v[25], v[26]}, v[27], v[28], v[29]},
        {v[30], v[31], v[32], v[33], v[34], v[35], v[36], v[37], v[38], v[39]}, v[40], v[41], v[42]};
}
GpuView row(MetalGraph& g, GpuView v, uint64_t width, uint32_t i) { return g.slice(v, uint64_t(i) * width * 4, width * 4); }
void bytes(GpuView v, uint64_t n) { require(v.bytes == n, "attention tile buffer geometry mismatch"); }
void workspace(const AttentionTileScratch& s, AttentionLayerShape shape) {
    const auto n = counts(shape); const auto views = s.rows.views();
    for (size_t i = 0; i < n.size(); ++i) bytes(views[i], checked_mul(checked_mul(n[i], 8), 4));
    bytes(s.prefix_sums, 7ull * shape.attention.index_width * 4); bytes(s.core_scores, maximum_qsa_verify_rows_scores); bytes(s.core_factors, maximum_qsa_verify_rows_factors);
}
void matrix(GpuMatrix w, uint32_t columns, uint32_t rows, uint32_t experts = 1) {
    require(can_decode(w.encoding), "unsupported attention tile matrix encoding"); const auto& e = encoding(w.encoding);
    require(columns && columns % e.elements_per_block == 0 && rows && experts, "attention tile matrix block geometry");
    bytes(w.view, checked_mul(checked_mul(checked_mul(columns / e.elements_per_block, e.bytes_per_block), rows), experts));
}
void weight_shapes(const MetalGraph& g, const AttentionLayerWeights& w, AttentionLayerShape s) {
    const auto c = s.residual.channels, b = s.residual.branches, rank = s.residual.rank;
    for (const auto& r : {w.attention_read, w.ffn_read}) {
        bytes(r.gains, uint64_t(b) * c * 4); matrix(r.down, b * c, rank); matrix(r.up, rank, b * c); matrix(r.inject, b * c, b);
    }
    const auto& a = w.attention; const auto q = s.attention.query_heads * s.attention.head_width;
    const auto k = s.attention.kv_heads * s.attention.head_width, iq = s.attention.index_heads * s.attention.index_width;
    require(a.query_gate.encoding == 8 && a.key.encoding == 8 && a.value.encoding == 8 && a.output.encoding == 8 &&
        a.index_query.encoding == 30 && a.index_key.encoding == 30, "attention tile requires original Q8/BF16 QSA projections");
    matrix(a.query_gate, c, 2 * q); matrix(a.key, c, k); matrix(a.value, c, k); matrix(a.output, q, c);
    matrix(a.index_query, c, iq); matrix(a.index_key, c, s.attention.index_width);
    bytes(a.query_gains, s.attention.head_width * 4ull); bytes(a.key_gains, a.query_gains.bytes);
    bytes(a.index_query_gains, s.attention.index_width * 4ull); bytes(a.index_key_gains, a.index_query_gains.bytes);
    bytes(a.frequencies, s.attention.rotary_width * 2ull);
    const auto& m = w.ffn; const auto h = s.moe.hidden, e = s.moe.experts;
    matrix(m.router, c, e);
    if (!is_planar_moe(g, m, s.moe)) {
        matrix(m.gate, c, h, e); matrix(m.up, c, h, e); matrix(m.down, h, c, e);
    }
    matrix(m.shared_gate, c, h); matrix(m.shared_up, c, h); matrix(m.shared_down, h, c); matrix(m.shared_router, c, 1);
}
std::array<GpuView, 27> weight_views(const AttentionLayerWeights& w) {
    const auto& a = w.attention; const auto& m = w.ffn;
    return {w.attention_read.gains, w.attention_read.down.view, w.attention_read.up.view, w.attention_read.inject.view,
        w.ffn_read.gains, w.ffn_read.down.view, w.ffn_read.up.view, w.ffn_read.inject.view,
        a.query_gate.view, a.key.view, a.value.view, a.output.view, a.index_query.view, a.index_key.view,
        a.query_gains, a.key_gains, a.index_query_gains, a.index_key_gains, a.frequencies,
        m.router.view, m.gate.view, m.up.view, m.down.view, m.shared_gate.view, m.shared_up.view, m.shared_down.view, m.shared_router.view};
}
void project(MetalGraph& g, GpuMatrix w, GpuView input, GpuView output, uint32_t columns, uint32_t rows,
             uint32_t steps, AttentionTileMode mode) {
    if (mode == AttentionTileMode::core_only) {
        for (uint32_t t = 0; t < steps; ++t) g.gemv(w.encoding, w.view, row(g, input, columns, t), row(g, output, rows, t), columns, rows);
        return;
    }
    uint32_t at = 0;
    while (at < steps) {
        const uint32_t n = g.q8_many_width(steps - at);
        g.q8_matvec_many(w.view, g.slice(input, uint64_t(at) * columns * 4, uint64_t(n) * columns * 4),
            g.slice(output, uint64_t(at) * rows * 4, uint64_t(n) * rows * 4), columns, rows, n); at += n;
    }
}
uint64_t extra(MetalGraph& g, GpuMatrix w, uint32_t columns, uint32_t rows) { return g.gemv_dispatches(w.encoding, columns, rows) - 1; }
}
std::array<GpuView, 46> AttentionTileScratch::views() const {
    std::array<GpuView, 46> out{}; const auto v = rows.views(); std::copy(v.begin(), v.end(), out.begin());
    out[43] = prefix_sums; out[44] = core_scores; out[45] = core_factors; return out;
}
void prepare_attention_tile(MetalGraph& g, AttentionTileHcMode hc) {
    graph_policy(g, hc); g.prepare_qsa_verify_rows(); g.prepare_q8_matvec_many_all();
}
uint64_t attention_tile_scratch_bytes(AttentionLayerShape shape) {
    uint64_t bytes = checked_add(maximum_qsa_verify_rows_scores, maximum_qsa_verify_rows_factors);
    bytes = checked_add(bytes, checked_mul(checked_mul(7, shape.attention.index_width), 4));
    for (const auto n : counts(shape)) {
        require(checked_mul(n, 8) <= 16'777'216, "attention tile scratch count bound");
        bytes = checked_add(bytes, checked_mul(checked_mul(n, 8), 4));
    }
    return bytes;
}
AttentionTileScratch allocate_attention_tile_scratch(MetalGraph& g, AttentionLayerShape shape) {
    const auto n = counts(shape); std::array<GpuView, 43> views{}; AttentionTileScratch s{};
    try {
        for (size_t i = 0; i < views.size(); ++i) {
            require(n[i] * 8 <= 16'777'216, "attention tile scratch count bound");
            views[i] = g.zeros(checked_mul(checked_mul(n[i], 8), 4), "private eight-row QSA layer scratch");
        }
        s.rows = assemble(views); s.prefix_sums = g.zeros(7ull * shape.attention.index_width * 4, "private QSA prefix-sum tape");
        s.core_scores = g.zeros(maximum_qsa_verify_rows_scores, "private QSA row fixed scores");
        s.core_factors = g.zeros(maximum_qsa_verify_rows_factors, "private QSA row fixed factors");
    } catch (...) {
        for (const auto v : views) if (v.owner) g.release(v);
        for (const auto v : {s.prefix_sums, s.core_scores, s.core_factors}) if (v.owner) g.release(v);
        throw;
    }
    return s;
}
QsaScratch allocate_attention_tile_qsa_scratch(MetalGraph& g, AttentionLayerShape shape) {
    const auto n = counts(shape); std::array<GpuView, 43> views{};
    try {
        for (size_t i = 6; i < 30; ++i) {
            require(n[i] * 8 <= 16'777'216, "attention tile QSA scratch count bound");
            views[i] = g.zeros(checked_mul(checked_mul(n[i], 8), 4), "private known-prefill QSA rows");
        }
    } catch (...) { for (auto v : views) if (v.owner) g.release(v); throw; }
    return assemble(views).attention;
}
void release_attention_tile_scratch(MetalGraph& g, const AttentionTileScratch& s) { for (auto v : s.views()) g.release(v); }
AttentionLayerScratch attention_tile_token(MetalGraph& g, const AttentionTileScratch& s, AttentionLayerShape shape, uint32_t i) {
    require(i < 8, "attention tile row outside capacity"); workspace(s, shape);
    const auto n = counts(shape); const auto full = s.rows.views(); std::array<GpuView, 43> views{};
    for (size_t at = 0; at < views.size(); ++at) views[at] = row(g, full[at], n[at], i);
    return assemble(views);
}
uint64_t attention_tile_dispatches(MetalGraph& g, const AttentionLayerWeights& w, uint32_t base, uint32_t steps,
                                  AttentionLayerShape shape, AttentionTileMode mode, AttentionTileHcMode hc) {
    geometry(shape); static_cast<void>(qsa_verify_rows_plan(base, steps, shape.attention)); graph_policy(g, hc); weight_shapes(g, w, shape);
    require(uint32_t(mode) <= 1, "invalid attention tile projection mode");
    const auto c = shape.residual.channels, b = shape.residual.branches, rank = shape.residual.rank, h = shape.moe.hidden;
    uint64_t common = 0;
    for (const auto& r : {w.attention_read, w.ffn_read})
        common += extra(g, r.down, b * c, rank) + extra(g, r.up, rank, b * c) + extra(g, r.inject, b * c, b);
    common += extra(g, w.ffn.router, c, shape.moe.experts) + extra(g, w.ffn.shared_gate, c, h) +
        extra(g, w.ffn.shared_up, c, h) + extra(g, w.ffn.shared_down, h, c) + extra(g, w.ffn.shared_router, c, 1);
    // Original28 + ordinary QSA schedule, replacing precisely N three-pass
    // chronological attention calls by3 (or by1 when single-pass sparse QSA is enabled).
    uint64_t count =
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        g.qsa_sparse_enabled() ? 1 :
#endif
        3;
    const uint64_t planar_saved = is_planar_moe(g, w.ffn, shape.moe) ? (g.swiglu_fused_enabled() ? 2ull : 3ull) : 0ull;
    for (uint32_t t = 0; t < steps; ++t) count += (g.swiglu_fused_enabled() ? 26 : 28) - planar_saved + qsa_dispatches(shape.attention, base + t) - 1 + common;
    const uint32_t chunks = g.q8_many_chunks(steps);
    if (mode == AttentionTileMode::core_and_q8) count -= 4ull * (steps - chunks);
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    if (g.q8_three_rows_enabled() && steps == 3) count -= moe_q8_three_rows_saved_dispatches(g,w.ffn,shape.moe);
#endif
    return count;
}
void encode_attention_layer_tile(MetalGraph& g, const AttentionLayerWeights& w, const AttentionTileScratch& s,
    const QsaCache& cache, GpuView sum, GpuView next, GpuView input, GpuView output, uint32_t base, uint32_t steps,
    AttentionLayerShape shape, QsaPolicy policy, RouterMode router, bool normalize, AttentionTileMode mode, AttentionTileHcMode hc
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    , const VerifyNaxMoeScratch* verify_nax_moe
#endif
    ) {
    try {
        static_cast<void>(attention_tile_dispatches(g, w, base, steps, shape, mode, hc)); validate_qsa_policy(policy);
        require(uint32_t(router) <= 1, "invalid attention tile router policy"); workspace(s, shape);
        const auto a = shape.attention; const auto c = a.channels, q = a.query_heads * a.head_width, k = a.kv_heads * a.head_width;
        const auto width = uint64_t(shape.residual.branches) * c;
        bytes(input, 8 * width * 4); bytes(output, input.bytes); bytes(sum, a.index_width * 4ull); bytes(next, sum.bytes);
        bytes(cache.keys, uint64_t(a.capacity) * k * 2); bytes(cache.values, cache.keys.bytes);
        bytes(cache.index_keys, uint64_t(qsa_block_capacity(a)) * a.index_width * 2);
        std::array<GpuView, 29> reads{input, sum}; const auto weights = weight_views(w); std::copy(weights.begin(), weights.end(), reads.begin() + 2);
        const auto cv = cache.views(); g.validate_disjoint(reads, cv);
        std::array<GpuView, 32> protected_reads{}; auto at = std::copy(reads.begin(), reads.end(), protected_reads.begin()); std::copy(cv.begin(), cv.end(), at);
        std::array<GpuView, 48> writes{output, next}; const auto scratch = s.views(); std::copy(scratch.begin(), scratch.end(), writes.begin() + 2);
        g.validate_disjoint(protected_reads, writes); // Every late capability/alias checked before encoding.
        bool use_verify_nax = false;
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        use_verify_nax = g.verify_nax_active() && verify_nax_moe && mode == AttentionTileMode::core_and_q8 && steps >= 2 && steps <= 8;
        if (use_verify_nax) {
            std::vector<GpuView> all_protected(protected_reads.begin(), protected_reads.end());
            all_protected.insert(all_protected.end(), writes.begin(), writes.end());
            g.validate_disjoint(all_protected, verify_nax_moe->views());
            encode_verify_nax_residual_read(g, w.attention_read, s.rows.residual, input, shape.residual, steps, policy.rms_epsilon);
            encode_verify_nax_projection(g, w.attention.query_gate, s.rows.residual.input, s.rows.attention.packed, c, 2 * q, steps);
            encode_verify_nax_projection(g, w.attention.key, s.rows.residual.input, s.rows.attention.raw_key, c, k, steps);
            encode_verify_nax_projection(g, w.attention.value, s.rows.residual.input, s.rows.attention.raw_value, c, k, steps);
            encode_verify_nax_projection(g, w.attention.index_query, s.rows.residual.input, s.rows.attention.raw_index_query, c, a.index_heads * a.index_width, steps);
            encode_verify_nax_projection(g, w.attention.index_key, s.rows.residual.input, s.rows.attention.raw_index_key, c, a.index_width, steps);
            TOKSPUR_OPERATOR_CHECKPOINT(g, "dense_q8", "verify_qsa_qkv", {steps, c, 2 * q + 2 * k, 8});
        } else
#endif
        {
            for (uint32_t t = 0; t < steps; ++t) encode_residual_read(g, w.attention_read, attention_tile_token(g, s, shape, t).residual,
                row(g, input, width, t), shape.residual, policy.rms_epsilon);
            project(g, w.attention.query_gate, s.rows.residual.input, s.rows.attention.packed, c, 2 * q, steps, mode);
            project(g, w.attention.key, s.rows.residual.input, s.rows.attention.raw_key, c, k, steps, mode);
            project(g, w.attention.value, s.rows.residual.input, s.rows.attention.raw_value, c, k, steps, mode);
            TOKSPUR_OPERATOR_CHECKPOINT(g, "dense_q8", "verify_qsa_qkv", {steps, c, 2 * q + 2 * k, 8});
        }
#if defined(TOKSPUR_PRIVATE_GROUPED_MM_F32) && defined(TOKSPUR_PRIVATE_QSA_PREFILL) && defined(TOKSPUR_PRIVATE_GDN_CHUNK)
        if (use_verify_nax && verify_nax_qsa_batch_supported(a)) {
            auto qpf = [&](QsaPrefillOp operation, std::initializer_list<GpuView> bufs, const QsaPrefillParams& p) {
                g.qsa_prefill_dispatch(operation, std::span<const GpuView>(bufs.begin(), bufs.size()), p);
            };
            const uint32_t iq = a.index_heads * a.index_width, ik = a.index_width, pairs = a.rotary_width / 2u;
            const uint32_t block_cap = qsa_block_capacity(a), budget = qsa_block_budget(a);
            const uint32_t slots = ((block_cap + 511u) / 512u) * budget;
            const uint32_t first_pool = base / 4u, max_blocks = (base + steps) / 4u, pool_count = max_blocks - first_pool;
            const uint32_t eps_bits = std::bit_cast<uint32_t>(policy.rms_epsilon);
            const uint64_t kv_off = uint64_t(base) * k * 2, kv_bytes = uint64_t(steps) * k * 2;
            const auto& ar = s.rows.attention;
            qpf(QsaPrefillOp::prep_query, {ar.packed, w.attention.query_gains, w.attention.frequencies, ar.rotary_query, ar.gate},
                {steps, a.query_heads, a.head_width, pairs, base, uint32_t(policy.packing), 2u * q, q, uint32_t(policy.rotary_pairs), eps_bits});
            qpf(QsaPrefillOp::prep_kv, {ar.raw_key, ar.raw_value, w.attention.key_gains, w.attention.frequencies,
                g.slice(cache.keys, kv_off, kv_bytes), g.slice(cache.values, kv_off, kv_bytes)},
                {steps, a.kv_heads, a.head_width, pairs, base, k, k, k, uint32_t(policy.rotary_pairs), eps_bits});
            qpf(QsaPrefillOp::prep_index_query, {ar.raw_index_query, w.attention.index_query_gains, w.attention.frequencies, ar.rotary_index_query},
                {steps, a.index_heads, ik, pairs, base, iq, iq, uint32_t(policy.rotary_pairs), eps_bits});
            qpf(QsaPrefillOp::verify_index_pool, {ar.raw_index_key, sum, ar.pooled, s.prefix_sums, next},
                {steps, ik, base, ik, ik, pool_count, 4u});
            if (pool_count) {
                g.rms_rows({g.slice(ar.pooled, 0, uint64_t(pool_count) * ik * 4), pool_count, ik, uint64_t(ik) * 4},
                    w.attention.index_key_gains, {g.slice(ar.normalized_index_key, 0, uint64_t(pool_count) * ik * 4), pool_count, ik, uint64_t(ik) * 4},
                    ik, policy.rms_epsilon, true);
                qpf(QsaPrefillOp::rope_rows, {ar.normalized_index_key, w.attention.frequencies, ar.rotary_index_key},
                    {pool_count, 1u, ik, pairs, first_pool * 4u, 4u, ik, ik, uint32_t(policy.rotary_pairs)});
                qpf(QsaPrefillOp::store_bf16,
                    {ar.rotary_index_key, g.slice(cache.index_keys, uint64_t(first_pool) * ik * 2, uint64_t(pool_count) * ik * 2)},
                    {pool_count, ik, ik, ik});
            }
            if (max_blocks > budget) {
                qpf(QsaPrefillOp::index_scores, {ar.rotary_index_query, cache.index_keys, ar.scores},
                    {steps, base, a.index_heads, ik, block_cap, iq, budget});
                qpf(QsaPrefillOp::select_sort, {ar.scores, ar.selection.scores_a, ar.selection.ids_a},
                    {steps, base, block_cap, slots, budget});
                GpuView in_scores = ar.selection.scores_a, in_ids = ar.selection.ids_a;
                GpuView out_scores = ar.selection.scores_b, out_ids = ar.selection.ids_b;
                for (uint32_t span = 512u; span < max_blocks; span *= 2u) {
                    qpf(QsaPrefillOp::select_merge, {in_scores, in_ids, out_scores, out_ids},
                        {steps, base, slots, budget, span});
                    std::swap(in_scores, out_scores);
                    std::swap(in_ids, out_ids);
                }
                qpf(QsaPrefillOp::select_ids, {in_ids, ar.selected_ids},
                    {steps, base, slots, budget, budget});
            }
        } else
#endif
        {
            auto previous = sum;
            for (uint32_t t = 0; t < steps; ++t) {
                const auto token = attention_tile_token(g, s, shape, t); const auto& r = token.attention; const auto position = base + t;
                const auto current = t + 1 == steps ? next : row(g, s.prefix_sums, a.index_width, t);
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
                if (!use_verify_nax)
#endif
                {
                    g.gemv(30, w.attention.index_query.view, token.residual.input, r.raw_index_query, c, a.index_heads * a.index_width);
                    g.gemv(30, w.attention.index_key.view, token.residual.input, r.raw_index_key, c, a.index_width);
                }
                g.split_query_gate(r.packed, r.query, r.gate, a.query_heads, a.head_width, policy.packing);
                g.rms(r.query, w.attention.query_gains, r.normalized_query, a.head_width, policy.rms_epsilon);
                g.rms(r.raw_key, w.attention.key_gains, r.normalized_key, a.head_width, policy.rms_epsilon);
                g.rms(r.raw_index_query, w.attention.index_query_gains, r.normalized_index_query, a.index_width, policy.rms_epsilon);
                g.rotary(r.normalized_query, w.attention.frequencies, r.rotary_query, a.query_heads, a.head_width, position, policy.rotary_pairs);
                g.rotary(r.normalized_key, w.attention.frequencies, r.rotary_key, a.kv_heads, a.head_width, position, policy.rotary_pairs);
                g.rotary(r.normalized_index_query, w.attention.frequencies, r.rotary_index_query, a.index_heads, a.index_width, position, policy.rotary_pairs);
                g.store_bf16(r.rotary_key, g.slice(cache.keys, uint64_t(position) * k * 2, k * 2));
                g.store_bf16(r.raw_value, g.slice(cache.values, uint64_t(position) * k * 2, k * 2));
                g.index_accumulate(previous, r.raw_index_key, current, r.pooled, position, a.compression);
                const auto blocks = (position + 1) / a.compression;
                if ((position + 1) % a.compression == 0) {
                    g.rms(r.pooled, w.attention.index_key_gains, r.normalized_index_key, a.index_width, policy.rms_epsilon);
                    g.rotary(r.normalized_index_key, w.attention.frequencies, r.rotary_index_key, 1, a.index_width, position + 1 - a.compression, policy.rotary_pairs);
                    g.store_bf16(r.rotary_index_key, g.slice(cache.index_keys, uint64_t(blocks - 1) * a.index_width * 2, a.index_width * 2));
                }
                if (blocks > qsa_block_budget(a)) {
                    g.index_scores(r.rotary_index_query, cache.index_keys, r.scores, a.index_heads, a.index_width, blocks);
                    g.select_blocks(r.scores, r.selected_ids, r.selection, blocks, qsa_block_budget(a));
                }
                previous = current;
            }
        }
#if defined(TOKSPUR_PRIVATE_GROUPED_MM_F32) && defined(TOKSPUR_PRIVATE_QSA_PREFILL)
        if (use_verify_nax && verify_nax_qsa_batch_supported(a) && g.verify_nax_qsa_attention_supported(a)) {
            const uint32_t budget = qsa_block_budget(a);
            g.nax_qsa_attention(
                g.slice(s.rows.attention.rotary_query, 0, uint64_t(steps) * q * 4),
                cache.keys,
                cache.values,
                g.slice(s.rows.attention.selected_ids, 0, uint64_t(steps) * budget * 4),
                g.slice(s.rows.attention.attention, 0, uint64_t(steps) * q * 4),
                base, steps, a, budget);
        } else
#endif
        {
            g.sparse_attention_rows_chronological(s.rows.attention.rotary_query, cache.keys, cache.values, s.rows.attention.selected_ids,
                s.rows.attention.attention, s.core_scores, s.core_factors, base, steps, a);
        }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        if (use_verify_nax) {
            const uint64_t active_q_bytes = uint64_t(steps) * q * 4;
            g.multiply(g.slice(s.rows.attention.attention, 0, active_q_bytes),
                g.slice(s.rows.attention.gate, 0, active_q_bytes),
                g.slice(s.rows.attention.gated, 0, active_q_bytes), true);
            TOKSPUR_OPERATOR_CHECKPOINT(g, "qsa", "verify_core_and_indexers", {steps, base, a.query_heads, a.head_width});
            encode_verify_nax_projection(g, w.attention.output, s.rows.attention.gated, s.rows.attention_output, q, c, steps);
            TOKSPUR_OPERATOR_CHECKPOINT(g, "dense_q8", "verify_qsa_output", {steps, q, c, 8});
            encode_verify_nax_residual_write_read(g, w.ffn_read, s.rows.residual, input, s.rows.attention_output, s.rows.residual.scales, s.rows.intermediate_streams, shape.residual, steps, policy.rms_epsilon);
            encode_verify_nax_moe(g, w.ffn, s.rows.moe, *verify_nax_moe, s.rows.residual.input, s.rows.ffn_output, shape.moe, router, normalize, steps);
            encode_verify_nax_residual_write(g, s.rows.intermediate_streams, s.rows.ffn_output, s.rows.residual.scales, output, shape.residual, steps);
            return;
        }
#endif
        for (uint32_t t = 0; t < steps; ++t) {
            const auto r = attention_tile_token(g, s, shape, t).attention; g.multiply(r.attention, r.gate, r.gated, true);
        }
        TOKSPUR_OPERATOR_CHECKPOINT(g, "qsa", "verify_core_and_indexers", {steps, base, a.query_heads, a.head_width});
        project(g, w.attention.output, s.rows.attention.gated, s.rows.attention_output, q, c, steps, mode);
        TOKSPUR_OPERATOR_CHECKPOINT(g, "dense_q8", "verify_qsa_output", {steps, q, c, 8});
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        if (g.q8_three_rows_enabled() && steps == 3) {
            for (uint32_t t = 0; t < 3; ++t) {
                const auto r = attention_tile_token(g,s,shape,t);
                g.residual_write(row(g,input,width,t),r.attention_output,r.residual.scales,r.intermediate_streams);
                encode_residual_read(g,w.ffn_read,r.residual,r.intermediate_streams,shape.residual,policy.rms_epsilon);
            }
            encode_moe_q8_three_rows(g,w.ffn,s.rows.moe,s.rows.residual.input,s.rows.ffn_output,shape.moe,router,normalize,8);
            for (uint32_t t = 0; t < 3; ++t) {
                const auto r = attention_tile_token(g,s,shape,t);
                g.residual_write(r.intermediate_streams,r.ffn_output,r.residual.scales,row(g,output,width,t));
            }
            return;
        }
#endif
        for (uint32_t t = 0; t < steps; ++t) {
            const auto token = attention_tile_token(g, s, shape, t);
            g.residual_write(row(g, input, width, t), token.attention_output, token.residual.scales, token.intermediate_streams);
            encode_residual_read(g, w.ffn_read, token.residual, token.intermediate_streams, shape.residual, policy.rms_epsilon);
            encode_moe(g, w.ffn, token.moe, token.residual.input, token.ffn_output, shape.moe, router, normalize
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                , t
#endif
                );
            g.residual_write(token.intermediate_streams, token.ffn_output, token.residual.scales, row(g, output, width, t));
        }
    } catch (...) { try { g.abort(); } catch (const FormatError&) {} throw; }
}
} // namespace tokspur
