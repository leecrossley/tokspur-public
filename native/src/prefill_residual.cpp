#include "tokspur/prefill_residual.hpp"
#include "tokspur/operator_checkpoints.hpp"
#include "tokspur/grouped_mm.hpp"
#include "tokspur/gguf.hpp"
#include <cmath>

#if !defined(TOKSPUR_PRIVATE_GROUPED_MM) || !defined(TOKSPUR_PRIVATE_GDN_CHUNK)
#error "Prefill residual rows require private dense Q8 and row RMS operators"
#endif
namespace tokspur {
namespace {
void need(bool ok, const char* why) { if (!ok) throw FormatError(why); }
std::array<uint64_t, 6> widths(ResidualShape s) {
    validate_residual_shape(s);
    return {checked_mul(s.branches, s.channels), s.rank, checked_mul(s.branches, s.channels), s.branches, s.channels, s.branches};
}
void rows_check(GpuRowsF32 v, uint32_t count, uint32_t columns) {
    need(v.rows == count && v.columns == columns && count && v.stride_bytes >= uint64_t(columns) * 4 && v.stride_bytes % 4 == 0 &&
        v.view.bytes >= checked_add(checked_mul(count - 1, v.stride_bytes), uint64_t(columns) * 4), "prefill residual row geometry");
}
void matrix(GpuMatrix w, uint32_t c, uint32_t r) {
    const auto& f = encoding(w.encoding);
    need(c % f.elements_per_block == 0 && w.view.bytes == checked_mul(checked_mul(c / f.elements_per_block, f.bytes_per_block), r), "prefill residual matrix geometry");
}
void weights(MetalGraph& g, const ResidualWeights& w, ResidualShape s) {
    static_cast<void>(widths(s)); const auto width = s.branches * s.channels;
    need(w.gains.bytes == uint64_t(width) * 4 && w.up.encoding == 8, "prefill residual gains or Q8 up projection");
    matrix(w.down, width, s.rank); matrix(w.up, s.rank, width); matrix(w.inject, width, s.branches);
    const auto down_layout = g.prefill_weight_layout(w.down.view);
    const auto inject_layout = g.prefill_weight_layout(w.inject.view);
    need((down_layout == PrefillWeightLayout::gguf || down_layout == PrefillWeightLayout::q8_lanes_v1) &&
        (inject_layout == PrefillWeightLayout::gguf || (w.inject.encoding == 8 && inject_layout == PrefillWeightLayout::q8_lanes_v1)),
        "prefill residual preserves original HC down and injection layouts");
}
GpuView row(MetalGraph& g, GpuRowsF32 rows, uint32_t index) { return g.slice(rows.view, uint64_t(index) * rows.stride_bytes, uint64_t(rows.columns) * 4); }
GpuRowsF32 tight(GpuView view, uint32_t n, uint32_t width) { return {view, n, width, uint64_t(width) * 4}; }
void scratch_check(const PrefillResidualScratch& s, ResidualShape shape, uint32_t n) {
    need(n && n <= s.capacity && s.capacity <= 2048, "prefill residual capacity");
    const auto ws = widths(shape); const auto views = s.rows.views();
    for (size_t i = 0; i < ws.size(); ++i) need(views[i].bytes == checked_mul(checked_mul(ws[i], s.capacity), 4), "prefill residual scratch geometry");
}
// The staged experiment covers partial rows at the original HC geometry. The legacy wide experiment retains its full-chunk boundary.
bool batch_hc_down(const MetalGraph& g, const ResidualWeights& w, ResidualShape s, uint32_t n) {
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    return g.prefill_hc_down_f32_rows(n) && s.branches == 4 && s.channels == 2560 && s.rank == 320 &&
        w.down.encoding == 8 && g.prefill_weight_layout(w.down.view) == PrefillWeightLayout::gguf;
#else
    static_cast<void>(g); static_cast<void>(w); static_cast<void>(s); static_cast<void>(n); return false;
#endif
}
uint32_t max_hc_rows(const MetalGraph& g) { return g.prefill_dense_nax() ? 2048u : 512u; }
bool batch_hc_down_window(const MetalGraph& g, const ResidualWeights& w, ResidualShape s, uint32_t n) {
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    const auto layout = g.prefill_weight_layout(w.down.view);
    return g.prefill_hc_down_window() && n >= 2 && n <= max_hc_rows(g) && s.branches == 4 && s.channels == 2560 && s.rank == 320 &&
        w.down.encoding == 8 && (layout == PrefillWeightLayout::gguf || layout == PrefillWeightLayout::q8_lanes_v1);
#else
    static_cast<void>(g); static_cast<void>(w); static_cast<void>(s); static_cast<void>(n); return false;
#endif
}
void abort(MetalGraph& g) { try { g.abort(); } catch (const FormatError&) {} }
bool batch_rows(const MetalGraph& g, ResidualShape s, uint32_t n) {
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    return g.prefill_hc_rows() && s.branches == 4 && s.channels == 2560 && n >= 2 && n <= max_hc_rows(g);
#else
    static_cast<void>(g); static_cast<void>(s); static_cast<void>(n); return false;
#endif
}
bool batch_q8_inject(const MetalGraph& g, const ResidualWeights& w, ResidualShape s, uint32_t n) {
    return batch_rows(g, s, n) && g.prefill_dense_nax() && w.inject.encoding == 8 &&
        (s.branches * 4) % 16 == 0 && (s.branches * s.channels) % 64 == 0;
}
void encode_prefill_residual_after_rms(MetalGraph& g, const ResidualWeights& w, const PrefillResidualScratch& s,
    GpuRowsF32 normalized, ResidualShape shape, uint32_t n) {
    const auto width = shape.branches * shape.channels;
    const auto bottleneck = tight(s.rows.bottleneck, n, shape.rank), scores = tight(s.rows.scores, n, width);
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    if (batch_hc_down_window(g, w, shape, n)) g.hc_down_q8_window(w.down.view, normalized, bottleneck);
    else
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    if (batch_hc_down(g, w, shape, n)) g.dense_hc_down_f32(w.down.view, normalized, bottleneck);
    else
#endif
    for (uint32_t t = 0; t < n; ++t) g.gemv(w.down.encoding, w.down.view, row(g, normalized, t), row(g, bottleneck, t), width, shape.rank);
    TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","down",{n,width,shape.rank,w.down.encoding});
    const auto active = g.slice(s.rows.bottleneck, 0, uint64_t(n) * shape.rank * 4);
    g.activate(active, active, Activation::silu, 1.0f / shape.branches);
    TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","activation",{n,shape.rank,shape.branches,0});
    g.dense_mm_q8(g.prefill_weight_layout(w.up.view), w.up.view, bottleneck, scores);
    TOKSPUR_OPERATOR_CHECKPOINT(g,"dense_q8","hc_up",{n,shape.rank,width,w.up.encoding});
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    if (batch_rows(g, shape, n)) {
        const auto inject = g.slice(s.rows.inject_logits, 0, uint64_t(n) * shape.branches * 4);
        const auto normalized_rows = g.slice(normalized.view, 0, uint64_t(n) * width * 4), score_rows = g.slice(scores.view, 0, uint64_t(n) * width * 4);
        if (w.inject.encoding == 0) g.prefill_hc_inject_rows(w.inject.view, normalized_rows, inject, n);
        else if (batch_q8_inject(g, w, shape, n)) g.dense_mm_q8(g.prefill_weight_layout(w.inject.view), w.inject.view, normalized, tight(inject, n, shape.branches));
        else for (uint32_t t = 0; t < n; ++t) g.gemv(w.inject.encoding, w.inject.view, row(g, normalized, t), g.slice(inject, uint64_t(t) * shape.branches * 4, uint64_t(shape.branches) * 4), width, shape.branches);
        g.hc_read_rows(normalized_rows, score_rows, inject, g.slice(s.rows.input, 0, uint64_t(n) * shape.channels * 4), g.slice(s.rows.scales, 0, uint64_t(n) * shape.branches * 4), n);
    } else
#endif
    for (uint32_t t = 0; t < n; ++t) {
        const auto inject = g.slice(s.rows.inject_logits, uint64_t(t) * shape.branches * 4, uint64_t(shape.branches) * 4);
        g.gemv(w.inject.encoding, w.inject.view, row(g, normalized, t), inject, width, shape.branches);
        g.residual_read(row(g, normalized, t), row(g, scores, t), inject,
            g.slice(s.rows.input, uint64_t(t) * shape.channels * 4, uint64_t(shape.channels) * 4),
            g.slice(s.rows.scales, uint64_t(t) * shape.branches * 4, uint64_t(shape.branches) * 4), shape.branches, shape.channels);
    }
    TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","inject_and_read_interleaved",{n,width,shape.channels,shape.branches});
}
}
bool can_fuse_prefill_hc_consumer(const MetalGraph& g, ResidualShape s, uint32_t n) {
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    return g.hc_consumer_fused_enabled() && batch_rows(g, s, n);
#else
    static_cast<void>(g); static_cast<void>(s); static_cast<void>(n); return false;
#endif
}
PrefillResidualScratch allocate_prefill_residual_scratch(MetalGraph& g, ResidualShape s, uint32_t capacity) {
    need(capacity && capacity <= 2048, "prefill residual allocation capacity");
    const auto ws = widths(s); std::array<GpuView, 6> v{};
    try {
        for (size_t i = 0; i < ws.size(); ++i) {
            const auto bytes = checked_mul(checked_mul(ws[i], capacity), 4);
            need(bytes <= (256ull << 20), "prefill residual workspace budget");
            v[i] = g.zeros(bytes, "committed prefill residual rows");
        }
    } catch (...) { for (auto view : v) if (view.owner) g.release(view); throw; }
    return {{v[0], v[1], v[2], v[3], v[4], v[5]}, capacity};
}
void release_prefill_residual_scratch(MetalGraph& g, const PrefillResidualScratch& s) { for (auto v : s.rows.views()) g.release(v); }
std::array<GpuView, 9> PrefillLayerScratch::views() const {
    const auto r = residual.rows.views(); return {r[0], r[1], r[2], r[3], r[4], r[5], intermediate_streams, attention_output, ffn_output};
}
PrefillLayerScratch allocate_prefill_layer_scratch(MetalGraph& g, ResidualShape shape, uint32_t capacity) {
    PrefillLayerScratch out{};
    try {
        out.residual = allocate_prefill_residual_scratch(g, shape, capacity);
        out.intermediate_streams = g.zeros(checked_mul(checked_mul(checked_mul(capacity, shape.branches), shape.channels), 4), "prefill intermediate streams");
        out.attention_output = g.zeros(checked_mul(checked_mul(capacity, shape.channels), 4), "prefill attention output rows");
        out.ffn_output = g.zeros(out.attention_output.bytes, "prefill FFN output rows");
    } catch (...) { for (auto view : out.views()) if (view.owner) g.release(view); throw; }
    return out;
}
void release_prefill_layer_scratch(MetalGraph& g, const PrefillLayerScratch& s) { for (auto view : s.views()) g.release(view); }
void prepare_prefill_residual(MetalGraph& g, const ResidualWeights& w, ResidualShape s) {
    weights(g, w, s); g.prepare_gdn_chunk(); g.prepare_dense_mm_q8(g.prefill_weight_layout(w.up.view));
    if (w.inject.encoding == 8 && g.prefill_dense_nax() && (s.branches * 4) % 16 == 0 && (s.branches * s.channels) % 64 == 0)
        g.prepare_dense_mm_q8(g.prefill_weight_layout(w.inject.view));
    if (batch_hc_down(g, w, s, 512)) g.prepare_dense_mm_q8(PrefillWeightLayout::gguf);
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    g.prepare_hc_down_window();
#endif
}
uint64_t prefill_residual_dispatches(const MetalGraph& g, const ResidualWeights& w, ResidualShape s, uint32_t n) {
    static_cast<void>(widths(s)); need(n && n <= 2048, "prefill residual dispatch row count");
    const uint32_t cap = max_hc_rows(g);
    if (n > cap) {
        uint64_t total = 0;
        for (uint32_t offset = 0; offset < n; offset += cap)
            total = checked_add(total, prefill_residual_dispatches(g, w, s, std::min(cap, n - offset)));
        return total;
    }
    const auto down = batch_hc_down_window(g, w, s, n) ? g.dense_mm_q8_dispatches(s.branches * s.channels)
        : (batch_hc_down(g, w, s, n) ? 1 : uint64_t(n) * g.gemv_dispatches(w.down.encoding, s.branches * s.channels, s.rank));
    const auto inject = (batch_rows(g, s, n) && w.inject.encoding == 0) ? 2
        : (batch_q8_inject(g, w, s, n) ? g.dense_mm_q8_dispatches(s.branches * s.channels)
            : uint64_t(n) * g.gemv_dispatches(w.inject.encoding, s.branches * s.channels, s.branches));
    return 2 + g.dense_mm_q8_dispatches(s.rank) + down + inject + (batch_rows(g, s, n) ? 1 : n);
}
uint64_t prefill_residual_write_dispatches(const MetalGraph& g, ResidualShape s, uint32_t n) {
    static_cast<void>(widths(s)); need(n && n <= 2048, "prefill residual write row count");
    const uint32_t cap = max_hc_rows(g);
    if (n > cap) {
        uint64_t total = 0;
        for (uint32_t offset = 0; offset < n; offset += cap)
            total = checked_add(total, prefill_residual_write_dispatches(g, s, std::min(cap, n - offset)));
        return total;
    }
    return batch_rows(g, s, n) ? 1 : n;
}
GpuRowsF32 prefill_residual_input(const PrefillResidualScratch& s, ResidualShape shape, uint32_t n) {
    scratch_check(s, shape, n); return tight(s.rows.input, n, shape.channels);
}
void encode_prefill_residual(MetalGraph& g, const ResidualWeights& w, const PrefillResidualScratch& s,
    GpuRowsF32 streams, ResidualShape shape, float epsilon) {
    const uint32_t cap = max_hc_rows(g);
    if (streams.rows > cap) {
        try {
            weights(g, w, shape); scratch_check(s, shape, streams.rows);
            const auto n = streams.rows, width = shape.branches * shape.channels;
            rows_check(streams, n, width);
            const auto ws = widths(shape); const auto all_v = s.rows.views();
            for (uint32_t offset = 0; offset < n; offset += cap) {
                const uint32_t chunk = std::min(cap, n - offset);
                std::array<GpuView, 6> sub_v{};
                for (size_t i = 0; i < ws.size(); ++i)
                    sub_v[i] = g.slice(all_v[i], uint64_t(offset) * ws[i] * 4, uint64_t(chunk) * ws[i] * 4);
                const PrefillResidualScratch sub_s{{sub_v[0], sub_v[1], sub_v[2], sub_v[3], sub_v[4], sub_v[5]}, chunk};
                const auto sub_streams_view = g.slice(streams.view, uint64_t(offset) * streams.stride_bytes,
                    checked_add(checked_mul(chunk - 1, streams.stride_bytes), uint64_t(width) * 4));
                encode_prefill_residual(g, w, sub_s, {sub_streams_view, chunk, width, streams.stride_bytes}, shape, epsilon);
            }
        } catch (...) { abort(g); throw; }
        return;
    }
    try {
        weights(g, w, shape); scratch_check(s, shape, streams.rows); const auto n = streams.rows, width = shape.branches * shape.channels;
        rows_check(streams, n, width); need(std::isfinite(epsilon) && epsilon > 0, "prefill residual RMS epsilon");
        const std::array reads{streams.view, w.gains, w.down.view, w.up.view, w.inject.view}; g.validate_disjoint(reads, s.rows.views());
        const auto normalized = tight(s.rows.normalized, n, width);

        TOKSPUR_OPERATOR_CHECKPOINT(g,"unclassified","before_hc",{n,width,shape.rank,shape.branches});
        g.rms_rows(streams, w.gains, normalized, shape.channels, epsilon, false);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","norm",{n,width,shape.channels,shape.branches});
        encode_prefill_residual_after_rms(g, w, s, normalized, shape, n);
    } catch (...) { abort(g); throw; }
}
void encode_prefill_residual_write(MetalGraph& g, const PrefillResidualScratch& s, GpuRowsF32 streams,
    GpuRowsF32 values, GpuRowsF32 output, ResidualShape shape) {
    const uint32_t cap = max_hc_rows(g);
    if (streams.rows > cap) {
        try {
            const auto n = streams.rows, width = shape.branches * shape.channels; scratch_check(s, shape, n);
            rows_check(streams, n, width); rows_check(values, n, shape.channels); rows_check(output, n, width);
            const auto ws = widths(shape); const auto all_v = s.rows.views();
            for (uint32_t offset = 0; offset < n; offset += cap) {
                const uint32_t chunk = std::min(cap, n - offset);
                std::array<GpuView, 6> sub_v{};
                for (size_t i = 0; i < ws.size(); ++i)
                    sub_v[i] = g.slice(all_v[i], uint64_t(offset) * ws[i] * 4, uint64_t(chunk) * ws[i] * 4);
                const PrefillResidualScratch sub_s{{sub_v[0], sub_v[1], sub_v[2], sub_v[3], sub_v[4], sub_v[5]}, chunk};
                const auto sub_streams = g.slice(streams.view, uint64_t(offset) * streams.stride_bytes,
                    checked_add(checked_mul(chunk - 1, streams.stride_bytes), uint64_t(width) * 4));
                const auto sub_vals = g.slice(values.view, uint64_t(offset) * values.stride_bytes,
                    checked_add(checked_mul(chunk - 1, values.stride_bytes), uint64_t(shape.channels) * 4));
                const auto sub_out = g.slice(output.view, uint64_t(offset) * output.stride_bytes,
                    checked_add(checked_mul(chunk - 1, output.stride_bytes), uint64_t(width) * 4));
                encode_prefill_residual_write(g, sub_s, {sub_streams, chunk, width, streams.stride_bytes},
                    {sub_vals, chunk, shape.channels, values.stride_bytes},
                    {sub_out, chunk, width, output.stride_bytes}, shape);
            }
        } catch (...) { abort(g); throw; }
        return;
    }
    try {
        const auto n = streams.rows, width = shape.branches * shape.channels; scratch_check(s, shape, n);
        rows_check(streams, n, width); rows_check(values, n, shape.channels); rows_check(output, n, width);
        const auto views = s.rows.views(); const std::array reads{streams.view, values.view, s.rows.scales}; const std::array writes{output.view};
        g.validate_disjoint(reads, writes); g.validate_disjoint(views, writes);

        TOKSPUR_OPERATOR_CHECKPOINT(g,"unclassified","before_hc_write",{n,width,shape.channels,shape.branches});
#ifdef TOKSPUR_PRIVATE_HC_ROWS
        if (batch_rows(g, shape, n)) {
            need(streams.stride_bytes == uint64_t(width) * 4 && values.stride_bytes == uint64_t(shape.channels) * 4 && output.stride_bytes == uint64_t(width) * 4, "batched HC write requires tight rows");
            g.hc_write_rows(g.slice(streams.view, 0, uint64_t(n) * width * 4), g.slice(values.view, 0, uint64_t(n) * shape.channels * 4), g.slice(s.rows.scales, 0, uint64_t(n) * shape.branches * 4), g.slice(output.view, 0, uint64_t(n) * width * 4), n);
        } else
#endif
        for (uint32_t t = 0; t < n; ++t) g.residual_write(row(g, streams, t), row(g, values, t),
            g.slice(s.rows.scales, uint64_t(t) * shape.branches * 4, uint64_t(shape.branches) * 4), row(g, output, t));
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","write",{n,width,shape.channels,shape.branches});
    } catch (...) { abort(g); throw; }
}
void encode_prefill_residual_write_and_read(MetalGraph& g, const ResidualWeights& w, const PrefillResidualScratch& s,
    GpuRowsF32 streams, GpuRowsF32 values, GpuRowsF32 intermediate, ResidualShape shape, float epsilon) {
    if (!can_fuse_prefill_hc_consumer(g, shape, streams.rows)) {
        encode_prefill_residual_write(g, s, streams, values, intermediate, shape);
        encode_prefill_residual(g, w, s, intermediate, shape, epsilon);
        return;
    }
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    try {
        weights(g, w, shape);
        const auto n = streams.rows, width = shape.branches * shape.channels;
        scratch_check(s, shape, n);
        rows_check(streams, n, width); rows_check(values, n, shape.channels); rows_check(intermediate, n, width);
        need(std::isfinite(epsilon) && epsilon > 0, "prefill residual RMS epsilon");
        need(streams.stride_bytes == uint64_t(width) * 4 && values.stride_bytes == uint64_t(shape.channels) * 4 &&
            intermediate.stride_bytes == uint64_t(width) * 4, "batched HC write+RMS requires tight rows");
        const auto views = s.rows.views();
        const std::array write_reads{streams.view, values.view, s.rows.scales};
        const std::array write_dst{intermediate.view};
        g.validate_disjoint(write_reads, write_dst);
        g.validate_disjoint(views, write_dst);
        const std::array read_reads{intermediate.view, w.gains, w.down.view, w.up.view, w.inject.view};
        g.validate_disjoint(read_reads, views);
        const auto normalized = tight(s.rows.normalized, n, width);

        TOKSPUR_OPERATOR_CHECKPOINT(g,"unclassified","before_hc_write_and_norm",{n,width,shape.channels,shape.branches});
        g.hc_write_rms_rows(
            g.slice(streams.view, 0, uint64_t(n) * width * 4),
            g.slice(values.view, 0, uint64_t(n) * shape.channels * 4),
            g.slice(s.rows.scales, 0, uint64_t(n) * shape.branches * 4),
            w.gains,
            g.slice(intermediate.view, 0, uint64_t(n) * width * 4),
            g.slice(normalized.view, 0, uint64_t(n) * width * 4),
            n, epsilon);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","write_and_norm",{n,width,shape.channels,shape.branches});
        encode_prefill_residual_after_rms(g, w, s, normalized, shape, n);
    } catch (...) { abort(g); throw; }
#endif
}
}
