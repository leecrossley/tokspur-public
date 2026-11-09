#include "tokspur/verify_nax.hpp"
#include "tokspur/gdn_chunk.hpp"
#include "tokspur/gdn_core_tile.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/grouped_mm.hpp"
#include "tokspur/operator_checkpoints.hpp"
#ifdef TOKSPUR_PRIVATE_QSA_PREFILL
#include "tokspur/qsa_prefill_plan.hpp"
#endif

#if !defined(TOKSPUR_PRIVATE_GROUPED_MM_F32) || !defined(TOKSPUR_PRIVATE_GDN_CHUNK)
#error "verify_nax requires the carrier NAX and GDN chunk build"
#endif

namespace tokspur {
namespace {
void need(bool ok, const char* why) {
    if (!ok) throw FormatError(why);
}

GpuView row(MetalGraph& g, GpuView view, uint64_t width, uint32_t step) {
    return g.slice(view, checked_mul(checked_mul(step, width), 4), checked_mul(width, 4));
}

GpuRowsF32 tight(MetalGraph& g, GpuView view, uint32_t rows, uint32_t columns) {
    return {g.slice(view, 0, checked_mul(checked_mul(rows, columns), 4)), rows, columns, checked_mul(columns, 4)};
}

bool verify_fused_hc_q8_supported(const MetalGraph& g, GpuMatrix matrix, uint32_t columns, uint32_t rows) {
    if (!g.verify_hc_gdn_fused_enabled() || matrix.encoding != 8 ||
        columns < 32 || columns > 65536 || columns % 32 != 0 || rows < 1 || rows > 65536) {
        return false;
    }
    if (!((columns == 10240 && rows == 320) || (columns == 320 && rows == 10240))) {
        return false;
    }
    const auto layout = g.prefill_weight_layout(matrix.view);
    if (layout == PrefillWeightLayout::gguf) return true;
    return layout == PrefillWeightLayout::q8_lanes_v1 && (columns < 512 || columns % 256 == 0);
}

bool verify_fused_hc_shape_supported(const MetalGraph& g, ResidualShape shape) {
    return g.verify_hc_gdn_fused_enabled() &&
        shape.branches >= 1 && shape.branches <= 16 &&
        shape.channels >= 32 && shape.branches * shape.channels <= 65536 &&
        (shape.branches * shape.channels) % 32 == 0;
}

bool verify_fused_hc_inject_supported(const MetalGraph& g, const ResidualWeights& weights, ResidualShape shape) {
    return verify_fused_hc_shape_supported(g, shape) &&
        weights.inject.encoding == 0 &&
        g.prefill_weight_layout(weights.inject.view) == PrefillWeightLayout::gguf;
}

bool verify_fused_gdn_gates_supported(const MetalGraph& g, const GdnWeights& weights, GdnShape shape) {
    return g.verify_hc_gdn_fused_enabled() &&
        shape.channels >= 32 && shape.channels <= 65536 && shape.channels % 32 == 0 &&
        shape.value_heads >= 1 && shape.value_heads <= 256 &&
        weights.alpha.encoding == 0 && weights.beta.encoding == 0 &&
        g.prefill_weight_layout(weights.alpha.view) == PrefillWeightLayout::gguf &&
        g.prefill_weight_layout(weights.beta.view) == PrefillWeightLayout::gguf;
}
} // namespace

bool verify_nax_dense_q8_supported(const MetalGraph& g, GpuMatrix matrix, uint32_t columns, uint32_t rows) {
    if (matrix.encoding != 8 || columns < 32 || columns > 65536 || columns % 32 != 0 || rows < 1 || rows > 524288) {
        return false;
    }
    if (!g.verify_nax_m8() && ((columns == 10240 && rows == 320) || (columns == 320 && rows == 10240))) {
        return false;
    }
    const auto layout = g.prefill_weight_layout(matrix.view);
    if (layout == PrefillWeightLayout::gguf) return true;
    return layout == PrefillWeightLayout::q8_lanes_v1 && (columns < 512 || columns % 256 == 0);
}

bool verify_nax_bf16_supported(const MetalGraph& g, GpuMatrix matrix, uint32_t columns, uint32_t rows) {
    if (matrix.encoding != 30 || columns < 32 || columns > 65536 || columns % 32 != 0 || rows < 1 || rows > 524288) {
        return false;
    }
    return g.prefill_weight_layout(matrix.view) == PrefillWeightLayout::gguf;
}

bool verify_nax_hc_rows_supported(const MetalGraph& g, ResidualShape shape) {
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    return shape.branches == 4 && shape.channels == 2560 && g.kernel_policy()[0] == 16 && !g.hc_q8_policy().splits;
#else
    static_cast<void>(g);
    static_cast<void>(shape);
    return false;
#endif
}

bool verify_nax_router_window_supported(const MetalGraph& g, const MoeWeights& weights, MoeShape shape) {
    static_cast<void>(g);
    static_cast<void>(weights);
    static_cast<void>(shape);
    return false;
}

bool verify_nax_grouped_moe_supported(const MoeWeights& weights, MoeShape shape, uint32_t steps) {
    const auto ok_mat = [](uint32_t enc, uint32_t cols) {
        if (enc == 12 || enc == 13) return cols >= 256 && cols % 256 == 0;
        if (enc == 7 || enc == 8) return cols >= 32 && cols % 32 == 0;
        return false;
    };
    return shape.experts >= 1 && shape.experts <= 512 &&
        shape.selected >= 1 && shape.selected <= 16 && shape.selected <= shape.experts &&
        steps >= 2 && steps <= 8 &&
        ok_mat(weights.gate.encoding, shape.channels) &&
        ok_mat(weights.up.encoding, shape.channels) &&
        ok_mat(weights.down.encoding, shape.hidden);
}

uint64_t verify_nax_projection_dispatches(const MetalGraph& g, GpuMatrix matrix, uint32_t columns, uint32_t rows, uint32_t steps) {
    if (verify_nax_dense_q8_supported(g, matrix, columns, rows) ||
        verify_nax_bf16_supported(g, matrix, columns, rows) ||
        verify_fused_hc_q8_supported(g, matrix, columns, rows)) {
        return 1ull;
    }
    if (matrix.encoding == 8) {
        return g.q8_many_chunks(steps);
    }
    return uint64_t(steps) * g.gemv_dispatches(matrix.encoding, columns, rows);
}

void encode_verify_nax_projection(MetalGraph& g, GpuMatrix matrix, GpuView input, GpuView output, uint32_t columns, uint32_t rows, uint32_t steps) {
    if (verify_nax_dense_q8_supported(g, matrix, columns, rows)) {
        g.verify_nax_dense_q8(matrix.view, tight(g, input, steps, columns), tight(g, output, steps, rows), false);
        return;
    }
    if (verify_nax_bf16_supported(g, matrix, columns, rows)) {
        g.verify_nax_bf16(matrix.view, tight(g, input, steps, columns), tight(g, output, steps, rows));
        return;
    }
    if (verify_fused_hc_q8_supported(g, matrix, columns, rows)) {
        g.verify_fused_hc_q8(matrix.view, tight(g, input, steps, columns), tight(g, output, steps, rows), false, 1.0f);
        return;
    }
    if (matrix.encoding == 8) {
        uint32_t at = 0;
        while (at < steps) {
            const uint32_t n = g.q8_many_width(steps - at);
            g.q8_matvec_many(matrix.view, g.slice(input, uint64_t(at) * columns * 4, uint64_t(n) * columns * 4),
                g.slice(output, uint64_t(at) * rows * 4, uint64_t(n) * rows * 4), columns, rows, n);
            at += n;
        }
        return;
    }
    for (uint32_t t = 0; t < steps; ++t) {
        g.gemv(matrix.encoding, matrix.view, row(g, input, columns, t), row(g, output, rows, t), columns, rows);
    }
}

namespace {
uint64_t verify_nax_residual_after_rms_dispatches(const MetalGraph& g, const ResidualWeights& weights, ResidualShape shape, uint32_t steps) {
    const uint32_t width = shape.branches * shape.channels;
    const uint64_t down_and_act = (verify_nax_dense_q8_supported(g, weights.down, width, shape.rank) ||
        verify_fused_hc_q8_supported(g, weights.down, width, shape.rank))
        ? 1ull
        : (verify_nax_projection_dispatches(g, weights.down, width, shape.rank, steps) + 1ull);
    uint64_t count = down_and_act + verify_nax_projection_dispatches(g, weights.up, shape.rank, width, steps);
    if (verify_fused_hc_inject_supported(g, weights, shape)) {
        count += 2ull;
    } else if (verify_nax_hc_rows_supported(g, shape)) {
        const uint64_t inject = weights.inject.encoding == 0 ? 2ull
            : uint64_t(steps) * g.gemv_dispatches(weights.inject.encoding, width, shape.branches);
        count += inject + 1ull;
    } else {
        count += uint64_t(steps) * (g.gemv_dispatches(weights.inject.encoding, width, shape.branches) + 1ull);
    }
    return count;
}

void encode_verify_nax_residual_after_rms(MetalGraph& g, const ResidualWeights& weights, const ResidualScratch& scratch,
    ResidualShape shape, uint32_t steps) {
    const uint32_t width = shape.branches * shape.channels;
    if (verify_nax_dense_q8_supported(g, weights.down, width, shape.rank)) {
        g.verify_nax_dense_q8(weights.down.view, tight(g, scratch.normalized, steps, width),
            tight(g, scratch.bottleneck, steps, shape.rank), false, 1.0f / float(shape.branches));
    } else if (verify_fused_hc_q8_supported(g, weights.down, width, shape.rank)) {
        g.verify_fused_hc_q8(weights.down.view, tight(g, scratch.normalized, steps, width),
            tight(g, scratch.bottleneck, steps, shape.rank), true, 1.0f / float(shape.branches));
    } else {
        encode_verify_nax_projection(g, weights.down, scratch.normalized, scratch.bottleneck, width, shape.rank, steps);
        const auto active_bn = g.slice(scratch.bottleneck, 0, uint64_t(steps) * shape.rank * 4);
        g.activate(active_bn, active_bn, Activation::silu, 1.0f / shape.branches);
    }
    encode_verify_nax_projection(g, weights.up, scratch.bottleneck, scratch.scores, shape.rank, width, steps);
    if (verify_fused_hc_inject_supported(g, weights, shape)) {
        const auto norm_rows = g.slice(scratch.normalized, 0, uint64_t(steps) * width * 4);
        const auto score_rows = g.slice(scratch.scores, 0, uint64_t(steps) * width * 4);
        const auto inject_rows = g.slice(scratch.inject_logits, 0, uint64_t(steps) * shape.branches * 4);
        const auto scale_rows = g.slice(scratch.scales, 0, uint64_t(steps) * shape.branches * 4);
        const auto out_rows = g.slice(scratch.input, 0, uint64_t(steps) * shape.channels * 4);
        g.verify_fused_hc_inject_read_rows(weights.inject.view, norm_rows, score_rows, inject_rows,
            out_rows, scale_rows, shape.branches, shape.channels, steps);
        TOKSPUR_OPERATOR_CHECKPOINT(g, "hc", "verify_nax_read", {steps, width, shape.rank, shape.branches});
        return;
    }
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    if (verify_nax_hc_rows_supported(g, shape)) {
        const auto norm_rows = g.slice(scratch.normalized, 0, uint64_t(steps) * width * 4);
        const auto score_rows = g.slice(scratch.scores, 0, uint64_t(steps) * width * 4);
        const auto inject_rows = g.slice(scratch.inject_logits, 0, uint64_t(steps) * shape.branches * 4);
        if (weights.inject.encoding == 0) {
            g.prefill_hc_inject_rows(weights.inject.view, norm_rows, inject_rows, steps);
        } else {
            for (uint32_t t = 0; t < steps; ++t) {
                g.gemv(weights.inject.encoding, weights.inject.view, row(g, scratch.normalized, width, t),
                    row(g, scratch.inject_logits, shape.branches, t), width, shape.branches);
            }
        }
        g.hc_read_rows(norm_rows, score_rows, inject_rows,
            g.slice(scratch.input, 0, uint64_t(steps) * shape.channels * 4),
            g.slice(scratch.scales, 0, uint64_t(steps) * shape.branches * 4), steps);
        TOKSPUR_OPERATOR_CHECKPOINT(g, "hc", "verify_nax_read", {steps, width, shape.rank, shape.branches});
        return;
    }
#endif
    for (uint32_t t = 0; t < steps; ++t) {
        const auto norm_t = row(g, scratch.normalized, width, t);
        const auto score_t = row(g, scratch.scores, width, t);
        const auto inj_t = row(g, scratch.inject_logits, shape.branches, t);
        g.gemv(weights.inject.encoding, weights.inject.view, norm_t, inj_t, width, shape.branches);
        g.residual_read(norm_t, score_t, inj_t, row(g, scratch.input, shape.channels, t),
            row(g, scratch.scales, shape.branches, t), shape.branches, shape.channels);
    }
    TOKSPUR_OPERATOR_CHECKPOINT(g, "hc", "verify_nax_read", {steps, width, shape.rank, shape.branches});
}

bool verify_nax_gdn_gates_supported(const GdnWeights& weights, GdnShape shape) {
    return shape.channels >= 32 && shape.channels <= 65536 && shape.channels % 4 == 0 &&
        shape.value_heads >= 1 && shape.value_heads <= 512 &&
        weights.alpha.encoding == 0 && weights.beta.encoding == 0;
}
} // namespace

uint64_t verify_nax_residual_read_dispatches(const MetalGraph& g, const ResidualWeights& weights, ResidualShape shape, uint32_t steps) {
    return 1ull + verify_nax_residual_after_rms_dispatches(g, weights, shape, steps);
}

void encode_verify_nax_residual_read(MetalGraph& g, const ResidualWeights& weights, const ResidualScratch& scratch,
    GpuView input_streams, ResidualShape shape, uint32_t steps, float epsilon) {
    need(g.verify_nax_active() && steps >= 2 && steps <= 8, "verify NAX residual read requires active 2..8-row verify scope");
    const uint32_t width = shape.branches * shape.channels;
    g.rms_rows(tight(g, input_streams, steps, width), weights.gains, tight(g, scratch.normalized, steps, width), shape.channels, epsilon, weights.gains.bytes == uint64_t(shape.channels) * 4);
    encode_verify_nax_residual_after_rms(g, weights, scratch, shape, steps);
}

uint64_t verify_nax_residual_write_dispatches(const MetalGraph& g, ResidualShape shape, uint32_t steps) {
    return (verify_fused_hc_shape_supported(g, shape) || verify_nax_hc_rows_supported(g, shape)) ? 1ull : uint64_t(steps);
}

void encode_verify_nax_residual_write(MetalGraph& g, GpuView streams, GpuView values, GpuView scales, GpuView destination,
    ResidualShape shape, uint32_t steps) {
    need(g.verify_nax_active() && steps >= 2 && steps <= 8, "verify NAX residual write requires active 2..8-row verify scope");
    const uint32_t width = shape.branches * shape.channels;
    if (verify_fused_hc_shape_supported(g, shape)) {
        g.verify_fused_hc_write_rows(g.slice(streams, 0, uint64_t(steps) * width * 4),
            g.slice(values, 0, uint64_t(steps) * shape.channels * 4),
            g.slice(scales, 0, uint64_t(steps) * shape.branches * 4),
            g.slice(destination, 0, uint64_t(steps) * width * 4),
            shape.branches, shape.channels, steps);
        TOKSPUR_OPERATOR_CHECKPOINT(g, "hc", "verify_nax_write", {steps, width, shape.channels, shape.branches});
        return;
    }
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    if (verify_nax_hc_rows_supported(g, shape)) {
        g.hc_write_rows(g.slice(streams, 0, uint64_t(steps) * width * 4),
            g.slice(values, 0, uint64_t(steps) * shape.channels * 4),
            g.slice(scales, 0, uint64_t(steps) * shape.branches * 4),
            g.slice(destination, 0, uint64_t(steps) * width * 4), steps);
        TOKSPUR_OPERATOR_CHECKPOINT(g, "hc", "verify_nax_write", {steps, width, shape.channels, shape.branches});
        return;
    }
#endif
    for (uint32_t t = 0; t < steps; ++t) {
        g.residual_write(row(g, streams, width, t), row(g, values, shape.channels, t),
            row(g, scales, shape.branches, t), row(g, destination, width, t));
    }
    TOKSPUR_OPERATOR_CHECKPOINT(g, "hc", "verify_nax_write", {steps, width, shape.channels, shape.branches});
}

bool verify_nax_hc_write_rms_supported(const MetalGraph& g, const ResidualWeights& next_read, ResidualShape shape) {
    const uint64_t width_bytes = uint64_t(shape.branches) * shape.channels * 4;
    const uint64_t chan_bytes = uint64_t(shape.channels) * 4;
    if (verify_fused_hc_shape_supported(g, shape) &&
        (next_read.gains.bytes == chan_bytes || next_read.gains.bytes == width_bytes)) {
        return true;
    }
    return verify_nax_hc_rows_supported(g, shape) && next_read.gains.bytes == width_bytes;
}

uint64_t verify_nax_residual_write_read_dispatches(const MetalGraph& g, const ResidualWeights& next_read, ResidualShape shape, uint32_t steps) {
    if (verify_nax_hc_write_rms_supported(g, next_read, shape)) {
        return 1ull + verify_nax_residual_after_rms_dispatches(g, next_read, shape, steps);
    }
    return verify_nax_residual_write_dispatches(g, shape, steps) +
        verify_nax_residual_read_dispatches(g, next_read, shape, steps);
}

void encode_verify_nax_residual_write_read(MetalGraph& g, const ResidualWeights& next_read, const ResidualScratch& scratch,
    GpuView streams, GpuView values, GpuView scales, GpuView destination, ResidualShape shape, uint32_t steps, float epsilon) {
    need(g.verify_nax_active() && steps >= 2 && steps <= 8, "verify NAX residual write+read requires active 2..8-row verify scope");
    const uint32_t width = shape.branches * shape.channels;
    const uint64_t width_bytes = uint64_t(width) * 4;
    const uint64_t chan_bytes = uint64_t(shape.channels) * 4;
    if (verify_fused_hc_shape_supported(g, shape) &&
        (next_read.gains.bytes == chan_bytes || next_read.gains.bytes == width_bytes)) {
        g.verify_fused_hc_write_rms_rows(
            g.slice(streams, 0, uint64_t(steps) * width * 4),
            g.slice(values, 0, uint64_t(steps) * shape.channels * 4),
            g.slice(scales, 0, uint64_t(steps) * shape.branches * 4),
            next_read.gains,
            g.slice(destination, 0, uint64_t(steps) * width * 4),
            g.slice(scratch.normalized, 0, uint64_t(steps) * width * 4),
            shape.branches, shape.channels, steps, epsilon, next_read.gains.bytes == chan_bytes);
        encode_verify_nax_residual_after_rms(g, next_read, scratch, shape, steps);
        return;
    }
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    if (verify_nax_hc_rows_supported(g, shape) && next_read.gains.bytes == width_bytes) {
        g.hc_write_rms_rows(
            g.slice(streams, 0, uint64_t(steps) * width * 4),
            g.slice(values, 0, uint64_t(steps) * shape.channels * 4),
            g.slice(scales, 0, uint64_t(steps) * shape.branches * 4),
            next_read.gains,
            g.slice(destination, 0, uint64_t(steps) * width * 4),
            g.slice(scratch.normalized, 0, uint64_t(steps) * width * 4),
            steps, epsilon);
        encode_verify_nax_residual_after_rms(g, next_read, scratch, shape, steps);
        return;
    }
#endif
    encode_verify_nax_residual_write(g, streams, values, scales, destination, shape, steps);
    encode_verify_nax_residual_read(g, next_read, scratch, destination, shape, steps, epsilon);
}

uint64_t verify_nax_moe_dispatches(const MetalGraph& g, const MoeWeights& weights, MoeShape shape, uint32_t steps) {
    const bool router_win = verify_nax_router_window_supported(g, weights, shape);
    const uint64_t router_part = router_win ? 2ull
        : uint64_t(steps) * (g.gemv_dispatches(weights.router.encoding, shape.channels, shape.experts) + 1ull);
    const uint64_t routed_part = verify_nax_grouped_moe_supported(weights, shape, steps) ? 3ull : uint64_t(steps) * 5ull;
    const bool fuse_shared = verify_nax_dense_q8_supported(g, weights.shared_gate, shape.channels, shape.hidden) &&
        verify_nax_dense_q8_supported(g, weights.shared_up, shape.channels, shape.hidden);
    const bool single_pass_shared = fuse_shared &&
        g.prefill_weight_layout(weights.shared_gate.view) == g.prefill_weight_layout(weights.shared_up.view);
    const uint64_t shared_part = fuse_shared
        ? ((single_pass_shared ? 1ull : 2ull) + verify_nax_projection_dispatches(g, weights.shared_down, shape.hidden, shape.channels, steps))
        : (verify_nax_projection_dispatches(g, weights.shared_gate, shape.channels, shape.hidden, steps) +
           verify_nax_projection_dispatches(g, weights.shared_up, shape.channels, shape.hidden, steps) +
           2ull +
           verify_nax_projection_dispatches(g, weights.shared_down, shape.hidden, shape.channels, steps));
    const uint64_t mix_part = router_win ? 2ull
        : uint64_t(steps) * (g.gemv_dispatches(weights.shared_router.encoding, shape.channels, 1) + 1ull);
    return router_part + routed_part + shared_part + mix_part;
}

void encode_verify_nax_moe(MetalGraph& g, const MoeWeights& weights, const MoeScratch& scratch,
    const VerifyNaxMoeScratch& nax_moe, GpuView input, GpuView output, MoeShape shape,
    RouterMode mode, bool normalise, uint32_t steps) {
    static_cast<void>(nax_moe);
    need(g.verify_nax_active() && steps >= 2 && steps <= 8, "verify NAX MoE requires active 2..8-row verify scope");
    const bool router_win = verify_nax_router_window_supported(g, weights, shape);
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    if (router_win) {
        g.moe_router_f32_window(weights.router.view, tight(g, input, steps, shape.channels), tight(g, scratch.logits, steps, shape.experts));
        g.moe_route_rows(g.slice(scratch.logits, 0, uint64_t(steps) * shape.experts * 4),
            g.slice(scratch.indices, 0, uint64_t(steps) * shape.selected * 4),
            g.slice(scratch.weights, 0, uint64_t(steps) * shape.selected * 4),
            shape.experts, shape.selected, steps, mode, normalise);
    } else
#endif
    {
        for (uint32_t t = 0; t < steps; ++t) {
            const auto log_t = row(g, scratch.logits, shape.experts, t);
            g.gemv(weights.router.encoding, weights.router.view, row(g, input, shape.channels, t), log_t, shape.channels, shape.experts);
            g.route(log_t, row(g, scratch.indices, shape.selected, t), row(g, scratch.weights, shape.selected, t),
                shape.selected, mode, normalise);
        }
    }
    TOKSPUR_OPERATOR_CHECKPOINT(g, "moe", "verify_nax_router_topk", {steps, shape.channels, shape.experts, shape.selected});
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    for (uint32_t t = 0; t < steps; ++t) {
        g.route_lock_rows(row(g, scratch.logits, shape.experts, t),
            row(g, scratch.indices, shape.selected, t),
            row(g, scratch.weights, shape.selected, t),
            1, shape.channels, shape.experts, shape.selected, t);
    }
#endif

    if (verify_nax_grouped_moe_supported(weights, shape, steps)) {
        const auto ids = g.slice(scratch.indices, 0, uint64_t(steps) * shape.selected * 4);
        const auto in_rows = tight(g, input, steps, shape.channels);
        const auto gate_rows = tight(g, scratch.gate, steps * shape.selected, shape.hidden);
        const auto down_rows = tight(g, scratch.down, steps * shape.selected, shape.channels);
        if (is_planar_moe(g, weights, shape)) {
            g.planar_moe_routed_m8(weights.gate.view, weights.up.view, weights.down.view,
                ids, in_rows, gate_rows, down_rows, shape.experts, shape.selected);
        } else {
            const auto gate_layout = g.prefill_weight_layout(weights.gate.view);
            const auto up_layout = g.prefill_weight_layout(weights.up.view);
            const auto down_layout = g.prefill_weight_layout(weights.down.view);
            if (weights.gate.encoding == weights.up.encoding && gate_layout == up_layout) {
                g.verify_nax_moe_union(weights.gate.encoding, gate_layout, weights.gate.view, weights.up.view,
                    weights.down.encoding, down_layout, weights.down.view, ids, in_rows, gate_rows, down_rows,
                    shape.experts, shape.selected);
            } else {
                g.verify_nax_moe_routed(weights.gate.encoding, gate_layout, weights.gate.view,
                    ids, in_rows, gate_rows, shape.experts, shape.selected, true, false);
                g.verify_nax_moe_routed(weights.up.encoding, up_layout, weights.up.view,
                    ids, in_rows, gate_rows, shape.experts, shape.selected, true, true);
                g.verify_nax_moe_routed(weights.down.encoding, down_layout, weights.down.view,
                    ids, gate_rows, down_rows, shape.experts, shape.selected, false, false);
            }
        }
    } else {
        const uint64_t route_hidden = uint64_t(shape.hidden) * shape.selected;
        const uint64_t route_out = uint64_t(shape.channels) * shape.selected;
        for (uint32_t t = 0; t < steps; ++t) {
            const auto ids_t = row(g, scratch.indices, shape.selected, t);
            const auto x_t = row(g, input, shape.channels, t);
            const auto g_t = row(g, scratch.gate, route_hidden, t);
            const auto u_t = row(g, scratch.up, route_hidden, t);
            const auto d_t = row(g, scratch.down, route_out, t);
            g.routed_gemv(weights.gate.encoding, weights.gate.view, ids_t, x_t, g_t, shape.channels, shape.hidden, shape.experts, true);
            g.routed_gemv(weights.up.encoding, weights.up.view, ids_t, x_t, u_t, shape.channels, shape.hidden, shape.experts, true);
            g.activate(g_t, g_t, Activation::silu);
            g.multiply(g_t, u_t, g_t);
            g.routed_gemv(weights.down.encoding, weights.down.view, ids_t, g_t, d_t, shape.hidden, shape.channels, shape.experts, false);
        }
    }
    TOKSPUR_OPERATOR_CHECKPOINT(g, "moe", "verify_nax_routed_experts", {steps, shape.channels, shape.hidden, shape.selected});

    const bool fuse_shared = verify_nax_dense_q8_supported(g, weights.shared_gate, shape.channels, shape.hidden) &&
        verify_nax_dense_q8_supported(g, weights.shared_up, shape.channels, shape.hidden);
    if (fuse_shared) {
        const auto in_rows = tight(g, input, steps, shape.channels);
        const auto sg_rows = tight(g, scratch.shared_gate, steps, shape.hidden);
        if (g.prefill_weight_layout(weights.shared_gate.view) == g.prefill_weight_layout(weights.shared_up.view)) {
            g.verify_nax_dense_q8_swiglu(weights.shared_gate.view, weights.shared_up.view, in_rows, sg_rows);
        } else {
            g.verify_nax_dense_q8(weights.shared_gate.view, in_rows, sg_rows, false);
            g.verify_nax_dense_q8(weights.shared_up.view, in_rows, sg_rows, true);
        }
    } else {
        encode_verify_nax_projection(g, weights.shared_gate, input, scratch.shared_gate, shape.channels, shape.hidden, steps);
        encode_verify_nax_projection(g, weights.shared_up, input, scratch.shared_up, shape.channels, shape.hidden, steps);
        const auto sgv = g.slice(scratch.shared_gate, 0, uint64_t(steps) * shape.hidden * 4);
        const auto suv = g.slice(scratch.shared_up, 0, uint64_t(steps) * shape.hidden * 4);
        g.activate(sgv, sgv, Activation::silu);
        g.multiply(sgv, suv, sgv);
    }
    encode_verify_nax_projection(g, weights.shared_down, scratch.shared_gate, scratch.shared_down, shape.hidden, shape.channels, steps);
    TOKSPUR_OPERATOR_CHECKPOINT(g, "dense_q8", "verify_nax_shared_experts", {steps, shape.channels, shape.hidden, 8});

#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    if (router_win) {
        const auto shared_logits = g.slice(scratch.shared_logit, 0, uint64_t(steps) * 4);
        g.moe_shared_router_rows(weights.shared_router.view, tight(g, input, steps, shape.channels), shared_logits);
        g.moe_mixture_rows(g.slice(scratch.down, 0, uint64_t(steps) * shape.selected * shape.channels * 4),
            g.slice(scratch.weights, 0, uint64_t(steps) * shape.selected * 4),
            g.slice(scratch.shared_down, 0, uint64_t(steps) * shape.channels * 4),
            shared_logits, tight(g, output, steps, shape.channels), shape.selected);
        TOKSPUR_OPERATOR_CHECKPOINT(g, "moe", "verify_nax_shared_router_mixture", {steps, shape.channels, shape.selected, 0});
        return;
    }
#endif
    const uint64_t route_out = uint64_t(shape.channels) * shape.selected;
    for (uint32_t t = 0; t < steps; ++t) {
        const auto sl_t = row(g, scratch.shared_logit, 1, t);
        g.gemv(weights.shared_router.encoding, weights.shared_router.view, row(g, input, shape.channels, t), sl_t, shape.channels, 1);
        g.mixture(row(g, scratch.down, route_out, t), row(g, scratch.weights, shape.selected, t),
            row(g, scratch.shared_down, shape.channels, t), sl_t, row(g, output, shape.channels, t));
    }
    TOKSPUR_OPERATOR_CHECKPOINT(g, "moe", "verify_nax_shared_router_mixture", {steps, shape.channels, shape.selected, 0});
}

uint64_t verify_nax_gdn_core_dispatches(const MetalGraph& g, const GdnWeights& weights, GdnShape shape, uint32_t steps, bool ported) {
    const uint32_t raw = static_cast<uint32_t>(gdn_qkv_count(shape));
    const uint32_t value = shape.value_heads * shape.value_width;
    const uint64_t proj_in = verify_nax_projection_dispatches(g, weights.qkv, shape.channels, raw, steps) +
        verify_nax_projection_dispatches(g, weights.gate, shape.channels, value, steps);
    const uint64_t gates_f32 = ((g.verify_nax_m8() && verify_nax_gdn_gates_supported(weights, shape)) ||
        verify_fused_gdn_gates_supported(g, weights, shape)) ? 1ull : 2ull;
    const uint64_t conv = 1ull;
    const uint64_t core = ported ? 4ull : (gdn_core_batch_elementwise ? (4ull + 2ull * steps) : (1ull + 5ull * steps));
    const uint64_t proj_out = verify_nax_projection_dispatches(g, weights.output, value, shape.channels, steps);
    return proj_in + gates_f32 + conv + core + proj_out;
}

void encode_verify_nax_gdn_f32_gates(MetalGraph& g, const GdnWeights& weights, const GdnScratch& scratch,
    GpuView input, GdnShape shape, GdnPolicy policy, uint32_t steps) {
    const auto active_in = g.slice(input, 0, uint64_t(steps) * shape.channels * 4);
    const auto alpha_out = g.slice(scratch.alpha, 0, uint64_t(steps) * shape.value_heads * 4);
    const auto beta_out = g.slice(scratch.beta, 0, uint64_t(steps) * shape.value_heads * 4);
    if (g.verify_nax_m8() && verify_nax_gdn_gates_supported(weights, shape)) {
        g.verify_nax_gdn_gates_f32(weights.alpha.view, weights.beta.view, active_in, alpha_out, beta_out,
            shape.channels, shape.value_heads, steps);
        return;
    }
    if (verify_fused_gdn_gates_supported(g, weights, shape)) {
        g.verify_fused_gdn_alpha_beta_f32(weights.alpha.view, weights.beta.view,
            tight(g, input, steps, shape.channels),
            tight(g, scratch.alpha, steps, shape.value_heads),
            tight(g, scratch.beta, steps, shape.value_heads));
        return;
    }
    GdnChunkDispatch d{};
    d.op = GdnChunkOp::projection_f32;
    d.params.rows = steps;
    d.params.columns = shape.channels;
    d.params.outputs = shape.value_heads;
    d.params.heads = shape.value_heads;
    d.params.key_heads = shape.key_heads;
    d.params.key_width = shape.key_width;
    d.params.value_width = shape.value_width;
    d.params.query_scale = policy.query_scale;
    d.params.l2_epsilon = policy.l2_epsilon;
    d.params.rms_epsilon = policy.rms_epsilon;
    d.params.l2_mode = uint32_t(policy.l2_mode);
    d.params.input_stride = uint64_t(shape.channels) * 4;
    d.params.output_stride = uint64_t(shape.value_heads) * 4;
    d.params.auxiliary_stride = uint64_t(shape.value_heads) * shape.value_width * 4;
    d.params.gate_stride = uint64_t(shape.value_heads) * 4;
    d.buffers[0] = weights.alpha.view;
    d.buffers[1] = active_in;
    d.buffers[2] = alpha_out;
    g.gdn_chunk(d);
    d.buffers[0] = weights.beta.view;
    d.buffers[2] = beta_out;
    g.gdn_chunk(d);
}

uint64_t verify_nax_recurrent_layer_dispatches(const MetalGraph& g, const RecurrentWeights& weights,
    RecurrentShape shape, uint32_t steps, bool ported) {
    return verify_nax_residual_read_dispatches(g, weights.attention_read, shape.residual, steps) +
        verify_nax_gdn_core_dispatches(g, weights.attention, shape.gdn, steps, ported) +
        verify_nax_residual_write_read_dispatches(g, weights.ffn_read, shape.residual, steps) +
        verify_nax_moe_dispatches(g, weights.ffn, shape.moe, steps) +
        verify_nax_residual_write_dispatches(g, shape.residual, steps);
}

bool verify_nax_qsa_batch_supported(QsaShape a) {
#if defined(TOKSPUR_PRIVATE_QSA_PREFILL) && defined(TOKSPUR_PRIVATE_GDN_CHUNK)
    return a.compression == 4 && a.query_heads && a.query_heads <= 24 && a.kv_heads && a.kv_heads <= 24 &&
        a.head_width && a.head_width <= 256 && a.index_heads && a.index_heads <= 8 &&
        a.index_width && a.index_width <= 256 && a.rotary_width && a.rotary_width % 2 == 0 &&
        a.rotary_width <= std::min(a.head_width, a.index_width) &&
        qsa_block_capacity(a) <= 16384 && qsa_block_budget(a) && qsa_block_budget(a) <= 512;
#else
    (void)a;
    return false;
#endif
}

uint64_t verify_nax_attention_layer_dispatches(const MetalGraph& g, const AttentionLayerWeights& weights,
    AttentionLayerShape shape, uint32_t base_prefix, uint32_t steps) {
    const auto& a = shape.attention;
    const uint32_t c = a.channels, q = a.query_heads * a.head_width, k = a.kv_heads * a.head_width;
    const uint32_t iq = a.index_heads * a.index_width, ik = a.index_width;
    const uint64_t qkv_proj = verify_nax_projection_dispatches(g, weights.attention.query_gate, c, 2 * q, steps) +
        verify_nax_projection_dispatches(g, weights.attention.key, c, k, steps) +
        verify_nax_projection_dispatches(g, weights.attention.value, c, k, steps) +
        verify_nax_projection_dispatches(g, weights.attention.index_query, c, iq, steps) +
        verify_nax_projection_dispatches(g, weights.attention.index_key, c, ik, steps);
    uint64_t per_step = 0;
    const bool batch_ok = verify_nax_qsa_batch_supported(a);
    if (batch_ok) {
        const uint32_t first_pool = base_prefix / 4u;
        const uint32_t max_blocks = (base_prefix + steps) / 4u;
        const uint32_t pool_count = max_blocks - first_pool;
        per_step = 4ull + (pool_count ? 3ull : 0ull);
        if (max_blocks > qsa_block_budget(a)) {
            per_step += 3ull;
            for (uint32_t span = 512u; span < max_blocks; span *= 2u) ++per_step;
        }
    } else {
        for (uint32_t t = 0; t < steps; ++t) {
            per_step += (qsa_dispatches(a, base_prefix + t) - 8ull);
        }
    }
    const uint64_t sparse_core =
#ifdef TOKSPUR_PRIVATE_QSA_PREFILL
        (batch_ok && g.verify_nax_qsa_attention_supported(a)) ? qsa_nax_attention_dispatches(base_prefix, steps, a) :
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        g.qsa_sparse_enabled() ? 1ull :
#endif
        3ull;
    const uint64_t gate_mul = 1ull;
    const uint64_t out_proj = verify_nax_projection_dispatches(g, weights.attention.output, q, c, steps);
    return verify_nax_residual_read_dispatches(g, weights.attention_read, shape.residual, steps) +
        qkv_proj + per_step + sparse_core + gate_mul + out_proj +
        verify_nax_residual_write_read_dispatches(g, weights.ffn_read, shape.residual, steps) +
        verify_nax_moe_dispatches(g, weights.ffn, shape.moe, steps) +
        verify_nax_residual_write_dispatches(g, shape.residual, steps);
}

bool verify_nax_ple_supported(const MetalGraph& g, const PleWeights& weights, const RecurrentScratch& batch_scratch, PleShape shape) {
    const uint32_t width = shape.branches * shape.channels;
    return batch_scratch.residual.normalized.owner != 0 &&
        batch_scratch.residual.scores.owner != 0 &&
        batch_scratch.intermediate_streams.owner != 0 &&
        batch_scratch.residual.input.owner != 0 &&
        verify_nax_dense_q8_supported(g, weights.key, shape.embedding, width) &&
        verify_nax_dense_q8_supported(g, weights.value, shape.embedding, shape.channels);
}

uint64_t verify_nax_ple_dispatches(const MetalGraph& g, const PleWeights& weights, const RecurrentScratch& batch_scratch,
    PleShape shape, uint32_t steps) {
    need(verify_nax_ple_supported(g, weights, batch_scratch, shape), "verify NAX PLE unsupported");
    return 6ull + 2ull * steps;
}

void encode_verify_nax_ple(MetalGraph& g, const PleWeights& weights, const PleScratch& single_scratch,
    const RecurrentScratch& batch_scratch, GpuView embeddings, GpuView streams, GpuView current_memory,
    GpuView prefix_memory, GpuView next_memory, GpuView output, PleShape shape, uint32_t steps, float epsilon) {
    validate_ple_shape(shape);
    need(g.verify_nax_active() && steps >= 2 && steps <= 8 &&
        verify_nax_ple_supported(g, weights, batch_scratch, shape),
        "verify NAX PLE requires active 2..8-row verify scope and supported geometry");
    const uint32_t width = shape.branches * shape.channels, c = shape.channels, emb = shape.embedding;
    const auto emb_rows = tight(g, embeddings, steps, emb);
    const auto in_rows = tight(g, streams, steps, width);
    const auto buf0 = tight(g, batch_scratch.residual.normalized, steps, width);
    const auto buf1 = tight(g, batch_scratch.residual.scores, steps, width);
    const auto buf2 = tight(g, batch_scratch.intermediate_streams, steps, width);
    const auto val_rows = tight(g, batch_scratch.residual.input, steps, c);
    g.verify_nax_dense_q8(weights.key.view, emb_rows, buf0, false);
    g.verify_nax_dense_q8(weights.value.view, emb_rows, val_rows, false);
    g.rms_rows(buf0, weights.key_gains, buf1, c, epsilon, weights.key_gains.bytes == uint64_t(c) * 4);
    g.rms_rows(in_rows, weights.query_gains, buf0, c, epsilon, weights.query_gains.bytes == uint64_t(c) * 4);
    g.ple_gate_rows(buf0.view, buf1.view, val_rows.view, buf2.view, shape.branches, c, steps, shape.gate_transform);
    g.rms_rows(buf2, weights.conv_gains, buf0, c, epsilon, weights.conv_gains.bytes == uint64_t(c) * 4);
    auto previous = current_memory;
    for (uint32_t t = 0; t < steps; ++t) {
        const auto next = (t + 1 == steps) ? next_memory : row(g, prefix_memory, current_memory.bytes / 4, t);
        g.convolution(previous, row(g, buf0.view, width, t), weights.convolution, next, single_scratch.convolved, shape.kernel, shape.dilation);
        g.ple_write(row(g, streams, width, t), row(g, buf2.view, width, t), single_scratch.convolved, row(g, output, width, t));
        previous = next;
    }
}

bool verify_nax_final_supported(const MetalGraph& g, const MixerWeights& mixer, GpuMatrix output,
    const ResidualScratch& scratch, ResidualShape shape, uint32_t vocabulary) {
    static_cast<void>(mixer);
    return scratch.normalized.owner != 0 && scratch.bottleneck.owner != 0 && scratch.scores.owner != 0 &&
        mixer.down.encoding == 8 && mixer.up.encoding == 8 &&
        verify_nax_dense_q8_supported(g, output, shape.channels, vocabulary);
}

uint64_t verify_nax_final_dispatches(const MetalGraph& g, const MixerWeights& mixer, GpuMatrix output,
    const ResidualScratch& scratch, ResidualShape shape, uint32_t vocabulary, uint32_t steps) {
    need(verify_nax_final_supported(g, mixer, output, scratch, shape, vocabulary), "verify NAX final head unsupported");
    const uint32_t width = shape.branches * shape.channels;
    const uint64_t down_and_act = (verify_nax_dense_q8_supported(g, mixer.down, width, shape.rank) ||
        verify_fused_hc_q8_supported(g, mixer.down, width, shape.rank))
        ? 1ull
        : (verify_nax_projection_dispatches(g, mixer.down, width, shape.rank, steps) + 1ull);
    return 1ull + down_and_act +
        verify_nax_projection_dispatches(g, mixer.up, shape.rank, width, steps) +
        1ull + 1ull;
}

void encode_verify_nax_final_logits(MetalGraph& g, const MixerWeights& mixer, GpuMatrix output,
    const ResidualScratch& scratch, GpuView streams, GpuView mixed, GpuView logits,
    ResidualShape shape, uint32_t vocabulary, uint32_t steps, float epsilon) {
    need(g.verify_nax_active() && steps >= 2 && steps <= 8 &&
        verify_nax_final_supported(g, mixer, output, scratch, shape, vocabulary),
        "verify NAX final head requires active 2..8-row verify scope and supported geometry");
    const uint32_t width = shape.branches * shape.channels;
    g.rms_rows(tight(g, streams, steps, width), mixer.gains, tight(g, scratch.normalized, steps, width), shape.channels, epsilon, mixer.gains.bytes == uint64_t(shape.channels) * 4);
    if (verify_nax_dense_q8_supported(g, mixer.down, width, shape.rank)) {
        g.verify_nax_dense_q8(mixer.down.view, tight(g, scratch.normalized, steps, width),
            tight(g, scratch.bottleneck, steps, shape.rank), false, 1.0f / float(shape.branches));
    } else if (verify_fused_hc_q8_supported(g, mixer.down, width, shape.rank)) {
        g.verify_fused_hc_q8(mixer.down.view, tight(g, scratch.normalized, steps, width),
            tight(g, scratch.bottleneck, steps, shape.rank), true, 1.0f / float(shape.branches));
    } else {
        encode_verify_nax_projection(g, mixer.down, scratch.normalized, scratch.bottleneck, width, shape.rank, steps);
        const auto active_bn = g.slice(scratch.bottleneck, 0, uint64_t(steps) * shape.rank * 4);
        g.activate(active_bn, active_bn, Activation::silu, 1.0f / shape.branches);
    }
    encode_verify_nax_projection(g, mixer.up, scratch.bottleneck, scratch.scores, shape.rank, width, steps);
    if (verify_fused_hc_shape_supported(g, shape)) {
        g.verify_fused_hc_read_rows(g.slice(scratch.normalized, 0, uint64_t(steps) * width * 4),
            g.slice(scratch.scores, 0, uint64_t(steps) * width * 4),
            g.slice(mixed, 0, uint64_t(steps) * shape.channels * 4),
            shape.branches, shape.channels, steps);
    } else {
        g.verify_nax_gated_mix_rows(
            g.slice(scratch.normalized, 0, uint64_t(steps) * width * 4),
            g.slice(scratch.scores, 0, uint64_t(steps) * width * 4),
            g.slice(mixed, 0, uint64_t(steps) * shape.channels * 4),
            shape.branches, shape.channels, steps);
    }
    TOKSPUR_OPERATOR_CHECKPOINT(g, "hc", "verify_nax_final_mixer", {steps, width, shape.rank, shape.branches});
    g.verify_nax_dense_q8(output.view, tight(g, mixed, steps, shape.channels), tight(g, logits, steps, vocabulary), false);
    TOKSPUR_OPERATOR_CHECKPOINT(g, "dense_q8", "verify_nax_final_logits", {steps, shape.channels, vocabulary, 8});
}

} // namespace tokspur
