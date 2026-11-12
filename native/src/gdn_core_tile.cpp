#include "tokspur/gdn_core_tile.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/operator_checkpoints.hpp"
#include "tokspur/q8_many.hpp"
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
#include "tokspur/verify_nax.hpp"
#endif

#include <algorithm>

namespace tokspur {
namespace {
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
void validate(GdnShape s, StateTile t) {
    validate_gdn_shape(s); validate_state_tile(t);
    require(s.key_width == 128 && s.kernel == 4 && s.value_heads <= 48 && s.value_width <= 128 &&
        s.channels % 32 == 0 && (uint64_t(s.value_heads) * s.value_width) % 32 == 0,
        "unsupported GDN core tile shape");
}
std::array<uint64_t, 13> counts(GdnShape s) {
    const auto raw = gdn_qkv_count(s), qk = uint64_t(s.value_heads) * 128, value = uint64_t(s.value_heads) * s.value_width;
    return {raw, raw, qk, qk, value, value, s.value_heads, s.value_heads,
        s.value_heads, s.value_heads, value, value, value};
}
void bytes(GpuView view, uint64_t expected) { require(view.bytes == expected, "GDN core tile buffer geometry mismatch"); }
GpuView row(MetalGraph& g, GpuView v, uint64_t width, uint32_t step) {
    return g.slice(v, checked_mul(checked_mul(step, width), 4), checked_mul(width, 4));
}
GdnScratch token(MetalGraph& g, const GdnScratch& s, GdnShape shape, uint32_t step) {
    const auto n = counts(shape); const auto views = s.views(); std::array<GpuView, 13> v{};
    for (size_t i = 0; i < v.size(); ++i) v[i] = row(g, views[i], n[i], step);
    return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12]};
}
void state_bytes(const GdnState& state, GdnShape s) {
    bytes(state.matrix, checked_mul(gdn_state_count(s), 4));
    bytes(state.convolution, checked_mul(checked_mul(gdn_qkv_count(s), 3), 4));
}
void tape_bytes(const GdnScratch& scratch, GdnShape s, StateTile t) {
    const auto n = counts(s); const auto views = scratch.views();
    for (size_t i = 0; i < n.size(); ++i) bytes(views[i], checked_mul(checked_mul(n[i], t.capacity), 4));
}
void many(MetalGraph& g, GpuMatrix matrix, GpuView input, GpuView output,
          uint32_t columns, uint32_t rows, uint32_t steps) {
    uint32_t at = 0;
    while (at < steps) {
        const uint32_t width = g.q8_many_width(steps - at);
        g.q8_matvec_many(matrix.view, g.slice(input, uint64_t(at) * columns * 4, uint64_t(width) * columns * 4),
            g.slice(output, uint64_t(at) * rows * 4, uint64_t(width) * rows * 4), columns, rows, width);
        at += width;
    }
}
void abort(MetalGraph& g) { try { g.abort(); } catch (const FormatError&) {} }
}

GdnScratch allocate_gdn_core_tile_scratch(MetalGraph& g, GdnShape s, uint32_t capacity) {
    validate(s, {capacity, 1}); const auto n = counts(s); std::array<GpuView, 13> v{};
    try {
        for (size_t i = 0; i < n.size(); ++i)
            v[i] = g.zeros(checked_mul(checked_mul(n[i], capacity), 4), "GDN core tile scratch");
    } catch (...) { for (const auto view : v) if (view.owner) g.release(view); throw; }
    return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12]};
}
void encode_gdn_core_tile(MetalGraph& g, const GdnWeights& w, const GdnScratch& s,
                          const GdnState& accepted, const GdnState& candidate,
                          GpuView input, GpuView output, GdnShape shape, GdnPolicy policy, StateTile tile) {
    try {
        validate(shape, tile); validate_gdn_policy(policy);
        const auto kernel_policy = g.kernel_policy();
        const auto splits = kernel_policy[0];
        require((splits == 0 || splits == 1 || splits == 4 || splits == 8 || splits == 16 || splits == 32) && (kernel_policy[1] & 1),
            "GDN core tile requires supported F32/block-Q8 policy");
        const auto raw = static_cast<uint32_t>(gdn_qkv_count(shape)), value = shape.value_heads * shape.value_width;
        require(w.qkv.encoding == 8 && w.gate.encoding == 8 && w.output.encoding == 8 &&
            w.alpha.encoding == 0 && w.beta.encoding == 0, "GDN core tile projection encoding mismatch");
        bytes(w.qkv.view, q8_many_bytes({shape.channels, raw, 1}).matrix);
        bytes(w.gate.view, q8_many_bytes({shape.channels, value, 1}).matrix);
        bytes(w.output.view, q8_many_bytes({value, shape.channels, 1}).matrix);
        bytes(w.alpha.view, uint64_t(shape.channels) * shape.value_heads * 4); bytes(w.beta.view, w.alpha.view.bytes);
        bytes(w.convolution, uint64_t(raw) * 16); bytes(w.negative_rates, uint64_t(shape.value_heads) * 4);
        bytes(w.dt_bias, w.negative_rates.bytes); bytes(w.gains, uint64_t(shape.value_width) * 4);
        state_bytes(accepted, shape); state_bytes(candidate, shape); tape_bytes(s, shape, tile);
        bytes(input, uint64_t(tile.capacity) * shape.channels * 4); bytes(output, input.bytes);
        const std::array reads{input, accepted.matrix, accepted.convolution, w.qkv.view, w.gate.view, w.alpha.view,
            w.beta.view, w.output.view, w.convolution, w.negative_rates, w.dt_bias, w.gains};
        std::array<GpuView, 16> writes{output, candidate.matrix, candidate.convolution};
        const auto workspace = s.views(); std::copy(workspace.begin(), workspace.end(), writes.begin() + 3);
        g.validate_disjoint(reads, writes);
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        const bool use_verify_nax = g.verify_nax_active() && tile.steps >= 2 && tile.steps <= 8;
        if (use_verify_nax) {
            encode_verify_nax_projection(g, w.qkv, input, s.projected, shape.channels, raw, tile.steps);
            encode_verify_nax_projection(g, w.gate, input, s.gate, shape.channels, value, tile.steps);
            encode_verify_nax_gdn_f32_gates(g, w, s, input, shape, policy, tile.steps);
        } else
#endif
        {
            many(g, w.qkv, input, s.projected, shape.channels, raw, tile.steps);
            many(g, w.gate, input, s.gate, shape.channels, value, tile.steps);
            for (uint32_t step = 0; step < tile.steps; ++step) {
                const auto t = token(g, s, shape, step); const auto x = row(g, input, shape.channels, step);
                g.gemv(0, w.alpha.view, x, t.alpha, shape.channels, shape.value_heads);
                g.gemv(0, w.beta.view, x, t.beta, shape.channels, shape.value_heads);
            }
        }
        g.gdn_convolution_tile(accepted.convolution, s.projected, w.convolution, candidate.convolution, s.convolved, raw, tile);
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        if (g.gdn_use_ported()) {
            const auto active_conv = g.slice(s.convolved, 0, uint64_t(tile.steps) * raw * 4);
            g.activate(active_conv, active_conv, Activation::silu);
            g.gdn_prepare_rows(s.convolved, s.query, s.key, s.value, shape, policy, tile);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_projections_convolution_activation_prepare",{tile.steps,shape.channels,shape.value_heads,shape.value_width});
            g.gdn_verify_main(accepted.matrix, s.query, s.key, s.value, s.alpha, s.beta, w.negative_rates, w.dt_bias,
                candidate.matrix, s.raw, s.decay, s.write_gate, shape.value_heads, shape.value_width, tile, true);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_core",{tile.steps,shape.value_heads,shape.key_width,shape.value_width});
            g.gdn_norm_gate(s.raw, w.gains, s.gate, s.gated, shape.value_heads, shape.value_width, policy.rms_epsilon, tile);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_rms_gate",{tile.steps,value,0,0});
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            if (use_verify_nax) encode_verify_nax_projection(g, w.output, s.gated, output, value, shape.channels, tile.steps);
            else
#endif
            many(g, w.output, s.gated, output, value, shape.channels, tile.steps);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_output_gemv",{tile.steps,value,shape.channels,w.output.encoding});
            return;
        }
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        if (verify_gdn_row_batch && gdn_core_batch_elementwise && use_verify_nax) {
            g.gdn_gates_verify_rows(s.alpha, s.beta, w.negative_rates, w.dt_bias,
                s.decay, s.write_gate, shape.value_heads, tile);
            g.gdn_silu_prepare_verify_rows(s.convolved, s.query, s.key, s.value, shape, policy, tile);
        } else
#endif
        {
            if constexpr (gdn_core_batch_elementwise) {
                const auto active = g.slice(s.convolved, 0, uint64_t(tile.steps) * raw * 4);
                g.activate(active, active, Activation::silu);
            }
            for (uint32_t step = 0; step < tile.steps; ++step) {
                const auto t = token(g, s, shape, step);
                if constexpr (!gdn_core_batch_elementwise) g.activate(t.convolved, t.convolved, Activation::silu);
                g.gdn_prepare(t.convolved, t.query, t.key, t.value, shape, policy);
                g.gdn_gates(t.alpha, t.beta, w.negative_rates, w.dt_bias, t.decay, t.write_gate);
            }
        }
        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_projections_convolution_activation_prepare",{tile.steps,shape.channels,shape.value_heads,shape.value_width});
        g.gdn_delta_tile(accepted.matrix, s.query, s.key, s.value, s.decay, s.write_gate,
            candidate.matrix, s.raw, shape.value_heads, shape.value_width, tile);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_core",{tile.steps,shape.value_heads,shape.key_width,shape.value_width});
        if constexpr (gdn_core_batch_elementwise) {
            const auto active = [&](GpuView v) { return g.slice(v, 0, uint64_t(tile.steps) * value * 4); };
            // RMS groups share the same per-channel gains. Slice only committed
            // input rows; inactive scratch/tape tails remain untouched.
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            if (verify_gdn_row_batch && use_verify_nax) {
                g.rms_sigmoid_gate(active(s.raw), w.gains, active(s.gate), active(s.gated), shape.value_width, policy.rms_epsilon);
            } else
#endif
            {
                g.rms(active(s.raw), w.gains, active(s.normalized), shape.value_width, policy.rms_epsilon);
                g.multiply(active(s.normalized), active(s.gate), active(s.gated), true);
            }
        } else {
            for (uint32_t step = 0; step < tile.steps; ++step) {
                const auto t = token(g, s, shape, step);
                g.rms(t.raw, w.gains, t.normalized, shape.value_width, policy.rms_epsilon);
                g.multiply(t.normalized, t.gate, t.gated, true);
            }
        }
        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_rms_gate",{tile.steps,value,0,0});
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        if (use_verify_nax) encode_verify_nax_projection(g, w.output, s.gated, output, value, shape.channels, tile.steps);
        else
#endif
        many(g, w.output, s.gated, output, value, shape.channels, tile.steps);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_output_gemv",{tile.steps,value,shape.channels,w.output.encoding});
    } catch (...) { abort(g); throw; }
}

GdnCoreReplayScratch allocate_gdn_core_replay_scratch(MetalGraph& g, GdnShape shape, uint32_t capacity) {
    validate(shape, {capacity, 1}); GdnCoreReplayScratch s{};
    try {
        s.convolution_output = g.zeros(uint64_t(capacity) * gdn_qkv_count(shape) * 4, "GDN prefix convolution scratch");
        s.delta_output = g.zeros(uint64_t(capacity) * shape.value_heads * shape.value_width * 4, "GDN prefix delta scratch");
    } catch (...) { if (s.convolution_output.owner) g.release(s.convolution_output); throw; }
    return s;
}
GdnCoreTape allocate_gdn_core_tape(MetalGraph& g, GdnShape shape, uint32_t capacity) {
    validate(shape, {capacity, 1}); const auto n = counts(shape);
    const std::array<uint64_t, 6> widths{n[0], n[2], n[3], n[4], n[8], n[9]};
    std::array<GpuView, 6> v{};
    try {
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = g.zeros(checked_mul(checked_mul(widths[i], capacity), 4), "retained GDN prefix tape");
    } catch (...) { for (auto view : v) if (view.owner) g.release(view); throw; }
    return {v[0], v[1], v[2], v[3], v[4], v[5]};
}
GdnScratch bind_gdn_core_tape(GdnScratch shared, const GdnCoreTape& tape) {
    shared.projected = tape.projected; shared.query = tape.query; shared.key = tape.key;
    shared.value = tape.value; shared.decay = tape.decay; shared.write_gate = tape.write_gate;
    return shared;
}
void encode_gdn_core_tape_prefix(MetalGraph& g, GpuView weights, const GdnCoreTape& tape,
    const GdnState& accepted, const GdnState& candidate, const GdnCoreReplayScratch& scratch,
    GdnShape shape, StateTile tile) {
    try {
        validate(shape, tile); state_bytes(accepted, shape); state_bytes(candidate, shape);
        const auto n = counts(shape); const auto views = tape.views();
        const std::array<uint64_t, 6> widths{n[0], n[2], n[3], n[4], n[8], n[9]};
        for (size_t i = 0; i < views.size(); ++i) bytes(views[i], checked_mul(checked_mul(widths[i], tile.capacity), 4));
        bytes(weights, gdn_qkv_count(shape) * 16); bytes(scratch.convolution_output, tape.projected.bytes);
        bytes(scratch.delta_output, tape.value.bytes);
        std::array<GpuView, 9> reads{accepted.matrix, accepted.convolution, weights};
        std::copy(views.begin(), views.end(), reads.begin() + 3);
        const std::array writes{candidate.matrix, candidate.convolution, scratch.convolution_output, scratch.delta_output};
        g.validate_disjoint(reads, writes);
        g.gdn_convolution_tile(accepted.convolution, tape.projected, weights, candidate.convolution,
            scratch.convolution_output, static_cast<uint32_t>(gdn_qkv_count(shape)), tile);
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        if (g.gdn_use_ported()) {
            g.gdn_verify_replay(accepted.matrix, tape.key, tape.value, tape.decay, tape.write_gate,
                candidate.matrix, shape.value_heads, shape.value_width, tile);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_replay",{tile.steps,shape.value_heads,shape.key_width,shape.value_width});
            return;
        }
#endif
        g.gdn_delta_tile(accepted.matrix, tape.query, tape.key, tape.value, tape.decay, tape.write_gate,
            candidate.matrix, scratch.delta_output, shape.value_heads, shape.value_width, tile);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_replay",{tile.steps,shape.value_heads,shape.key_width,shape.value_width});
    } catch (...) { abort(g); throw; }
}
void release_gdn_core_replay_scratch(MetalGraph& g, const GdnCoreReplayScratch& s) {
    g.release(s.convolution_output); g.release(s.delta_output);
}
void encode_gdn_core_prefix(MetalGraph& g, GpuView weights, const GdnScratch& tape,
                            GpuView retained_outputs, const GdnState& accepted, const GdnState& candidate,
                            const GdnCoreReplayScratch& scratch, GdnShape shape, StateTile tile) {
    try {
        validate(shape, tile); state_bytes(accepted, shape); state_bytes(candidate, shape); tape_bytes(tape, shape, tile);
        bytes(weights, gdn_qkv_count(shape) * 16); bytes(scratch.convolution_output, tape.projected.bytes);
        bytes(scratch.delta_output, tape.value.bytes); bytes(retained_outputs, uint64_t(tile.capacity) * shape.channels * 4);
        std::array<GpuView, 17> reads{accepted.matrix, accepted.convolution, weights, retained_outputs};
        const auto views = tape.views(); std::copy(views.begin(), views.end(), reads.begin() + 4);
        const std::array writes{candidate.matrix, candidate.convolution, scratch.convolution_output, scratch.delta_output};
        g.validate_disjoint(reads, writes);
        g.gdn_convolution_tile(accepted.convolution, tape.projected, weights, candidate.convolution,
            scratch.convolution_output, static_cast<uint32_t>(gdn_qkv_count(shape)), tile);
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        if (g.gdn_use_ported()) {
            g.gdn_verify_replay(accepted.matrix, tape.key, tape.value, tape.decay, tape.write_gate,
                candidate.matrix, shape.value_heads, shape.value_width, tile);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_replay",{tile.steps,shape.value_heads,shape.key_width,shape.value_width});
            return;
        }
#endif
        g.gdn_delta_tile(accepted.matrix, tape.query, tape.key, tape.value, tape.decay, tape.write_gate,
            candidate.matrix, scratch.delta_output, shape.value_heads, shape.value_width, tile);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","verify_replay",{tile.steps,shape.value_heads,shape.key_width,shape.value_width});
    } catch (...) { abort(g); throw; }
}
} // namespace tokspur
