#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
#include "tokspur/moe_q8_three_rows.hpp"
#endif
#include "tokspur/recurrent_tile.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <cmath>

namespace tokspur {
namespace {
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
void validate(RecurrentShape s, StateTile tile) {
    validate_recurrent_shape(s); validate_state_tile(tile);
    require(s.gdn.key_width == 128 && s.gdn.kernel == 4 && s.gdn.value_heads <= 48 && s.gdn.value_width <= 128 &&
        s.gdn.channels % 32 == 0 && (uint64_t(s.gdn.value_heads) * s.gdn.value_width) % 32 == 0,
        "unsupported recurrent tile GDN shape");
}
std::array<uint64_t, 32> counts(RecurrentShape s) {
    const auto b = s.residual.branches, c = s.residual.channels, r = s.residual.rank;
    const auto bc = checked_mul(b, c), raw = gdn_qkv_count(s.gdn), qk = checked_mul(s.gdn.value_heads, s.gdn.key_width);
    const auto value = checked_mul(s.gdn.value_heads, s.gdn.value_width);
    const auto h = s.moe.hidden, k = s.moe.selected;
    return {bc, r, bc, b, c, b, raw, raw, qk, qk, value, value, s.gdn.value_heads, s.gdn.value_heads,
        s.gdn.value_heads, s.gdn.value_heads, value, value, value, s.moe.experts, k, k,
        checked_mul(h, k), checked_mul(h, k), checked_mul(c, k), h, h, c, 1, bc, c, c};
}
RecurrentScratch assemble(const std::array<GpuView, 32>& v) {
    return {{v[0], v[1], v[2], v[3], v[4], v[5]},
        {v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16], v[17], v[18]},
        {v[19], v[20], v[21], v[22], v[23], v[24], v[25], v[26], v[27], v[28]}, v[29], v[30], v[31]};
}
void bytes(GpuView v, uint64_t expected) { require(v.bytes == expected, "recurrent tile buffer geometry mismatch"); }
void workspace(const RecurrentScratch& scratch, RecurrentShape shape, uint32_t capacity) {
    const auto n = counts(shape); const auto views = scratch.views();
    for (size_t i = 0; i < n.size(); ++i) {
        const auto count = checked_mul(n[i], capacity); require(count && count <= 16'777'216, "recurrent tile scratch count bound");
        bytes(views[i], checked_mul(count, 4));
    }
}
std::array<GpuView, 25> weights(const RecurrentWeights& w) {
    return {w.attention_read.gains, w.attention_read.down.view, w.attention_read.up.view, w.attention_read.inject.view,
        w.ffn_read.gains, w.ffn_read.down.view, w.ffn_read.up.view, w.ffn_read.inject.view,
        w.attention.qkv.view, w.attention.gate.view, w.attention.alpha.view, w.attention.beta.view, w.attention.output.view,
        w.attention.convolution, w.attention.negative_rates, w.attention.dt_bias, w.attention.gains,
        w.ffn.router.view, w.ffn.gate.view, w.ffn.up.view, w.ffn.down.view, w.ffn.shared_gate.view,
        w.ffn.shared_up.view, w.ffn.shared_down.view, w.ffn.shared_router.view};
}
void matrix(GpuMatrix w, uint32_t columns, uint32_t rows, uint32_t experts = 1) {
    require(can_decode(w.encoding), "unsupported recurrent tile matrix encoding"); const auto& format = encoding(w.encoding);
    require(columns && columns % format.elements_per_block == 0 && rows && experts, "recurrent tile matrix block geometry");
    bytes(w.view, checked_mul(checked_mul(checked_mul(columns / format.elements_per_block, format.bytes_per_block), rows), experts));
}
void weight_shapes(const MetalGraph& g, const RecurrentWeights& w, RecurrentShape s) {
    const auto c = s.residual.channels, b = s.residual.branches, r = s.residual.rank, bc = b * c;
    for (const auto& part : {w.attention_read, w.ffn_read}) {
        bytes(part.gains, uint64_t(bc) * 4); matrix(part.down, bc, r); matrix(part.up, r, bc); matrix(part.inject, bc, b);
    }
    const auto raw = static_cast<uint32_t>(gdn_qkv_count(s.gdn)), value = s.gdn.value_heads * s.gdn.value_width;
    const auto& a = w.attention;
    require(a.qkv.encoding == 8 && a.gate.encoding == 8 && a.output.encoding == 8 && a.alpha.encoding == 0 && a.beta.encoding == 0,
        "recurrent tile requires Q8/F32 GDN projections");
    matrix(a.qkv, c, raw); matrix(a.gate, c, value); matrix(a.output, value, c);
    matrix(a.alpha, c, s.gdn.value_heads); matrix(a.beta, c, s.gdn.value_heads);
    bytes(a.convolution, uint64_t(raw) * 16); bytes(a.negative_rates, uint64_t(s.gdn.value_heads) * 4);
    bytes(a.dt_bias, a.negative_rates.bytes); bytes(a.gains, uint64_t(s.gdn.value_width) * 4);
    const auto& m = w.ffn; const auto h = s.moe.hidden, e = s.moe.experts;
    matrix(m.router, c, e);
    if (!is_planar_moe(g, m, s.moe)) {
        matrix(m.gate, c, h, e); matrix(m.up, c, h, e); matrix(m.down, h, c, e);
    }
    matrix(m.shared_gate, c, h); matrix(m.shared_up, c, h); matrix(m.shared_down, h, c); matrix(m.shared_router, c, 1);
}
void state(const GdnState& v, GdnShape s) {
    bytes(v.matrix, gdn_state_count(s) * 4); bytes(v.convolution, gdn_qkv_count(s) * 12);
}
GpuView row(MetalGraph& g, GpuView v, uint64_t count, uint32_t step) { return g.slice(v, uint64_t(step) * count * 4, count * 4); }
void projection_rows(MetalGraph& g, GpuMatrix matrix, GpuView input, GpuView output,
    uint32_t columns, uint32_t rows, uint32_t steps) {
    // Only unchanged Q8 block arithmetic has a qualified multi-input primitive.
    // Every other encoding retains the existing per-row GEMV and split policy.
    if (matrix.encoding == 8) {
        uint32_t at = 0;
        while (at < steps) {
            const uint32_t n = g.q8_many_width(steps - at);
            g.q8_matvec_many(matrix.view, g.slice(input, uint64_t(at) * columns * 4, uint64_t(n) * columns * 4),
                g.slice(output, uint64_t(at) * rows * 4, uint64_t(n) * rows * 4), columns, rows, n);
            at += n;
        }
    } else for (uint32_t t = 0; t < steps; ++t)
        g.gemv(matrix.encoding, matrix.view, row(g, input, columns, t), row(g, output, rows, t), columns, rows);
}
void residual_rows(MetalGraph& g, const ResidualWeights& w, const RecurrentScratch& scratch,
    GpuView input, RecurrentShape shape, StateTile tile, float epsilon) {
    const auto width = shape.residual.branches * shape.residual.channels;
    for (uint32_t t = 0; t < tile.steps; ++t) {
        const auto s = recurrent_tile_token(g, scratch, shape, tile.capacity, t).residual;
        g.rms(row(g, input, width, t), w.gains, s.normalized, shape.residual.channels, epsilon);
    }
    projection_rows(g, w.down, scratch.residual.normalized, scratch.residual.bottleneck, width, shape.residual.rank, tile.steps);
    for (uint32_t t = 0; t < tile.steps; ++t) {
        const auto s = recurrent_tile_token(g, scratch, shape, tile.capacity, t).residual;
        g.activate(s.bottleneck, s.bottleneck, Activation::silu, 1.0f / shape.residual.branches);
    }
    projection_rows(g, w.up, scratch.residual.bottleneck, scratch.residual.scores, shape.residual.rank, width, tile.steps);
    for (uint32_t t = 0; t < tile.steps; ++t) {
        const auto s = recurrent_tile_token(g, scratch, shape, tile.capacity, t).residual;
        g.gemv(w.inject.encoding, w.inject.view, s.normalized, s.inject_logits, width, shape.residual.branches);
        g.residual_read(s.normalized, s.scores, s.inject_logits, s.input, s.scales, shape.residual.branches, shape.residual.channels);
    }
}
void abort(MetalGraph& g) { try { g.abort(); } catch (const FormatError&) {} }
}
RecurrentScratch allocate_recurrent_tile_scratch(MetalGraph& g, RecurrentShape shape, uint32_t capacity) {
    validate(shape, {capacity, 1}); const auto n = counts(shape); std::array<GpuView, 32> views{};
    for (auto count : n) require(checked_mul(count, capacity) <= 16'777'216, "recurrent tile allocation count bound");
    try {
        for (size_t i = 0; i < views.size(); ++i) views[i] = g.zeros(checked_mul(checked_mul(n[i], capacity), 4), "recurrent tile scratch");
    } catch (...) { for (auto view : views) if (view.owner) g.release(view); throw; }
    return assemble(views);
}
RecurrentScratch recurrent_tile_token(MetalGraph& g, const RecurrentScratch& scratch, RecurrentShape shape, uint32_t capacity, uint32_t step) {
    validate(shape, {capacity, 1}); require(step < capacity, "recurrent tile token outside capacity"); workspace(scratch, shape, capacity);
    const auto n = counts(shape); const auto views = scratch.views(); std::array<GpuView, 32> result{};
    for (size_t i = 0; i < result.size(); ++i) result[i] = row(g, views[i], n[i], step);
    return assemble(result);
}
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
void encode_known_residual_rows512(MetalGraph& g, const ResidualWeights& w, const RecurrentScratch& scratch,
    GpuView input, RecurrentShape shape, StateTile tile, float epsilon) {
    try {
        validate(shape, tile); workspace(scratch, shape, tile.capacity);
        require(tile.capacity == 512 && std::isfinite(epsilon) && epsilon > 0,
            "private residual row reuse requires capacity512 and finite positive RMS epsilon");
        const auto width = shape.residual.branches * shape.residual.channels;
        bytes(input, uint64_t(tile.capacity) * width * 4);
        bytes(w.gains, uint64_t(width) * 4);
        matrix(w.down, width, shape.residual.rank); matrix(w.up, shape.residual.rank, width);
        matrix(w.inject, width, shape.residual.branches);
        const std::array reads{input, w.gains, w.down.view, w.up.view, w.inject.view};
        g.validate_disjoint(reads, scratch.residual.views());
        residual_rows(g, w, scratch, input, shape, tile, epsilon);
    } catch (...) { abort(g); throw; }
}
#endif
void encode_recurrent_layer_tile(MetalGraph& g, const RecurrentWeights& w, const RecurrentScratch& s,
    const GdnState& accepted, const GdnState& candidate, GpuView input, GpuView output,
    RecurrentShape shape, GdnPolicy policy, RouterMode mode, bool normalize, StateTile tile, bool reuse_residual_projections
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
    , const MoeManyScratch* grouped_ffn, bool reuse_residual_rows512
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    , const VerifyNaxMoeScratch* verify_nax_moe,
    bool input_prenormalized, GpuView next_rms_gains, float next_rms_epsilon
#endif
    ) {
    try {
        validate(shape, tile); validate_gdn_policy(policy); require(uint32_t(mode) <= 1, "invalid recurrent tile router mode");
        const auto kernel_policy = g.kernel_policy(); const auto splits = kernel_policy[0];
        require((splits == 0 || splits == 1 || splits == 4 || splits == 8 || splits == 16 || splits == 32) && (kernel_policy[1] & 1),
            "recurrent tile requires supported F32/block-Q8 policy");
        weight_shapes(g, w, shape); workspace(s, shape, tile.capacity); state(accepted, shape.gdn); state(candidate, shape.gdn);
        const auto width = uint64_t(shape.residual.branches) * shape.residual.channels;
        bytes(input, uint64_t(tile.capacity) * width * 4); bytes(output, input.bytes);
        std::array<GpuView, 28> reads{input, accepted.matrix, accepted.convolution}; const auto protected_weights = weights(w);
        std::copy(protected_weights.begin(), protected_weights.end(), reads.begin() + 3);
        std::array<GpuView, 35> writes{output, candidate.matrix, candidate.convolution}; const auto scratch = s.views();
        std::copy(scratch.begin(), scratch.end(), writes.begin() + 3); g.validate_disjoint(reads, writes);
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        require(!reuse_residual_rows512 || (tile.capacity == 512 && !reuse_residual_projections && !grouped_ffn),
            "private capacity512 residual rows require ordinary FFN and their separate selector");
        if (grouped_ffn) {
            require(!reuse_residual_projections && grouped_ffn->capacity == tile.capacity && tile.capacity == 512,
                "grouped recurrent prefill requires separate capacity512 resources without residual reuse");
            const auto extra = grouped_ffn->views();
            std::vector<GpuView> protected_reads(reads.begin(), reads.end());
            protected_reads.insert(protected_reads.end(), writes.begin(), writes.end());
            g.validate_disjoint(protected_reads, extra);
        }
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        if (g.verify_nax_active() && verify_nax_moe && tile.steps >= 2 && tile.steps <= 8) {
            if (next_rms_gains.owner != 0) g.validate_disjoint(std::array{next_rms_gains}, writes);
            std::vector<GpuView> protected_reads(reads.begin(), reads.end());
            protected_reads.insert(protected_reads.end(), writes.begin(), writes.end());
            g.validate_disjoint(protected_reads, verify_nax_moe->views());
            if (input_prenormalized) {
                encode_verify_nax_residual_read_prenorm(g, w.attention_read, s.residual, shape.residual, tile.steps);
            } else {
                encode_verify_nax_residual_read(g, w.attention_read, s.residual, input, shape.residual, tile.steps, policy.rms_epsilon);
            }
            encode_gdn_core_tile(g, w.attention, s.gdn, accepted, candidate, s.residual.input, s.attention_output, shape.gdn, policy, tile);
            encode_verify_nax_residual_write_read(g, w.ffn_read, s.residual, input, s.attention_output, s.residual.scales, s.intermediate_streams, shape.residual, tile.steps, policy.rms_epsilon);
            encode_verify_nax_moe(g, w.ffn, s.moe, *verify_nax_moe, s.residual.input, s.ffn_output, shape.moe, mode, normalize, tile.steps);
            if (next_rms_gains.owner != 0) {
                encode_verify_nax_residual_write_rms(g, s.intermediate_streams, s.ffn_output, s.residual.scales,
                    next_rms_gains, output, s.residual.normalized, shape.residual, tile.steps, next_rms_epsilon);
            } else {
                encode_verify_nax_residual_write(g, s.intermediate_streams, s.ffn_output, s.residual.scales, output, shape.residual, tile.steps);
            }
            return;
        }
#endif
        if (reuse_residual_projections
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
            || reuse_residual_rows512
#endif
            ) residual_rows(g, w.attention_read, s, input, shape, tile, policy.rms_epsilon);
        else for (uint32_t step = 0; step < tile.steps; ++step) {
            const auto t = recurrent_tile_token(g, s, shape, tile.capacity, step);
            encode_residual_read(g, w.attention_read, t.residual, row(g, input, width, step), shape.residual, policy.rms_epsilon);
        }
        encode_gdn_core_tile(g, w.attention, s.gdn, accepted, candidate, s.residual.input, s.attention_output, shape.gdn, policy, tile);
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        if (tile.capacity == 512) {
            // Preserve the chronological core and per-row residual equations.
            // The grouped-FFN selector retains its existing arithmetic arm;
            // separate HC row reuse shares projections without changing it.
            if (reuse_residual_rows512) {
                // All attention scales must be consumed before residual_rows
                // overwrites the shared per-row HC scratch for the FFN read.
                for (uint32_t step = 0; step < tile.steps; ++step) {
                    const auto t = recurrent_tile_token(g, s, shape, tile.capacity, step);
                    g.residual_write(row(g, input, width, step), t.attention_output, t.residual.scales, t.intermediate_streams);
                }
                residual_rows(g, w.ffn_read, s, s.intermediate_streams, shape, tile, policy.rms_epsilon);
            } else for (uint32_t step = 0; step < tile.steps; ++step) {
                const auto t = recurrent_tile_token(g, s, shape, tile.capacity, step);
                g.residual_write(row(g, input, width, step), t.attention_output, t.residual.scales, t.intermediate_streams);
                encode_residual_read(g, w.ffn_read, t.residual, t.intermediate_streams, shape.residual, policy.rms_epsilon);
            }
            if (grouped_ffn) encode_moe_many(g, w.ffn, *grouped_ffn, s.residual.input, s.ffn_output, shape.moe, mode, normalize, tile.steps);
            else for (uint32_t step = 0; step < tile.steps; ++step) {
                const auto t = recurrent_tile_token(g, s, shape, tile.capacity, step);
                encode_moe(g, w.ffn, t.moe, t.residual.input, t.ffn_output, shape.moe, mode, normalize
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                    , step
#endif
                    );
            }
            for (uint32_t step = 0; step < tile.steps; ++step) {
                const auto t = recurrent_tile_token(g, s, shape, tile.capacity, step);
                g.residual_write(t.intermediate_streams, t.ffn_output, t.residual.scales, row(g, output, width, step));
            }
            return;
        }
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        if (g.q8_three_rows_enabled() && tile.steps == 3) {
            for (uint32_t t = 0; t < 3; ++t) {
                const auto r = recurrent_tile_token(g,s,shape,tile.capacity,t);
                g.residual_write(row(g,input,width,t),r.attention_output,r.residual.scales,r.intermediate_streams);
            }
            if (reuse_residual_projections) residual_rows(g,w.ffn_read,s,s.intermediate_streams,shape,tile,policy.rms_epsilon);
            else for (uint32_t t = 0; t < 3; ++t) {
                const auto r = recurrent_tile_token(g,s,shape,tile.capacity,t);
                encode_residual_read(g,w.ffn_read,r.residual,r.intermediate_streams,shape.residual,policy.rms_epsilon);
            }
            encode_moe_q8_three_rows(g,w.ffn,s.moe,s.residual.input,s.ffn_output,shape.moe,mode,normalize,tile.capacity);
            for (uint32_t t = 0; t < 3; ++t) {
                const auto r = recurrent_tile_token(g,s,shape,tile.capacity,t);
                g.residual_write(r.intermediate_streams,r.ffn_output,r.residual.scales,row(g,output,width,t));
            }
            return;
        }
#endif
        if (reuse_residual_projections) {
            // Consume EVERY attention-read scale before reusing the same
            // capacity-sized residual scratch for the FFN read. No accepted
            // recurrent state is published by this encoder.
            for (uint32_t step = 0; step < tile.steps; ++step) {
                const auto t = recurrent_tile_token(g, s, shape, tile.capacity, step);
                g.residual_write(row(g, input, width, step), t.attention_output, t.residual.scales, t.intermediate_streams);
            }
            residual_rows(g, w.ffn_read, s, s.intermediate_streams, shape, tile, policy.rms_epsilon);
            for (uint32_t step = 0; step < tile.steps; ++step) {
                const auto t = recurrent_tile_token(g, s, shape, tile.capacity, step);
                encode_moe(g, w.ffn, t.moe, t.residual.input, t.ffn_output, shape.moe, mode, normalize
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                    , step
#endif
                    );
                g.residual_write(t.intermediate_streams, t.ffn_output, t.residual.scales, row(g, output, width, step));
            }
            return;
        }
        for (uint32_t step = 0; step < tile.steps; ++step) {
            const auto t = recurrent_tile_token(g, s, shape, tile.capacity, step);
            g.residual_write(row(g, input, width, step), t.attention_output, t.residual.scales, t.intermediate_streams);
            encode_residual_read(g, w.ffn_read, t.residual, t.intermediate_streams, shape.residual, policy.rms_epsilon);
            encode_moe(g, w.ffn, t.moe, t.residual.input, t.ffn_output, shape.moe, mode, normalize
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                , step
#endif
                );
            g.residual_write(t.intermediate_streams, t.ffn_output, t.residual.scales, row(g, output, width, step));
        }
    } catch (...) { abort(g); throw; }
}
void encode_recurrent_tile_prefix(MetalGraph& g, const RecurrentWeights& w, const RecurrentScratch& tape,
    const GdnState& accepted, const GdnState& candidate, GpuView output, const GdnCoreReplayScratch& replay, RecurrentShape shape, StateTile tile) {
    try {
        validate(shape, tile); weight_shapes(g, w, shape); workspace(tape, shape, tile.capacity); state(accepted, shape.gdn); state(candidate, shape.gdn);
        bytes(output, uint64_t(tile.capacity) * shape.residual.branches * shape.residual.channels * 4);
        bytes(replay.convolution_output, tape.gdn.projected.bytes); bytes(replay.delta_output, tape.gdn.value.bytes);
        std::array<GpuView, 60> reads{accepted.matrix, accepted.convolution, output};
        const auto protected_weights = weights(w); auto at = std::copy(protected_weights.begin(), protected_weights.end(), reads.begin() + 3);
        const auto views = tape.views(); std::copy(views.begin(), views.end(), at);
        const std::array writes{candidate.matrix, candidate.convolution, replay.convolution_output, replay.delta_output};
        g.validate_disjoint(reads, writes);
        encode_gdn_core_prefix(g, w.attention.convolution, tape.gdn, tape.attention_output, accepted, candidate, replay, shape.gdn, tile);
    } catch (...) { abort(g); throw; }
}
} // namespace tokspur
