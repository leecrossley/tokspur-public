#include "tokspur/recurrent.hpp"
#include "tokspur/operator_checkpoints.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cmath>

namespace tokspur {
namespace {
template<size_t N> std::array<GpuView, N> allocate(MetalGraph& g, const std::array<uint64_t, N>& counts) {
    std::array<GpuView, N> result{};
    size_t made = 0;
    try {
        for (; made < N; ++made) result[made] = g.zeros(checked_mul(counts[made], 4), "recurrent layer workspace");
    } catch (...) {
        for (size_t i = 0; i < made; ++i) g.release(result[i]);
        throw;
    }
    return result;
}
std::array<GpuView, 9> gdn_weights(const GdnWeights& w) {
    return {w.qkv.view, w.gate.view, w.alpha.view, w.beta.view, w.output.view,
        w.convolution, w.negative_rates, w.dt_bias, w.gains};
}
std::array<GpuView, 4> residual_weights(const ResidualWeights& w) { return {w.gains, w.down.view, w.up.view, w.inject.view}; }
void abort_on_failure(MetalGraph& g) { try { g.abort(); } catch (const FormatError&) {} }
}

void validate_residual_shape(ResidualShape s) {
    if (!s.branches || s.branches > 64 || !s.channels || s.channels > 65536 ||
        !s.rank || s.rank > 4096 || uint64_t(s.branches) * s.channels > 65536) throw FormatError("unsupported residual shape");
}
void validate_recurrent_shape(RecurrentShape s) {
    validate_residual_shape(s.residual); validate_gdn_shape(s.gdn);
    if (s.residual.channels != s.gdn.channels || s.residual.channels != s.moe.channels ||
        !s.moe.hidden || s.moe.hidden > 65536 || !s.moe.experts || s.moe.experts > 4096 ||
        !s.moe.selected || s.moe.selected > 64 || s.moe.selected > s.moe.experts) throw FormatError("inconsistent recurrent layer shape");
}
std::array<GpuView, 6> ResidualScratch::views() const { return {normalized, bottleneck, scores, inject_logits, input, scales}; }
std::array<GpuView, 2> GdnState::views() const { return {matrix, convolution}; }
std::array<GpuView, 13> GdnScratch::views() const { return {projected, convolved, query, key, value, gate, alpha, beta, decay, write_gate, raw, normalized, gated}; }
std::array<GpuView, 32> RecurrentScratch::views() const {
    std::array<GpuView, 32> out;
    // Store arrays before taking their iterators (temporaries have separate lives).
    const auto rv = residual.views(); auto at = std::copy(rv.begin(), rv.end(), out.begin());
    const auto gv = gdn.views(); at = std::copy(gv.begin(), gv.end(), at);
    const auto mv = moe.views(); at = std::copy(mv.begin(), mv.end(), at);
    *at++ = intermediate_streams; *at++ = attention_output; *at = ffn_output;
    return out;
}
ResidualScratch allocate_residual_scratch(MetalGraph& g, ResidualShape s) {
    validate_residual_shape(s);
    const auto width = uint64_t(s.branches) * s.channels;
    const auto v = allocate(g, std::array<uint64_t, 6>{width, s.rank, width, s.branches, s.channels, s.branches});
    return {v[0], v[1], v[2], v[3], v[4], v[5]};
}
void release_residual_scratch(MetalGraph& g, const ResidualScratch& s) { for (const auto v : s.views()) g.release(v); }
GdnState allocate_gdn_state(MetalGraph& g, GdnShape s) {
    const auto v = allocate(g, std::array<uint64_t, 2>{gdn_state_count(s), checked_mul(gdn_qkv_count(s), s.kernel - 1)});
    return {v[0], v[1]};
}
void release_gdn_state(MetalGraph& g, const GdnState& s) { for (const auto v : s.views()) g.release(v); }
GdnScratch allocate_gdn_scratch(MetalGraph& g, GdnShape s) {
    const auto raw = gdn_qkv_count(s), qk = uint64_t(s.value_heads) * s.key_width, value = uint64_t(s.value_heads) * s.value_width;
    const auto v = allocate(g, std::array<uint64_t, 13>{raw, raw, qk, qk, value, value, s.value_heads, s.value_heads, s.value_heads, s.value_heads, value, value, value});
    return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12]};
}
void release_gdn_scratch(MetalGraph& g, const GdnScratch& s) { for (const auto v : s.views()) g.release(v); }
RecurrentScratch allocate_recurrent_scratch(MetalGraph& g, RecurrentShape s) {
    validate_recurrent_shape(s);
    RecurrentScratch out{};
    try {
        out.residual = allocate_residual_scratch(g, s.residual);
        out.gdn = allocate_gdn_scratch(g, s.gdn);
        out.moe = allocate_moe_scratch(g, s.moe);
        const auto v = allocate(g, std::array<uint64_t, 3>{uint64_t(s.residual.branches) * s.residual.channels, s.residual.channels, s.residual.channels});
        out.intermediate_streams = v[0]; out.attention_output = v[1]; out.ffn_output = v[2];
    } catch (...) {
        for (const auto v : out.views()) if (v.owner) g.release(v);
        throw;
    }
    return out;
}
void release_recurrent_scratch(MetalGraph& g, const RecurrentScratch& s) { for (const auto v : s.views()) g.release(v); }

void encode_residual_read(MetalGraph& g, const ResidualWeights& w, const ResidualScratch& s, GpuView streams, ResidualShape shape, float epsilon) {
    try {
        validate_residual_shape(shape);
        if (!std::isfinite(epsilon) || epsilon <= 0) throw FormatError("invalid residual RMS epsilon");
        const auto weight_views = residual_weights(w);
        std::array<GpuView, 5> inputs{streams, weight_views[0], weight_views[1], weight_views[2], weight_views[3]};
        g.validate_disjoint(inputs, s.views());

        TOKSPUR_OPERATOR_CHECKPOINT(g,"unclassified","before_hc",{1,shape.channels*shape.branches,shape.rank,shape.branches});
        g.rms(streams, w.gains, s.normalized, shape.channels, epsilon);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","norm",{1,shape.channels*shape.branches,shape.channels,shape.branches});
        g.gemv(w.down.encoding, w.down.view, s.normalized, s.bottleneck, shape.channels * shape.branches, shape.rank);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","down",{1,shape.channels*shape.branches,shape.rank,w.down.encoding});
        g.activate(s.bottleneck, s.bottleneck, Activation::silu, 1.0f / shape.branches);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","activation",{1,shape.rank,shape.branches,0});
        g.gemv(w.up.encoding, w.up.view, s.bottleneck, s.scores, shape.rank, shape.channels * shape.branches);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","up_gemv",{1,shape.rank,shape.channels*shape.branches,w.up.encoding});
        g.gemv(w.inject.encoding, w.inject.view, s.normalized, s.inject_logits, shape.channels * shape.branches, shape.branches);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","inject_gemv",{1,shape.channels*shape.branches,shape.branches,w.inject.encoding});
        g.residual_read(s.normalized, s.scores, s.inject_logits, s.input, s.scales, shape.branches, shape.channels);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","read",{1,shape.channels*shape.branches,shape.channels,shape.branches});
    } catch (...) { abort_on_failure(g); throw; }
}

void encode_gdn(MetalGraph& g, const GdnWeights& w, const GdnScratch& s, const GdnState& current, const GdnState& next,
                 GpuView input, GpuView output, GdnShape shape, GdnPolicy policy) {
    try {
        validate_gdn_shape(shape); validate_gdn_policy(policy);
        const auto weights = gdn_weights(w);
        std::array<GpuView, 12> reads{input, current.matrix, current.convolution};
        std::copy(weights.begin(), weights.end(), reads.begin() + 3);
        std::array<GpuView, 16> writes{output, next.matrix, next.convolution};
        const auto scratch = s.views(); std::copy(scratch.begin(), scratch.end(), writes.begin() + 3);
        g.validate_disjoint(reads, writes);
        const auto width = static_cast<uint32_t>(gdn_qkv_count(shape)), value = shape.value_heads * shape.value_width;
        g.gemv(w.qkv.encoding, w.qkv.view, input, s.projected, shape.channels, width);
        g.gemv(w.gate.encoding, w.gate.view, input, s.gate, shape.channels, value);
        g.gemv(w.alpha.encoding, w.alpha.view, input, s.alpha, shape.channels, shape.value_heads);
        g.gemv(w.beta.encoding, w.beta.view, input, s.beta, shape.channels, shape.value_heads);
        g.convolution(current.convolution, s.projected, w.convolution, next.convolution, s.convolved, shape.kernel);
        g.activate(s.convolved, s.convolved, Activation::silu);
        g.gdn_prepare(s.convolved, s.query, s.key, s.value, shape, policy);
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        if (g.gdn_use_ported()) {
            TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","projections_convolution_activation_prepare_gates",{1,shape.channels,shape.value_heads,shape.value_width});
            g.gdn_verify_main(current.matrix, s.query, s.key, s.value, s.alpha, s.beta, w.negative_rates, w.dt_bias,
                next.matrix, s.raw, s.decay, s.write_gate, shape.value_heads, shape.value_width, {1, 1}, false);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","core",{1,shape.value_heads,shape.key_width,shape.value_width});
            g.gdn_norm_gate(s.raw, w.gains, s.gate, s.gated, shape.value_heads, shape.value_width, policy.rms_epsilon, {1, 1});
            TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","rms_gate",{1,value,0,0});
            g.gemv(w.output.encoding, w.output.view, s.gated, output, value, shape.channels);
            TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","output_gemv",{1,value,shape.channels,w.output.encoding});
            return;
        }
#endif
        g.gdn_gates(s.alpha, s.beta, w.negative_rates, w.dt_bias, s.decay, s.write_gate);

        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","projections_convolution_activation_prepare_gates",{1,shape.channels,shape.value_heads,shape.value_width});
        g.delta(current.matrix, s.query, s.key, s.value, s.decay, s.write_gate, next.matrix, s.raw, shape.value_heads, shape.key_width, shape.value_width);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","core",{1,shape.value_heads,shape.key_width,shape.value_width});
        g.rms(s.raw, w.gains, s.normalized, shape.value_width, policy.rms_epsilon);
        g.multiply(s.normalized, s.gate, s.gated, true);

        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","rms_gate",{1,value,0,0});
        g.gemv(w.output.encoding, w.output.view, s.gated, output, value, shape.channels);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"gdn","output_gemv",{1,value,shape.channels,w.output.encoding});
    } catch (...) { abort_on_failure(g); throw; }
}

void encode_recurrent_layer(MetalGraph& g, const RecurrentWeights& w, const RecurrentScratch& s, const GdnState& current, const GdnState& next,
                             GpuView input, GpuView output, RecurrentShape shape, GdnPolicy policy, RouterMode mode, bool normalize) {
    try {
        validate_recurrent_shape(shape); validate_gdn_policy(policy);
        if (uint32_t(mode) > 1) throw FormatError("invalid recurrent router policy");
        std::array<GpuView, 28> reads{input, current.matrix, current.convolution};
        auto at = reads.begin() + 3;
        const auto rw1 = residual_weights(w.attention_read), rw2 = residual_weights(w.ffn_read);
        at = std::copy(rw1.begin(), rw1.end(), at); at = std::copy(rw2.begin(), rw2.end(), at);
        const auto gw = gdn_weights(w.attention); at = std::copy(gw.begin(), gw.end(), at);
        const std::array mw{w.ffn.router.view, w.ffn.gate.view, w.ffn.up.view, w.ffn.down.view, w.ffn.shared_gate.view,
            w.ffn.shared_up.view, w.ffn.shared_down.view, w.ffn.shared_router.view};
        std::copy(mw.begin(), mw.end(), at);
        std::array<GpuView, 35> writes{output, next.matrix, next.convolution};
        const auto scratch = s.views(); std::copy(scratch.begin(), scratch.end(), writes.begin() + 3);
        g.validate_disjoint(reads, writes);
        encode_residual_read(g, w.attention_read, s.residual, input, shape.residual, policy.rms_epsilon);
        encode_gdn(g, w.attention, s.gdn, current, next, s.residual.input, s.attention_output, shape.gdn, policy);
        g.residual_write(input, s.attention_output, s.residual.scales, s.intermediate_streams);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","attention_write",{1,shape.residual.channels*shape.residual.branches,shape.residual.channels,shape.residual.branches});
        encode_residual_read(g, w.ffn_read, s.residual, s.intermediate_streams, shape.residual, policy.rms_epsilon);
        encode_moe(g, w.ffn, s.moe, s.residual.input, s.ffn_output, shape.moe, mode, normalize);
        g.residual_write(s.intermediate_streams, s.ffn_output, s.residual.scales, output);
        TOKSPUR_OPERATOR_CHECKPOINT(g,"hc","ffn_write",{1,shape.residual.channels*shape.residual.branches,shape.residual.channels,shape.residual.branches});
    } catch (...) { abort_on_failure(g); throw; }
}

} // namespace tokspur
