#include "tokspur/recurrent_reference.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <cmath>

namespace tokspur {
namespace {
std::vector<double> project(ReferenceMatrix matrix, std::span<const double> x, uint32_t rows) {
    if (x.empty() || x.size() > 65536) throw FormatError("reference projection width limit");
    return reference_gemv_doubleinput(matrix.encoding, matrix.bytes, static_cast<uint32_t>(x.size()), rows, x);
}
ReferenceMatrix expert(ReferenceMatrix matrix, uint32_t index, uint32_t count) {
    if (!count || index >= count || matrix.bytes.size() % count) throw FormatError("reference expert span mismatch");
    const auto stride = matrix.bytes.size() / count;
    return {matrix.bytes.subspan(index * stride, stride), matrix.encoding};
}
ReferenceResidual residual(const ReferenceResidualWeights& w, std::span<const double> streams, ResidualShape s, double epsilon) {
    // Kept separate from the GPU assembler: ordinary loops and scalar references
    // express the equations, rather than replaying a GPU command list on a CPU.
    if (streams.size() != uint64_t(s.branches) * s.channels) throw FormatError("reference residual width mismatch");
    ReferenceResidual out;
    out.normalized = group_rms_reference(streams, w.gains, s.channels, epsilon);
    out.bottleneck = project(w.down, out.normalized, s.rank);
    for (auto& x : out.bottleneck) x = stable_silu(x / s.branches);
    out.scores = project(w.up, out.bottleneck, s.branches * s.channels);
    out.inject_logits = project(w.inject, out.normalized, s.branches);
    out.read = {std::vector<double>(s.channels, 0), std::vector<double>(s.branches)};
    for (uint32_t b = 0; b < s.branches; ++b) {
        out.read.write_scales[b] = 2 * stable_sigmoid(out.inject_logits[b] / s.branches);
        for (uint32_t d = 0; d < s.channels; ++d) out.read.input[d] += out.normalized[b * s.channels + d] * stable_sigmoid(out.scores[b * s.channels + d]) / s.branches;
    }
    return out;
}
ReferenceGdn gdn(const ReferenceGdnWeights& w, std::span<const double> x, std::span<const double> state,
                 std::span<const double> history, GdnShape s, GdnPolicy p) {
    ReferenceGdn out;
    out.projected = project(w.qkv, x, static_cast<uint32_t>(gdn_qkv_count(s)));
    out.gate = project(w.gate, x, s.value_heads * s.value_width);
    out.alpha = project(w.alpha, x, s.value_heads); out.beta = project(w.beta, x, s.value_heads);
    out.convolution = causal_depthwise_reference(history, out.projected, w.convolution, s.kernel);
    auto activated = out.convolution.output;
    for (auto& y : activated) y = stable_silu(y);
    out.prepared = prepare_gdn_reference(activated, s, p);
    out.gates = gdn_gates_reference(out.alpha, out.beta, w.negative_rates, w.dt_bias);
    out.delta = gated_delta_reference(state, out.prepared.query, out.prepared.key, out.prepared.value,
        out.gates.decay, out.gates.beta, s.value_heads, s.key_width, s.value_width);
    if (w.gains.size() != s.value_width) throw FormatError("reference GDN gain width mismatch");
    std::vector<double> gains(out.delta.output.size());
    for (size_t i = 0; i < gains.size(); ++i) gains[i] = w.gains[i % w.gains.size()];
    out.normalized = group_rms_reference(out.delta.output, gains, s.value_width, p.rms_epsilon);
    out.gated = out.normalized;
    for (size_t i = 0; i < out.gated.size(); ++i) out.gated[i] *= stable_sigmoid(out.gate[i]);
    out.output = project(w.output, out.gated, s.channels);
    return out;
}
ReferenceMoe moe(const ReferenceMoeWeights& w, std::span<const double> x, MoeShape s, RouterMode mode, bool normalize) {
    ReferenceMoe out;
    out.logits = project(w.router, x, s.experts);
    out.routes = route_reference(out.logits, s.selected, mode, normalize);
    for (const auto id : out.routes.indices) {
        auto gate = project(expert(w.gate, id, s.experts), x, s.hidden);
        const auto up = project(expert(w.up, id, s.experts), x, s.hidden);
        for (size_t i = 0; i < gate.size(); ++i) gate[i] = stable_silu(gate[i]) * up[i];
        const auto down = project(expert(w.down, id, s.experts), gate, s.channels);
        out.activated.insert(out.activated.end(), gate.begin(), gate.end()); out.routed.insert(out.routed.end(), down.begin(), down.end());
    }
    out.shared_activated = project(w.shared_gate, x, s.hidden);
    const auto up = project(w.shared_up, x, s.hidden);
    for (size_t i = 0; i < up.size(); ++i) out.shared_activated[i] = stable_silu(out.shared_activated[i]) * up[i];
    out.shared_output = project(w.shared_down, out.shared_activated, s.channels);
    out.output = mixture_reference(out.routed, out.routes.weights, out.shared_output, project(w.shared_router, x, 1)[0]);
    return out;
}
}
ReferenceResidual residual_projection_reference(const ReferenceResidualWeights& w, std::span<const double> streams, ResidualShape s, double epsilon) {
    return residual(w, streams, s, epsilon);
}
ReferenceGdn gdn_projection_reference(const ReferenceGdnWeights& w, std::span<const double> input,
                                      std::span<const double> state, std::span<const double> history,
                                      GdnShape shape, GdnPolicy policy) {
    validate_gdn_shape(shape); validate_gdn_policy(policy);
    if (input.size() != shape.channels || state.size() != gdn_state_count(shape) ||
        history.size() != gdn_qkv_count(shape) * (shape.kernel - 1)) throw FormatError("reference GDN core geometry mismatch");
    return gdn(w, input, state, history, shape, policy);
}
ReferenceMoe moe_projection_reference(const ReferenceMoeWeights& w, std::span<const double> input, MoeShape s, RouterMode mode, bool normalize) {
    return moe(w, input, s, mode, normalize);
}
ReferenceLayer recurrent_reference(const ReferenceLayerWeights& w, std::span<const double> streams,
                                    std::span<const double> state, std::span<const double> history,
                                    RecurrentShape s, GdnPolicy p, RouterMode mode, bool normalize) {
    // Shape guards here don't call the Metal-facing validation implementation.
    validate_gdn_shape(s.gdn); validate_gdn_policy(p);
    if (!s.residual.branches || s.residual.branches > 64 || !s.residual.channels || s.residual.channels > 65536 ||
        !s.residual.rank || s.residual.rank > 4096 || s.residual.channels != s.gdn.channels || s.moe.channels != s.gdn.channels ||
        !s.moe.hidden || s.moe.hidden > 65536 || !s.moe.experts || s.moe.experts > 4096 || !s.moe.selected || s.moe.selected > 64 || s.moe.selected > s.moe.experts) throw FormatError("reference recurrent shape mismatch");
    ReferenceLayer out;
    out.attention_read = residual(w.attention_read, streams, s.residual, p.rms_epsilon);
    out.attention = gdn(w.attention, out.attention_read.read.input, state, history, s.gdn, p);
    out.intermediate_streams = residual_write_reference(streams, out.attention.output, out.attention_read.read.write_scales);
    out.ffn_read = residual(w.ffn_read, out.intermediate_streams, s.residual, p.rms_epsilon);
    out.ffn = moe(w.ffn, out.ffn_read.read.input, s.moe, mode, normalize);
    out.output_streams = residual_write_reference(out.intermediate_streams, out.ffn.output, out.ffn_read.read.write_scales);
    return out;
}
} // namespace tokspur
