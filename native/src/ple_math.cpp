#include "tokspur/ple.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <cmath>

namespace tokspur {
void validate_ple_shape(PleShape s) {
    if (!s.branches || s.branches > 64 || !s.channels || s.channels > 65536 || uint64_t(s.branches) * s.channels > 65536 ||
        !s.embedding || s.embedding > 65536 || s.kernel < 2 || s.kernel > 64 || !s.dilation || s.dilation > 16 ||
        uint32_t(s.gate_transform) > uint32_t(PleGateTransform::signed_root)) throw FormatError("unsupported PLE geometry");
}
double ple_gate_reference(double score, PleGateTransform transform) {
    if (!std::isfinite(score) || uint32_t(transform) > uint32_t(PleGateTransform::signed_root))
        throw FormatError("invalid PLE gate score/transform");
    if (transform == PleGateTransform::signed_root && score != 0)
        score = std::copysign(std::sqrt(std::max(std::abs(score), 1e-6)), score);
    return stable_sigmoid(score);
}
ReferencePle ple_reference(const ReferencePleWeights& w, std::span<const double> embedding, std::span<const double> streams,
                             std::span<const double> history, PleShape s, double epsilon) {
    validate_ple_shape(s);
    const auto width = s.branches * s.channels;
    if (embedding.size() != s.embedding || streams.size() != width) throw FormatError("PLE reference input geometry mismatch");
    ReferencePle out;
    out.key = reference_gemv_doubleinput(w.key.encoding, w.key.bytes, s.embedding, width, embedding);
    out.value = reference_gemv_doubleinput(w.value.encoding, w.value.bytes, s.embedding, s.channels, embedding);
    out.normalized_key = group_rms_reference(out.key, w.key_gains, s.channels, epsilon);
    out.normalized_query = group_rms_reference(streams, w.query_gains, s.channels, epsilon);
    out.gates.resize(s.branches); out.gated.resize(width);
    for (uint32_t b = 0; b < s.branches; ++b) {
        double dot = 0;
        for (uint32_t c = 0; c < s.channels; ++c) dot += out.normalized_key[b * s.channels + c] * out.normalized_query[b * s.channels + c];
        out.gates[b] = ple_gate_reference(dot / std::sqrt(double(s.channels)), s.gate_transform);
        for (uint32_t c = 0; c < s.channels; ++c) out.gated[b * s.channels + c] = out.gates[b] * out.value[c];
    }
    out.normalized_gated = group_rms_reference(out.gated, w.conv_gains, s.channels, epsilon);
    out.convolution = causal_depthwise_reference(history, out.normalized_gated, w.convolution, s.kernel, s.dilation);
    out.output.resize(width);
    for (uint32_t i = 0; i < width; ++i) out.output[i] = streams[i] + out.gated[i] + stable_silu(out.convolution.output[i]);
    return out;
}
ReferenceMixer mixer_reference(std::span<const double> gains, ReferenceMatrix down, ReferenceMatrix up,
                                 std::span<const double> streams, ResidualShape s, double epsilon) {
    if (!s.branches || s.branches > 64 || !s.channels || s.channels > 65536 || uint64_t(s.branches) * s.channels > 65536 || !s.rank || s.rank > 4096 || streams.size() != uint64_t(s.branches) * s.channels) throw FormatError("final mixer reference geometry mismatch");
    ReferenceMixer out;
    out.normalized = group_rms_reference(streams, gains, s.channels, epsilon);
    out.bottleneck = reference_gemv_doubleinput(down.encoding, down.bytes, s.branches * s.channels, s.rank, out.normalized);
    for (auto& x : out.bottleneck) x = stable_silu(x / s.branches);
    out.scores = reference_gemv_doubleinput(up.encoding, up.bytes, s.rank, s.branches * s.channels, out.bottleneck);
    out.mixed.resize(s.channels, 0);
    for (uint32_t b = 0; b < s.branches; ++b) for (uint32_t c = 0; c < s.channels; ++c) {
        const auto i = b * s.channels + c; out.mixed[c] += out.normalized[i] * stable_sigmoid(out.scores[i]) / s.branches;
    }
    return out;
}
} // namespace tokspur
