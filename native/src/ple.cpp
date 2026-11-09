#include "tokspur/ple.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cmath>

namespace tokspur {
namespace {
template<size_t N> std::array<GpuView, N> allocate(MetalGraph& g, const std::array<uint64_t, N>& counts) {
    std::array<GpuView, N> out{}; size_t made = 0;
    try { for (; made < N; ++made) out[made] = g.zeros(checked_mul(counts[made], 4), "PLE/mixer scratch"); }
    catch (...) { for (size_t i = 0; i < made; ++i) g.release(out[i]); throw; }
    return out;
}
void abort(MetalGraph& g) { try { g.abort(); } catch (const FormatError&) {} }
}
std::array<GpuView, 8> PleScratch::views() const { return {key, value, normalized_key, normalized_query, gated, gates, normalized_gated, convolved}; }
PleScratch allocate_ple_scratch(MetalGraph& g, PleShape s) {
    validate_ple_shape(s); const uint64_t n = s.branches * s.channels;
    const auto v = allocate(g, std::array<uint64_t, 8>{n, s.channels, n, n, n, s.branches, n, n});
    return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
}
void release_ple_scratch(MetalGraph& g, const PleScratch& s) { for (const auto v : s.views()) g.release(v); }
void encode_ple(MetalGraph& g, const PleWeights& w, const PleScratch& s, GpuView embedding, GpuView streams,
                  GpuView history, GpuView next, GpuView output, PleShape shape, float epsilon) {
    try {
        validate_ple_shape(shape);
        if (!std::isfinite(epsilon) || epsilon <= 0) throw FormatError("invalid PLE RMS epsilon");
        const std::array reads{embedding, streams, history, w.key.view, w.value.view, w.key_gains, w.query_gains, w.conv_gains, w.convolution};
        std::array<GpuView, 10> writes{next, output}; const auto sv = s.views(); std::copy(sv.begin(), sv.end(), writes.begin() + 2);
        g.validate_disjoint(reads, writes);
        const auto width = shape.branches * shape.channels;
        g.gemv(w.key.encoding, w.key.view, embedding, s.key, shape.embedding, width);
        g.gemv(w.value.encoding, w.value.view, embedding, s.value, shape.embedding, shape.channels);
        g.rms(s.key, w.key_gains, s.normalized_key, shape.channels, epsilon);
        g.rms(streams, w.query_gains, s.normalized_query, shape.channels, epsilon);
        g.ple_gate(s.normalized_query, s.normalized_key, s.value, s.gated, s.gates, shape.branches, shape.channels, shape.gate_transform);
        g.rms(s.gated, w.conv_gains, s.normalized_gated, shape.channels, epsilon);
        g.convolution(history, s.normalized_gated, w.convolution, next, s.convolved, shape.kernel, shape.dilation);
        g.ple_write(streams, s.gated, s.convolved, output);
    } catch (...) { abort(g); throw; }
}
std::array<GpuView, 4> MixerScratch::views() const { return {normalized, bottleneck, scores, mixed}; }
MixerScratch allocate_mixer_scratch(MetalGraph& g, ResidualShape s) {
    validate_residual_shape(s); const uint64_t n = s.branches * s.channels;
    const auto v = allocate(g, std::array<uint64_t, 4>{n, s.rank, n, s.channels});
    return {v[0], v[1], v[2], v[3]};
}
void release_mixer_scratch(MetalGraph& g, const MixerScratch& s) { for (const auto v : s.views()) g.release(v); }
void encode_mixer(MetalGraph& g, const MixerWeights& w, const MixerScratch& s, GpuView streams, ResidualShape shape, float epsilon) {
    try {
        validate_residual_shape(shape);
        g.validate_disjoint(std::array{streams, w.gains, w.down.view, w.up.view}, s.views());
        g.rms(streams, w.gains, s.normalized, shape.channels, epsilon);
        g.gemv(w.down.encoding, w.down.view, s.normalized, s.bottleneck, shape.branches * shape.channels, shape.rank);
        g.activate(s.bottleneck, s.bottleneck, Activation::silu, 1.0f / shape.branches);
        g.gemv(w.up.encoding, w.up.view, s.bottleneck, s.scores, shape.rank, shape.branches * shape.channels);
        g.gated_mix(s.normalized, s.scores, s.mixed, shape.branches, shape.channels);
    } catch (...) { abort(g); throw; }
}
} // namespace tokspur
