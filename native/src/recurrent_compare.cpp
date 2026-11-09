#include "tokspur/recurrent_check.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tokspur {
namespace {
template<class T> std::vector<T> read(MetalGraph& g, GpuView v) {
    const auto data = g.read(v);
    if (data.size() % sizeof(T)) throw FormatError("unaligned layer diagnostic readback");
    std::vector<T> out(data.size() / sizeof(T)); std::memcpy(out.data(), data.data(), data.size()); return out;
}
}
LayerComparison compare_recurrent_layer(MetalGraph& g, const RecurrentScratch& s, const GdnState& next, GpuView output, const ReferenceLayer& e) {
    LayerComparison out{{}, false, 0, 0, false};
    const auto check = [&](const char* name, GpuView view, std::span<const double> expected) {
        const auto actual = read<float>(g, view);
        if (actual.size() != expected.size()) throw FormatError("layer diagnostic reference shape mismatch");
        double maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) throw FormatError("nonfinite layer diagnostic value");
            maximum = std::max(maximum, std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i])));
        }
        out.compared_values += actual.size(); out.maximum = std::max(out.maximum, maximum);
        out.scaled_errors.emplace_back(name, maximum);
    };
    check("output_streams", output, e.output_streams);
    check("intermediate_streams", s.intermediate_streams, e.intermediate_streams);
    check("attention_output", s.attention_output, e.attention.output);
    check("ffn_output", s.ffn_output, e.ffn.output);
    check("ffn_read_normalized", s.residual.normalized, e.ffn_read.normalized);
    check("ffn_read_bottleneck", s.residual.bottleneck, e.ffn_read.bottleneck);
    check("ffn_read_scores", s.residual.scores, e.ffn_read.scores);
    check("ffn_read_inject", s.residual.inject_logits, e.ffn_read.inject_logits);
    check("ffn_read_input", s.residual.input, e.ffn_read.read.input);
    check("ffn_read_scales", s.residual.scales, e.ffn_read.read.write_scales);
    check("gdn_projected", s.gdn.projected, e.attention.projected);
    auto activated = e.attention.convolution.output;
    for (auto& x : activated) x = stable_silu(x);
    check("gdn_convolved_activated", s.gdn.convolved, activated);
    check("gdn_query", s.gdn.query, e.attention.prepared.query);
    check("gdn_key", s.gdn.key, e.attention.prepared.key);
    check("gdn_value", s.gdn.value, e.attention.prepared.value);
    check("gdn_gate", s.gdn.gate, e.attention.gate);
    check("gdn_alpha", s.gdn.alpha, e.attention.alpha);
    check("gdn_beta_logits", s.gdn.beta, e.attention.beta);
    check("gdn_decay", s.gdn.decay, e.attention.gates.decay);
    check("gdn_write_gate", s.gdn.write_gate, e.attention.gates.beta);
    check("gdn_raw", s.gdn.raw, e.attention.delta.output);
    check("gdn_normalized", s.gdn.normalized, e.attention.normalized);
    check("gdn_gated", s.gdn.gated, e.attention.gated);
    check("next_recurrent_state", next.matrix, e.attention.delta.state);
    check("next_convolution_history", next.convolution, e.attention.convolution.history);
    check("moe_logits", s.moe.logits, e.ffn.logits);
    check("moe_route_weights", s.moe.weights, e.ffn.routes.weights);
    check("moe_activated", s.moe.gate, e.ffn.activated);
    check("moe_routed", s.moe.down, e.ffn.routed);
    check("moe_shared_activated", s.moe.shared_gate, e.ffn.shared_activated);
    check("moe_shared_output", s.moe.shared_down, e.ffn.shared_output);
    out.routes_match = read<uint32_t>(g, s.moe.indices) == e.ffn.routes.indices;
    out.passed = out.routes_match && out.maximum <= 5e-5;
    return out;
}
} // namespace tokspur
