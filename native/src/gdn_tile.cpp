#include "tokspur/gdn_tile.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/operators.hpp"

#include <algorithm>
#include <utility>

namespace tokspur {
namespace {
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
void count(std::span<const double> input, uint64_t float_bytes) {
    require(input.size() == float_bytes / 4, "GDN tile reference buffer geometry mismatch");
}
void returned(uint64_t state_count, uint64_t output_count, uint64_t maximum) {
    require(checked_mul(checked_add(state_count, output_count), 8) <= maximum,
        "GDN tile reference returned-data budget exceeded");
}
}
void validate_state_tile(StateTile t) {
    require((t.capacity == 2 || t.capacity == 4 || t.capacity == 8 || t.capacity == 16 || t.capacity == 32
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        || t.capacity == 512
#endif
        ) &&
        t.steps && t.steps <= t.capacity, "unsupported GDN state tile capacity/steps");
}
GdnConvTileBytes gdn_conv_tile_bytes(uint32_t channels, StateTile t) {
    validate_state_tile(t); require(channels && channels <= 65536, "unsupported GDN tile convolution channels");
    return {checked_mul(channels, 12), checked_mul(checked_mul(channels, t.capacity), 4),
        checked_mul(channels, 16), checked_mul(checked_mul(channels, t.capacity), 4)};
}
GdnDeltaTileBytes gdn_delta_tile_bytes(uint32_t heads, uint32_t value_width, StateTile t) {
    validate_state_tile(t);
    require(heads && heads <= 48 && value_width && value_width <= 128, "unsupported GDN tile delta shape");
    return {checked_mul(checked_mul(checked_mul(heads, 128), value_width), 4),
        checked_mul(checked_mul(checked_mul(t.capacity, heads), 128), 4),
        checked_mul(checked_mul(checked_mul(t.capacity, heads), value_width), 4),
        checked_mul(checked_mul(t.capacity, heads), 4)};
}
ConvTileReference gdn_convolution_tile_reference(
    std::span<const double> history, std::span<const double> projected,
    std::span<const double> weights, uint32_t channels, StateTile t,
    uint64_t maximum) {
    const auto b = gdn_conv_tile_bytes(channels, t);
    count(history, b.history); count(projected, b.projected); count(weights, b.weights);
    const auto outputs = checked_mul(channels, t.steps); returned(history.size(), outputs, maximum);
    ConvTileReference out{{history.begin(), history.end()}, std::vector<double>(outputs)};
    for (uint32_t step = 0; step < t.steps; ++step) {
        auto next = causal_depthwise_reference(out.history, projected.subspan(uint64_t(step) * channels, channels), weights, 4, 1);
        std::copy(next.output.begin(), next.output.end(), out.output.begin() + uint64_t(step) * channels);
        out.history = std::move(next.history);
    }
    return out;
}
DeltaTileReference gdn_delta_tile_reference(
    std::span<const double> state, std::span<const double> query,
    std::span<const double> key, std::span<const double> value,
    std::span<const double> decay, std::span<const double> beta,
    uint32_t heads, uint32_t width, StateTile t, uint64_t maximum) {
    const auto b = gdn_delta_tile_bytes(heads, width, t);
    count(state, b.state); count(query, b.query_key); count(key, b.query_key);
    count(value, b.value_output); count(decay, b.gates); count(beta, b.gates);
    const auto stride = uint64_t(heads) * width, qstride = uint64_t(heads) * 128;
    returned(state.size(), checked_mul(stride, t.steps), maximum);
    DeltaTileReference out{{state.begin(), state.end()}, std::vector<double>(stride * t.steps)};
    for (uint32_t step = 0; step < t.steps; ++step) {
        auto next = gated_delta_reference(out.state, query.subspan(step * qstride, qstride), key.subspan(step * qstride, qstride),
            value.subspan(step * stride, stride), decay.subspan(uint64_t(step) * heads, heads),
            beta.subspan(uint64_t(step) * heads, heads), heads, 128, width);
        std::copy(next.output.begin(), next.output.end(), out.output.begin() + step * stride);
        out.state = std::move(next.state);
    }
    return out;
}
} // namespace tokspur
