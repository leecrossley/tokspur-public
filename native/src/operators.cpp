#include "tokspur/operators.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cmath>

namespace tokspur {
namespace {

void finite(std::span<const double> values) {
    for (const auto value : values) if (!std::isfinite(value)) throw FormatError("nonfinite operator input/result");
}

void scalar(double value) { if (!std::isfinite(value)) throw FormatError("nonfinite operator scalar"); }

std::vector<double> matrix_vector(std::span<const double> matrix, std::span<const double> input, size_t rows) {
    if (!rows || checked_mul(rows, input.size()) != matrix.size()) throw FormatError("reference matrix shape mismatch");
    std::vector<double> output(rows, 0);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < input.size(); ++column) output[row] += matrix[row * input.size() + column] * input[column];
    }
    finite(output);
    return output;
}

} // namespace

double stable_sigmoid(double value) {
    scalar(value);
    if (value >= 0) return 1 / (1 + std::exp(-value));
    const auto positive = std::exp(value);
    return positive / (1 + positive);
}

double stable_silu(double value) { return value * stable_sigmoid(value); }

double stable_softplus(double value) {
    scalar(value);
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

std::vector<double> group_rms_reference(std::span<const double> input, std::span<const double> gains,
                                        size_t group_width, double epsilon) {
    if (!group_width || input.empty() || input.size() != gains.size() || input.size() % group_width ||
        !std::isfinite(epsilon) || epsilon <= 0) throw FormatError("invalid grouped RMS shape/epsilon");
    finite(input);
    finite(gains);
    std::vector<double> output(input.size());
    for (size_t group = 0; group < input.size(); group += group_width) {
        double squared = 0;
        for (size_t i = 0; i < group_width; ++i) squared += input[group + i] * input[group + i];
        if (!std::isfinite(squared)) throw FormatError("RMS reference overflow");
        const auto inverse = 1 / std::sqrt(squared / group_width + epsilon);
        for (size_t i = 0; i < group_width; ++i) output[group + i] = input[group + i] * inverse * gains[group + i];
    }
    finite(output);
    return output;
}

ResidualRead gated_residual_reference(std::span<const double> streams, std::span<const double> gains,
                                      std::span<const double> down, std::span<const double> up,
                                      std::span<const double> inject, size_t branches, size_t channels,
                                      size_t rank, double epsilon) {
    const auto width = checked_mul(branches, channels);
    if (!branches || branches > 64 || !channels || channels > 65536 || !rank || rank > 4096 || streams.size() != width ||
        down.size() != checked_mul(rank, width) || up.size() != checked_mul(width, rank) ||
        inject.size() != checked_mul(branches, width)) throw FormatError("gated residual reference shape mismatch");
    finite(down); finite(up); finite(inject);
    const auto normalized = group_rms_reference(streams, gains, channels, epsilon);
    auto bottleneck = matrix_vector(down, normalized, rank);
    for (auto& value : bottleneck) value = stable_silu(value / branches);
    auto scores = matrix_vector(up, bottleneck, static_cast<size_t>(width));
    auto scales = matrix_vector(inject, normalized, branches);
    ResidualRead result{std::vector<double>(channels, 0), std::vector<double>(branches)};
    for (size_t branch = 0; branch < branches; ++branch) {
        result.write_scales[branch] = 2 * stable_sigmoid(scales[branch] / branches);
        for (size_t channel = 0; channel < channels; ++channel) {
            const auto position = branch * channels + channel;
            result.input[channel] += normalized[position] * stable_sigmoid(scores[position]) / branches;
        }
    }
    finite(result.input);
    return result;
}

std::vector<double> residual_write_reference(std::span<const double> streams, std::span<const double> output,
                                              std::span<const double> scales) {
    if (output.empty() || scales.empty() || checked_mul(output.size(), scales.size()) != streams.size()) throw FormatError("residual write shape mismatch");
    finite(streams); finite(output); finite(scales);
    std::vector<double> result(streams.begin(), streams.end());
    for (size_t branch = 0; branch < scales.size(); ++branch) {
        for (size_t channel = 0; channel < output.size(); ++channel) result[branch * output.size() + channel] += scales[branch] * output[channel];
    }
    finite(result);
    return result;
}

DeltaStep gated_delta_reference(std::span<const double> state, std::span<const double> query,
                                std::span<const double> key, std::span<const double> value,
                                std::span<const double> decay, std::span<const double> beta,
                                size_t heads, size_t key_width, size_t value_width) {
    const auto qk_count = checked_mul(heads, key_width);
    const auto v_count = checked_mul(heads, value_width);
    if (!heads || heads > 64 || !key_width || key_width > 256 || !value_width || value_width > 256 ||
        state.size() != checked_mul(qk_count, value_width) || query.size() != qk_count || key.size() != qk_count ||
        value.size() != v_count || decay.size() != heads || beta.size() != heads) throw FormatError("GDN reference shape mismatch");
    finite(state); finite(query); finite(key); finite(value); finite(decay); finite(beta);
    for (size_t h = 0; h < heads; ++h) if (decay[h] < 0 || decay[h] > 1 || beta[h] < 0 || beta[h] > 1) throw FormatError("GDN gates outside unit interval");
    DeltaStep result{std::vector<double>(state.size()), std::vector<double>(v_count, 0)};
    std::vector<double> error(value_width);
    for (size_t head = 0; head < heads; ++head) {
        const auto state_base = head * key_width * value_width;
        for (size_t v = 0; v < value_width; ++v) {
            double predicted = 0;
            for (size_t k = 0; k < key_width; ++k) predicted += decay[head] * state[state_base + k * value_width + v] * key[head * key_width + k];
            error[v] = value[head * value_width + v] - predicted;
        }
        for (size_t k = 0; k < key_width; ++k) {
            for (size_t v = 0; v < value_width; ++v) {
                const auto at = state_base + k * value_width + v;
                result.state[at] = decay[head] * state[at] + beta[head] * key[head * key_width + k] * error[v];
                result.output[head * value_width + v] += result.state[at] * query[head * key_width + k];
            }
        }
    }
    finite(result.state); finite(result.output);
    return result;
}

ConvStep causal_depthwise_reference(std::span<const double> history, std::span<const double> input,
                                    std::span<const double> weights, size_t kernel, size_t dilation) {
    if (!kernel || kernel > 64 || !dilation || dilation > 16 || input.empty() || input.size() > 65536 ||
        history.size() != checked_mul(checked_mul(kernel - 1, dilation), input.size()) || weights.size() != checked_mul(kernel, input.size())) {
        throw FormatError("causal convolution reference shape mismatch");
    }
    finite(history); finite(input); finite(weights);
    ConvStep result{std::vector<double>(history.size()), std::vector<double>(input.size(), 0)};
    for (size_t channel = 0; channel < input.size(); ++channel) {
        for (size_t t = 0; t + 1 < kernel; ++t) result.output[channel] += history[t * dilation * input.size() + channel] * weights[channel * kernel + t];
        result.output[channel] += input[channel] * weights[channel * kernel + kernel - 1];
    }
    if (kernel > 1) {
        std::copy(history.begin() + static_cast<ptrdiff_t>(input.size()), history.end(), result.history.begin());
        std::copy(input.begin(), input.end(), result.history.end() - static_cast<ptrdiff_t>(input.size()));
    }
    finite(result.output);
    return result;
}

} // namespace tokspur
