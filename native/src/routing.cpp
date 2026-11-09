#include "tokspur/routing.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/operators.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tokspur {

Routes route_reference(std::span<const double> logits, size_t selected, RouterMode mode, bool normalize_selected) {
    if (logits.empty() || logits.size() > 4096 || !selected || selected > 64 || selected > logits.size() ||
        uint32_t(mode) > 1) throw FormatError("invalid router shape/mode");
    for (const auto value : logits) if (!std::isfinite(value)) throw FormatError("nonfinite router logit");
    std::vector<uint32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + static_cast<ptrdiff_t>(selected), order.end(), [&](uint32_t a, uint32_t b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    });
    order.resize(selected);
    const double maximum = logits[order[0]];
    const auto activate = [&](size_t index) {
        if (mode == RouterMode::softmax) return std::exp(logits[index] - maximum);
        // Selected sigmoid weights can all underflow before division although
        // their normalized ratios are representable. Work in log space first.
        if (normalize_selected) return std::exp(-stable_softplus(-logits[index]) + stable_softplus(-maximum));
        return stable_sigmoid(logits[index]);
    };
    Routes result{std::move(order), std::vector<double>(selected)};
    double denominator = mode == RouterMode::sigmoid ? 1 : 0;
    if (normalize_selected) {
        denominator = 0;
        for (const auto index : result.indices) denominator += activate(index);
    } else if (mode == RouterMode::softmax) {
        for (size_t i = 0; i < logits.size(); ++i) denominator += activate(i);
    }
    if (!std::isfinite(denominator) || denominator <= 0) throw FormatError("router normalization is not representable");
    for (size_t i = 0; i < selected; ++i) result.weights[i] = activate(result.indices[i]) / denominator;
    return result;
}

std::vector<double> mixture_reference(std::span<const double> experts, std::span<const double> weights,
                                       std::span<const double> shared, double shared_logit) {
    if (shared.empty() || shared.size() > 65536 || weights.empty() || weights.size() > 64 ||
        checked_mul(shared.size(), weights.size()) != experts.size()) throw FormatError("mixture shape mismatch");
    for (const auto weight : weights) if (!std::isfinite(weight) || weight < 0) throw FormatError("invalid route weight");
    const auto gate = stable_sigmoid(shared_logit);
    std::vector<double> result(shared.size());
    for (size_t channel = 0; channel < shared.size(); ++channel) {
        if (!std::isfinite(shared[channel])) throw FormatError("nonfinite shared expert output");
        double value = gate * shared[channel];
        for (size_t expert = 0; expert < weights.size(); ++expert) {
            const auto contribution = experts[expert * shared.size() + channel];
            if (!std::isfinite(contribution)) throw FormatError("nonfinite routed expert output");
            value += weights[expert] * contribution;
        }
        if (!std::isfinite(value)) throw FormatError("nonfinite mixture result");
        result[channel] = value;
    }
    return result;
}

} // namespace tokspur
