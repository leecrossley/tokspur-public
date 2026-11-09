#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tokspur {

enum class RouterMode : uint32_t { softmax, sigmoid };

struct Routes {
    std::vector<uint32_t> indices;
    std::vector<double> weights;
};

// Select by descending finite logit, breaking exact ties by ascending expert ID.
// Activation and selected-weight normalization are explicit: checkpoint-specific
// router semantics must be qualified rather than silently inferred from shapes.
Routes route_reference(std::span<const double> logits, size_t selected, RouterMode mode, bool normalize_selected);

// Expert results are [selected,channel]; the shared expert uses a scalar sigmoid
// gate. This is a mathematical reference, not an external tensor implementation.
std::vector<double> mixture_reference(std::span<const double> experts, std::span<const double> route_weights,
                                       std::span<const double> shared, double shared_logit);

} // namespace tokspur
