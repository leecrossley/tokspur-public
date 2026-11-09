#pragma once

// Internal CPU test/implementation surface; not a serving or TextStack API.
#include "tokspur/gguf.hpp"
#include "tokspur/sampled_path.hpp"

#include <vector>

namespace tokspur::sampled_path_detail {

inline constexpr uint32_t maximum_uniform_attempts = 64;

// Rejection partitions the U53 grid into equally sized residue classes. Never
// approximate a rational acceptance probability by a rounded U53 threshold.
// Exhaustion is an unpublished operational refusal, not a fallback sample.
template<class NextWord, class Poll>
uint64_t uniform_below(uint64_t upper, NextWord&& next, Poll&& poll) {
    if (!upper || upper > sampled_path_grid_size) throw FormatError("sampled path unsupported integer draw bound");
    if (upper == 1) return 0;
    const uint64_t limit = sampled_path_grid_size - sampled_path_grid_size % upper;
    for (uint32_t attempt = 0; attempt < maximum_uniform_attempts; ++attempt) {
        poll();
        const uint64_t word = next();
        if (word >= sampled_path_grid_size) throw FormatError("sampled path entropy is outside native U53 grid");
        if (word < limit) return word % upper;
    }
    throw FormatError("sampled path integer entropy budget exhausted; no publication");
}

struct SparseTokenMass {
    uint32_t token_id = 0;
    uint64_t mass = 0;
};

// Count the exact U53 inverse-CDF intervals of the EXISTING sample_at primitive,
// including its final roundoff-gap fallback. This does not change its policy.
[[nodiscard]] std::vector<uint64_t> target_grid_units(const SamplingDistribution& distribution);
[[nodiscard]] std::vector<SparseTokenMass> target_sparse_grid_units(const SamplingDistribution& distribution);

template<class Next, class Poll>
uint32_t draw_mass_sparse(std::span<const SparseTokenMass> masses, Next&& next, Poll&& poll) {
    uint64_t total = 0;
    uint32_t positives = 0, last = 0;
    for (size_t i = 0; i < masses.size(); ++i) {
        if (!masses[i].mass) continue;
        if (masses[i].mass > sampled_path_grid_size - total) throw FormatError("sampled path mass overflow");
        total += masses[i].mass;
        ++positives;
        last = masses[i].token_id;
    }
    if (!total || !positives) throw FormatError("sampled path collapsed residual; no fallback");
    if (positives == 1) return last;
    const uint64_t ticket = uniform_below(total, next, poll);
    uint64_t cumulative = 0;
    for (size_t i = 0; i < masses.size(); ++i) {
        cumulative += masses[i].mass;
        if (ticket < cumulative) return masses[i].token_id;
    }
    throw FormatError("sampled path integer CDF exhausted; no fallback");
}

} // namespace tokspur::sampled_path_detail
