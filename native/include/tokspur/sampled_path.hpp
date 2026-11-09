#pragma once

#include "tokspur/sampling.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace tokspur {

inline constexpr uint32_t sampled_path_maximum_proposals = 7;
// Admit the production top-k20 proposal law without truncating or renormalizing
// its actual support. The U53 grid and probability/entropy algorithms are unchanged.
inline constexpr uint32_t sampled_path_maximum_support = 20;
inline constexpr uint32_t sampled_path_maximum_stops = 64;
inline constexpr uint64_t sampled_path_grid_size = uint64_t{1} << 53;

// This identifies the proposal sampler, not just its storage precision. Other
// samplers (a framework multinomial or categorical sampler, or a selector confidence) do not
// acquire this identity by converting their returned probabilities to double.
enum class SampledPathGrid : uint32_t { native_u53_inverse_cdf_v1 = 1 };

// A bounded actual categorical law over the same 53-bit uniform grid as
// SamplingRng::uniform(). Input order defines the proposal inverse CDF. Zero
// masses are removed; duplicates, unsupported grids and malformed laws fail.
// No normalization or new sampling filter is silently applied to the input.
class SampledSparseLaw {
public:
    static SampledSparseLaw from_masses(uint32_t vocabulary, std::span<const uint32_t> ids,
        std::span<const double> probabilities, SampledPathGrid grid = SampledPathGrid::native_u53_inverse_cdf_v1);
    static SampledSparseLaw from_grid_units(uint32_t vocabulary, std::span<const uint32_t> ids,
        std::span<const uint64_t> grid_units, SampledPathGrid grid = SampledPathGrid::native_u53_inverse_cdf_v1);
    [[nodiscard]] uint32_t vocabulary() const noexcept { return vocabulary_; }
    [[nodiscard]] uint32_t size() const noexcept { return size_; }
    // Read-only sparse bins, in the exact proposal CDF order. These accessors
    // do not alter the b66 rejection sampler or expose mutable law storage.
    [[nodiscard]] std::span<const uint32_t> identifiers() const noexcept { return {ids_.data(), size_}; }
    [[nodiscard]] std::span<const uint64_t> mass_units() const noexcept { return {units_.data(), size_}; }
    [[nodiscard]] uint64_t units(uint32_t token) const noexcept;
    [[nodiscard]] uint32_t sample_at(uint64_t uniform_grid_index) const;
    [[nodiscard]] bool deterministic() const noexcept;

private:
    uint32_t vocabulary_ = 0, size_ = 0;
    std::array<uint32_t, sampled_path_maximum_support> ids_{};
    std::array<uint64_t, sampled_path_maximum_support> units_{};
};

// The owner supplies these identities; this component cannot establish that
// logits really came from those weights/cache histories. Kernel/cache/weights
// identity must be bound to policy_epoch by the future engine adapter.
struct SampledPathIdentity {
    uint64_t owner = 0, epoch = 0, policy_epoch = 0;
    uint32_t prefix = 0, vocabulary = 0, last_token = 0;
    bool operator==(const SampledPathIdentity&) const = default;
};

struct SampledProposalRow {
    uint32_t position = 0, predecessor = 0, selected = 0;
    SampledSparseLaw actual_q;
};

struct SampledTargetRow {
    uint32_t position = 0;
    std::span<const float> logits;
};

struct SampledTargetTopKRow {
    uint32_t position = 0;
    GpuLogitTopKRow topk{};
};

struct SampledPathCancellation {
    bool (*requested)(void*) noexcept = nullptr;
    void* context = nullptr;
};

enum class SampledPathTerminal : uint32_t { correction, target_bonus, accepted_stop };

struct SampledPathToken {
    uint32_t id = 0;
    float source_logit = 0;
    bool accepted_proposal = false;
};

struct SampledPathDecision {
    explicit SampledPathDecision(const SamplingRng& after_proposals) : random_after(after_proposals) {}
    SampledPathIdentity identity;
    SamplingPolicy target_policy;
    std::array<SampledPathToken, sampled_path_maximum_proposals + 1> tokens{};
    uint32_t count = 0, accepted = 0, active_proposals = 0;
    bool stopped = false, output_budget_exhausted = false;
    SampledPathTerminal terminal = SampledPathTerminal::correction;
    uint64_t proposal_words = 0, acceptance_words = 0, terminal_words = 0;
    SamplingRng random_after;
};

// Pure decision, not model/cache publication. Target rows are p0..pk: p0 is
// retained at the fully consumed base prefix, pi follows i supplied proposals.
// Proposal row i must have been drawn from actual_q conditional on exactly the
// preceding selected proposal path (first predecessor is last_token).
//
// after_proposals already includes ALL proposal draws, including unused tails:
// exactly one word for each non-point-mass q, zero for each effective point mass.
// It is never mutated. The returned copy includes acceptance and terminal
// draws; a caller may publish it only with its corresponding model/logit state.
// Cancellation, malformed input, allocation failure or entropy-draw exhaustion
// returns no decision and advances no caller RNG. No target/serving integration
// or arbitrary-real/other-runtime sampling equivalence is implied.
[[nodiscard]] SampledPathDecision decide_sampled_path(
    SampledPathIdentity proposals_identity, std::span<const SampledProposalRow> proposals,
    SampledPathIdentity target_identity, std::span<const SampledTargetRow> target_rows,
    SamplingPolicy target_policy, uint32_t remaining_outputs, std::span<const uint32_t> stop_ids,
    const SamplingRng& after_proposals, uint64_t proposal_begin_words,
    SampledPathCancellation cancellation = {});

[[nodiscard]] SampledPathDecision decide_sampled_path_topk(
    SampledPathIdentity proposals_identity, std::span<const SampledProposalRow> proposals,
    SampledPathIdentity target_identity, std::span<const SampledTargetTopKRow> target_rows,
    SamplingPolicy target_policy, uint32_t remaining_outputs, std::span<const uint32_t> stop_ids,
    const SamplingRng& after_proposals, uint64_t proposal_begin_words,
    SampledPathCancellation cancellation = {});

} // namespace tokspur
