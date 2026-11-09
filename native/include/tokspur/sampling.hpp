#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <utility>
#include <vector>
#ifdef TOKSPUR_PRIVATE_SAMPLED_PATH_TRACE
#include <ostream>
#endif

namespace tokspur {

inline constexpr size_t maximum_sampling_vocabulary = 1U << 20;
inline constexpr uint32_t gpu_logit_topk_capacity = 64;
inline constexpr uint32_t gpu_logit_topk_tile_span = 4096;
inline constexpr uint32_t gpu_logit_topk_max_tiles = 256;

struct GpuLogitTopKRow {
    uint32_t vocabulary = 0;
    uint32_t count = 0;
    uint32_t first_nonfinite_col = 0xffffffffu;
    uint32_t first_subnormal_col = 0xffffffffu;
    std::array<uint32_t, gpu_logit_topk_capacity> ids{};
    std::array<float, gpu_logit_topk_capacity> logits{};
};
static_assert(sizeof(GpuLogitTopKRow) == 528, "GpuLogitTopKRow ABI");

struct SamplingPolicy {
    // Zero is greedy, with the lowest token ID winning an exact logit tie.
    // Positive finite values divide the logits before softmax.
    double temperature = 0;
    // Zero means no top-k restriction. Values larger than the vocabulary are
    // clamped to its size; values beyond the absolute vocabulary bound fail.
    uint32_t top_k = 0;
    double top_p = 1; // (0, 1], smallest inclusive probability-ranked prefix.
    double min_p = 0; // [0, 1], retained weight / maximum retained weight.
};

// A bounded, immutable distribution. The builder validates every policy field
// and every logit (including otherwise discarded entries). NaN and either
// infinity are refused, rather than being treated as an implicit masking API.
//
// Filter order is temperature -> top-k -> min-p -> top-p -> normalization.
// Top-p's denominator is the mass remaining AFTER top-k and min-p. Both cuts
// retain equality at their boundary; equal logits are ordered by token ID.
// Softmax weights or normalized masses that underflow to zero are excluded
// from support. Positive temperature and a sufficiently sharp distribution may be a point
// mass. At zero temperature, all filters are ignored AFTER policy validation.
class SamplingRowScan;
class SamplingDistribution {
public:
    SamplingDistribution(const SamplingDistribution&) = default;
    SamplingDistribution& operator=(const SamplingDistribution&) = default;
    SamplingDistribution(SamplingDistribution&& other) noexcept
        : vocabulary_(std::exchange(other.vocabulary_, 0)),
          probabilities_(std::move(other.probabilities_)),
          support_(std::move(other.support_)),
          support_probabilities_(std::move(other.support_probabilities_)) {}
    SamplingDistribution& operator=(SamplingDistribution&& other) noexcept {
        if (this != &other) {
            vocabulary_ = std::exchange(other.vocabulary_, 0);
            probabilities_ = std::move(other.probabilities_);
            support_ = std::move(other.support_);
            support_probabilities_ = std::move(other.support_probabilities_);
        }
        return *this;
    }

    [[nodiscard]] size_t vocabulary() const noexcept { return vocabulary_; }
    [[nodiscard]] std::span<const double> probabilities() const;
    [[nodiscard]] std::span<const uint32_t> support() const noexcept { return support_; }
    [[nodiscard]] std::span<const double> support_probabilities() const noexcept { return support_probabilities_; }
    [[nodiscard]] double probability(uint32_t token) const;
    [[nodiscard]] bool deterministic() const noexcept { return support_.size() == 1; }

private:
    SamplingDistribution(size_t vocabulary, std::vector<double> probabilities,
                         std::vector<uint32_t> support, std::vector<double> support_probabilities)
        : vocabulary_(vocabulary),
          probabilities_(std::move(probabilities)),
          support_(std::move(support)),
          support_probabilities_(std::move(support_probabilities)) {}
    size_t vocabulary_ = 0;
    mutable std::vector<double> probabilities_; // Dense target mass, materialised lazily when built from GPU top-k or fast top-k.
    std::vector<uint32_t> support_; // Descending logits, lowest-ID tie break.
    std::vector<double> support_probabilities_; // Masses aligned 1:1 with support_.
    friend SamplingDistribution sampling_distribution(std::span<const float>, SamplingPolicy);
    friend SamplingDistribution sampling_distribution_after_scan(std::span<const float>, SamplingPolicy, SamplingRowScan&);
    friend SamplingDistribution sampling_distribution(std::span<const double>, SamplingPolicy);
    friend SamplingDistribution sampling_distribution_from_topk(const GpuLogitTopKRow&, SamplingPolicy);
};

[[nodiscard]] SamplingDistribution sampling_distribution(std::span<const float> logits,
                                                        SamplingPolicy policy = {});
[[nodiscard]] SamplingDistribution sampling_distribution(std::span<const double> logits,
                                                        SamplingPolicy policy = {});
[[nodiscard]] bool sampling_topk_eligible(SamplingPolicy policy,
                                          size_t vocabulary = maximum_sampling_vocabulary) noexcept;
[[nodiscard]] GpuLogitTopKRow cpu_logit_topk_row(std::span<const float> logits);
[[nodiscard]] SamplingDistribution sampling_distribution_from_topk(const GpuLogitTopKRow& row,
                                                                   SamplingPolicy policy = {});

// Caller-owned entropy, deliberately separate from reusable model/cache state.
// This uses the C++ standard's specified mt19937_64 engine, seeded with exactly
// one uint64_t. One uniform consumes one engine word; (word >> 11) * 2^-53 maps
// it to the 2^53-point grid in [0,1), without uniform_real_distribution.
// This is reproducible but NOT a cryptographic generator or a parallel stream
// allocation scheme. Same seed implies same stream, not independent requests.
//
// Copy to checkpoint a speculative transaction, and assign the accepted copy
// only when its corresponding model state is committed. Rejected proposals do
// not by themselves justify restoring RNG: the future verification algorithm
// must specify which draws remain consumed. No method is thread safe; the
// caller grants exclusive access and owns any cancellation/publication epoch.
class SamplingRng {
public:
    SamplingRng() : SamplingRng(0) {}
    explicit SamplingRng(uint64_t seed) : engine_(seed), seed_(seed) {}
    SamplingRng(uint64_t seed, uint64_t words, const std::mt19937_64& engine)
        : engine_(engine), seed_(seed), words_(words) {}
    [[nodiscard]] uint64_t seed() const noexcept { return seed_; }
    [[nodiscard]] uint64_t words_consumed() const noexcept { return words_; }
    [[nodiscard]] const std::mt19937_64& engine() const noexcept { return engine_; }
    [[nodiscard]] double uniform();
    bool operator==(const SamplingRng&) const = default;
#ifdef TOKSPUR_PRIVATE_SAMPLED_PATH_TRACE
    // Component-test trace only, including the complete standard engine state.
    friend std::ostream& operator<<(std::ostream& out, const SamplingRng& value) {
        return out << value.seed_ << ' ' << value.words_ << ' ' << value.engine_;
    }
#endif

private:
    std::mt19937_64 engine_;
    uint64_t seed_;
    uint64_t words_ = 0;
};

// Pure inverse-CDF operation: unit must be finite and in [0,1). The intervals
// follow support order and are left-closed/right-open. Reported probabilities
// and cumulative sums use double precision; only a final roundoff gap can fall
// back to the last positive token. A moved-from distribution is refused. This
// is finite-precision sampling, not an
// exact real-number or arbitrary-precision probability implementation.
[[nodiscard]] uint32_t sample_at(const SamplingDistribution& distribution, double unit);

// Point masses, including greedy, consume NO entropy. Otherwise exactly one
// word is consumed. State advances transactionally only after a successful draw.
[[nodiscard]] uint32_t sample(const SamplingDistribution& distribution, SamplingRng& random);

} // namespace tokspur
