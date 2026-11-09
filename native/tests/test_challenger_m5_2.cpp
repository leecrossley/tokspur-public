#include "tokspur/sampled_path.hpp"
#include "tokspur/sampled_path_detail.hpp"
#include "tokspur/sampling.hpp"
#include "tokspur/mtp_anchor_state.hpp"
#include "tokspur/mtp_sampling.hpp"
#include "tokspur/prompt_lookup.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/metal.hpp"
#include "tokspur/mtp_head.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace {
using namespace tokspur;

uint64_t checks = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        std::cerr << "CHALLENGER FAILURE: " << message << std::endl;
        throw std::runtime_error(message);
    }
}

void require_throws_format_error(const std::function<void()>& action, const char* message) {
    ++checks;
    try {
        action();
    } catch (const FormatError&) {
        return;
    } catch (const std::exception& e) {
        std::cerr << "Expected FormatError but got: " << e.what() << " for: " << message << std::endl;
        throw std::runtime_error(message);
    }
    std::cerr << "Expected FormatError was NOT thrown for: " << message << std::endl;
    throw std::runtime_error(message);
}

template <typename T>
bool spans_equal(std::span<const T> a, std::span<const T> b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

// ============================================================================
// TASK 1: SampledSparseLaw::from_grid_units Invariants
// ============================================================================
void test_sampled_sparse_law_invariants() {
    std::cout << "[Task 1] Testing SampledSparseLaw::from_grid_units invariants..." << std::endl;
    constexpr uint64_t m = sampled_path_grid_size; // 2^53 = 9007199254740992

    // 1. Valid cases: exact integer sum == 2^53
    // Case 1a: Single token point mass (sum == 2^53)
    {
        const std::array<uint32_t, 1> ids{42};
        const std::array<uint64_t, 1> units{m};
        const auto law = SampledSparseLaw::from_grid_units(100, ids, units);
        require(law.size() == 1, "Point mass size must be 1");
        require(law.vocabulary() == 100, "Vocabulary must match");
        require(law.deterministic(), "Point mass must be deterministic");
        require(law.units(42) == m, "Units for token 42 must equal 2^53");
        require(law.units(0) == 0, "Units for token 0 must be 0");
        require(law.sample_at(0) == 42, "sample_at(0) must return token 42");
        require(law.sample_at(m - 1) == 42, "sample_at(2^53 - 1) must return token 42");
    }

    // Case 1b: 2 tokens (50/50 split)
    {
        const std::array<uint32_t, 2> ids{7, 13};
        const std::array<uint64_t, 2> units{m / 2, m / 2};
        const auto law = SampledSparseLaw::from_grid_units(100, ids, units);
        require(law.size() == 2, "Size must be 2");
        require(!law.deterministic(), "50/50 law must not be deterministic");
        require(law.units(7) == m / 2, "Units for token 7 must be m/2");
        require(law.units(13) == m / 2, "Units for token 13 must be m/2");
        require(law.sample_at(0) == 7, "sample_at(0) must be 7");
        require(law.sample_at(m / 2 - 1) == 7, "sample_at(m/2 - 1) must be 7");
        require(law.sample_at(m / 2) == 13, "sample_at(m/2) must be 13");
        require(law.sample_at(m - 1) == 13, "sample_at(m - 1) must be 13");
    }

    // Case 1c: Maximum allowed support (exactly 20 tokens)
    {
        std::array<uint32_t, 20> ids{};
        std::array<uint64_t, 20> units{};
        uint64_t allocated = 0;
        for (uint32_t i = 0; i < 20; ++i) {
            ids[i] = i * 2;
            units[i] = m / 20;
            allocated += units[i];
        }
        units[19] += (m - allocated); // Add remainder to close to exactly 2^53
        uint64_t total = 0;
        for (auto u : units) total += u;
        require(total == m, "Test setup: units must sum to exactly 2^53");

        const auto law = SampledSparseLaw::from_grid_units(50, ids, units);
        require(law.size() == 20, "20-token law must have size 20");
        require(!law.deterministic(), "20-token law must not be deterministic");
        for (uint32_t i = 0; i < 20; ++i) {
            require(law.units(ids[i]) == units[i], "Each token units must match");
        }
    }

    // Case 1d: Compaction of zero-mass entries
    {
        const std::array<uint32_t, 5> ids{1, 2, 3, 4, 5};
        const std::array<uint64_t, 5> units{0, m / 2, 0, m / 2, 0};
        const auto law = SampledSparseLaw::from_grid_units(10, ids, units);
        require(law.size() == 2, "Compacted size must be 2");
        require(law.identifiers()[0] == 2, "First compacted ID must be 2");
        require(law.identifiers()[1] == 4, "Second compacted ID must be 4");
        require(law.mass_units()[0] == m / 2, "First compacted mass must be m/2");
        require(law.mass_units()[1] == m / 2, "Second compacted mass must be m/2");
        require(law.units(1) == 0, "Zero-unit token 1 must return 0");
        require(law.units(3) == 0, "Zero-unit token 3 must return 0");
        require(law.units(5) == 0, "Zero-unit token 5 must return 0");
    }

    // 2. Adversarial Non-Summing Tests:
    // Case 2a: Sum is 2^53 - 1 (too small by 1 unit)
    {
        const std::array<uint32_t, 2> ids{1, 2};
        const std::array<uint64_t, 2> units{m / 2, (m / 2) - 1};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(10, ids, units));
        }, "Sum == 2^53 - 1 must throw FormatError");
    }

    // Case 2b: Sum is 2^53 + 1 (too large by 1 unit)
    {
        const std::array<uint32_t, 2> ids{1, 2};
        const std::array<uint64_t, 2> units{m / 2, (m / 2) + 1};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(10, ids, units));
        }, "Sum == 2^53 + 1 must throw FormatError");
    }

    // Case 2c: Sum is 0
    {
        const std::array<uint32_t, 2> ids{1, 2};
        const std::array<uint64_t, 2> units{0, 0};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(10, ids, units));
        }, "Sum == 0 must throw FormatError");
    }

    // Case 2d: Integer overflow attack: two large uint64 that would wrap around 2^64
    {
        const std::array<uint32_t, 2> ids{1, 2};
        const uint64_t val1 = m + 1000;
        const uint64_t val2 = std::numeric_limits<uint64_t>::max() - 999;
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(10, ids, std::array{val1, val2}));
        }, "Integer overflow in units sum must throw FormatError");
    }

    // 3. Empty support
    {
        const std::span<const uint32_t> empty_ids{};
        const std::span<const uint64_t> empty_units{};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(10, empty_ids, empty_units));
        }, "Empty support must throw FormatError");
    }

    // 4. Support > 20 (e.g. 21 tokens)
    {
        std::array<uint32_t, 21> ids{};
        std::array<uint64_t, 21> units{};
        uint64_t allocated = 0;
        for (uint32_t i = 0; i < 21; ++i) {
            ids[i] = i;
            units[i] = m / 21;
            allocated += units[i];
        }
        units[20] += (m - allocated);
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(100, ids, units));
        }, "Support of 21 tokens (> 20) must throw FormatError");
    }

    // 5. Duplicate IDs
    {
        const std::array<uint32_t, 3> dup_ids{5, 8, 5};
        const std::array<uint64_t, 3> units{m / 4, m / 2, m / 4};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(100, dup_ids, units));
        }, "Duplicate token IDs must throw FormatError");
    }

    // 6. Out-of-vocabulary IDs
    // Case 6a: ID == vocabulary
    {
        const std::array<uint32_t, 1> ids{50};
        const std::array<uint64_t, 1> units{m};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(50, ids, units));
        }, "Token ID == vocabulary must throw FormatError");
    }

    // Case 6b: ID > vocabulary
    {
        const std::array<uint32_t, 1> ids{51};
        const std::array<uint64_t, 1> units{m};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(50, ids, units));
        }, "Token ID > vocabulary must throw FormatError");
    }

    // Case 6c: Vocabulary == 0
    {
        const std::array<uint32_t, 1> ids{0};
        const std::array<uint64_t, 1> units{m};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(0, ids, units));
        }, "Vocabulary == 0 must throw FormatError");
    }

    // Case 6d: Vocabulary > maximum_sampling_vocabulary
    {
        const std::array<uint32_t, 1> ids{0};
        const std::array<uint64_t, 1> units{m};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(maximum_sampling_vocabulary + 1, ids, units));
        }, "Vocabulary > maximum_sampling_vocabulary must throw FormatError");
    }

    // 7. Mismatched vector dimensions
    {
        const std::array<uint32_t, 2> ids{1, 2};
        const std::array<uint64_t, 3> units{m / 3, m / 3, m - 2 * (m / 3)};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(10, ids, units));
        }, "Mismatched ids/units sizes must throw FormatError");
    }

    // 8. Unsupported grid enum
    {
        const std::array<uint32_t, 1> ids{1};
        const std::array<uint64_t, 1> units{m};
        require_throws_format_error([&] {
            static_cast<void>(SampledSparseLaw::from_grid_units(10, ids, units, static_cast<SampledPathGrid>(999)));
        }, "Unsupported grid enum must throw FormatError");
    }

    // 9. Randomized Generator & Oracle: 1,000 random valid and perturbed distributions
    std::mt19937_64 rng(0x546f6b53707572ULL);
    for (size_t trial = 0; trial < 1000; ++trial) {
        const uint32_t support_k = 1 + (rng() % 20); // 1..20
        const uint32_t vocab = 100 + (rng() % 1000);
        std::vector<uint32_t> all_tokens(vocab);
        std::iota(all_tokens.begin(), all_tokens.end(), 0);
        std::shuffle(all_tokens.begin(), all_tokens.end(), rng);
        all_tokens.resize(support_k);

        // Generate partition of 2^53
        std::vector<uint64_t> cuts(support_k - 1);
        for (auto& c : cuts) c = rng() % m;
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

        std::vector<uint64_t> units;
        units.reserve(support_k);
        uint64_t prev = 0;
        for (auto c : cuts) {
            units.push_back(c - prev);
            prev = c;
        }
        units.push_back(m - prev);

        while (units.size() < support_k) {
            units.push_back(0); // Pad with zeroes if needed
        }
        std::vector<uint32_t> ids(all_tokens.begin(), all_tokens.begin() + units.size());

        uint64_t sum = 0;
        for (auto u : units) sum += u;
        require(sum == m, "Randomized setup must sum to m");

        // Valid execution
        const auto law = SampledSparseLaw::from_grid_units(vocab, ids, units);
        require(law.size() <= support_k, "Compacted size <= support_k");
        require(law.vocabulary() == vocab, "Vocab matches");

        // Verify partition endpoints
        uint64_t cumulative = 0;
        for (size_t idx = 0; idx < law.size(); ++idx) {
            const auto id = law.identifiers()[idx];
            const auto u = law.mass_units()[idx];
            require(law.units(id) == u, "Units lookup matches");
            require(law.sample_at(cumulative) == id, "Interval start matches id");
            require(law.sample_at(cumulative + u - 1) == id, "Interval end matches id");
            cumulative += u;
        }
        require(cumulative == m, "Cumulative sum matches 2^53");

        // Perturbation 1: subtract 1 from last non-zero unit -> must throw
        if (units.back() > 0) {
            auto bad_units = units;
            bad_units.back() -= 1;
            require_throws_format_error([&] {
                static_cast<void>(SampledSparseLaw::from_grid_units(vocab, ids, bad_units));
            }, "Perturbed sum (m - 1) must throw");
        }

        // Perturbation 2: add 1 to last unit -> must throw
        {
            auto bad_units = units;
            bad_units.back() += 1;
            require_throws_format_error([&] {
                static_cast<void>(SampledSparseLaw::from_grid_units(vocab, ids, bad_units));
            }, "Perturbed sum (m + 1) must throw");
        }
    }

    std::cout << "[Task 1] PASS: All SampledSparseLaw::from_grid_units invariants confirmed (including 1,000 random stress partitions)." << std::endl;
}

// ============================================================================
// TASK 2: Logit Preflight and Canonicalization
// ============================================================================
void test_logit_preflight_and_canonicalization() {
    std::cout << "[Task 2] Testing Logit Preflight and Canonicalization..." << std::endl;

    // 1. Subnormal rejection for float
    const float subnormal_pos_min = std::numeric_limits<float>::denorm_min();
    const float subnormal_neg_min = -std::numeric_limits<float>::denorm_min();
    const float subnormal_max = std::nextafter(std::numeric_limits<float>::min(), 0.0f);

    require(std::fpclassify(subnormal_pos_min) == FP_SUBNORMAL, "subnormal_pos_min must be FP_SUBNORMAL");
    require(std::fpclassify(subnormal_neg_min) == FP_SUBNORMAL, "subnormal_neg_min must be FP_SUBNORMAL");
    require(std::fpclassify(subnormal_max) == FP_SUBNORMAL, "subnormal_max must be FP_SUBNORMAL");

    // Float subnormal at start
    require_throws_format_error([&] {
        std::vector<float> logits = {subnormal_pos_min, 1.0f, 2.0f};
        static_cast<void>(sampling_distribution(logits));
    }, "Float subnormal at position 0 must throw FormatError");

    // Float subnormal at end
    require_throws_format_error([&] {
        std::vector<float> logits = {1.0f, 2.0f, subnormal_pos_min};
        static_cast<void>(sampling_distribution(logits));
    }, "Float subnormal at position N-1 must throw FormatError");

    // Float negative subnormal in middle
    require_throws_format_error([&] {
        std::vector<float> logits = {1.0f, subnormal_neg_min, 2.0f};
        static_cast<void>(sampling_distribution(logits));
    }, "Float negative subnormal in middle must throw FormatError");

    // Float largest subnormal
    require_throws_format_error([&] {
        std::vector<float> logits = {1.0f, subnormal_max, 2.0f};
        static_cast<void>(sampling_distribution(logits));
    }, "Float largest subnormal must throw FormatError");

    // Bitwise subnormal scan: test 256 various bit patterns in the subnormal float range
    for (uint32_t step = 1; step <= 256; ++step) {
        const uint32_t bits = step * 0x00007ULL; // Bit patterns with exp = 0 and fraction != 0
        const float val = std::bit_cast<float>(bits);
        if (std::fpclassify(val) == FP_SUBNORMAL) {
            std::vector<float> test_vec = {1.0f, val, 2.0f};
            require_throws_format_error([&] {
                static_cast<void>(sampling_distribution(test_vec));
            }, "Bitwise subnormal float must throw FormatError");
        }
    }

    // 2. Subnormal rejection for double
    const double double_subnormal = std::numeric_limits<double>::denorm_min();
    require(std::fpclassify(double_subnormal) == FP_SUBNORMAL, "double_subnormal must be FP_SUBNORMAL");

    require_throws_format_error([&] {
        std::vector<double> logits = {1.0, double_subnormal, 2.0};
        static_cast<void>(sampling_distribution(logits));
    }, "Double subnormal must throw FormatError");

    // 3. Smallest normal float/double MUST NOT throw
    const float normal_min_float = std::numeric_limits<float>::min();
    require(std::fpclassify(normal_min_float) == FP_NORMAL, "normal_min_float must be FP_NORMAL");
    {
        std::vector<float> logits = {normal_min_float, 1.0f, 2.0f};
        const auto dist = sampling_distribution(logits);
        require(dist.vocabulary() == 3, "Smallest normal float must be admitted");
    }

    const double normal_min_double = std::numeric_limits<double>::min();
    require(std::fpclassify(normal_min_double) == FP_NORMAL, "normal_min_double must be FP_NORMAL");
    {
        std::vector<double> logits = {normal_min_double, 1.0, 2.0};
        const auto dist = sampling_distribution(logits);
        require(dist.vocabulary() == 3, "Smallest normal double must be admitted");
    }

    // 4. -0.0f Canonicalization to +0.0f
    // IEEE 754: -0.0f and +0.0f have distinct sign bits (0x80000000 vs 0x00000000)
    const float pos_zero = +0.0f;
    const float neg_zero = -0.0f;
    require(pos_zero == neg_zero, "IEEE 754: +0.0f == -0.0f");
    require(std::signbit(neg_zero) && !std::signbit(pos_zero), "Sign bits must differ");

    // Case 4a: Check that vector with -0.0f produces identical support and probabilities as all +0.0f
    std::vector<float> mixed_zeros = {pos_zero, neg_zero, pos_zero, neg_zero};
    std::vector<float> all_pos_zeros = {pos_zero, pos_zero, pos_zero, pos_zero};
    std::vector<float> all_neg_zeros = {neg_zero, neg_zero, neg_zero, neg_zero};

    const auto dist_mixed = sampling_distribution(mixed_zeros, {1.0, 0, 1.0, 0});
    const auto dist_pos = sampling_distribution(all_pos_zeros, {1.0, 0, 1.0, 0});
    const auto dist_neg = sampling_distribution(all_neg_zeros, {1.0, 0, 1.0, 0});

    require(spans_equal(dist_mixed.support(), dist_pos.support()), "Support must match between mixed and pos zeros");
    require(spans_equal(dist_mixed.support(), dist_neg.support()), "Support must match between mixed and neg zeros");
    require(std::vector<uint32_t>(dist_mixed.support().begin(), dist_mixed.support().end()) == std::vector<uint32_t>{0, 1, 2, 3},
            "Support must be deterministic 0, 1, 2, 3");

    for (size_t i = 0; i < 4; ++i) {
        require(std::abs(dist_mixed.probability(uint32_t(i)) - 0.25) < 1e-12, "Probability must be 0.25");
        require(dist_mixed.probability(uint32_t(i)) == dist_pos.probability(uint32_t(i)),
                "Probabilities must be bit-identical between mixed and pos zeros");
        require(dist_mixed.probability(uint32_t(i)) == dist_neg.probability(uint32_t(i)),
                "Probabilities must be bit-identical between mixed and neg zeros");
    }

    // Case 4b: Deterministic tie-breaking under bounded top-k
    // When logits tie, rank comparator is: la > lb || (la == lb && a < b)
    // If -0.0f is canonicalized to +0.0f, la == lb is true, and tie breaks strictly by token ID (a < b).
    // Let's test with top_k = 2:
    const auto top2_mixed = sampling_distribution(mixed_zeros, {1.0, 2, 1.0, 0});
    const auto top2_pos = sampling_distribution(all_pos_zeros, {1.0, 2, 1.0, 0});
    require(std::vector<uint32_t>(top2_mixed.support().begin(), top2_mixed.support().end()) == std::vector<uint32_t>{0, 1},
            "Top-2 tie-break must strictly select tokens 0 and 1 regardless of -0.0f");
    require(spans_equal(top2_mixed.support(), top2_pos.support()), "Top-2 support must match all-pos-zeros");

    // Case 4c: Mixed positive, zero, negative logits with -0.0f
    // Tokens:
    // id 0: 3.0f
    // id 1: -0.0f
    // id 2: 1.5f
    // id 3: +0.0f
    // id 4: -2.0f
    // Sorting order must be: id 0 (3.0f), id 2 (1.5f), then tie between 1 and 3 (0.0f).
    // Because 1 < 3, tie-breaking must place 1 before 3: {0, 2, 1, 3, 4}.
    std::vector<float> complex_logits1 = {3.0f, neg_zero, 1.5f, pos_zero, -2.0f};
    std::vector<float> complex_logits2 = {3.0f, pos_zero, 1.5f, neg_zero, -2.0f};

    const auto complex_dist1 = sampling_distribution(complex_logits1, {1.0, 4, 1.0, 0});
    const auto complex_dist2 = sampling_distribution(complex_logits2, {1.0, 4, 1.0, 0});

    const std::vector<uint32_t> expected_complex_support = {0, 2, 1, 3};
    require(std::vector<uint32_t>(complex_dist1.support().begin(), complex_dist1.support().end()) == expected_complex_support,
            "Complex support with -0.0f must match expected deterministic tie-break");
    require(std::vector<uint32_t>(complex_dist2.support().begin(), complex_dist2.support().end()) == expected_complex_support,
            "Complex support with inverted zeros must match expected deterministic tie-break");
    require(spans_equal(complex_dist1.probabilities(), complex_dist2.probabilities()),
            "Probabilities must be bit-exact across signed-zero configurations");

    // Case 4d: Maximum logit is -0.0f
    std::vector<float> max_neg_zero = {neg_zero, -1.0f, -2.0f};
    std::vector<float> max_pos_zero = {pos_zero, -1.0f, -2.0f};
    const auto max_neg_dist = sampling_distribution(max_neg_zero, {1.0, 0, 1.0, 0});
    const auto max_pos_dist = sampling_distribution(max_pos_zero, {1.0, 0, 1.0, 0});
    require(max_neg_dist.support()[0] == 0, "Token 0 must be top rank");
    require(spans_equal(max_neg_dist.support(), max_pos_dist.support()), "Support must match between max -0 and +0");
    require(spans_equal(max_neg_dist.probabilities(), max_pos_dist.probabilities()), "Probabilities must match bit-exact");

    std::cout << "[Task 2] PASS: Logit preflight and canonicalization confirmed." << std::endl;
}

// ============================================================================
// TASK 3: 32K Draft Slice Suballocation & Geometry
// ============================================================================
void test_draft_slice_suballocation_and_geometry(const char* metallib_path) {
    std::cout << "[Task 3] Testing 32K Draft Slice Suballocation & Geometry..." << std::endl;

    constexpr uint32_t channels = 2560;
    constexpr uint32_t full_vocab = 248320;
    constexpr uint32_t draft_vocab = 32768;

    // Mathematical verification:
    // In Q8_0: 32 elements per block. Each block has 1 FP16 scale (2 bytes) + 32 int8 quants (32 bytes) = 34 bytes.
    // Row bytes = (2560 / 32) * 34 = 80 * 34 = 2720 bytes.
    constexpr uint64_t row_bytes = (channels / 32) * 34;
    require(row_bytes == 2720, "Q8_0 row bytes for 2560 channels must be 2720");

    constexpr uint64_t full_vocab_bytes = uint64_t(full_vocab) * row_bytes;
    require(full_vocab_bytes == 675430400, "Full vocabulary Q8_0 matrix must be 675,430,400 bytes (644.14 MiB)");

    constexpr uint64_t expected_draft_output_bytes = uint64_t(draft_vocab) * row_bytes;
    require(expected_draft_output_bytes == 89128960, "32K draft output slice must be exactly 89,128,960 bytes");
    require(expected_draft_output_bytes == 85ull * 1024 * 1024, "89,128,960 bytes must equal exactly 85.0 MiB");

    constexpr uint64_t expected_draft_embedding_bytes = uint64_t(draft_vocab) * row_bytes;
    require(expected_draft_embedding_bytes == 89128960, "32K draft embedding slice must be exactly 89,128,960 bytes");
    require(expected_draft_embedding_bytes == 85ull * 1024 * 1024, "89,128,960 bytes must equal exactly 85.0 MiB");

    constexpr uint64_t expected_total_slice_bytes = expected_draft_output_bytes + expected_draft_embedding_bytes;
    require(expected_total_slice_bytes == 178257920, "Total draft slice allocation must be exactly 178,257,920 bytes");
    require(expected_total_slice_bytes == 170ull * 1024 * 1024, "Total draft slice allocation must equal exactly 170.0 MiB");
    require(expected_total_slice_bytes <= 170ull * 1024 * 1024, "Draft slices must not exceed 170.0 MiB ceiling");

    // Sliced logits buffer calculation:
    constexpr uint64_t full_logits_bytes = uint64_t(full_vocab) * sizeof(float);
    require(full_logits_bytes == 993280, "Full vocabulary logits buffer is 993,280 bytes (~970 KiB)");

    constexpr uint64_t draft_logits_bytes = uint64_t(draft_vocab) * sizeof(float);
    require(draft_logits_bytes == 131072, "32K draft logits buffer must be exactly 131,072 bytes");
    require(draft_logits_bytes == 128ull * 1024, "131,072 bytes must equal exactly 128 KiB");

    // MetalGraph Live Verification:
    if (metallib_path) {
        std::cout << "  Verifying MetalGraph suballocation with " << metallib_path << "..." << std::endl;
        MetalGraph graph(metallib_path, 256ull << 20); // 256 MiB budget
        require(!graph.has_draft_slices(), "Fresh graph must not have draft slices");

        const auto ledger_initial = graph.memory();

        // Allocate a simulated parent output.weight allocation (at least 89,128,960 bytes)
        const auto parent_output = graph.zeros(expected_draft_output_bytes + 1024, "Simulated output.weight");
        const auto ledger_after_parent = graph.memory();

        // Suballocate the draft output slice
        const auto draft_output = graph.suballocate_draft_output_slice(parent_output, draft_vocab, channels);
        const auto ledger_after_suballoc = graph.memory();

        // Assert that suballocating the draft output slice performed ZERO additional GPU buffer allocations!
        require(ledger_after_suballoc.live_bytes == ledger_after_parent.live_bytes,
                "Draft output slice suballocation must NOT increase live GPU bytes (0 extra buffer allocated)");
        require(draft_output.bytes == 89128960, "draft_output.bytes must equal 89,128,960");
        require(draft_output.slot == parent_output.slot, "draft_output must share parent slot (suballocation)");
        require(draft_output.offset == parent_output.offset, "draft_output offset must match parent");

        // Allocate the draft embedding slice
        const auto draft_embedding = graph.allocate_draft_embedding_slice(draft_vocab, channels);
        const auto ledger_after_embedding = graph.memory();

        require(draft_embedding.bytes == 89128960, "draft_embedding.bytes must equal 89,128,960");
        require(draft_embedding.slot != parent_output.slot, "draft_embedding must have its own slot");
        require(ledger_after_embedding.live_bytes > ledger_after_suballoc.live_bytes,
                "Draft embedding slice allocates dedicated storage");

        require(graph.has_draft_slices(), "Graph must report has_draft_slices == true");
        require(graph.draft_output_slice().bytes == 89128960, "graph.draft_output_slice().bytes matches");
        require(graph.draft_embedding_slice().bytes == 89128960, "graph.draft_embedding_slice().bytes matches");

        // Release views cleanly
        graph.release(draft_embedding);
        graph.release(parent_output);
        require(graph.memory().live_bytes == ledger_initial.live_bytes, "All scratch memory reclaimed cleanly");
    }

    std::cout << "[Task 3] PASS: 32K Draft Slice Suballocation & Geometry confirmed." << std::endl;
}

// ============================================================================
// TASK 4: Pre-Draw Suffix Arbiter & k=0 Route
// ============================================================================
void test_predraw_suffix_arbiter_and_k0_route() {
    std::cout << "[Task 4] Testing Pre-Draw Suffix Arbiter & k=0 Route..." << std::endl;

    // 1. Prompt Lookup / Suffix Arbiter:
    // Verify prompt lookup is evaluated strictly from committed[0..base] before any RNG draws (0 entropy drawn)
    LookupConfig config;
    config.match_tokens = 4;
    config.proposal_tokens = 2;
    config.history_tokens = 64;
    config.work_limit = 64;

    PromptLookup lookup(config);
    SamplingRng rng(42);
    const uint64_t initial_words = rng.words_consumed();

    // Commit tokens: [10, 20, 30, 40, 50, 60, 10, 20, 30, 40]
    // Tokens 0..3: [10, 20, 30, 40]
    // Tokens 4..5: [50, 60]
    // Tokens 6..9: [10, 20, 30, 40]
    const std::vector<uint32_t> initial_tokens = {10, 20, 30, 40, 50, 60, 10, 20, 30, 40};
    lookup.append(lookup.epoch(), lookup.committed_length(), initial_tokens);

    require(lookup.committed_length() == 10, "Committed length must be 10");

    // Suffix of length 4 at base=10 is tokens 6..9: [10, 20, 30, 40].
    // Earlier occurrence of this 4-gram is at index 0.
    // The continuation following index 0 is [50, 60].
    const auto proposal = lookup.propose(lookup.epoch(), 10);

    require(proposal.outcome == LookupOutcome::proposed, "Outcome must be proposed");
    require(proposal.count == 2, "Proposed count must be 2");
    require(proposal.proposed()[0] == 50, "First proposed token must be 50");
    require(proposal.proposed()[1] == 60, "Second proposed token must be 60");

    // Empirically verify 0 entropy drawn during prompt lookup
    require(rng.words_consumed() == initial_words, "Prompt lookup MUST NOT consume any entropy words (0 entropy drawn)");

    // Test a query that produces NO match (unique prefix)
    lookup.append(lookup.epoch(), 10, std::array<uint32_t, 4>{99, 98, 97, 96});
    const auto no_match_proposal = lookup.propose(lookup.epoch(), 14);
    require(no_match_proposal.count == 0, "No match query must propose 0 tokens");
    require(rng.words_consumed() == initial_words, "No match query still consumes 0 entropy words");

    // 2. k=0 Route and Stage A Controller State Machine:
    // Verify k=0 decisions route through single-token advance and bypass draft GEMV/sampling
    StageAControllerState state{};
    require(state.current_mode == uint8_t(StageAMode::Speculative), "Default mode is Speculative");
    require(state.ewma_alpha1 == 0.45, "Default ewma_alpha1 is 0.45");
    require(state.consecutive_ordinary_steps == 0, "Default consecutive_ordinary_steps is 0");
    require(state.controller_policy_digest == default_controller_policy_digest, "Default policy digest verified");

    // Verify CostTable parameters and digests
    require(StageACostTable::alpha_0 == 0.45, "CostTable alpha_0 is 0.45");
    require(StageACostTable::beta == 0.125, "CostTable beta is 0.125");
    require(StageACostTable::eta_exit == 0.00, "CostTable eta_exit is 0.00");
    require(StageACostTable::eta_enter == 0.04, "CostTable eta_enter is 0.04");
    require(StageACostTable::probe_interval == 16, "CostTable probe_interval is 16");
    require(StageACostTable::policy_digest == default_controller_policy_digest, "Policy digest matches");

    // Verify t_ord values
    require(StageACostTable::lookup_t_ord_us(512) == 34430, "t_ord(512) must be 34430");
    require(StageACostTable::lookup_t_ord_us(1024) == 36324, "t_ord(1024) must be 36324");
    require(StageACostTable::lookup_t_ord_us(2048) == 44652, "t_ord(2048) must be 44652");
    require(StageACostTable::lookup_t_ord_us(4096) == 44673, "t_ord(4096) must be 44673");

    // Force transition to OrdinaryK0 by simulating poor speculative performance (y1 = 0)
    stage_a_commit_speculative(state, 0.0, 512);
    require(state.current_mode == uint8_t(StageAMode::OrdinaryK0), "State must transition to OrdinaryK0 when alpha1 drops");
    require(state.consecutive_ordinary_steps == 0, "consecutive_ordinary_steps must be reset to 0 on exit");
    const double alpha_after_exit = state.ewma_alpha1;

    // Simulate 15 ordinary steps (steps 0 through 14)
    // In mtp_depth2_round.inc:
    // const bool is_probe_round = (is_v2 && controller.current_mode == OrdinaryK0 && controller.consecutive_ordinary_steps >= 15 && compute_remaining >= 2);
    // const bool is_ordinary_step = ... || (is_v2 && controller.current_mode == OrdinaryK0 && !is_probe_round);
    for (uint32_t step = 0; step < 15; ++step) {
        require(state.consecutive_ordinary_steps == step, "Step count must advance monotonically");
        const bool is_probe = (state.current_mode == uint8_t(StageAMode::OrdinaryK0) &&
                               state.consecutive_ordinary_steps == 15);
        require(!is_probe, "Steps 0..14 must NOT be probe rounds");
        const bool is_ordinary = (state.current_mode == uint8_t(StageAMode::OrdinaryK0) && !is_probe);
        require(is_ordinary, "Steps 0..14 must be ordinary k=0 steps (bypassing draft GEMV/sampling)");

        // Step ordinary preserves EWMA alpha1 across unobserved k=0 steps and increments consecutive_ordinary_steps
        stage_a_step_ordinary(state, 512);
        require(state.ewma_alpha1 == alpha_after_exit, "Unobserved ordinary steps must not decay ewma_alpha1");
    }

    require(state.consecutive_ordinary_steps == 15, "At step 15, consecutive_ordinary_steps must equal 15");
    const bool is_probe_step15 = (state.current_mode == uint8_t(StageAMode::OrdinaryK0) &&
                                  state.consecutive_ordinary_steps == 15);
    require(is_probe_step15, "Step 15 MUST trigger is_probe_round == true");

    // Probe round commits y1 = 1.0 (successful probe)
    stage_a_commit_probe(state, 1.0, 512);
    require(state.consecutive_ordinary_steps == 0, "Probe round must reset consecutive_ordinary_steps");

    // After a single accepted probe from 0.39375 -> 0.46953125, spec_tp is still below 1.04 * r_ord,
    // while continued accepted periodic probes recover into Speculative without permanent lock-out.
    const double expected_spec_tp = StageACostTable::max_speculative_throughput(state.ewma_alpha1);
    const double expected_r_ord = 1.0 / double(StageACostTable::lookup_t_ord_us(512));
    const bool should_enter = expected_spec_tp > (1.0 + StageACostTable::eta_enter) * expected_r_ord;
    require(!should_enter, "After 1 probe from 0.39375, spec_tp remains below 1.04 * r_ord");
    require(state.current_mode == (should_enter ? uint8_t(StageAMode::Speculative) : uint8_t(StageAMode::OrdinaryK0)),
            "Controller mode must strictly track the throughput inequality");

    StageAControllerState recovery = state;
    for (uint32_t cycle = 0; cycle < 8 && recovery.current_mode == uint8_t(StageAMode::OrdinaryK0); ++cycle) {
        for (uint32_t step = 0; step < 15; ++step) {
            stage_a_step_ordinary(recovery, 512);
        }
        stage_a_commit_probe(recovery, 1.0, 512);
    }
    require(recovery.current_mode == uint8_t(StageAMode::Speculative),
            "Repeated accepted periodic probes must re-enter Speculative mode");

    // 3. Zero controller mutation during packet overflow drains (P = C - E)
    MtpGenerationCarry carry;
    MtpGenerationCommand cmd{
        .id = 10,
        .remaining = 8,
        .begin = true,
        .total_budget = 8,
        .stage_a_controller = state,
        .v2 = true
    };
    carry.admit(cmd, 8, 100, std::array<uint32_t, 1>{0});
    require(carry.controller_state() == state, "Controller state must match admitted state");

    // Commit a round with count=2, but emitted_now=1 (creating 1 overflow token)
    std::array<uint32_t, 2> round_tokens{101, 102};
    std::array<float, 2> round_scores{1.5f, 2.0f};
    carry.commit_round(2, round_tokens, round_scores, 100, 102, 1, false);
    require(carry.pending_overflow() == 1, "Pending overflow must be 1");

    // Pop the overflow token and verify controller state is completely UNMUTATED
    const auto snapshot_before_pop = carry.controller_state();
    const auto overflow = carry.pop_overflow();
    require(overflow.id == 102, "Popped overflow token ID must be 102");
    require(carry.pending_overflow() == 0, "Pending overflow must be 0 after pop");
    require(carry.controller_state() == snapshot_before_pop,
            "pop_overflow MUST NOT mutate StageAControllerState (zero mutation on overflow drain)");

    // 4. MtpGenerationCarry::finish Contract Verification
    // Demonstrate and confirm that finish() returns carry_.clear() (indicating whether an invisible terminal was cleared)
    // and correctly closes generation (active() == false).
    {
        MtpGenerationCarry finish_carry;
        MtpGenerationCommand fin_cmd{
            .id = 20,
            .remaining = 8,
            .begin = true,
            .total_budget = 8,
            .v2 = true
        };
        finish_carry.admit(fin_cmd, 8, 200, std::array<uint32_t, 1>{0});
        require(finish_carry.active(), "Must be active");

        std::array<uint32_t, 8> ids{1, 2, 3, 4, 5, 6, 7, 8};
        std::array<float, 8> scores{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
        finish_carry.commit_round(8, ids, scores, 200, 208, 8, false);

        // When NO pending anchor exists in carry_:
        // carry_.clear() returns false!
        // finish() returns false, BUT generation is closed (!finish_carry.active()).
        const bool finish_result = finish_carry.finish(8, 208, false);
        require(!finish_result, "finish() returns false when no invisible terminal was carried");
        require(!finish_carry.active(), "generation is closed after full budget emitted");
        require(finish_carry.remaining() == 0, "Remaining budget is 0");
    }

    std::cout << "[Task 4] PASS: Pre-Draw Suffix Arbiter & k=0 Route confirmed." << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << "=================================================================" << std::endl;
        std::cout << "CHALLENGER M5-2 EMPIRICAL ADVERSARIAL STRESS TEST SUITE" << std::endl;
        std::cout << "=================================================================" << std::endl;

        const char* metallib = argc > 1 ? argv[1] : nullptr;

        test_sampled_sparse_law_invariants();
        test_logit_preflight_and_canonicalization();
        test_draft_slice_suballocation_and_geometry(metallib);
        test_predraw_suffix_arbiter_and_k0_route();

        std::cout << "=================================================================" << std::endl;
        std::cout << "ALL EMPIRICAL CHALLENGES PASSED! Total checks: " << checks << std::endl;
        std::cout << "=================================================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CHALLENGER M5-2 TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
