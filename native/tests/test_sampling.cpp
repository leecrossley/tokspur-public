#include "tokspur/sampling.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <utility>

namespace {
size_t checks = 0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
void near(double actual, double expected, const char* message) {
    require(std::abs(actual - expected) <= 2e-14, message);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid sampling input accepted");
}
void support(const tokspur::SamplingDistribution& actual, const std::vector<uint32_t>& expected) {
    require(std::vector<uint32_t>(actual.support().begin(), actual.support().end()) == expected,
            "sampling support/order differs");
    double sum = 0;
    for (size_t i = 0; i < actual.vocabulary(); ++i) {
        const auto p = actual.probability(uint32_t(i));
        require(std::isfinite(p) && p >= 0 && p <= 1, "invalid dense probability");
        const bool retained = std::find(expected.begin(), expected.end(), uint32_t(i)) != expected.end();
        require(retained ? p > 0 : p == 0, "dense/support disagreement");
        sum += p;
    }
    near(sum, 1, "probabilities not normalized");
}
} // namespace

int main() {
    try {
        using tokspur::sampling_distribution;
        using tokspur::SamplingPolicy;
        const auto inf = std::numeric_limits<double>::infinity();
        const auto nan = std::numeric_limits<double>::quiet_NaN();
        const auto largest = std::numeric_limits<double>::max();
        const auto tiny = std::numeric_limits<double>::denorm_min();
        const std::array<double, 5> ties{3, 3, 2, 1, 3};
        auto greedy = sampling_distribution(ties);
        support(greedy, {0});
        require(greedy.deterministic(), "greedy is not deterministic");
        auto tied = sampling_distribution(ties, {1, 2, 1, 0});
        support(tied, {0, 1});
        near(tied.probability(0), 0.5, "top-k tie mass");
        support(sampling_distribution(ties, {1, 0, 1, 1}), {0, 1, 4});
        support(sampling_distribution(ties, {0, 1, 0.25, 1}), {0});
        const std::array<float, 5> float_ties{3, 3, 2, 1, 3};
        require(sampling_distribution(float_ties, {1, 2, 1, 0}).probabilities()[0] == 0.5,
                "float sampling overload differs");

        // Ratios 8:4:2:1 permit an independently calculable normalization.
        const std::array<double, 4> powers{std::log(8.0), std::log(4.0), std::log(2.0), 0};
        const auto all = sampling_distribution(powers, {1, 0, 1, 0});
        support(all, {0, 1, 2, 3});
        for (size_t i = 0; i < 4; ++i) near(all.probability(uint32_t(i)), double(8U >> i) / 15,
                                           "softmax ratio golden");
        const auto top_two = sampling_distribution(powers, {1, 2, 1, 0});
        support(top_two, {0, 1});
        near(top_two.probability(0), 2.0 / 3, "top-k normalization golden");
        support(sampling_distribution(powers, {1, 99, 1, 0}), {0, 1, 2, 3});
        const auto minimum = sampling_distribution(powers, {1, 0, 1, 0.3});
        support(minimum, {0, 1});
        near(minimum.probability(1), 1.0 / 3, "min-p golden");
        const auto nucleus = sampling_distribution(powers, {1, 0, 0.7, 0});
        support(nucleus, {0, 1});
        near(nucleus.probability(0), 2.0 / 3, "top-p golden");
        // Top-p applies after top-k/min-p: after each first cut, max mass=2/3.
        support(sampling_distribution(powers, {1, 2, 0.6, 0}), {0});
        support(sampling_distribution(powers, {1, 0, 0.6, 0.3}), {0});
        const auto hot = sampling_distribution(powers, {0.5, 0, 1, 0});
        for (size_t i = 0; i < 4; ++i) near(hot.probability(uint32_t(i)), double(64U >> (2 * i)) / 85,
                                           "temperature ratio golden");

        // Exact binary probability boundaries, including both ends of [0,1).
        const std::array<double, 4> equal{0, 0, 0, 0};
        const auto uniform = sampling_distribution(equal, {1, 0, 1, 0});
        support(sampling_distribution(equal, {1, 0, 0.5, 0}), {0, 1});
        support(sampling_distribution(equal, {1, 0, std::nextafter(0.5, 1.0), 0}), {0, 1, 2});
        support(sampling_distribution(equal, {1, 0, tiny, 0}), {0});
        const std::array<double, 2> min_equal{0, -std::log(2.0)};
        support(sampling_distribution(min_equal, {1, 0, 1, 0.5}), {0, 1});
        support(sampling_distribution(min_equal, {1, 0, 1, std::nextafter(0.5, 1.0)}), {0});
        require(tokspur::sample_at(uniform, 0) == 0, "zero draw not first interval");
        require(tokspur::sample_at(uniform, std::nextafter(0.25, 0.0)) == 0, "lower boundary interval");
        require(tokspur::sample_at(uniform, 0.25) == 1, "CDF tie is not right interval");
        require(tokspur::sample_at(uniform, 0.5) == 2, "CDF half boundary");
        require(tokspur::sample_at(uniform, 0.75) == 3, "CDF final boundary");
        require(tokspur::sample_at(uniform, std::nextafter(1.0, 0.0)) == 3, "largest valid draw");
        for (size_t i = 0; i < 4096; ++i)
            require(tokspur::sample_at(uniform, double(i) / 4096) == i / 1024, "exact uniform CDF partition");

        // Huge/tiny finite logits and temperatures must not overflow softmax.
        const std::array<double, 2> opposite{largest, -largest};
        const auto broad = sampling_distribution(opposite, {largest, 0, 1, 0});
        support(broad, {0, 1});
        near(broad.probability(1), std::exp(-2.0) / (1 + std::exp(-2.0)), "cross-zero overflow guard");
        support(sampling_distribution(opposite, {1, 0, 1, 0}), {0});
        support(sampling_distribution(powers, {tiny, 0, 1, 0}), {0});
        const std::array<double, 3> huge_equal{largest, largest, largest};
        const auto huge_uniform = sampling_distribution(huge_equal, {tiny, 0, 1, 0});
        support(huge_uniform, {0, 1, 2});
        near(huge_uniform.probability(0), 1.0 / 3, "huge equal logits");
        const std::array<double, 3> negative{-largest, -largest, -largest};
        near(sampling_distribution(negative, {1, 0, 1, 0}).probability(0), 1.0 / 3, "negative huge offset");
        const std::array<double, 1> singleton{-largest};
        support(sampling_distribution(singleton, {tiny, 0, tiny, 1}), {0});
        const std::array<double, 3> normalized_underflow{0, 0, std::log(tiny)};
        support(sampling_distribution(normalized_underflow, {1, 0, 1, 0}), {0, 1});

        // Randomized independent direct-exp reference, deliberately moderate
        // values so it need not use the production max-shift calculation.
        std::mt19937 fixture(0x73616d70);
        for (size_t trial = 0; trial < 500; ++trial) {
            const size_t count = 1 + fixture() % 128;
            std::vector<double> logits(count), weights(count);
            const double temperature = 0.25 + double(fixture() % 8) / 4;
            const uint32_t k = fixture() % (count + 1);
            const double p = double(1 + fixture() % 16) / 16;
            const double m = double(fixture() % 9) / 8;
            std::vector<uint32_t> ids(count);
            std::iota(ids.begin(), ids.end(), 0);
            for (size_t i = 0; i < count; ++i) {
                logits[i] = double(int(fixture() % 33) - 16) / 8;
                weights[i] = std::exp(logits[i] / temperature);
            }
            std::stable_sort(ids.begin(), ids.end(), [&](auto a, auto b) { return logits[a] > logits[b]; });
            if (k) ids.resize(std::min(size_t(k), ids.size()));
            // Avoid randomized cutpoints exactly on a transcendental equality:
            // precise equal-boundary behaviour is covered by the binary cases.
            std::erase_if(ids, [&](auto id) { return weights[id] / weights[ids[0]] < m - 1e-15; });
            double total = 0;
            for (const auto id : ids) total += weights[id];
            double cumulative = 0;
            size_t prefix = 0;
            do { cumulative += weights[ids[prefix++]]; } while (prefix < ids.size() && cumulative < p * total - 1e-13);
            ids.resize(prefix);
            total = 0;
            for (const auto id : ids) total += weights[id];
            const auto actual = sampling_distribution(logits, {temperature, k, p, m});
            support(actual, ids);
            for (const auto id : ids) near(actual.probability(id), weights[id] / total, "independent filtered reference");
            std::vector<double> shifted = logits;
            for (auto& value : shifted) value += 64;
            const auto offset = sampling_distribution(shifted, {temperature, k, p, m});
            require(actual.support().size() == offset.support().size(), "offset changes support");
            for (size_t i = 0; i < count; ++i) near(actual.probability(uint32_t(i)), offset.probability(uint32_t(i)), "offset changes mass");
        }

        tokspur::SamplingRng random(0x746f6b73707572);
        const auto initial = random;
        for (size_t i = 0; i < 100; ++i) require(tokspur::sample(greedy, random) == 0, "greedy draw changed");
        require(random == initial && random.words_consumed() == 0, "greedy consumed entropy");
        const auto sharp = sampling_distribution(powers, {tiny, 0, 1, 0});
        static_cast<void>(tokspur::sample(sharp, random));
        require(random == initial, "stochastic point mass consumed entropy");
        std::mt19937_64 engine(random.seed());
        for (size_t i = 0; i < 10000; ++i) {
            const double expected = double(engine() >> 11) * 0x1p-53;
            require(random.uniform() == expected, "RNG unit mapping differs from specified bits");
            require(random.words_consumed() == i + 1, "RNG count differs");
        }
        auto checkpoint = random;
        std::array<uint32_t, 1000> sequence{};
        for (auto& id : sequence) id = tokspur::sample(all, random);
        require(random.words_consumed() == checkpoint.words_consumed() + sequence.size(), "one-word sampling count");
        auto replay = checkpoint;
        for (const auto id : sequence) require(tokspur::sample(all, replay) == id, "RNG checkpoint replay changed");
        require(replay == random, "replayed entropy state differs");
        // Reject an unpublished speculative transaction by discarding its copy.
        auto speculative = checkpoint;
        static_cast<void>(tokspur::sample(all, speculative));
        require(checkpoint.words_consumed() + 1 == speculative.words_consumed(), "private entropy transaction count");
        require(checkpoint.words_consumed() == 10000, "private transaction mutated original");

        // Deterministic empirical check complements exact CDF partitions; the
        // finite seeded sequence is not a statistical claim about model output.
        tokspur::SamplingRng empirical(0x637075);
        std::array<size_t, 4> bins{};
        for (size_t i = 0; i < 120000; ++i) ++bins[tokspur::sample(all, empirical)];
        for (size_t i = 0; i < bins.size(); ++i)
            require(std::abs(double(bins[i]) - 120000 * all.probability(uint32_t(i))) < 800,
                    "seeded sampling histogram out of broad bounds");

        for (const auto value : {nan, inf, -inf}) {
            const std::array<double, 2> bad{0, value};
            rejected([&] { static_cast<void>(sampling_distribution(bad)); });
            rejected([&] { static_cast<void>(sampling_distribution(bad, {1, 1, 1, 0})); });
        }
        for (const auto value : {nan, inf, -inf, -1.0})
            rejected([&] { static_cast<void>(sampling_distribution(ties, {value, 0, 1, 0})); });
        for (const auto value : {nan, inf, -inf, -1.0, 0.0, 1.01})
            rejected([&] { static_cast<void>(sampling_distribution(ties, {0, 0, value, 0})); });
        for (const auto value : {nan, inf, -inf, -1.0, 1.01})
            rejected([&] { static_cast<void>(sampling_distribution(ties, {0, 0, 1, value})); });
        rejected([&] { static_cast<void>(sampling_distribution(ties, {0, uint32_t(tokspur::maximum_sampling_vocabulary + 1), 1, 0})); });
        rejected([&] { static_cast<void>(sampling_distribution(std::span<const double>{})); });
        const std::vector<float> oversized(tokspur::maximum_sampling_vocabulary + 1, 0);
        rejected([&] { static_cast<void>(sampling_distribution(oversized)); });
        rejected([&] { static_cast<void>(all.probability(4)); });
        for (const auto value : {nan, inf, -inf, -tiny, -1.0, 1.0, 1.01})
            rejected([&] { static_cast<void>(tokspur::sample_at(all, value)); });
        auto movable = uniform;
        auto moved = std::move(movable);
        require(moved.vocabulary() == 4, "distribution move lost data");
        // The standard permits a moved-from container to be empty or retain
        // data. If empty, it must be safely refused without advancing entropy.
        if (movable.vocabulary() == 0 || movable.support().empty()) {
            const auto before = random;
            rejected([&] { static_cast<void>(tokspur::sample(movable, random)); });
            require(random == before, "refused sample advanced entropy");
        }
        const std::vector<float> maximum_vocab(tokspur::maximum_sampling_vocabulary, 0);
        const auto bounded = sampling_distribution(maximum_vocab, {1, 1, 1, 0});
        require(bounded.vocabulary() == tokspur::maximum_sampling_vocabulary && bounded.deterministic(),
                "maximum vocabulary boundary refused");
        // Full sorting is an independent selection oracle for the small-k
        // heap, including exact ties and both sides of its capacity boundary.
        for (const uint32_t count : {1U, 31U, 67U, 513U, 248320U}) {
            std::vector<float> values(count);
            for (uint32_t i = 0; i < count; ++i)
                values[i] = float(int((uint64_t(i) * 73 + 11) % 17) - 8) / 16;
            const std::vector<double> wide(values.begin(), values.end());
            std::vector<uint32_t> ordered(count);
            std::iota(ordered.begin(), ordered.end(), 0);
            std::sort(ordered.begin(), ordered.end(), [&](uint32_t a, uint32_t b) {
                return values[a] > values[b] || (values[a] == values[b] && a < b);
            });
            const auto float_greedy = sampling_distribution(values);
            require(float_greedy.deterministic() && float_greedy.support().front() == ordered.front(), "float greedy scan differs from full-sort oracle");
            tokspur::SamplingRng greedy_random(44);
            const auto greedy_before = greedy_random;
            require(tokspur::sample(float_greedy, greedy_random) == ordered.front() && greedy_random == greedy_before, "float greedy scan consumed entropy or changed token");
            const auto filtered_float = sampling_distribution(values, {.75, 20, .95, .125});
            const auto filtered_double = sampling_distribution(wide, {.75, 20, .95, .125});
            require(std::ranges::equal(filtered_float.support(), filtered_double.support()), "float filtered support differs from double oracle");
            require(std::ranges::equal(filtered_float.probabilities(), filtered_double.probabilities()), "float filtered masses differ from double oracle");
            for (const uint32_t k : {0U, 1U, 20U, 64U, 65U}) {
                auto expected = ordered;
                if (k && k < count) expected.resize(k);
                const auto small = sampling_distribution(values, {1, k, 1, 0});
                const auto large = sampling_distribution(wide, {1, k, 1, 0});
                require(std::vector<uint32_t>(small.support().begin(), small.support().end()) == expected,
                        "float top-k differs from full-sort oracle");
                require(std::vector<uint32_t>(large.support().begin(), large.support().end()) == expected,
                        "double top-k differs from full-sort oracle");
                require(std::equal(small.probabilities().begin(), small.probabilities().end(),
                                   large.probabilities().begin()), "dtype changes exact binary-input masses");
                tokspur::SamplingRng a(44), b(44);
                for (uint32_t draw = 0; draw < 9; ++draw)
                    require(tokspur::sample(small, a) == tokspur::sample(large, b), "dtype changes sampled token");
                require(a == b, "dtype changes complete entropy state");
            }
        }
        // A request without top_k ranks the whole vocabulary on every row. Float
        // rows rank it by radix and double rows by comparison sort, so exact
        // agreement checks the order, its ties and every mass against the sort.
        std::mt19937 ranked_fixture(0x72616e6b);
        for (const uint32_t count : {65U, 2047U, 2049U, 4099U, 248320U}) {
            for (uint32_t shape = 0; shape < 5; ++shape) {
                std::vector<float> values(count);
                for (uint32_t i = 0; i < count; ++i) {
                    const uint32_t word = ranked_fixture();
                    if (shape == 0) values[i] = std::ldexp(float(int32_t(word >> 8) - (1 << 23)), -19); // dense in [-16, 16)
                    else if (shape == 1) values[i] = (word & 1) ? -0.0f : 0.0f;                        // signed zeros only
                    else if (shape == 2) values[i] = float(int(word % 7) - 3) * 1e30f;                 // both signs, far apart
                    else if (shape == 3) values[i] = -std::ldexp(float(word >> 8) + 1, -120);          // negative, near zero
                    else values[i] = word % 4099 ? std::ldexp(float(int32_t(word >> 8) - (1 << 23)), -21) : 12 + float(word % 5); // peaked head
                }
                const std::vector<double> wide(values.begin(), values.end());
                // A min-p of one keeps exactly the tokens tied with the maximum: the cut's equality boundary.
                std::vector<SamplingPolicy> policies{{1, 0, 1, 0}, {0.7, 0, 0.95, 0}, {1, count - 1, 1, 0}, {1, 0, 1, 1}};
                if (count < 248320U) policies.insert(policies.end(),
                    {{1, 0, 0.9, 0.01}, {0.02, 0, 1, 0}, {1, 65, 1, 0}, {1.3, 1000, 0.5, 0.001}, {1e30, 0, 1, 0}});
                for (const auto& policy : policies) {
                    const auto ranked = sampling_distribution(values, policy);
                    const auto sorted = sampling_distribution(wide, policy);
                    require(std::ranges::equal(ranked.support(), sorted.support()), "radix ranking differs from comparison sort");
                    require(std::ranges::equal(ranked.support_probabilities(), sorted.support_probabilities()),
                            "radix-ranked masses differ from comparison sort");
                    require(std::ranges::equal(ranked.probabilities(), sorted.probabilities()),
                            "radix-ranked dense masses differ from comparison sort");
                    tokspur::SamplingRng a(45), b(45);
                    for (uint32_t draw = 0; draw < 9; ++draw)
                        require(tokspur::sample(ranked, a) == tokspur::sample(sorted, b), "radix ranking changes sampled token");
                    require(a == b, "radix ranking changes entropy state");
                }
            }
        }
        for (const float value : {std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::denorm_min()}) {
            // An invalid entry is refused before a wide ranking, wherever it sits.
            for (const size_t position : {size_t(0), size_t(2048), size_t(4098)}) {
                std::vector<float> bad(4099, -1); bad[position] = value;
                rejected([&] { static_cast<void>(sampling_distribution(bad, {1, 0, 1, 0})); });
                rejected([&] { static_cast<void>(sampling_distribution(bad, {1, 100, 0.9, 0})); });
            }
        }
        for (const float value : {std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity()}) {
            std::vector<float> bad(100, -1); bad.back() = value;
            rejected([&] { static_cast<void>(sampling_distribution(bad, {1, 20, 1, 0})); });
        }
        const float subnormal_f = std::numeric_limits<float>::denorm_min();
        std::vector<float> subnormal_logits(10, 0.0f); subnormal_logits[3] = subnormal_f;
        rejected([&] { static_cast<void>(sampling_distribution(subnormal_logits)); });
        const double subnormal_d = std::numeric_limits<double>::denorm_min();
        std::vector<double> subnormal_d_logits(10, 0.0); subnormal_d_logits[3] = subnormal_d;
        rejected([&] { static_cast<void>(sampling_distribution(subnormal_d_logits)); });

        // Invalid entries must be rejected in the initial selection, vector scan and scalar tail, even when they cannot win.
        for (const float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), subnormal_f, -subnormal_f}) {
            for (const size_t position : {0u, 20u, 64u, 128u, 192u}) {
                std::vector<float> bad(193, -1.0f);
                bad[position] = value;
                rejected([&] { static_cast<void>(sampling_distribution(bad)); });
                rejected([&] { static_cast<void>(sampling_distribution(bad, {1, 20, .95, 0})); });
            }
        }

        const std::array<float, 4> signed_zero_ties{-0.0f, +0.0f, -0.0f, +0.0f};
        const auto sz_dist = sampling_distribution(signed_zero_ties, {1, 4, 1, 0});
        support(sz_dist, {0, 1, 2, 3});
        near(sz_dist.probability(0), 0.25, "signed zero tie probability 0");
        near(sz_dist.probability(1), 0.25, "signed zero tie probability 1");
        near(sz_dist.probability(2), 0.25, "signed zero tie probability 2");
        near(sz_dist.probability(3), 0.25, "signed zero tie probability 3");

        std::cout << "Owned sampling checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Owned sampling check failed: " << error.what() << '\n';
        return 1;
    }
}
