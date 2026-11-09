#include "tokspur/prepared_sampling.hpp"
#include "tokspur/mtp_sampling.hpp"
#include <array>
#include <bit>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace tokspur;
void check(bool b) { if (!b) throw std::runtime_error("prepared sampling parity failed"); }
std::string error(const std::function<void()>& fn) {
    try { fn(); } catch (const FormatError& e) { return e.what(); }
    throw std::runtime_error("invalid sampling row accepted");
}
int main() {
    try {
        const std::array<float, 3> a{0.2f, .5f, -.1f}, b{-.4f, .1f, .6f};
        for (const auto policy : {SamplingPolicy{}, SamplingPolicy{1, 0, 1, 0}, SamplingPolicy{1, 20, 1, 0}, SamplingPolicy{1, 20, .95, 0}}) {
            for (uint64_t seed = 0; seed < 256; ++seed) {
                const SamplingRng random(seed);
                const auto first = draw_mtp_depth1(a, policy, random), second = draw_mtp_depth1(b, policy, first.random_after);
                configure_prepared_sampling(false);
                const auto old = decide_mtp_depth2(20, 0, first, second, b, a, b, policy, {});
                configure_prepared_sampling(true);
                const auto now = decide_mtp_depth2(20, 0, first, second, b, a, b, policy, {});
                check(old.random_after == now.random_after && old.accepted == now.accepted && old.count == now.count && old.terminal == now.terminal);
                check(old.acceptance_words == now.acceptance_words && old.terminal_words == now.terminal_words);
                for (uint32_t i = 0; i < old.count; ++i) check(old.tokens[i].id == now.tokens[i].id &&
                    std::bit_cast<uint32_t>(old.tokens[i].source_logit) == std::bit_cast<uint32_t>(now.tokens[i].source_logit));
            }
        }
        // Large complete supports exercise both exact ordering optimisations,
        // including dense ties, signed zero, filtered supports and residual draws.
        constexpr uint32_t vocabulary = 65536;
        std::vector<float> wide(vocabulary), other(vocabulary);
        for (uint32_t i = 0; i < vocabulary; ++i) {
            wide[i] = i % 64 ? float(int((i * 17) % 97) - 48) / 8 : -0.0f;
            other[i] = float(int((i * 31) % 71) - 35) / 8;
        }
        for (const auto policy : {SamplingPolicy{1, 0, 1, 0}, SamplingPolicy{1, vocabulary, 1, 0}, SamplingPolicy{1, 0, .9, .01}, SamplingPolicy{1, 20, .95, 0}}) {
            configure_prepared_sampling(false); const auto reference = sampling_distribution(wide, policy);
            configure_prepared_sampling(true); const auto candidate = sampling_distribution(wide, policy);
            check(reference.support().size() == candidate.support().size() &&
                std::equal(reference.support().begin(), reference.support().end(), candidate.support().begin()) &&
                std::equal(reference.support_probabilities().begin(), reference.support_probabilities().end(), candidate.support_probabilities().begin()) &&
                std::equal(reference.probabilities().begin(), reference.probabilities().end(), candidate.probabilities().begin()));
            for (uint64_t seed = 44; seed < 52; ++seed) {
                configure_prepared_sampling(false);
                const auto proposal = draw_mtp_depth1(other, policy, SamplingRng(seed));
                const auto old = decide_mtp_depth1(17, 2, proposal, wide, other, policy, {});
                configure_prepared_sampling(true);
                const auto now = decide_mtp_depth1(17, 2, proposal, wide, other, policy, {});
                check(old.random_after == now.random_after && old.accepted == now.accepted && old.count == now.count && old.terminal == now.terminal);
                check(old.acceptance_words == now.acceptance_words && old.terminal_words == now.terminal_words);
                for (uint32_t i = 0; i < old.count; ++i) check(old.tokens[i].id == now.tokens[i].id &&
                    std::bit_cast<uint32_t>(old.tokens[i].source_logit) == std::bit_cast<uint32_t>(now.tokens[i].source_logit));
            }
        }
        std::array<float, 3> invalid{std::numeric_limits<float>::denorm_min(), 1, std::numeric_limits<float>::quiet_NaN()};
        check(error([&] { static_cast<void>(sampling_distribution(invalid)); }) == "subnormal sampling logit");
        const SamplingRng random(44); const auto proposal = draw_mtp_depth1(a, {}, random);
        for (bool enabled : {false, true}) {
            configure_prepared_sampling(enabled);
            // The invalid bonus row is still checked when the first proposal is rejected.
            check(error([&] { static_cast<void>(decide_mtp_depth1(17, 2, proposal, b, invalid, {}, {})); }) == "sampled path nonfinite target logit");
            const auto summary = cpu_logit_topk_row(invalid);
            check(error([&] { static_cast<void>(decide_mtp_depth1_topk(17, 2, proposal, cpu_logit_topk_row(b), summary, {}, {})); }) == "sampled path nonfinite target logit");
            check(error([&] { static_cast<void>(sampling_distribution_from_topk(summary)); }) == "subnormal sampling logit");
        }
        auto certificate = SamplingRowScan::inspect(std::span<const float>(a), [] {});
        const auto distribution = sampling_distribution_after_scan(a, {1, 0, 1, 0}, certificate);
        const auto reference = sampling_distribution(a, {1, 0, 1, 0});
        check(std::equal(distribution.probabilities().begin(), distribution.probabilities().end(), reference.probabilities().begin()));
        check(error([&] { static_cast<void>(sampling_distribution_after_scan(a, {}, certificate)); }) == "expired sampling validation");
        auto previous = SamplingRowScan::inspect(std::span<const float>(a), [] {});
        auto moved = std::move(previous);
        check(error([&] { static_cast<void>(previous.consume(a)); }) == "expired sampling validation");
        check(error([&] { static_cast<void>(moved.consume(b)); }) == "expired sampling validation");
        configure_prepared_sampling(false);
        std::cout << "prepared sampling distribution, RNG, rejected-row, precedence and expiry checks passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
