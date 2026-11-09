#include "tokspur/routing.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

namespace {
size_t checks = 0;
void require(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
void rejected(const std::function<void()>& action) {
    ++checks; try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid router input accepted");
}
}

int main() {
    try {
        std::mt19937 random(0x726f7574);
        for (size_t trial = 0; trial < 1000; ++trial) {
            const size_t count = 1 + random() % 1024, selected = 1 + random() % std::min(size_t(64), count);
            std::vector<double> logits(count), probabilities(count);
            for (auto& value : logits) value = (int(random() % 2001) - 1000) / 100.0;
            for (const auto mode : {tokspur::RouterMode::softmax, tokspur::RouterMode::sigmoid}) {
                double sum = 0;
                for (size_t i = 0; i < count; ++i) {
                    probabilities[i] = mode == tokspur::RouterMode::softmax ? std::exp(logits[i]) : 1 / (1 + std::exp(-logits[i]));
                    sum += probabilities[i];
                }
                if (mode == tokspur::RouterMode::softmax) for (auto& value : probabilities) value /= sum;
                std::vector<uint32_t> order(count);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return probabilities[a] > probabilities[b]; });
                order.resize(selected);
                for (const bool normalize : {false, true}) {
                    const auto actual = tokspur::route_reference(logits, selected, mode, normalize);
                    require(actual.indices == order, "route IDs differ from independent probability ordering");
                    double denominator = 1;
                    if (normalize) { denominator = 0; for (const auto id : order) denominator += probabilities[id]; }
                    for (size_t i = 0; i < selected; ++i) require(std::abs(actual.weights[i] - probabilities[order[i]] / denominator) < 1e-13, "route weights differ");
                }
            }
        }
        const std::array<double, 5> ties{3, 3, 2, 1, 3};
        const auto tied = tokspur::route_reference(ties, 3, tokspur::RouterMode::softmax, true);
        require(tied.indices == std::vector<uint32_t>({0, 1, 4}), "tie IDs are not stable");
        const std::array<double, 4> experts{1, 2, 3, 4};
        const std::array<double, 2> weights{0.25, 0.75}, shared{5, 6};
        const auto mix = tokspur::mixture_reference(experts, weights, shared, 0);
        require(mix == std::vector<double>({5, 6.5}), "mixture scalar golden");
        rejected([&] { static_cast<void>(tokspur::route_reference(ties, 0, tokspur::RouterMode::softmax, true)); });
        rejected([&] { static_cast<void>(tokspur::route_reference(ties, 6, tokspur::RouterMode::softmax, true)); });
        rejected([&] { static_cast<void>(tokspur::route_reference({}, 1, tokspur::RouterMode::softmax, true)); });
        rejected([&] { static_cast<void>(tokspur::route_reference(ties, 1, static_cast<tokspur::RouterMode>(99), true)); });
        const std::array<double, 1> nan{std::numeric_limits<double>::quiet_NaN()}, extreme{-1000};
        rejected([&] { static_cast<void>(tokspur::route_reference(nan, 1, tokspur::RouterMode::softmax, true)); });
        require(tokspur::route_reference(extreme, 1, tokspur::RouterMode::sigmoid, true).weights == std::vector<double>({1}), "normalized sigmoid underflow");
        const auto stable = tokspur::route_reference(extreme, 1, tokspur::RouterMode::softmax, true);
        require(stable.indices == std::vector<uint32_t>({0}) && stable.weights == std::vector<double>({1}), "shifted softmax underflow");
        rejected([&] { static_cast<void>(tokspur::mixture_reference(experts, weights, {}, 0)); });
        rejected([&] { static_cast<void>(tokspur::mixture_reference(experts, nan, shared, 0)); });
        std::cout << "Routing reference checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Routing reference check failed: " << error.what() << '\n'; return 1;
    }
}
