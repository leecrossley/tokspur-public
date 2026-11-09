#include "tokspur/operators.hpp"
#include "tokspur/gguf.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
size_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, const char* message) {
    require(std::abs(actual - expected) <= 2e-12 * std::max(1.0, std::abs(expected)), message);
}
void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid operator reference request accepted");
}

void gates_and_residual() {
    close(tokspur::stable_sigmoid(0), 0.5, "sigmoid zero");
    close(tokspur::stable_sigmoid(1000), 1, "sigmoid large positive");
    close(tokspur::stable_sigmoid(-1000), 0, "sigmoid large negative");
    close(tokspur::stable_softplus(1000), 1000, "softplus large positive");
    close(tokspur::stable_softplus(-1000), 0, "softplus large negative");
    close(tokspur::stable_softplus(0), std::log(2.0), "softplus zero");
    close(tokspur::stable_silu(0), 0, "SiLU zero");
    rejected([] { static_cast<void>(tokspur::stable_sigmoid(std::numeric_limits<double>::infinity())); });
    const std::array<double, 2> streams{2, 3};
    const std::array<double, 2> gains{1, 2};
    const std::array<double, 2> down{0.25, -0.5};
    const std::array<double, 2> up{0.3, -0.7};
    const std::array<double, 4> inject{0.1, 0.2, -0.5, 0.7};
    const double a = 2 / std::sqrt(5.0);
    const double b = 6 / std::sqrt(10.0);
    const double projected = (0.25 * a - 0.5 * b) / 2;
    const double hidden = projected / (1 + std::exp(-projected));
    const double expected = (a / (1 + std::exp(-0.3 * hidden)) + b / (1 + std::exp(0.7 * hidden))) / 2;
    const auto read = tokspur::gated_residual_reference(streams, gains, down, up, inject, 2, 1, 1, 1);
    close(read.input[0], expected, "GR scalar mathematical oracle");
    close(read.write_scales[0], 2 / (1 + std::exp(-(0.1 * a + 0.2 * b) / 2)), "GR first write gate");
    close(read.write_scales[1], 2 / (1 + std::exp(-(-0.5 * a + 0.7 * b) / 2)), "GR second write gate");
    const auto written = tokspur::residual_write_reference(streams, read.input, read.write_scales);
    close(written[0], 2 + read.input[0] * read.write_scales[0], "GR first residual write");
    close(written[1], 3 + read.input[0] * read.write_scales[1], "GR second residual write");
    const std::array<double, 4> grouped{3, 4, 0, 2};
    const std::array<double, 4> ones{1, 1, 1, 1};
    const auto normalized = tokspur::group_rms_reference(grouped, ones, 2, 0.5);
    close(normalized[0], 3 / std::sqrt(13.0), "group RMS first group");
    close(normalized[3], 2 / std::sqrt(2.5), "group RMS independent groups");
    rejected([&] { static_cast<void>(tokspur::group_rms_reference(grouped, ones, 3, 1e-6)); });
    rejected([&] { static_cast<void>(tokspur::group_rms_reference(grouped, ones, 2, 0)); });
    rejected([&] { static_cast<void>(tokspur::gated_residual_reference(streams, gains, down, up, inject, 3, 1, 1, 1)); });
}

void delta() {
    std::mt19937 random(0x64656c74);
    const auto number = [&] { return (int(random() % 2001) - 1000) / 1000.0; };
    for (size_t trial = 0; trial < 500; ++trial) {
        const size_t heads = 1 + random() % 3;
        const size_t dk = 1 + random() % 7;
        const size_t dv = 1 + random() % 8;
        std::vector<double> state(heads * dk * dv), q(heads * dk), k(heads * dk), v(heads * dv), alpha(heads), beta(heads);
        for (auto& x : state) x = number();
        for (auto& x : q) x = number();
        for (auto& x : k) x = number();
        for (auto& x : v) x = number();
        for (auto& x : alpha) x = std::abs(number());
        for (auto& x : beta) x = std::abs(number());
        const auto original = state;
        const auto actual = tokspur::gated_delta_reference(state, q, k, v, alpha, beta, heads, dk, dv);
        // Independent dense transition alpha*(I - beta*k*k^T)*S + beta*k*v^T.
        std::vector<double> expected(state.size(), 0);
        for (size_t h = 0; h < heads; ++h) {
            for (size_t row = 0; row < dk; ++row) {
                for (size_t column = 0; column < dv; ++column) {
                    double transformed = 0;
                    for (size_t source = 0; source < dk; ++source) {
                        const auto transition = (row == source ? 1.0 : 0.0) - beta[h] * k[h * dk + row] * k[h * dk + source];
                        transformed += transition * original[h * dk * dv + source * dv + column];
                    }
                    const auto at = h * dk * dv + row * dv + column;
                    expected[at] = alpha[h] * transformed + beta[h] * k[h * dk + row] * v[h * dv + column];
                    close(actual.state[at], expected[at], "GDN dense matrix oracle");
                }
            }
            for (size_t column = 0; column < dv; ++column) {
                double output = 0;
                for (size_t row = 0; row < dk; ++row) output += expected[h * dk * dv + row * dv + column] * q[h * dk + row];
                close(actual.output[h * dv + column], output, "GDN independent readout");
            }
        }
        require(state == original, "GDN reference must not mutate committed input state");
        const auto replay = tokspur::gated_delta_reference(state, q, k, v, alpha, beta, heads, dk, dv);
        require(replay.state == actual.state && replay.output == actual.output, "GDN discarded candidate replay");
    }
    const std::array<double, 1> state{2}, q{0.5}, k{0.4}, v{1}, alpha{0.7}, beta{0.3};
    const auto one = tokspur::gated_delta_reference(state, q, k, v, alpha, beta, 1, 1, 1);
    close(one.state[0], 1.4528, "GDN scalar golden state");
    close(one.output[0], 0.7264, "GDN scalar golden output");
    const std::array<double, 1> bad_gate{1.1};
    rejected([&] { static_cast<void>(tokspur::gated_delta_reference(state, q, k, v, bad_gate, beta, 1, 1, 1)); });
    rejected([&] { static_cast<void>(tokspur::gated_delta_reference(state, q, k, v, alpha, beta, 1, 2, 1)); });
}

void convolution() {
    std::vector<double> history(6, 0);
    const std::array<double, 8> weights{1, 2, 3, 4, -1, -2, -3, -4};
    for (size_t step = 0; step < 10; ++step) {
        const std::array<double, 2> input{double(step + 1), double(2 * step + 1)};
        const auto original = history;
        const auto next = tokspur::causal_depthwise_reference(history, input, weights, 4);
        for (size_t channel = 0; channel < 2; ++channel) {
            double expected = 0;
            for (size_t lag = 0; lag < 4; ++lag) {
                if (lag > step) continue;
                const auto time = step - lag;
                const auto sample = channel ? double(2 * time + 1) : double(time + 1);
                expected += sample * weights[channel * 4 + 3 - lag];
            }
            close(next.output[channel], expected, "causal convolution sequence oracle");
        }
        require(history == original, "convolution reference must preserve committed input");
        history = next.history;
    }
    const std::array<double, 1> input{3}, weight{2};
    const auto instant = tokspur::causal_depthwise_reference({}, input, weight, 1);
    require(instant.history.empty() && instant.output[0] == 6, "kernel-one convolution");
    rejected([&] { static_cast<void>(tokspur::causal_depthwise_reference({}, input, weight, 0)); });
}
} // namespace

int main() {
    try {
        gates_and_residual(); delta(); convolution();
        std::cout << "Graph operator reference checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Graph operator reference check failed: " << error.what() << '\n';
        return 1;
    }
}
