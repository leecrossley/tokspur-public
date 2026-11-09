#include "tokspur/recurrent_reference.hpp"
#include "tokspur/gguf.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>

namespace {
size_t checks = 0;
void close(double a, double b) {
    ++checks; if (!std::isfinite(a) || std::abs(a - b) > 1e-12) throw std::runtime_error("hand-calculated recurrent reference differs");
}
void reject(const std::function<void()>& f) {
    ++checks; try { f(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid recurrent CPU reference accepted");
}
}
int main() {
    try {
        using namespace tokspur;
        const std::array<float, 1> one{1}, zero{0};
        const std::array<float, 2> ones{1, 1}, zeros{0, 0};
        const std::array<float, 3> qkv{1, 1, 1};
        const std::array<double, 1> gain{1}, rate{0}, bias{0};
        const std::array<double, 6> conv{0.25, 1, 0.25, 1, 0.25, 1};
        const auto mat = [](const auto& v) { return ReferenceMatrix{std::as_bytes(std::span(v)), 0}; };
        const ReferenceResidualWeights residual{gain, mat(zero), mat(zero), mat(zero)};
        const ReferenceGdnWeights gdn{mat(qkv), mat(zero), mat(zero), mat(zero), mat(one), conv, rate, bias, gain};
        const ReferenceMoeWeights moe{mat(zeros), mat(ones), mat(ones), mat(ones), mat(one), mat(one), mat(one), mat(zero)};
        const ReferenceLayerWeights w{residual, residual, gdn, moe};
        const RecurrentShape s{{1, 1, 1}, {1, 1, 1, 1, 1, 2}, {1, 1, 2, 2}};
        const GdnPolicy p{1, 1e-4f, L2Mode::add_squared_epsilon, 1e-4f};
        std::array<double, 1> state{0}; std::array<double, 3> history{0, 0, 0};
        const auto silu = [](double x) { return x / (1 + std::exp(-x)); };
        double previous = 0, matrix = 0;
        for (double token : {1.0, -0.7, 0.2, 0.0}) {
            const std::array<double, 1> input{token};
            const auto result = recurrent_reference(w, input, state, history, s, p, RouterMode::softmax, true);
            const double read = 0.5 * token / std::sqrt(token * token + p.rms_epsilon);
            const double value = silu(read + 0.25 * previous);
            const double key = value / std::sqrt(value * value + p.l2_epsilon);
            matrix = matrix * (1 - 0.5 * key * key) + 0.5 * key * value;
            const double raw = matrix * key;
            const double attention = 0.5 * raw / std::sqrt(raw * raw + p.rms_epsilon);
            const double intermediate = token + attention;
            const double ffn_read = 0.5 * intermediate / std::sqrt(intermediate * intermediate + p.rms_epsilon);
            const double ffn = 1.5 * silu(ffn_read) * ffn_read;
            const auto core = gdn_projection_reference(gdn, std::array<double, 1>{read}, state, history, s.gdn, p);
            close(core.delta.state[0], matrix); close(core.output[0], attention);
            close(result.attention_read.read.input[0], read); close(result.attention.delta.state[0], matrix);
            close(result.attention.delta.output[0], raw); close(result.attention.output[0], attention);
            close(result.intermediate_streams[0], intermediate); close(result.ffn_read.read.input[0], ffn_read);
            close(result.ffn.output[0], ffn); close(result.output_streams[0], intermediate + ffn);
            ++checks; if (result.ffn.routes.indices != std::vector<uint32_t>({0, 1})) throw std::runtime_error("CPU reference route tie mismatch");
            state[0] = matrix;
            for (auto& x : history) x = read;
            previous = read;
        }
        auto bad = w;
        bad.attention.qkv.bytes = std::as_bytes(std::span(one));
        reject([&] { static_cast<void>(recurrent_reference(bad, std::array<double, 1>{1}, state, history, s, p, RouterMode::softmax, true)); });
        bad = w; const std::array<double, 1> invalid_rate{1}; bad.attention.negative_rates = invalid_rate;
        reject([&] { static_cast<void>(recurrent_reference(bad, std::array<double, 1>{1}, state, history, s, p, RouterMode::softmax, true)); });
        auto shape = s; shape.moe.channels = 2;
        reject([&] { static_cast<void>(recurrent_reference(w, std::array<double, 1>{1}, state, history, shape, p, RouterMode::softmax, true)); });
        reject([&] { static_cast<void>(gdn_projection_reference(gdn, std::array<double, 2>{1, 1}, state, history, s.gdn, p)); });
        reject([&] { static_cast<void>(gdn_projection_reference(gdn, std::array<double, 1>{1}, std::span<const double>{}, history, s.gdn, p)); });
        reject([&] { static_cast<void>(gdn_projection_reference(gdn, std::array<double, 1>{1}, state, std::span<const double>{}, s.gdn, p)); });
        std::cout << "Connected CPU reference checks passed: " << checks << '\n';
    } catch (const std::exception& e) { std::cerr << "CPU reference check failed: " << e.what() << '\n'; return 1; }
}
