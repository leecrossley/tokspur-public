#include "tokspur/gdn_math.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

namespace {
size_t checks = 0;
void require(bool b, const char* s) { ++checks; if (!b) throw std::runtime_error(s); }
void reject(const std::function<void()>& f) {
    ++checks; try { f(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("invalid GDN input accepted");
}
}
int main() {
    try {
        using namespace tokspur;
        std::mt19937 rng(0x67646e31);
        for (const GdnShape s : std::array<GdnShape, 4>{{{7, 2, 6, 5, 3, 4}, {1, 1, 1, 1, 1, 2}, {2560, 16, 48, 128, 128, 4}, {32, 4, 8, 31, 17, 3}}}) {
            std::vector<double> raw(gdn_qkv_count(s));
            for (auto& x : raw) x = (int(rng() % 2001) - 1000) / 1000.0;
            for (const auto mode : {L2Mode::add_squared_epsilon, L2Mode::floor_squared_epsilon}) for (float scale : {1.0f, 0.088388346f}) {
                const GdnPolicy p{scale, 1e-6f, mode, 1e-6f};
                const auto out = prepare_gdn_reference(raw, s, p);
                const auto block = s.key_heads * s.key_width;
                for (uint32_t h = 0; h < s.value_heads; ++h) {
                    const auto base = (h % s.key_heads) * s.key_width;
                    double qs = 0, ks = 0;
                    for (uint32_t i = 0; i < s.key_width; ++i) { qs += raw[base + i] * raw[base + i]; ks += raw[block + base + i] * raw[block + base + i]; }
                    const auto qd = std::sqrt(mode == L2Mode::add_squared_epsilon ? qs + p.l2_epsilon : std::max(qs, double(p.l2_epsilon)));
                    const auto kd = std::sqrt(mode == L2Mode::add_squared_epsilon ? ks + p.l2_epsilon : std::max(ks, double(p.l2_epsilon)));
                    for (uint32_t i = 0; i < s.key_width; ++i) {
                        require(std::abs(out.query[h * s.key_width + i] - scale * raw[base + i] / qd) < 1e-12, "expanded Q order or normalization");
                        require(std::abs(out.key[h * s.key_width + i] - raw[block + base + i] / kd) < 1e-12, "expanded K order or normalization");
                    }
                    for (uint32_t i = 0; i < s.value_width; ++i) require(out.value[h * s.value_width + i] == raw[2 * block + h * s.value_width + i], "V head order");
                }
            }
            std::fill(raw.begin(), raw.end(), 0);
            const auto zero = prepare_gdn_reference(raw, s, {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f});
            require(std::all_of(zero.query.begin(), zero.query.end(), [](double x) { return x == 0; }), "zero L2 normalization");
        }
        const std::array<double, 5> alpha{-1000, 1000, 0, 0, 2}, beta{-1000, 1000, 0, 2, -2}, rates{-1, -2, 0, -1, -4}, bias{0, 0, 1, 0, -2};
        const auto g = gdn_gates_reference(alpha, beta, rates, bias);
        require(g.decay[0] == 1 && g.decay[1] == 0 && g.decay[2] == 1, "stable decay extremes");
        require(std::abs(g.decay[3] - 0.5) < 1e-12 && std::abs(g.decay[4] - 0.0625) < 1e-12, "negative rates used directly");
        require(g.beta[0] == 0 && g.beta[1] == 1 && g.beta[2] == 0.5, "beta gate extremes");
        reject([&] { validate_gdn_shape({7, 2, 5, 5, 3, 4}); });
        reject([&] { validate_gdn_shape({7, 0, 6, 5, 3, 4}); });
        reject([&] { validate_gdn_policy({1, 0, L2Mode::add_squared_epsilon, 1e-6f}); });
        reject([&] { validate_gdn_policy({1, 1e-6f, L2Mode(2), 1e-6f}); });
        auto bad = rates; bad[0] = 1;
        reject([&] { static_cast<void>(gdn_gates_reference(alpha, beta, bad, bias)); });
        bad[0] = std::numeric_limits<double>::quiet_NaN();
        reject([&] { static_cast<void>(gdn_gates_reference(alpha, beta, bad, bias)); });
        std::vector<double> large(7, 1e300);
        reject([&] { static_cast<void>(prepare_gdn_reference(large, {1, 1, 1, 2, 3, 2}, {1, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f})); });
        std::cout << "Owned GDN math checks passed: " << checks << '\n';
    } catch (const std::exception& e) { std::cerr << "GDN math check failed: " << e.what() << '\n'; return 1; }
}
