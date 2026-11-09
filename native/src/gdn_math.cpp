#include "tokspur/gdn_math.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/operators.hpp"

#include <algorithm>
#include <cmath>

namespace tokspur {
namespace {
void finite(std::span<const double> input) {
    for (const auto x : input) if (!std::isfinite(x)) throw FormatError("nonfinite GDN preparation input/result");
}
}
void validate_gdn_shape(GdnShape s) {
    if (!s.channels || s.channels > 65536 || !s.key_heads || s.key_heads > 64 ||
        !s.value_heads || s.value_heads > 64 || s.value_heads % s.key_heads ||
        !s.key_width || s.key_width > 256 || !s.value_width || s.value_width > 256 ||
        s.kernel < 2 || s.kernel > 64) throw FormatError("unsupported GDN shape");
    if (2ull * s.key_heads * s.key_width + uint64_t(s.value_heads) * s.value_width > 65536) throw FormatError("GDN projection width limit");
}
void validate_gdn_policy(GdnPolicy p) {
    if (!std::isfinite(p.query_scale) || p.query_scale <= 0 || !std::isfinite(p.l2_epsilon) || p.l2_epsilon <= 0 ||
        uint32_t(p.l2_mode) > 1 || !std::isfinite(p.rms_epsilon) || p.rms_epsilon <= 0) throw FormatError("invalid GDN numeric policy");
}
uint64_t gdn_qkv_count(GdnShape s) {
    validate_gdn_shape(s);
    return 2ull * s.key_heads * s.key_width + uint64_t(s.value_heads) * s.value_width;
}
uint64_t gdn_state_count(GdnShape s) {
    validate_gdn_shape(s);
    return uint64_t(s.value_heads) * s.key_width * s.value_width;
}
GdnPrepared prepare_gdn_reference(std::span<const double> input, GdnShape s, GdnPolicy p) {
    validate_gdn_policy(p);
    if (input.size() != gdn_qkv_count(s)) throw FormatError("GDN preparation shape mismatch");
    finite(input);
    GdnPrepared out{std::vector<double>(s.value_heads * s.key_width), std::vector<double>(s.value_heads * s.key_width),
                    std::vector<double>(s.value_heads * s.value_width)};
    const auto block = s.key_heads * s.key_width;
    for (uint32_t h = 0; h < s.value_heads; ++h) {
        const auto base = (h % s.key_heads) * s.key_width;
        double qs = 0, ks = 0;
        for (uint32_t k = 0; k < s.key_width; ++k) {
            qs += input[base + k] * input[base + k];
            ks += input[block + base + k] * input[block + base + k];
        }
        if (!std::isfinite(qs) || !std::isfinite(ks)) throw FormatError("GDN L2 accumulation overflow");
        const auto denominator = [&](double sum) {
            const auto value = p.l2_mode == L2Mode::add_squared_epsilon ? sum + p.l2_epsilon : std::max(sum, double(p.l2_epsilon));
            if (!std::isfinite(value)) throw FormatError("GDN L2 denominator overflow");
            return std::sqrt(value);
        };
        const auto qd = denominator(qs), kd = denominator(ks);
        for (uint32_t k = 0; k < s.key_width; ++k) {
            out.query[h * s.key_width + k] = input[base + k] / qd * p.query_scale;
            out.key[h * s.key_width + k] = input[block + base + k] / kd;
        }
        std::copy_n(input.begin() + 2 * block + h * s.value_width, s.value_width, out.value.begin() + h * s.value_width);
    }
    finite(out.query); finite(out.key);
    return out;
}
GdnGates gdn_gates_reference(std::span<const double> alpha, std::span<const double> beta,
                             std::span<const double> rates, std::span<const double> bias) {
    if (alpha.empty() || alpha.size() > 64 || beta.size() != alpha.size() || rates.size() != alpha.size() || bias.size() != alpha.size()) throw FormatError("GDN gate shape mismatch");
    finite(alpha); finite(beta); finite(rates); finite(bias);
    GdnGates out{std::vector<double>(alpha.size()), std::vector<double>(alpha.size())};
    for (size_t h = 0; h < alpha.size(); ++h) {
        if (rates[h] > 0) throw FormatError("GDN stored rate must be nonpositive");
        const auto exponent = rates[h] * stable_softplus(alpha[h] + bias[h]);
        if (!std::isfinite(exponent)) throw FormatError("GDN rate product overflow");
        out.decay[h] = std::exp(exponent);
        out.beta[h] = stable_sigmoid(beta[h]);
    }
    return out;
}
} // namespace tokspur
