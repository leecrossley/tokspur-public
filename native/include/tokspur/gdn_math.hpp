#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace tokspur {

struct GdnShape {
    uint32_t channels, key_heads, value_heads, key_width, value_width, kernel;
};
// Epsilon is in squared-norm units in both modes. Policies stay explicit until
// actual checkpoint layer/logit comparison qualifies the complete model path.
enum class L2Mode : uint32_t { add_squared_epsilon, floor_squared_epsilon };
struct GdnPolicy {
    float query_scale, l2_epsilon;
    L2Mode l2_mode;
    float rms_epsilon;
};
void validate_gdn_shape(GdnShape shape);
void validate_gdn_policy(GdnPolicy policy);
uint64_t gdn_qkv_count(GdnShape shape);
uint64_t gdn_state_count(GdnShape shape);

struct GdnPrepared { std::vector<double> query, key, value; };
// Stored GGUF order is all Q, all K, then ratio-major V. For value head h,
// the corresponding Q/K head is h % key_heads, not h / head_ratio.
GdnPrepared prepare_gdn_reference(std::span<const double> activated_qkv, GdnShape shape, GdnPolicy policy);
struct GdnGates { std::vector<double> decay, beta; };
// The checkpoint already stores negative exp(A_log); applying exp again is a
// different model. This API deliberately accepts only finite nonpositive rates.
GdnGates gdn_gates_reference(std::span<const double> alpha_logits, std::span<const double> beta_logits,
                             std::span<const double> negative_rates, std::span<const double> dt_bias);

} // namespace tokspur
