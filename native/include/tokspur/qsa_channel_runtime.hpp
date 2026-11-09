#pragma once

#include "tokspur/qsa_stage_v8.hpp"
#include <optional>
#include <string_view>

namespace tokspur {
inline constexpr uint64_t maximum_qsa_channel_factors = 2 * maximum_qsa_stage_v8_scores;
inline constexpr uint32_t qsa_channel_threads = 256;
inline std::optional<bool> qsa_channel_mode(std::string_view mode) {
    if (mode == "off") return false;
    if (mode == "on") return true;
    return {};
}
// Other startup/execution/override guards remain those of Stage-V8. This
// additional closed gate applies equally to explicit OFF and ON, not absence.
inline bool qsa_channel_startup_allowed(bool selected, bool stage_enabled, uint32_t exact_hc) {
    return !selected || (stage_enabled && exact_hc == 2);
}
inline uint64_t qsa_channel_factor_bytes(uint32_t prefix, QsaShape shape) {
    return 2 * qsa_stage_v8_bytes(prefix, shape).scores;
}
} // namespace tokspur
