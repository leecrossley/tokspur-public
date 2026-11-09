#pragma once

#include "tokspur/qsa_score_stage.hpp"
#include <algorithm>

namespace tokspur {
// New composition admission, not a widening of qsa_score_stage_bytes. The same
// owned generic shader expressions serve bounded generated stacks and the
// exact real architecture. The private target still admits only actual weights.
inline QsaScoreStageBytes qsa_stage_v8_bytes(uint32_t prefix, QsaShape s) {
    validate_qsa_shape(s);
    if (s.channels > 2560 || s.query_heads > 24 || s.kv_heads > 2 || s.head_width > 256 ||
        s.index_heads > 4 || s.index_width > 128 || s.rotary_width > 64 ||
        s.compression != 4 || s.token_budget > 2048 || !prefix || prefix > s.capacity)
        throw FormatError("stage-v8 requires bounded actual-or-generated QSA geometry/prefix");
    const auto blocks = prefix / 4, budget = qsa_block_budget(s);
    const uint32_t tokens = std::min(blocks, budget) * 4 + prefix % 4;
    return {tokens, uint64_t(s.query_heads) * tokens * sizeof(float)};
}
inline constexpr uint64_t maximum_qsa_stage_v8_scores = 24ull * 2051 * sizeof(float);
} // namespace tokspur
