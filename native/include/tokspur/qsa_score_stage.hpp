#pragma once

#include "tokspur/attention_math.hpp"
#include "tokspur/gguf.hpp"

namespace tokspur {
struct QsaScoreStageBytes { uint32_t attended_tokens; uint64_t scores; };
// Fixed actual-target component, not a generic attention plan or allocation.
inline QsaScoreStageBytes qsa_score_stage_bytes(uint32_t prefix, QsaShape s) {
    validate_qsa_shape(s);
    if (s.channels != 2560 || s.query_heads != 24 || s.kv_heads != 2 || s.head_width != 256 ||
        s.index_heads != 4 || s.index_width != 128 || s.rotary_width != 64 ||
        s.compression != 4 || s.token_budget != 2048 || s.capacity != 65536 || !prefix || prefix > s.capacity)
        throw FormatError("score staging requires the fixed actual-target QSA geometry/prefix");
    const auto blocks = prefix / 4;
    const uint32_t tokens = (blocks < 512 ? blocks : 512) * 4 + prefix % 4;
    return {tokens, uint64_t(24) * tokens * sizeof(float)};
}
} // namespace tokspur
