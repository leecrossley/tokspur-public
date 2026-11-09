#pragma once

#include "tokspur/qsa_stage_v8.hpp"
#include <array>
#include <string_view>

namespace tokspur {
// Private COST-only schedule. Absence owns no extra GPU payload; explicit OFF
// prepares exactly the same workspace as both candidate modes.
enum class QsaVerifyRowsMode : uint32_t { off, core_only, core_and_q8 };
struct QsaVerifyRowsConfig { bool prepared = false; QsaVerifyRowsMode mode = QsaVerifyRowsMode::off; };
struct QsaVerifyRowsMemory {
    QsaVerifyRowsConfig config{};
    uint64_t logical_bytes = 0, charged_bytes = 0;
    uint32_t allocations = 0, borrowed_common_views = 0;
};
inline constexpr uint32_t qsa_verify_rows_capacity = 8, qsa_verify_rows_stride = 2051;
inline constexpr uint64_t qsa_verify_rows_row_scores = 24ull * qsa_verify_rows_stride;
inline constexpr uint64_t maximum_qsa_verify_rows_scores = 8 * qsa_verify_rows_row_scores * sizeof(float);
inline constexpr uint64_t maximum_qsa_verify_rows_factors = 2 * maximum_qsa_verify_rows_scores;
// Successful physical tiles, logical rows, score/factor/value passes. Legacy
// Stage/channel counters count ONLY legacy calls and remain unchanged.
using QsaVerifyRowsRoutes = std::array<uint64_t, 5>;
struct QsaVerifyRowsPlan {
    uint32_t base_prefix, rows, maximum_attended;
    std::array<uint32_t, 8> attended{};
    uint64_t vector_row_bytes, id_row_bytes;
};
inline QsaVerifyRowsPlan qsa_verify_rows_plan(uint32_t base, uint32_t rows, QsaShape shape) {
    validate_qsa_shape(shape);
    if (!rows || rows > 8 || base >= shape.capacity || rows > shape.capacity - base)
        throw FormatError("QSA verification rows require 1..8 causal prefixes within capacity");
    QsaVerifyRowsPlan out{base, rows, 0, {}, uint64_t(shape.query_heads) * shape.head_width * 4,
        uint64_t(qsa_block_budget(shape)) * 4};
    for (uint32_t row = 0; row < rows; ++row) {
        const auto b = qsa_stage_v8_bytes(base + row + 1, shape);
        if (shape.query_heads > 24 || b.attended_tokens > qsa_verify_rows_stride)
            throw FormatError("QSA verification row exceeds its fixed score stride");
        out.attended[row] = b.attended_tokens;
        out.maximum_attended = std::max(out.maximum_attended, b.attended_tokens);
    }
    return out;
}
inline const char* qsa_verify_rows_name(QsaVerifyRowsMode mode) {
    switch (mode) { case QsaVerifyRowsMode::off: return "off";
        case QsaVerifyRowsMode::core_only: return "core"; case QsaVerifyRowsMode::core_and_q8: return "q8"; }
    throw FormatError("invalid private QSA row mode");
}
inline QsaVerifyRowsMode qsa_verify_rows_mode(std::string_view value) {
    if (value == "off") return QsaVerifyRowsMode::off;
    if (value == "core") return QsaVerifyRowsMode::core_only;
    if (value == "q8") return QsaVerifyRowsMode::core_and_q8;
    throw FormatError("private QSA row mode must be off, core or q8");
}
} // namespace tokspur
