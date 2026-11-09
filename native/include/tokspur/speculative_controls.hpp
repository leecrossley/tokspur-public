#pragma once

#include "tokspur/gguf.hpp"

#include <cstdint>
#include <string_view>

namespace tokspur {

enum class DFlash2Precision : uint32_t { q8_0 = 0, bf16 = 1 };

inline constexpr uint64_t maximum_dflash2_tap_budget_bytes = 256ull << 20;

inline uint64_t dflash2_tap_budget(uint32_t known_capacity) {
    switch (known_capacity) {
    case 0: case 8: return 8ull << 20;
    case 512: return 48ull << 20;
    case 1024: return 64ull << 20;
    case 2048: return 160ull << 20;
    default: throw FormatError("unsupported DFlash2 tap capacity");
    }
}

struct DFlash2TapPlan {
    uint32_t known_capacity;
    uint64_t known_bytes, verify_bytes;
};

// Plan both live banks before any Metal allocation. The graph still checks the actual allocated-size charge against this budget.
inline DFlash2TapPlan dflash2_tap_plan(uint32_t known_capacity, uint32_t slots, uint32_t width, uint64_t budget) {
    static_cast<void>(dflash2_tap_budget(known_capacity));
    if (!slots || slots > 16 || !width || width % 4 || !budget || budget > maximum_dflash2_tap_budget_bytes)
        throw FormatError("invalid DFlash2 tap geometry or budget");
    const uint32_t capacity = known_capacity ? known_capacity : 8;
    // The admitted capacity and slot bounds keep these products within uint64_t even at the largest uint32_t width.
    const uint64_t row_bytes = uint64_t(slots) * width * sizeof(float);
    const DFlash2TapPlan plan{capacity, uint64_t(capacity) * row_bytes, 8ull * row_bytes};
    if (plan.known_bytes + plan.verify_bytes > budget) throw FormatError("DFlash tap exceeds its explicit GPU budget");
    return plan;
}

inline DFlash2Precision parse_dflash2_precision(std::string_view value) {
    if (value == "q8_0") return DFlash2Precision::q8_0;
    if (value == "bf16") return DFlash2Precision::bf16;
    throw FormatError("native DFlash2 precision must be q8_0 or bf16");
}

inline uint32_t parse_native_speculative_depth(std::string_view value) {
    if (value == "1") return 1;
    if (value == "2") return 2;
    if (value == "3") return 3;
    if (value == "7") return 7;
    throw FormatError("native speculative depth must be 1, 2, 3 or 7");
}

inline constexpr uint32_t dflash2_default_max_drafts = 7;
inline constexpr uint32_t dflash2_width4_max_drafts = 3;
inline constexpr uint32_t dflash2_default_vocab_slice = 32768;

inline uint32_t parse_dflash2_max_drafts(std::string_view value) {
    if (value.size() == 1 && value[0] >= '1' && value[0] <= '7') return uint32_t(value[0] - '0');
    throw FormatError("native DFlash2 max_drafts must be an integer in 1..7");
}

inline uint32_t parse_dflash2_vocab_slice(std::string_view value) {
    if (value == "0" || value == "full") return 0;
    if (value.empty() || value[0] == '0') {
        throw FormatError("native DFlash2 vocab_slice must be 0, full, or a positive multiple of 32 in [256, 1048576]");
    }
    uint64_t parsed = 0;
    for (char ch : value) {
        if (ch < '0' || ch > '9') {
            throw FormatError("native DFlash2 vocab_slice must be 0, full, or a positive multiple of 32 in [256, 1048576]");
        }
        parsed = parsed * 10ull + uint64_t(ch - '0');
        if (parsed > 1048576ull) {
            throw FormatError("native DFlash2 vocab_slice must be 0, full, or a positive multiple of 32 in [256, 1048576]");
        }
    }
    if (parsed < 256ull || (parsed % 32ull) != 0ull) {
        throw FormatError("native DFlash2 vocab_slice must be 0, full, or a positive multiple of 32 in [256, 1048576]");
    }
    return uint32_t(parsed);
}

} // namespace tokspur
