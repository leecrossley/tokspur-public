#pragma once

#include "tokspur/gguf.hpp"

#include <cstdint>

namespace tokspur {
struct Q8SplitShape { uint32_t columns, rows, splits; };
struct Q8SplitBytes { uint64_t matrix, input, output, partials; };

// Original bounded column-split experiment; encoded Q8_0 weights are unchanged.
// These are logical view sizes, not actual/padded Metal resource charges.
inline Q8SplitBytes q8_split_bytes(Q8SplitShape shape) {
    const auto [columns, rows, splits] = shape;
    if (!columns || columns > 65536 || columns % 32 || !rows || rows > 1024 ||
        !(splits == 1 || splits == 2 || splits == 4 || splits == 8 || splits == 16 || splits == 32))
        throw FormatError("unsupported Q8 split geometry");
    return {uint64_t(columns) / 32 * rows * 34, uint64_t(columns) * 4,
            uint64_t(rows) * 4, uint64_t(rows) * splits * 4};
}
} // namespace tokspur
