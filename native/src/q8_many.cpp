#include "tokspur/q8_many.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"

#include <algorithm>
#include <cmath>

namespace tokspur {
Q8ManyBytes q8_many_bytes(Q8ManyShape s) {
    if (!s.columns || s.columns > 65536 || s.columns % 32 || !s.rows || s.rows > 1'000'000 ||
        (s.width != 1 && s.width != 2 && s.width != 4 && s.width != 8
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
         && s.width != 3
#endif
        ))
        throw FormatError("unsupported Q8-many projection shape");
    return {checked_mul(checked_mul(s.columns / 32, s.rows), 34),
        checked_mul(checked_mul(s.columns, s.width), 4), checked_mul(checked_mul(s.rows, s.width), 4)};
}
uint32_t q8_many_width(uint32_t remaining, bool three_rows) {
    if (!remaining) throw FormatError("empty Q8-many width plan");
#ifndef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    if (three_rows) throw FormatError("three-row Q8 requires the private build");
#endif
    if (remaining >= 8) return 8;
    if (remaining >= 4) return 4;
    if (three_rows && remaining == 3) return 3;
    return remaining >= 2 ? 2 : 1;
}
uint32_t q8_many_chunks(uint32_t rows, bool three_rows) {
    if (!rows || rows > 65536) throw FormatError("Q8-many chunk row bound");
    uint32_t chunks = 0;
    while (rows) { rows -= q8_many_width(rows, three_rows); ++chunks; }
    return chunks;
}
std::vector<double> reference_q8_many(std::span<const std::byte> matrix,
                                    std::span<const float> input, Q8ManyShape s,
                                    uint64_t max_output_bytes) {
    const auto bytes = q8_many_bytes(s);
    if (matrix.size() != bytes.matrix || input.size_bytes() != bytes.input || checked_mul(bytes.output, 2) > max_output_bytes)
        throw FormatError("Q8-many reference geometry/output budget mismatch");
    for (auto x : input) if (!std::isfinite(x)) throw FormatError("nonfinite Q8-many reference input");
    std::vector<double> result(bytes.output / 4);
    for (uint32_t token = 0; token < s.width; ++token) {
        const auto row = reference_gemv(8, matrix, s.columns, s.rows, input.subspan(uint64_t(token) * s.columns, s.columns));
        std::copy(row.begin(), row.end(), result.begin() + uint64_t(token) * s.rows);
    }
    return result;
}
} // namespace tokspur
