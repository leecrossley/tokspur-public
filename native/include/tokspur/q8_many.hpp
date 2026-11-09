#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tokspur {
struct Q8ManyShape { uint32_t columns, rows, width; };
struct Q8ManyBytes { uint64_t matrix, input, output; };
// Explicit bounded experiment. Vectors/results are token-major; the encoded
// Q8_0 matrix is unchanged and shared by every token.
Q8ManyBytes q8_many_bytes(Q8ManyShape shape);
// Greedy physical widths. Three is admitted only by the explicit private build and selector.
uint32_t q8_many_width(uint32_t remaining, bool three_rows = false);
uint32_t q8_many_chunks(uint32_t rows, bool three_rows = false);
// Scalar double-accumulating numerical reference, never the fast path.
std::vector<double> reference_q8_many(std::span<const std::byte> matrix,
                                    std::span<const float> input, Q8ManyShape shape,
                                    uint64_t max_output_bytes);
int q8_many_check_cli(int argc, char** argv);
} // namespace tokspur
