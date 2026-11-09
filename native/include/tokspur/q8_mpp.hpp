#pragma once

#include <cstdint>

namespace tokspur {
struct Q8MppBytes { uint64_t matrix, input, output; };
enum class Q8MppLayout : uint32_t { single, independent_four, single_wide };
// Private O-tier operator. Original Q8_0 weights and FP32 token-major rows;
// arbitrary batches through 2048, bounded views, different sum association.
Q8MppBytes q8_mpp_bytes(uint32_t columns, uint32_t rows, uint32_t batches);
const char* q8_mpp_kernel(Q8MppLayout layout);
uint32_t q8_mpp_simdgroups(Q8MppLayout layout);
}
