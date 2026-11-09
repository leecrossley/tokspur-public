#pragma once
// Private lossless expert-layout spike. Not linked into the engine, server or ordinary Makefile targets. Q4_K, Q5_1 and Q8_0 rows are admitted.
#include <cstddef>
#include <cstdint>
#include <span>

namespace tokspur::lanes {

// Each row keeps its exact byte count. Block headers move to a leading metadata run; quant bytes are regrouped so that the bytes one SIMD lane reads for consecutive blocks are adjacent. No value is decoded or rounded.
struct RowLayout {
    uint32_t type = 0;       // GGUF ggml_type: 12 (Q4_K), 7 (Q5_1) or 8 (Q8_0)
    uint32_t columns = 0;    // elements per row
    uint32_t blocks = 0;     // quant blocks per row
    uint32_t meta_bytes = 0; // leading per-block header bytes
    uint64_t row_bytes = 0;  // identical to the original GGUF row
};

RowLayout row_layout(uint32_t type, uint32_t columns);

// Whole rows only: span sizes must be a multiple of row_bytes and equal.
void pack_rows(const RowLayout& layout, std::span<const std::byte> original, std::span<std::byte> packed);
void unpack_rows(const RowLayout& layout, std::span<const std::byte> packed, std::span<std::byte> original);

// Independent CPU decoders of one element, for value-equality checks. Both use the arithmetic of the owned gemv.metal element() decoder in FP32.
float decode_original(const RowLayout& layout, const std::byte* row, uint32_t column);
float decode_packed(const RowLayout& layout, const std::byte* row, uint32_t column);

// Byte position of original quant byte `index` (within block `block`) inside a packed row. Exposed for tests; metadata bytes are not covered.
uint64_t packed_quant_offset(const RowLayout& layout, uint32_t block, uint32_t index);

} // namespace tokspur::lanes
