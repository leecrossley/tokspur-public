#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tokspur::raw_decode {

enum class Format : std::uint32_t { q4_k = 12, q5_1 = 7 };

// Raw GGUF only. Strides are bytes and IDs remain GPU inputs in the operator.
struct alignas(8) Args {
    std::uint32_t columns, rows, experts, selected;
    std::uint32_t row_bytes, broadcast, reserved0 = 0, reserved1 = 0;
    std::uint64_t input_stride, output_stride, expert_stride, matrix_bytes;
};
static_assert(sizeof(Args) == 64);
static_assert(offsetof(Args, input_stride) == 32);
static_assert(offsetof(Args, matrix_bytes) == 56);

struct Plan {
    Format format;
    Args args;
    std::uint64_t input_bytes, output_bytes, index_bytes;
    std::uint32_t grid_x, grid_y, threads = 64;
    const char* kernel;
};

inline constexpr std::uint64_t matrix_limit = 2ull << 30;
inline constexpr std::uint64_t vector_limit = 256ull << 20;
inline constexpr std::uint64_t generated_gpu_budget = 64ull << 20;
inline constexpr double component_scaled_error_limit = 5e-5;

Plan make_plan(Format, std::uint32_t columns, std::uint32_t rows, std::uint32_t experts,
    std::uint32_t selected, bool broadcast, std::uint64_t input_stride = 0, std::uint64_t output_stride = 0);
void validate_bindings(const Plan&, std::uint64_t weights_bytes, std::uint64_t input_bytes,
    std::uint64_t output_bytes, std::uint64_t indices_bytes, std::uint64_t weights_offset = 0,
    std::uint64_t input_offset = 0, std::uint64_t output_offset = 0, std::uint64_t indices_offset = 0);
std::uint32_t block_columns(Format);
std::uint32_t block_bytes(Format);
float half_value(std::uint16_t bits);
std::uint16_t read16(const std::uint8_t*);
void write16(std::uint8_t*, std::uint16_t);
std::uint32_t read32(const std::uint8_t*);
void write32(std::uint8_t*, std::uint32_t);
void set_q4_group(std::uint8_t* block, unsigned group, unsigned scale, unsigned minimum);
void set_quant(Format, std::uint8_t* block, unsigned column, unsigned value);
double represented_weight(Format, const std::uint8_t* block, unsigned column);
double dot_f64(Format, const std::uint8_t* row, std::span<const float> input);
float dot_source_order(Format, const std::uint8_t* row, std::span<const float> input);
void fill_generated(Format, const Plan&, std::span<std::uint8_t> weights, std::span<float> input, std::uint32_t seed = 44);
std::vector<std::uint32_t> generated_routes(std::uint32_t selected, std::uint32_t experts);

}
