#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace tokspur {
// Explicit component experiment. Capacity is 2/4/8/16/32; steps is 1..capacity.
// Buffers are capacity-sized, but unprocessed rows are never read/written.
// k=0 rollback means retain the accepted base without a dispatch.
struct StateTile { uint32_t capacity, steps; };
struct GdnConvTileBytes { uint64_t history, projected, weights, output; };
struct GdnDeltaTileBytes { uint64_t state, query_key, value_output, gates; };

void validate_state_tile(StateTile tile);
GdnConvTileBytes gdn_conv_tile_bytes(uint32_t channels, StateTile tile);
GdnDeltaTileBytes gdn_delta_tile_bytes(uint32_t heads, uint32_t value_width, StateTile tile);

struct ConvTileReference { std::vector<double> history, output; };
struct DeltaTileReference { std::vector<double> state, output; };

// Double references, not the production path. Inputs are capacity-sized; output
// includes only the processed prefix. The explicit budget covers final state
// and returned output bytes (not caller inputs or temporary allocator overhead).
ConvTileReference gdn_convolution_tile_reference(
    std::span<const double> accepted_history, std::span<const double> projected,
    std::span<const double> weights, uint32_t channels, StateTile tile,
    uint64_t max_returned_bytes);
DeltaTileReference gdn_delta_tile_reference(
    std::span<const double> accepted_state, std::span<const double> query,
    std::span<const double> key, std::span<const double> value,
    std::span<const double> decay, std::span<const double> beta,
    uint32_t heads, uint32_t value_width, StateTile tile,
    uint64_t max_returned_bytes);
} // namespace tokspur
