#pragma once

#include "tokspur/gdn_core_tile.hpp"
#include "tokspur/recurrent_reference.hpp"

namespace tokspur {
struct GdnCoreTileComparison {
    uint64_t checks = 0, batches = 0, compared_values = 0;
    double maximum_scaled_error = 0;
    uint64_t serial_dispatches = 0, tile_dispatches = 0;
    std::string output_sha256, state_sha256, history_sha256;
};
// Generated or bounded actual weights only. Owns diagnostic scratch/state, not
// supplied weights; synchronous readback happens after complete batches only.
// Compares every valid stage and final-state bit to ordinary encode_gdn, then
// checks the independent double chain without changing tolerance on failure.
GdnCoreTileComparison check_gdn_core_tile(MetalGraph& graph, const GdnWeights& weights,
    const ReferenceGdnWeights& reference_weights, GdnShape shape, GdnPolicy policy,
    StateTile tile, std::span<const float> inputs, std::span<const float> accepted_matrix,
    std::span<const float> accepted_history, bool every_replay_prefix);
} // namespace tokspur
