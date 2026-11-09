#pragma once

#include "tokspur/gdn_core_tile.hpp"
#include "tokspur/recurrent_reference.hpp"
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
#include "tokspur/moe_many.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
#include "tokspur/verify_nax.hpp"
#endif

namespace tokspur {
// Explicit known-input tile primitive. Every RecurrentScratch view is
// capacity-sized/token-major. Residual projection reuse is separately default
// off; ordinary target/stack callers retain their existing encoding policy.
RecurrentScratch allocate_recurrent_tile_scratch(MetalGraph& graph, RecurrentShape shape, uint32_t capacity);
RecurrentScratch recurrent_tile_token(MetalGraph& graph, const RecurrentScratch& scratch,
                                      RecurrentShape shape, uint32_t capacity, uint32_t step);
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
// Private capacity512 HC read: unchanged per-row RMS/activation/reduction,
// sharing only existing Q8 projection weights across independent rows.
void encode_known_residual_rows512(MetalGraph& graph, const ResidualWeights& weights,
    const RecurrentScratch& scratch, GpuView streams, RecurrentShape shape,
    StateTile tile, float rms_epsilon);
#endif
void encode_recurrent_layer_tile(MetalGraph& graph, const RecurrentWeights& weights,
    const RecurrentScratch& scratch, const GdnState& accepted, const GdnState& candidate,
    GpuView input_streams, GpuView output_streams, RecurrentShape shape, GdnPolicy policy,
    RouterMode mode, bool normalize_selected, StateTile tile, bool reuse_residual_projections = false
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
    , const MoeManyScratch* grouped_ffn = nullptr, bool reuse_residual_rows512 = false
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    , const VerifyNaxMoeScratch* verify_nax_moe = nullptr
#endif
    );
// Recover only this layer's GDN matrix/history, from the SAME accepted base.
// All layer outputs and all scratch/tape are protected from the replay writes.
// k=0 retains base without work. No publication occurs in either encode API.
void encode_recurrent_tile_prefix(MetalGraph& graph, const RecurrentWeights& weights,
    const RecurrentScratch& tape, const GdnState& accepted, const GdnState& candidate,
    GpuView retained_outputs, const GdnCoreReplayScratch& replay, RecurrentShape shape, StateTile tile);

struct RecurrentTileComparison {
    uint64_t checks = 0, batches = 0, compared_values = 0;
    double maximum_scaled_error = 0;
    uint64_t serial_dispatches = 0, tile_dispatches = 0;
    std::string output_sha256, state_sha256, history_sha256;
};
// Diagnostic only. Exact all32 scratch/output/final-state comparisons and owned
// independent double references; no timing result or production fallback.
RecurrentTileComparison check_recurrent_tile(MetalGraph& graph, const RecurrentWeights& weights,
    const ReferenceLayerWeights& reference_weights, RecurrentShape shape, GdnPolicy policy,
    RouterMode mode, bool normalize_selected, StateTile tile, std::span<const float> inputs,
    std::span<const float> accepted_matrix, std::span<const float> accepted_history, bool every_replay_prefix,
    bool reuse_residual_projections = false);
int recurrent_tile_check_cli(int argc, char** argv);
} // namespace tokspur
