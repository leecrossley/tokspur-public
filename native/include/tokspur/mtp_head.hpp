#pragma once

#ifndef TOKSPUR_PRIVATE_NATIVE_MTP
#error "native MTP head requires the explicit private build"
#endif

#include "tokspur/mtp.hpp"
#include "tokspur/ple.hpp"
#include "tokspur/sampling.hpp"

namespace tokspur {

struct MtpDenseWeights {
    GpuMatrix query_gate, key, value, output;
    GpuView query_gains, key_gains, frequencies;
};
struct MtpHeadWeights {
    GpuView hidden_gains, embedding_gains;
    GpuMatrix embedding_hidden;
    ResidualWeights attention_read, ffn_read;
    MtpDenseWeights attention;
    MoeWeights ffn;
    MixerWeights mixer;
    GpuMatrix output; // Borrowed original target vocabulary projection.
};
struct MtpHeadShape {
    ResidualShape residual;
    MoeShape experts;
    uint32_t query_heads, kv_heads, head_width, rotary_width, vocabulary, capacity;
};
// Auxiliary proposal arithmetic only. These kernels never evaluate target P.
// Serial remains the default; tiles change FP32 association in draft Q.
enum class MtpDraftAttention : uint32_t { serial, tiled128, tiled256, split16 };
inline constexpr uint32_t mtp_draft_attention_tile_tokens(MtpDraftAttention mode) {
    return mode == MtpDraftAttention::tiled128 ? 128 : mode == MtpDraftAttention::tiled256 ? 256 : 0;
}
struct MtpHeadPolicy {
    QueryPacking query_packing = QueryPacking::head_query_gate;
    RotaryPairs rotary_pairs = RotaryPairs::split_half;
    float rms_epsilon = 1e-6f;
    RouterMode router = RouterMode::softmax;
    bool normalize_selected = true;
    MtpDraftAttention draft_attention = MtpDraftAttention::serial;
};
struct MtpHeadResult {
    std::span<const float> logits;
    BatchResult batch;
    uint32_t absolute_position = 0, cache_rows = 0;
};
struct MtpHeadFastResult {
    GpuLogitTopKRow topk{};
    BatchResult batch{};
    uint32_t absolute_position = 0, cache_rows = 0;
};
struct MtpHeadFastChainResult {
    std::array<GpuLogitTopKRow, 3> topks{};
    uint32_t depth = 0;
    BatchResult batch{};
    uint32_t first_position = 0, cache_rows = 0;
};
struct GpuMtpSampledLawRow {
    uint32_t count = 0;
    uint32_t chosen_token = 0;
    uint32_t used_ticket = 0;
    uint32_t ticket_index = 0;
    std::array<uint32_t, 20> ids{};
    std::array<uint32_t, 20> units24{};
    std::array<uint32_t, 20> reserved{};
};
static_assert(sizeof(GpuMtpSampledLawRow) == 256, "GpuMtpSampledLawRow ABI");
struct MtpHeadSampledChainResult {
    std::array<GpuLogitTopKRow, 3> topks{};
    std::array<GpuMtpSampledLawRow, 3> laws{};
    uint32_t depth = 0;
    BatchResult batch{};
    uint32_t first_position = 0, cache_rows = 0;
};

// The caller gives this head exclusive idle graph use. The imported shared
// sidecar sources are retained; the borrowed target output capability must
// remain live until destruction. No token-dependent allocations/imports.
// Teacher forcing consumes (target HC after token p-1, embedding token p) at p.
// The initial bridge explicitly starts at p=1, omitting unavailable H[-1].
class MtpHead {
public:
    MtpHead(MetalGraph& graph, const MtpSharedBinding& binding, GpuMatrix borrowed_output,
            uint32_t capacity = 65536, MtpDraftAttention draft_attention = MtpDraftAttention::serial,
            const std::map<std::string, GpuView>* pack_views = nullptr);
    // Generated controls only; caller retains immutable weight capabilities.
    MtpHead(MetalGraph& graph, MtpHeadWeights weights, MtpHeadShape shape, MtpHeadPolicy policy = {});
    ~MtpHead();
    MtpHead(const MtpHead&) = delete;
    MtpHead& operator=(const MtpHead&) = delete;

    void reset(uint32_t first_absolute_position = 1);
    void truncate(uint32_t committed_cache_rows); // Discards logits; stale suffix is invisible.
    [[nodiscard]] uint32_t cache_rows() const;
    [[nodiscard]] uint32_t next_absolute_position() const;
    [[nodiscard]] MtpHeadResult advance(std::span<const float> target_hc,
                                       std::span<const float> next_token_embedding,
                                       uint32_t absolute_position);
    [[nodiscard]] MtpHeadResult advance(GpuView target_hc,
                                       std::span<const float> next_token_embedding,
                                       uint32_t absolute_position);
    // Cache-only teacher forcing: future rows use fresh target HC, so the
    // historical attention output/FFN/mixer/vocabulary projection is unused.
    // The exact same EH/HC-read/K/V path is encoded as advance. No logits or
    // head HC are available after this call.
    [[nodiscard]] BatchResult prime_pair(std::span<const float> target_hc,
                                        std::span<const float> next_token_embedding,
                                        uint32_t absolute_position);
    [[nodiscard]] BatchResult prime_pair(GpuView target_hc,
                                        std::span<const float> next_token_embedding,
                                        uint32_t absolute_position);
    // Explicit chronological reset/rebuild over caller-owned flattened pairs.
    void prime(std::span<const float> target_hc_rows, std::span<const float> embedding_rows,
               uint32_t first_absolute_position = 1);
#ifdef TOKSPUR_PRIVATE_MTP_PRIMING_BATCH
    // Explicit startup-only allocation. Appending preserves the existing cache and absolute frontier, stages all rows before encoding, and submits one prefix-only head batch.
    void prepare_priming_batch(uint32_t capacity, uint64_t gpu_scratch_budget_bytes);
    [[nodiscard]] BatchResult prime_batch(std::span<const float> target_hc_rows,
        std::span<const float> embedding_rows, uint32_t first_absolute_position);
    [[nodiscard]] BatchResult prime_batch(GpuView target_hc_rows,
        std::span<const float> embedding_rows, uint32_t first_absolute_position);
    [[nodiscard]] BatchResult prime_batch_with_prefix(std::span<const float> first_hc_row,
        GpuView tail_hc_rows, std::span<const float> embedding_rows, uint32_t first_absolute_position);
    [[nodiscard]] uint32_t priming_batch_capacity() const;
    [[nodiscard]] uint64_t priming_batch_bytes() const;
    // Idle caller-buffered diagnostic. Keys then values, both complete published BF16 prefixes. No allocation, reset, entropy or model work.
    [[nodiscard]] uint64_t priming_cache_prefix_bytes() const;
    void read_priming_cache_prefix(std::span<std::byte> output) const;
#endif
    [[nodiscard]] std::span<const float> logits() const;
    [[nodiscard]] std::span<const float> hidden() const; // Last head HC, diagnostic/deeper-step input.
    [[nodiscard]] GpuView hidden_gpu() const;
    [[nodiscard]] uint64_t workspace_bytes() const;
    [[nodiscard]] MtpDraftAttention draft_attention() const;
    void configure_draft_vocabulary(uint32_t draft_vocab, GpuMatrix draft_output);
    [[nodiscard]] uint32_t draft_vocabulary() const noexcept;
    [[nodiscard]] bool has_draft_slice() const noexcept;
    void configure_fast_round(bool enabled, GpuMatrix token_embedding = {});
    [[nodiscard]] bool fast_round() const noexcept;
    [[nodiscard]] bool can_greedy_chain() const noexcept;
    [[nodiscard]] MtpHeadFastResult advance_fast(GpuView target_hc_gpu,
                                                 std::span<const float> next_token_embedding,
                                                 uint32_t absolute_position);
    [[nodiscard]] MtpHeadFastResult advance_fast_host(std::span<const float> target_hc,
                                                      std::span<const float> next_token_embedding,
                                                      uint32_t absolute_position);
    [[nodiscard]] MtpHeadFastResult advance_fast_chain(std::span<const float> next_token_embedding,
                                                       uint32_t absolute_position);
    [[nodiscard]] MtpHeadFastChainResult advance_fast_greedy_chain(GpuView target_hc_gpu,
                                                                   std::span<const float> anchor_embedding,
                                                                   uint32_t first_position,
                                                                   uint32_t depth);
    [[nodiscard]] MtpHeadFastChainResult advance_fast_greedy_chain_host(std::span<const float> target_hc,
                                                                        std::span<const float> anchor_embedding,
                                                                        uint32_t first_position,
                                                                        uint32_t depth);
    [[nodiscard]] MtpHeadSampledChainResult advance_fast_sampled_chain(GpuView target_hc_gpu,
                                                                       std::span<const float> anchor_embedding,
                                                                       uint32_t first_position,
                                                                       uint32_t depth,
                                                                       SamplingPolicy draft_policy,
                                                                       std::span<const uint64_t, 3> tickets);
    [[nodiscard]] MtpHeadSampledChainResult advance_fast_sampled_chain_host(std::span<const float> target_hc,
                                                                            std::span<const float> anchor_embedding,
                                                                            uint32_t first_position,
                                                                            uint32_t depth,
                                                                            SamplingPolicy draft_policy,
                                                                            std::span<const uint64_t, 3> tickets);
    void defer_prime_batch(GpuView target_hc_gpu, std::span<const float> embedding_rows,
                           uint32_t first_absolute_position);
    void defer_prime_batch_host(std::span<const float> target_hc_rows, std::span<const float> embedding_rows,
                                uint32_t first_absolute_position);
    [[nodiscard]] BatchResult flush_deferred_prime();
    [[nodiscard]] uint32_t deferred_prime_rows() const noexcept;
    [[nodiscard]] uint32_t first_absolute_position() const noexcept;
    [[nodiscard]] uint64_t ssd_cache_bytes(uint32_t cache_rows) const;
    [[nodiscard]] uint64_t ssd_cache_bytes() const;
    [[nodiscard]] bool read_ssd_cache_span(uint64_t offset, std::span<std::byte> target) const;
    [[nodiscard]] bool write_ssd_cache_span(uint32_t first_pos, uint32_t cache_rows, uint64_t offset, std::span<const std::byte> source);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tokspur
