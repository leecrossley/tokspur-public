#pragma once

#include "tokspur/graph.hpp"
#include "tokspur/sampled_path.hpp"
#include "tokspur/speculative_controls.hpp"

#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace tokspur {

struct DFlash2DraftResult {
    std::array<SampledProposalRow, 7> proposals{};
    std::array<float, 7> confidences{};
    uint32_t count = 0;
    uint32_t evaluated_proposals = 0;
    uint64_t proposal_begin_words = 0;
    SamplingRng random_after{0};
    BatchResult batch{};
};

// Adapted from incoai/splash 1.4.0 runtime/model/VerifyBudget.hpp (Apache-2.0).
class DFlash2VerifyBudget {
public:
    static constexpr uint32_t kPositions = 7;
    static constexpr uint32_t kBlockRows = 8;
    static constexpr uint32_t kLabels = 4096;
    static constexpr uint32_t kRefitCycles = 64;
    static constexpr uint32_t kExploreCycles = 32;
    static constexpr double kDecay = 0.97;
    static constexpr double kPriorCycles = 24.0;
    static constexpr double kSlopePrecision = 50.0;
    static constexpr double kInterceptPrecision = 4.0;

    struct Acceptance {
        float slope = 0.7f;
        std::array<float, kPositions> intercepts{0.0f, -0.35f, -0.8f, -0.75f, -0.75f, -0.85f, -0.45f};
    };

    struct Label {
        uint32_t position = 0;
        float feature = 0.0f;
        bool accepted = false;
    };

    static double seed_cost(uint32_t rows);
    static Acceptance fit(std::span<const Label> labels, const Acceptance& start);

    [[nodiscard]] const Acceptance& acceptance() const noexcept { return acceptance_; }
    [[nodiscard]] uint64_t cycles() const noexcept { return cycles_; }
    [[nodiscard]] size_t label_count() const noexcept { return labels_.size(); }
    [[nodiscard]] std::vector<Label> labels() const;
    [[nodiscard]] bool explores() const noexcept { return (cycles_ % kExploreCycles) == (kExploreCycles - 1u); }
    [[nodiscard]] std::array<float, kBlockRows + 1> costs() const;
    [[nodiscard]] uint32_t choose_verify_rows(std::span<const float, kPositions> confidences, uint32_t max_rows) const;

    void observe(std::span<const float, kPositions> confidences, uint32_t verify_rows, uint32_t accepted,
                 double milliseconds, bool record_labels = true);

private:
    struct CostStats {
        double weight = 0.0;
        double rows = 0.0;
        double rows_squared = 0.0;
        double time = 0.0;
        double rows_time = 0.0;
    };

    [[nodiscard]] std::optional<std::pair<double, double>> fit_cost_line() const noexcept;

    Acceptance acceptance_{};
    CostStats cost_{};
    std::vector<Label> labels_{};
    size_t next_label_ = 0;
    uint64_t cycles_ = 0;
};

// The caller owns the fixed scratch allocation. Pending rows never borrow the target's reusable tap storage.
class DFlash2ContextStage {
public:
    void prepare(GpuView storage, uint32_t width);
    void stage(MetalGraph& gpu, GpuView source, uint32_t rows, uint32_t start);
    void truncate(uint32_t position) noexcept;
    void clear() noexcept { rows_ = start_ = 0; }
    [[nodiscard]] GpuView storage() const noexcept { return storage_; }
    [[nodiscard]] uint32_t rows() const noexcept { return rows_; }
    [[nodiscard]] uint32_t start() const noexcept { return start_; }
private:
    GpuView storage_{};
    uint32_t width_ = 0, rows_ = 0, start_ = 0;
};

class DFlash2Drafter {
public:
    using EmbedFn = std::function<void(uint32_t token_id, std::span<float> destination)>;

    DFlash2Drafter(MetalGraph& gpu, const std::filesystem::path& directory, GpuView borrowed_output_weight,
                   uint32_t known_token_capacity, const EmbedFn& embed_token,
                   DFlash2Precision precision = DFlash2Precision::q8_0, bool defer_context = false,
                   bool bind_checkpoint_identity = true);
    ~DFlash2Drafter();

    DFlash2Drafter(const DFlash2Drafter&) = delete;
    DFlash2Drafter& operator=(const DFlash2Drafter&) = delete;

    void reset() noexcept;
    [[nodiscard]] uint32_t first_absolute_position() const noexcept { return context_len_ - cache_rows(); }
    [[nodiscard]] uint32_t cache_rows() const noexcept { return std::min(context_len_, sliding_window_); }
    [[nodiscard]] uint64_t ssd_cache_bytes(uint32_t rows) const;
    [[nodiscard]] uint64_t ssd_cache_bytes() const { return ssd_cache_bytes(cache_rows()); }
    bool read_ssd_cache_span(uint64_t offset, std::span<std::byte> target);
    bool write_ssd_cache_span(uint32_t first, uint32_t rows, uint64_t offset, std::span<const std::byte> source);
    void truncate(uint32_t position);
    void set_max_drafts(uint32_t max_drafts);
    [[nodiscard]] uint32_t max_drafts() const noexcept { return max_drafts_; }
    void configure_draft_vocab_slice(uint32_t slice_vocab);
    [[nodiscard]] uint32_t draft_vocab_slice() const noexcept { return active_output_vocab_ < vocab_size_ ? active_output_vocab_ : 0u; }
    [[nodiscard]] uint32_t active_output_vocabulary() const noexcept { return active_output_vocab_; }

    void stage_context_commit(GpuView captured_target_hidden, uint32_t rows, uint32_t start_position);
    BatchResult flush_context();
    BatchResult commit_context(GpuView captured_target_hidden, uint32_t rows, uint32_t start_position,
                               uint64_t remaining_after = 0);

    DFlash2DraftResult draft_block8(uint32_t k, uint32_t base_position, uint32_t anchor_id,
                                    std::span<const float> anchor_row, SamplingPolicy policy,
                                    const SamplingRng& after_anchor, bool adaptive_budget = true);

    void observe_verify_round(std::span<const float, DFlash2VerifyBudget::kPositions> confidences,
                              uint32_t verify_rows, uint32_t accepted, double milliseconds,
                              bool record_labels = true);

    [[nodiscard]] const DFlash2VerifyBudget& verify_budget() const noexcept { return verify_budget_; }
    void reset_verify_budget() { verify_budget_ = DFlash2VerifyBudget{}; }
    using MeasuredBudget = std::function<uint32_t(uint32_t, std::span<const float, 7>, uint32_t)>;
    void set_measured_budget(MeasuredBudget callback) { measured_budget_ = std::move(callback); }
    void set_adaptive_budget(bool enabled) noexcept { adaptive_budget_ = enabled; }
    [[nodiscard]] bool adaptive_budget() const noexcept { return adaptive_budget_; }

    [[nodiscard]] DFlash2Precision precision() const noexcept { return precision_; }
    [[nodiscard]] bool deferred_context() const noexcept { return defer_context_; }
    [[nodiscard]] std::span<const uint32_t> target_layer_ids() const noexcept { return target_layer_ids_; }
    [[nodiscard]] uint32_t context_len() const noexcept { return context_len_; }
    [[nodiscard]] uint32_t mask_token_id() const noexcept { return mask_token_id_; }
    [[nodiscard]] uint32_t vocabulary() const noexcept { return vocab_size_; }
    [[nodiscard]] uint64_t weight_bytes() const noexcept { return weight_bytes_; }
    [[nodiscard]] uint64_t workspace_bytes() const noexcept { return workspace_bytes_; }
    [[nodiscard]] uint32_t priming_batch_capacity() const noexcept { return priming_cap_; }
    [[nodiscard]] uint64_t priming_batch_bytes() const noexcept { return priming_batch_bytes_; }
    [[nodiscard]] const std::string& checkpoint_sha256() const noexcept { return checkpoint_sha256_; }
    [[nodiscard]] const std::string& state_identity_sha256() const noexcept { return state_identity_sha256_; }

private:
    void dense_proj(GpuView matrix, GpuRowsF32 in, GpuRowsF32 out, bool swiglu = false);
    void encode_context_commit_chunk(GpuView captured_target_hidden, uint32_t rows, uint32_t start_position);

    DFlash2Precision precision_ = DFlash2Precision::q8_0;
    bool defer_context_ = false;
    DFlash2ContextStage pending_context_;
    struct LayerGpu {
        GpuView input_norm{};
        GpuView attn_conv_base{};
        GpuView attn_conv_proj{};
        GpuView qkv_proj{};
        GpuView kv_proj{};
        GpuView qk_norm_freq{};
        GpuView o_proj{};
        GpuView post_attn_norm{};
        GpuView mlp_conv_base{};
        GpuView mlp_conv_proj{};
        GpuView gate_proj{};
        GpuView up_proj{};
        GpuView down_proj{};
        GpuView ring_k{};
        GpuView ring_v{};
    };

    MetalGraph& gpu_;
    GpuView borrowed_output_weight_{};
    GpuView active_output_weight_{};
    uint32_t active_output_vocab_ = 248320;
    uint32_t max_drafts_ = 7;

    uint32_t hidden_size_ = 2560;
    uint32_t intermediate_size_ = 7680;
    uint32_t num_layers_ = 5;
    uint32_t q_heads_ = 32;
    uint32_t kv_heads_ = 8;
    uint32_t head_dim_ = 128;
    uint32_t sliding_window_ = 2048;
    uint32_t vocab_size_ = 248320;
    uint32_t mask_token_id_ = 248077;
    uint32_t selector_rank_ = 256;
    uint32_t selector_top_k_ = 16;
    uint32_t priming_cap_ = 2048;
    float rms_eps_ = 1e-6f;
    float rope_theta_ = 10000.0f;

    std::vector<uint32_t> target_layer_ids_;
    std::vector<float> mask_embedding_;
    std::vector<float> host_input_8_;
    std::vector<std::byte> host_selector_out_;
    std::string checkpoint_sha256_;
    std::string state_identity_sha256_;

    GpuView fc_proj_{};
    GpuView hidden_norm_{};
    GpuView final_norm_{};
    GpuView selector_proj_{};
    GpuView predecessor_codebook_{};
    GpuView successor_codebook_{};
    std::vector<LayerGpu> layers_;

    GpuView prime_projected_{};
    GpuView prime_hidden_{};
    GpuView prime_kv_{};

    std::array<GpuView, 2> draft_hidden_{};
    GpuView draft_norm_{};
    GpuView draft_dynamic_{};
    GpuView draft_conv_{};
    GpuView draft_qkv_{};
    GpuView draft_queries_{};
    GpuView draft_query_keys_{};
    GpuView draft_query_values_{};
    GpuView draft_partials_{};
    GpuView draft_attn_out_{};
    GpuView draft_proj_{};
    GpuView draft_residual_{};
    GpuView draft_inter_{};
    GpuView draft_final_{};
    GpuView draft_logits_7_{};
    GpuView draft_selector_hidden_7_{};
    GpuView draft_top_ids_{};
    GpuView draft_top_vals_{};
    GpuView draft_selector_out_{};

    std::vector<GpuView> owned_views_;
    uint64_t weight_bytes_ = 0;
    uint64_t workspace_bytes_ = 0;
    uint64_t priming_batch_bytes_ = 0;
    uint32_t context_len_ = 0;
    bool adaptive_budget_ = false;
    DFlash2VerifyBudget verify_budget_{};
    MeasuredBudget measured_budget_;
};

} // namespace tokspur
