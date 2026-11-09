#pragma once

#include "tokspur/mtp_anchor_state.hpp"
#include "tokspur/sampling.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace tokspur {

constexpr uint64_t default_checkpoint_ssd_budget = 8ull << 30;  // 8.0 GiB
constexpr uint64_t maximum_checkpoint_ssd_budget = 32ull << 30; // 32.0 GiB
constexpr uint64_t apfs_free_space_floor = 20ull << 30;          // 20.0 GiB
constexpr size_t staging_slab_bytes = 64ull << 20;               // 64 MiB
constexpr size_t checkpoint_chunk_bytes = 16ull << 20;           // 16 MiB
constexpr uint32_t checkpoint_format_version = 3;
constexpr double lambda_recompute_prefill = 1.0e7;               // 1.0e7 bytes / second

#pragma pack(push, 1)
struct CheckpointFileHeader {
    std::array<char, 8> magic{'T', 'O', 'K', 'C', 'K', 'P', 'T', '1'};
    uint32_t version = checkpoint_format_version;
    uint32_t prefix_tokens = 0;
    uint64_t total_payload_bytes = 0;
    uint32_t chunk_size = uint32_t(checkpoint_chunk_bytes);
    uint32_t chunk_count = 0;
    uint64_t access_epoch = 0;
    uint8_t is_system_anchor = 0;
    uint8_t reserved[7]{0};
    std::array<uint8_t, 32> state_digest{};
    std::array<uint8_t, 32> controller_policy_digest{};
    std::array<uint8_t, 32> model_execution_digest{};
    std::array<uint8_t, 32> metadata_sha256{};
    std::array<uint8_t, 32> header_sha256{};
};
#pragma pack(pop)

struct CheckpointMetadata {
    SamplingRng rng{};
    StageAControllerState controller_state{};
    uint32_t mtp_first = 1;
    uint32_t mtp_rows = 0;
    std::optional<SampledPathToken> carried_anchor{};
    uint32_t pending_overflow = 0;
    std::vector<float> logits;
};

struct SlotHandshake {
    std::atomic<uint64_t> reader_epoch{0};
    std::atomic<bool> cancel_spill{false};
};

// Calibrated prefill recompute time formula:
// T_hat_prefill(N) = 1.8e-4 * N + 4.2e-8 * N^2 seconds
inline double calibrated_prefill_time(uint32_t prefix_tokens) noexcept {
    const double n = double(prefix_tokens);
    return 1.8e-4 * n + 4.2e-8 * n * n;
}

inline double compute_eviction_priority(uint64_t last_access_epoch, uint32_t prefix_tokens,
                                        uint64_t file_bytes, bool is_system_anchor) noexcept {
    const double t_hat = calibrated_prefill_time(prefix_tokens);
    const double bytes = file_bytes > 0 ? double(file_bytes) : 1.0;
    return double(last_access_epoch) + lambda_recompute_prefill * (t_hat / bytes) * (is_system_anchor ? 2.0 : 1.0);
}

class SsdCheckpointManager {
public:
    using FreeSpaceQuery = std::function<bool(int dir_fd, uint64_t& out_free_bytes)>;

    SsdCheckpointManager(const std::filesystem::path& state_dir,
                         uint64_t budget_bytes = default_checkpoint_ssd_budget,
                         uint64_t floor_bytes = apfs_free_space_floor,
                         std::string eviction_policy = "recompute-cost");
    ~SsdCheckpointManager();
    SsdCheckpointManager(const SsdCheckpointManager&) = delete;
    SsdCheckpointManager& operator=(const SsdCheckpointManager&) = delete;

    [[nodiscard]] bool enabled() const noexcept;
    [[nodiscard]] uint64_t budget_bytes() const noexcept;
    [[nodiscard]] uint64_t occupied_bytes() const noexcept;
    [[nodiscard]] uint64_t floor_bytes() const noexcept;
    [[nodiscard]] int dir_fd() const noexcept;
    [[nodiscard]] const std::string& eviction_policy() const noexcept;
    [[nodiscard]] bool has_checkpoint(const std::string& filename) const noexcept;
    void set_execution_identity(const std::array<uint8_t, 32>& digest) noexcept;
    [[nodiscard]] const std::array<uint8_t, 32>& execution_identity() const noexcept;
    void set_free_space_query(FreeSpaceQuery query);

    // Checks APFS free space floor (statvfs) and budget limits
    [[nodiscard]] bool admit_spill(uint64_t incoming_bytes);

    // Eviction management
    void run_eviction_if_needed(uint64_t incoming_bytes);

    // Staging slab buffer (64 MiB page-aligned)
    [[nodiscard]] std::span<std::byte> staging_slab() noexcept;

    // Coalescing check (Feature 71: >= 5.0 s or >= 256 committed tokens)
    [[nodiscard]] bool should_coalesce(double elapsed_seconds, uint32_t newly_committed_tokens) const noexcept;

    // Save checkpoint
    bool spill_checkpoint(const std::string& filename,
                          SlotHandshake& handshake,
                          CheckpointFileHeader header,
                          const CheckpointMetadata& metadata,
                          const std::function<bool(size_t chunk_idx, std::span<std::byte> target)>& read_chunk_fn);

    // Restore checkpoint
    bool restore_checkpoint(const std::string& filename,
                            CheckpointFileHeader& out_header,
                            CheckpointMetadata& out_metadata,
                            const std::function<bool(size_t chunk_idx, std::span<const std::byte> source)>& write_chunk_fn,
                            const std::function<bool(const CheckpointFileHeader&, const CheckpointMetadata&)>& validate_fn = {});

    // Cleanly unlinks a checkpoint
    bool remove_checkpoint(const std::string& filename);

    // Lists all valid checkpoints with their eviction priorities
    struct EntryInfo {
        std::string filename;
        uint64_t file_bytes = 0;
        uint32_t prefix_tokens = 0;
        uint64_t access_epoch = 0;
        bool is_system_anchor = false;
        double priority = 0.0;
        uint64_t device = 0;
        uint64_t inode = 0;
        timespec birth_time{};
    };
    [[nodiscard]] std::vector<EntryInfo> list_entries() const;

private:
    int dir_fd_ = -1;
    std::filesystem::path checkpoint_dir_;
    uint64_t budget_bytes_ = default_checkpoint_ssd_budget;
    uint64_t floor_bytes_ = apfs_free_space_floor;
    std::string eviction_policy_ = "recompute-cost";
    std::byte* staging_slab_ptr_ = nullptr;
    std::atomic<uint64_t> global_access_epoch_{1};
    mutable uint64_t cached_occupied_bytes_ = 0;
    std::array<uint8_t, 32> execution_identity_digest_{};
    FreeSpaceQuery free_space_query_{};

    void open_secure_directory();
    [[nodiscard]] bool reconcile_directory(bool initialise_epoch = false) noexcept;
    [[nodiscard]] bool take_access_epoch(uint64_t& value) noexcept;
    [[nodiscard]] bool query_available_space(uint64_t& out_free_bytes) const;
    [[nodiscard]] std::vector<EntryInfo> scan_entries(bool include_incompatible) const;
    [[nodiscard]] uint64_t evict_for_headroom(uint64_t incoming_bytes, uint64_t initial_free_bytes);
};

} // namespace tokspur
