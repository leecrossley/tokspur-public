#pragma once
// Private, Darwin-only lossless derived weight pack (tokspur-pack-v2).
// Implements 16 KiB page-aligned Metal mapping, byte-exact inverse reconstruction proof,
// 64-bit uint64_t addressing, SHA-256 integrity, and disk reserve verification.

#include "tokspur/gguf.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tokspur::derived_pack {

inline constexpr uint64_t pack_alignment = 16384; // 16 KiB page boundary for Metal newBufferWithBytesNoCopy
inline constexpr uint32_t pack_version = 2;
inline constexpr const char* pack_recipe = "tokspur-pack-v2";
inline constexpr const char* pack_architecture = "tokspur_pack";
inline constexpr const char* source_architecture = "qwen4exp";
inline constexpr const char* target_revision = "766911a6b7369840a91dbcd95f9f997acaab6cd6";
inline constexpr const char* mtp_sidecar_sha256 = "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6";
inline constexpr size_t expected_gpu_tensors = 1254;

inline constexpr uint64_t default_reserve_bytes = 50ULL * 1024 * 1024 * 1024; // 50 GiB safety reserve
inline constexpr uint64_t default_chunk_bytes = 4'194'304; // 4 MiB streaming chunk
inline constexpr uint32_t max_authentication_workers = 16;
inline constexpr uint64_t max_pack_bytes = 128'000'000'000ULL; // 128 GB safety ceiling

struct PackTensor {
    std::string name;
    uint32_t type = 0;
    std::string layout = "plain";
    std::vector<uint64_t> dimensions;
    uint64_t bytes = 0;
    uint64_t offset = 0; // Aligned relative to data_offset
    std::string source_sha256;
    std::string packed_sha256;
};

struct PackPlan {
    std::vector<PackTensor> tensors;
    uint64_t payload_bytes = 0;
    uint64_t file_bytes = 0;
    uint64_t data_offset = 0;
    std::string recipe = pack_recipe;
    uint32_t version = pack_version;
    uint32_t alignment = pack_alignment;
};

using Bytes = std::vector<std::byte>;

// Compute SHA-256 hex digest of text
[[nodiscard]] std::string hash(std::string_view text);

// Build GGUF header bytes for pack plan
[[nodiscard]] Bytes build_header(const PackPlan& plan);

struct AdoptedPack {
    std::shared_ptr<const ModelFiles> model;
    PackPlan plan;
    std::string manifest_sha256;
    uint32_t authentication_workers = 1;
};

// Check if a tensor name is a CPU-pinned embedding table (retained on host CPU)
[[nodiscard]] inline constexpr bool is_cpu_pinned_embedding(std::string_view name) noexcept {
    return name == "token_embd.weight" || name == "per_layer_token_embd.weight" || name == "per_layer_token_embd";
}

// Check if a tensor name is eligible for GPU packing (excludes CPU embedding tables)
[[nodiscard]] bool is_gpu_pack_tensor(std::string_view name);

// Verify census-driven GPU prototype checks across real GDN, layer-3 QSA, layer-1 ple_*, global output HC, output.weight, and blk.48 MTP
[[nodiscard]] bool verify_gpu_pack_census(const PackPlan& plan);

// Plan derived pack from source model files and optional MTP sidecar
[[nodiscard]] PackPlan plan_derived_pack(
    const ModelFiles& source,
    const ModelFiles* mtp_sidecar = nullptr,
    uint64_t byte_budget = max_pack_bytes
);

// Build derived pack with byte-exact inverse reconstruction proof across all tensors
[[nodiscard]] PackPlan build_derived_pack(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint64_t reserve_bytes = default_reserve_bytes,
    uint64_t chunk_bytes = default_chunk_bytes
);

// Build derived pack using an explicit directory free-space query callback
[[nodiscard]] PackPlan build_derived_pack_with_space_query(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint64_t reserve_bytes,
    uint64_t chunk_bytes,
    const std::function<uint64_t(int)>& query_available_bytes
);

enum class TrustMode { full, local_seal };

// Adopt derived pack with full SHA-256 and inverse reconstruction verification,
// or local-seal fast-start validation (< 1.5 s) with fallback to full on missing/stale seal.
[[nodiscard]] AdoptedPack adopt_derived_pack(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& path,
    uint64_t chunk_bytes = default_chunk_bytes,
    uint32_t authentication_workers = 1,
    TrustMode trust_mode = TrustMode::full,
    bool require_full_census = true
);

// Check disk budget given available space, peak allocation, and required reserve
[[nodiscard]] bool check_disk_reserve(
    uint64_t available_bytes,
    uint64_t peak_additional_bytes,
    uint64_t reserve_bytes = default_reserve_bytes
);

} // namespace tokspur::derived_pack

namespace tokspur {
using derived_pack::is_cpu_pinned_embedding;
}

