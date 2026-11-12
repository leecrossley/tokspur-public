#pragma once
// Lossless planar weight pack v3 (tokspur-pack-v3).
// Eliminates GGUF scale bit-twiddling and AoS unaligned strides while preserving
// 100 per cent identical FP32 dequantised operands and byte-exact GGUF reconstruction
// with zero sidecar footer bytes.

#include "tokspur/derived_pack.hpp"
#include "tokspur/gguf.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tokspur::planar_v3 {

inline constexpr uint32_t pack_version_v3 = 3;
inline constexpr const char* pack_recipe_v3 = "tokspur-pack-v3";
inline constexpr const char* block_recipe_prefix_v3 = "tokspur-pack-v3-block-";
inline constexpr uint64_t expected_pack_v3_payload_bytes = 85'881'974'272ULL;
inline constexpr uint64_t max_streaming_scratch_bytes = 96ULL * 1024 * 1024; // 96 MiB (well within 4.00 GB ceiling)

enum class PlanarKind : uint32_t {
    plain = 0,
    q8_lanes_v1 = 1,
    q4k_coloc_gate_v3 = 2,
    q4k_coloc_up_v3 = 3,
    q5k_coloc_gate_v3 = 4,
    q5k_coloc_up_v3 = 5,
    q8_coloc_gate_v3 = 6,
    q8_coloc_up_v3 = 7,
    q51_planar_v3 = 8,
    q8_planar_v3 = 9,
};

[[nodiscard]] const char* planar_kind_name(PlanarKind kind);
[[nodiscard]] PlanarKind parse_planar_kind(std::string_view name);
[[nodiscard]] bool is_planar_v3_kind(PlanarKind kind) noexcept;
[[nodiscard]] bool is_coloc_gate_kind(PlanarKind kind) noexcept;
[[nodiscard]] bool is_coloc_up_kind(PlanarKind kind) noexcept;

struct PlanarExpertSpec {
    uint32_t source_type = 0; // 7 (Q5_1), 8 (Q8_0), 12 (Q4_K), 13 (Q5_K)
    uint32_t columns = 0;
    uint32_t rows = 0;
    uint32_t blocks_per_row = 0;
    uint64_t source_row_bytes = 0;
    uint64_t planar_row_bytes = 0;
    uint64_t source_expert_bytes = 0;
    uint64_t planar_expert_bytes = 0;
    uint64_t qs_offset = 0;
    uint64_t qh_offset = 0;
    uint64_t sm_offset = 0;
    uint64_t dm_offset = 0;
};

[[nodiscard]] PlanarExpertSpec make_expert_spec(uint32_t source_type, uint32_t columns, uint32_t rows);

// Bijective 12-byte GGUF Q4_K/Q5_K scales[12] <-> 16-byte {uint8 sc[8], uint8 mn[8]} in [0, 63].
void pack_sm16(std::span<const std::byte, 12> g_scales, std::span<std::byte, 16> out_sm) noexcept;
void unpack_sm16(std::span<const std::byte, 16> in_sm, std::span<std::byte, 12> out_scales);

// Pack / unpack one complete expert matrix ([columns, rows]) between GGUF AoS and planar v3 SoA.
void pack_expert_planar_v3(const PlanarExpertSpec& spec, std::span<const std::byte> gguf_expert, std::span<std::byte> planar_expert);
void unpack_expert_planar_v3(const PlanarExpertSpec& spec, std::span<const std::byte> planar_expert, std::span<std::byte> gguf_expert);

// Exact CPU dequantisation of one planar v3 expert into FP32 (matching tokspur::dequantize and GPU FP32 decode).
void dequantize_expert_planar_v3(const PlanarExpertSpec& spec, std::span<const std::byte> planar_expert, std::span<float> output);
void dequantize_expert_planar_v3_f32(const PlanarExpertSpec& spec, std::span<const std::byte> planar_expert, std::span<float> output);

struct PackTensorV3 {
    std::string name;
    uint32_t type = 0;               // GGUF directory type (24 = I8 when bytes != source_bytes, else source_type)
    uint32_t source_type = 0;        // Original GGUF encoding
    PlanarKind kind = PlanarKind::plain;
    std::string layout = "plain";
    std::vector<uint64_t> dimensions;        // GGUF directory dimensions
    std::vector<uint64_t> source_dimensions; // Original GGUF dimensions
    uint64_t source_bytes = 0;
    uint64_t bytes = 0;
    uint64_t offset = 0; // Relative to data_offset, 16 KiB aligned
    std::string partner_name; // Non-empty for co-located gate/up pairs
    std::string source_sha256;
    std::string packed_sha256;
};

struct PackPlanV3 {
    std::vector<PackTensorV3> tensors;
    uint64_t source_payload_bytes = 0;
    uint64_t payload_bytes = 0;
    uint64_t file_bytes = 0;
    uint64_t data_offset = 0;
    uint64_t q4k_superblocks = 0;
    uint64_t q5k_superblocks = 0;
    uint64_t original_12b_scale_header_bytes = 0;
    uint64_t expanded_16b_scale_header_bytes = 0;
    std::string recipe = pack_recipe_v3;
    uint32_t version = pack_version_v3;
    uint32_t alignment = static_cast<uint32_t>(derived_pack::pack_alignment);
};

struct BudgetReportV3 {
    uint64_t filesystem_available_bytes = 0;
    uint64_t existing_pack_v2_file_bytes = 0;
    uint64_t planned_pack_v3_payload_bytes = 0;
    uint64_t planned_pack_v3_file_bytes = 0;
    uint64_t free_after_pack_v3_bytes = 0;
    uint64_t required_reserve_bytes = derived_pack::default_reserve_bytes;
    uint64_t streaming_ring_buffer_bytes = 0;
    uint64_t streaming_ram_ceiling_bytes = 4'000'000'000ULL;
    uint64_t q4k_planar_bytes = 0;
    uint64_t q5k_planar_bytes = 0;
    uint64_t q51_planar_bytes = 0;
    uint64_t q8_planar_moe_bytes = 0;
    uint64_t original_12b_scale_header_bytes = 0;
    uint64_t expanded_16b_scale_header_bytes = 0;
    bool reserve_satisfied = false;
    bool ram_ceiling_satisfied = false;
};

struct AdoptedPackV3 {
    std::shared_ptr<const ModelFiles> model;
    PackPlanV3 plan;
    std::string manifest_sha256;
    uint32_t authentication_workers = 1;
    // Describes completed source reads, not pages released by the OS.
    std::string_view authentication_source_read_mode = "not_read";
};

[[nodiscard]] std::vector<std::byte> build_header_v3(const PackPlanV3& plan);

// Full 1,254-tensor tokspur-pack-v3 planner and disk/RAM budget inspector.
[[nodiscard]] PackPlanV3 plan_derived_pack_v3(
    const ModelFiles& source,
    const ModelFiles* mtp_sidecar = nullptr,
    uint64_t byte_budget = derived_pack::max_pack_bytes
);

[[nodiscard]] BudgetReportV3 measure_pack_v3_budget(
    const ModelFiles& source,
    const ModelFiles* mtp_sidecar,
    const std::filesystem::path& output_directory,
    const std::filesystem::path& existing_pack_v2_path = {},
    uint32_t experts_per_chunk = 8,
    uint64_t reserve_bytes = derived_pack::default_reserve_bytes
);

[[nodiscard]] BudgetReportV3 measure_pack_v3_budget(
    const PackPlanV3& plan, const std::filesystem::path& output_directory,
    const std::filesystem::path& existing_pack_v2_path, uint32_t experts_per_chunk,
    uint64_t reserve_bytes, uint64_t chunk_bytes
);

struct PublishedPackV3 {
    AdoptedPackV3 verified;
    bool existed;
    double build_seconds;
};

// Fully authenticate against the supplied sources before publishing or accepting a seal.
[[nodiscard]] PublishedPackV3 build_or_verify_pack_v3(
    PackPlanV3 plan, std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar, const std::filesystem::path& output,
    uint32_t experts_per_chunk, uint64_t reserve_bytes, uint64_t chunk_bytes,
    void (*check_cancelled)() = nullptr
);

void validate_pack_v3_output_name(const std::filesystem::path& output);
void require_full_pack_v3_census(const PackPlanV3& plan);
int pack_v3_cli(int argc, char** argv);

// Single-block fixture planner (e.g. block_id in {0, 2, 3, 48}) for sequential D1 prototype.
[[nodiscard]] PackPlanV3 plan_block_pack_v3(
    const ModelFiles& source,
    const ModelFiles* mtp_sidecar,
    uint32_t block_id,
    uint64_t byte_budget = 4'000'000'000ULL
);

[[nodiscard]] PackPlanV3 build_pack_v3_from_plan(
    PackPlanV3 plan,
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint32_t experts_per_chunk = 8,
    uint64_t reserve_bytes = derived_pack::default_reserve_bytes,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes,
    void (*check_cancelled)() = nullptr
);

[[nodiscard]] PackPlanV3 build_derived_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint32_t experts_per_chunk = 8,
    uint64_t reserve_bytes = derived_pack::default_reserve_bytes,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes
);

[[nodiscard]] PackPlanV3 build_block_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    uint32_t block_id,
    const std::filesystem::path& output,
    uint32_t experts_per_chunk = 8,
    uint64_t reserve_bytes = derived_pack::default_reserve_bytes,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes
);

[[nodiscard]] AdoptedPackV3 adopt_pack_v3_with_plan(
    PackPlanV3 expected_plan,
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& path,
    uint32_t experts_per_chunk = 8,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes,
    uint32_t authentication_workers = 1,
    derived_pack::TrustMode trust_mode = derived_pack::TrustMode::full,
    bool require_full_census = true,
    void (*check_cancelled)() = nullptr
);

[[nodiscard]] AdoptedPackV3 adopt_derived_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& path,
    uint32_t experts_per_chunk = 8,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes,
    uint32_t authentication_workers = 1,
    derived_pack::TrustMode trust_mode = derived_pack::TrustMode::full,
    bool require_full_census = true
);

[[nodiscard]] AdoptedPackV3 adopt_block_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    uint32_t block_id,
    const std::filesystem::path& path,
    uint32_t experts_per_chunk = 8,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes,
    uint32_t authentication_workers = 1
);

} // namespace tokspur::planar_v3
