#pragma once
// Private, Darwin-only lossless lane sidecar. No model or inference is started.
#include "tokspur/gguf.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace tokspur::lanes {

inline constexpr uint64_t sidecar_max_bytes = 32'000'000'000;
inline constexpr uint64_t sidecar_default_chunk_bytes = 4'194'304;
inline constexpr uint32_t sidecar_max_authentication_workers = 16;
inline constexpr uint32_t sidecar_version = 1;
inline constexpr const char* sidecar_recipe = "q51-q8-lanes-v1";

struct SidecarTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> dimensions;
    uint64_t bytes = 0;
    std::string source_sha256, packed_sha256;
};
struct SidecarPlan {
    std::vector<SidecarTensor> tensors;
    uint64_t payload_bytes = 0, file_bytes = 0;
};
struct AdoptedSidecar {
    std::shared_ptr<const ModelFiles> model;
    SidecarPlan plan;
    std::string manifest_sha256;
    uint32_t authentication_workers = 1;
};

// Empty names select all eligible tensors. Explicit names must be unique and eligible.
// Planning inspects metadata only. It neither reads payloads nor creates files.
[[nodiscard]] bool lane_sidecar_eligible(const Tensor& tensor);
[[nodiscard]] SidecarPlan plan_sidecar(const ModelFiles& source,
    std::span<const std::string> tensor_names, uint64_t byte_budget);

// Builds one bounded sibling temporary GGUF, proves every inverse byte, seals its
// fixed-length manifest and publishes read-only with an exclusive atomic rename.
// An existing output, including a symlink, is never replaced.
[[nodiscard]] SidecarPlan build_sidecar(std::shared_ptr<const ModelFiles> source,
    const std::filesystem::path& output, std::span<const std::string> tensor_names,
    uint64_t byte_budget, uint64_t chunk_bytes = sidecar_default_chunk_bytes);

// Enforces the exact selected set, geometry, canonical header, both payload hashes
// and a complete inverse-byte comparison against the unchanged original source.
// The returned ModelFiles owns its regular read-only descriptor and mapping.
// Serial remains the default. Explicit workers authenticate independent tensors;
// all checks finish before return, with the lowest tensor-index failure rethrown.
[[nodiscard]] AdoptedSidecar adopt_sidecar(std::shared_ptr<const ModelFiles> source,
    const std::filesystem::path& sidecar, std::span<const std::string> expected_names,
    uint64_t byte_budget, uint64_t chunk_bytes = sidecar_default_chunk_bytes,
    uint32_t authentication_workers = 1);

} // namespace tokspur::lanes
