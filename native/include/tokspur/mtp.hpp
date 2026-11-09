#pragma once

#include "tokspur/model.hpp"

#include <memory>

namespace tokspur {

// This is a closed catalogue/ownership component, not a learned draft graph.
// Asset digests identify the prospective assets; no function here computes or
// verifies a full-weight digest, and none provides a runnable capability.
inline constexpr std::string_view mtp_shared_q8_revision = "766911a6b7369840a91dbcd95f9f997acaab6cd6";
inline constexpr std::string_view mtp_shared_q8_sha256 = "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6";
inline constexpr uint64_t mtp_shared_q8_file_bytes = 2'786'568'256;
inline constexpr uint64_t mtp_shared_q8_header_bytes = 10'946'624;

struct MtpTensorRequirement {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t encoding = 0;
    uint64_t offset = 0; // Absolute offset in this one pinned sidecar profile.
    uint64_t bytes = 0;
};

[[nodiscard]] std::vector<MtpTensorRequirement> mtp_shared_q8_requirements();

enum class MtpTemplatePolicy {
    require_exact_identity,
    // An explicit metadata inspection policy. All input IDs/rendering remain
    // target-owned. A differing sidecar template is reported, never adopted.
    target_rendered_ids_only,
};

struct MtpSharedLayout {
    size_t owned_tensor_count = 0;
    size_t identical_tokenizer_records = 0;
    uint64_t owned_tensor_bytes = 0;
    uint64_t borrowed_tensor_bytes = 0;
    uint64_t sidecar_template_bytes = 0;
    uint64_t target_template_bytes = 0;
    bool templates_identical = false;
    bool full_target_profile_checked = false;
    MtpTemplatePolicy template_policy = MtpTemplatePolicy::require_exact_identity;
    [[nodiscard]] Json describe() const;
};

// Low-level catalogue admission. Target descriptors must be supplied from the
// caller's target catalogue. This does not prove a complete target profile or
// asset payload identity. Useful with explicit Gguf header-only test views.
// Exact typed tokenizer.ggml values are compared without copying their arrays.
[[nodiscard]] MtpSharedLayout inspect_mtp_shared_q8_catalogue(
    const Gguf& sidecar, const Gguf& target_metadata,
    const Tensor& token_embedding, const Tensor& output,
    MtpTemplatePolicy template_policy);

// Owns the sidecar mapping and retains the complete target mapping lifetime.
// The read-only source accessors retain the admitted mappings for the explicit
// native head importer. They allocate/import nothing and grant no graph or
// execution qualification. There is no startup/serving integration here.
class MtpSharedBinding {
public:
    MtpSharedBinding(std::shared_ptr<const ModelFiles> target,
                     const std::filesystem::path& sidecar,
                     MtpTemplatePolicy template_policy,
                     std::shared_ptr<const ModelFiles> pack = nullptr);
    [[nodiscard]] const MtpSharedLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] const Tensor& borrowed_tensor(std::string_view name) const;
    [[nodiscard]] const Tensor& sidecar_tensor(std::string_view name) const;
    [[nodiscard]] std::shared_ptr<const ModelFiles> sidecar_source() const;
    [[nodiscard]] std::shared_ptr<const ModelFiles> target_source() const;
    void assert_unchanged() const;

private:
    std::shared_ptr<const ModelFiles> target_;
    std::shared_ptr<const ModelFiles> sidecar_;
    std::shared_ptr<const ModelFiles> pack_;
    MtpSharedLayout layout_;
};

} // namespace tokspur
