#pragma once
#if defined(TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL) && !defined(TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL)
#error Sampled input8 requires the separately built scalar-channel transport.
#endif

#if defined(TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT) && (!defined(TOKSPUR_PRIVATE_CONCURRENT_DISPATCH) || !defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2))
#error Concurrent HTTP requires the explicit concurrent/depth-two carrier.
#endif
#if defined(TOKSPUR_PRIVATE_MTP_OUTPUT64) && (!defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2) || !defined(TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT))
#error Output-packet selection requires the explicit depth-two concurrent HTTP carrier.
#endif
#if defined(TOKSPUR_PRIVATE_HTTP_PREFILL512) && (!defined(TOKSPUR_PRIVATE_NATIVE_MTP) || !defined(TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL) || !defined(TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT))
#error Input512 requires the explicit native MTP scalar/concurrent HTTP carrier.
#endif
#include "tokspur/json.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tokspur {

using WorkerDeadline = std::chrono::steady_clock::time_point;
class WorkerError : public std::runtime_error { public: using std::runtime_error::runtime_error; };
class WorkerCancelled : public std::runtime_error { public: using std::runtime_error::runtime_error; };

struct OwnedWorkerOptions {
    std::filesystem::path executable, policy, library;
    std::vector<std::filesystem::path> shards;
    std::string stack_execution = "direct"; // Startup only: direct or recurrent-replay.
    uint32_t checkpoint_slots = 0;
    uint64_t checkpoint_budget_bytes = 0; // Paired opt-in; enables zero-step broker controls.
#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
    std::optional<bool> concurrent_dispatch; // Presence is explicit startup consent, including OFF.
#endif
    std::chrono::milliseconds startup_timeout{1200000}, step_timeout{60000}, drain_timeout{10000};
    std::function<bool()> startup_cancelled; // Parent only; never invoked after fork.
    std::filesystem::path derived_pack;
    std::optional<uint32_t> derived_pack_auth_workers;
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    std::filesystem::path weight_lanes; // Explicit startup-only derived layout; absent keeps original imports.
    std::optional<uint32_t> weight_lanes_auth_workers; // Absent retains serial argv and metadata.
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    std::filesystem::path mtp_sidecar; // Explicit sampled-v2 transport, default absent.
    std::filesystem::path dflash2_dir; // Explicit DFlash2 block-8 non-causal speculative drafter directory.
    std::optional<std::string> drafter; // Optional explicit "mtp" or "dflash2" selector.
    std::optional<std::string> dflash2_precision; // Optional explicit "q8_0" or "bf16" DFlash2 dense precision.
    std::optional<uint32_t> dflash2_max_drafts;
    std::optional<uint32_t> dflash2_vocab_slice;
    std::optional<bool> logit_topk;
    std::optional<bool> ple_async_gather;
    std::optional<bool> mtp_fast_round;
    bool prepared_validation = false;
    std::optional<std::filesystem::path> verify_cost_profile, verify_m8_splits;
    std::string dflash2_draft_policy = "always";
    std::optional<bool> dflash2_defer_context, verify_nax_m8;
    std::optional<std::string> dflash2_verify_rows;
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    std::optional<bool> raw_decode; // Startup-only raw routed Q4_K/Q5_1, absent preserves existing argv.
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    std::optional<bool> q8_three_rows; // Absence preserves the legacy child argv.
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
    std::optional<uint32_t> mtp_depth; // Explicit separately built fixed-depth arm.
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
    std::optional<uint32_t> mtp_output_packet; // Explicit 8/64; no input-capacity change.
#endif
#endif
    bool mtp_request_carry = false; // Explicit HTTP-generation lifetime; omission preserves packet-local v2.
    std::optional<std::string> mtp_draft_attention; // Absent preserves serial argv/captures.
    std::optional<bool> mtp_draft_slice;
    std::optional<uint32_t> known_token_capacity;
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    bool prefill_input512 = false; // Explicit chunked known inputs; verifier/output capacity stays separate.
    std::optional<bool> prefill_hc_rows; // Explicit startup selection, refused when uncompiled.
    std::optional<bool> prefill_router_window; // Explicit startup selection, refused when uncompiled.
    std::optional<bool> prefill_hc_down_window; // Explicit startup selection, refused when uncompiled.
    std::optional<bool> qsa_sparse; // Explicit startup selection, refused when uncompiled.
    std::optional<bool> gdn_ported; // Explicit startup selection, refused when uncompiled.
    std::optional<bool> gdn_prefill_ported; // Decoupled prefill-only gdn_pipelined_tb12 selector.
    std::optional<bool> prefill_hc_down_f32; // Presence is refused before child startup when uncompiled; compiled explicit off is forwarded.
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    bool gdn_old_order_batch = false; // Explicit recurrence-only ablation, absent/off keeps existing routes.
#endif
    bool prefill_final_logits = false; // Explicit final-input-v1 contract; absent/off keeps per-packet logits.
    std::optional<std::string> prefill_dense; // Absent preserves the existing f16 input512 argv; f32-staged is an explicit private backend, while wire precision remains f32.
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    std::optional<std::string> prefill_grouped; // Explicit serial/f32 choice; absent preserves the native default.
    std::optional<bool> verify_nax;
    std::optional<bool> verify_hc_gdn_fused;
#endif
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    bool verify8 = false; // Immutable, private greedy-only HTTP capability.
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    bool branch8 = false; // Explicit new broker wire; never inferred from slots.
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    bool stage8_present = false, stage8_enabled = false; // Absent is not prepared-off.
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_FAST_MATH
    std::optional<bool> fast_channel; // Explicit staged/channel pair; both use corrected signed-root PLE.
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
    bool scalar_channel = false; // Separate sampled-capable scalar transport; never selects V8.
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    bool sampled_prefill = false; // Distinct input1..8/output1 transport; never a verifier.
#endif
    bool checkpoint_ssd = false;
    uint64_t checkpoint_ssd_budget_bytes = 8ull << 30;
    uint64_t checkpoint_ssd_floor_bytes = 20ull << 30;
    std::filesystem::path checkpoint_ssd_dir;
    std::string checkpoint_policy = "recompute-cost";
    bool overlap_prefill = false;
    std::optional<bool> swiglu_fused;
    std::optional<std::string> q8_gemv_rg4;
    std::optional<std::string> q8_three_msg;
    std::optional<bool> moe_three_union;
    std::optional<std::string> mtp_lookup;
    std::optional<uint32_t> mtp_lookup_match;
    std::optional<std::string> prefill_m64;
    std::optional<bool> prefill_multiseg;
    std::optional<bool> prefill_swizzle;
    std::optional<bool> prefill_router_fast;
    std::optional<bool> qsa_prefill_fused;
    std::optional<bool> hc_consumer_fused;
    std::optional<bool> ple_overlap;
    std::optional<bool> shared_prefix_cache;
};

#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
inline constexpr std::array<std::string_view, 5> owned_weight_lane_fields{
    "weight_lanes_schema", "weight_lanes_manifest_sha256", "weight_lanes_tensor_count",
    "weight_lanes_file_bytes", "weight_lanes_original_payload_bytes_skipped"};
inline constexpr std::array<std::string_view, 2> owned_lane_auth_fields{
    "lane_auth_workers_requested", "lane_auth_workers"};
// HTTP consent binds presence at READY and the complete typed identity at every completion.
// File authentication and adoption remain in the pinned native child.
std::optional<uint32_t> owned_lane_auth_workers_requested(const Json&);
void validate_owned_weight_lanes_ready(const Json&, bool selected, std::optional<uint32_t> workers = std::nullopt);
void validate_owned_weight_lanes_done(const Json& ready, const Json& done);
#endif

#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
#if !defined(TOKSPUR_PRIVATE_HTTP_FAST_MATH) || !defined(TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8) || !defined(TOKSPUR_PRIVATE_HTTP_BRANCH8)
#error Scalar channel serving requires the explicitly built math/transport validators.
#endif
inline bool scalar_channel_transport(const OwnedWorkerOptions& options) {
    return options.scalar_channel && options.stack_execution == "direct" &&
        !options.verify8 && !options.branch8 && options.checkpoint_slots <= 4 && options.checkpoint_budget_bytes <= (512u << 20) &&
        options.stage8_present && options.stage8_enabled && options.fast_channel == std::optional<bool>(true);
}
#endif

#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
inline bool owned_weight_lanes_transport(const OwnedWorkerOptions& options) {
#ifdef TOKSPUR_PRIVATE_HTTP_SCALAR_CHANNEL
    return scalar_channel_transport(options);
#else
    static_cast<void>(options); return false;
#endif
}
#endif

#ifdef TOKSPUR_PRIVATE_HTTP_MTP_CONCURRENT
bool owned_concurrent_transport(const OwnedWorkerOptions&);
void validate_owned_concurrent_ready(const Json&, std::optional<bool> selected);
void validate_owned_concurrent_done(const Json& ready, const Json& done);
#endif

#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
void validate_owned_q8_three_ready(const Json&, std::optional<bool> selected);
void validate_owned_q8_three_previous(const Json& ready, const Json& previous);
// The caller may pass cold_resync only after the existing cold known-input reset guard has admitted it. OwnedWorker retains every drained DONE and uses strict deltas.
void validate_owned_q8_three_done(const Json& ready, const Json& previous, const Json& done, bool cold_resync = false);
#endif

// One persistent owned target, fixed argv and one command in flight. Each
// admitted command performs at most one model step (one input OR one output).
// Cancellation drains that command before throwing; the next request must
// explicitly reset. EOF, malformed framing and uncertain failures kill/drain
// the child, invalidate admission and never trigger restart, fallback or retry.
// Public methods must not be called reentrantly from callbacks.
// Prepared direct workers also admit branch_begin/branch_capture controls.
// Their future prompt IDs are metadata only: no control executes model tokens.
// Private V8 mode instead admits <=8 input OR <=8 output tokens. Its callback
// receives ordered round/token records ONLY after the complete done is admitted.
class OwnedWorker {
public:
    explicit OwnedWorker(OwnedWorkerOptions options);
    ~OwnedWorker();
    OwnedWorker(const OwnedWorker&) = delete;
    OwnedWorker& operator=(const OwnedWorker&) = delete;
    [[nodiscard]] const Json& ready() const;
    Json request(const Json&, WorkerDeadline, const std::function<bool()>& cancelled,
                 const std::function<void(const Json&)>& on_token = {});
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    void abort_mtp_generation(WorkerDeadline); // Zero model steps after an admitted drain; failures invalidate.
#endif
    void close(); // Throws unless the exact guardian wait/worker exit/pipe drain is verified.
    [[nodiscard]] bool alive() const;
private:
    // Both callers hold Impl::mutex through admission, drain and publication.
    Json request_locked(const Json&, WorkerDeadline, const std::function<bool()>&,
                        const std::function<void(const Json&)>&);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tokspur
