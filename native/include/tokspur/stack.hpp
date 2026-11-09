#pragma once

#include "tokspur/attention.hpp"
#include "tokspur/memory.hpp"
#include "tokspur/ple.hpp"
#include "tokspur/sampling.hpp"

#include <memory>
#include <string>
#include <variant>

namespace tokspur {
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
enum class GdnChunkMode : uint32_t;
#endif
using StackLayer = std::variant<RecurrentWeights, AttentionLayerWeights>;
struct StackWeights {
    std::vector<StackLayer> layers;
    PleWeights memory;
    MixerWeights mixer;
    GpuMatrix output;
};
struct StackShape {
    ResidualShape residual;
    GdnShape recurrent;
    QsaShape attention;
    MoeShape experts;
    PleShape memory;
    uint32_t vocabulary, memory_layer;
};
struct StackPolicy {
    GdnPolicy recurrent;
    QsaPolicy attention;
    RouterMode router;
    bool normalize_selected;
    NgramPolicy ngram;
};
struct StackResult { std::vector<float> logits; BatchResult batch; uint32_t position; bool logits_available = true; };
// Explicit host-only tap of the complete final HC streams, before the final
// mixer. Two banks make failed/cancelled forward work invisible to the owner.
inline constexpr uint64_t maximum_stack_final_hc_tap_budget = 16ull << 20;
struct StackFinalHcTapMemory {
    bool prepared = false;
    uint32_t capacity = 0, residual_width = 0;
    uint64_t host_charged_bytes = 0;
};
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
struct StackDFlashTapMemory {
    bool prepared = false;
    uint32_t known_capacity = 0, verify_capacity = 0, slot_count = 0, slot_width = 0;
    uint64_t gpu_charged_bytes = 0;
};
struct StackDFlashCapture {
    GpuView rows{};
    uint32_t count = 0, slot_count = 0, slot_width = 0;
    uint64_t stride_bytes = 0;
};
#endif
inline constexpr uint64_t maximum_stack_verification_budget = 128ull << 20;
struct StackVerificationMemory {
    bool prepared = false;
    uint64_t gpu_charged_bytes = 0, host_charged_bytes = 0, tape_payload_bytes = 0;
};
// Probe runs only with an idle graph, before staging and after each completed
// batch. It may throw/cancel; nested stack operations are refused throughout.
struct StackVerificationCancellation { bool (*requested)(void*) = nullptr; void* context = nullptr; };
struct StackVerificationLookup { NgramHistory next; std::span<const uint64_t> rows; };
struct StackVerificationResult {
    uint32_t accepted = 0, position = 0; // position is the original base prefix.
    // Prepared host storage, valid until the next verification call/destruction.
    // k=0: pre-round logits; k=1/2: logits after exactly that many inputs.
    std::span<const float> logits;
    BatchResult verification{}, recovery{};
    double wall_ms = 0;
    // Fixed values, not borrowed rows: pre-round[a], then first-row[b].
    // The second value is meaningful only when accepted == 2.
    std::array<float, 2> source_logits{};
};
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
// p1/p2 are the target rows AFTER each supplied input, never proposal logits.
// final_hc_rows contains every evaluated row even when only the first commits.
// All spans are borrowed until the next stack mutation/destruction.
struct StackSampledVerificationRows {
    uint32_t position = 0, count = 0, residual_width = 0, vocabulary = 0;
    bool has_topk = false;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
    std::array<std::span<const float>, 8> logits{};
    std::array<GpuLogitTopKRow, 8> topk{};
#elif defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST)
    std::array<std::span<const float>, 3> logits{};
    std::array<GpuLogitTopKRow, 3> topk{};
#else
    std::array<std::span<const float>, 2> logits{};
    std::array<GpuLogitTopKRow, 2> topk{};
#endif
    std::span<const float> final_hc_rows{};
    GpuView final_hc_gpu_rows{};
};
struct StackSampledVerificationSelector {
    uint32_t (*choose)(void*, const StackSampledVerificationRows&) = nullptr;
    void* context = nullptr;
    bool allow_topk = false;
};
struct StackSampledVerificationResult {
    uint32_t committed = 0, position = 0;
    std::span<const float> logits{};
    StackSampledVerificationRows evaluated{};
    BatchResult verification{}, recovery{};
    double wall_ms = 0;
};
// Separate opt-in transaction; the old depth-two API/storage stays unchanged.
struct StackVerificationLookups8 {
    uint32_t count = 0;
    std::array<StackVerificationLookup, 8> rows{};
};
struct StackVerificationResult8 {
    uint32_t accepted = 0, position = 0, evaluated = 0, head_dispatches = 0;
    std::span<const float> logits; // Prepared row after exactly accepted inputs.
    BatchResult verification{}, recovery{};
    double wall_ms = 0;
    std::array<float, 8> source_logits{};
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    // Borrowed diagnostic rows, valid until next verification/destruction.
    std::span<const float> evaluated_logits{};
#endif
};
#endif
// Original scheduling experiments, not different model arithmetic. The replay
// arm keeps sparse attention dynamic and executes recorded recurrent runs in
// the same serial token batch. Captures and kernel policy are fixed at creation.
enum class StackExecution : uint32_t { direct, indirect_compatible, recurrent_replay };
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
struct StackKnownQsaRowsMemory {
    bool prepared = false, enabled = false;
    uint64_t logical_bytes = 0, charged_bytes = 0;
    uint32_t allocations = 0;
};
inline constexpr uint64_t maximum_known_qsa_rows_bytes = 64ull << 20;
#endif

inline constexpr uint32_t maximum_stack_checkpoint_slots = 4;
inline constexpr uint64_t maximum_stack_checkpoint_budget = 512ull << 20;
struct StackCheckpointConfig { uint32_t slots; uint64_t byte_budget; };
struct StackCheckpointMemory {
    uint32_t slots = 0, retained = 0;
    uint64_t payload_bytes_per_slot = 0, gpu_charged_bytes = 0, host_metadata_bytes = 0;
};
// Process-local capability, not a serialized cache key or a token-prefix claim.
// Only its originating TextStack can interpret it. A default handle is invalid.
class StackCheckpoint {
public:
    StackCheckpoint() = default;
    bool operator==(const StackCheckpoint&) const = default;
private:
    uint64_t owner_ = 0, generation_ = 0;
    uint32_t slot_ = 0;
    StackCheckpoint(uint64_t owner, uint64_t generation, uint32_t slot)
        : owner_(owner), generation_(generation), slot_(slot) {}
    friend class TextStack;
};

// Serial target-only stack. Graph and immutable weights must outlive this object;
// the caller grants it exclusive graph use during each method. No external model
// executor or mathematical reference is called on the encoded path.
// HC Q8 policy is separately pinned for this entire object's lifetime. A nonzero
// policy admits direct execution only, without residual-many reuse or verifier.
// This is not yet an asynchronous epoch/lease manager or persistent cache.
class TextStack {
public:
    TextStack(MetalGraph& graph, StackWeights weights, StackShape shape,
              StackPolicy policy, NgramSpec ngram, StackExecution execution = StackExecution::direct,
              uint32_t known_token_capacity = 0, bool known_residual_reuse = false);
    ~TextStack();
    TextStack(const TextStack&) = delete;
    TextStack& operator=(const TextStack&) = delete;

    [[nodiscard]] uint32_t position() const;
    [[nodiscard]] StackExecution execution() const;
    [[nodiscard]] bool replay_prepared() const;
    [[nodiscard]] uint32_t replay_capture_count() const;
    [[nodiscard]] uint32_t known_token_capacity() const; // Zero/eight, with 512/1024/2048 admitted in private PREFILL_CHUNK builds.
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    // Startup-only experiment. Evaluated views remain valid until the next verification or destruction; callers copy retained results before another stack operation.
    void configure_sampled_logit_views(bool enabled);
    [[nodiscard]] bool sampled_logit_views() const;
    void configure_sampled_logit_topk(bool enabled);
    [[nodiscard]] bool sampled_logit_topk() const;
    void read_verified_committed_logits(uint32_t committed_1based, std::span<float> destination);
#endif
    [[nodiscard]] bool known_residual_reuse() const; // Never changes ordinary step.
    // Prepare once at an empty direct stack, before a verifier. Capacity eight
    // covers ordinary steps/known prefill. Persistent checkpoints are refused.
    void prepare_final_hc_tap(uint64_t byte_budget);
    void configure_gpu_final_hc_tap(bool enabled);
    [[nodiscard]] bool gpu_final_hc_tap() const;
    [[nodiscard]] StackFinalHcTapMemory final_hc_tap_memory() const;
    [[nodiscard]] uint32_t last_final_hc_count() const;
    [[nodiscard]] std::span<const float> last_final_hc_rows() const;
    [[nodiscard]] std::span<const float> last_final_hc_residual() const;
    [[nodiscard]] bool has_last_final_hc_gpu() const;
    [[nodiscard]] GpuView last_final_hc_gpu_rows() const;
    [[nodiscard]] GpuView last_final_hc_gpu_residual() const;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    void prepare_dflash_tap(std::span<const uint32_t> layer_ids, uint64_t byte_budget);
    [[nodiscard]] StackDFlashTapMemory dflash_tap_memory() const;
    [[nodiscard]] StackDFlashCapture last_dflash_capture() const;
#endif
    // Prepared-resource comparator only, at an empty prefix with unchanged
    // direct kernel policy. Allocations remain identical in both modes.
    void configure_known_residual_reuse(bool enabled);
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
    // One-time empty-prefix preparation. Explicit OFF and ON own identical
    // bounded workspace. Only step_known uses it; ordinary/verifier math stays
    // unchanged. Initial spike admits capacity8, not capacity512 or HTTP.
    void prepare_known_qsa_rows(bool enabled);
    [[nodiscard]] StackKnownQsaRowsMemory known_qsa_rows_memory() const;
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
    // Empty-prefix diagnostic selector; prepared resources stay identical.
    // OFF is the chronological large-tile candidate with ordinary FFN math.
    [[nodiscard]] bool known_grouped_ffn() const;
    void configure_known_grouped_ffn(bool enabled);
    [[nodiscard]] bool known_layer_submissions() const;
    void configure_known_layer_submissions(bool enabled);
    // Separate default-off selector; never enables capacity-eight reuse.
    [[nodiscard]] bool known_hc_residual_rows() const;
    void configure_known_hc_residual_rows(bool enabled);
#endif
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    void prepare_chunked_prefill(GdnChunkMode mode, uint64_t budget_bytes, bool legacy_order = false);
    [[nodiscard]] bool chunked_prefill_enabled() const;
    [[nodiscard]] uint64_t chunked_prefill_bytes() const;
    // Bounded generated-data diagnostic only, including unpublished capacity bytes.
    [[nodiscard]] std::vector<std::byte> chunked_cache_snapshot() const;
#endif
    // Same-resident comparator only: the stack must have been constructed in
    // replay mode and have an empty committed prefix. All captures stay resident
    // and the startup kernel policy stays fixed in every selected arm.
    void configure_execution(StackExecution execution);
    [[nodiscard]] NgramHistory committed_history() const;
    [[nodiscard]] NgramStep lookup(uint32_t token) const;
    // Stages chronological n-gram lookups from a COPY of the accepted history.
    // Requires prepared known-input resources and1..capacity tokens.
    // No state is published, and no borrowed embedding-cache row is retained.
    [[nodiscard]] std::vector<NgramStep> lookups(std::span<const uint32_t> tokens) const;
    // The asset owner supplies the requested token row and concatenated lookup
    // rows. They are copied before encoding. All state publication is internal.
    StackResult step(uint32_t token, std::span<const float> token_row,
                     std::span<const float> memory_rows);
    // Explicit direct/unprofiled known-token PREFILL only. Rows are flattened
    // token-major, and every layer/state advances atomically by all N tokens.
    // Returns only LAST-row logits and the original base position. This is NOT
    // speculative verification, partial acceptance, sampling or RNG correction.
    // Ordinary step remains ordinary even when these resources are resident.
    // Explicit input512 calls may defer logits. Trunk state and all final HC rows still publish atomically; deferred results contain no logits and are never ready for generation.
    StackResult step_known(std::span<const uint32_t> tokens,
                           std::span<const float> token_rows,
                           std::span<const float> memory_rows, bool emit_logits = true);
    // Default-off, fixed depth-two transient transaction, separately budgeted
    // from persistent checkpoints. Prepare once while empty/idle under the same
    // direct, unprofiled split0/16 block-Q8 policy as known-token prefill.
    void prepare_transient_verification(uint64_t byte_budget);
    [[nodiscard]] StackVerificationMemory verification_memory() const;
    // Borrowed prepared rows, valid until next verification_lookups/verify call.
    [[nodiscard]] std::array<StackVerificationLookup, 2> verification_lookups(std::span<const uint32_t> tokens) const;
    // Caller supplies authoritative committed prefix and its matching retained
    // logits, never StackResult.position (which is a pre-batch base) or proposal
    // logits. Prefix must equal position(). Lowest-ID argmax is exact greedy. Fewer
    // than two remaining outputs, a first-token stop or first mismatch returns
    // k=0 without GPU work: caller performs its ordinary target step/stop path.
    // Second mismatch publishes only k=1; correction is NOT appended here.
    // Exceptions/cancellation preserve the base; no hidden retry is performed.
    StackVerificationResult verify_greedy2(std::span<const uint32_t> tokens,
        std::span<const float> token_rows, std::span<const float> memory_rows,
        uint32_t committed_prefix, std::span<const float> committed_logits, uint32_t remaining_outputs,
        std::span<const uint32_t> stop_tokens = {}, StackVerificationCancellation cancellation = {});
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    // Explicit depth-one sampled transaction using the existing V8 tapes and
    // optional QSA row kernels. Unlike legacy verifier preparation, this admits
    // an already-prepared known-QSA prefill workspace; allocations stay separate.
    // Requires the final-HC tap, capacity eight and no persistent checkpoints.
    void prepare_sampled_verification2(uint64_t byte_budget
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        , QsaVerifyRowsConfig qsa_rows = {}
#endif
        );
    [[nodiscard]] std::array<StackVerificationLookup, 2> sampled_verification_lookups(
        std::span<const uint32_t> tokens) const;
    // Evaluate arbitrary 1..2 inputs. The synchronous selector sees all target
    // logits/HC rows and chooses a nonzero consumed prefix <= input count.
    // It may throw. No RNG is owned here and no proposal must equal an argmax.
    // All fallible work/cancellation precedes state+history+HC publication;
    // nested stack operations are refused while the selector runs. A rejected
    // tail uses the existing compact GDN replay and PLE/QSA prefix recovery.
    StackSampledVerificationResult verify_sampled2(std::span<const uint32_t> tokens,
        std::span<const float> token_rows, std::span<const float> memory_rows,
        uint32_t committed_prefix, StackSampledVerificationSelector selector,
        StackVerificationCancellation cancellation = {});
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
    [[nodiscard]] std::array<StackVerificationLookup, 3> sampled_verification_depth2_lookups(std::span<const uint32_t>) const;
    StackSampledVerificationResult verify_sampled_depth2(std::span<const uint32_t>, std::span<const float>, std::span<const float>,
        uint32_t committed_prefix, StackSampledVerificationSelector, StackVerificationCancellation = {});
    [[nodiscard]] std::array<StackVerificationLookup, 4> sampled_verification_depth3_lookups(std::span<const uint32_t>) const;
    StackSampledVerificationResult verify_sampled_depth3(std::span<const uint32_t>, std::span<const float>, std::span<const float>,
        uint32_t committed_prefix, StackSampledVerificationSelector, StackVerificationCancellation = {});
    [[nodiscard]] std::array<StackVerificationLookup, 8> sampled_verification8_lookups(std::span<const uint32_t> tokens) const;
    StackSampledVerificationResult verify_sampled8(std::span<const uint32_t> tokens, std::span<const float> token_rows,
        std::span<const float> memory_rows, uint32_t committed_prefix, StackSampledVerificationSelector selector,
        StackVerificationCancellation cancellation = {});
#endif
#if defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
    // Retained diagnostic entry. D2 uses its separately named exact-three-row entry.
    [[nodiscard]] std::array<StackVerificationLookup, 3> sampled_verification_cost_lookups(
        std::span<const uint32_t> tokens) const;
    StackSampledVerificationResult verify_sampled_cost3(std::span<const uint32_t> tokens,
        std::span<const float> token_rows, std::span<const float> memory_rows,
        uint32_t committed_prefix, StackSampledVerificationSelector selector,
        StackVerificationCancellation cancellation = {});
#endif
    // Mutually exclusive with old verification; checkpoints stay refused unless
    // separately compiled private startup consent was explicitly granted.
    // Known capacity8,
    // Q8 vocabulary head and direct/unprofiled immutable policies required.
    void prepare_transient_verification8(uint64_t byte_budget
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        , QsaVerifyRowsConfig qsa_rows = {}
#endif
        );
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    [[nodiscard]] QsaVerifyRowsMemory verification_qsa_rows_memory() const;
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
    // Consent only: allocates nothing, before either store on an empty direct
    // capacity-eight stack. The <=512MiB persistent and <=128MiB transient
    // budgets remain separate and both are included in the native ledger.
    void permit_verification8_branch_reuse();
#endif
    // Input proposals contain2..8 IDs. count is trimmed by remaining/context
    // reservation and first proposed stop; count<2 stages nothing (ordinary).
    [[nodiscard]] StackVerificationLookups8 verification_lookups8(std::span<const uint32_t> tokens,
        uint32_t remaining_outputs, std::span<const uint32_t> stop_tokens = {}) const;
    // Rows cover the supplied IDs; only the trimmed active prefix is consumed.
    // k0 never submits. Other partial prefixes replay compact GDN tapes, restore
    // PLE/QSA sums and publish atomically after every fallible/cancel check.
    StackVerificationResult8 verify_greedy8(std::span<const uint32_t> tokens,
        std::span<const float> token_rows, std::span<const float> memory_rows,
        uint32_t committed_prefix, std::span<const float> committed_logits, uint32_t remaining_outputs,
        std::span<const uint32_t> stop_tokens = {}, StackVerificationCancellation cancellation = {});
#endif
    // Transactional empty-prefix publication, reuses allocations. A replay-
    // prepared stack cannot accept a changed kernel policy even while executing
    // direct/compatible mode or at prefix zero; construct a new stack. The same
    // immutable startup-policy rule applies to a known-token-prepared stack.
    void reset();
    // Optional synchronous model-state foundation. Default construction retains
    // no checkpoint resources. Prepare once, while empty and idle, in direct
    // execution (no replay preparation). Pins the current kernel/reuse policy.
    // Budget includes actual GPU allocation charges and bounded host metadata;
    // no QSA KV/index cache copies, logits, terminal flags or RNG are retained.
    void prepare_checkpoints(StackCheckpointConfig config);
    [[nodiscard]] StackCheckpointMemory checkpoint_memory() const;
    // Fixed prepared storage; full capacity evicts the oldest retained handle.
    // Refusals leave both accepted state and all retained handles unchanged.
    [[nodiscard]] StackCheckpoint retain_checkpoint();
    // Restores only a retained ancestor/current prefix on this stack's active
    // lineage. Copies into candidate banks, then atomically publishes. Handles
    // strictly beyond the restored prefix become stale before any later append;
    // older/equal handles survive. Successful reset invalidates every handle.
    // Caller must separately reconcile its token history, retained logits,
    // terminal metadata and entropy. Restore alone cannot authorize generation.
    uint32_t restore_checkpoint(StackCheckpoint checkpoint);
    // Truncates in-place shared KV prefix to an ancestor position, invalidating descendants.
    void truncate_shared_kv(uint32_t prefix);
    [[nodiscard]] uint64_t ssd_checkpoint_payload_bytes(uint32_t prefix) const;
    [[nodiscard]] uint64_t ssd_checkpoint_payload_bytes() const;
    [[nodiscard]] bool read_ssd_checkpoint_span(uint64_t offset, std::span<std::byte> target) const;
    [[nodiscard]] bool write_ssd_checkpoint_span(uint32_t target_prefix, uint64_t offset, std::span<const std::byte> source);
    // Explicit, bounded development diagnostic, never full model checkpoint I/O.
    [[nodiscard]] std::vector<std::byte> accepted_snapshot() const;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    // Correctness-only streaming digest of the accepted_snapshot layout. No aggregate snapshot allocation and no change to its existing size ceiling.
    [[nodiscard]] std::string accepted_state_sha256() const;
#endif
#if defined(TOKSPUR_PRIVATE_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_CONCURRENT_DISPATCH)
    struct AcceptedFingerprint {
        uint64_t bytes = 0;
        std::array<unsigned char, 32> sha256{};
        bool operator==(const AcceptedFingerprint&) const = default;
    };
    // Diagnostic only, exact accepted_snapshot byte order, without its8MB copy
    // ceiling. Caller scratch is bounded to1MB; never used inside a timed round.
    [[nodiscard]] AcceptedFingerprint accepted_fingerprint(std::span<std::byte> scratch) const;
#endif

private:
    StackResult step_with_logits(uint32_t token, std::span<const float> token_row,
                                 std::span<const float> memory_rows, bool emit_logits);
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    StackSampledVerificationResult verify_sampled_bounded(std::span<const uint32_t> tokens,
        std::span<const float> token_rows, std::span<const float> memory_rows,
        uint32_t committed_prefix, StackSampledVerificationSelector selector,
        StackVerificationCancellation cancellation, uint32_t maximum_rows);
#endif
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tokspur
