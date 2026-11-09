#include "tokspur/operator_checkpoints.hpp"
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
#include "tokspur/moe_q8_three_rows.hpp"
#endif
#include "tokspur/stack.hpp"
#include "tokspur/carrier_phase.hpp"
#if defined(TOKSPUR_PRIVATE_KNOWN_QSA_ROWS) && (!defined(TOKSPUR_PRIVATE_VERIFY_QSA_ROWS) || !defined(TOKSPUR_PRIVATE_VERIFY_EIGHT))
#error "known QSA row spike requires the isolated verifier-row build lineage"
#endif
#if defined(TOKSPUR_PRIVATE_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_CONCURRENT_DISPATCH)
#include <CommonCrypto/CommonDigest.h>
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
#include "tokspur/qsa_stage_v8.hpp"
#endif
#include "tokspur/gguf.hpp"
#include "tokspur/speculative_controls.hpp"
#include "tokspur/recurrent_tile.hpp"
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
#include "tokspur/prefill_gdn.hpp"
#include "tokspur/prefill_attention.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
#include "tokspur/attention_tile.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
#include "tokspur/verify_nax.hpp"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>

namespace tokspur {
namespace {
struct RecurrentPair { GdnState current, next; };
struct AttentionPair { QsaCache cache; GpuView current, next; };
using LayerState = std::variant<RecurrentPair, AttentionPair>;
struct RecurrentRun {
    size_t begin, end; // Half-open, all recurrent; sparse attention is never recorded.
    uint32_t dispatches;
    std::array<GraphCapture, 2> captures{};
};
void require(bool ok, const char* message) { if (!ok) throw FormatError(message); }
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
struct ChunkedPrefillStore {
    MetalGraph& g;
    GdnChunkMode mode = GdnChunkMode::fused;
    bool legacy_order = false;
    PrefillLayerScratch layer{};
    GpuView prefill_core_arena{};
    GdnChunkScratch gdn{};
    GdnState gdn_work{};
    QsaPrefillScratch qsa{};
    MoeMmManyScratch moe{};
    std::vector<std::variant<GdnPrefillLayouts, AttentionPrefillLayouts>> layouts;
    std::vector<std::array<GpuView, 3>> cache_backup;
    std::vector<GpuView> owned;
    uint64_t charged = 0;
    explicit ChunkedPrefillStore(MetalGraph& graph) : g(graph) { owned.reserve(1024); }
    template<class Range> void keep(const Range& views) { for (auto v : views) if (v.owner) owned.push_back(v); }
    GpuView zeros(uint64_t bytes, const char* label = "prefill physical cache rollback") { const auto v = g.zeros(bytes, label); owned.push_back(v); return v; }
    ~ChunkedPrefillStore() { for (auto i = owned.rbegin(); i != owned.rend(); ++i) try { g.release(*i); } catch (...) {} }
};
#endif
std::atomic<uint64_t> next_checkpoint_owner{1};
uint64_t checkpoint_owner() {
    auto owner = next_checkpoint_owner.load(std::memory_order_relaxed);
    do {
        require(owner < std::numeric_limits<uint64_t>::max(), "checkpoint owner space exhausted");
    } while (!next_checkpoint_owner.compare_exchange_weak(owner, owner + 1, std::memory_order_relaxed));
    return owner;
}
struct CheckpointSlot {
    GpuView storage{};
    NgramHistory history{};
    uint64_t generation = 0;
    uint32_t prefix = 0;
    bool valid = false;
    std::atomic<uint64_t> reader_epoch{0};
    std::atomic<bool> cancel_spill{false};

    CheckpointSlot() = default;
    CheckpointSlot(const CheckpointSlot& other)
        : storage(other.storage), history(other.history), generation(other.generation),
          prefix(other.prefix), valid(other.valid),
          reader_epoch(other.reader_epoch.load(std::memory_order_relaxed)),
          cancel_spill(other.cancel_spill.load(std::memory_order_relaxed)) {}
    CheckpointSlot& operator=(const CheckpointSlot& other) {
        if (this != &other) {
            storage = other.storage;
            history = other.history;
            generation = other.generation;
            prefix = other.prefix;
            valid = other.valid;
            reader_epoch.store(other.reader_epoch.load(std::memory_order_relaxed), std::memory_order_relaxed);
            cancel_spill.store(other.cancel_spill.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
};
struct CheckpointStore {
    std::array<CheckpointSlot, maximum_stack_checkpoint_slots> slots{};
    uint64_t owner = 0, generation = 0, payload_bytes = 0, gpu_charged_bytes = 0;
    uint32_t capacity = 0;
    std::array<uint32_t, 2> kernel_policy{};
    bool residual_reuse = false;
};
struct VerificationStore {
    RecurrentScratch recurrent{};
    GdnCoreReplayScratch replay{};
    std::vector<GdnCoreTape> tapes;
    std::vector<GpuView> first_sums, owned;
    GpuView streams{}, second{}, memory_streams{}, embeddings{}, first_memory{}, logits{};
    std::vector<float> host_streams;
    std::array<std::vector<float>, 3> host_logits; // pre-round, after first, after second
    std::array<std::vector<uint64_t>, 2> lookup_rows;
    uint64_t gpu_charged_bytes = 0, host_charged_bytes = 0, tape_payload_bytes = 0;
    std::array<uint32_t, 2> kernel_policy{};
    bool residual_reuse = false;
};
struct FinalHcTap {
    uint32_t capacity = 8;
    std::vector<float> current, candidate;
    uint32_t count = 0, width = 0;
    uint64_t charged_bytes = 0;
};
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
struct DFlashTap {
    std::vector<int32_t> slot_for_layer;
    uint32_t slot_count = 0, slot_width = 0, known_capacity = 0, verify_capacity = 8;
    GpuView known_gpu{}, verify_gpu{};
    std::vector<GpuView> owned;
    uint64_t gpu_charged_bytes = 0;
    StackDFlashCapture last_capture{};
};
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
struct VerificationStore8 {
    RecurrentScratch recurrent{};
    GdnCoreReplayScratch replay{};
    std::vector<GdnCoreTape> tapes;
    std::vector<GpuView> prefix_sums, owned;
    GpuView streams{}, second{}, memory_streams{}, embeddings{}, prefix_memory{}, mixed{}, logits{}, topk_scratch{}, topk_rows{};
    std::vector<float> host_streams, host_logits; // Nine fixed rows: base, then1..8.
    std::array<std::vector<uint64_t>, 8> lookup_rows;
    uint64_t gpu_charged_bytes = 0, host_charged_bytes = 0, tape_payload_bytes = 0;
    std::array<uint32_t, 2> kernel_policy{};
    bool residual_reuse = false;
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    QsaVerifyRowsMemory qsa_rows_memory{};
    QsaScratch qsa_rows{};
    GpuView qsa_scores{}, qsa_factors{};
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    VerifyNaxMoeScratch nax_moe{};
#endif
};
#endif
uint32_t greedy_id(std::span<const float> logits) {
    require(!logits.empty(), "empty greedy target logits");
    uint32_t best = 0;
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    float best_val = -std::numeric_limits<float>::infinity();
    size_t base = 0;
    for (; base + 64 <= logits.size(); base += 64) {
        const float* ptr = logits.data() + base;
        uint32_t bad = 0;
        float block_max = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < 64; ++j) {
            const float v = ptr[j];
            bad |= uint32_t((std::bit_cast<uint32_t>(v) & 0x7f800000u) == 0x7f800000u);
            block_max = v > block_max ? v : block_max;
        }
        require(!bad, "nonfinite greedy target logits");
        if (base == 0 || block_max > best_val) {
            for (size_t j = 0; j < 64; ++j) {
                if (base + j == 0 || ptr[j] > best_val) {
                    best_val = ptr[j];
                    best = uint32_t(base + j);
                }
            }
        }
    }
    for (size_t i = base; i < logits.size(); ++i) {
        require(std::isfinite(logits[i]), "nonfinite greedy target logits");
        if (i == 0 || logits[i] > best_val) {
            best_val = logits[i];
            best = uint32_t(i);
        }
    }
#else
    for (uint32_t i = 0; i < logits.size(); ++i) {
        require(std::isfinite(logits[i]), "nonfinite greedy target logits");
        if (logits[i] > logits[best]) best = i; // Strict comparison preserves lowest-ID ties.
    }
#endif
    return best;
}
bool same_view(GpuView a, GpuView b) {
    return a.owner == b.owner && a.slot == b.slot && a.offset == b.offset && a.bytes == b.bytes;
}
void shape_check(StackShape s, StackPolicy p, const NgramSpec& n, size_t layers) {
    validate_recurrent_shape({s.residual, s.recurrent, s.experts});
    validate_qsa_shape(s.attention); validate_qsa_policy(p.attention);
    validate_gdn_policy(p.recurrent); validate_ple_shape(s.memory); validate_ngram(n, p.ngram);
    require(layers && layers <= 128 && s.memory_layer < layers && s.vocabulary && s.vocabulary <= 1'048'576 &&
        s.vocabulary == n.token_limit && s.attention.channels == s.residual.channels &&
        s.memory.branches == s.residual.branches && s.memory.channels == s.residual.channels &&
        s.memory.embedding == uint64_t(n.order - 1) * n.heads * n.row_width && uint32_t(p.router) <= 1,
        "inconsistent full stack geometry/policy");
    require(p.recurrent.rms_epsilon == p.attention.rms_epsilon, "stack RMS policies differ");
}
}
struct TextStack::Impl {
    MetalGraph& g;
    StackWeights weights;
    StackShape shape;
    StackPolicy policy;
    NgramSpec spec;
    NgramHistory history;
    StackExecution execution;
    const bool replay_prepared;
    const uint32_t known_capacity;
    bool residual_reuse;
    uint32_t prefix = 0;
    std::array<uint32_t, 2> kernel_policy{};
    uint32_t hc_q8_splits = 0;
    bool q8_three_rows = false;
    bool swiglu_fused = false;
    Q8GemvRg4Mode q8_gemv_rg4 = Q8GemvRg4Mode::off;
    Q8ThreeMsgMode q8_three_msg = Q8ThreeMsgMode::off;
    bool moe_three_union = false;
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    QsaStageV8Policy qsa_stage_v8{};
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    QsaChannelPolicy qsa_channel{};
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    bool qsa_sparse = false;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    bool gdn_ported = false;
    bool gdn_prefill_ported = false;
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    HcExactMode hc_exact = HcExactMode::off;
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    bool wide_q8_exact = false;
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    bool wide_q8_many = false;
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    bool verify_nax = false;
    bool q8_matrices_registered = false;
    bool verify_hc_gdn_fused = false;
#endif
    const std::thread::id thread = std::this_thread::get_id();
    std::vector<LayerState> states;
    std::vector<GpuView> owned;
    std::vector<RecurrentRun> runs;
    std::vector<GraphCapture> captures;
    RecurrentScratch r{};
    AttentionLayerScratch a{};
    PleScratch p{};
    MixerScratch m{};
    GpuView current_memory{}, next_memory{}, token_streams{}, second_streams{}, memory_streams{}, embedding{}, logits{};
    RecurrentScratch known_recurrent{};
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
    AttentionTileScratch known_qsa_rows{};
    StackKnownQsaRowsMemory known_qsa_rows_memory{};
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
    MoeManyScratch known_grouped{};
    bool grouped_known_enabled = false;
    bool known_layer_submit = false;
    bool known_hc_rows = false;
#endif
    GpuView known_streams{}, known_second{}, known_memory_streams{}, known_embeddings{}, known_memory_work{}, known_index_work{};
    std::array<GpuView, 2> memory_orientations{};
    std::vector<float> host_streams, known_host_streams, zero;
    std::unique_ptr<CheckpointStore> checkpoints;
    std::unique_ptr<VerificationStore> verification;
    std::unique_ptr<FinalHcTap> final_hc_tap;
    GpuView final_hc_current_gpu{}, final_hc_candidate_gpu{};
    bool final_hc_gpu_mode = false, final_hc_gpu_valid = false, final_hc_candidate_host_valid = false;
    mutable bool final_hc_host_valid = false;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    std::unique_ptr<DFlashTap> dflash_tap;
#endif
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    std::unique_ptr<ChunkedPrefillStore> chunked;
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    std::unique_ptr<VerificationStore8> verification8;
    bool preparing_sampled_verification = false, sampled_verification_prepared = false, sampled_logit_views = false, sampled_logit_topk = false;
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
    bool verification8_branch_reuse = false;
#endif
    bool pending_verification = false;

    void check_thread() const {
        require(thread == std::this_thread::get_id(), "text stack used from another host thread");
        require(!pending_verification, "nested text stack operation during verification");
        require(g.hc_q8_policy().splits == hc_q8_splits, "immutable stack HC Q8 policy changed");
        require(g.q8_three_rows_enabled() == q8_three_rows, "immutable stack three-row Q8 policy changed");
        require(g.swiglu_fused_enabled() == swiglu_fused, "immutable stack fused SwiGLU policy changed");
        require(g.q8_gemv_rg4_mode() == q8_gemv_rg4, "immutable stack multi-SG Q8 GEMV policy changed");
        require(g.q8_three_msg_mode() == q8_three_msg, "immutable stack multi-SG three-row Q8 policy changed");
        require(g.moe_three_union_enabled() == moe_three_union, "immutable stack unioned three-row MoE policy changed");
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
        require(g.qsa_stage_v8_policy() == qsa_stage_v8, "immutable stack staged QSA composition changed");
        require(!qsa_stage_v8.prepared || g.kernel_policy() == kernel_policy, "immutable staged QSA GEMV policy changed");
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
        require(g.qsa_channel_policy() == qsa_channel, "immutable stack chronological QSA policy changed");
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        require(g.qsa_sparse_enabled() == qsa_sparse, "immutable stack sparse QSA policy changed");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        require(g.gdn_ported_enabled() == gdn_ported, "immutable stack ported GDN policy changed");
        require(g.gdn_prefill_ported_enabled() == gdn_prefill_ported, "immutable stack prefill-only ported GDN policy changed");
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
        require(g.hc_exact_mode() == hc_exact, "immutable private exact-HC stack policy changed");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
        require(g.wide_q8_exact_enabled() == wide_q8_exact, "immutable private wide-Q8 stack policy changed");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        require(g.wide_q8_many_enabled() == wide_q8_many, "immutable private wide-Q8-many stack policy changed");
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        require(g.verify_nax() == verify_nax, "immutable stack verify NAX policy changed");
        require(g.verify_hc_gdn_fused_enabled() == verify_hc_gdn_fused, "immutable stack verify fused HC/GDN policy changed");
#endif
    }
    template <class Range> void keep(const Range& views) { for (const auto v : views) owned.push_back(v); }
    GpuView zeros(uint64_t count, const char* label) { const auto v = g.zeros(checked_mul(count, 4), label); owned.push_back(v); return v; }
    void dispose() noexcept {
        // Captures lease every bound allocation, including immutable imported
        // weights. Drain those leases before attempting any owned-buffer release.
        for (auto i = captures.rbegin(); i != captures.rend(); ++i) { try { g.release_capture(*i); } catch (...) {} }
        captures.clear(); runs.clear();
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
        chunked.reset();
#endif
        if (checkpoints) for (auto& slot : checkpoints->slots)
            if (slot.storage.owner) { try { g.release(slot.storage); } catch (...) {} }
        checkpoints.reset();
        if (verification) for (auto i = verification->owned.rbegin(); i != verification->owned.rend(); ++i)
            { try { g.release(*i); } catch (...) {} }
        verification.reset();
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        if (verification8) for (auto i = verification8->owned.rbegin(); i != verification8->owned.rend(); ++i)
            { try { g.release(*i); } catch (...) {} }
        verification8.reset();
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (dflash_tap) for (auto i = dflash_tap->owned.rbegin(); i != dflash_tap->owned.rend(); ++i)
            { try { g.release(*i); } catch (...) {} }
        dflash_tap.reset();
#endif
        for (auto i = owned.rbegin(); i != owned.rend(); ++i) { try { g.release(*i); } catch (...) {} }
        owned.clear();
    }
    Impl(MetalGraph& graph, StackWeights w, StackShape s, StackPolicy pol, NgramSpec ns, StackExecution mode, uint32_t known, bool reuse)
        : g(graph), weights(std::move(w)), shape(s), policy(pol), spec(std::move(ns)), history(empty_ngram_history(spec)), execution(mode),
          replay_prepared(mode == StackExecution::recurrent_replay), known_capacity(known), residual_reuse(reuse) {
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        grouped_known_enabled = known == 512;
#endif
        shape_check(s, pol, spec, weights.layers.size()); kernel_policy = g.kernel_policy();
        hc_q8_splits = g.hc_q8_policy().splits;
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
        qsa_stage_v8 = g.qsa_stage_v8_policy();
        if (qsa_stage_v8.prepared) {
            static_cast<void>(qsa_stage_v8_bytes(1, s.attention));
            require(mode == StackExecution::direct && !hc_q8_splits && kernel_policy == std::array<uint32_t, 2>{16, 3} &&
                (!g.memory().profile_counter_payload_bytes || std::getenv("TOKSPUR_ALLOW_KNOWN_PROFILE")) && !g.memory().live_captures &&
                qsa_stage_v8.scratch_logical_bytes == maximum_qsa_stage_v8_scores &&
                qsa_stage_v8.scratch_charged_bytes >= maximum_qsa_stage_v8_scores,
                "staged QSA composition requires bounded-shape fixed direct unprofiled HC0 startup");
        }
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
        hc_exact = g.hc_exact_mode();
        require(hc_exact == HcExactMode::off || mode == StackExecution::direct,
            "private exact-HC requires direct stack execution");
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
        qsa_channel = g.qsa_channel_policy();
        require(!qsa_channel.prepared || (qsa_stage_v8.prepared && qsa_stage_v8.enabled &&
            hc_exact == HcExactMode::combined && mode == StackExecution::direct &&
            qsa_channel.factor_logical_bytes == maximum_qsa_channel_factors &&
            qsa_channel.factor_charged_bytes >= maximum_qsa_channel_factors),
            "chronological QSA stack requires its matched prepared Stage/HCcombined policy");
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        qsa_sparse = g.qsa_sparse_enabled();
        require(!qsa_sparse || (qsa_channel.prepared && qsa_channel.enabled),
            "sparse QSA stack requires prepared chronological QSA policy");
#endif
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        gdn_ported = g.gdn_ported_enabled();
        gdn_prefill_ported = g.gdn_prefill_ported_enabled();
        require((!gdn_ported && !gdn_prefill_ported) || (mode == StackExecution::direct && s.recurrent.key_width == 128),
            "ported GDN stack requires direct execution and key_width 128");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
        wide_q8_exact = g.wide_q8_exact_enabled();
        require(!wide_q8_exact || mode == StackExecution::direct, "private wide-Q8 requires direct stack execution");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
        wide_q8_many = g.wide_q8_many_enabled();
        require(!wide_q8_many || (mode == StackExecution::direct && known == 8),
            "private wide-Q8-many requires direct capacity-eight stack execution");
#endif
        q8_three_rows = g.q8_three_rows_enabled();
        swiglu_fused = g.swiglu_fused_enabled();
        q8_gemv_rg4 = g.q8_gemv_rg4_mode();
        q8_three_msg = g.q8_three_msg_mode();
        moe_three_union = g.moe_three_union_enabled();
        require(!q8_three_rows || mode == StackExecution::direct, "three-row Q8 requires direct stack execution");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        verify_nax = g.verify_nax();
        verify_hc_gdn_fused = g.verify_hc_gdn_fused_enabled();
        require(!verify_nax || mode == StackExecution::direct, "verify NAX requires direct stack execution");
        require(!verify_hc_gdn_fused || verify_nax, "verify fused HC/GDN requires verify NAX");
#endif
        require(q8_gemv_rg4 == Q8GemvRg4Mode::off || mode == StackExecution::direct, "multi-SG Q8 GEMV requires direct stack execution");
        require(q8_three_msg == Q8ThreeMsgMode::off || (q8_three_rows && mode == StackExecution::direct), "multi-SG three-row Q8 requires direct three-row execution");
        require(!moe_three_union || (q8_three_rows && mode == StackExecution::direct), "unioned three-row MoE requires direct three-row execution");
        require(uint32_t(mode) <= uint32_t(StackExecution::recurrent_replay), "unsupported stack execution mode");
        require(!hc_q8_splits || (mode == StackExecution::direct && !reuse),
            "HC Q8 split policy requires direct execution without known residual reuse");
        require(known == 0 || known == 8
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
            || known == 512 || known == 1024 || known == 2048
#endif
            , "unsupported explicit known-token capacity");
        require(!reuse || known == 8
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            || known == 512 || known == 1024 || known == 2048
#endif
            , "residual reuse requires a supported fixed known-token capacity");
        if (known) check_known_policy();
        owned.reserve(128 * 8 + 128); states.reserve(weights.layers.size());
        runs.reserve(weights.layers.size()); captures.reserve(weights.layers.size() * 2);
        const uint64_t memory_count = uint64_t(s.memory.branches) * s.memory.channels * (s.memory.kernel - 1) * s.memory.dilation;
        const uint64_t maximum = std::max({gdn_state_count(s.recurrent), gdn_qkv_count(s.recurrent) * (s.recurrent.kernel - 1), memory_count, uint64_t(s.attention.index_width)});
        require(maximum <= 16'777'216, "unsupported stack reset scratch");
        zero.resize(maximum, 0); host_streams.resize(uint64_t(s.residual.branches) * s.residual.channels);
        try {
            r = allocate_recurrent_scratch(g, {s.residual, s.recurrent, s.experts}); keep(r.views());
            a = allocate_attention_layer_scratch(g, {s.residual, s.attention, s.experts}); keep(a.views());
            p = allocate_ple_scratch(g, s.memory); keep(p.views());
            m = allocate_mixer_scratch(g, s.residual); keep(m.views());
            current_memory = zeros(memory_count, "accepted PLE history"); next_memory = zeros(memory_count, "candidate PLE history");
            memory_orientations = {current_memory, next_memory};
            token_streams = zeros(host_streams.size(), "token branch streams"); second_streams = zeros(host_streams.size(), "alternating branch streams");
            memory_streams = zeros(host_streams.size(), "PLE branch streams"); embedding = zeros(s.memory.embedding, "lookup rows"); logits = zeros(s.vocabulary, "target logits");
            for (const auto& layer : weights.layers) {
                if (std::holds_alternative<RecurrentWeights>(layer)) {
                    const auto current = allocate_gdn_state(g, s.recurrent); keep(current.views());
                    const auto next = allocate_gdn_state(g, s.recurrent); keep(next.views());
                    states.emplace_back(RecurrentPair{current, next});
                } else {
                    const auto cache = allocate_qsa_cache(g, s.attention); keep(cache.views());
                    const auto current = zeros(s.attention.index_width, "accepted index sum"), next = zeros(s.attention.index_width, "candidate index sum");
                    states.emplace_back(AttentionPair{cache, current, next});
                }
            }
            if (known_capacity) prepare_known(memory_count);
            std::vector<GpuView> protected_weights; protected_weights.reserve(weights.layers.size() * 32 + 16);
            const auto residual = [&](const ResidualWeights& w) { for (auto v : {w.gains, w.down.view, w.up.view, w.inject.view}) protected_weights.push_back(v); };
            const auto experts = [&](const MoeWeights& w) { for (auto v : {w.router.view, w.gate.view, w.up.view, w.down.view, w.shared_gate.view, w.shared_up.view, w.shared_down.view, w.shared_router.view}) protected_weights.push_back(v); };
            for (const auto& layer : weights.layers) {
                if (const auto* rw = std::get_if<RecurrentWeights>(&layer)) {
                    residual(rw->attention_read); residual(rw->ffn_read); experts(rw->ffn); const auto& w = rw->attention;
                    for (auto v : {w.qkv.view, w.gate.view, w.alpha.view, w.beta.view, w.output.view, w.convolution, w.negative_rates, w.dt_bias, w.gains}) protected_weights.push_back(v);
                } else {
                    const auto& aw = std::get<AttentionLayerWeights>(layer); residual(aw.attention_read); residual(aw.ffn_read); experts(aw.ffn); const auto& w = aw.attention;
                    for (auto v : {w.query_gate.view, w.key.view, w.value.view, w.output.view, w.index_query.view, w.index_key.view,
                        w.query_gains, w.key_gains, w.index_query_gains, w.index_key_gains, w.frequencies}) protected_weights.push_back(v);
                }
            }
            for (auto v : {weights.memory.key.view, weights.memory.value.view, weights.memory.key_gains, weights.memory.query_gains,
                weights.memory.conv_gains, weights.memory.convolution, weights.mixer.gains, weights.mixer.down.view, weights.mixer.up.view, weights.output.view}) protected_weights.push_back(v);
            // Protect every layer's accepted state from every supplied weight,
            // including a forged view of a later layer's future allocation.
            g.validate_disjoint(protected_weights, owned);
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            register_nax_q8_matrices();
#endif
            if (replay_prepared) prepare_replay();
        } catch (...) { dispose(); throw; }
    }
    ~Impl() { dispose(); }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    void register_nax_q8_matrices() {
        if (q8_matrices_registered || !(g.prefill_dense_nax() || g.prefill_grouped_nax() || verify_nax)) return;
        const auto reg_q8 = [&](const GpuMatrix& mat, uint32_t cols) {
            if (mat.encoding != 8u || cols < 32u || (cols % 32u) != 0u) return;
            const uint64_t rb = uint64_t(cols / 32u) * 34ull;
            if (!rb || mat.view.bytes < rb || (mat.view.bytes % rb) != 0u) return;
            const uint64_t r64 = mat.view.bytes / rb;
            if (r64 == 0u || r64 > 1'000'000ull) return;
            g.register_q8_matrix(mat.view, cols, uint32_t(r64));
        };
        const uint32_t hc_width = shape.residual.branches * shape.residual.channels;
        const auto reg_res = [&](const ResidualWeights& rw_res) {
            reg_q8(rw_res.down, hc_width);
            reg_q8(rw_res.up, shape.residual.rank);
        };
        const auto reg_moe = [&](const MoeWeights& mw) {
            reg_q8(mw.shared_gate, shape.residual.channels);
            reg_q8(mw.shared_up, shape.residual.channels);
            reg_q8(mw.shared_down, shape.experts.hidden);
        };
        for (const auto& layer : weights.layers) {
            if (const auto* rw = std::get_if<RecurrentWeights>(&layer)) {
                reg_res(rw->attention_read); reg_res(rw->ffn_read); reg_moe(rw->ffn);
                reg_q8(rw->attention.qkv, shape.residual.channels);
                reg_q8(rw->attention.gate, shape.residual.channels);
                reg_q8(rw->attention.output, shape.recurrent.value_heads * shape.recurrent.value_width);
            } else {
                const auto& aw = std::get<AttentionLayerWeights>(layer);
                reg_res(aw.attention_read); reg_res(aw.ffn_read); reg_moe(aw.ffn);
                reg_q8(aw.attention.query_gate, shape.residual.channels);
                reg_q8(aw.attention.key, shape.residual.channels);
                reg_q8(aw.attention.value, shape.residual.channels);
                reg_q8(aw.attention.output, shape.attention.query_heads * shape.attention.head_width);
                reg_q8(aw.attention.index_query, shape.residual.channels);
                reg_q8(aw.attention.index_key, shape.residual.channels);
            }
        }
        reg_q8(weights.memory.key, shape.memory.embedding);
        reg_q8(weights.memory.value, shape.memory.embedding);
        reg_q8(weights.mixer.down, hc_width);
        reg_q8(weights.mixer.up, shape.residual.rank);
        reg_q8(weights.output, shape.residual.channels);
        q8_matrices_registered = true;
    }
#endif

    void check_known_policy() const {
        const auto memory = g.memory();
        require(execution == StackExecution::direct && !replay_prepared &&
            (!memory.profile_counter_payload_bytes || std::getenv("TOKSPUR_ALLOW_KNOWN_PROFILE")
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                || g.carrier_diagnostic_active()
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
                || g.mtp_profile_reserved()
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
                || g.task4_profile_reserved()
#endif
            ) && !memory.live_captures &&
            g.kernel_policy() == kernel_policy &&
            (kernel_policy[0] == 0 || kernel_policy[0] == 16) && (kernel_policy[1] & 1),
            "known-token prefill requires immutable split0/16 block-Q8 direct execution without profiling or captures");
        // Also require exclusive idle graph ownership before staging any rows.
        static_cast<void>(g.last_profile());
    }
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
    void check_large_known_selector() const {
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
        require(!chunked, "prepared chunked prefill refuses legacy selectors");
#endif
        check_thread(); check_known_policy();
        require(known_capacity == 512 && !prefix && !checkpoints && !verification,
            "large-prefill selector requires prepared capacity512, empty prefix and no checkpoint/verifier");
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
        require(!verification8, "large-prefill selector refuses prepared verification8");
#endif
    }
#endif
    void check_checkpoint_policy() const {
        require(checkpoints && execution == StackExecution::direct && !replay_prepared &&
            g.kernel_policy() == checkpoints->kernel_policy && residual_reuse == checkpoints->residual_reuse,
            "checkpoint store requires its unchanged direct kernel/reuse policy");
        static_cast<void>(g.last_profile()); // Exclusive idle graph, no device work.
    }
    void check_verification_policy() const {
        require(verification && residual_reuse == verification->residual_reuse &&
            g.kernel_policy() == verification->kernel_policy, "verification requires its immutable kernel/reuse policy");
        check_known_policy();
    }
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    void check_verification8_policy() const {
        const bool history = !checkpoints
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
            || verification8_branch_reuse
#endif
            ;
        require(verification8 && !verification && history && (known_capacity == 8
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            || ((known_capacity == 512 || known_capacity == 1024 || known_capacity == 2048) && chunked)
#endif
            ) &&
            residual_reuse == verification8->residual_reuse && g.kernel_policy() == verification8->kernel_policy,
            "verification8 requires its exclusive immutable kernel/reuse policy");
        check_known_policy();
        if (checkpoints) check_checkpoint_policy();
    }
#endif
    void invalidate_checkpoints() noexcept {
        if (checkpoints) for (auto& slot : checkpoints->slots) slot.valid = false;
    }
    uint64_t checkpoint_payload_bytes() const {
        uint64_t bytes = current_memory.bytes;
        for (const auto& state : states) {
            if (const auto* r = std::get_if<RecurrentPair>(&state)) {
                bytes = checked_add(bytes, r->current.matrix.bytes);
                bytes = checked_add(bytes, r->current.convolution.bytes);
            } else bytes = checked_add(bytes, std::get<AttentionPair>(state).current.bytes);
        }
        return bytes;
    }
    size_t checkpoint_copies(GpuView storage, bool restore, std::array<GraphCopy, 257>& copies) const {
        uint64_t offset = 0; size_t count = 0;
        const auto append = [&](GpuView current, GpuView next) {
            const auto packed = g.slice(storage, offset, current.bytes);
            copies[count++] = restore ? GraphCopy{packed, next} : GraphCopy{current, packed};
            offset = checked_add(offset, current.bytes);
        };
        append(current_memory, next_memory);
        for (const auto& state : states) {
            if (const auto* r = std::get_if<RecurrentPair>(&state)) {
                append(r->current.matrix, r->next.matrix); append(r->current.convolution, r->next.convolution);
            } else { const auto& a = std::get<AttentionPair>(state); append(a.current, a.next); }
        }
        require(offset == storage.bytes, "checkpoint packed-state geometry mismatch");
        return count;
    }
    void prepare_known(uint64_t memory_count) {
        const auto width = uint64_t(shape.residual.branches) * shape.residual.channels;
        const auto streams = checked_mul(known_capacity, width);
        const auto embeddings = checked_mul(known_capacity, shape.memory.embedding);
        require(streams <= 33'554'432 && embeddings <= 33'554'432,
            "known-token row storage exceeds bounded scratch");
        known_host_streams.resize(streams, 0);
        known_streams = zeros(streams, "known-token branch streams");
        known_second = zeros(streams, "known-token alternating streams");
        known_memory_streams = zeros(streams, "known-token PLE output rows");
        known_embeddings = zeros(embeddings, "known-token lookup rows");
        known_memory_work = zeros(memory_count, "known-token temporary PLE history");
        known_index_work = zeros(shape.attention.index_width, "known-token temporary index sum");
        bool recurrent = false;
        for (const auto& layer : weights.layers) if (const auto* w = std::get_if<RecurrentWeights>(&layer)) {
            const auto& a = w->attention;
            require(a.qkv.encoding == 8 && a.gate.encoding == 8 && a.output.encoding == 8 &&
                a.alpha.encoding == 0 && a.beta.encoding == 0,
                "known-token recurrent prefill requires Q8/F32 core projections");
            recurrent = true;
        }
        if (known_capacity < 512 && (recurrent
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
            || known_capacity == 512
#endif
            )) {
            known_recurrent = allocate_recurrent_tile_scratch(g, {shape.residual, shape.recurrent, shape.experts}, known_capacity);
            keep(known_recurrent.views());
        }
        if (recurrent
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
            || known_capacity >= 512
#endif
            ) {
            g.prepare_q8_matvec_many_all();
            g.prepare_gdn_tiles(128);
        }
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        if (known_capacity == 512
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            && !residual_reuse
#endif
            ) {
            require(!residual_reuse, "grouped prefill cannot change the residual reuse policy");
            for (const auto& layer : weights.layers) std::visit([&](const auto& w) { prepare_moe_many(g, w.ffn); }, layer);
            known_grouped = allocate_moe_many_scratch(g, shape.experts, known_capacity);
            keep(known_grouped.views());
        }
#endif
    }
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    size_t chunked_cache_copies(uint32_t steps, std::span<GraphCopy, 384> save, std::span<GraphCopy, 384> restore) const {
        if (!chunked) return 0;
        require(steps && steps <= known_capacity && prefix < shape.attention.capacity && steps <= shape.attention.capacity - prefix,
            "QSA prefill cache backup step bound");
        const uint64_t kv_stride = uint64_t(shape.attention.kv_heads) * shape.attention.head_width * 2;
        const uint64_t key_append_offset = uint64_t(prefix) * kv_stride;
        const uint64_t key_append_bytes = uint64_t(steps) * kv_stride;
        const uint32_t pool_count = (prefix + steps) / shape.attention.compression - prefix / shape.attention.compression;
        const uint64_t index_append_offset = uint64_t(prefix / shape.attention.compression) * shape.attention.index_width * 2;
        const uint64_t index_append_bytes = uint64_t(pool_count) * shape.attention.index_width * 2;
        size_t count = 0;
        for (size_t i = 0; i < states.size(); ++i) if (const auto* state = std::get_if<AttentionPair>(&states[i])) {
            const auto& backup = chunked->cache_backup[i];
            const std::array sources{g.slice(state->cache.keys, key_append_offset, key_append_bytes),
                g.slice(state->cache.values, key_append_offset, key_append_bytes),
                pool_count ? g.slice(state->cache.index_keys, index_append_offset, index_append_bytes) : GpuView{}};
            for (size_t k = 0; k < sources.size(); ++k) if (sources[k].owner) {
                const auto saved = g.slice(backup[k], 0, sources[k].bytes);
                save[count] = {sources[k], saved}; restore[count++] = {saved, sources[k]};
            }
        }
        return count;
    }
    uint32_t chunked_ple_slice_cap() const {
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        return (g.prefill_dense_nax() && g.prefill_grouped_nax()) ? 2048u : 512u;
#else
        return 512u;
#endif
    }
    bool chunked_batched_ple(uint32_t steps) const {
        const uint64_t width = uint64_t(shape.residual.branches) * shape.residual.channels;
        const uint32_t ple_cap = chunked_ple_slice_cap();
        return chunked && g.prefill_dense_nax() && steps >= 2 &&
            weights.memory.key.encoding == 8 && weights.memory.value.encoding == 8 &&
            shape.memory.embedding % 64 == 0 &&
            shape.memory.branches == shape.residual.branches && shape.memory.channels == shape.residual.channels &&
            chunked->prefill_core_arena.bytes >= checked_mul(checked_mul(uint64_t(std::min(steps, ple_cap)), 3 * width + shape.memory.channels), 4);
    }
#endif
    void publish_pairs() noexcept {
        for (auto& state : states) {
            if (auto* pair = std::get_if<RecurrentPair>(&state)) std::swap(pair->current, pair->next);
            else { auto& attention = std::get<AttentionPair>(state); std::swap(attention.current, attention.next); }
        }
        std::swap(current_memory, next_memory);
    }
    void capture_final_hc(GpuView rows, uint32_t count) {
        if (!final_hc_tap) return;
        auto& tap = *final_hc_tap;
        require(count && count <= tap.capacity, "final HC tap row bound");
        const auto byte_len = checked_mul(checked_mul(uint64_t(count), tap.width), sizeof(float));
        final_hc_candidate_gpu = g.slice(rows, 0, byte_len);
        auto out = std::span(tap.candidate).first(uint64_t(count) * tap.width);
        g.read(final_hc_candidate_gpu, std::as_writable_bytes(out));
        for (auto value : out) require(std::isfinite(value), "nonfinite final HC residual tap");
        final_hc_candidate_host_valid = true;
    }
    void capture_final_hc_gpu_only(GpuView rows, uint32_t count) {
        if (!final_hc_tap) return;
        auto& tap = *final_hc_tap;
        require(count && count <= tap.capacity, "final HC tap row bound");
        const auto byte_len = checked_mul(checked_mul(uint64_t(count), tap.width), sizeof(float));
        final_hc_candidate_gpu = g.slice(rows, 0, byte_len);
        final_hc_candidate_host_valid = false;
    }
    void publish_final_hc(uint32_t count) noexcept {
        if (final_hc_tap) {
            if (final_hc_candidate_host_valid) {
                final_hc_tap->current.swap(final_hc_tap->candidate);
            }
            final_hc_tap->count = count;
            final_hc_host_valid = final_hc_candidate_host_valid;
            if (count && final_hc_candidate_gpu.owner) {
                final_hc_current_gpu = g.slice(final_hc_candidate_gpu, 0, uint64_t(count) * final_hc_tap->width * sizeof(float));
                final_hc_gpu_valid = true;
            } else {
                final_hc_current_gpu = {};
                final_hc_gpu_valid = false;
            }
            final_hc_candidate_gpu = {};
            final_hc_candidate_host_valid = false;
        }
    }
    void ensure_final_hc_host() const {
        if (!final_hc_tap || final_hc_host_valid || final_hc_tap->count == 0) return;
        require(final_hc_gpu_valid && final_hc_current_gpu.owner, "final HC tap has no committed backing");
        auto out = std::span(final_hc_tap->current).first(uint64_t(final_hc_tap->count) * final_hc_tap->width);
        g.read(final_hc_current_gpu, std::as_writable_bytes(out));
        for (auto value : out) require(std::isfinite(value), "nonfinite final HC residual tap");
        final_hc_host_valid = true;
    }
    uint64_t dflash_tap_dispatches() const noexcept {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        return dflash_tap ? dflash_tap->slot_count : 0;
#else
        return 0;
#endif
    }
    void encode_dflash_tap_layer(size_t i, GpuView layer_out, GpuView dest, uint32_t rows) {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (!dflash_tap || i >= dflash_tap->slot_for_layer.size()) return;
        const int32_t slot = dflash_tap->slot_for_layer[i];
        if (slot < 0) return;
        const uint64_t width = uint64_t(shape.residual.branches) * shape.residual.channels;
        const uint64_t out_stride = uint64_t(dflash_tap->slot_count) * dflash_tap->slot_width * sizeof(float);
        g.dflash2_stream_mean(layer_out, dest, rows, dflash_tap->slot_width, uint32_t(slot), dflash_tap->slot_count,
            width * sizeof(float), out_stride);
#else
        static_cast<void>(i); static_cast<void>(layer_out); static_cast<void>(dest); static_cast<void>(rows);
#endif
    }
    void publish_dflash_tap(GpuView view, uint32_t rows) noexcept {
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (!dflash_tap) return;
        const uint64_t stride = uint64_t(dflash_tap->slot_count) * dflash_tap->slot_width * sizeof(float);
        dflash_tap->last_capture = rows ? StackDFlashCapture{view, rows, dflash_tap->slot_count, dflash_tap->slot_width, stride}
            : StackDFlashCapture{};
#else
        static_cast<void>(view); static_cast<void>(rows);
#endif
    }
    uint64_t extra_dispatches(const GpuMatrix& w, uint32_t columns, uint32_t rows) const {
        return g.gemv_dispatches(w.encoding, columns, rows) - 1;
    }
    uint64_t residual_extra(const ResidualWeights& w) const {
        const auto width = shape.residual.branches * shape.residual.channels;
        return extra_dispatches(w.down, width, shape.residual.rank) + extra_dispatches(w.up, shape.residual.rank, width) +
            extra_dispatches(w.inject, width, shape.residual.branches);
    }
    uint64_t moe_extra(const MoeWeights& w) const {
        const auto s = shape.experts;
        return extra_dispatches(w.router, s.channels, s.experts) + extra_dispatches(w.shared_gate, s.channels, s.hidden) +
            extra_dispatches(w.shared_up, s.channels, s.hidden) + extra_dispatches(w.shared_down, s.hidden, s.channels) +
            extra_dispatches(w.shared_router, s.channels, 1);
    }
    uint64_t ple_dispatches() const {
        return 8 + extra_dispatches(weights.memory.key, shape.memory.embedding, shape.memory.branches * shape.memory.channels) +
            extra_dispatches(weights.memory.value, shape.memory.embedding, shape.memory.channels);
    }
    uint64_t layer_dispatches(size_t i, uint32_t position, std::optional<bool> ported_override = std::nullopt) const {
        if (const auto* rw = std::get_if<RecurrentWeights>(&weights.layers[i])) {
            const auto& w = rw->attention; const auto s = shape.recurrent;
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
            const bool ported = ported_override.value_or(gdn_ported);
#else
            static_cast<void>(ported_override);
#endif
            const uint64_t planar_saved = is_planar_moe(g, rw->ffn, shape.experts) ? (swiglu_fused ? 2ull : 3ull) : 0ull;
            return 40 - (swiglu_fused ? 2ull : 0ull) - planar_saved
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
                - (ported ? 2ull : 0ull)
#endif
                + residual_extra(rw->attention_read) + residual_extra(rw->ffn_read) + moe_extra(rw->ffn) +
                extra_dispatches(w.qkv, s.channels, uint32_t(gdn_qkv_count(s))) +
                extra_dispatches(w.gate, s.channels, s.value_heads * s.value_width) +
                extra_dispatches(w.alpha, s.channels, s.value_heads) + extra_dispatches(w.beta, s.channels, s.value_heads) +
                extra_dispatches(w.output, s.value_heads * s.value_width, s.channels);
        }
        const auto& aw = std::get<AttentionLayerWeights>(weights.layers[i]); const auto& w = aw.attention; const auto s = shape.attention;
        const uint64_t planar_saved = is_planar_moe(g, aw.ffn, shape.experts) ? (swiglu_fused ? 2ull : 3ull) : 0ull;
        return 28 - (swiglu_fused ? 2ull : 0ull) - planar_saved + qsa_dispatches(shape.attention, position)
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
            + uint64_t(qsa_stage_v8.enabled)
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
            + uint64_t(qsa_channel.enabled)
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
            - (qsa_sparse ? 2ull : 0ull)
#endif
#endif
            + residual_extra(aw.attention_read) + residual_extra(aw.ffn_read) + moe_extra(aw.ffn) +
            extra_dispatches(w.query_gate, s.channels, s.query_heads * s.head_width * 2) +
            extra_dispatches(w.key, s.channels, s.kv_heads * s.head_width) + extra_dispatches(w.value, s.channels, s.kv_heads * s.head_width) +
            extra_dispatches(w.index_query, s.channels, s.index_heads * s.index_width) + extra_dispatches(w.index_key, s.channels, s.index_width) +
            extra_dispatches(w.output, s.query_heads * s.head_width, s.channels);
    }
    uint64_t final_dispatches() const {
        return 6 +
            extra_dispatches(weights.mixer.down, shape.residual.branches * shape.residual.channels, shape.residual.rank) +
            extra_dispatches(weights.mixer.up, shape.residual.rank, shape.residual.branches * shape.residual.channels) +
            extra_dispatches(weights.output, shape.residual.channels, shape.vocabulary);
    }
    uint64_t diagnostic_route_dispatches(uint32_t rows = 1) const {
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        return g.carrier_route_snapshot_dispatches(uint32_t(weights.layers.size()), rows);
#else
        static_cast<void>(rows);
        return 0;
#endif
    }
    uint64_t expected_dispatches() const {
        // Compute from fixed geometry and the current QSA position before encode;
        // never bless the observed dispatch delta as its own expected result.
        uint64_t count = ple_dispatches() + final_dispatches();
        for (size_t i = 0; i < weights.layers.size(); ++i) {
            const auto dispatches = layer_dispatches(i, prefix);
            count += dispatches;
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_QSA_SPARSE)
            if (g.qsa_route_attribution_enabled() && std::holds_alternative<AttentionLayerWeights>(weights.layers[i]))
                count += dispatches - 2;
#endif
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_GDN_PORTED)
            if (g.gdn_route_attribution_enabled() && std::holds_alternative<RecurrentWeights>(weights.layers[i]))
                count += dispatches - 2;
#endif
        }
        return count;
    }
    uint64_t known_dispatches(uint32_t steps) const {
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
        if (chunked) {
            const uint32_t ple_cap = chunked_ple_slice_cap();
            const uint64_t ple_total = chunked_batched_ple(steps)
                ? uint64_t((steps + ple_cap - 1) / ple_cap) * (g.dense_mm_q8_dispatches(shape.memory.embedding) + 6ull)
                : uint64_t(steps) * ple_dispatches();
            uint64_t count = ple_total + final_dispatches();
            for (const auto& layer : weights.layers) {
                if (const auto* w = std::get_if<RecurrentWeights>(&layer)) {
                    const auto layer_count = gdn_prefill_layer_dispatches(g, *w, {shape.residual, shape.recurrent, shape.experts}, steps, chunked->mode);
                    count += layer_count;
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_GDN_PORTED)
                    if (chunked->mode == GdnChunkMode::fused && g.gdn_route_attribution_prefill(prefix, steps))
                        count += layer_count;
#endif
                } else count += attention_prefill_dispatches(g, std::get<AttentionLayerWeights>(layer),
                    {shape.residual, shape.attention, shape.experts}, prefix, steps);
            }
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
            if (g.route_attribution_enabled() && prefix == 0 && steps >= 2) {
                require(shape.residual.branches == 4 && shape.residual.channels == 2560 && shape.residual.rank == 320 && shape.experts.experts == 512 && shape.experts.selected == 10, "route attribution stack geometry");
                const auto layers = weights.layers.size();
                const auto layer_count = count - ple_total - final_dispatches();
                // Two HC-down scalar projections become staged in each shadow layer, then a separate router and n top-k calls inspect the exact layer input.
                count += layer_count - layers * 2 * (steps - 1) + layers * (1 + steps);
            }
#endif
            return count;
        }
#endif
        return legacy_known_dispatches(steps);
    }
    uint64_t recurrent_known_layer_dispatches(size_t i, uint32_t steps, bool verification_rows, bool ported) const {
        static_cast<void>(verification_rows);
        [[maybe_unused]] const uint32_t chunks = g.q8_many_chunks(steps);
        const auto& w = std::get<RecurrentWeights>(weights.layers[i]);
        const uint64_t saved = ported ? (steps ? uint64_t(steps - 1) * 3 : 0) : gdn_core_elementwise_saved_dispatches(steps);
        uint64_t count = uint64_t(steps) * layer_dispatches(i, prefix, ported) -
            2 * uint64_t(steps - 1) - 3 * uint64_t(steps - chunks) - saved;
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        if (g.q8_three_rows_enabled() && steps == 3 && (verification_rows || known_capacity != 512))
            count -= moe_q8_three_rows_saved_dispatches(g, w.ffn, shape.experts);
#endif
        if (residual_reuse
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
            || known_hc_rows
#endif
            ) for (const auto& read : {w.attention_read, w.ffn_read})
            count -= (uint64_t(read.down.encoding == 8) + uint64_t(read.up.encoding == 8)) * (steps - chunks);
        return count;
    }
    uint64_t legacy_known_dispatches(uint32_t steps, bool verification_rows = false) const {
        static_cast<void>(verification_rows); // Ordinary builds may omit both scheduling extensions.
        // Q8 projections and their estimates use the same admitted physical widths.
        // Verifier rows use their own capacity8 scalar/shared FFN schedule. Committed capacity512 rows retain their separate per-row/grouped encoder.
        [[maybe_unused]] const uint32_t chunks = g.q8_many_chunks(steps);
        uint64_t count = uint64_t(steps) * ple_dispatches() + final_dispatches();
        for (size_t i = 0; i < weights.layers.size(); ++i) {
            if (std::holds_alternative<RecurrentWeights>(weights.layers[i])) {
                const bool ported =
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
                    gdn_ported;
#else
                    false;
#endif
                count += recurrent_known_layer_dispatches(i, steps, verification_rows, ported);
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_GDN_PORTED)
                if (verification_rows && g.gdn_route_attribution_enabled())
                    count += recurrent_known_layer_dispatches(i, steps, verification_rows, true);
#endif
            } else {
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
                if (known_qsa_rows_memory.enabled) {
                    count += attention_tile_dispatches(g, std::get<AttentionLayerWeights>(weights.layers[i]), prefix,
                        steps, {shape.residual, shape.attention, shape.experts}, AttentionTileMode::core_and_q8,
                        AttentionTileHcMode::combined);
                    continue;
                }
#endif
                for (uint32_t t = 0; t < steps; ++t) count += layer_dispatches(i, prefix + t);
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
                if (known_hc_rows) {
                    const auto& w = std::get<AttentionLayerWeights>(weights.layers[i]);
                    for (const auto& read : {w.attention_read, w.ffn_read})
                        count -= (uint64_t(read.down.encoding == 8) + uint64_t(read.up.encoding == 8)) * (steps - chunks);
                }
#endif
            }
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
            if (grouped_known_enabled && !verification_rows) std::visit([&](const auto& w) {
                // Ordinary FFN: fourteen base dispatches (twelve with fused SwiGLU) plus split projections.
                const uint64_t planar_saved = is_planar_moe(g, w.ffn, shape.experts) ? (swiglu_fused ? 2ull : 3ull) : 0ull;
                count -= uint64_t(steps) * ((swiglu_fused ? 12ull : 14ull) - planar_saved + moe_extra(w.ffn));
                count += moe_many_dispatches(g, w.ffn, shape.experts, steps);
            }, weights.layers[i]);
#endif
        }
        return count;
    }
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    uint64_t verification8_dispatches(uint32_t steps) const {
        auto count = legacy_known_dispatches(steps, true);
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
        // Sampled verification may explicitly compose with known-QSA prefill.
        // Its encoding uses its OWN row policy/scratch, not that prefill route.
        if (known_qsa_rows_memory.enabled) for (size_t i = 0; i < weights.layers.size(); ++i)
            if (const auto* w = std::get_if<AttentionLayerWeights>(&weights.layers[i])) {
                count -= attention_tile_dispatches(g, *w, prefix, steps,
                    {shape.residual, shape.attention, shape.experts}, AttentionTileMode::core_and_q8,
                    AttentionTileHcMode::combined);
                for (uint32_t t = 0; t < steps; ++t) count += layer_dispatches(i, prefix + t);
            }
#endif
        const auto mode = verification8->qsa_rows_memory.config.mode;
        if (mode != QsaVerifyRowsMode::off) for (size_t i = 0; i < weights.layers.size(); ++i)
            if (const auto* w = std::get_if<AttentionLayerWeights>(&weights.layers[i])) {
                for (uint32_t t = 0; t < steps; ++t) count -= layer_dispatches(i, prefix + t);
                const auto tile_count = attention_tile_dispatches(g, *w, prefix, steps, {shape.residual, shape.attention, shape.experts},
                    mode == QsaVerifyRowsMode::core_only ? AttentionTileMode::core_only : AttentionTileMode::core_and_q8,
                    AttentionTileHcMode::combined);
                count += tile_count;
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_QSA_SPARSE)
                if (g.qsa_route_attribution_enabled()) count += tile_count - 2;
#endif
            }
        return count;
    }
#endif
#if defined(TOKSPUR_PRIVATE_VERIFY_EIGHT) && defined(TOKSPUR_PRIVATE_GROUPED_MM_F32)
    void prepare_stack_verify_nax() {
        std::vector<std::pair<uint32_t, PrefillWeightLayout>> moe_layouts;
        const auto add_mat = [&](const GpuMatrix& mat) {
            if (g.planar_matrix_shape(mat.view).has_value()) return;
            const bool ok = mat.encoding == 7 || mat.encoding == 8 || mat.encoding == 12 || mat.encoding == 13;
            if (!ok) return;
            const auto pair = std::pair{mat.encoding, g.prefill_weight_layout(mat.view)};
            if (std::find(moe_layouts.begin(), moe_layouts.end(), pair) == moe_layouts.end()) moe_layouts.push_back(pair);
        };
        for (const auto& layer : weights.layers) {
            std::visit([&](const auto& w) { add_mat(w.ffn.gate); add_mat(w.ffn.up); add_mat(w.ffn.down); }, layer);
        }
        g.prepare_verify_nax(moe_layouts);
        if (g.verify_hc_gdn_fused_enabled()) g.prepare_verify_hc_gdn_fused();
    }
    uint64_t verify_nax_batch_dispatches(uint32_t steps) const {
        const auto& v = *verification8;
        uint64_t count = verify_nax_ple_supported(g, weights.memory, v.recurrent, shape.memory)
            ? verify_nax_ple_dispatches(g, weights.memory, v.recurrent, shape.memory, steps)
            : uint64_t(steps) * ple_dispatches();
        for (size_t i = 0; i < weights.layers.size(); ++i) {
            if (const auto* rw = std::get_if<RecurrentWeights>(&weights.layers[i])) {
                const bool ported =
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
                    gdn_ported;
#else
                    false;
#endif
                count += verify_nax_recurrent_layer_dispatches(g, *rw, {shape.residual, shape.recurrent, shape.experts}, steps, ported);
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_GDN_PORTED)
                if (g.gdn_route_attribution_enabled())
                    count += verify_nax_recurrent_layer_dispatches(g, *rw, {shape.residual, shape.recurrent, shape.experts}, steps, true);
#endif
            } else {
                const auto& aw = std::get<AttentionLayerWeights>(weights.layers[i]);
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
                const auto mode = v.qsa_rows_memory.config.mode;
                if (mode != QsaVerifyRowsMode::off) {
                    const uint64_t tile_count = mode == QsaVerifyRowsMode::core_and_q8
                        ? verify_nax_attention_layer_dispatches(g, aw, {shape.residual, shape.attention, shape.experts}, prefix, steps)
                        : attention_tile_dispatches(g, aw, prefix, steps, {shape.residual, shape.attention, shape.experts},
                            AttentionTileMode::core_only, AttentionTileHcMode::combined);
                    count += tile_count;
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_QSA_SPARSE)
                    if (g.qsa_route_attribution_enabled()) count += tile_count - 2;
#endif
                    continue;
                }
#endif
                for (uint32_t t = 0; t < steps; ++t) count += layer_dispatches(i, prefix + t);
            }
        }
        if (verify_nax_final_supported(g, weights.mixer, weights.output, v.recurrent.residual, shape.residual, shape.vocabulary)) {
            count += verify_nax_final_dispatches(g, weights.mixer, weights.output, v.recurrent.residual, shape.residual, shape.vocabulary, steps);
        } else {
            count += uint64_t(steps) * (final_dispatches() - 1ull) + g.q8_many_chunks(steps);
        }
        return count;
    }
#endif
    GpuView known_row(GpuView view, uint64_t width, uint32_t row) const {
        return g.slice(view, checked_mul(checked_mul(row, width), 4), checked_mul(width, 4));
    }
    static GpuView known_destination(GpuView next, GpuView work, uint32_t row, uint32_t steps) {
        // With either odd or even N, final state lands in next. Accepted state
        // is NEVER reused as scratch, even though ordinary steps swap pairs.
        return (row + steps) % 2 ? next : work;
    }
    void encode_known_layer(size_t i, uint32_t steps) {
        TOKSPUR_OPERATOR_CONTEXT(g,"prefill_layer",int32_t(i),prefix,steps);
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        g.set_task4_profile_tag({Task4ProfilePhase::prefill_layer, int32_t(i), prefix, steps});
#endif
        auto input = i % 2 ? known_second : known_streams;
        const auto output = i % 2 ? known_streams : known_second;
        const auto width = uint64_t(shape.residual.branches) * shape.residual.channels;
        if (i == shape.memory_layer) {
            auto history = current_memory;
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
            if (chunked_batched_ple(steps)) {
                const auto& w = weights.memory;
                const uint32_t c = shape.memory.channels, emb = shape.memory.embedding;
                const float eps = policy.attention.rms_epsilon;
                const uint32_t ple_cap = chunked_ple_slice_cap();
                const uint32_t num_slices = (steps + ple_cap - 1) / ple_cap;
                for (uint32_t s_idx = 0; s_idx < num_slices; ++s_idx) {
                    const uint32_t offset = s_idx * ple_cap, chunk = std::min(ple_cap, steps - offset);
                    const uint64_t w_bytes = uint64_t(chunk) * width * 4, c_bytes = uint64_t(chunk) * c * 4;
                    const auto buf0 = g.slice(chunked->prefill_core_arena, 0, w_bytes);
                    const auto buf1 = g.slice(chunked->prefill_core_arena, w_bytes, w_bytes);
                    const auto buf2 = g.slice(chunked->prefill_core_arena, 2 * w_bytes, w_bytes);
                    const auto val_buf = g.slice(chunked->prefill_core_arena, 3 * w_bytes, c_bytes);
                    const auto emb_slice = g.slice(known_embeddings, uint64_t(offset) * emb * 4, uint64_t(chunk) * emb * 4);
                    const auto in_slice = g.slice(input, uint64_t(offset) * width * 4, w_bytes);
                    const auto out_slice = g.slice(known_memory_streams, uint64_t(offset) * width * 4, w_bytes);
                    const GpuRowsF32 emb_rows{emb_slice, chunk, emb, uint64_t(emb) * 4};
                    const GpuRowsF32 in_rows{in_slice, chunk, uint32_t(width), width * 4};
                    const GpuRowsF32 buf0_rows{buf0, chunk, uint32_t(width), width * 4};
                    const GpuRowsF32 buf1_rows{buf1, chunk, uint32_t(width), width * 4};
                    const GpuRowsF32 buf2_rows{buf2, chunk, uint32_t(width), width * 4};
                    const GpuRowsF32 val_rows{val_buf, chunk, c, uint64_t(c) * 4};
                    g.dense_mm_q8(g.prefill_weight_layout(w.key.view), w.key.view, emb_rows, buf0_rows);
                    g.dense_mm_q8(g.prefill_weight_layout(w.value.view), w.value.view, emb_rows, val_rows, g.can_reuse_nax_dense_bf16(emb_rows));
                    g.rms_rows(buf0_rows, w.key_gains, buf1_rows, c, eps, w.key_gains.bytes == uint64_t(c) * 4);
                    g.rms_rows(in_rows, w.query_gains, buf0_rows, c, eps, w.query_gains.bytes == uint64_t(c) * 4);
                    g.ple_gate_rows(buf0, buf1, val_buf, buf2, shape.memory.branches, c, chunk, shape.memory.gate_transform);
                    g.rms_rows(buf2_rows, w.conv_gains, buf0_rows, c, eps, w.conv_gains.bytes == uint64_t(c) * 4);
                    const auto next = known_destination(next_memory, known_memory_work, s_idx, num_slices);
                    g.ple_conv_write_rows(history, buf0, w.convolution, in_slice, buf2, next, out_slice,
                        uint32_t(width), shape.memory.kernel, shape.memory.dilation, chunk, width * 4, width * 4);
                    history = next;
                }
            } else
#endif
            for (uint32_t t = 0; t < steps; ++t) {
                const auto next = known_destination(next_memory, known_memory_work, t, steps);
                encode_ple(g, weights.memory, p, known_row(known_embeddings, shape.memory.embedding, t),
                    known_row(input, width, t), history, next, known_row(known_memory_streams, width, t),
                    shape.memory, policy.attention.rms_epsilon);
                history = next;
            }
            input = known_memory_streams;
        }
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
        if (chunked) {
            const GpuRowsF32 in{input, steps, uint32_t(width), width * 4}, out{output, steps, uint32_t(width), width * 4};
            const auto encode_layer = [&] {
            if (const auto* w = std::get_if<RecurrentWeights>(&weights.layers[i])) {
                const auto& state = std::get<RecurrentPair>(states[i]);
                encode_gdn_prefill_layer(g, *w, chunked->layer, chunked->gdn, chunked->moe, state.current, state.next, in, out,
                    {shape.residual, shape.recurrent, shape.experts}, policy.recurrent, policy.router, policy.normalize_selected,
                    chunked->mode, std::get<GdnPrefillLayouts>(chunked->layouts[i]), chunked->legacy_order,
                    chunked->gdn_work.matrix.owner ? &chunked->gdn_work : nullptr);
            } else {
                const auto& state = std::get<AttentionPair>(states[i]);
                encode_attention_prefill(g, std::get<AttentionLayerWeights>(weights.layers[i]), chunked->layer, chunked->qsa,
                    chunked->moe, state.cache, state.current, state.next, in, out, prefix,
                    {shape.residual, shape.attention, shape.experts}, policy.attention, policy.router, policy.normalize_selected,
                    std::get<AttentionPrefillLayouts>(chunked->layouts[i]), known_index_work);
            }
            };
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
            if (g.route_attribution_enabled() && prefix == 0 && steps >= 2) {
                std::array<GpuView,6> common{g.slice(input,0,uint64_t(steps)*width*4)};
                std::array<GraphCopy,3> restore{};size_t restore_count=0;
                if (const auto* state=std::get_if<RecurrentPair>(&states[i])) {common[1]=state->current.matrix;common[2]=state->current.convolution;}
                else {
                    const auto& attention_state=std::get<AttentionPair>(states[i]);common[1]=attention_state.current;
                    const uint64_t kv_stride = uint64_t(shape.attention.kv_heads) * shape.attention.head_width * 2;
                    const uint64_t key_append_offset = uint64_t(prefix) * kv_stride, key_append_bytes = uint64_t(steps) * kv_stride;
                    const uint32_t pool_count = (prefix + steps) / shape.attention.compression - prefix / shape.attention.compression;
                    const uint64_t index_append_offset = uint64_t(prefix / shape.attention.compression) * shape.attention.index_width * 2;
                    const uint64_t index_append_bytes = uint64_t(pool_count) * shape.attention.index_width * 2;
                    const std::array parts{g.slice(attention_state.cache.keys,key_append_offset,key_append_bytes),g.slice(attention_state.cache.values,key_append_offset,key_append_bytes),pool_count?g.slice(attention_state.cache.index_keys,index_append_offset,index_append_bytes):GpuView{}};
                    for(size_t k=0;k<parts.size();++k)if(parts[k].owner){common[2+k]=parts[k];restore[restore_count++]={g.slice(chunked->cache_backup[i][k],0,parts[k].bytes),parts[k]};}
                }
                g.route_attribution_common(1,common);
                g.route_attribution_phase(2);encode_layer();
                g.route_attribution_common(5,common,std::span(restore).first(restore_count));
                g.route_attribution_phase(3);encode_layer();g.route_attribution_phase(0);
            } else
#endif
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_GDN_PORTED)
            if (chunked->mode == GdnChunkMode::fused && std::holds_alternative<RecurrentWeights>(weights.layers[i]) &&
                g.gdn_route_attribution_prefill(prefix, steps)) {
                encode_gdn_route_attribution(std::get<RecurrentPair>(states[i]),
                    g.slice(input, 0, uint64_t(steps) * width * 4), steps,
                    prefill_residual_input(chunked->layer.residual, shape.residual, steps).view,
                    g.slice(chunked->moe.rows.logits, 0, uint64_t(steps) * shape.experts.experts * 4),
                    g.slice(chunked->moe.rows.indices, 0, uint64_t(steps) * shape.experts.selected * 4),
                    g.slice(chunked->moe.rows.weights, 0, uint64_t(steps) * shape.experts.selected * 4), encode_layer);
            } else
#endif
            encode_layer();
            return;
        }
#endif
        if (const auto* w = std::get_if<RecurrentWeights>(&weights.layers[i])) {
            const auto& state = std::get<RecurrentPair>(states[i]);
            encode_recurrent_layer_tile(g, *w, known_recurrent, state.current, state.next, input, output,
                {shape.residual, shape.recurrent, shape.experts}, policy.recurrent, policy.router,
                policy.normalize_selected, {known_capacity, steps}, residual_reuse
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
                , grouped_known_enabled ? &known_grouped : nullptr, known_hc_rows
#endif
                );
        } else {
            const auto& state = std::get<AttentionPair>(states[i]);
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
            if (known_qsa_rows_memory.enabled) {
                // All input rows already exist. The tile writes each row's KV
                // and compression boundary chronologically, then selects blocks
                // separately for every query. No shared/borrowed selection IDs.
                encode_attention_layer_tile(g, std::get<AttentionLayerWeights>(weights.layers[i]), known_qsa_rows,
                    state.cache, state.current, state.next, input, output, prefix, steps,
                    {shape.residual, shape.attention, shape.experts}, policy.attention, policy.router,
                    policy.normalize_selected, AttentionTileMode::core_and_q8, AttentionTileHcMode::combined);
                return;
            }
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
            if (known_capacity == 512) {
                const auto& w = std::get<AttentionLayerWeights>(weights.layers[i]);
                auto sum = state.current;
                if (known_hc_rows) encode_known_residual_rows512(g, w.attention_read, known_recurrent, input,
                    {shape.residual, shape.recurrent, shape.experts}, {known_capacity, steps}, policy.attention.rms_epsilon);
                for (uint32_t t = 0; t < steps; ++t) {
                    const auto next = known_destination(state.next, known_index_work, t, steps);
                    const auto s = recurrent_tile_token(g, known_recurrent,
                        {shape.residual, shape.recurrent, shape.experts}, known_capacity, t);
                    if (!known_hc_rows) encode_residual_read(g, w.attention_read, s.residual, known_row(input, width, t),
                        shape.residual, policy.attention.rms_epsilon);
                    encode_qsa(g, w.attention, a.attention, state.cache, sum, next, s.residual.input,
                        s.attention_output, prefix + t, shape.attention, policy.attention);
                    g.residual_write(known_row(input, width, t), s.attention_output, s.residual.scales, s.intermediate_streams);
                    if (!known_hc_rows) encode_residual_read(g, w.ffn_read, s.residual, s.intermediate_streams,
                        shape.residual, policy.attention.rms_epsilon);
                    sum = next;
                }
                // The complete causal QSA loop above consumes every attention
                // scale. Only now may the FFN read reuse that capacity scratch.
                if (known_hc_rows) encode_known_residual_rows512(g, w.ffn_read, known_recurrent,
                    known_recurrent.intermediate_streams, {shape.residual, shape.recurrent, shape.experts},
                    {known_capacity, steps}, policy.attention.rms_epsilon);
                if (grouped_known_enabled) encode_moe_many(g, w.ffn, known_grouped, known_recurrent.residual.input,
                    known_recurrent.ffn_output, shape.experts, policy.router, policy.normalize_selected, steps);
                else for (uint32_t t = 0; t < steps; ++t) {
                    const auto s = recurrent_tile_token(g, known_recurrent,
                        {shape.residual, shape.recurrent, shape.experts}, known_capacity, t);
                    encode_moe(g, w.ffn, s.moe, s.residual.input, s.ffn_output, shape.experts, policy.router, policy.normalize_selected
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                        , t
#endif
                        );
                }
                for (uint32_t t = 0; t < steps; ++t) {
                    const auto s = recurrent_tile_token(g, known_recurrent,
                        {shape.residual, shape.recurrent, shape.experts}, known_capacity, t);
                    g.residual_write(s.intermediate_streams, s.ffn_output, s.residual.scales, known_row(output, width, t));
                }
                return;
            }
#endif
            auto sum = state.current;
            for (uint32_t t = 0; t < steps; ++t) {
                const auto next = known_destination(state.next, known_index_work, t, steps);
                // Each call appends just its causal KV row/new compression
                // block. All accepted rows below the original prefix survive.
                encode_attention_layer(g, std::get<AttentionLayerWeights>(weights.layers[i]), a,
                    state.cache, sum, next, known_row(input, width, t), known_row(output, width, t),
                    prefix + t, {shape.residual, shape.attention, shape.experts}, policy.attention,
                    policy.router, policy.normalize_selected
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                    , t
#endif
                    );
                sum = next;
            }
        }
    }
    void encode_verification_layer(size_t i) {
        TOKSPUR_OPERATOR_CONTEXT(g,"verification_layer",int32_t(i),prefix,2);
        auto& v = *verification;
        auto input = i % 2 ? v.second : v.streams;
        const auto output = i % 2 ? v.streams : v.second;
        const auto width = uint64_t(shape.residual.branches) * shape.residual.channels;
        if (i == shape.memory_layer) {
            encode_ple(g, weights.memory, p, known_row(v.embeddings, shape.memory.embedding, 0),
                known_row(input, width, 0), current_memory, v.first_memory, known_row(v.memory_streams, width, 0),
                shape.memory, policy.attention.rms_epsilon);
            encode_ple(g, weights.memory, p, known_row(v.embeddings, shape.memory.embedding, 1),
                known_row(input, width, 1), v.first_memory, next_memory, known_row(v.memory_streams, width, 1),
                shape.memory, policy.attention.rms_epsilon);
            input = v.memory_streams;
        }
        if (const auto* w = std::get_if<RecurrentWeights>(&weights.layers[i])) {
            const auto& state = std::get<RecurrentPair>(states[i]);
            auto scratch = v.recurrent; scratch.gdn = bind_gdn_core_tape(scratch.gdn, v.tapes[i]);
            encode_recurrent_layer_tile(g, *w, scratch, state.current, state.next, input, output,
                {shape.residual, shape.recurrent, shape.experts}, policy.recurrent, policy.router,
                policy.normalize_selected, {2, 2}, residual_reuse);
        } else {
            const auto& state = std::get<AttentionPair>(states[i]);
            for (uint32_t t = 0; t < 2; ++t)
                encode_attention_layer(g, std::get<AttentionLayerWeights>(weights.layers[i]), a, state.cache,
                    t ? v.first_sums[i] : state.current, t ? state.next : v.first_sums[i],
                    known_row(input, width, t), known_row(output, width, t), prefix + t,
                    {shape.residual, shape.attention, shape.experts}, policy.attention, policy.router, policy.normalize_selected
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                    , t
#endif
                    );
        }
    }
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_QSA_SPARSE)
    template<typename Fn>
    void encode_qsa_route_attribution(const AttentionPair& state, GpuView active_input, uint32_t steps,
        GpuView ffn_input, GpuView router_logits, GpuView selected_ids, GpuView selected_weights, Fn&& encode_pass) {
        const uint64_t kv_row = uint64_t(shape.attention.kv_heads) * shape.attention.head_width * 2;
        const uint64_t index_row = uint64_t(shape.attention.index_width) * 2;
        const uint32_t compression = shape.attention.compression;
        const uint32_t first_block = prefix / compression, end_block = (prefix + steps) / compression, pool_blocks = end_block - first_block;
        const auto backup = g.qsa_route_attribution_backup();
        const std::array parts{
            g.slice(state.cache.keys, uint64_t(prefix) * kv_row, uint64_t(steps) * kv_row),
            g.slice(state.cache.values, uint64_t(prefix) * kv_row, uint64_t(steps) * kv_row),
            pool_blocks ? g.slice(state.cache.index_keys, uint64_t(first_block) * index_row, uint64_t(pool_blocks) * index_row) : GpuView{}
        };
        std::array<GpuView, 6> common{active_input, state.current};
        std::array<GraphCopy, 3> save{}, restore{}; size_t copy_count = 0;
        for (size_t k = 0; k < parts.size(); ++k) if (parts[k].owner) {
            require(parts[k].bytes <= backup[k].bytes, "QSA route attribution backup capacity");
            common[2 + k] = parts[k];
            const auto slot = g.slice(backup[k], 0, parts[k].bytes);
            save[copy_count] = {parts[k], slot}; restore[copy_count] = {slot, parts[k]}; ++copy_count;
        }
        QsaRouteAttributionRecord common_rec{1, steps, shape.experts.channels, shape.experts.experts, shape.experts.selected, 0, -1, common};
        g.qsa_route_attribution_emit(common_rec, std::span(save).first(copy_count));
        g.qsa_route_attribution_phase(2); encode_pass();
        g.qsa_route_attribution_emit({2, steps, shape.experts.channels, shape.experts.experts, shape.experts.selected, 0, -1,
            {ffn_input, router_logits, selected_ids, selected_weights, {}, {}}});
        common_rec.phase = 5; g.qsa_route_attribution_emit(common_rec, std::span(restore).first(copy_count));
        g.qsa_route_attribution_phase(3); encode_pass();
        g.qsa_route_attribution_emit({3, steps, shape.experts.channels, shape.experts.experts, shape.experts.selected, 0, -1,
            {ffn_input, router_logits, selected_ids, selected_weights, {}, {}}});
        g.qsa_route_attribution_phase(0);
    }
#endif
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_GDN_PORTED)
    template<typename Fn>
    void encode_gdn_route_attribution(const RecurrentPair& state, GpuView active_input, uint32_t steps,
        GpuView ffn_input, GpuView router_logits, GpuView selected_ids, GpuView selected_weights, Fn&& encode_pass) {
        GdnRouteAttributionRecord common_rec{1, steps, shape.experts.channels, shape.experts.experts, shape.experts.selected, 0, -1,
            {active_input, state.current.matrix, state.current.convolution, {}, {}, {}}};
        g.gdn_route_attribution_emit(common_rec);
        g.gdn_route_attribution_phase(2); encode_pass();
        g.gdn_route_attribution_emit({2, steps, shape.experts.channels, shape.experts.experts, shape.experts.selected, 0, -1,
            {ffn_input, router_logits, selected_ids, selected_weights, {}, {}}});
        common_rec.phase = 5; g.gdn_route_attribution_emit(common_rec);
        g.gdn_route_attribution_phase(3); encode_pass();
        g.gdn_route_attribution_emit({3, steps, shape.experts.channels, shape.experts.experts, shape.experts.selected, 0, -1,
            {ffn_input, router_logits, selected_ids, selected_weights, {}, {}}});
        g.gdn_route_attribution_phase(0);
    }
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    void encode_verification8_layer(size_t i, uint32_t steps) {
        TOKSPUR_OPERATOR_CONTEXT(g,"verification_layer",int32_t(i),prefix,steps);
        auto& v = *verification8;
        auto input = i % 2 ? v.second : v.streams;
        const auto output = i % 2 ? v.streams : v.second;
        const auto width = uint64_t(shape.residual.branches) * shape.residual.channels;
        if (i == shape.memory_layer) {
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            if (g.verify_nax_active() && verify_nax_ple_supported(g, weights.memory, v.recurrent, shape.memory)) {
                encode_verify_nax_ple(g, weights.memory, p, v.recurrent, v.embeddings, input,
                    current_memory, v.prefix_memory, next_memory, v.memory_streams,
                    shape.memory, steps, policy.attention.rms_epsilon);
            } else
#endif
            {
                auto previous = current_memory;
                for (uint32_t t = 0; t < steps; ++t) {
                    const auto next = t + 1 == steps ? next_memory : known_row(v.prefix_memory, current_memory.bytes / 4, t);
                    encode_ple(g, weights.memory, p, known_row(v.embeddings, shape.memory.embedding, t),
                        known_row(input, width, t), previous, next, known_row(v.memory_streams, width, t),
                        shape.memory, policy.attention.rms_epsilon);
                    previous = next;
                }
            }
            input = v.memory_streams;
        }
        if (const auto* w = std::get_if<RecurrentWeights>(&weights.layers[i])) {
            const auto& state = std::get<RecurrentPair>(states[i]);
            auto scratch = v.recurrent; scratch.gdn = bind_gdn_core_tape(scratch.gdn, v.tapes[i]);
            const auto encode_tile = [&] {
                encode_recurrent_layer_tile(g, *w, scratch, state.current, state.next, input, output,
                    {shape.residual, shape.recurrent, shape.experts}, policy.recurrent, policy.router,
                    policy.normalize_selected, {8, steps}, residual_reuse
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
                    , nullptr, false
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
                    , g.verify_nax_active() ? &v.nax_moe : nullptr
#endif
                    );
            };
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_GDN_PORTED)
            if (g.gdn_route_attribution_enabled()) {
                encode_gdn_route_attribution(state, g.slice(input, 0, uint64_t(steps) * width * 4), steps,
                    g.slice(v.recurrent.residual.input, 0, uint64_t(steps) * shape.experts.channels * 4),
                    g.slice(v.recurrent.moe.logits, 0, uint64_t(steps) * shape.experts.experts * 4),
                    g.slice(v.recurrent.moe.indices, 0, uint64_t(steps) * shape.experts.selected * 4),
                    g.slice(v.recurrent.moe.weights, 0, uint64_t(steps) * shape.experts.selected * 4), encode_tile);
                return;
            }
#endif
            encode_tile();
        } else {
            const auto& state = std::get<AttentionPair>(states[i]);
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
            if (v.qsa_rows_memory.config.mode != QsaVerifyRowsMode::off) {
                const AttentionTileScratch scratch{{v.recurrent.residual, v.qsa_rows, v.recurrent.moe,
                    v.recurrent.intermediate_streams, v.recurrent.attention_output, v.recurrent.ffn_output},
                    v.prefix_sums[i], v.qsa_scores, v.qsa_factors};
                const auto encode_tile = [&] {
                    const auto attn_mode = v.qsa_rows_memory.config.mode == QsaVerifyRowsMode::core_only
                        ? AttentionTileMode::core_only : AttentionTileMode::core_and_q8;
                    encode_attention_layer_tile(g, std::get<AttentionLayerWeights>(weights.layers[i]), scratch,
                        state.cache, state.current, state.next, input, output, prefix, steps,
                        {shape.residual, shape.attention, shape.experts}, policy.attention, policy.router,
                        policy.normalize_selected, attn_mode, AttentionTileHcMode::combined
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
                        , (g.verify_nax_active() && attn_mode == AttentionTileMode::core_and_q8) ? &v.nax_moe : nullptr
#endif
                        );
                };
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_QSA_SPARSE)
                if (g.qsa_route_attribution_enabled()) {
                    encode_qsa_route_attribution(state, g.slice(input, 0, uint64_t(steps) * width * 4), steps,
                        g.slice(v.recurrent.residual.input, 0, uint64_t(steps) * shape.experts.channels * 4),
                        g.slice(v.recurrent.moe.logits, 0, uint64_t(steps) * shape.experts.experts * 4),
                        g.slice(v.recurrent.moe.indices, 0, uint64_t(steps) * shape.experts.selected * 4),
                        g.slice(v.recurrent.moe.weights, 0, uint64_t(steps) * shape.experts.selected * 4), encode_tile);
                    return;
                }
#endif
                encode_tile();
                return;
            }
#endif
            auto previous = state.current;
            for (uint32_t t = 0; t < steps; ++t) {
                const auto next = t + 1 == steps ? state.next : known_row(v.prefix_sums[i], shape.attention.index_width, t);
                encode_attention_layer(g, std::get<AttentionLayerWeights>(weights.layers[i]), a, state.cache,
                    previous, next, known_row(input, width, t), known_row(output, width, t), prefix + t,
                    {shape.residual, shape.attention, shape.experts}, policy.attention, policy.router, policy.normalize_selected
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
                    , t
#endif
                    );
                previous = next;
            }
        }
    }
#endif
    GpuView layer_input(size_t i) const { return i % 2 ? second_streams : token_streams; }
    GpuView layer_output(size_t i) const { return i % 2 ? token_streams : second_streams; }
    void encode_layer(size_t i, bool reverse = false) {
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        g.set_carrier_diagnostic_layer(uint32_t(i));
#endif
        TOKSPUR_OPERATOR_CONTEXT(g,"ordinary_layer",int32_t(i),prefix,1);
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
        g.set_task4_profile_tag({Task4ProfilePhase::decode_layer, int32_t(i), prefix, 1});
#endif
        auto input = layer_input(i); const auto output = layer_output(i);
        if (i == shape.memory_layer) {
            encode_ple(g, weights.memory, p, embedding, input, reverse ? next_memory : current_memory,
                reverse ? current_memory : next_memory, memory_streams, shape.memory, policy.attention.rms_epsilon);
            input = memory_streams;
        }
        if (const auto* w = std::get_if<RecurrentWeights>(&weights.layers[i])) {
            const auto& state = std::get<RecurrentPair>(states[i]);
            const auto encode_single = [&] {
                encode_recurrent_layer(g, *w, r, reverse ? state.next : state.current, reverse ? state.current : state.next,
                    input, output, {shape.residual, shape.recurrent, shape.experts}, policy.recurrent, policy.router, policy.normalize_selected);
            };
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_GDN_PORTED)
            if (g.gdn_route_attribution_enabled()) {
                require(!reverse, "GDN route attribution cannot use static reversed bindings");
                encode_gdn_route_attribution(state, input, 1, r.residual.input, r.moe.logits, r.moe.indices, r.moe.weights, encode_single);
                return;
            }
#endif
            encode_single();
        } else {
            require(!reverse, "sparse attention cannot use static reversed bindings");
            const auto& state = std::get<AttentionPair>(states[i]);
            const auto encode_single = [&] {
                encode_attention_layer(g, std::get<AttentionLayerWeights>(weights.layers[i]), a, state.cache, state.current, state.next,
                    input, output, prefix, {shape.residual, shape.attention, shape.experts}, policy.attention, policy.router, policy.normalize_selected);
            };
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && defined(TOKSPUR_PRIVATE_QSA_SPARSE)
            if (g.qsa_route_attribution_enabled()) {
                encode_qsa_route_attribution(state, input, 1, a.residual.input, a.moe.logits, a.moe.indices, a.moe.weights, encode_single);
                return;
            }
#endif
            encode_single();
        }
    }
    void prepare_replay() {
        // One-time immutable command preparation. Chunk a longer generic run if
        // necessary to retain the graph's 1024-command cap; the 48-layer target
        // has twelve three-layer runs, with 143/135 commands at split16/block3.
        try {
            for (size_t i = 0; i < weights.layers.size();) {
                if (!std::holds_alternative<RecurrentWeights>(weights.layers[i])) { ++i; continue; }
                const auto begin = i; uint64_t count = 0;
                while (i < weights.layers.size() && std::holds_alternative<RecurrentWeights>(weights.layers[i])) {
                    const auto extra = layer_dispatches(i, prefix) + (i == shape.memory_layer ? ple_dispatches() : 0);
                    require(extra && extra <= 1024, "recurrent layer exceeds static command capacity");
                    if (count && extra > 1024 - count) break;
                    count += extra; ++i;
                }
                RecurrentRun run{begin, i, uint32_t(count), {}};
                for (uint32_t orientation = 0; orientation < 2; ++orientation) {
                    g.begin_capture(run.dispatches);
                    for (size_t layer = begin; layer < i; ++layer) encode_layer(layer, bool(orientation));
                    const auto capture = g.finish_capture();
                    // reserve() above makes publication nonallocating after the
                    // graph publishes a capture handle, including partial failure.
                    captures.push_back(capture); run.captures[orientation] = capture;
                }
                runs.push_back(run);
            }
        } catch (...) { try { g.abort(); } catch (const FormatError&) {} throw; }
    }
    uint32_t physical_orientation() const {
        // reset() swaps physical buffers even at position zero. Prefix parity
        // is therefore not a binding identity and must never select a capture.
        for (uint32_t i = 0; i < memory_orientations.size(); ++i)
            if (same_view(current_memory, memory_orientations[i]) && same_view(next_memory, memory_orientations[i ^ 1])) return i;
        throw FormatError("accepted physical stack orientation is invalid");
    }
};
TextStack::TextStack(MetalGraph& g, StackWeights w, StackShape s, StackPolicy p, NgramSpec n, StackExecution execution, uint32_t known_token_capacity, bool known_residual_reuse)
    : impl_(std::make_unique<Impl>(g, std::move(w), s, p, std::move(n), execution, known_token_capacity, known_residual_reuse)) {}
TextStack::~TextStack() = default;
uint32_t TextStack::position() const { impl_->check_thread(); return impl_->prefix; }
StackExecution TextStack::execution() const { impl_->check_thread(); return impl_->execution; }
bool TextStack::replay_prepared() const { impl_->check_thread(); return impl_->replay_prepared; }
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
bool TextStack::chunked_prefill_enabled() const { impl_->check_thread(); return bool(impl_->chunked); }
uint64_t TextStack::chunked_prefill_bytes() const { impl_->check_thread(); return impl_->chunked ? impl_->chunked->charged : 0; }
std::vector<std::byte> TextStack::chunked_cache_snapshot() const {
    const auto& p = *impl_; p.check_thread(); require(bool(p.chunked), "physical cache snapshot requires prepared chunked prefill");
    uint64_t total = 0;
    for (const auto& state : p.states) if (const auto* a = std::get_if<AttentionPair>(&state))
        for (auto v : a->cache.views()) total = checked_add(total, v.bytes);
    require(total <= (8ull << 20), "generated physical cache snapshot exceeds its bound");
    std::vector<std::byte> result; result.reserve(total);
    for (const auto& state : p.states) if (const auto* a = std::get_if<AttentionPair>(&state))
        for (auto v : a->cache.views()) { const auto bytes = p.g.read(v); result.insert(result.end(), bytes.begin(), bytes.end()); }
    return result;
}
void TextStack::prepare_chunked_prefill(GdnChunkMode mode, uint64_t budget, bool legacy_order) {
    auto& p = *impl_; p.check_thread(); p.check_known_policy();
    const bool grouped_nax =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        p.g.prefill_grouped_nax();
#else
        false;
#endif
    const bool nax_2k = p.g.prefill_dense_nax() && grouped_nax;
    const uint64_t max_budget = (p.known_capacity >= 1024 || p.g.prefill_dense_nax() || grouped_nax) ? (1792ull << 20) : (1024ull << 20);
    require(!p.chunked && !p.prefix && (p.known_capacity == 512 || p.known_capacity == 1024 || p.known_capacity == 2048) && p.execution == StackExecution::direct &&
#ifndef TOKSPUR_PRIVATE_HTTP_PREFILL512
        !p.residual_reuse &&
#endif
        !p.known_layer_submit && !p.known_hc_rows && !p.checkpoints && !p.verification && !p.final_hc_tap &&
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        !p.dflash_tap &&
#endif
        uint32_t(mode) <= 1 && budget && budget <= max_budget, "chunked prefill requires an empty exclusive capacity>=512 stack and bounded budget");
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    require(!p.verification8, "chunked prefill preparation precedes verifier resources");
#endif
    require(!legacy_order || mode == GdnChunkMode::fused, "old-order recurrence requires fused chunk scheduling");
#ifndef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    require(!legacy_order, "old-order recurrence is not compiled");
#endif
    uint64_t before = p.g.memory().live_bytes;
    auto store = std::make_unique<ChunkedPrefillStore>(p.g); store->mode = mode; store->legacy_order = legacy_order;
    store->layouts.reserve(p.weights.layers.size()); store->cache_backup.resize(p.weights.layers.size());
    const bool candidate_prefill = p.g.prefill_dense_nax() || grouped_nax;
    const auto prepare_layer = [&](size_t i) {
        if (const auto* w = std::get_if<RecurrentWeights>(&p.weights.layers[i])) {
            const auto l = gdn_prefill_layouts(p.g, *w); prepare_gdn_prefill_layer(p.g, *w, {p.shape.residual, p.shape.recurrent, p.shape.experts}, l, legacy_order);
            store->layouts.emplace_back(l);
        } else {
            const auto& aw = std::get<AttentionLayerWeights>(p.weights.layers[i]); const auto l = attention_prefill_layouts(p.g, aw);
            prepare_attention_prefill(p.g, aw, {p.shape.residual, p.shape.attention, p.shape.experts}, l); store->layouts.emplace_back(l);
        }
    };
    if (candidate_prefill) {
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        p.register_nax_q8_matrices();
#endif
        if (p.g.prefill_dense_nax() && p.weights.memory.key.encoding == 8 && p.weights.memory.value.encoding == 8 && p.shape.memory.embedding % 64 == 0) {
            p.g.prepare_gdn_chunk();
            p.g.prepare_dense_mm_q8(p.g.prefill_weight_layout(p.weights.memory.key.view));
            p.g.prepare_dense_mm_q8(p.g.prefill_weight_layout(p.weights.memory.value.view));
        }
        for (size_t i = 0; i < p.weights.layers.size(); ++i) prepare_layer(i);
        before = p.g.memory().live_bytes;
    }
    const uint32_t prefill_cap = p.known_capacity;
    const uint32_t chunk_cap = nax_2k ? prefill_cap : std::min(prefill_cap, 512u);
    const auto gdn_bytes = gdn_chunk_bytes(p.shape.recurrent, chunk_cap, mode, nax_2k).total;
    const auto qsa_plan = qsa_prefill_plan(0, chunk_cap, chunk_cap, p.shape.attention, nax_2k);
    const uint64_t q_count = p.shape.attention.query_heads * p.shape.attention.head_width, k_count = p.shape.attention.kv_heads * p.shape.attention.head_width, iq_count = p.shape.attention.index_heads * p.shape.attention.index_width;
    const uint64_t pool_rows = (chunk_cap + 3) / 4, stripe_n = std::min(chunk_cap, 32u), tiles = (p.shape.attention.token_budget + 3 + 127) / 128;
    const std::array<uint64_t, 25> qsa_counts{chunk_cap * q_count * 2, chunk_cap * k_count, chunk_cap * k_count, chunk_cap * iq_count, uint64_t(chunk_cap) * p.shape.attention.index_width,
        chunk_cap * q_count, chunk_cap * q_count, chunk_cap * q_count, chunk_cap * k_count, chunk_cap * q_count, chunk_cap * k_count, chunk_cap * iq_count, chunk_cap * iq_count,
        pool_rows * p.shape.attention.index_width, pool_rows * p.shape.attention.index_width, pool_rows * p.shape.attention.index_width, stripe_n * qsa_plan.block_capacity,
        stripe_n * qsa_plan.selection_stride, stripe_n * qsa_plan.selection_stride, stripe_n * qsa_plan.selection_stride, stripe_n * qsa_plan.selection_stride,
        uint64_t(chunk_cap) * qsa_plan.block_budget, stripe_n * p.shape.attention.query_heads * tiles * (p.shape.attention.head_width + 2), chunk_cap * q_count, chunk_cap * q_count};
    uint64_t qsa_bytes = 0;
    for (auto c : qsa_counts) qsa_bytes = checked_add(qsa_bytes, checked_mul(c, 4));
    const uint64_t core_arena_bytes = std::max(gdn_bytes, qsa_bytes);

    store->prefill_core_arena = store->zeros(core_arena_bytes, "prefill core arena");
    store->layer = allocate_prefill_layer_scratch(p.g, p.shape.residual, prefill_cap); store->keep(store->layer.views());
    store->gdn = allocate_gdn_chunk_scratch(p.g, p.shape.recurrent, chunk_cap, mode, store->prefill_core_arena);
    store->qsa = allocate_qsa_prefill_scratch(p.g, p.shape.attention, chunk_cap, store->prefill_core_arena);
    if (prefill_cap > chunk_cap) {
        store->gdn_work = allocate_gdn_state(p.g, p.shape.recurrent);
        store->keep(store->gdn_work.views());
    }
    const uint32_t moe_cap = grouped_nax ? prefill_cap : std::min(prefill_cap, uint32_t(10240 / p.shape.experts.selected));
    store->moe = allocate_moe_mm_many_scratch(p.g, p.shape.experts, moe_cap, grouped_nax); store->keep(store->moe.views());
    for (size_t i = 0; i < p.weights.layers.size(); ++i) {
        if (!candidate_prefill) prepare_layer(i);
        if (!std::holds_alternative<RecurrentWeights>(p.weights.layers[i])) {
            const auto key_bytes = checked_mul(checked_mul(prefill_cap, p.shape.attention.kv_heads * p.shape.attention.head_width), 2);
            const auto pool_bytes = checked_mul(checked_mul(prefill_cap / p.shape.attention.compression + 1, p.shape.attention.index_width), 2);
            store->cache_backup[i] = {store->zeros(key_bytes), store->zeros(key_bytes), store->zeros(pool_bytes)};
        }
    }
    store->charged = p.g.memory().live_bytes - before; require(store->charged <= budget, "chunked prefill exceeds its explicit GPU budget");
    p.g.validate_disjoint(p.owned, store->owned); p.chunked = std::move(store);
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    p.grouped_known_enabled = false;
#endif
}
#endif
uint32_t TextStack::replay_capture_count() const { impl_->check_thread(); return uint32_t(impl_->captures.size()); }
uint32_t TextStack::known_token_capacity() const { impl_->check_thread(); return impl_->known_capacity; }
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
void TextStack::configure_sampled_logit_views(bool enabled) {
    auto& p = *impl_; p.check_thread();
    require(!p.prefix && !p.sampled_verification_prepared, "sampled logit views are fixed before verifier preparation");
    p.sampled_logit_views = enabled;
}
bool TextStack::sampled_logit_views() const { impl_->check_thread(); return impl_->sampled_logit_views; }
void TextStack::configure_sampled_logit_topk(bool enabled) {
    auto& p = *impl_; p.check_thread();
    require(!p.prefix && !p.sampled_verification_prepared, "sampled logit top-k is fixed before verifier preparation");
    p.sampled_logit_topk = enabled;
}
bool TextStack::sampled_logit_topk() const { impl_->check_thread(); return impl_->sampled_logit_topk; }
void TextStack::read_verified_committed_logits(uint32_t committed_1based, std::span<float> destination) {
    auto& p = *impl_; p.check_thread(); p.check_verification8_policy();
    const uint32_t vocabulary = p.shape.vocabulary;
    require(p.sampled_verification_prepared && committed_1based >= 1 && committed_1based <= 8 &&
        destination.size() == vocabulary, "invalid committed verification logit row readback");
    auto& v = *p.verification8;
    if (p.sampled_logit_views) {
        const float* shared = p.g.host_floats(v.logits);
        const auto src = std::span<const float>(shared + uint64_t(committed_1based - 1u) * vocabulary, vocabulary);
        for (float val : src) require(std::isfinite(val), "nonfinite sampled target logits");
        std::copy(src.begin(), src.end(), destination.begin());
    } else {
        p.g.read(p.g.slice(v.logits, uint64_t(committed_1based - 1u) * vocabulary * sizeof(float),
                           uint64_t(vocabulary) * sizeof(float)),
                 std::as_writable_bytes(destination));
        for (float val : destination) require(std::isfinite(val), "nonfinite sampled target logits");
    }
}
#endif
void TextStack::prepare_final_hc_tap(uint64_t budget) {
    auto& p = *impl_; p.check_thread();
    uint32_t tap_capacity = 8; uint64_t tap_budget = maximum_stack_final_hc_tap_budget;
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    if (p.known_capacity >= 512 && p.chunked) {
        tap_capacity = p.known_capacity;
        tap_budget = std::max(p.known_capacity >= 2048 ? (256ull << 20) : (p.known_capacity >= 1024 ? (128ull << 20) : (64ull << 20)), uint64_t(tap_capacity) * 10240 * 4 * 2 + (1ull << 20));
    }
#endif
    require(!p.final_hc_tap && !p.prefix && !p.checkpoints && !p.verification &&
        p.execution == StackExecution::direct && !p.replay_prepared && p.known_capacity <= tap_capacity &&
        budget && budget <= tap_budget,
        "final HC tap requires empty direct capacity<=tap_capacity stack and explicit bounded budget");
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    require(!p.verification8, "final HC tap must precede verifier preparation");
#endif
    static_cast<void>(p.g.last_profile());
    const auto width = checked_mul(p.shape.residual.branches, p.shape.residual.channels);
    const auto minimum = checked_add(sizeof(FinalHcTap), checked_mul(checked_mul(2 * tap_capacity, width), sizeof(float)));
    require(width <= std::numeric_limits<uint32_t>::max() && minimum <= budget,
        "final HC tap host banks exceed explicit budget");
    auto tap = std::make_unique<FinalHcTap>();
    tap->width = uint32_t(width); tap->capacity = tap_capacity;
    tap->current.resize(tap_capacity * width); tap->candidate.resize(tap_capacity * width);
    tap->charged_bytes = sizeof(FinalHcTap) +
        checked_mul(tap->current.capacity() + tap->candidate.capacity(), sizeof(float));
    require(tap->charged_bytes <= budget, "final HC tap actual host capacity exceeds budget");
    p.final_hc_tap = std::move(tap);
    p.final_hc_current_gpu = {}; p.final_hc_candidate_gpu = {};
    p.final_hc_gpu_valid = false; p.final_hc_host_valid = false; p.final_hc_candidate_host_valid = false;
}
void TextStack::configure_gpu_final_hc_tap(bool enabled) {
    auto& p = *impl_; p.check_thread();
    require(bool(p.final_hc_tap) && !p.prefix, "GPU final HC tap requires an empty prepared final HC tap");
    p.final_hc_gpu_mode = enabled;
}
bool TextStack::gpu_final_hc_tap() const {
    const auto& p = *impl_; p.check_thread();
    return p.final_hc_gpu_mode;
}
StackFinalHcTapMemory TextStack::final_hc_tap_memory() const {
    const auto& p = *impl_; p.check_thread();
    return p.final_hc_tap ? StackFinalHcTapMemory{true, p.final_hc_tap->capacity, p.final_hc_tap->width, p.final_hc_tap->charged_bytes}
        : StackFinalHcTapMemory{};
}
uint32_t TextStack::last_final_hc_count() const {
    const auto& p = *impl_; p.check_thread();
    require(bool(p.final_hc_tap), "final HC tap is not prepared");
    return p.final_hc_tap->count;
}
std::span<const float> TextStack::last_final_hc_rows() const {
    auto& p = *impl_; p.check_thread();
    require(bool(p.final_hc_tap), "final HC tap is not prepared");
    p.ensure_final_hc_host();
    return std::span<const float>(p.final_hc_tap->current).first(uint64_t(p.final_hc_tap->count) * p.final_hc_tap->width);
}
std::span<const float> TextStack::last_final_hc_residual() const {
    auto& p = *impl_; p.check_thread();
    require(bool(p.final_hc_tap), "final HC tap is not prepared");
    auto& tap = *p.final_hc_tap;
    require(tap.count > 0, "final HC tap has no committed target row");
    if (!p.final_hc_host_valid) {
        if (tap.count > 8) {
            require(p.final_hc_gpu_valid && p.final_hc_current_gpu.owner, "final HC tap has no committed backing");
            const uint64_t row_bytes = uint64_t(tap.width) * sizeof(float);
            auto last_out = std::span(tap.current).subspan(uint64_t(tap.count - 1) * tap.width, tap.width);
            p.g.read(p.g.slice(p.final_hc_current_gpu, uint64_t(tap.count - 1) * row_bytes, row_bytes),
                     std::as_writable_bytes(last_out));
            for (auto value : last_out) require(std::isfinite(value), "nonfinite final HC residual tap");
            return last_out;
        }
        p.ensure_final_hc_host();
    }
    return std::span<const float>(tap.current).first(uint64_t(tap.count) * tap.width).last(tap.width);
}
bool TextStack::has_last_final_hc_gpu() const {
    const auto& p = *impl_; p.check_thread();
    return p.final_hc_tap && p.final_hc_tap->count > 0 && p.final_hc_gpu_valid && p.final_hc_current_gpu.owner != 0;
}
GpuView TextStack::last_final_hc_gpu_rows() const {
    const auto& p = *impl_; p.check_thread();
    require(bool(p.final_hc_tap), "final HC tap is not prepared");
    require(p.final_hc_tap->count > 0 && p.final_hc_gpu_valid && p.final_hc_current_gpu.owner != 0,
        "final HC tap has no committed GPU rows");
    return p.final_hc_current_gpu;
}
GpuView TextStack::last_final_hc_gpu_residual() const {
    const auto& p = *impl_; p.check_thread();
    const auto rows = last_final_hc_gpu_rows();
    const uint64_t row_bytes = uint64_t(p.final_hc_tap->width) * sizeof(float);
    return p.g.slice(rows, uint64_t(p.final_hc_tap->count - 1) * row_bytes, row_bytes);
}
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
void TextStack::prepare_dflash_tap(std::span<const uint32_t> layer_ids, uint64_t budget) {
    auto& p = *impl_; p.check_thread();
    require(!p.dflash_tap && !p.prefix && !p.checkpoints && !p.verification &&
        p.execution == StackExecution::direct && !p.replay_prepared &&
        p.shape.residual.branches == 4 && p.shape.residual.channels % 4 == 0 &&
        !layer_ids.empty() && layer_ids.size() <= 16 && budget && budget <= maximum_dflash2_tap_budget_bytes,
        "DFlash tap requires an empty direct 4-branch stack and bounded budget");
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    require(!p.verification8, "DFlash tap must precede verifier preparation");
#endif
    static_cast<void>(p.g.last_profile());
    std::vector<int32_t> slots(p.weights.layers.size(), -1);
    for (size_t s = 0; s < layer_ids.size(); ++s) {
        const uint32_t id = layer_ids[s];
        require(id < p.weights.layers.size() && slots[id] < 0, "DFlash tap layer index out of bounds or duplicate");
        slots[id] = int32_t(s);
    }
    const uint32_t slot_count = uint32_t(layer_ids.size());
    const uint32_t slot_width = p.shape.residual.channels;
    const auto plan = dflash2_tap_plan(p.known_capacity, slot_count, slot_width, budget);
    p.g.prepare_dflash2();
    const auto before = p.g.memory().live_bytes;
    auto tap = std::make_unique<DFlashTap>();
    tap->slot_for_layer = std::move(slots);
    tap->slot_count = slot_count;
    tap->slot_width = slot_width;
    tap->known_capacity = plan.known_capacity;
    tap->verify_capacity = 8u;
    try {
        tap->known_gpu = p.g.zeros(plan.known_bytes, "DFlash known tap rows");
        tap->owned.push_back(tap->known_gpu);
        tap->verify_gpu = p.g.zeros(plan.verify_bytes, "DFlash verify tap rows");
        tap->owned.push_back(tap->verify_gpu);
        tap->gpu_charged_bytes = p.g.memory().live_bytes - before;
        require(tap->gpu_charged_bytes <= budget, "DFlash tap exceeds its explicit GPU budget");
        p.g.validate_disjoint(p.owned, tap->owned);
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
        if (p.chunked) p.g.validate_disjoint(p.chunked->owned, tap->owned);
#endif
    } catch (...) {
        for (auto i = tap->owned.rbegin(); i != tap->owned.rend(); ++i) { try { p.g.release(*i); } catch (...) {} }
        throw;
    }
    p.dflash_tap = std::move(tap);
}
StackDFlashTapMemory TextStack::dflash_tap_memory() const {
    const auto& p = *impl_; p.check_thread();
    return p.dflash_tap ? StackDFlashTapMemory{true, p.dflash_tap->known_capacity, p.dflash_tap->verify_capacity,
        p.dflash_tap->slot_count, p.dflash_tap->slot_width, p.dflash_tap->gpu_charged_bytes}
        : StackDFlashTapMemory{};
}
StackDFlashCapture TextStack::last_dflash_capture() const {
    const auto& p = *impl_; p.check_thread();
    require(bool(p.dflash_tap), "DFlash tap is not prepared");
    return p.dflash_tap->last_capture;
}
#endif
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
void TextStack::prepare_known_qsa_rows(bool enabled) {
    auto& p = *impl_; p.check_thread(); p.check_known_policy();
    require(!p.known_qsa_rows_memory.prepared && !p.prefix && p.known_capacity == 8 &&
        !p.checkpoints && !p.verification && !p.verification8,
        "known QSA rows require one-time capacity8 preparation before checkpoints/verifiers");
    const AttentionLayerShape shape{p.shape.residual, p.shape.attention, p.shape.experts};
    bool attention = false;
    for (const auto& layer : p.weights.layers) if (const auto* w = std::get_if<AttentionLayerWeights>(&layer)) {
        static_cast<void>(attention_tile_dispatches(p.g, *w, 0, 1, shape, AttentionTileMode::core_and_q8,
            AttentionTileHcMode::combined));
        attention = true;
    }
    require(attention, "known QSA rows require an attention layer");
    const auto logical = attention_tile_scratch_bytes(shape);
    require(logical <= maximum_known_qsa_rows_bytes, "known QSA rows exceed fixed workspace budget");
    p.owned.reserve(p.owned.size() + 46); // No throwing owner append after allocation.
    prepare_attention_tile(p.g, AttentionTileHcMode::combined);
    const auto before = p.g.memory();
    auto scratch = allocate_attention_tile_scratch(p.g, shape);
    try {
        const auto after = p.g.memory();
        require(after.allocations == before.allocations + 46 && after.live_bytes >= before.live_bytes + logical &&
            after.live_bytes - before.live_bytes <= maximum_known_qsa_rows_bytes,
            "known QSA row allocation accounting/budget mismatch");
        p.keep(scratch.views()); p.known_qsa_rows = scratch;
        p.known_qsa_rows_memory = {true, enabled, logical, after.live_bytes - before.live_bytes, 46};
    } catch (...) { release_attention_tile_scratch(p.g, scratch); throw; }
}
StackKnownQsaRowsMemory TextStack::known_qsa_rows_memory() const {
    impl_->check_thread(); return impl_->known_qsa_rows_memory;
}
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
bool TextStack::known_grouped_ffn() const { impl_->check_thread(); return impl_->grouped_known_enabled; }
void TextStack::configure_known_grouped_ffn(bool enabled) {
    auto& p = *impl_; p.check_large_known_selector();
    require(!enabled || !p.known_hc_rows, "private HC residual rows require ordinary FFN");
    p.grouped_known_enabled = enabled;
}
bool TextStack::known_layer_submissions() const { impl_->check_thread(); return impl_->known_layer_submit; }
void TextStack::configure_known_layer_submissions(bool enabled) {
    auto& p = *impl_; p.check_large_known_selector(); p.known_layer_submit = enabled;
}
bool TextStack::known_hc_residual_rows() const { impl_->check_thread(); return impl_->known_hc_rows; }
void TextStack::configure_known_hc_residual_rows(bool enabled) {
    auto& p = *impl_; p.check_large_known_selector();
    require(!enabled || (!p.grouped_known_enabled && !p.residual_reuse && !p.hc_q8_splits),
        "private HC residual rows require ordinary FFN, separate capacity512 policy and unsplit HC");
    p.known_hc_rows = enabled;
}
#endif
bool TextStack::known_residual_reuse() const { impl_->check_thread(); return impl_->residual_reuse; }
void TextStack::configure_known_residual_reuse(bool enabled) {
    auto& p = *impl_; p.check_thread();
    require(p.known_capacity == 8 && !p.prefix, "residual reuse switch requires prepared capacity eight and empty prefix");
    require(!enabled || !p.hc_q8_splits, "HC Q8 split policy does not support known residual reuse");
    p.check_known_policy();
    require(!p.checkpoints || enabled == p.residual_reuse, "prepared checkpoints pin residual reuse policy");
    require(!p.verification || enabled == p.residual_reuse, "prepared verification pins residual reuse policy");
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    require(!p.verification8 || enabled == p.residual_reuse, "prepared verification8 pins residual reuse policy");
#endif
    p.residual_reuse = enabled;
}
void TextStack::configure_execution(StackExecution execution) {
    auto& p = *impl_; p.check_thread();
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    require(p.hc_exact == HcExactMode::off || execution == StackExecution::direct, "private exact-HC refuses replay/indirect stack execution");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(!p.wide_q8_exact || execution == StackExecution::direct, "private wide-Q8 refuses replay/indirect stack execution");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!p.wide_q8_many || execution == StackExecution::direct, "private wide-Q8-many refuses replay/indirect stack execution");
#endif
    require(uint32_t(execution) <= uint32_t(StackExecution::recurrent_replay), "unsupported stack execution mode");
    require(p.replay_prepared && !p.prefix, "stack execution switch requires a replay-prepared empty prefix");
    require(p.g.kernel_policy() == p.kernel_policy, "immutable replay kernel policy changed before execution switch");
    // This existing read-only graph API requires idle state and performs no
    // device work/allocation. Never change scheduling inside an active batch.
    static_cast<void>(p.g.last_profile());
    p.execution = execution;
}
NgramHistory TextStack::committed_history() const {
    const auto& p = *impl_; p.check_thread();
    return p.history;
}
NgramStep TextStack::lookup(uint32_t token) const {
    const auto& p = *impl_; p.check_thread();
    if (p.checkpoints) p.check_checkpoint_policy();
    if (p.known_capacity) p.check_known_policy();
    if (p.verification) p.check_verification_policy();
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    if (p.verification8) p.check_verification8_policy();
#endif
    require((!p.prefix && !p.replay_prepared) || p.g.kernel_policy() == p.kernel_policy,
        "kernel policy changed under an accepted prefix or immutable replay plan");
    require(p.prefix < p.shape.attention.capacity, "complete stack context capacity exceeded");
    return ngram_step(p.spec, p.policy.ngram, p.history, token);
}
std::vector<NgramStep> TextStack::lookups(std::span<const uint32_t> tokens) const {
    const auto& p = *impl_; p.check_thread();
    require(p.known_capacity && !tokens.empty() && tokens.size() <= p.known_capacity,
        "known-token lookup exceeds explicit prepared capacity");
    p.check_known_policy();
    if (p.checkpoints) p.check_checkpoint_policy();
    if (p.verification) p.check_verification_policy();
    require(tokens.size() <= p.shape.attention.capacity - p.prefix, "known-token prefill exceeds context capacity");
    std::vector<NgramStep> staged; staged.reserve(tokens.size()); auto history = p.history;
    for (auto token : tokens) {
        auto next = ngram_step(p.spec, p.policy.ngram, history, token);
        history = next.next; staged.push_back(std::move(next));
    }
    return staged;
}
StackResult TextStack::step(uint32_t token, std::span<const float> row, std::span<const float> memory) {
    return step_with_logits(token, row, memory, true);
}
StackResult TextStack::step_with_logits(uint32_t token, std::span<const float> row, std::span<const float> memory, bool emit_logits) {
    auto& p = *impl_; p.check_thread(); const auto candidate = lookup(token);
    const auto kernel_policy = p.g.kernel_policy();
    require(row.size() == p.shape.residual.channels && memory.size() == p.shape.memory.embedding, "stack embedding input shape mismatch");
    for (auto value : row) require(std::isfinite(value), "nonfinite token embedding");
    for (auto value : memory) require(std::isfinite(value), "nonfinite lookup embedding");
    for (uint32_t b = 0; b < p.shape.residual.branches; ++b) std::copy(row.begin(), row.end(), p.host_streams.begin() + uint64_t(b) * row.size());
    p.g.write(p.token_streams, std::as_bytes(std::span(p.host_streams))); p.g.write(p.embedding, std::as_bytes(memory));
    const auto allocations = p.g.memory().allocations; const auto expected = p.expected_dispatches() - (emit_logits ? 0 : p.final_dispatches()) + p.dflash_tap_dispatches();
#if defined(TOKSPUR_PRIVATE_CHUNKED_PREFILL) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && (defined(TOKSPUR_PRIVATE_QSA_SPARSE) || defined(TOKSPUR_PRIVATE_GDN_PORTED))
    const bool route_step_rollback = p.chunked && (
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        p.g.qsa_route_attribution_enabled() ||
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        p.g.gdn_route_attribution_enabled() ||
#endif
        false);
    std::array<GraphCopy, 384> qsa_route_save{}, qsa_route_restore{};
    const auto qsa_route_copies = route_step_rollback ? p.chunked_cache_copies(1, qsa_route_save, qsa_route_restore) : 0;
    if (qsa_route_copies) p.g.copy(std::span(qsa_route_save).first(qsa_route_copies));
#endif
    try {
        const auto orientation = p.execution == StackExecution::recurrent_replay ? p.physical_orientation() : 0;
        if (p.execution == StackExecution::indirect_compatible) p.g.begin_indirect_compatible(); else p.g.begin();
        size_t next_run = 0;
        for (size_t i = 0; i < p.weights.layers.size();) {
            if (p.execution == StackExecution::recurrent_replay && next_run < p.runs.size() && p.runs[next_run].begin == i) {
                const auto& run = p.runs[next_run++]; p.g.execute_capture(run.captures[orientation]); i = run.end;
            } else {
                // Position-dependent QSA projections/cache work remains dynamic
                // between captured runs, in this same begin/submit transaction.
                p.encode_layer(i);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (p.dflash_tap) p.encode_dflash_tap_layer(i, p.layer_output(i), p.dflash_tap->verify_gpu, 1);
#endif
                ++i;
            }
        }
        const auto streams = p.layer_output(p.weights.layers.size() - 1);
        if (emit_logits) {
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            p.g.set_task4_profile_tag({Task4ProfilePhase::decode_mixer, -1, p.prefix, 1});
#endif
            encode_mixer(p.g, p.weights.mixer, p.m, streams, p.shape.residual, p.policy.attention.rms_epsilon);
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            p.g.set_task4_profile_tag({Task4ProfilePhase::decode_output, -1, p.prefix, 1});
#endif
            p.g.gemv(p.weights.output.encoding, p.weights.output.view, p.m.mixed, p.logits, p.shape.residual.channels, p.shape.vocabulary);
        }
        const auto batch = p.g.submit();
        require(batch.dispatches == expected + p.diagnostic_route_dispatches() && batch.capture_executes == (p.execution == StackExecution::recurrent_replay ? p.runs.size() : 0) &&
            p.g.memory().allocations == allocations, "full stack dispatch/allocation/capture invariant failed");
        std::vector<float> logits;
        if (emit_logits) {
            const auto bytes = p.g.read(p.logits); logits.resize(p.shape.vocabulary); std::memcpy(logits.data(), bytes.data(), bytes.size());
            for (auto value : logits) require(std::isfinite(value), "nonfinite target logits");
        }
        p.capture_final_hc(streams, 1);
        // No fallible work after publication. Returning/moving the prepared
        // vector, view swaps, scalar prefix and fixed-array assignment cannot throw.
        const auto position = p.prefix; p.publish_pairs(); p.history = candidate.next; p.kernel_policy = kernel_policy; ++p.prefix;
        p.publish_final_hc(1);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (p.dflash_tap) p.publish_dflash_tap(p.dflash_tap->verify_gpu, 1);
#endif
        return {std::move(logits), batch, position, emit_logits};
    } catch (...) {
        try { p.g.abort(); } catch (const FormatError&) {}
#if defined(TOKSPUR_PRIVATE_CHUNKED_PREFILL) && defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && (defined(TOKSPUR_PRIVATE_QSA_SPARSE) || defined(TOKSPUR_PRIVATE_GDN_PORTED))
        if (qsa_route_copies) p.g.copy(std::span(qsa_route_restore).first(qsa_route_copies));
#endif
        throw;
    }
}
StackResult TextStack::step_known(std::span<const uint32_t> tokens, std::span<const float> rows, std::span<const float> memory, bool emit_logits) {
    auto& p = *impl_; p.check_thread();
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
    const bool has_prefill_tap = bool(p.final_hc_tap)
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        || bool(p.dflash_tap)
#endif
        ;
    require(emit_logits || (p.chunked && has_prefill_tap), "deferred logits require prepared chunked input and final HC or DFlash tap");
#else
    require(emit_logits, "deferred logits require the explicit input512 build");
#endif
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    if (p.chunked && tokens.size() == 1) {
#if defined(TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS) && (defined(TOKSPUR_PRIVATE_QSA_SPARSE) || defined(TOKSPUR_PRIVATE_GDN_PORTED))
        if (
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
            p.g.qsa_route_attribution_enabled() ||
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
            p.g.gdn_route_attribution_enabled() ||
#endif
            false) return step_with_logits(tokens.front(), rows, memory, emit_logits);
#endif
        // A single input uses the same native arithmetic as ordinary decoding. The chunk transaction still owns full physical cache rollback.
        std::array<GraphCopy, 384> save{}, restore{}; const auto count = p.chunked_cache_copies(1, save, restore);
        if (count) p.g.copy(std::span(save).first(count));
        try { return step_with_logits(tokens.front(), rows, memory, emit_logits); }
        catch (...) { if (count) p.g.copy(std::span(restore).first(count)); throw; }
    }
#endif
    const auto staged = lookups(tokens);
    const auto steps = uint32_t(tokens.size());
    const auto channels = p.shape.residual.channels;
    const auto width = uint64_t(p.shape.residual.branches) * channels;
    require(rows.size() == checked_mul(steps, channels) && memory.size() == checked_mul(steps, p.shape.memory.embedding),
        "known-token flattened embedding input shape mismatch");
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
    const auto require_all_finite = [](std::span<const float> vals, const char* msg) {
        size_t idx = 0;
#if defined(__aarch64__)
        uint32x4_t bad = vdupq_n_u32(0u);
        const uint32x4_t exp_mask = vdupq_n_u32(0x7f800000u);
        for (; idx + 16 <= vals.size(); idx += 16) {
            const float* fp = vals.data() + idx;
            const uint32x4_t c0 = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(fp)), exp_mask), exp_mask);
            const uint32x4_t c1 = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(fp + 4)), exp_mask), exp_mask);
            const uint32x4_t c2 = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(fp + 8)), exp_mask), exp_mask);
            const uint32x4_t c3 = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(fp + 12)), exp_mask), exp_mask);
            bad = vorrq_u32(bad, vorrq_u32(vorrq_u32(c0, c1), vorrq_u32(c2, c3)));
        }
        require(vmaxvq_u32(bad) == 0u, msg);
#endif
        for (; idx < vals.size(); ++idx) require(std::isfinite(vals[idx]), msg);
    };
    require_all_finite(rows, "nonfinite known-token embedding");
    require_all_finite(memory, "nonfinite known-token lookup embedding");
#else
    for (auto value : rows) require(std::isfinite(value), "nonfinite known-token embedding");
    for (auto value : memory) require(std::isfinite(value), "nonfinite known-token lookup embedding");
#endif
    for (uint32_t t = 0; t < steps; ++t) for (uint32_t b = 0; b < p.shape.residual.branches; ++b) {
        const auto row = rows.subspan(uint64_t(t) * channels, channels);
        std::copy(row.begin(), row.end(), p.known_host_streams.begin() + uint64_t(t) * width + uint64_t(b) * channels);
    }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    if (p.g.prefill_dense_nax() || p.g.prefill_grouped_nax()) {
        const uint64_t stream_floats = checked_mul(uint64_t(steps), width);
        p.g.write(p.g.slice(p.known_streams, 0, checked_mul(stream_floats, 4)), std::as_bytes(std::span(p.known_host_streams).first(stream_floats)));
    } else
#endif
    {
        p.g.write(p.known_streams, std::as_bytes(std::span(p.known_host_streams)));
    }
    p.g.write(p.g.slice(p.known_embeddings, 0, checked_mul(memory.size(), 4)), std::as_bytes(memory));
    const auto allocations = p.g.memory().allocations; const auto expected = p.known_dispatches(steps) - (emit_logits ? 0 : p.final_dispatches()) + p.dflash_tap_dispatches();
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    require(!p.g.task4_profile_reserved() || expected <= 16384, "Task4 full-packet profile exceeds counter capacity; splitting refused");
#endif
    std::vector<float> logits(emit_logits ? p.shape.vocabulary : 0);
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    std::array<GraphCopy, 384> cache_save{}, cache_restore{};
    const auto cache_copies = p.chunked_cache_copies(steps, cache_save, cache_restore);
    if (cache_copies) p.g.copy(std::span(cache_save).first(cache_copies));
#endif
    try {
        BatchResult batch{};
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        const auto logical_started = p.known_layer_submit ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto accumulate = [&](BatchResult part) {
            batch.dispatches = checked_add(batch.dispatches, part.dispatches);
            batch.capture_executes = checked_add(batch.capture_executes, part.capture_executes);
            batch.wall_ms += part.wall_ms; batch.gpu_ms += part.gpu_ms;
        };
#endif
        p.g.begin();
        for (size_t i = 0; i < p.weights.layers.size(); ++i) {
            p.encode_known_layer(i, steps);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (p.dflash_tap) p.encode_dflash_tap_layer(i, i % 2 ? p.known_streams : p.known_second, p.dflash_tap->known_gpu, steps);
#endif
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
            if (p.known_layer_submit) {
                accumulate(p.g.submit());
                p.g.begin();
            }
#endif
        }
        const auto last_streams = p.weights.layers.size() % 2 ? p.known_second : p.known_streams;
        if (emit_logits) {
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            p.g.set_task4_profile_tag({Task4ProfilePhase::prefill_mixer, -1, p.prefix + steps - 1, 1});
#endif
            encode_mixer(p.g, p.weights.mixer, p.m, p.known_row(last_streams, width, steps - 1), p.shape.residual, p.policy.attention.rms_epsilon);
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
            p.g.set_task4_profile_tag({Task4ProfilePhase::prefill_output, -1, p.prefix + steps - 1, 1});
#endif
            p.g.gemv(p.weights.output.encoding, p.weights.output.view, p.m.mixed, p.logits, channels, p.shape.vocabulary);
        }
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        accumulate(p.g.submit());
        if (p.known_layer_submit)
            batch.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - logical_started).count();
#else
        batch = p.g.submit();
#endif
        if (p.g.last_profile().timings_valid && std::getenv("TOKSPUR_DUMP_VERIFY_PROFILE")) {
            std::map<std::string, std::pair<uint32_t, double>> agg;
            for (const auto& kt : p.g.last_profile().kernels) {
                auto& e = agg[kt.kernel];
                e.first += 1;
                e.second += kt.milliseconds;
            }
            std::cerr << "PREFILL_PROFILE sampled_ms=" << p.g.last_profile().sampled_milliseconds
                      << " gpu_ms=" << batch.gpu_ms
                      << " dispatches=" << batch.dispatches;
            for (const auto& [kname, kv] : agg) {
                std::cerr << " " << kname << "=" << kv.first << ":" << kv.second;
            }
            std::cerr << "\n";
        }
        require(batch.dispatches == expected && batch.capture_executes == 0 && p.g.memory().allocations == allocations,
            "known-token full stack dispatch/allocation/capture invariant failed");
        if (emit_logits) {
            const auto bytes = p.g.read(p.logits); std::memcpy(logits.data(), bytes.data(), bytes.size());
            for (auto value : logits) require(std::isfinite(value), "nonfinite known-token target logits");
        }
        if (p.final_hc_gpu_mode) {
            p.capture_final_hc_gpu_only(last_streams, steps);
        } else {
            p.capture_final_hc(last_streams, steps);
        }
        // Every candidate is complete. No fallible operation may follow this
        // all-N publication; no partial acceptance/verification contract exists.
        const auto position = p.prefix; p.publish_pairs(); p.history = staged.back().next; p.prefix += steps;
        p.publish_final_hc(steps);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (p.dflash_tap) p.publish_dflash_tap(p.dflash_tap->known_gpu, steps);
#endif
        return {std::move(logits), batch, position, emit_logits};
    } catch (...) {
        try { p.g.abort(); } catch (const FormatError&) {}
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
        if (cache_copies) p.g.copy(std::span(cache_restore).first(cache_copies));
#endif
        throw;
    }
}
void TextStack::reset() {
    auto& p = *impl_; p.check_thread(); const auto empty = empty_ngram_history(p.spec);
    if (p.checkpoints) p.check_checkpoint_policy();
    if (p.known_capacity) p.check_known_policy();
    if (p.verification) p.check_verification_policy();
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    if (p.verification8) p.check_verification8_policy();
#endif
    const auto kernel_policy = p.g.kernel_policy();
    require(!p.replay_prepared || kernel_policy == p.kernel_policy,
        "immutable replay kernel policy changed; construct a new stack for a different policy");
    const auto clear = [&](GpuView v) { p.g.write(v, std::as_bytes(std::span(p.zero)).first(v.bytes)); };
    // Clear candidates, never accepted state. A refusal before publication
    // leaves the previous composite prefix intact. Old KV rows are ignored at 0.
    for (const auto& state : p.states) {
        if (const auto* r = std::get_if<RecurrentPair>(&state)) { clear(r->next.matrix); clear(r->next.convolution); }
        else clear(std::get<AttentionPair>(state).next);
    }
    clear(p.next_memory); p.publish_pairs(); p.history = empty; p.kernel_policy = kernel_policy; p.prefix = 0;
    if (p.final_hc_tap) p.final_hc_tap->count = 0;
    p.final_hc_current_gpu = {}; p.final_hc_candidate_gpu = {};
    p.final_hc_gpu_valid = false; p.final_hc_host_valid = false; p.final_hc_candidate_host_valid = false;
    p.publish_dflash_tap({}, 0);
    p.invalidate_checkpoints();
}
void TextStack::prepare_transient_verification(uint64_t budget) {
    auto& p = *impl_; p.check_thread();
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    require(!p.chunked, "chunked prefill refuses checkpoint or ordinary verifier stores");
#endif
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
    require(!p.known_qsa_rows_memory.prepared, "known QSA row spike does not admit verifier composition");
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
    require(!p.verification8_branch_reuse, "branch-verifier consent refuses the old depth-two store");
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    require(!p.verification8, "old and eight-row verification stores are mutually exclusive");
#endif
    require(!p.hc_q8_splits, "HC Q8 split policy does not support speculative verification");
    require(!p.verification && !p.prefix && budget && budget <= maximum_stack_verification_budget,
        "verification preparation requires empty unprepared stack and bounded budget");
    p.check_known_policy();
    if (p.checkpoints) p.check_checkpoint_policy();
    for (const auto& layer : p.weights.layers) if (const auto* w = std::get_if<RecurrentWeights>(&layer)) {
        const auto& a = w->attention;
        require(a.qkv.encoding == 8 && a.gate.encoding == 8 && a.output.encoding == 8 &&
            a.alpha.encoding == 0 && a.beta.encoding == 0, "verification requires Q8/F32 recurrent projections");
    }
    const auto width = uint64_t(p.shape.residual.branches) * p.shape.residual.channels;
    const auto host_minimum = checked_add(sizeof(VerificationStore), checked_mul(3ull * p.shape.vocabulary + 2 * width, 4));
    require(host_minimum < budget, "verification host storage exceeds explicit budget");
    auto store = std::make_unique<VerificationStore>();
    store->tapes.resize(p.states.size()); store->first_sums.resize(p.states.size());
    store->owned.reserve(p.states.size() * 6 + 50);
    store->host_streams.resize(2 * width);
    for (auto& row : store->host_logits) row.resize(p.shape.vocabulary);
    for (auto& row : store->lookup_rows) row.resize(p.spec.offsets.size());
    store->host_charged_bytes = sizeof(VerificationStore) + store->tapes.capacity() * sizeof(GdnCoreTape) +
        (store->first_sums.capacity() + store->owned.capacity()) * sizeof(GpuView) + store->host_streams.capacity() * sizeof(float);
    for (const auto& row : store->host_logits) store->host_charged_bytes += row.capacity() * sizeof(float);
    for (const auto& row : store->lookup_rows) store->host_charged_bytes += row.capacity() * sizeof(uint64_t);
    require(store->host_charged_bytes < budget, "verification host capacities exceed explicit budget");
    const auto before = p.g.memory().live_bytes;
    const auto admit = [&] {
        store->gpu_charged_bytes = p.g.memory().live_bytes - before;
        require(checked_add(store->gpu_charged_bytes, store->host_charged_bytes) <= budget,
            "actual verification allocations exceed separate transient budget");
    };
    const auto keep = [&](auto views) { for (auto v : views) store->owned.push_back(v); admit(); };
    const auto zeros = [&](uint64_t count, const char* label) {
        const auto view = p.g.zeros(checked_mul(count, 4), label); store->owned.push_back(view); admit(); return view;
    };
    try {
        store->streams = zeros(2 * width, "verification branch rows");
        store->second = zeros(2 * width, "verification alternating rows");
        store->memory_streams = zeros(2 * width, "verification PLE output rows");
        store->embeddings = zeros(2ull * p.shape.memory.embedding, "verification lookup rows");
        store->first_memory = zeros(p.current_memory.bytes / 4, "verification first PLE history");
        store->logits = zeros(2ull * p.shape.vocabulary, "verification two target logit rows");
        bool recurrent = false;
        for (size_t i = 0; i < p.states.size(); ++i) {
            if (std::holds_alternative<RecurrentPair>(p.states[i])) {
                recurrent = true; store->tapes[i] = allocate_gdn_core_tape(p.g, p.shape.recurrent, 2);
                keep(store->tapes[i].views());
                for (auto view : store->tapes[i].views()) store->tape_payload_bytes += view.bytes;
            } else store->first_sums[i] = zeros(p.shape.attention.index_width, "verification first QSA partial sum");
        }
        if (recurrent) {
            store->recurrent = allocate_recurrent_tile_scratch(p.g, {p.shape.residual, p.shape.recurrent, p.shape.experts}, 2);
            keep(store->recurrent.views());
            store->replay = allocate_gdn_core_replay_scratch(p.g, p.shape.recurrent, 2);
            keep(std::array{store->replay.convolution_output, store->replay.delta_output});
            p.g.prepare_q8_matvec_many(2); p.g.prepare_gdn_tiles(128);
        }
        admit();
    } catch (...) {
        for (auto i = store->owned.rbegin(); i != store->owned.rend(); ++i) p.g.release(*i);
        throw;
    }
    store->kernel_policy = p.g.kernel_policy(); store->residual_reuse = p.residual_reuse;
    p.verification = std::move(store);
}
StackVerificationMemory TextStack::verification_memory() const {
    const auto& p = *impl_; p.check_thread();
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    if (p.verification8) { const auto& v = *p.verification8; return {true, v.gpu_charged_bytes, v.host_charged_bytes, v.tape_payload_bytes}; }
#endif
    if (!p.verification) return {};
    const auto& v = *p.verification;
    return {true, v.gpu_charged_bytes, v.host_charged_bytes, v.tape_payload_bytes};
}
std::array<StackVerificationLookup, 2> TextStack::verification_lookups(std::span<const uint32_t> tokens) const {
    const auto& p = *impl_; p.check_thread(); p.check_verification_policy();
    require(tokens.size() == 2 && p.prefix <= p.shape.attention.capacity &&
        tokens.size() <= p.shape.attention.capacity - p.prefix, "verification requires two tokens within context");
    auto& v = *p.verification;
    const auto first = ngram_step(p.spec, p.policy.ngram, p.history, tokens[0], v.lookup_rows[0]);
    const auto second = ngram_step(p.spec, p.policy.ngram, first, tokens[1], v.lookup_rows[1]);
    return {{{first, v.lookup_rows[0]}, {second, v.lookup_rows[1]}}};
}
StackVerificationResult TextStack::verify_greedy2(std::span<const uint32_t> tokens,
    std::span<const float> rows, std::span<const float> memory, uint32_t committed_prefix,
    std::span<const float> committed_logits, uint32_t remaining_outputs,
    std::span<const uint32_t> stops, StackVerificationCancellation cancellation) {
    const auto started = std::chrono::steady_clock::now();
    auto& p = *impl_; p.check_thread(); p.check_verification_policy();
    if (p.checkpoints) p.check_checkpoint_policy();
    const auto staged = verification_lookups(tokens);
    require(p.prefix && committed_prefix == p.prefix && committed_logits.size() == p.shape.vocabulary,
        "verification requires matching authoritative committed logits/prefix");
    const auto first_id = greedy_id(committed_logits);
    require(stops.size() <= p.shape.vocabulary, "verification stop-list bound");
    for (auto id : stops) require(id < p.shape.vocabulary, "verification stop outside vocabulary");
    const auto channels = p.shape.residual.channels;
    const auto width = uint64_t(p.shape.residual.branches) * channels;
    require(rows.size() == 2ull * channels && memory.size() == 2ull * p.shape.memory.embedding,
        "verification flattened input shape mismatch");
    for (auto value : rows) require(std::isfinite(value), "nonfinite verification token embedding");
    for (auto value : memory) require(std::isfinite(value), "nonfinite verification lookup embedding");
    auto& v = *p.verification;
    std::copy(committed_logits.begin(), committed_logits.end(), v.host_logits[0].begin());
    StackVerificationResult result{0, p.prefix, v.host_logits[0]};
    result.source_logits[0] = committed_logits[first_id];
    p.pending_verification = true;
    struct Pending { bool& value; ~Pending() { value = false; } } pending{p.pending_verification};
    const auto poll = [&] {
        require(!cancellation.requested || !cancellation.requested(cancellation.context), "greedy verification cancelled before publication");
        p.check_verification_policy();
    };
    try {
        poll();
        if (remaining_outputs < 2 || tokens[0] != first_id || std::find(stops.begin(), stops.end(), tokens[0]) != stops.end()) {
            result.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            return result;
        }
        for (uint32_t t = 0; t < 2; ++t) for (uint32_t b = 0; b < p.shape.residual.branches; ++b)
            std::copy_n(rows.begin() + uint64_t(t) * channels, channels, v.host_streams.begin() + uint64_t(t) * width + uint64_t(b) * channels);
        p.g.write(v.streams, std::as_bytes(std::span(v.host_streams)));
        p.g.write(v.embeddings, std::as_bytes(memory));
        const auto allocations = p.g.memory().allocations;
        const auto expected = p.legacy_known_dispatches(2) + p.final_dispatches();
        p.g.begin();
        for (size_t i = 0; i < p.weights.layers.size(); ++i) p.encode_verification_layer(i);
        const auto streams = p.weights.layers.size() % 2 ? v.second : v.streams;
        for (uint32_t t = 0; t < 2; ++t) {
            encode_mixer(p.g, p.weights.mixer, p.m, p.known_row(streams, width, t), p.shape.residual, p.policy.attention.rms_epsilon);
            p.g.gemv(p.weights.output.encoding, p.weights.output.view, p.m.mixed,
                p.known_row(v.logits, p.shape.vocabulary, t), channels, p.shape.vocabulary);
        }
        result.verification = p.g.submit();
        require(result.verification.dispatches == expected && !result.verification.capture_executes &&
            p.g.memory().allocations == allocations, "verification dispatch/allocation invariant failed");
        for (uint32_t t = 0; t < 2; ++t) {
            p.g.read(p.known_row(v.logits, p.shape.vocabulary, t), std::as_writable_bytes(std::span(v.host_logits[t + 1])));
            static_cast<void>(greedy_id(v.host_logits[t + 1]));
        }
        result.accepted = tokens[1] == greedy_id(v.host_logits[1]) ? 2 : 1;
        if (result.accepted == 2) result.source_logits[1] = v.host_logits[1][tokens[1]];
        poll(); // Forward work is drained; accepted banks are still untouched.
        if (result.accepted == 1) {
            uint64_t recurrent_layers = 0;
            for (const auto& state : p.states) recurrent_layers += std::holds_alternative<RecurrentPair>(state);
            if (recurrent_layers) {
                p.g.begin();
                for (size_t i = 0; i < p.states.size(); ++i) if (const auto* r = std::get_if<RecurrentPair>(&p.states[i]))
                    encode_gdn_core_tape_prefix(p.g, std::get<RecurrentWeights>(p.weights.layers[i]).attention.convolution,
                        v.tapes[i], r->current, r->next, v.replay, p.shape.recurrent, {2, 1});
                result.recovery = p.g.submit();
                require(result.recovery.dispatches == recurrent_layers * 2 && !result.recovery.capture_executes &&
                    p.g.memory().allocations == allocations, "verification recovery invariant failed");
            }
            // Candidate-only copies have global alias validation before any
            // bytes change. No accepted state or complete QSA cache is copied.
            std::array<GraphCopy, 129> copies{}; size_t count = 0;
            copies[count++] = {v.first_memory, p.next_memory};
            for (size_t i = 0; i < p.states.size(); ++i) if (const auto* a = std::get_if<AttentionPair>(&p.states[i]))
                copies[count++] = {v.first_sums[i], a->next};
            p.g.copy(std::span(copies).first(count));
            poll();
        }
        result.logits = v.host_logits[result.accepted];
        p.capture_final_hc(streams, 2);
        result.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        // QSA appended only rows P/P+1 and at most newly completed index rows.
        // Prefix bounds hide a rejected tail; future append overwrites it. Rows
        // below P and retained checkpoint ancestors are never changed.
        // ALL fallible work precedes composite bank/history/prefix publication.
        p.publish_pairs(); p.history = staged[result.accepted - 1].next; p.prefix += result.accepted;
        p.publish_final_hc(result.accepted);
        return result;
    } catch (...) { try { p.g.abort(); } catch (const FormatError&) {} throw; }
}
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
void TextStack::prepare_sampled_verification2(uint64_t budget
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    , QsaVerifyRowsConfig qsa_rows
#endif
    ) {
    auto& p = *impl_; p.check_thread();
    const bool has_verify_tap = bool(p.final_hc_tap)
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        || bool(p.dflash_tap)
#endif
        ;
    require(has_verify_tap && !p.sampled_verification_prepared && !p.preparing_sampled_verification &&
        !p.checkpoints && !p.verification && !p.verification8 && !p.prefix && (p.known_capacity == 8
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        || ((p.known_capacity == 512 || p.known_capacity == 1024 || p.known_capacity == 2048) && p.chunked && p.residual_reuse)
#endif
        ),
        "sampled verification requires final HC or DFlash tap and empty exclusive capacity8 stack");
    p.preparing_sampled_verification = true;
    struct Preparing { bool& value; ~Preparing() { value = false; } } preparing{p.preparing_sampled_verification};
    prepare_transient_verification8(budget
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        , qsa_rows
#endif
        );
    p.sampled_verification_prepared = true;
}
std::array<StackVerificationLookup, 2> TextStack::sampled_verification_lookups(
    std::span<const uint32_t> tokens) const {
    const auto& p = *impl_; p.check_thread(); p.check_verification8_policy();
    require(p.sampled_verification_prepared && !tokens.empty() && tokens.size() <= 2 &&
        p.prefix <= p.shape.attention.capacity && tokens.size() <= p.shape.attention.capacity - p.prefix,
        "sampled verification requires one or two inputs within context");
    std::array<StackVerificationLookup, 2> out{};
    auto history = p.history;
    for (size_t t = 0; t < tokens.size(); ++t) {
        history = ngram_step(p.spec, p.policy.ngram, history, tokens[t], p.verification8->lookup_rows[t]);
        out[t] = {history, p.verification8->lookup_rows[t]};
    }
    return out;
}
StackSampledVerificationResult TextStack::verify_sampled2(std::span<const uint32_t> tokens,
    std::span<const float> rows, std::span<const float> memory, uint32_t committed_prefix,
    StackSampledVerificationSelector selector, StackVerificationCancellation cancellation) {
    return verify_sampled_bounded(tokens, rows, memory, committed_prefix, selector, cancellation, 2);
}
#if defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
std::array<StackVerificationLookup, 3> TextStack::sampled_verification_cost_lookups(
    std::span<const uint32_t> tokens) const {
    const auto& p = *impl_; p.check_thread(); p.check_verification8_policy();
    require(p.sampled_verification_prepared && !tokens.empty() && tokens.size() <= 3 &&
        p.prefix <= p.shape.attention.capacity && tokens.size() <= p.shape.attention.capacity - p.prefix,
        "sampled COST requires one to three inputs within context");
    std::array<StackVerificationLookup, 3> out{}; auto history = p.history;
    for (size_t t = 0; t < tokens.size(); ++t) {
        history = ngram_step(p.spec, p.policy.ngram, history, tokens[t], p.verification8->lookup_rows[t]);
        out[t] = {history, p.verification8->lookup_rows[t]};
    }
    return out;
}
StackSampledVerificationResult TextStack::verify_sampled_cost3(std::span<const uint32_t> tokens,
    std::span<const float> rows, std::span<const float> memory, uint32_t committed_prefix,
    StackSampledVerificationSelector selector, StackVerificationCancellation cancellation) {
    return verify_sampled_bounded(tokens, rows, memory, committed_prefix, selector, cancellation, 3);
}
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
std::array<StackVerificationLookup, 3> TextStack::sampled_verification_depth2_lookups(std::span<const uint32_t> tokens) const {
    return sampled_verification_cost_lookups(tokens);
}
StackSampledVerificationResult TextStack::verify_sampled_depth2(std::span<const uint32_t> tokens,
    std::span<const float> rows, std::span<const float> memory, uint32_t prefix,
    StackSampledVerificationSelector selector, StackVerificationCancellation cancellation) {
    require(tokens.size() == 3, "depth-two MTP verification requires three target rows");
    return verify_sampled_bounded(tokens, rows, memory, prefix, selector, cancellation, 3);
}
std::array<StackVerificationLookup, 4> TextStack::sampled_verification_depth3_lookups(std::span<const uint32_t> tokens) const {
    require(tokens.size() == 4, "depth-three MTP lookup staging requires four target rows");
    const auto all = sampled_verification8_lookups(tokens);
    return {all[0], all[1], all[2], all[3]};
}
StackSampledVerificationResult TextStack::verify_sampled_depth3(std::span<const uint32_t> tokens,
    std::span<const float> rows, std::span<const float> memory, uint32_t prefix,
    StackSampledVerificationSelector selector, StackVerificationCancellation cancellation) {
    require(tokens.size() == 4, "depth-three MTP verification requires four target rows");
    return verify_sampled_bounded(tokens, rows, memory, prefix, selector, cancellation, 4);
}
std::array<StackVerificationLookup, 8> TextStack::sampled_verification8_lookups(std::span<const uint32_t> tokens) const {
    const auto& p = *impl_; p.check_thread(); p.check_verification8_policy();
    require(p.sampled_verification_prepared && !tokens.empty() && tokens.size() <= 8 &&
        p.prefix <= p.shape.attention.capacity && tokens.size() <= p.shape.attention.capacity - p.prefix,
        "sampled verification8 requires one to eight inputs within context");
    std::array<StackVerificationLookup, 8> out{}; auto history = p.history;
    for (size_t t = 0; t < tokens.size(); ++t) {
        history = ngram_step(p.spec, p.policy.ngram, history, tokens[t], p.verification8->lookup_rows[t]);
        out[t] = {history, p.verification8->lookup_rows[t]};
    }
    return out;
}
StackSampledVerificationResult TextStack::verify_sampled8(std::span<const uint32_t> tokens,
    std::span<const float> rows, std::span<const float> memory, uint32_t committed_prefix,
    StackSampledVerificationSelector selector, StackVerificationCancellation cancellation) {
    return verify_sampled_bounded(tokens, rows, memory, committed_prefix, selector, cancellation, 8);
}
#endif

StackSampledVerificationResult TextStack::verify_sampled_bounded(std::span<const uint32_t> tokens,
    std::span<const float> rows, std::span<const float> memory, uint32_t committed_prefix,
    StackSampledVerificationSelector selector, StackVerificationCancellation cancellation, uint32_t maximum_rows) {
    const auto started = std::chrono::steady_clock::now();
    auto& p = *impl_; p.check_thread(); p.check_verification8_policy();
    require(!tokens.empty() && tokens.size() <= maximum_rows, "sampled transaction input count exceeds its explicit entry bound");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
    require(maximum_rows == 2 || maximum_rows == 3 || maximum_rows == 4 || maximum_rows == 8, "invalid sampled verification entry bound");
    const auto staged = sampled_verification8_lookups(tokens);
#elif defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST)
    require(maximum_rows == 2 || maximum_rows == 3, "invalid sampled COST entry bound");
    const auto staged = sampled_verification_cost_lookups(tokens);
#else
    require(maximum_rows == 2, "production sampled verification remains width two");
    const auto staged = sampled_verification_lookups(tokens);
#endif
    const auto steps = uint32_t(tokens.size()), channels = p.shape.residual.channels, vocabulary = p.shape.vocabulary;
    const auto width = uint64_t(p.shape.residual.branches) * channels;
    const bool has_verify_tap = bool(p.final_hc_tap)
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        || bool(p.dflash_tap)
#endif
        ;
    require(has_verify_tap && p.sampled_verification_prepared && !p.checkpoints &&
        committed_prefix == p.prefix && selector.choose,
        "sampled verification requires matching committed prefix and selector");
    require(rows.size() == uint64_t(steps) * channels && memory.size() == uint64_t(steps) * p.shape.memory.embedding,
        "sampled verification flattened input shape mismatch");
    for (auto value : rows) require(std::isfinite(value), "nonfinite sampled token embedding");
    for (auto value : memory) require(std::isfinite(value), "nonfinite sampled lookup embedding");
    auto& v = *p.verification8;
    const auto logits = [&](uint32_t row) { return std::span(v.host_logits).subspan(uint64_t(row) * vocabulary, vocabulary); };
    StackSampledVerificationResult result;
    result.position = p.prefix;
    result.evaluated.position = p.prefix; result.evaluated.count = steps;
    result.evaluated.residual_width = uint32_t(width);
    p.pending_verification = true;
    struct Pending { bool& value; ~Pending() { value = false; } } pending{p.pending_verification};
    const auto poll = [&] {
        require(!cancellation.requested || !cancellation.requested(cancellation.context),
            "sampled verification cancelled before publication");
        p.check_verification8_policy();
    };
#if defined(TOKSPUR_PRIVATE_CHUNKED_PREFILL) && (defined(TOKSPUR_PRIVATE_KERNEL_QSA_F32) || defined(TOKSPUR_PRIVATE_QSA_SPARSE) || defined(TOKSPUR_PRIVATE_GDN_PORTED))
    // Explicit QSA/GDN diagnostics and ported arms own the bounded cache backup in both arms. Unselected serving is unchanged.
    std::array<GraphCopy,384> cache_save{},cache_restore{};
    const bool rollback_enabled =
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
        p.g.kernel_step2_rollback_enabled() ||
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
        p.g.qsa_sparse_rollback_enabled() ||
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
        p.g.gdn_ported_rollback_enabled() ||
#endif
        false;
    const auto cache_copies=rollback_enabled?p.chunked_cache_copies(steps,cache_save,cache_restore):0;
    if(cache_copies)p.g.copy(std::span(cache_save).first(cache_copies));
#endif
    try {
        poll();
        for (uint32_t t = 0; t < steps; ++t) for (uint32_t b = 0; b < p.shape.residual.branches; ++b)
            std::copy_n(rows.begin() + uint64_t(t) * channels, channels,
                v.host_streams.begin() + uint64_t(t) * width + uint64_t(b) * channels);
        p.g.write(v.streams, std::as_bytes(std::span(v.host_streams)));
        p.g.write(p.g.slice(v.embeddings, 0, uint64_t(memory.size()) * sizeof(float)), std::as_bytes(memory));
        const auto allocations = p.g.memory().allocations;
        // The head and layer estimates use the same selected physical row widths.
        const uint32_t chunks = p.g.q8_many_chunks(steps);
        const bool use_verify_nax =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            p.g.verify_nax() && steps >= 2;
#else
            false;
#endif
        const bool use_topk = p.sampled_logit_topk && selector.allow_topk;
        const auto expected = (
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            use_verify_nax ? p.verify_nax_batch_dispatches(steps) :
#endif
            (
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
            p.verification8_dispatches(steps)
#else
            p.legacy_known_dispatches(steps, true)
#endif
            + uint64_t(steps - 1) * p.final_dispatches() - (steps - chunks))) + p.dflash_tap_dispatches() + (use_topk ? 2ull : 0ull);
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
        p.g.set_mtp_profile_phase(MtpProfilePhase::verification);
#endif
        const auto streams = p.weights.layers.size() % 2 ? v.second : v.streams;
        bool gpu_checked_final = false;
        BatchResult capture_parts{};
        uint64_t route_snapshot_dispatches = 0;
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        uint32_t capture_first_layer = 0;
#endif
        {
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            VerifyNaxScopeGuard scope_guard(p.g, use_verify_nax, true);
#endif
            p.g.begin();
            for (size_t i = 0; i < p.weights.layers.size(); ++i) {
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                p.g.set_carrier_diagnostic_layer(uint32_t(i));
#endif
                p.encode_verification8_layer(i, steps);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (p.dflash_tap) p.encode_dflash_tap_layer(i, i % 2 ? v.streams : v.second, p.dflash_tap->verify_gpu, steps);
#endif
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                // Wide correctness captures are not timing evidence. Keep each
                // recorded batch bounded without changing the serving schedule.
                if (steps >= 5 && p.g.carrier_diagnostic_active() && p.g.carrier_capture_enabled() &&
                    (i + 1) % 16 == 0 && i + 1 < p.weights.layers.size()) {
                    const auto part = p.g.submit();
                    route_snapshot_dispatches += p.g.carrier_route_snapshot_dispatches(uint32_t(i + 1) - capture_first_layer, steps, capture_first_layer);
                    capture_first_layer = uint32_t(i + 1);
                    capture_parts.dispatches += part.dispatches; capture_parts.wall_ms += part.wall_ms;
                    capture_parts.gpu_ms += part.gpu_ms; capture_parts.capture_executes += part.capture_executes;
                    p.g.begin();
                }
#endif
            }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            if (p.g.verify_nax_active() && verify_nax_final_supported(p.g, p.weights.mixer, p.weights.output, v.recurrent.residual, p.shape.residual, vocabulary)) {
                encode_verify_nax_final_logits(p.g, p.weights.mixer, p.weights.output, v.recurrent.residual,
                    streams, v.mixed, v.logits, p.shape.residual, vocabulary, steps, p.policy.attention.rms_epsilon);
                gpu_checked_final = true;
            } else
#endif
            {
                for (uint32_t t = 0; t < steps; ++t) {
                    auto mixer = p.m; mixer.mixed = p.known_row(v.mixed, channels, t);
                    encode_mixer(p.g, p.weights.mixer, mixer, p.known_row(streams, width, t),
                        p.shape.residual, p.policy.attention.rms_epsilon);
                }
                TOKSPUR_OPERATOR_CHECKPOINT(p.g, "hc", "verify_final_mixer", {steps, uint32_t(width), p.shape.residual.rank, p.shape.residual.branches});
                uint32_t at = 0;
                while (at < steps) {
                    const uint32_t n = p.g.q8_many_width(steps - at);
                    p.g.q8_matvec_many(p.weights.output.view,
                        p.g.slice(v.mixed, uint64_t(at) * channels * 4, uint64_t(n) * channels * 4),
                        p.g.slice(v.logits, uint64_t(at) * vocabulary * 4, uint64_t(n) * vocabulary * 4), channels, vocabulary, n);
                    at += n;
                }
                TOKSPUR_OPERATOR_CHECKPOINT(p.g, "dense_q8", "verify_final_logits", {steps, channels, vocabulary, 8});
            }
            if (use_topk) {
                p.g.logit_topk(
                    GpuRowsF32{p.g.slice(v.logits, 0, uint64_t(steps) * vocabulary * sizeof(float)),
                               steps, vocabulary, uint64_t(vocabulary) * sizeof(float)},
                    v.topk_scratch,
                    p.g.slice(v.topk_rows, 0, uint64_t(steps) * sizeof(GpuLogitTopKRow)));
            }
            result.verification = p.g.submit();
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
            route_snapshot_dispatches += p.g.carrier_route_snapshot_dispatches(uint32_t(p.weights.layers.size()) - capture_first_layer, steps, capture_first_layer);
#endif
            result.verification.dispatches += capture_parts.dispatches; result.verification.wall_ms += capture_parts.wall_ms;
            result.verification.gpu_ms += capture_parts.gpu_ms; result.verification.capture_executes += capture_parts.capture_executes;
            if (p.g.last_profile().timings_valid && std::getenv("TOKSPUR_DUMP_VERIFY_PROFILE")) {
                std::map<std::string, std::pair<uint32_t, double>> agg;
                for (const auto& kt : p.g.last_profile().kernels) {
                    auto& e = agg[kt.kernel];
                    e.first += 1;
                    e.second += kt.milliseconds;
                }
                std::cerr << "VERIFY_PROFILE sampled_ms=" << p.g.last_profile().sampled_milliseconds
                          << " gpu_ms=" << result.verification.gpu_ms
                          << " dispatches=" << result.verification.dispatches;
                for (const auto& [kname, kv] : agg) {
                    std::cerr << " " << kname << "=" << kv.first << ":" << kv.second;
                }
                std::cerr << "\n";
            }
        }
        require(result.verification.dispatches == expected + route_snapshot_dispatches && !result.verification.capture_executes &&
            p.g.memory().allocations == allocations, "sampled verification dispatch/allocation invariant failed");
        result.evaluated.vocabulary = vocabulary;
        if (use_topk) {
            p.g.read(p.g.slice(v.topk_rows, 0, uint64_t(steps) * sizeof(GpuLogitTopKRow)),
                     std::as_writable_bytes(std::span(result.evaluated.topk).first(steps)));
            for (uint32_t t = 0; t < steps; ++t) {
                require(result.evaluated.topk[t].vocabulary == vocabulary, "GPU logit top-k vocabulary mismatch");
                require(result.evaluated.topk[t].first_nonfinite_col == 0xffffffffu, "nonfinite sampled target logits");
            }
            result.evaluated.has_topk = true;
        } else {
            const float* shared_logits = nullptr;
            if (p.sampled_logit_views) shared_logits = p.g.host_floats(v.logits);
            else p.g.read(p.g.slice(v.logits, 0, uint64_t(steps) * vocabulary * 4),
                std::as_writable_bytes(std::span(v.host_logits).subspan(vocabulary, uint64_t(steps) * vocabulary)));
            for (uint32_t t = 0; t < steps; ++t) {
                const auto row_logits = shared_logits ? std::span<const float>(shared_logits + uint64_t(t) * vocabulary, vocabulary) : logits(t + 1);
                if (!gpu_checked_final) {
                    TOKSPUR_CARRIER_PHASE(numerical);
#ifdef TOKSPUR_PRIVATE_M5_CPU_CANDIDATES
                    for (size_t base = 0; base < row_logits.size(); base += 4096) {
                        poll();
                        const size_t end = std::min(base + 4096, row_logits.size());
                        uint32_t bad = 0;
                        for (size_t i = base; i < end; ++i) {
                            bad |= uint32_t((std::bit_cast<uint32_t>(row_logits[i]) & 0x7f800000u) == 0x7f800000u);
                        }
                        require(!bad, "nonfinite sampled target logits");
                    }
#else
                    for (size_t i = 0; i < row_logits.size(); ++i) {
                        if (!(i & 4095)) poll();
                        require(std::isfinite(row_logits[i]), "nonfinite sampled target logits");
                    }
#endif
                }
                result.evaluated.logits[t] = row_logits;
            }
        }
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
        if (p.g.carrier_diagnostic_active() && p.g.carrier_capture_enabled()) {
            // Independent full readback, including rejected rows and top-k-only serving paths.
            std::vector<float> complete(uint64_t(steps) * vocabulary);
            p.g.read(p.g.slice(v.logits, 0, complete.size() * sizeof(float)), std::as_writable_bytes(std::span(complete)));
            for (uint32_t t = 0; t < steps; ++t)
                p.g.capture_carrier_logits(std::span<const float>(complete).subspan(uint64_t(t) * vocabulary, vocabulary), t, steps,
                    use_topk ? &result.evaluated.topk[t] : nullptr);
        }
#endif
        if (p.final_hc_tap) {
            if (p.final_hc_gpu_mode) {
                p.capture_final_hc_gpu_only(streams, steps);
                result.evaluated.final_hc_gpu_rows = p.final_hc_candidate_gpu;
            } else {
                p.capture_final_hc(streams, steps);
                result.evaluated.final_hc_gpu_rows = p.final_hc_candidate_gpu;
                result.evaluated.final_hc_rows = std::span<const float>(p.final_hc_tap->candidate).first(uint64_t(steps) * width);
            }
        }
        poll();
        result.committed = selector.choose(selector.context, result.evaluated);
        require(result.committed && result.committed <= steps, "sampled selector returned an invalid consumed prefix");
        poll();
        if (result.committed < steps) {
            uint64_t recurrent_layers = 0;
            for (const auto& state : p.states) recurrent_layers += std::holds_alternative<RecurrentPair>(state);
            if (recurrent_layers) {
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
                p.g.set_mtp_profile_phase(MtpProfilePhase::recovery);
#endif
                p.g.begin();
                for (size_t i = 0; i < p.states.size(); ++i) if (const auto* r = std::get_if<RecurrentPair>(&p.states[i]))
                    encode_gdn_core_tape_prefix(p.g, std::get<RecurrentWeights>(p.weights.layers[i]).attention.convolution,
                        v.tapes[i], r->current, r->next, v.replay, p.shape.recurrent, {8, result.committed});
                result.recovery = p.g.submit();
                require(result.recovery.dispatches == 2 * recurrent_layers && !result.recovery.capture_executes &&
                    p.g.memory().allocations == allocations, "sampled verification prefix recovery invariant failed");
            }
            std::array<GraphCopy, 513> copies{}; size_t count = 0;
            copies[count++] = {p.known_row(v.prefix_memory, p.current_memory.bytes / 4, result.committed - 1), p.next_memory};
            for (size_t i = 0; i < p.states.size(); ++i) if (const auto* a = std::get_if<AttentionPair>(&p.states[i]))
                copies[count++] = {p.known_row(v.prefix_sums[i], p.shape.attention.index_width, result.committed - 1), a->next};
            if (!use_verify_nax) {
                p.g.copy(std::span(copies).first(count));
                count = 0;
            }
#if defined(TOKSPUR_PRIVATE_CHUNKED_PREFILL) && (defined(TOKSPUR_PRIVATE_KERNEL_QSA_F32) || defined(TOKSPUR_PRIVATE_QSA_SPARSE) || defined(TOKSPUR_PRIVATE_GDN_PORTED))
            if(cache_copies){
                const uint64_t row_bytes=uint64_t(p.shape.attention.kv_heads)*p.shape.attention.head_width*2;
                const uint64_t index_bytes=uint64_t(p.shape.attention.index_width)*2;
                const auto compression=p.shape.attention.compression;
                const uint32_t first_index=(p.prefix+result.committed)/compression,last_index=(p.prefix+steps)/compression,saved_index=p.prefix/compression;
                for(size_t i=0;i<p.states.size();++i)if(const auto* state=std::get_if<AttentionPair>(&p.states[i])){
                    const auto& backup=p.chunked->cache_backup[i];
                    const auto bytes=uint64_t(steps-result.committed)*row_bytes;
                    copies[count++]={p.g.slice(backup[0],uint64_t(result.committed)*row_bytes,bytes),p.g.slice(state->cache.keys,uint64_t(p.prefix+result.committed)*row_bytes,bytes)};
                    copies[count++]={p.g.slice(backup[1],uint64_t(result.committed)*row_bytes,bytes),p.g.slice(state->cache.values,uint64_t(p.prefix+result.committed)*row_bytes,bytes)};
                    if(last_index>first_index)copies[count++]={p.g.slice(backup[2],uint64_t(first_index-saved_index)*index_bytes,uint64_t(last_index-first_index)*index_bytes),p.g.slice(state->cache.index_keys,uint64_t(first_index)*index_bytes,uint64_t(last_index-first_index)*index_bytes)};
                }
            }
#endif
            if (count) p.g.copy(std::span(copies).first(count));
        }
        poll();
        require(p.g.memory().allocations == allocations, "sampled selector changed prepared GPU allocation count");
        result.logits = result.evaluated.logits[result.committed - 1];
        result.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        // No fallible work after composite target/tap publication. Ordinary builds retain their existing prefix-hidden future rows. The explicit step-2 diagnostic has restored its rejected physical rows above.
        p.publish_pairs(); p.history = staged[result.committed - 1].next; p.prefix += result.committed;
        p.publish_final_hc(result.committed);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (p.dflash_tap) p.publish_dflash_tap(p.dflash_tap->verify_gpu, result.committed);
#endif
        return result;
    } catch (...) {
        try { p.g.abort(); } catch (const FormatError&) {}
#if defined(TOKSPUR_PRIVATE_CHUNKED_PREFILL) && (defined(TOKSPUR_PRIVATE_KERNEL_QSA_F32) || defined(TOKSPUR_PRIVATE_QSA_SPARSE) || defined(TOKSPUR_PRIVATE_GDN_PORTED))
        if(cache_copies)p.g.copy(std::span(cache_restore).first(cache_copies));
#endif
        throw;
    }
}
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
void TextStack::permit_verification8_branch_reuse() {
    auto& p = *impl_; p.check_thread(); p.check_known_policy();
    require(!p.verification8_branch_reuse && !p.prefix && p.known_capacity == 8 &&
        !p.verification && !p.verification8 && !p.checkpoints && !p.hc_q8_splits,
        "branch-verifier consent requires an empty unprepared capacity-eight stack");
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(!p.wide_q8_exact, "branch verifier refuses scalar wide-Q8 policy");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!p.wide_q8_many, "branch verifier refuses wide-Q8-many policy");
#endif
    p.verification8_branch_reuse = true; // Explicit consent, not prepared storage.
}
#endif
void TextStack::prepare_transient_verification8(uint64_t budget
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    , QsaVerifyRowsConfig qsa_rows
#endif
    ) {
    auto& p = *impl_; p.check_thread();
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    require(!p.chunked
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        || p.preparing_sampled_verification
#endif
        , "chunked prefill requires explicit sampled verifier composition");
#endif
    p.check_known_policy();
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
    require(!p.known_qsa_rows_memory.prepared || p.preparing_sampled_verification,
        "known QSA row spike requires explicit sampled verifier composition");
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    require(uint32_t(qsa_rows.mode) <= 2 && (qsa_rows.prepared || qsa_rows.mode == QsaVerifyRowsMode::off),
        "invalid private verification QSA row preparation/mode");
    if (qsa_rows.prepared) {
        require(p.residual_reuse && p.qsa_channel.prepared && p.qsa_channel.enabled, "private QSA rows require residual reuse and corrected channel policy");
        bool recurrent = false, attention = false;
        for (const auto& layer : p.weights.layers) {
            if (std::holds_alternative<RecurrentWeights>(layer)) recurrent = true;
            else { attention = true; static_cast<void>(attention_tile_dispatches(p.g,
                std::get<AttentionLayerWeights>(layer), 0, 1, {p.shape.residual, p.shape.attention, p.shape.experts},
                AttentionTileMode::core_only, AttentionTileHcMode::combined)); }
        }
        require(recurrent && attention, "private QSA rows require existing recurrent common scratch and QSA layers");
        prepare_attention_tile(p.g, AttentionTileHcMode::combined);
    }
#endif
    const bool history = !p.checkpoints
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
        || p.verification8_branch_reuse
#endif
        ;
    require(!p.verification8 && !p.verification && history && !p.prefix && (p.known_capacity == 8
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
        || ((p.known_capacity == 512 || p.known_capacity == 1024 || p.known_capacity == 2048) && p.chunked && p.preparing_sampled_verification)
#endif
        ) &&
        !p.hc_q8_splits && budget && budget <= maximum_stack_verification_budget,
        "verification8 requires empty exclusive capacity8 direct preparation and bounded budget");
    if (p.checkpoints) p.check_checkpoint_policy();
    require(p.weights.output.encoding == 8 && p.shape.residual.channels % 32 == 0,
        "verification8 requires an original Q8 vocabulary head");
    const auto width = uint64_t(p.shape.residual.branches) * p.shape.residual.channels;
    const auto host_minimum = checked_add(sizeof(VerificationStore8), checked_mul(9ull * p.shape.vocabulary + 8 * width, 4));
    require(host_minimum < budget, "verification8 host rows exceed explicit combined budget");
    auto store = std::make_unique<VerificationStore8>();
    store->tapes.resize(p.states.size()); store->prefix_sums.resize(p.states.size());
    store->owned.reserve(p.states.size() * 7 + 52
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        + (qsa_rows.prepared ? 26 : 0)
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
        + 3
#endif
        );
    store->host_streams.resize(8 * width); store->host_logits.resize(9ull * p.shape.vocabulary);
    for (auto& row : store->lookup_rows) row.resize(p.spec.offsets.size());
    store->host_charged_bytes = sizeof(VerificationStore8) + store->tapes.capacity() * sizeof(GdnCoreTape) +
        (store->prefix_sums.capacity() + store->owned.capacity()) * sizeof(GpuView) +
        (store->host_streams.capacity() + store->host_logits.capacity()) * sizeof(float);
    for (const auto& row : store->lookup_rows) store->host_charged_bytes += row.capacity() * sizeof(uint64_t);
    require(store->host_charged_bytes < budget, "verification8 actual host capacities exceed combined budget");
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    if (p.g.verify_nax()) p.prepare_stack_verify_nax();
#endif
    if (p.sampled_logit_topk) p.g.prepare_logit_topk();
    const auto before = p.g.memory().live_bytes;
    const auto admit = [&] {
        store->gpu_charged_bytes = p.g.memory().live_bytes - before;
        require(checked_add(store->gpu_charged_bytes, store->host_charged_bytes) <= budget,
            "verification8 actual GPU/host charges exceed combined budget");
    };
    const auto keep = [&](auto views) { for (auto view : views) store->owned.push_back(view); admit(); };
    const auto zeros = [&](uint64_t count, const char* label) {
        const auto view = p.g.zeros(checked_mul(count, 4), label); store->owned.push_back(view); admit(); return view;
    };
    const auto zeros_bytes = [&](uint64_t bytes, const char* label) {
        const auto view = p.g.zeros(bytes, label); store->owned.push_back(view); admit(); return view;
    };
    try {
        store->streams = zeros(8 * width, "verification8 branch rows");
        store->second = zeros(8 * width, "verification8 alternating rows");
        store->memory_streams = zeros(8 * width, "verification8 PLE rows");
        store->embeddings = zeros(8ull * p.shape.memory.embedding, "verification8 lookup embeddings");
        store->prefix_memory = zeros(7 * (p.current_memory.bytes / 4), "verification8 PLE prefix histories");
        store->mixed = zeros(8ull * p.shape.residual.channels, "verification8 mixed head inputs");
        store->logits = zeros(8ull * p.shape.vocabulary, "verification8 full vocabulary rows");
        if (p.sampled_logit_topk) {
            store->topk_scratch = zeros_bytes(MetalGraph::logit_topk_scratch_bytes(8, p.shape.vocabulary),
                "verification8 logit top-k partial scratch");
            store->topk_rows = zeros_bytes(MetalGraph::logit_topk_output_bytes(8),
                "verification8 logit top-k output rows");
        }
        bool recurrent = false;
        for (size_t i = 0; i < p.states.size(); ++i) {
            if (std::holds_alternative<RecurrentPair>(p.states[i])) {
                recurrent = true; store->tapes[i] = allocate_gdn_core_tape(p.g, p.shape.recurrent, 8);
                keep(store->tapes[i].views());
                for (auto view : store->tapes[i].views()) store->tape_payload_bytes += view.bytes;
            } else store->prefix_sums[i] = zeros(7ull * p.shape.attention.index_width, "verification8 QSA prefix sums");
        }
        if (recurrent) {
            store->recurrent = allocate_recurrent_tile_scratch(p.g, {p.shape.residual, p.shape.recurrent, p.shape.experts}, 8);
            keep(store->recurrent.views());
            store->replay = allocate_gdn_core_replay_scratch(p.g, p.shape.recurrent, 8);
            keep(std::array{store->replay.convolution_output, store->replay.delta_output});
            p.g.prepare_gdn_tiles(128);
        }
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        store->qsa_rows_memory.config = qsa_rows;
        if (qsa_rows.prepared) {
            const auto prior = p.g.memory();
            store->qsa_rows = allocate_attention_tile_qsa_scratch(p.g, {p.shape.residual, p.shape.attention, p.shape.experts});
            // Register ALL newly returned owners before admission may throw.
            // Reserved capacity covers26 entries, so failed admission releases all.
            for (auto view : store->qsa_rows.views()) store->owned.push_back(view);
            admit();
            store->qsa_scores = zeros(maximum_qsa_verify_rows_scores / 4, "verification QSA row scores");
            store->qsa_factors = zeros(maximum_qsa_verify_rows_factors / 4, "verification QSA row factors");
            uint64_t logical = store->qsa_scores.bytes + store->qsa_factors.bytes;
            for (auto view : store->qsa_rows.views()) logical += view.bytes;
            const auto after = p.g.memory();
            require(after.allocations == prior.allocations + 26 && after.live_bytes >= prior.live_bytes + logical,
                "verification QSA row workspace must charge exactly26 GPU owners");
            store->qsa_rows_memory = {qsa_rows, logical, after.live_bytes - prior.live_bytes, 26, 19};
        }
#endif
        p.g.prepare_q8_matvec_many_all(); // Head uses these even without recurrent layers.
        admit();
    } catch (...) {
        for (auto i = store->owned.rbegin(); i != store->owned.rend(); ++i) p.g.release(*i);
        throw;
    }
    store->kernel_policy = p.g.kernel_policy(); store->residual_reuse = p.residual_reuse;
    p.verification8 = std::move(store); // No request-time allocation or partial preparation publication.
}
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
QsaVerifyRowsMemory TextStack::verification_qsa_rows_memory() const {
    const auto& p = *impl_; p.check_thread();
    return p.verification8 ? p.verification8->qsa_rows_memory : QsaVerifyRowsMemory{};
}
#endif
StackVerificationLookups8 TextStack::verification_lookups8(std::span<const uint32_t> tokens,
    uint32_t remaining, std::span<const uint32_t> stops) const {
    const auto& p = *impl_; p.check_thread(); p.check_verification8_policy();
    require(tokens.size() >= 2 && tokens.size() <= 8 && p.prefix <= p.shape.attention.capacity &&
        remaining <= 4096 && stops.size() <= p.shape.vocabulary, "invalid verification8 reservation");
    for (auto token : tokens) require(token < p.shape.vocabulary, "verification8 token outside vocabulary");
    for (auto token : stops) require(token < p.shape.vocabulary, "verification8 stop outside vocabulary");
    StackVerificationLookups8 out;
    auto steps = std::min({uint32_t(tokens.size()), remaining, p.shape.attention.capacity - p.prefix});
    for (uint32_t i = 0; i < steps; ++i) if (std::find(stops.begin(), stops.end(), tokens[i]) != stops.end()) { steps = i + 1; break; }
    if (steps < 2) return out; // First stop or short remaining/context reservation uses ordinary decoding.
    out.count = steps; auto previous = p.history;
    for (uint32_t i = 0; i < steps; ++i) {
        previous = ngram_step(p.spec, p.policy.ngram, previous, tokens[i], p.verification8->lookup_rows[i]);
        out.rows[i] = {previous, p.verification8->lookup_rows[i]};
    }
    return out;
}
StackVerificationResult8 TextStack::verify_greedy8(std::span<const uint32_t> tokens,
    std::span<const float> rows, std::span<const float> memory, uint32_t committed_prefix,
    std::span<const float> committed_logits, uint32_t remaining, std::span<const uint32_t> stops,
    StackVerificationCancellation cancellation) {
    const auto started = std::chrono::steady_clock::now(); auto& p = *impl_; p.check_thread(); p.check_verification8_policy();
    const auto staged = verification_lookups8(tokens, remaining, stops); const auto steps = staged.count;
    require(p.prefix && committed_prefix == p.prefix && committed_logits.size() == p.shape.vocabulary,
        "verification8 requires authoritative committed logits/prefix");
    const auto channels = p.shape.residual.channels, vocabulary = p.shape.vocabulary;
    const auto width = uint64_t(p.shape.residual.branches) * channels;
    require(rows.size() == checked_mul(tokens.size(), channels) && memory.size() == checked_mul(tokens.size(), p.shape.memory.embedding),
        "verification8 flattened input rows differ");
    for (auto value : rows) require(std::isfinite(value), "nonfinite verification8 token embedding");
    for (auto value : memory) require(std::isfinite(value), "nonfinite verification8 lookup embedding");
    auto& v = *p.verification8;
    const auto logits = [&](uint32_t row) { return std::span(v.host_logits).subspan(uint64_t(row) * vocabulary, vocabulary); };
    std::copy(committed_logits.begin(), committed_logits.end(), logits(0).begin());
    const auto first = greedy_id(committed_logits); StackVerificationResult8 result{0, p.prefix, 0, 0, logits(0)};
    result.source_logits[0] = committed_logits[first];
    p.pending_verification = true;
    struct Pending { bool& value; ~Pending() { value = false; } } pending{p.pending_verification};
    const auto poll = [&] {
        require(!cancellation.requested || !cancellation.requested(cancellation.context), "verification8 cancelled before publication");
        p.check_verification8_policy();
    };
    try {
        poll();
        if (!steps || tokens[0] != first) { result.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); return result; }
        for (uint32_t t = 0; t < steps; ++t) for (uint32_t b = 0; b < p.shape.residual.branches; ++b)
            std::copy_n(rows.begin() + uint64_t(t) * channels, channels, v.host_streams.begin() + uint64_t(t) * width + uint64_t(b) * channels);
        p.g.write(v.streams, std::as_bytes(std::span(v.host_streams)));
        p.g.write(p.g.slice(v.embeddings, 0, uint64_t(steps) * p.shape.memory.embedding * 4), std::as_bytes(memory.first(uint64_t(steps) * p.shape.memory.embedding)));
        const uint32_t chunks = p.g.q8_many_chunks(steps);
        const auto allocations = p.g.memory().allocations;
        const bool use_verify_nax =
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            p.g.verify_nax() && steps >= 2;
#else
            false;
#endif
        const auto expected = (
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            use_verify_nax ? p.verify_nax_batch_dispatches(steps) :
#endif
            (
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
            p.verification8_dispatches(steps)
#else
            p.legacy_known_dispatches(steps, true)
#endif
            + uint64_t(steps - 1) * p.final_dispatches() - (steps - chunks))) + p.dflash_tap_dispatches();
        const auto streams = p.weights.layers.size() % 2 ? v.second : v.streams;
        bool used_nax_final = false;
        {
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            VerifyNaxScopeGuard scope_guard(p.g, use_verify_nax, true);
#endif
            p.g.begin();
            for (size_t i = 0; i < p.weights.layers.size(); ++i) {
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
                p.g.set_carrier_diagnostic_layer(uint32_t(i));
#endif
                p.encode_verification8_layer(i, steps);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (p.dflash_tap) p.encode_dflash_tap_layer(i, i % 2 ? v.streams : v.second, p.dflash_tap->verify_gpu, steps);
#endif
            }
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
            if (p.g.verify_nax_active() && verify_nax_final_supported(p.g, p.weights.mixer, p.weights.output, v.recurrent.residual, p.shape.residual, vocabulary)) {
                encode_verify_nax_final_logits(p.g, p.weights.mixer, p.weights.output, v.recurrent.residual,
                    streams, v.mixed, v.logits, p.shape.residual, vocabulary, steps, p.policy.attention.rms_epsilon);
                used_nax_final = true;
            } else
#endif
            {
                for (uint32_t t = 0; t < steps; ++t) {
                    auto mixer = p.m; mixer.mixed = p.known_row(v.mixed, channels, t);
                    encode_mixer(p.g, p.weights.mixer, mixer, p.known_row(streams, width, t), p.shape.residual, p.policy.attention.rms_epsilon);
                }
                TOKSPUR_OPERATOR_CHECKPOINT(p.g, "hc", "verify_final_mixer", {steps, uint32_t(width), p.shape.residual.rank, p.shape.residual.branches});
                uint32_t at = 0;
                while (at < steps) {
                    const uint32_t n = p.g.q8_many_width(steps - at);
                    p.g.q8_matvec_many(p.weights.output.view,
                        p.g.slice(v.mixed, uint64_t(at) * channels * 4, uint64_t(n) * channels * 4),
                        p.g.slice(v.logits, uint64_t(at) * vocabulary * 4, uint64_t(n) * vocabulary * 4), channels, vocabulary, n);
                    at += n;
                }
                TOKSPUR_OPERATOR_CHECKPOINT(p.g, "dense_q8", "verify_final_logits", {steps, channels, vocabulary, 8});
            }
            result.verification = p.g.submit();
            if (p.g.last_profile().timings_valid && std::getenv("TOKSPUR_DUMP_VERIFY_PROFILE")) {
                std::map<std::string, std::pair<uint32_t, double>> agg;
                for (const auto& kt : p.g.last_profile().kernels) {
                    auto& e = agg[kt.kernel];
                    e.first += 1;
                    e.second += kt.milliseconds;
                }
                std::cerr << "VERIFY_PROFILE sampled_ms=" << p.g.last_profile().sampled_milliseconds
                          << " gpu_ms=" << result.verification.gpu_ms
                          << " dispatches=" << result.verification.dispatches;
                for (const auto& [kname, kv] : agg) {
                    std::cerr << " " << kname << "=" << kv.first << ":" << kv.second;
                }
                std::cerr << "\n";
            }
        }
        require(result.verification.dispatches == expected && !result.verification.capture_executes && p.g.memory().allocations == allocations,
            "verification8 dispatch/allocation invariant failed");
        result.evaluated = steps; result.head_dispatches = used_nax_final ? 1u : chunks;
        p.g.read(p.g.slice(v.logits, 0, uint64_t(steps) * vocabulary * 4), std::as_writable_bytes(std::span(v.host_logits).subspan(vocabulary, uint64_t(steps) * vocabulary)));
        for (uint32_t t = 1; t <= steps; ++t) static_cast<void>(greedy_id(logits(t))); // Every evaluated row must be finite.
        result.accepted = 1;
        for (uint32_t t = 1; t < steps && tokens[t] == greedy_id(logits(t)); ++t) {
            result.source_logits[t] = logits(t)[tokens[t]]; ++result.accepted;
        }
        poll();
        if (result.accepted < steps) {
            uint64_t recurrent_layers = 0;
            for (const auto& state : p.states) recurrent_layers += std::holds_alternative<RecurrentPair>(state);
            if (recurrent_layers) {
                p.g.begin();
                for (size_t i = 0; i < p.states.size(); ++i) if (const auto* r = std::get_if<RecurrentPair>(&p.states[i]))
                    encode_gdn_core_tape_prefix(p.g, std::get<RecurrentWeights>(p.weights.layers[i]).attention.convolution,
                        v.tapes[i], r->current, r->next, v.replay, p.shape.recurrent, {8, result.accepted});
                result.recovery = p.g.submit();
                require(result.recovery.dispatches == 2 * recurrent_layers && !result.recovery.capture_executes && p.g.memory().allocations == allocations,
                    "verification8 prefix recovery invariant failed");
            }
            std::array<GraphCopy, 129> copies{}; size_t count = 0;
            copies[count++] = {p.known_row(v.prefix_memory, p.current_memory.bytes / 4, result.accepted - 1), p.next_memory};
            for (size_t i = 0; i < p.states.size(); ++i) if (const auto* a = std::get_if<AttentionPair>(&p.states[i]))
                copies[count++] = {p.known_row(v.prefix_sums[i], p.shape.attention.index_width, result.accepted - 1), a->next};
            p.g.copy(std::span(copies).first(count)); poll();
        }
        result.logits = logits(result.accepted);
        p.capture_final_hc(streams, steps);
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
        result.evaluated_logits = std::span<const float>(v.host_logits).subspan(vocabulary, uint64_t(steps) * vocabulary);
#endif
        result.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        // Only future QSA rows were appended. Prefix bounds hide rejected tails;
        // later appends overwrite them. No fallible work after composite publication.
        p.publish_pairs(); p.history = staged.rows[result.accepted - 1].next; p.prefix += result.accepted;
        p.publish_final_hc(result.accepted);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (p.dflash_tap) p.publish_dflash_tap(p.dflash_tap->verify_gpu, result.accepted);
#endif
        return result;
    } catch (...) { try { p.g.abort(); } catch (const FormatError&) {} throw; }
}
#endif
void TextStack::prepare_checkpoints(StackCheckpointConfig config) {
    auto& p = *impl_; p.check_thread();
#ifdef TOKSPUR_PRIVATE_CHUNKED_PREFILL
    require(!p.chunked, "chunked prefill refuses checkpoint or ordinary verifier stores");
#endif
    require(!p.final_hc_tap
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        && !p.dflash_tap
#endif
        , "persistent checkpoints do not retain the final HC or DFlash tap");
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
    require(!p.known_qsa_rows_memory.prepared, "known QSA row spike does not admit checkpoint composition");
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
    require(!p.verification8
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
        || p.verification8_branch_reuse
#endif
        , "verification8 requires explicit private consent for persistent checkpoints");
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    require(p.hc_exact == HcExactMode::off
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
        || p.verification8_branch_reuse
#endif
        , "private exact-HC requires explicit private branch-verifier consent for checkpoints");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    require(!p.wide_q8_exact, "private wide-Q8 first cohort refuses checkpoints");
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    require(!p.wide_q8_many, "private wide-Q8-many first cohort refuses checkpoints");
#endif
    require(!p.checkpoints && !p.prefix && p.execution == StackExecution::direct && !p.replay_prepared,
        "checkpoint preparation requires unprepared empty direct stack");
    require(config.slots && config.slots <= maximum_stack_checkpoint_slots &&
        config.byte_budget && config.byte_budget <= maximum_stack_checkpoint_budget,
        "checkpoint slot/byte budget limit");
    if (p.known_capacity) p.check_known_policy();
    if (p.verification) p.check_verification_policy();
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
    if (p.verification8) p.check_verification8_policy();
#endif
    static_cast<void>(p.g.last_profile());
    const auto payload = p.checkpoint_payload_bytes();
    require(checked_add(checked_mul(payload, config.slots), sizeof(CheckpointStore)) <= config.byte_budget,
        "checkpoint payload exceeds explicit budget");
    auto store = std::make_unique<CheckpointStore>();
    store->owner = checkpoint_owner(); store->capacity = config.slots; store->payload_bytes = payload;
    store->kernel_policy = p.g.kernel_policy(); store->residual_reuse = p.residual_reuse;
    const auto before = p.g.memory().live_bytes;
    try {
        for (uint32_t i = 0; i < config.slots; ++i) {
            store->slots[i].storage = p.g.zeros(payload, "retained composite checkpoint");
            store->gpu_charged_bytes = p.g.memory().live_bytes - before;
            require(checked_add(store->gpu_charged_bytes, sizeof(CheckpointStore)) <= config.byte_budget,
                "actual checkpoint allocations exceed explicit budget");
        }
    } catch (...) {
        for (auto& slot : store->slots) if (slot.storage.owner) p.g.release(slot.storage);
        throw;
    }
    // Publication is a nonthrowing pointer move after all allocations/admission.
    p.checkpoints = std::move(store);
}
StackCheckpointMemory TextStack::checkpoint_memory() const {
    const auto& p = *impl_; p.check_thread();
    if (!p.checkpoints) return {};
    const auto& store = *p.checkpoints; uint32_t retained = 0;
    for (const auto& slot : store.slots) retained += slot.valid;
    return {store.capacity, retained, store.payload_bytes, store.gpu_charged_bytes, sizeof(CheckpointStore)};
}
StackCheckpoint TextStack::retain_checkpoint() {
    auto& p = *impl_; p.check_thread(); p.check_checkpoint_policy();
    auto& store = *p.checkpoints;
    require(store.generation < std::numeric_limits<uint64_t>::max(), "checkpoint generation space exhausted");
    uint32_t index = 0;
    for (uint32_t i = 0; i < store.capacity; ++i) {
        if (!store.slots[i].valid) { index = i; break; }
        if (store.slots[i].generation < store.slots[index].generation) index = i;
    }
    auto& slot = store.slots[index]; std::array<GraphCopy, 257> copies{};
    const auto count = p.checkpoint_copies(slot.storage, false, copies);
    p.g.copy(std::span(copies).first(count));
    // The complete copy was prevalidated; no fallible work remains. Replacing
    // this generation atomically retires its previous opaque handle.
    slot.history = p.history; slot.prefix = p.prefix; slot.generation = ++store.generation; slot.valid = true;
    return {store.owner, slot.generation, index};
}
uint32_t TextStack::restore_checkpoint(StackCheckpoint checkpoint) {
    auto& p = *impl_; p.check_thread(); p.check_checkpoint_policy();
    auto& store = *p.checkpoints;
    require(checkpoint.owner_ == store.owner && checkpoint.slot_ < store.capacity,
        "foreign or invalid stack checkpoint");
    const auto& slot = store.slots[checkpoint.slot_];
    require(slot.valid && checkpoint.generation_ == slot.generation && slot.prefix <= p.prefix,
        "stale or non-ancestor stack checkpoint");
    std::array<GraphCopy, 257> copies{};
    const auto count = p.checkpoint_copies(slot.storage, true, copies);
    p.g.copy(std::span(copies).first(count));
    // Shared QSA rows below slot.prefix still belong to this lineage. Discard
    // every descendant before a later append can overwrite any of their rows.
    // These scalar/array assignments and bank swaps cannot throw.
    for (auto& other : store.slots) if (other.valid && other.prefix > slot.prefix) other.valid = false;
    p.publish_pairs(); p.history = slot.history; p.prefix = slot.prefix;
    return p.prefix;
}
void TextStack::truncate_shared_kv(uint32_t prefix) {
    auto& p = *impl_; p.check_thread();
    require(prefix <= p.prefix, "truncate_shared_kv is ancestor-only");
    if (p.checkpoints) {
        for (auto& other : p.checkpoints->slots) {
            if (other.valid && other.prefix > prefix) other.valid = false;
        }
    }
    p.prefix = prefix;
}
uint64_t TextStack::ssd_checkpoint_payload_bytes(uint32_t prefix) const {
    const auto& p = *impl_; p.check_thread();
    require(prefix <= p.shape.attention.capacity, "SSD checkpoint prefix exceeds stack capacity");
    uint64_t total = sizeof(uint32_t) + sizeof(p.history.previous);
    if (p.final_hc_tap && prefix > 0) {
        total = checked_add(total, checked_mul(uint64_t(p.final_hc_tap->width), sizeof(float)));
    }
    total = checked_add(total, p.current_memory.bytes);
    for (const auto& state : p.states) {
        if (const auto* r = std::get_if<RecurrentPair>(&state)) {
            total = checked_add(total, r->current.matrix.bytes);
            total = checked_add(total, r->current.convolution.bytes);
        } else {
            const auto& a = std::get<AttentionPair>(state);
            total = checked_add(total, a.current.bytes);
            const uint64_t kv_bytes = uint64_t(prefix) * p.shape.attention.kv_heads * p.shape.attention.head_width * 2;
            total = checked_add(total, checked_mul(kv_bytes, 2));
            const uint64_t index_bytes = uint64_t(prefix / p.shape.attention.compression) * p.shape.attention.index_width * 2;
            total = checked_add(total, index_bytes);
        }
    }
    return total;
}
uint64_t TextStack::ssd_checkpoint_payload_bytes() const {
    const auto& p = *impl_; p.check_thread();
    return ssd_checkpoint_payload_bytes(p.prefix);
}
bool TextStack::read_ssd_checkpoint_span(uint64_t offset, std::span<std::byte> target) const {
    const auto& p = *impl_; p.check_thread();
    if (p.pending_verification) return false;
    const uint64_t total = ssd_checkpoint_payload_bytes(p.prefix);
    if (offset > total || target.size() > total - offset) return false;
    if (target.empty()) return true;
    const uint64_t req_end = offset + target.size();
    uint64_t cursor = 0;
    const auto copy_host = [&](std::span<const std::byte> src) {
        const uint64_t seg_start = cursor, seg_end = cursor + src.size();
        cursor = seg_end;
        if (seg_end <= offset || seg_start >= req_end) return;
        const uint64_t ov_start = std::max(seg_start, offset);
        const uint64_t ov_len = std::min(seg_end, req_end) - ov_start;
        std::memcpy(target.data() + (ov_start - offset), src.data() + (ov_start - seg_start), size_t(ov_len));
    };
    const auto copy_gpu = [&](GpuView src) {
        const uint64_t seg_start = cursor, seg_end = cursor + src.bytes;
        cursor = seg_end;
        if (seg_end <= offset || seg_start >= req_end) return;
        const uint64_t ov_start = std::max(seg_start, offset);
        const uint64_t ov_len = std::min(seg_end, req_end) - ov_start;
        p.g.read(p.g.slice(src, ov_start - seg_start, ov_len), target.subspan(size_t(ov_start - offset), size_t(ov_len)));
    };
    copy_host(std::as_bytes(std::span(&p.prefix, 1)));
    copy_host(std::as_bytes(std::span(p.history.previous)));
    if (p.final_hc_tap && p.prefix > 0) {
        if (p.final_hc_tap->count == 0) return false;
        p.ensure_final_hc_host();
        const auto rows = std::span<const float>(p.final_hc_tap->current).first(uint64_t(p.final_hc_tap->count) * p.final_hc_tap->width);
        copy_host(std::as_bytes(rows.last(p.final_hc_tap->width)));
    }
    copy_gpu(p.current_memory);
    for (const auto& state : p.states) {
        if (const auto* r = std::get_if<RecurrentPair>(&state)) {
            copy_gpu(r->current.matrix);
            copy_gpu(r->current.convolution);
        } else {
            const auto& a = std::get<AttentionPair>(state);
            copy_gpu(a.current);
            const uint64_t kv_bytes = uint64_t(p.prefix) * p.shape.attention.kv_heads * p.shape.attention.head_width * 2;
            if (kv_bytes) {
                copy_gpu(p.g.slice(a.cache.keys, 0, kv_bytes));
                copy_gpu(p.g.slice(a.cache.values, 0, kv_bytes));
            }
            const uint64_t index_bytes = uint64_t(p.prefix / p.shape.attention.compression) * p.shape.attention.index_width * 2;
            if (index_bytes) copy_gpu(p.g.slice(a.cache.index_keys, 0, index_bytes));
        }
    }
    return true;
}
bool TextStack::write_ssd_checkpoint_span(uint32_t target_prefix, uint64_t offset, std::span<const std::byte> source) {
    auto& p = *impl_; p.check_thread();
    if (p.pending_verification || target_prefix > p.shape.attention.capacity) return false;
    const uint64_t total = ssd_checkpoint_payload_bytes(target_prefix);
    if (offset > total || source.size() > total - offset) return false;
    if (offset == 0) {
        if (!source.empty()) {
            if (source.size() < sizeof(uint32_t)) return false;
            uint32_t check_prefix = 0;
            std::memcpy(&check_prefix, source.data(), sizeof(uint32_t));
            if (check_prefix != target_prefix) return false;
        }
        if (p.checkpoints) {
            for (auto& other : p.checkpoints->slots) {
                if (other.valid && other.prefix > target_prefix) other.valid = false;
            }
        }
        p.prefix = target_prefix;
        if (p.final_hc_tap) p.final_hc_tap->count = target_prefix > 0 ? 1 : 0;
        p.final_hc_current_gpu = {}; p.final_hc_candidate_gpu = {};
        p.final_hc_gpu_valid = false; p.final_hc_host_valid = target_prefix > 0; p.final_hc_candidate_host_valid = false;
    }
    if (source.empty()) return true;
    const uint64_t req_end = offset + source.size();
    uint64_t cursor = 0;
    uint32_t stored_prefix = target_prefix;
    const auto write_host = [&](std::span<std::byte> dst) {
        const uint64_t seg_start = cursor, seg_end = cursor + dst.size();
        cursor = seg_end;
        if (seg_end <= offset || seg_start >= req_end) return;
        const uint64_t ov_start = std::max(seg_start, offset);
        const uint64_t ov_len = std::min(seg_end, req_end) - ov_start;
        std::memcpy(dst.data() + (ov_start - seg_start), source.data() + (ov_start - offset), size_t(ov_len));
    };
    const auto write_gpu = [&](GpuView dst) {
        const uint64_t seg_start = cursor, seg_end = cursor + dst.bytes;
        cursor = seg_end;
        if (seg_end <= offset || seg_start >= req_end) return;
        const uint64_t ov_start = std::max(seg_start, offset);
        const uint64_t ov_len = std::min(seg_end, req_end) - ov_start;
        p.g.write(p.g.slice(dst, ov_start - seg_start, ov_len), source.subspan(size_t(ov_start - offset), size_t(ov_len)));
    };
    write_host(std::as_writable_bytes(std::span(&stored_prefix, 1)));
    if (offset == 0 && source.size() >= sizeof(uint32_t) && stored_prefix != target_prefix) return false;
    write_host(std::as_writable_bytes(std::span(p.history.previous)));
    if (p.final_hc_tap && target_prefix > 0) {
        auto row = std::span<float>(p.final_hc_tap->current).first(p.final_hc_tap->width);
        write_host(std::as_writable_bytes(row));
        p.final_hc_tap->count = 1;
        p.final_hc_host_valid = true;
    }
    write_gpu(p.current_memory);
    for (auto& state : p.states) {
        if (auto* r = std::get_if<RecurrentPair>(&state)) {
            write_gpu(r->current.matrix);
            write_gpu(r->current.convolution);
        } else {
            auto& a = std::get<AttentionPair>(state);
            write_gpu(a.current);
            const uint64_t kv_bytes = uint64_t(target_prefix) * p.shape.attention.kv_heads * p.shape.attention.head_width * 2;
            if (kv_bytes) {
                write_gpu(p.g.slice(a.cache.keys, 0, kv_bytes));
                write_gpu(p.g.slice(a.cache.values, 0, kv_bytes));
            }
            const uint64_t index_bytes = uint64_t(target_prefix / p.shape.attention.compression) * p.shape.attention.index_width * 2;
            if (index_bytes) write_gpu(p.g.slice(a.cache.index_keys, 0, index_bytes));
        }
    }
    p.prefix = target_prefix;
    return true;
}
#if defined(TOKSPUR_PRIVATE_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_CONCURRENT_DISPATCH)
TextStack::AcceptedFingerprint TextStack::accepted_fingerprint(std::span<std::byte> scratch) const {
    const auto& p = *impl_; p.check_thread();
    require(!p.pending_verification && !scratch.empty() && scratch.size() <= 1000000,
        "invalid or in-flight verification fingerprint scratch");
    AcceptedFingerprint out; CC_SHA256_CTX hash; require(CC_SHA256_Init(&hash), "cannot initialize accepted-state fingerprint");
    const auto append = [&](std::span<const std::byte> bytes) {
        require(bytes.size() <= std::numeric_limits<CC_LONG>::max() && CC_SHA256_Update(&hash, bytes.data(), CC_LONG(bytes.size())),
            "cannot update accepted-state fingerprint"); out.bytes = checked_add(out.bytes, bytes.size());
    };
    const auto view = [&](GpuView source) {
        for (uint64_t at = 0; at < source.bytes;) {
            const auto count = std::min<uint64_t>(scratch.size(), source.bytes - at);
            p.g.read(p.g.slice(source, at, count), scratch.first(count)); append(scratch.first(count)); at += count;
        }
    };
    // Same stable order and visible-prefix bounds as accepted_snapshot().
    append(std::as_bytes(std::span(&p.prefix, 1))); append(std::as_bytes(std::span(p.history.previous))); view(p.current_memory);
    for (const auto& state : p.states) {
        if (const auto* r = std::get_if<RecurrentPair>(&state)) { view(r->current.matrix); view(r->current.convolution); }
        else {
            const auto& a = std::get<AttentionPair>(state); view(a.current);
            const uint64_t bytes = uint64_t(p.prefix) * p.shape.attention.kv_heads * p.shape.attention.head_width * 2;
            if (bytes) { view(p.g.slice(a.cache.keys, 0, bytes)); view(p.g.slice(a.cache.values, 0, bytes)); }
            const uint64_t index_bytes = uint64_t(p.prefix / p.shape.attention.compression) * p.shape.attention.index_width * 2;
            if (index_bytes) view(p.g.slice(a.cache.index_keys, 0, index_bytes));
        }
    }
    require(CC_SHA256_Final(out.sha256.data(), &hash), "cannot finalize accepted-state fingerprint"); return out;
}
#endif
std::vector<std::byte> TextStack::accepted_snapshot() const {
    const auto& p = *impl_; p.check_thread(); std::vector<GpuView> views;
    views.reserve(p.states.size() * 4 + 1); views.push_back(p.current_memory);
    for (const auto& state : p.states) {
        if (const auto* r = std::get_if<RecurrentPair>(&state)) { views.push_back(r->current.matrix); views.push_back(r->current.convolution); }
        else {
            const auto& a = std::get<AttentionPair>(state); views.push_back(a.current);
            const uint64_t bytes = uint64_t(p.prefix) * p.shape.attention.kv_heads * p.shape.attention.head_width * 2;
            if (bytes) { views.push_back(p.g.slice(a.cache.keys, 0, bytes)); views.push_back(p.g.slice(a.cache.values, 0, bytes)); }
            const uint64_t index_bytes = uint64_t(p.prefix / p.shape.attention.compression) * p.shape.attention.index_width * 2;
            if (index_bytes) views.push_back(p.g.slice(a.cache.index_keys, 0, index_bytes));
        }
    }
    uint64_t total = sizeof(p.prefix) + sizeof(p.history.previous);
    for (auto v : views) total = checked_add(total, v.bytes);
    require(total <= 8 * 1024 * 1024, "diagnostic stack snapshot exceeds 8-MiB ceiling");
    std::vector<std::byte> out; out.reserve(total);
    const auto append = [&](std::span<const std::byte> bytes) { out.insert(out.end(), bytes.begin(), bytes.end()); };
    append(std::as_bytes(std::span(&p.prefix, 1))); append(std::as_bytes(std::span(p.history.previous)));
    for (auto v : views) append(p.g.read(v));
    return out;
}
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
std::string TextStack::accepted_state_sha256() const {
    const auto& p = *impl_; p.check_thread(); std::vector<GpuView> views;
    views.reserve(p.states.size() * 4 + 1); views.push_back(p.current_memory);
    for (const auto& state : p.states) {
        if (const auto* r = std::get_if<RecurrentPair>(&state)) { views.push_back(r->current.matrix); views.push_back(r->current.convolution); }
        else {
            const auto& a = std::get<AttentionPair>(state); views.push_back(a.current);
            const uint64_t bytes = uint64_t(p.prefix) * p.shape.attention.kv_heads * p.shape.attention.head_width * 2;
            if (bytes) { views.push_back(p.g.slice(a.cache.keys, 0, bytes)); views.push_back(p.g.slice(a.cache.values, 0, bytes)); }
            const uint64_t index_bytes = uint64_t(p.prefix / p.shape.attention.compression) * p.shape.attention.index_width * 2;
            if (index_bytes) views.push_back(p.g.slice(a.cache.index_keys, 0, index_bytes));
        }
    }
    CC_SHA256_CTX context; CC_SHA256_Init(&context);
    const auto feed = [&](std::span<const std::byte> bytes) {
        size_t done = 0;
        while (done < bytes.size()) {
            const auto count = static_cast<CC_LONG>(std::min<size_t>(bytes.size() - done, 1024 * 1024));
            CC_SHA256_Update(&context, bytes.data() + done, count); done += count;
        }
    };
    feed(std::as_bytes(std::span(&p.prefix, 1))); feed(std::as_bytes(std::span(p.history.previous)));
    for (const auto view : views) { const auto bytes = p.g.read(view); feed(bytes); }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{}; CC_SHA256_Final(digest.data(), &context);
    constexpr char hex[] = "0123456789abcdef"; std::string out; out.reserve(64);
    for (const auto value : digest) { out += hex[value >> 4]; out += hex[value & 15]; }
    return out;
}
#endif
} // namespace tokspur
