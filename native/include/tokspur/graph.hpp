#pragma once
#if defined(TOKSPUR_PRIVATE_HTTP_FAST_MATH) && (!defined(TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8) || !defined(TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL) || !defined(TOKSPUR_PRIVATE_PLE_GATE_COMPAT))
#error "private fast HTTP math requires explicit staged-HTTP, channel and PLE builds"
#endif
#if defined(TOKSPUR_PRIVATE_WIDE_Q8_EXACT) && !defined(TOKSPUR_PRIVATE_HC_EXACT)
#error "private wide caller requires its independent HC metadata build"
#endif
#if defined(TOKSPUR_PRIVATE_WIDE_Q8_MANY) && !defined(TOKSPUR_PRIVATE_WIDE_Q8_EXACT)
#error "private many caller requires its independent off-wide/HC metadata build"
#endif
#if defined(TOKSPUR_PRIVATE_VERIFY8_BRANCH) && (!defined(TOKSPUR_PRIVATE_VERIFY_EIGHT) || !defined(TOKSPUR_PRIVATE_HC_EXACT))
#error "private branch verifier requires the independent V8 and exact-HC builds"
#endif
#if defined(TOKSPUR_PRIVATE_HTTP_BRANCH8) && (!defined(TOKSPUR_PRIVATE_VERIFY8_BRANCH) || !defined(TOKSPUR_PRIVATE_HTTP_VERIFY8))
#error "private branch8 HTTP requires both explicit branch and HTTP verifier builds"
#endif
#if defined(TOKSPUR_PRIVATE_QSA_STAGE_V8) && !defined(TOKSPUR_PRIVATE_VERIFY8_BRANCH)
#error "private staged QSA composition requires the explicit branch-verifier build"
#endif
#if defined(TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8) && (!defined(TOKSPUR_PRIVATE_HTTP_BRANCH8) || !defined(TOKSPUR_PRIVATE_QSA_STAGE_V8))
#error "private HTTP staged QSA requires both branch8 and staged-QSA builds"
#endif
#if defined(TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL) && (!defined(TOKSPUR_PRIVATE_QSA_STAGE_V8) || !defined(TOKSPUR_PRIVATE_HC_EXACT) || !defined(TOKSPUR_PRIVATE_WIDE_Q8_EXACT) || !defined(TOKSPUR_PRIVATE_WIDE_Q8_MANY))
#error "private chronological channels require the independent Stage-V8/HC/wide-off metadata builds"
#endif

#if defined(TOKSPUR_PRIVATE_VERIFY_QSA_ROWS) && ((!defined(TOKSPUR_PRIVATE_VERIFICATION_COST) && !(defined(TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL) && defined(TOKSPUR_PRIVATE_KNOWN_QSA_ROWS))) || !defined(TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL) || !defined(TOKSPUR_PRIVATE_VERIFY_EIGHT))
#error "private QSA rows require COST or the explicit sampled known-row caller, plus channel/V8 builds"
#endif
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
#include "tokspur/qsa_verify_rows.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
#include "tokspur/task4_profile.hpp"
#endif
#include "tokspur/routing.hpp"
#include "tokspur/memory.hpp"
#include "tokspur/gdn_math.hpp"
#include "tokspur/gdn_tile.hpp"
#include "tokspur/attention_math.hpp"
#ifdef TOKSPUR_PRIVATE_Q8_MPP
#include "tokspur/q8_mpp.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MPP
#include "tokspur/grouped_mpp.hpp"
#endif

#include <cstddef>
#include <array>
#include <cstdint>
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
#include "tokspur/carrier_diagnostic.hpp"
#endif
#include <cmath>
#include <filesystem>
#include <memory>
#include "tokspur/nax_split_policy.hpp"
#include <optional>
#include <span>
#include <string>
#include <vector>
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
#include "tokspur/batch_phase_clock.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
#if !defined(TOKSPUR_PRIVATE_NATIVE_MTP) || !defined(TOKSPUR_PRIVATE_LANE_LAYOUT) || !defined(TOKSPUR_PRIVATE_VERIFY_QSA_ROWS) || !defined(TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL)
#error "MTP dispatch profiling requires the explicit native MTP, lane-layout and channel/QSA-row build"
#endif
#include "tokspur/mtp_profile.hpp"
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
#include "tokspur/qsa_channel_runtime.hpp"
#endif

namespace tokspur {
#ifdef TOKSPUR_PRIVATE_VERIFY_NAX_BATCH_TOP10_MIXTURE
static_assert(TOKSPUR_PRIVATE_VERIFY_NAX_BATCH_TOP10_MIXTURE == 0 || TOKSPUR_PRIVATE_VERIFY_NAX_BATCH_TOP10_MIXTURE == 1,
    "verify NAX top10/mixture batch policy must be 0 or 1");
inline constexpr bool verify_nax_batch_top10_mixture_enabled = TOKSPUR_PRIVATE_VERIFY_NAX_BATCH_TOP10_MIXTURE == 1;
#else
inline constexpr bool verify_nax_batch_top10_mixture_enabled = false;
#endif
#if defined(TOKSPUR_PRIVATE_VERIFY_NAX_BATCH_TOP10_MIXTURE) && TOKSPUR_PRIVATE_VERIFY_NAX_BATCH_TOP10_MIXTURE == 1
#if !defined(TOKSPUR_PRIVATE_PREFILL_BATCH_TOP10) || TOKSPUR_PRIVATE_PREFILL_BATCH_TOP10 != 1 || !defined(TOKSPUR_PRIVATE_PREFILL_BATCH_MIXTURE) || TOKSPUR_PRIVATE_PREFILL_BATCH_MIXTURE != 1
#error "verify NAX top10/mixture batching requires both existing batch kernels"
#endif
#endif
class ModelFiles;
enum class PrefillWeightLayout : uint32_t;
enum class QsaPrefillOp : uint32_t;
struct GdnChunkDispatch;
struct GpuLogitTopKRow;

// Opaque, checked capabilities. A released or foreign view is never silently
// rebound to another allocation. Offsets/lengths are bytes, not float elements.
struct GpuView {
    uint64_t owner = 0;
    size_t slot = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};
// Borrowed FP32 token rows for prefill operators. Strides are bytes, and the caller owns allocation, submission and state publication.
struct GpuRowsF32 {
    GpuView view;
    uint32_t rows = 0;
    uint32_t columns = 0;
    uint64_t stride_bytes = 0;
};
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
// A per-layer borrowed routing capability. Only its Graph owner can create it, and end/abort/owner submission makes it unusable. Internal operator cuts retain the protected lease within the same owner. No routes or allocation ownership escape that owner.
class GroupedMmRoutes {
    friend class MetalGraph;
    uint64_t owner_ = 0, generation_ = 0;
};
#endif
struct GraphCopy { GpuView source, destination; };
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
struct CarrierStateRead { GpuView storage; uint64_t source_offset, bytes, output_offset; };
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
struct LaneMatrixShape { uint32_t encoding, columns, rows, experts; };
#endif
struct PlanarMatrixShape {
    uint32_t source_type = 0; // 7 (Q5_1), 8 (Q8_0), 12 (Q4_K), 13 (Q5_K)
    uint32_t kind = 0;        // uint32_t(planar_v3::PlanarKind)
    uint32_t columns = 0;
    uint32_t rows = 0;
    uint32_t experts = 0;
};

// Immutable recorded bindings, owned by one graph. Slots are never recycled.
struct GraphCapture { uint64_t owner = 0; size_t slot = 0; };

enum class Activation : uint32_t { identity, silu, sigmoid, softplus };
enum class PleGateTransform : uint32_t { plain, signed_root };
enum class Q8GemvRg4Mode : uint8_t { off = 0, on = 1, wide = 2 };
enum class Q8ThreeMsgMode : uint8_t { off = 0, on = 1, wide = 2 };
enum class PrefillM64Mode : uint8_t { off = 0, on = 1, short_only = 2 };

struct GraphMemory {
    uint64_t budget = 0;
    uint64_t live_bytes = 0;
    uint64_t peak_bytes = 0;
    uint64_t allocations = 0;
    uint64_t max_buffer_bytes = 0;
    uint64_t recommended_working_set_bytes = 0;
    uint64_t imported_backing_bytes = 0;
    uint64_t profile_counter_payload_bytes = 0;
    uint64_t capture_resource_bytes = 0; // Constants + reported ICB allocatedSize.
    uint64_t live_captures = 0;
    // Immutable no-copy capture binding aliases share already charged imported
    // CPU pages. Span is diagnostic; each full reported GPU resource allocation
    // is conservatively added to live_bytes. Cached until source release.
    uint64_t capture_binding_aliases = 0;
    uint64_t capture_binding_alias_span_bytes = 0;
    uint64_t capture_binding_alias_resource_bytes = 0;
};
struct HcQ8Policy {
    uint32_t splits = 0;
    uint64_t scratch_logical_bytes = 0, scratch_charged_bytes = 0;
};
struct RouterMarginTelemetry {
    float min_margin = std::numeric_limits<float>::infinity();
    double sum_margin = 0.0;
    uint64_t total_tokens = 0;
    uint64_t violation_count = 0;
    uint64_t fallback_layers = 0;
    uint64_t total_layers = 0;
    [[nodiscard]] float mean_margin() const {
        return total_tokens > 0 ? float(sum_margin / double(total_tokens)) : 0.0f;
    }
};
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
struct QsaStageV8Policy {
    bool prepared = false, enabled = false;
    uint64_t scratch_logical_bytes = 0, scratch_charged_bytes = 0;
    bool operator==(const QsaStageV8Policy&) const = default;
};
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
struct QsaChannelPolicy {
    bool prepared = false, enabled = false;
    uint64_t factor_logical_bytes = 0, factor_charged_bytes = 0;
    bool operator==(const QsaChannelPolicy&) const = default;
};
#endif
inline bool qsa_stage_v8_startup_allowed(uint32_t execution, bool profiling, uint32_t splits, uint32_t blocks,
    uint32_t old_hc, uint32_t capacity, bool kernels, bool scheduling, bool prefill,
    bool lookup_selectors, bool broker, bool scalar_wide, bool many_wide
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    , bool bounded_branch_broker = false
#endif
    ) {
    const bool transport = !broker
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        || (bounded_branch_broker && capacity == 8)
#endif
        ;
    return !execution && !profiling && splits == 16 && blocks == 3 && !old_hc &&
        (capacity == 0 || capacity == 8
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
         || capacity == 512 || capacity == 1024 || capacity == 2048
#endif
        ) && !kernels && !scheduling && !prefill &&
        !lookup_selectors && transport && !scalar_wide && !many_wide;
}
#endif
#ifdef TOKSPUR_PRIVATE_HC_EXACT
enum class HcExactMode : uint32_t { off, scalar, combined };
// Completed successful GPU dispatches, not logical token projections.
using HcExactRoutes = std::array<uint64_t, 5>; // scalar,many1,many2,many4,many8.
inline bool hc_exact_startup_allowed(HcExactMode mode, uint32_t execution, bool profiling,
    uint32_t splits, uint32_t blocks, uint32_t old_hc, uint32_t capacity,
    bool kernels, bool scheduling, bool prefill, bool broker, uint32_t checkpoints
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
    , bool bounded_branch_verifier = false
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    , bool bounded_branch_broker = false
#endif
    ) {
#ifdef TOKSPUR_PRIVATE_VERIFY8_BRANCH
    // Existing callers remain closed. Explicit consent admits only the combined
    // capacity-eight path and at most the already bounded four ancestor slots.
    const bool history = !checkpoints || (bounded_branch_verifier && checkpoints <= 4 &&
        mode == HcExactMode::combined && capacity == 8);
#else
    const bool history = !checkpoints;
#endif
    const bool transport = !broker
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        || (bounded_branch_broker && bounded_branch_verifier && checkpoints && history)
#endif
        ;
    return uint32_t(mode) <= 2 && (mode == HcExactMode::off ||
        (!execution && !profiling && splits == 16 && blocks == 3 && !old_hc &&
         (capacity == 0 || capacity == 8
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
          || capacity == 512 || capacity == 1024 || capacity == 2048
#endif
         ) && !kernels && !scheduling && !prefill && transport && history));
}
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
// Private caller is built together with the independent HC instrumentation.
// Normal builds have neither this API nor its metadata.
using WideQ8ExactRoutes = std::array<uint64_t, 4>; // qkv scalar/many1, output scalar/many1.
inline bool wide_q8_exact_startup_allowed(bool enabled, uint32_t execution, bool profiling,
    uint32_t splits, uint32_t blocks, uint32_t old_hc, uint32_t capacity,
    bool kernels, bool scheduling, bool prefill, bool broker, uint32_t checkpoints) {
    return !enabled || (!execution && !profiling && splits == 16 && blocks == 3 && !old_hc &&
        (capacity == 0 || capacity == 8) && !kernels && !scheduling && !prefill && !broker && !checkpoints);
}
#endif

#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
// Completed physical dispatches: qkv many2/4/8, output many2/4/8.
using WideQ8ManyRoutes = std::array<uint64_t, 6>;
inline bool wide_q8_many_startup_allowed(bool enabled, uint32_t execution, bool profiling,
    uint32_t splits, uint32_t blocks, uint32_t old_hc, uint32_t capacity,
    bool kernels, bool scheduling, bool prefill, bool broker, uint32_t checkpoints, bool scalar_wide) {
    return !scalar_wide && (!enabled || (!execution && !profiling && splits == 16 && blocks == 3 &&
        !old_hc && capacity == 8 && !kernels && !scheduling && !prefill && !broker && !checkpoints));
}
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
// Completed physical dispatches: prefill, decode (T=1), verify (T>1), replay, norm_gate.
using GdnPortedRoutes = std::array<uint64_t, 5>;
#endif

struct BatchResult {
    uint64_t dispatches = 0;
    double wall_ms = 0;
    double gpu_ms = 0;
    uint64_t capture_executes = 0; // Not physical dispatches.
};
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
// Intrusive direct-batch diagnostics. The sink must consume borrowed strings during the callback.
struct OperatorCheckpointRecord {
    const char* category = "unclassified";
    const char* tag = "owner_tail";
    const char* phase = "unassigned";
    std::array<uint32_t, 4> dimensions{};
    int32_t layer = -1;
    uint32_t prefix = 0, rows = 0;
    uint64_t owner_batch = 0, segment = 0;
    bool owner_final = false;
};
// Physical is null only when an explicit diagnostic owner closes an empty final batch.
using OperatorCheckpointSink = bool (*)(void*, const OperatorCheckpointRecord&, const BatchResult* physical, const BatchResult& rollup) noexcept;
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
class MetalGraph;
struct QsaRouteAttributionRecord {
    uint32_t phase = 0, rows = 0, channels = 0, experts = 0, selected = 0, prefix = 0;
    int32_t layer = -1;
    std::array<GpuView, 6> views{};
};
using QsaRouteAttributionSink = bool (*)(void*, MetalGraph&, const QsaRouteAttributionRecord&) noexcept;
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
class MetalGraph;
struct GdnRouteAttributionRecord {
    uint32_t phase = 0, rows = 0, channels = 0, experts = 0, selected = 0, prefix = 0;
    int32_t layer = -1;
    std::array<GpuView, 6> views{};
};
using GdnRouteAttributionSink = bool (*)(void*, MetalGraph&, const GdnRouteAttributionRecord&) noexcept;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
class MetalGraph;
struct RouteAttributionRecord {
    uint32_t phase = 0, rows = 0, channels = 0, experts = 0, selected = 0;
    int32_t layer = -1;
    std::array<GpuView, 6> views{};
};
using RouteAttributionSink = bool (*)(void*, MetalGraph&, const RouteAttributionRecord&) noexcept;
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
class MetalGraph;
struct RouteLockRecord {
    OperatorCheckpointRecord context;
    uint32_t rows = 0, channels = 0, experts = 0, selected = 0, row_offset = 0;
    GpuView logits{}, ids{}, weights{};
};
using RouteLockSink = bool (*)(void*, MetalGraph&, const RouteLockRecord&) noexcept;
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
struct KernelSearchCaptureRecord {
    OperatorCheckpointRecord context;
    uint32_t kind = 0, encoding = 0, parameter_bytes = 0;
    std::array<uint32_t, 8> dimensions{};
    std::array<uint32_t, 32> parameters{};
    std::array<GpuView, 6> views{};
};
using KernelSearchCaptureSink = bool (*)(void*, MetalGraph&, const KernelSearchCaptureRecord&) noexcept;
#endif
enum class ProfileMode { disabled, dispatch_boundary, encoder_boundary };
// Clock quality is separate from successful GPU work. Keep this pure so the
// inclusive boundary and invalid calibrations are testable without GPU clocks.
inline bool profile_calibration_valid(double tick_nanoseconds, double mismatch_fraction) {
    return std::isfinite(tick_nanoseconds) && tick_nanoseconds > 0 &&
        std::isfinite(mismatch_fraction) && mismatch_fraction >= 0 && mismatch_fraction <= 0.03;
}
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
// Private attribution metadata only. No kernel, buffer or command scheduling change.
enum class Task4ProfilePhase : uint32_t { unassigned, prefill_layer, decode_layer, prefill_mixer, prefill_output, decode_mixer, decode_output, mtp_prime, mtp_advance, mtp_mixer, mtp_output };
struct Task4ProfileTag {
    Task4ProfilePhase phase = Task4ProfilePhase::unassigned;
    int32_t layer = -1;
    uint32_t prefix = 0, rows = 0;
};
#endif
struct KernelTiming {
    std::string kernel;
    uint32_t encoding;
    std::array<uint32_t, 4> shape;
    uint64_t start, end;
    double milliseconds;
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    Task4ProfileTag task4_tag{};
    uint32_t task4_dispatch_index = 0;
    std::array<uint64_t, 3> task4_grid{}, task4_threads_per_threadgroup{};
    bool task4_grid_is_threadgroups = true;
#endif
};
struct PrefillTop10MixtureTelemetry {
    uint64_t moe_batch_calls = 0;
    uint64_t moe_batch_rows = 0;
    uint32_t moe_batch_min_rows = 0;
    uint32_t moe_batch_max_rows = 0;
    uint64_t top10_batched_calls = 0;
    uint64_t top10_batched_rows = 0;
    uint64_t top10_scalar_calls = 0;
    uint64_t top10_fallbacks = 0;
    uint64_t mixture_batched_calls = 0;
    uint64_t mixture_batched_rows = 0;
    uint64_t mixture_scalar_calls = 0;
    uint64_t mixture_fallbacks = 0;
};
// Lifetime completed operator-group encodings (before submission), including later aborts.
// Partial groups that fail during CPU encoding are not counted; prefill/drafter are excluded.
struct VerifyNaxTop10MixtureTelemetry {
    uint64_t top10_batched_calls = 0, top10_batched_rows = 0, top10_scalar_calls = 0;
    uint64_t mixture_batched_calls = 0, mixture_batched_rows = 0, mixture_scalar_calls = 0;
};
struct GraphProfile {
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    uint64_t task4_calibration_gpu_start = 0, task4_calibration_gpu_end = 0;
    uint64_t task4_calibration_cpu_start = 0, task4_calibration_cpu_end = 0;
    double task4_calibration_steady_span_ns = 0;
    bool task4_counter_ticks_valid = true;
#endif
    ProfileMode mode = ProfileMode::disabled;
    bool timings_valid = false; // False retains raw samples, not usable calibrated measurements.
    bool cpu_clock_is_mach = false;
    double gpu_tick_nanoseconds = 0;
    double cpu_clock_mismatch_fraction = 0; // Relative disagreement with steady_clock over the sampled span.
    double sampled_milliseconds = 0;
    std::vector<KernelTiming> kernels;
};
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
struct Task4ProfileBatch {
    uint32_t command_buffer_id = 0;
    double encode_ms = 0, submit_ms = 0, wait_ms = 0;
    uint32_t encoder_count = 0;
};
// Synchronous after-submit consumer. Copy or serialise every batch here before begin clears last_profile.
using Task4ProfileSink = bool (*)(void*, const Task4ProfileBatch&, const GraphProfile&, const BatchResult&) noexcept;
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
struct MtpProfileBatch {
    uint32_t round = 0, base = 0, batch = 0;
    MtpProfilePhase phase = MtpProfilePhase::none;
    double encode_ms = 0, submit_ms = 0, wait_ms = 0;
};
// Called synchronously after collection, before any following submit can replace
// last_profile. A failed sink invalidates the current diagnostic request; no continuation.
using MtpProfileSink = bool (*)(void*, const MtpProfileBatch&, const GraphProfile&, const BatchResult&) noexcept;
#endif
struct SelectionScratch { GpuView scores_a, ids_a, scores_b, ids_b; };
struct TensorSlice { std::string name; uint64_t offset = 0, bytes = 0; };
struct ImportedWeights {
    std::vector<GpuView> tensors; // Same order as requested slices.
    std::vector<GpuView> regions; // Full capabilities, release each at most once.
};

// Persistent, single-host-thread execution context. Upload/allocate/read/release
// are permitted only between batches. One serial encoder and one command buffer
// hold a batch; there is no per-operator commit, wait, readback or weight copy.
// This is execution machinery, not yet complete model state or inference.
class MetalGraph {
public:
    explicit MetalGraph(const std::filesystem::path& owned_library, uint64_t byte_budget);
    ~MetalGraph();
    MetalGraph(const MetalGraph&) = delete;
    MetalGraph& operator=(const MetalGraph&) = delete;

    [[nodiscard]] GpuView zeros(uint64_t bytes, std::string label);
    [[nodiscard]] GpuView upload(std::span<const std::byte> data, std::string label, bool immutable = true);
    // One batched immutable source import at a time. Page-adjacent ranges in a
    // shard merge by default, preventing overlapping backing/ledger entries. The shared
    // source owns read-only mmap/FD lifetimes until every region is released.
    // Optional page-aligned region cap retains every complete tensor slice.
    // A capped boundary may share immutable edge pages with its neighbour;
    // each actual resource span is charged conservatively (no hidden aliases).
    [[nodiscard]] ImportedWeights import_weights(std::shared_ptr<const ModelFiles> source, std::span<const TensorSlice> slices,
                                                 uint64_t maximum_region_bytes = 0
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                                                 // Explicit auxiliary mapping; every new resource is fully charged.
                                                 , bool additional_source = false
#endif
                                                 );
    [[nodiscard]] GpuView slice(GpuView buffer, uint64_t offset, uint64_t bytes) const;
    [[nodiscard]] GpuView suballocate_draft_output_slice(GpuView output_weight_view, uint32_t entries = 32768, uint32_t channels = 2560);
    [[nodiscard]] GpuView allocate_draft_embedding_slice(uint32_t entries = 32768, uint32_t channels = 2560);
    [[nodiscard]] GpuView draft_output_slice() const noexcept;
    [[nodiscard]] GpuView draft_embedding_slice() const noexcept;
    [[nodiscard]] bool has_draft_slices() const noexcept;
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    // Register complete immutable sidecar matrices before the first batch. Every packed binding must use a matching lane kernel; unsupported consumers refuse.
    void register_lane_matrix(GpuView matrix, LaneMatrixShape shape);
    [[nodiscard]] uint64_t lane_matrix_count() const;
#endif
    // Lossless planar v3 (tokspur-pack-v3) immutable registration and MoE execution.
    void register_planar_matrix(GpuView matrix, PlanarMatrixShape shape);
    [[nodiscard]] uint64_t planar_matrix_count() const;
    [[nodiscard]] std::optional<PlanarMatrixShape> planar_matrix_shape(GpuView matrix) const;
    void register_q8_matrix(GpuView matrix, uint32_t columns, uint32_t rows);
    void prepare_planar_moe_v3(uint32_t gate_up_encoding, uint32_t down_encoding, bool prepare_nax = true,
                               uint32_t max_nax_tokens = 2048, uint32_t max_experts = 512,
                               uint32_t max_selected = 10, uint32_t max_channels = 2560, uint32_t max_hidden = 640);
    void planar_moe_gate_up_swiglu_m8(GpuView pair_lo, GpuView pair_hi, GpuView indices,
                                      GpuRowsF32 input, GpuRowsF32 expert_hidden,
                                      uint32_t experts, uint32_t selected);
    void planar_moe_routed_m8(GpuView pair_lo, GpuView pair_hi, GpuView down_weights, GpuView indices,
                              GpuRowsF32 input, GpuRowsF32 expert_hidden, GpuRowsF32 expert_down,
                              uint32_t experts, uint32_t selected);
    void planar_nax_grouped_moe(GpuView pair_lo, GpuView pair_hi, GpuView down_weights, GpuView indices,
                                GpuRowsF32 input, GpuRowsF32 expert_down,
                                GpuView token_map, GpuView counts, GpuView tasks,
                                uint32_t experts, uint32_t selected);
    void write(GpuView destination, std::span<const std::byte> data);
    void write_bytes(GpuView storage, uint64_t offset, std::span<const std::byte> data);
    void read_bytes(GpuView storage, uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::vector<std::byte> read(GpuView source) const;
    // Exact-size idle readback into caller-owned storage; no payload allocation.
    // memmove permits a caller span that aliases the shared source buffer.
    void read(GpuView source, std::span<std::byte> destination) const;
    [[nodiscard]] const float* host_floats(GpuView source) const;
    // Synchronous idle-only copies between existing shared Metal buffers.
    // All 1..1024 pairs are checked before any bytes change: equal lengths,
    // writable destinations and no destination overlap with any source or
    // other destination. No allocation, encoding or intermediate payload.
    void copy(std::span<const GraphCopy> copies);
    void release(GpuView full_buffer);
    [[nodiscard]] GraphMemory memory() const;
    // Allocates a telemetry snapshot only; never called on encoding paths.
    [[nodiscard]] Q8Tile64CacheStats q8_tile64_cache_stats() const;
    [[nodiscard]] std::string device_name() const;
    // Opt-in diagnostic only. Stage-only devices use one encoder per operation;
    // this deliberately perturbs timing and is never an uninstrumented benchmark.
    // Counter result payload is charged, opaque driver overhead is not exposed.
    ProfileMode configure_profiling(uint32_t max_dispatches); // Zero disables.
#ifdef TOKSPUR_PRIVATE_CARRIER_DIAGNOSTIC
    // Idle diagnostic state inspection. Read complete shared views through a
    // bounded GPU staging buffer, then hash caller-owned host bytes. No model
    // write or decoded action is performed; the normal read path is unchanged.
    void begin_carrier_gpu_readback();
    void end_carrier_gpu_readback() noexcept;
    [[nodiscard]] uint32_t carrier_gpu_readback_commands() const;
    [[nodiscard]] bool carrier_gpu_readback_enabled() const;
    // Ordered, complete coverage of one external host span, at most 16 MiB.
    // All source ranges and aliases are checked before GPU submission.
    void read_carrier_state_segments(std::span<const CarrierStateRead> reads, std::span<std::byte> destination) const;
    [[nodiscard]] uint32_t prime_carrier_mapped_weights() const;
    void prepare_carrier_diagnostic(CarrierDiagnosticMode mode, uint32_t actions, bool record_dispatches = true);
    bool begin_carrier_diagnostic(bool generation_v2, bool component = false);
    void end_carrier_diagnostic();
    void finish_carrier_diagnostic();
    [[nodiscard]] uint32_t carrier_diagnostic_action() const;
    [[nodiscard]] bool carrier_diagnostic_active() const;
    [[nodiscard]] bool carrier_capture_enabled() const;
    [[nodiscard]] uint32_t carrier_route_snapshot_dispatches(uint32_t layers, uint32_t rows, uint32_t first_layer = 0) const;
    void set_carrier_diagnostic_layer(uint32_t layer);
    void configure_carrier_capture(const std::string& path);
    void capture_carrier_logits(std::span<const float> logits, uint32_t row, uint32_t rows, const GpuLogitTopKRow* topk = nullptr);
#endif
    [[nodiscard]] const GraphProfile& last_profile() const; // Invalidated at begin.
#ifdef TOKSPUR_PRIVATE_TASK4_PROFILE
    // Diagnostic setup after model/stack/MTP preparation. The explicit boundary must be supported, with no automatic fallback or concurrent dispatch. Encoder-late8 reserves counters while inactive until the caller starts its final-packet window.
    void prepare_task4_profile(uint32_t max_dispatches, uint32_t max_batches, Task4ProfileSink sink, void* context,
                               Task4ProfileBoundary boundary = Task4ProfileBoundary::dispatch);
    void begin_task4_profile_window(uint32_t prefix);
    void set_task4_profile_tag(Task4ProfileTag tag);
    [[nodiscard]] bool task4_profile_reserved() const;
#endif
#ifdef TOKSPUR_PRIVATE_MTP_DISPATCH_PROFILE
    void prepare_mtp_generation_profile(uint32_t rounds, MtpProfileBoundary boundary, MtpProfileSink sink, void* context);
    [[nodiscard]] bool begin_mtp_profile_round(uint32_t base);
    void end_mtp_profile_round();
    void set_mtp_profile_phase(MtpProfilePhase phase);
    [[nodiscard]] bool mtp_profile_reserved() const;
    [[nodiscard]] MtpProfileWindow mtp_profile_window() const;
#endif
    // Experimental FP32 small-output GEMV: deterministic split-column reduction,
    // 64 output rows or fewer, >=1,024 columns. Zero preserves ordinary execution.
    // Allocates bounded scratch only while idle; different FP32 sum association.
    // One selects aligned FP32 reads with ordinary sum association, no scratch.
    void configure_small_gemv(uint32_t column_splits); // 0, 1, 4, 8, 16 or 32.
    // Opt-in block-wise decoding; unchanged encoded weights, lane sequence and
    // FP32 operations. Bit 0 selects Q8_0, bit 1 selects Q4_K; zero is ordinary.
    // No layout copy, scratch allocation or extra dispatch is introduced.
    void configure_block_gemv(uint32_t formats); // 0..3, idle only.
#ifdef TOKSPUR_PRIVATE_NATIVE_PLAIN_LOADS
    // Component-only selector. Same arithmetic with aligned F32/BF16 loads.
    void configure_native_plain_loads(bool enabled);
    [[nodiscard]] bool native_plain_loads() const;
#endif

    // Numeric execution identity for reusable accepted state. Profiling is not
    // included: it changes instrumentation, not the chosen arithmetic path.
    [[nodiscard]] std::array<uint32_t, 2> kernel_policy() const;
    // Fixed opt-in HC-down family only: Q8_0, 10240 columns, 320 rows.
    // Direct/unprofiled; one maximum 320*32 FP32 scratch, prepared while idle.
    // Split 1 matches block-Q8; larger values change FP32 sum association.
    // Default zero prepares/allocates nothing. Reusable stacks pin this policy.
    void configure_hc_q8_splits(uint32_t splits); // 0,1,2,4,8,16,32.
    [[nodiscard]] HcQ8Policy hc_q8_policy() const;
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    void configure_hc_exact(HcExactMode mode); // Startup-only; all PSOs or no publication.
    [[nodiscard]] HcExactMode hc_exact_mode() const;
    [[nodiscard]] HcExactRoutes hc_exact_routes() const;
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_EXACT
    void configure_wide_q8_exact(bool enabled); // Before any batch; both selected PSOs or no enabled publication.
    [[nodiscard]] bool wide_q8_exact_enabled() const;
    [[nodiscard]] WideQ8ExactRoutes wide_q8_exact_routes() const;
#endif
#ifdef TOKSPUR_PRIVATE_WIDE_Q8_MANY
    void configure_wide_q8_many(bool enabled); // Startup-only; all six U4 PSOs or no enabled publication.
    [[nodiscard]] bool wide_q8_many_enabled() const;
    [[nodiscard]] WideQ8ManyRoutes wide_q8_many_routes() const;
#endif
    [[nodiscard]] uint32_t gemv_dispatches(uint32_t encoding, uint32_t columns, uint32_t rows) const;
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    // New prepared-off/on composition: BOTH arms prepare both PSOs and the same
    // one maximum score allocation. Immutable after first begin/capture attempt.
    void configure_qsa_stage_v8(bool enabled);
    [[nodiscard]] QsaStageV8Policy qsa_stage_v8_policy() const;
    // Successful attention calls, each containing two physical passes (+1 vs
    // ordinary). Failed/aborted batches discard their pending count.
    [[nodiscard]] uint64_t qsa_stage_v8_calls() const;
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    // Absent: no preparation, allocation or routing change. Explicit OFF/ON
    // BOTH prepare the same two PSOs and one fixed factor buffer, before begin.
    void configure_qsa_channel_chronological(bool enabled);
    [[nodiscard]] QsaChannelPolicy qsa_channel_policy() const;
    // Successful three-pass calls. Each adds one physical dispatch to the
    // existing two-pass Stage count; aborted/failed batches publish none.
    [[nodiscard]] uint64_t qsa_channel_calls() const;
#ifdef TOKSPUR_PRIVATE_KERNEL_QSA_F32
    void configure_kernel_qsa_f32(bool enabled);
    [[nodiscard]] bool kernel_qsa_f32_enabled() const;
    [[nodiscard]] bool kernel_step2_rollback_enabled() const;
#ifdef TOKSPUR_PRIVATE_KERNEL_GROUPED_FP16
    void configure_kernel_grouped_fp16(bool enabled);
    [[nodiscard]] bool kernel_grouped_fp16_enabled() const;
    [[nodiscard]] uint64_t kernel_grouped_fp16_encoded_calls() const;
#endif
#endif
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    void configure_qsa_sparse(bool enabled);
    [[nodiscard]] bool qsa_sparse_enabled() const;
    [[nodiscard]] bool qsa_sparse_rollback_enabled() const;
    [[nodiscard]] uint64_t qsa_sparse_decode_calls() const;
    [[nodiscard]] uint64_t qsa_sparse_verify_calls() const;
    void prepare_qsa_sparse_variants();
    void sparse_attention_bn(GpuView query, GpuView bf16_keys, GpuView bf16_values,
        GpuView selected_ids, GpuView output, uint32_t prefix, QsaShape shape, uint32_t bn);
    void sparse_attention_rows_bn(GpuView queries, GpuView bf16_keys, GpuView bf16_values,
        GpuView selected_ids, GpuView outputs, uint32_t base_prefix, uint32_t rows, QsaShape shape, uint32_t bn);
#endif
#endif
    // Original fixed-geometry component APIs and geometry guard remain separate.
    void prepare_qsa_score_stage();
    void sparse_attention_score_stage(GpuView query, GpuView bf16_keys, GpuView bf16_values,
        GpuView selected_ids, GpuView output, GpuView scores, uint32_t prefix, QsaShape shape);
    // Explicit new bounded generated/actual geometry; same owned shader math.
    void sparse_attention_stage_v8(GpuView query, GpuView bf16_keys, GpuView bf16_values,
        GpuView selected_ids, GpuView output, GpuView scores, uint32_t prefix, QsaShape shape);
#endif
#ifdef TOKSPUR_PRIVATE_BATCH_PHASE_CLOCK
    // Separately compiled CPU-only telemetry, startup-only. ON requires regular
    // writable stderr before any batch; no profiling or GPU storage is added.
    void configure_batch_phase_clock(bool enabled);
    [[nodiscard]] BatchPhaseClockState batch_phase_clock_state() const;
#endif

#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
    void prepare_qsa_verify_rows();
    [[nodiscard]] QsaVerifyRowsRoutes qsa_verify_rows_routes() const;
    void sparse_attention_rows_chronological(GpuView queries, GpuView bf16_keys, GpuView bf16_values,
        GpuView selected_ids, GpuView outputs, GpuView scores, GpuView factors,
        uint32_t base_prefix, uint32_t rows, QsaShape shape);
#endif

#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    // Arm B of the phase B dispatch-graph measurement, private build only: each direct batch uses one concurrent compute encoder and the same dispatches in the same order, with a buffer barrier only before a dispatch that reads or writes a byte range written since the last barrier or writes one read since it. Off keeps the barrier after every dispatch. Startup-only (before the first batch), direct unprofiled batches.
    void configure_concurrent_dispatch(bool enabled);
    [[nodiscard]] bool concurrent_dispatch() const;
    [[nodiscard]] uint64_t lifetime_dispatches() const;
    [[nodiscard]] uint64_t lifetime_command_buffers() const;
    [[nodiscard]] uint64_t lifetime_barriers() const;
    [[nodiscard]] uint64_t concurrent_barriers() const; // Actual buffer barriers in the last completed batch, in either arm.
#endif
    void dispatch_indirect(GpuView indirect_buffer, uint64_t offset, uint32_t group_x, uint32_t group_y = 1, uint32_t group_z = 1, bool aligned = true);
#ifdef __OBJC__
    void dispatch_indirect(GpuView indirect_buffer, uint64_t offset, MTLSize group, bool aligned = true);
#endif
#ifdef TOKSPUR_PRIVATE_DISPATCH_GRAPH
    // Development tooling only (separate dispatch-graph build): appends one JSON
    // record per line for every allocation, host access, pipeline, dispatch,
    // barrier and batch, plus digests of every mutable allocation after each
    // completed batch. Direct unprofiled batches only; configure before the
    // first allocation. Bytes are never written, only SHA-256 digests.
    void configure_dispatch_trace(const std::filesystem::path& path);
#endif

    // Host-only capability/alias preflight; no synchronization or GPU work.
    // Every destination must be writable and disjoint from every other view.
    void validate_disjoint(std::span<const GpuView> inputs, std::span<const GpuView> destinations) const;

#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
    void configure_operator_checkpoints(OperatorCheckpointSink sink, void* context, uint32_t maximum_records = 8192);
    void operator_checkpoint(const char* category, const char* tag, std::array<uint32_t, 4> dimensions = {});
    void set_operator_checkpoint_context(const char* phase, int32_t layer, uint32_t prefix, uint32_t rows);
#ifdef TOKSPUR_PRIVATE_QSA_SPARSE
    void configure_qsa_route_attribution(QsaRouteAttributionSink, void*);
    [[nodiscard]] bool qsa_route_attribution_enabled() const;
    [[nodiscard]] std::array<GpuView, 3> qsa_route_attribution_backup() const;
    void qsa_route_attribution_phase(uint32_t);
    void qsa_route_attribution_emit(QsaRouteAttributionRecord, std::span<const GraphCopy> copies = {});
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    void configure_gdn_route_attribution(GdnRouteAttributionSink, void*, std::optional<uint32_t> prefill_prefix = std::nullopt);
    [[nodiscard]] bool gdn_route_attribution_enabled() const;
    [[nodiscard]] bool gdn_route_attribution_prefill(uint32_t prefix, uint32_t steps) const;
    void gdn_route_attribution_phase(uint32_t);
    void gdn_route_attribution_emit(GdnRouteAttributionRecord, std::span<const GraphCopy> copies = {});
#endif
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
    // Private common-input replay only. It never changes the serving kernel policy or publishes the shadow layer.
    void configure_route_attribution(RouteAttributionSink, void*);
    [[nodiscard]] bool route_attribution_enabled() const;
    void route_attribution_phase(uint32_t);
    void route_attribution_common(uint32_t phase, std::array<GpuView, 6>, std::span<const GraphCopy> restore = {});
    void route_attribution_rows(GpuView matrix, GpuRowsF32 input, GpuView logits, GpuView ids, GpuView weights, uint32_t selected, RouterMode, bool normalise);
    void route_attribution_emit(RouteAttributionRecord, std::span<const GraphCopy> restore = {});
#endif
#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
    void configure_route_lock(RouteLockSink, void*);
    void route_lock_rows(GpuView logits, GpuView ids, GpuView weights, uint32_t rows, uint32_t channels, uint32_t experts, uint32_t selected, uint32_t row_offset = 0);
#endif
#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
    void configure_kernel_search_capture(KernelSearchCaptureSink, void*);
    void kernel_search_capture(KernelSearchCaptureRecord, uint32_t bit);
#endif
    void begin();
    // Diagnostic comparator only: ordinary host encoding, but descriptor-built
    // ICB-capable pipelines. Keeps compilation effects separate from replay.
    void begin_indirect_compatible();
    void abort(); // Discards an unsubmitted batch; not rollback after execution.
    [[nodiscard]] BatchResult submit(); // Waits once, checks completion/error flag.
    // Commits the dispatches encoded so far without waiting and continues the same logical batch in a new physical command buffer, so the GPU starts while the host encodes the rest. Returns false and changes nothing for an empty, recorded, profiled or diagnostic batch. submit() and abort() wait for every part.
    bool commit_partial();

    // Explicit static-command experiment. Recording performs no GPU execution;
    // normal owned composite encoders supply all bindings and copied constants.
    // No rebinding, policy adaptation, profiling, or automatic stack routing.
    // 1..1024 physical commands; persistent resources charged to this budget.
    // Release of any captured allocation is refused until release_capture.
    // As for direct batches, a caught primitive encode error requires abort;
    // owned composite encoders abort the complete recording on exceptions.
    void begin_capture(uint32_t max_dispatches);
    [[nodiscard]] GraphCapture finish_capture();
    void execute_capture(GraphCapture capture); // Inside a serial ordinary batch.
    void release_capture(GraphCapture capture); // Idle only; stale handles refuse.

    void gemv(uint32_t encoding, GpuView matrix, GpuView input, GpuView output, uint32_t columns, uint32_t rows);
    // gemv() once per row: input.columns matrix columns in, output.columns matrix rows out. Each row keeps gemv()'s kernels, arguments and arithmetic.
    // In a concurrent batch a two-stage small GEMV runs stage by stage over separate partial-sum regions, so its rows need no barrier between them.
    void gemv_rows(uint32_t encoding, GpuView matrix, GpuRowsF32 input, GpuRowsF32 output);
    // Token-major Q8 projections reuse decoded weights without scratch or conversion. Widths 1/2/4/8 retain their arithmetic. Private width three requires explicit startup selection. Prepare every selected width while idle.
    // Fixed before any batch or many-width preparation. Off in every ordinary build.
    void configure_q8_three_rows(bool enabled);
    [[nodiscard]] bool q8_three_rows_enabled() const;
    // Successful physical three-row dispatches: raw, packed, raw HC-combined, raw wide-QKV, raw wide-output. Legacy route arrays are unchanged.
    using Q8ThreeRoutes = std::array<uint64_t, 5>;
    [[nodiscard]] Q8ThreeRoutes q8_three_rows_routes() const;
    [[nodiscard]] uint32_t q8_many_width(uint32_t remaining) const;
    [[nodiscard]] uint32_t q8_many_chunks(uint32_t rows) const;
    void prepare_q8_matvec_many_all();
    void prepare_q8_matvec_many(uint32_t width);
    void q8_matvec_many(GpuView matrix, GpuView input, GpuView output,
                        uint32_t columns, uint32_t rows, uint32_t width);
#ifdef TOKSPUR_PRIVATE_GROUPED_MM
    // Return only the layout registered for this complete owned weight view.
    [[nodiscard]] PrefillWeightLayout prefill_weight_layout(GpuView matrix) const;
    void prepare_grouped_mm(uint32_t encoding, PrefillWeightLayout layout);
    void grouped_mm(uint32_t encoding, PrefillWeightLayout layout, GpuView matrices, GpuView indices,
        GpuRowsF32 input, GpuRowsF32 output, GpuView token_map, GpuView counts, GpuView tasks,
        uint32_t experts, uint32_t selected, bool broadcast);
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    // Explicit route preparation for an FP32 MoE layer. IDs and metadata stay protected from writes until end; a later layer must create its own scope.
    [[nodiscard]] GroupedMmRoutes grouped_mm_routes(GpuView indices, GpuView token_map, GpuView counts, GpuView tasks, uint32_t experts, uint32_t selected, uint32_t tokens);
    void grouped_mm_project(uint32_t encoding, PrefillWeightLayout layout, GpuView matrices, const GroupedMmRoutes& routes, GpuRowsF32 input, GpuRowsF32 output, bool broadcast);
    void grouped_mm_gate_up_swiglu(uint32_t encoding, PrefillWeightLayout layout, GpuView gate_matrices, GpuView up_matrices, const GroupedMmRoutes& routes, GpuRowsF32 input, GpuRowsF32 output, bool broadcast, bool handoff_bf16 = false);
    [[nodiscard]] bool can_grouped_mm_handoff_bf16(uint32_t experts, uint32_t selected, uint32_t tokens, uint32_t in_columns, uint32_t out_columns) const;
    void end_grouped_mm_routes(const GroupedMmRoutes& routes);
#endif
    // Fixed before preparation. The FP32 diagnostic accepts raw, tight rows only.
    void configure_prefill_dense_f32(bool enabled);
    [[nodiscard]] bool prefill_dense_f32() const;
    void configure_prefill_dense_nax(bool enabled);
    [[nodiscard]] bool prefill_dense_nax() const;
    [[nodiscard]] uint32_t dense_mm_q8_dispatches(uint32_t columns) const;
    [[nodiscard]] bool can_reuse_nax_dense_bf16(GpuRowsF32 input) const;
    void set_mtp_priming_scope(bool active);
    void configure_verify_nax(bool enabled);
    [[nodiscard]] bool verify_nax() const;
    void configure_verify_nax_m8(bool enabled);
    void configure_target_m8_splits(NaxSplitOverrides overrides);
    [[nodiscard]] const std::string& target_m8_split_digest() const;
    void set_target_verification_role(bool active);
    [[nodiscard]] bool verify_nax_m8() const;
    [[nodiscard]] bool verify_nax_active() const;
    void set_verify_nax_scope(bool active);
    void prepare_verify_nax(std::span<const std::pair<uint32_t, PrefillWeightLayout>> moe_layouts);
    void verify_nax_dense_q8(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool swiglu = false, float silu_scale = 0.0f);
    void verify_nax_dense_q8_swiglu(GpuView gate_matrix, GpuView up_matrix, GpuRowsF32 input, GpuRowsF32 output);
    void verify_nax_gate_up_swiglu_q8(GpuView gate_matrix, GpuView up_matrix, GpuRowsF32 input, GpuRowsF32 output);
    void nax_m8_dense_q8(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool swiglu = false, float silu_scale = 0.0f);
    void verify_nax_bf16(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output);
    void verify_nax_dense_bf16(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output);
    void verify_nax_gdn_gates_f32(GpuView alpha_w, GpuView beta_w, GpuView input, GpuView alpha_out, GpuView beta_out,
        uint32_t columns, uint32_t heads, uint32_t steps);
    void verify_nax_gated_mix_rows(GpuView normalized, GpuView scores, GpuView output,
        uint32_t branches, uint32_t channels, uint32_t steps);
    void verify_nax_moe_routed(uint32_t encoding, PrefillWeightLayout layout, GpuView matrices, GpuView indices,
        GpuRowsF32 input, GpuRowsF32 output, uint32_t experts, uint32_t selected, bool broadcast, bool swiglu = false);
    void verify_nax_moe_union(uint32_t up_encoding, PrefillWeightLayout up_layout, GpuView gate_matrices, GpuView up_matrices,
        uint32_t down_encoding, PrefillWeightLayout down_layout, GpuView down_matrices, GpuView indices,
        GpuRowsF32 input, GpuRowsF32 gate_scratch, GpuRowsF32 down_output, uint32_t experts, uint32_t selected);
#ifdef TOKSPUR_PRIVATE_PREFILL_HC_DOWN_F32
    // Default off, fixed before dense preparation. Requires the FP32 dense policy first.
    void configure_prefill_hc_down_f32(bool enabled);
    [[nodiscard]] bool prefill_hc_down_f32() const;
    [[nodiscard]] bool prefill_hc_down_f32_rows(uint32_t rows) const;
    void dense_hc_down_f32(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output);
    [[nodiscard]] uint64_t prefill_hc_down_f32_encoded_calls() const; // Encoded calls, including later aborted submissions.
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MM_F32
    void configure_prefill_grouped_f32(bool enabled);
    [[nodiscard]] bool prefill_grouped_f32() const;
    void configure_prefill_grouped_auto(bool enabled);
    [[nodiscard]] bool prefill_grouped_auto() const;
    void configure_prefill_grouped_nax(bool enabled);
    [[nodiscard]] bool prefill_grouped_nax() const;
#endif
    void configure_router_margin_epsilon(float epsilon);
    [[nodiscard]] float router_margin_epsilon() const;
    void configure_router_margin_audit(bool enabled);
    [[nodiscard]] bool router_margin_audit() const;
    [[nodiscard]] RouterMarginTelemetry router_margin_telemetry() const;
    void reset_router_margin_telemetry();
    bool evaluate_router_margin(uint32_t tokens);
    [[nodiscard]] GpuView router_margins_scratch() const;
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_F32_BATCH
    // Private projection only. Fixed before preparation, default off, no serving selector.
    void configure_moe_router_f32_batch(bool enabled);
    [[nodiscard]] bool moe_router_f32_batch_enabled() const;
    [[nodiscard]] const char* moe_router_f32_projection_kernel() const;
    void prepare_moe_router_f32_batch();
    void moe_router_f32_batch(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output);
#endif
#ifdef TOKSPUR_PRIVATE_MOE_ROUTER_WINDOW
    // Exact windowed FP32 router and row-batched top-k, shared router and mixture. Fixed before preparation, default off.
    void configure_prefill_router_window(bool enabled);
    [[nodiscard]] bool prefill_router_window() const;
    void configure_prefill_router_fast(bool enabled);
    [[nodiscard]] bool prefill_router_fast() const;
    [[nodiscard]] const char* moe_router_window_projection_kernel() const;
    [[nodiscard]] std::array<uint64_t, 4> router_window_encoded() const;
    void prepare_moe_router_window();
    void moe_router_f32_window(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output);
    void moe_route_rows(GpuView logits, GpuView indices, GpuView weights, uint32_t experts, uint32_t selected, uint32_t tokens, RouterMode mode, bool normalize_selected);
    void moe_shared_router_rows(GpuView weights, GpuRowsF32 input, GpuView output);
    void moe_mixture_rows(GpuView experts, GpuView weights, GpuView shared, GpuView shared_logit, GpuRowsF32 output, uint32_t selected);
#endif
#ifdef TOKSPUR_PRIVATE_HC_DOWN_WINDOW
    // Exact windowed Q8_0 HC down projection (10240 to 320). Fixed before preparation, default off.
    void configure_prefill_hc_down_window(bool enabled);
    [[nodiscard]] bool prefill_hc_down_window() const;
    [[nodiscard]] uint64_t hc_down_window_encoded() const;
    void prepare_hc_down_window();
    void hc_down_q8_window(GpuView weights, GpuRowsF32 input, GpuRowsF32 output);
#endif
    [[nodiscard]] bool prefill_batch_top10() const;
    [[nodiscard]] bool prefill_batch_mixture() const;
    void prepare_prefill_top10_mixture();
    void prefill_route_batch(GpuView logits, GpuView indices, GpuView weights, uint32_t experts, uint32_t selected, uint32_t tokens, RouterMode mode, bool normalize_selected);
    void prefill_mixture_batch(GpuView experts, GpuView weights, GpuView shared, GpuView shared_logit, GpuRowsF32 output, uint32_t selected);
    void record_prefill_moe_chunk(uint32_t rows, bool used_batch_top10, bool top10_fallback, bool used_batch_mixture, bool mixture_fallback);
    [[nodiscard]] PrefillTop10MixtureTelemetry prefill_top10_mixture_telemetry() const;
    void reset_prefill_top10_mixture_telemetry();
    void record_verify_nax_moe_ops(uint32_t rows, bool top10, bool batched);
    [[nodiscard]] VerifyNaxTop10MixtureTelemetry verify_nax_top10_mixture_telemetry() const;
    void prepare_dense_mm_q8(PrefillWeightLayout layout);
    void dense_mm_q8(PrefillWeightLayout layout, GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool reuse_bf16 = false);
    void nax_dense_bf16(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool reuse_bf16 = false);
    [[nodiscard]] bool can_nax_dense_q8_swiglu(PrefillWeightLayout l0, PrefillWeightLayout l1, uint32_t columns, uint32_t hidden, uint32_t batches) const;
    void nax_dense_q8_swiglu(PrefillWeightLayout l0, GpuView w0, PrefillWeightLayout l1, GpuView w1,
        GpuRowsF32 input, GpuRowsF32 output, bool reuse_bf16 = false);
    [[nodiscard]] bool can_dense_mm_q8_multiseg(std::initializer_list<PrefillWeightLayout> layouts, uint32_t batches) const;
    void dense_mm_q8_multiseg2(PrefillWeightLayout l0, GpuView w0, PrefillWeightLayout l1, GpuView w1,
        GpuRowsF32 input, GpuRowsF32 out0, GpuRowsF32 out1);
    void dense_mm_q8_multiseg3(PrefillWeightLayout l0, GpuView w0, PrefillWeightLayout l1, GpuView w1,
        PrefillWeightLayout l2, GpuView w2, GpuRowsF32 input, GpuRowsF32 out0, GpuRowsF32 out1, GpuRowsF32 out2);
#endif
#ifdef TOKSPUR_PRIVATE_GDN_CHUNK
    void prepare_gdn_chunk();
    void gdn_chunk(const GdnChunkDispatch& dispatch);
#ifdef TOKSPUR_PRIVATE_GDN_LEGACY_BATCH
    void prepare_gdn_legacy_batch();
    void gdn_legacy_batch(const GdnChunkDispatch& dispatch);
    [[nodiscard]] uint64_t gdn_legacy_batch_encoded_calls() const;
#endif
    void rms_rows(GpuRowsF32 input, GpuView gains, GpuRowsF32 output,
        uint32_t group_width, float epsilon, bool shared_gains);
#endif
#ifdef TOKSPUR_PRIVATE_GDN_PORTED
    void configure_gdn_ported(bool enabled);
    [[nodiscard]] bool gdn_ported_enabled() const;
    void configure_gdn_prefill_ported(bool enabled);
    [[nodiscard]] bool gdn_prefill_ported_enabled() const;
    [[nodiscard]] bool gdn_ported_rollback_enabled() const;
    [[nodiscard]] bool gdn_use_ported() const;
    [[nodiscard]] GdnPortedRoutes gdn_ported_routes() const;
    void prepare_gdn_ported_variants();
    void gdn_prepare_rows(GpuView activated_qkv, GpuView query, GpuView key, GpuView value,
        GdnShape shape, GdnPolicy policy, StateTile tile);
    void gdn_verify_main(GpuView accepted_state, GpuView query, GpuView key, GpuView value,
        GpuView alpha_logits, GpuView beta_logits, GpuView negative_rates, GpuView dt_bias,
        GpuView candidate_state, GpuView output, GpuView tape_decay, GpuView tape_beta,
        uint32_t heads, uint32_t value_width, StateTile tile, bool write_tape);
    void gdn_verify_main_db(GpuView accepted_state, GpuView query, GpuView key, GpuView value,
        GpuView alpha_logits, GpuView beta_logits, GpuView negative_rates, GpuView dt_bias,
        GpuView candidate_state, GpuView output, GpuView tape_decay, GpuView tape_beta,
        uint32_t heads, uint32_t value_width, StateTile tile, bool write_tape, uint32_t db);
    void gdn_verify_replay(GpuView accepted_state, GpuView key, GpuView value,
        GpuView decay, GpuView beta, GpuView replayed_state,
        uint32_t heads, uint32_t value_width, StateTile tile);
    void gdn_verify_replay_db(GpuView accepted_state, GpuView key, GpuView value,
        GpuView decay, GpuView beta, GpuView replayed_state,
        uint32_t heads, uint32_t value_width, StateTile tile, uint32_t db);
    void gdn_verify_main_replay(GpuView base_state, GpuView prefix_key, GpuView prefix_value,
        GpuView prefix_decay, GpuView prefix_beta, GpuView query, GpuView key, GpuView value,
        GpuView alpha_logits, GpuView beta_logits, GpuView negative_rates, GpuView dt_bias,
        GpuView next_state, GpuView output, uint32_t heads, uint32_t value_width,
        StateTile prefix_tile, StateTile main_tile);
    void gdn_norm_gate(GpuView raw, GpuView effective_gains, GpuView gate, GpuView output,
        uint32_t heads, uint32_t value_width, float epsilon, StateTile tile);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_PREFILL
    void prepare_qsa_prefill();
    void qsa_prefill_dispatch(QsaPrefillOp op, std::span<const GpuView> buffers,
        const std::array<uint32_t, 16>& parameters);
    [[nodiscard]] bool nax_qsa_attention_supported(QsaShape shape) const;
    [[nodiscard]] bool verify_nax_qsa_attention_supported(QsaShape shape) const;
    void nax_qsa_select_blocks(GpuView rotary_index_query, GpuView bf16_index_keys, GpuView selected_ids,
        uint32_t base, uint32_t rows, QsaShape shape, uint32_t block_budget);
    void nax_qsa_attention(GpuView query, GpuView bf16_keys, GpuView bf16_values, GpuView selected_ids,
        GpuView output, uint32_t base, uint32_t rows, QsaShape shape, uint32_t block_budget);
#endif
#ifdef TOKSPUR_PRIVATE_Q8_MPP
    // Experimental O-tier TensorOps, no target routing or capture support.
    // Q8_0 weights and FP32 rows are retained; sums may differ in bits.
    void prepare_q8_mpp(Q8MppLayout layout = Q8MppLayout::single);
    void q8_mpp(GpuView matrix, GpuView input, GpuView output,
                uint32_t columns, uint32_t rows, uint32_t batches, Q8MppLayout layout = Q8MppLayout::single);
#endif
#ifdef TOKSPUR_PRIVATE_DENSE_Q8_STAGED
    // Private raw FP32 backend, default false and fixed before dense preparation or the first batch. Target/worker startup consent is checked separately from operator diagnostics.
    void configure_prefill_dense_staged_f32(bool enabled);
    [[nodiscard]] bool prefill_dense_staged_f32() const;
    [[nodiscard]] const char* prefill_dense_kernel() const;
    // Private direct component only. True explicitly selects half operands with FP32 accumulation; false stages full FP32.
    void prepare_dense_q8_staged(bool half_operands = false);
    void dense_q8_staged(GpuView matrix, GpuView input, GpuView output, uint32_t columns, uint32_t rows, uint32_t batches, bool half_operands = false);
    void dense_q8_staged(GpuView matrix, GpuView input, GpuView output, uint32_t columns, uint32_t rows, uint32_t batches, bool half_operands, PrefillWeightLayout layout);
#endif
#ifdef TOKSPUR_PRIVATE_GROUPED_MPP
    void prepare_routed_matmul_many(uint32_t encoding, uint32_t tile, uint32_t staging_k = 0);
    void routed_matmul_many(uint32_t encoding, GpuView matrices, GpuView indices, GpuView input,
        GpuView output, GpuView token_map, GpuView counts, GpuView tasks,
        GroupedMppShape shape, bool broadcast);
#endif
    // Explicit direct-only Q8_0 column-split experiment. Prepare both pipelines
    // while idle; allocate caller-owned exact-size partials via q8_split_bytes.
    // Bounds: columns multiple of 32 <=65536, rows <=1024, splits 1/2/4/8/16/32.
    // Always two dispatches, no allocation/policy mutation/capture support.
    // Split 1 retains fast block-Q8 arithmetic; larger splits reassociate FP32
    // sums (no wider accumulation), then reduce partials in fixed split order.
    // All outputs/scratch must be mutually disjoint and disjoint from inputs.
    // As for other primitives, abort the batch after a caught encode refusal.
    void prepare_q8_matvec_split();
    void q8_matvec_split(GpuView matrix, GpuView input, GpuView output, GpuView partials,
                         uint32_t columns, uint32_t rows, uint32_t splits);
    // Explicit lossless-layout experiments, not automatic model conversion.
    // Both retain ordinary column/lane arithmetic and use one physical pass.
    void q8_soa_gemv(GpuView scales, GpuView values, GpuView input, GpuView output, uint32_t columns, uint32_t rows);
    void typed_f32_gemv(GpuView matrix, GpuView input, GpuView output, uint32_t columns, uint32_t rows);
    // Explicit ordered-state tiles, not a target or stack execution policy.
    // Prepare while idle; delta groups are 32/64/128. No encode-time GPU-buffer allocation.
    // Capacity-sized buffers, steps 1..capacity; unused output rows stay untouched.
    // Candidates must be disjoint. Publish only after the whole batch succeeds.
    void prepare_gdn_tiles(uint32_t delta_group_threads);
    // Same per-row arithmetic and head order as gdn_prepare/gdn_gates, one
    // dispatch per operation. Capacity-sized tapes, active steps 2..8 only.
    void gdn_prepare_verify_rows(GpuView input, GpuView query, GpuView key,
        GpuView value, GdnShape shape, GdnPolicy policy, StateTile tile);
    void gdn_silu_prepare_verify_rows(GpuView input, GpuView query, GpuView key,
        GpuView value, GdnShape shape, GdnPolicy policy, StateTile tile);
    void gdn_gates_verify_rows(GpuView alpha, GpuView beta_logits, GpuView rates,
        GpuView bias, GpuView decay, GpuView beta, uint32_t heads, StateTile tile);
    void gdn_convolution_tile(GpuView accepted_history, GpuView projected, GpuView weights,
                              GpuView candidate_history, GpuView output,
                              uint32_t channels, StateTile tile);
    void gdn_delta_tile(GpuView accepted_state, GpuView query, GpuView key, GpuView value,
                        GpuView decay, GpuView beta, GpuView candidate_state,
                        GpuView output, uint32_t heads, uint32_t value_width, StateTile tile);
    void rms(GpuView input, GpuView effective_gains, GpuView output, uint32_t group_width, float epsilon);
    void rms_sigmoid_gate(GpuView input, GpuView effective_gains, GpuView gate, GpuView output, uint32_t group_width, float epsilon);
    void activate(GpuView input, GpuView output, Activation kind, float scale = 1);
    void multiply(GpuView input, GpuView gate, GpuView output, bool sigmoid_gate = false);
    void configure_swiglu_fused(bool enabled);
    [[nodiscard]] bool swiglu_fused_enabled() const;
    [[nodiscard]] uint64_t swiglu_fused_calls() const;
    void swiglu(GpuView gate, GpuView up, GpuView output, float scale = 1);
    void swiglu_direct(GpuView gate, GpuView up, GpuView output, float scale = 1);
    void configure_q8_gemv_rg4(Q8GemvRg4Mode mode);
    [[nodiscard]] Q8GemvRg4Mode q8_gemv_rg4_mode() const;
    using Q8GemvRg4Routes = std::array<uint64_t, 2>;
    [[nodiscard]] Q8GemvRg4Routes q8_gemv_rg4_routes() const;
    void configure_q8_three_msg(Q8ThreeMsgMode mode);
    [[nodiscard]] Q8ThreeMsgMode q8_three_msg_mode() const;
    using Q8ThreeMsgRoutes = std::array<uint64_t, 2>;
    [[nodiscard]] Q8ThreeMsgRoutes q8_three_msg_routes() const;
    void configure_moe_three_union(bool enabled);
    [[nodiscard]] bool moe_three_union_enabled() const;
    [[nodiscard]] uint64_t moe_three_union_calls() const;
    [[nodiscard]] bool moe_three_union_matrix_supported(uint32_t encoding, GpuView matrix,
        uint32_t columns, uint32_t rows, uint32_t experts, bool allow_packed_q51) const;
    void moe_three_router(GpuView weights, GpuView input, GpuView logits, uint32_t columns, uint32_t experts);
    void moe_three_route(GpuView logits, GpuView indices, GpuView weights,
        uint32_t experts, uint32_t selected, RouterMode mode, bool normalize_selected);
    void moe_three_build_union(GpuView indices, uint32_t experts, uint32_t selected, uint32_t hidden, uint32_t channels);
    void moe_three_gate_up_swiglu(uint32_t encoding, GpuView gate_weights, GpuView up_weights,
        GpuView input, GpuView expert_hidden, uint32_t columns, uint32_t rows, uint32_t experts, uint32_t selected);
    void moe_three_down(uint32_t encoding, GpuView down_weights, GpuView expert_hidden,
        GpuView expert_down, uint32_t columns, uint32_t rows, uint32_t experts, uint32_t selected);
    void moe_three_shared_router(GpuView weights, GpuView input, GpuView output, uint32_t columns);
    void moe_three_mixture(GpuView experts, GpuView weights, GpuView shared, GpuView shared_logit,
        GpuView output, uint32_t channels, uint32_t selected);
    void configure_prefill_m64(PrefillM64Mode mode);
    [[nodiscard]] PrefillM64Mode prefill_m64_mode() const;
    [[nodiscard]] bool prefill_m64_active(uint32_t batches) const;
    [[nodiscard]] uint64_t prefill_m64_calls() const;
    void configure_prefill_multiseg(bool enabled);
    [[nodiscard]] bool prefill_multiseg_enabled() const;
    [[nodiscard]] uint64_t prefill_multiseg_calls() const;
    void configure_prefill_swizzle(bool enabled);
    [[nodiscard]] bool prefill_swizzle_enabled() const;
    void configure_qsa_prefill_fused(bool enabled);
    [[nodiscard]] bool qsa_prefill_fused_enabled() const;
    void configure_hc_consumer_fused(bool enabled);
    [[nodiscard]] bool hc_consumer_fused_enabled() const;
    [[nodiscard]] uint64_t hc_consumer_fused_calls() const;
    static uint64_t logit_topk_scratch_bytes(uint32_t max_rows, uint32_t max_vocabulary = 1u << 20);
    static uint64_t logit_topk_output_bytes(uint32_t max_rows);
    void prepare_logit_topk();
    void logit_topk(GpuRowsF32 logits, GpuView partial_scratch, GpuView out_rows);
    void configure_verify_hc_gdn_fused(bool enabled);
    [[nodiscard]] bool verify_hc_gdn_fused_enabled() const;
    void prepare_verify_hc_gdn_fused();
    void verify_fused_hc_q8(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool silu = false, float scale = 1.0f);
    void verify_fused_hc_inject_scales_rows(GpuView weights, GpuView normalized,
                                            GpuView inject_logits, GpuView scales,
                                            uint32_t branches, uint32_t channels, uint32_t tokens);
    void verify_fused_hc_inject_read_rows(GpuView weights, GpuView normalized, GpuView scores,
                                          GpuView inject_logits, GpuView output, GpuView scales,
                                          uint32_t branches, uint32_t channels, uint32_t tokens);
    void verify_fused_hc_read_rows(GpuView normalized, GpuView scores, GpuView output,
                                   uint32_t branches, uint32_t channels, uint32_t tokens);
    void verify_fused_hc_write_rows(GpuView streams, GpuView input, GpuView scales, GpuView destination,
                                    uint32_t branches, uint32_t channels, uint32_t tokens);
    void verify_fused_hc_write_rms_rows(GpuView streams, GpuView input, GpuView scales, GpuView gains,
                                        GpuView destination, GpuView normalized,
                                        uint32_t branches, uint32_t channels, uint32_t tokens,
                                        float epsilon, bool shared_gains);
    void verify_fused_gdn_alpha_beta_f32(GpuView w_alpha, GpuView w_beta, GpuRowsF32 input,
                                         GpuRowsF32 out_alpha, GpuRowsF32 out_beta);
    void verify_fused_expert_gate_f32(GpuView weights, GpuRowsF32 input, GpuRowsF32 output);
    void verify_fused_shared_logit_f32(GpuView weights, GpuRowsF32 input, GpuRowsF32 output);
    void residual_read(GpuView normalized, GpuView scores, GpuView inject_logits, GpuView output,
                       GpuView scales, uint32_t branches, uint32_t channels);
    void residual_write(GpuView streams, GpuView output, GpuView scales, GpuView destination);
#ifdef TOKSPUR_PRIVATE_HC_ROWS
    // Component-only batching at four branches, 2560 channels and split16. No serving selector.
    void prepare_hc_rows();
    void configure_prefill_hc_rows(bool enabled);
    [[nodiscard]] bool prefill_hc_rows() const;
    [[nodiscard]] std::array<uint64_t, 3> hc_rows_encoded() const;
    void prefill_hc_inject_rows(GpuView weights, GpuView normalized, GpuView inject, uint32_t tokens);
    void hc_inject_rows(GpuView weights, GpuView normalized, GpuView partials, GpuView inject, uint32_t tokens);
    void hc_read_rows(GpuView normalized, GpuView scores, GpuView inject, GpuView output, GpuView scales, uint32_t tokens);
    void hc_write_rows(GpuView streams, GpuView input, GpuView scales, GpuView destination, uint32_t tokens);
    void hc_write_rms_rows(GpuView streams, GpuView input, GpuView scales, GpuView gains,
        GpuView destination, GpuView normalized, uint32_t tokens, float epsilon);
#endif
    void convolution(GpuView history, GpuView input, GpuView weights, GpuView next_history,
                     GpuView output, uint32_t kernel, uint32_t dilation = 1);
    void ple_gate(GpuView normalized_query, GpuView normalized_key, GpuView shared_value, GpuView gated, GpuView gates,
                  uint32_t branches, uint32_t channels,
#ifdef TOKSPUR_PRIVATE_PLE_GATE_COMPAT
                  PleGateTransform transform = PleGateTransform::signed_root);
#else
                  PleGateTransform transform = PleGateTransform::plain);
#endif
    void ple_write(GpuView streams, GpuView gated, GpuView convolution, GpuView output);
    void ple_gate_rows(GpuView normalized_query, GpuView normalized_key, GpuView shared_value, GpuView gated,
                       uint32_t branches, uint32_t channels, uint32_t rows, PleGateTransform transform);
    void ple_conv_write_rows(GpuView history, GpuView norm_gated, GpuView weights, GpuView streams, GpuView gated,
                             GpuView next_history, GpuView output, uint32_t width, uint32_t kernel, uint32_t dilation,
                             uint32_t rows, uint64_t stream_stride_bytes, uint64_t output_stride_bytes);
    void gated_mix(GpuView normalized, GpuView scores, GpuView output, uint32_t branches, uint32_t channels);
    void delta(GpuView state, GpuView query, GpuView key, GpuView value, GpuView decay,
               GpuView beta, GpuView next_state, GpuView output,
               uint32_t heads, uint32_t key_width, uint32_t value_width);
    void route(GpuView logits, GpuView indices, GpuView weights, uint32_t selected,
               RouterMode mode, bool normalize_selected);
    void routed_gemv(uint32_t encoding, GpuView matrices, GpuView indices, GpuView input,
                     GpuView output, uint32_t columns, uint32_t rows, uint32_t experts,
                     bool broadcast_input);
    // Explicit raw routed Q4_K/Q5_1 only, fixed before the first batch. No dense, grouped, packed or indirect route is selected.
    void configure_raw_decode(bool enabled);
    [[nodiscard]] bool raw_decode_enabled() const;
    using RawDecodeRoutes = std::array<uint64_t, 2>;
    // Successful completed physical dispatches, Q4_K then Q5_1. Aborted or failed submissions publish no routes.
    [[nodiscard]] RawDecodeRoutes raw_decode_routes() const;
    void mixture(GpuView expert_outputs, GpuView weights, GpuView shared_output,
                 GpuView shared_logit, GpuView output);
    void gdn_prepare(GpuView activated_qkv, GpuView query, GpuView key, GpuView value,
                     GdnShape shape, GdnPolicy policy);
    void gdn_gates(GpuView alpha_logits, GpuView beta_logits, GpuView negative_rates,
                   GpuView dt_bias, GpuView decay, GpuView beta);
    void split_query_gate(GpuView packed, GpuView query, GpuView gate, uint32_t heads, uint32_t width, QueryPacking packing);
    void rotary(GpuView input, GpuView frequencies, GpuView output, uint32_t heads, uint32_t width, uint32_t position, RotaryPairs pairs);
    void store_bf16(GpuView input, GpuView destination_row);
    void index_accumulate(GpuView current_sum, GpuView key, GpuView next_sum, GpuView pooled, uint32_t position, uint32_t compression);
    void index_scores(GpuView query, GpuView bf16_keys, GpuView scores, uint32_t heads, uint32_t width, uint32_t blocks);
    void select_blocks(GpuView scores, GpuView selected_ids, SelectionScratch scratch, uint32_t blocks, uint32_t budget);
    void sparse_attention(GpuView query, GpuView bf16_keys, GpuView bf16_values, GpuView selected_ids, GpuView output, uint32_t prefix, QsaShape shape);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    // Explicit depth-one head primitives. Dense attention visits every valid
    // cache row and has no trunk QSA/indexer policy or selected-ID dependency.
    void prepare_mtp_head();
    void prepare_mtp_head_tiled();
    void mtp_concat(GpuView normalized_embedding, GpuView normalized_hc, GpuView output,
                    uint32_t branches, uint32_t channels);
    void mtp_dense_attention(GpuView query, GpuView bf16_keys, GpuView bf16_values, GpuView output,
                             uint32_t prefix, uint32_t query_heads, uint32_t kv_heads, uint32_t head_width);
    // Explicit numerical alternative for proposal Q only. Partials contain
    // unnormalised softmax numerators, maxima and denominators. The immutable
    // caller-owned workspace must cover all active head/tile records.
    void mtp_dense_attention_tiled(GpuView query, GpuView bf16_keys, GpuView bf16_values, GpuView output,
                                  GpuView partials, uint32_t prefix, uint32_t heads,
                                  uint32_t kv_heads, uint32_t width, uint32_t tile_tokens);
    void mtp_concat_rows(GpuView normalized_embedding, GpuView normalized_hc, GpuView output,
                         uint32_t branches, uint32_t channels, uint32_t rows);
    void mtp_rope_store_kv_bf16(GpuView normalized_key, GpuView raw_value, GpuView frequencies,
                                GpuView bf16_keys, GpuView bf16_values, uint32_t kv_heads, uint32_t head_width,
                                uint32_t first_position, RotaryPairs pairs, uint32_t rows);
    void mtp_embed_from_topk_q8(GpuView token_embd, GpuView topk_row, GpuView output,
                                uint32_t channels, uint32_t vocabulary);
    void mtp_sample_and_embed_from_topk_q8(GpuView token_embd, GpuView topk_row, GpuView tickets_buf,
                                           GpuView law_out, GpuView output, uint32_t channels,
                                           uint32_t embd_vocabulary, uint32_t max_k, bool embed_output,
                                           float inv_temp, float min_p, float top_p);
    void prepare_dflash2();
    void set_dflash2_scope(bool active);
    void dflash2_stream_mean(GpuView streams, GpuView captured, uint32_t rows, uint32_t width, uint32_t slot_index,
                             uint32_t total_slots, uint64_t input_stride_bytes, uint64_t output_stride_bytes);
    void dflash2_context_kv_write(GpuView context_kv, GpuView qk_norm_freq, GpuView ring_k, GpuView ring_v,
                                  uint32_t rows, uint32_t start_pos, uint32_t window = 2048, float eps = 1e-6f);
    void dflash2_conv_f32(GpuView input, GpuView dynamic, GpuView base, GpuView residual, GpuView output,
                          uint32_t hidden, bool finish, uint32_t rows = 8, uint32_t group_size = 16);
    void dflash2_qkv_prepare(GpuView proposal_qkv, GpuView qk_norm_freq, GpuView queries, GpuView query_keys,
                             GpuView query_values, uint32_t cache_len, uint32_t rows = 8, float eps = 1e-6f);
    void dflash2_attention(GpuView queries, GpuView ring_k, GpuView ring_v, GpuView query_keys, GpuView query_values,
                           GpuView partials, GpuView output, uint32_t cache_len, uint32_t window = 2048);
    void dflash2_select(GpuView logits, GpuView partial_ids, GpuView partial_values, GpuView selector_out,
                        GpuView selector_hidden, GpuView predecessor_codebook, GpuView successor_codebook,
                        uint32_t positions, uint32_t vocabulary, uint32_t anchor,
                        uint32_t codebook_vocabulary = 0);
    void dflash2_dense_bf16(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool swiglu = false);
    void dflash2_dense_q8(GpuView matrix, GpuRowsF32 input, GpuRowsF32 output, bool swiglu = false);
    void dflash2_gate_up_swiglu(GpuView gate_matrix, GpuView up_matrix, GpuRowsF32 input, GpuRowsF32 output, bool bf16 = false);
    void dflash2_conv_finish_rms_f32(GpuView input, GpuView dynamic, GpuView base, GpuView residual, GpuView gains,
                                     GpuView residual_out, GpuView norm_out, uint32_t hidden, float eps = 1e-6f,
                                     uint32_t rows = 8, uint32_t group_size = 16);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tokspur
